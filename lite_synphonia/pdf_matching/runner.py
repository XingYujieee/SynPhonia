from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..stage_state import make_stage_status
from .api_embedder import APIEmbeddingClient
from .pdf_reader import PdfDocument, read_pdf_document
from .scorer import PageScore, aggregate_page_scores
from .state_machine import MatchingConfig, PageMatcherState, PageTimelineEntry

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TimedTranscriptSegment:
    segment_id: str
    text: str
    start_time: float
    end_time: float
    is_final: bool = True


@dataclass
class PdfMatchingResult:
    pages: list[dict[str, object]]
    segment_matches: list[dict[str, object]]
    timeline: list[dict[str, object]]
    runtime: dict[str, object]
    warnings: dict[str, object]
    stage_status: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "pages": self.pages,
            "segment_matches": self.segment_matches,
            "timeline": self.timeline,
            "runtime": self.runtime,
            "warnings": self.warnings,
            "stage_status": self.stage_status or _default_pdf_stage_status(self),
        }


@dataclass
class _PreparedPdf:
    document: PdfDocument
    chunk_page_indexes: list[int]
    chunk_texts: list[str]
    chunk_embeddings: np.ndarray
    page_texts: list[str]
    cache_hit: bool = False


# ── PDF embedding cache ───────────────────────────────────────────────────────

def _pdf_content_hash(pdf_path: Path) -> str:
    """Return a short SHA-256 hex digest of the PDF file bytes."""
    h = hashlib.sha256()
    with open(pdf_path, "rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            h.update(block)
    return h.hexdigest()[:16]


def _cache_key(pdf_path: Path, model: str) -> str:
    """Unique cache filename for a (pdf, model) pair."""
    safe_model = model.replace("/", "_").replace("\\", "_")
    return f"{_pdf_content_hash(pdf_path)}_{safe_model}.npz"


def _load_pdf_cache(
    cache_dir: Path,
    pdf_path: Path,
    model: str,
) -> np.ndarray | None:
    """Return cached chunk embeddings if they exist, else None."""
    cache_file = cache_dir / _cache_key(pdf_path, model)
    if not cache_file.exists():
        return None
    try:
        data = np.load(str(cache_file))
        embeddings = data["embeddings"]
        log.info(
            "[pdf_match] PDF cache hit: %s  shape=%s",
            cache_file.name, embeddings.shape,
        )
        return embeddings
    except Exception as exc:
        log.warning("[pdf_match] PDF cache load failed (%s), will re-embed.", exc)
        return None


def _save_pdf_cache(
    cache_dir: Path,
    pdf_path: Path,
    model: str,
    embeddings: np.ndarray,
) -> None:
    """Persist chunk embeddings to disk for future runs."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / _cache_key(pdf_path, model)
    try:
        np.savez_compressed(str(cache_file), embeddings=embeddings)
        log.info(
            "[pdf_match] PDF cache saved: %s  shape=%s",
            cache_file.name, embeddings.shape,
        )
    except Exception as exc:
        log.warning("[pdf_match] PDF cache save failed: %s", exc)


# ── Main entry point ──────────────────────────────────────────────────────────

def run_pdf_matching_api(
    *,
    pdf_path: str | Path,
    transcription_payload: dict[str, Any],
    output_path: str | Path,
    embedder: APIEmbeddingClient,
    matching_config: MatchingConfig | None = None,
    pdf_cache_dir: Path | None = None,
    passage_prefix: str = "",
    query_prefix: str = "",
) -> PdfMatchingResult:
    """Match transcript segments to PDF pages using API-based embeddings.

    Parameters
    ----------
    pdf_path:
        Path to the slide PDF.
    transcription_payload:
        The transcription.json payload dict.
    output_path:
        Destination for pdf_match/results.json.
    embedder:
        Ready ``APIEmbeddingClient`` instance.
    matching_config:
        Optional tuning overrides.
    pdf_cache_dir:
        Directory for on-disk PDF embedding cache.  When provided, chunk
        embeddings are loaded from cache on the first hit and recomputed only
        when the PDF content changes.  Pass ``None`` to disable caching.
    passage_prefix:
        String prepended to each PDF chunk text before embedding.
        Use ``"passage: "`` for E5-family models, ``""`` for BGE-M3 / OpenAI.
    query_prefix:
        String prepended to each query segment text before embedding.
        Use ``"query: "`` for E5-family models, ``""`` for BGE-M3 / OpenAI.
    """
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    segments = extract_segments_from_payload(transcription_payload)
    if not segments:
        raise RuntimeError(
            "PDF matching requires timestamped transcription results. "
            "Use a transcription.json payload or run the transcription stage."
        )

    config = matching_config or MatchingConfig()

    # ── 1. Prepare PDF (chunk + embed, with cache) ────────────────────────
    prepared = _prepare_pdf(
        pdf_path, embedder,
        pdf_cache_dir=pdf_cache_dir,
        passage_prefix=passage_prefix,
    )

    # ── 2. Pre-embed ALL query segments in ONE batch call ─────────────────
    # Previously, encode_texts was called inside the per-segment loop,
    # producing N sequential API requests (one per segment).  Pre-embedding
    # reduces that to a single batched request regardless of segment count.
    unique_texts = list(dict.fromkeys(s.text for s in segments if s.text.strip()))
    if unique_texts:
        prefixed = [f"{query_prefix}{t}" if query_prefix else t for t in unique_texts]
        raw_embeddings = embedder.encode_texts(prefixed, input_type="query")
        query_cache: dict[str, np.ndarray] = {
            text: raw_embeddings[i] for i, text in enumerate(unique_texts)
        }
        log.info(
            "[pdf_match] Pre-embedded %d unique query segments in 1 batch call.",
            len(unique_texts),
        )
    else:
        query_cache = {}

    # ── 3. Matching loop — pure numpy, no API calls ───────────────────────
    matcher_state = PageMatcherState(config=config)
    segment_matches: list[dict[str, object]] = []
    page_scores_history: list[list[PageScore]] = []
    final_segments: list[TimedTranscriptSegment] = []

    for segment in segments:
        final_segments.append(segment)
        final_segments = _trim_segments(final_segments, segment.end_time, config)
        query_texts = [s.text for s in final_segments if s.text.strip()]
        if not query_texts:
            continue

        # Build window embedding matrix from cache — zero API calls
        cached = [query_cache[t] for t in query_texts if t in query_cache]
        if not cached:
            continue
        query_embeddings = np.stack(cached, axis=0)
        weighted_query = _weighted_query_embedding(query_embeddings)

        page_scores = aggregate_page_scores(
            query_embedding=weighted_query,
            chunk_embeddings=prepared.chunk_embeddings,
            chunk_page_indexes=prepared.chunk_page_indexes,
            page_count=len(prepared.document.pages),
            query_texts=query_texts,
            page_texts=prepared.page_texts,
        )
        page_scores_history.append(page_scores)
        decision = matcher_state.update(
            start_time=segment.start_time,
            end_time=segment.end_time,
            page_scores=page_scores,
        )
        segment_matches.append(
            {
                "segment_id": segment.segment_id,
                "start": round(segment.start_time, 3),
                "end": round(segment.end_time, 3),
                "text": segment.text,
                "page_index": decision.assigned_page,
                "confidence": round(max(decision.confidence, 0.0), 4),
                "switched": decision.switched,
                "reason": decision.reason,
                "page_scores": _page_scores_payload(page_scores),
            }
        )

    end_time = segments[-1].end_time if segments else 0.0
    matcher_state.finalize(float(end_time))
    segment_matches, timeline_entries = _apply_global_monotonic_smoothing(
        segment_matches,
        page_scores_history,
        page_count=len(prepared.document.pages),
    )
    timeline_quality = _timeline_quality_payload(
        segment_matches,
        timeline_entries,
        page_scores_history,
    )
    result = PdfMatchingResult(
        pages=_pages_payload(prepared.document),
        segment_matches=segment_matches,
        timeline=[item.to_dict() for item in timeline_entries],
        runtime={
            "embedding_backend": "api",
            "embedding_model": embedder._model,
            "pdf_path": str(Path(pdf_path).resolve()),
            "page_count": len(prepared.document.pages),
            "chunk_count": len(prepared.chunk_texts),
            "segment_count": len(segments),
            "unique_query_segments": len(unique_texts),
            "pdf_cache_hit": prepared.cache_hit,
            "passage_prefix": passage_prefix,
            "query_prefix": query_prefix,
            "query_max_seconds": config.query_max_seconds,
            "query_max_segments": config.query_max_segments,
            "switch_margin": config.switch_margin,
            "low_confidence_threshold": config.low_confidence_threshold,
            "min_page_dwell_seconds": config.min_page_dwell_seconds,
            "cooldown_seconds": config.cooldown_seconds,
            "confirmations_required": config.confirmations_required,
        },
        warnings={
            **prepared.document.warnings,
            "low_confidence_segments": matcher_state.low_confidence_segments,
            "timeline_quality": timeline_quality,
        },
    )
    write_pdf_matching_result(output_path, result)
    return result


# ── Internal helpers ──────────────────────────────────────────────────────────

def write_pdf_matching_result(path: str | Path, result: PdfMatchingResult) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if result.stage_status is None:
        result.stage_status = _default_pdf_stage_status(result)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        **result.to_dict(),
    }
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return destination


def extract_segments_from_payload(payload: dict[str, object]) -> list[TimedTranscriptSegment]:
    results = payload.get("results")
    if isinstance(results, list) and results:
        segments: list[TimedTranscriptSegment] = []
        for index, item in enumerate(results):
            if not isinstance(item, dict):
                continue
            text = " ".join(str(item.get("text", "")).split()).strip()
            if not text:
                continue
            if "t0" not in item or "t1" not in item:
                continue
            segments.append(
                TimedTranscriptSegment(
                    segment_id=f"result-{index:05d}",
                    text=text,
                    start_time=float(item["t0"]),
                    end_time=float(item["t1"]),
                    is_final=True,
                )
            )
        if segments:
            return segments

    events = payload.get("events")
    if not isinstance(events, list):
        return []

    segments = []
    for index, item in enumerate(sorted(events, key=_event_sort_key)):
        if not isinstance(item, dict) or not bool(item.get("is_final", False)):
            continue
        text = " ".join(str(item.get("text", "")).split()).strip()
        if not text:
            continue
        segments.append(
            TimedTranscriptSegment(
                segment_id=str(item.get("event_id") or f"event-{index:05d}"),
                text=text,
                start_time=float(item.get("timestamp_start", 0.0)),
                end_time=float(item.get("timestamp_end", 0.0)),
                is_final=True,
            )
        )
    return segments


def _prepare_pdf(
    path: str | Path,
    embedder: APIEmbeddingClient,
    *,
    pdf_cache_dir: Path | None = None,
    passage_prefix: str = "",
) -> _PreparedPdf:
    """Read the PDF, chunk it, and embed chunks — loading from cache when available."""
    pdf_path = Path(path).expanduser().resolve()
    document = read_pdf_document(pdf_path)
    text_pages = [page for page in document.pages if page.text_available]
    if not text_pages:
        raise RuntimeError(
            "PDF matching failed because no extractable text was found in the PDF."
        )

    chunk_texts: list[str] = []
    chunk_page_indexes: list[int] = []
    for page in document.pages:
        for chunk in page.chunks:
            chunk_texts.append(chunk.text)
            chunk_page_indexes.append(page.page_index)

    # Try loading from cache first
    cached_embeddings: np.ndarray | None = None
    if pdf_cache_dir is not None:
        cached_embeddings = _load_pdf_cache(pdf_cache_dir, pdf_path, embedder._model)

    if cached_embeddings is not None and cached_embeddings.shape[0] == len(chunk_texts):
        chunk_embeddings = cached_embeddings
        cache_hit = True
        print(f"[pdf_match] PDF embeddings loaded from cache ({len(chunk_texts)} chunks).")
    else:
        # Apply passage prefix only if specified (BGE-M3 → no prefix, E5 → "passage: ")
        texts_to_embed = [
            f"{passage_prefix}{t}" if passage_prefix else t
            for t in chunk_texts
        ]
        log.info("[pdf_match] Embedding %d PDF chunks via API...", len(texts_to_embed))
        chunk_embeddings = embedder.encode_texts(texts_to_embed, input_type="passage")
        cache_hit = False
        if pdf_cache_dir is not None:
            _save_pdf_cache(pdf_cache_dir, pdf_path, embedder._model, chunk_embeddings)

    return _PreparedPdf(
        document=document,
        chunk_page_indexes=chunk_page_indexes,
        chunk_texts=chunk_texts,
        chunk_embeddings=chunk_embeddings,
        page_texts=[page.text for page in document.pages],
        cache_hit=cache_hit,
    )


def _default_pdf_stage_status(result: PdfMatchingResult) -> dict[str, object]:
    empty_pages = result.warnings.get("empty_pages", [])
    extraction_failures = result.warnings.get("extraction_failures", [])
    low_confidence = result.warnings.get("low_confidence_segments", [])
    timeline_quality = result.warnings.get("timeline_quality", {})
    collapsed_timeline = bool(timeline_quality.get("collapsed_timeline", False))
    if empty_pages or extraction_failures or low_confidence or collapsed_timeline:
        return make_stage_status(
            "pdf_matching",
            "warning",
            (
                "PDF matching produced a collapsed timeline that is likely untrustworthy."
                if collapsed_timeline
                else "PDF matching completed with warnings that may reduce timeline trust."
            ),
            details={
                "empty_pages": empty_pages,
                "extraction_failures": extraction_failures,
                "low_confidence_segments": low_confidence,
                "timeline_quality": timeline_quality,
            },
        )
    return make_stage_status(
        "pdf_matching",
        "success",
        "PDF matching completed without structural warnings.",
    )


def _pages_payload(document: PdfDocument) -> list[dict[str, object]]:
    pages: list[dict[str, object]] = []
    for page in document.pages:
        pages.append(
            {
                "page_index": page.page_index,
                "page_label": page.page_label,
                "text_available": page.text_available,
                "text_preview": page.text_preview,
                "chunk_count": len(page.chunks),
            }
        )
    return pages


def _event_sort_key(item: object) -> tuple[float, float]:
    if not isinstance(item, dict):
        return (0.0, 0.0)
    return (
        float(item.get("timestamp_start", 0.0)),
        float(item.get("timestamp_end", 0.0)),
    )


def _weighted_query_embedding(query_embeddings: np.ndarray) -> np.ndarray:
    if len(query_embeddings) == 1:
        return query_embeddings[0]

    weights = np.ones((len(query_embeddings),), dtype=np.float32)
    if len(query_embeddings) >= 2:
        weights[-2:] = 3.0
    if len(query_embeddings) >= 4:
        weights[-4:-2] = 2.0

    weighted = (query_embeddings * weights[:, None]).sum(axis=0)
    norm = float(np.linalg.norm(weighted))
    if norm == 0.0:
        return weighted.astype(np.float32, copy=False)
    return (weighted / norm).astype(np.float32, copy=False)


def _page_scores_payload(page_scores: list[PageScore], *, limit: int = 5) -> list[dict[str, float | int]]:
    payload: list[dict[str, float | int]] = []
    for score in page_scores[:limit]:
        payload.append(
            {
                "page_index": score.page_index,
                "score": round(score.score, 4),
                "dense_score": round(score.dense_score, 4),
                "lexical_score": round(score.lexical_score, 4),
                "max_score": round(score.max_score, 4),
                "mean_top_score": round(score.mean_top_score, 4),
                "chunk_hits": score.chunk_hits,
            }
        )
    return payload


def _score_for_page(page_scores: list[PageScore], page_index: int | None) -> PageScore | None:
    if page_index is None:
        return None
    for score in page_scores:
        if score.page_index == page_index:
            return score
    return None


def _apply_global_monotonic_smoothing(
    segment_matches: list[dict[str, object]],
    page_scores_history: list[list[PageScore]],
    *,
    page_count: int,
) -> tuple[list[dict[str, object]], list[PageTimelineEntry]]:
    if not segment_matches or not page_scores_history:
        return segment_matches, []

    assignments = _smooth_page_sequence(page_scores_history, page_count=page_count)
    if not assignments:
        return segment_matches, []

    updated_matches: list[dict[str, object]] = []
    previous_page: int | None = None
    for match, assigned_page, page_scores in zip(segment_matches, assignments, page_scores_history):
        assigned_score = _score_for_page(page_scores, assigned_page)
        updated_matches.append(
            {
                **match,
                "page_index": assigned_page,
                "confidence": round(max(assigned_score.score if assigned_score is not None else 0.0, 0.0), 4),
                "switched": previous_page is None or assigned_page != previous_page,
                "reason": (
                    "global_monotonic_smoothing"
                    if match.get("page_index") != assigned_page
                    else match.get("reason", "")
                ),
                "page_scores": _page_scores_payload(page_scores),
            }
        )
        previous_page = assigned_page

    return updated_matches, _build_timeline_from_segment_matches(updated_matches)


def _smooth_page_sequence(
    page_scores_history: list[list[PageScore]],
    *,
    page_count: int,
    skip_penalty: float = 0.040,
) -> list[int]:
    if not page_scores_history or page_count <= 0:
        return []

    score_matrix: list[list[float]] = []
    for page_scores in page_scores_history:
        row = [-1.0] * page_count
        for score in page_scores:
            if 0 <= score.page_index < page_count:
                row[score.page_index] = score.score
        score_matrix.append(row)

    segment_count = len(score_matrix)
    dp = [[float("-inf")] * page_count for _ in range(segment_count)]
    backpointers = [[0] * page_count for _ in range(segment_count)]

    for page_index in range(page_count):
        dp[0][page_index] = score_matrix[0][page_index]

    for segment_index in range(1, segment_count):
        for page_index in range(page_count):
            best_score = float("-inf")
            best_prev = 0
            for prev_page in range(page_index + 1):
                transition_penalty = max(0, page_index - prev_page - 1) * skip_penalty
                candidate_score = dp[segment_index - 1][prev_page] - transition_penalty
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_prev = prev_page
            dp[segment_index][page_index] = best_score + score_matrix[segment_index][page_index]
            backpointers[segment_index][page_index] = best_prev

    last_page = max(range(page_count), key=lambda page_index: dp[-1][page_index])
    assignments = [last_page]
    for segment_index in range(segment_count - 1, 0, -1):
        last_page = backpointers[segment_index][last_page]
        assignments.append(last_page)
    assignments.reverse()
    return assignments


def _build_timeline_from_segment_matches(
    segment_matches: list[dict[str, object]]
) -> list[PageTimelineEntry]:
    timeline: list[PageTimelineEntry] = []
    for match in segment_matches:
        page_index = match.get("page_index")
        start_time = float(match.get("start", 0.0))
        end_time = float(match.get("end", start_time))
        confidence = float(match.get("confidence", 0.0))
        if page_index is None:
            continue
        if timeline and timeline[-1].page_index == page_index:
            timeline[-1].end_time = max(timeline[-1].end_time, end_time)
            timeline[-1].confidence = max(timeline[-1].confidence, confidence)
            continue
        timeline.append(
            PageTimelineEntry(
                page_index=int(page_index),
                start_time=start_time,
                end_time=end_time,
                confidence=confidence,
            )
        )
    return timeline


def _timeline_quality_payload(
    segment_matches: list[dict[str, object]],
    timeline_entries: list[PageTimelineEntry],
    page_scores_history: list[list[PageScore]],
) -> dict[str, object]:
    if not segment_matches:
        return {}

    raw_top_pages = [scores[0].page_index for scores in page_scores_history if scores]
    distinct_raw_top_pages = len(set(raw_top_pages))
    assigned_pages = [int(match["page_index"]) for match in segment_matches if match.get("page_index") is not None]
    distinct_assigned_pages = len(set(assigned_pages))
    total_duration = sum(entry.duration for entry in timeline_entries)
    dominant_page_ratio = 0.0
    if total_duration > 0 and timeline_entries:
        dominant_page_ratio = max(entry.duration for entry in timeline_entries) / total_duration

    collapsed_timeline = (
        len(segment_matches) >= 4
        and distinct_raw_top_pages >= 3
        and distinct_assigned_pages <= 1
        and dominant_page_ratio >= 0.9
    )
    return {
        "collapsed_timeline": collapsed_timeline,
        "distinct_raw_top_pages": distinct_raw_top_pages,
        "distinct_assigned_pages": distinct_assigned_pages,
        "dominant_page_ratio": round(dominant_page_ratio, 4),
    }


def _trim_segments(
    segments: list[TimedTranscriptSegment],
    current_end_time: float,
    config: MatchingConfig,
) -> list[TimedTranscriptSegment]:
    while len(segments) > config.query_max_segments:
        segments = segments[1:]
    while segments:
        oldest = segments[0]
        if current_end_time - oldest.end_time <= config.query_max_seconds:
            break
        segments = segments[1:]
    return segments

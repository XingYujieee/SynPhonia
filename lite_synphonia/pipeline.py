"""LiteSynphonia — fully in-process, zero-local-model pipeline.

Differences from merge_syn/pipeline.py
---------------------------------------
* No whisper.cpp binary check — transcription goes directly to Deepgram.
* No torch/transformers check — summarization goes directly to the LLM API.
* No PDF embedding model download — embeddings come from the embedding API.
* Everything runs in the same process (no subprocess spawning).
* Single provider registry file manages all API keys.

Stage flow
----------
1. Audio capture      → MicrophoneAudioSource + enhance_audio (unchanged)
2. Transcription      → Deepgram /v1/listen                   (API)
3. Transcript quality → same heuristics as merge_syn           (local)
4. Summarization      → OpenAI-compatible LLM API             (API)
5. PDF matching       → APIEmbeddingClient + merge_syn scorer  (API)
6. Output             → merged_results.json                    (same format)
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# ── Local utilities ───────────────────────────────────────────────────────────
from .transcription.audio_processing import enhance_audio
from .transcription.config import TranscriptionConfig
from .transcription.deepgram_client import (
    DeepgramCallError,
    DeepgramTranscriptionClient,
)
from .transcription.glossary import build_initial_prompt, load_glossary_file
from .transcription.models import TranscriptionEvent, TranscriptionMetrics
from .transcription.preflight import analyze_preflight_audio
from .transcription.quality import (
    assess_transcription_quality,
    build_transcription_stage_status,
)
from .transcription.result_export import write_json, write_wav
from .transcription.text_postprocess import is_plausible_text, normalize_text
from .transcription.config import build_runtime_summary

from .summarization.config import ApiConfig, AppConfig
from .provider_registry import get_registry
from .summarization.runner import run_pipeline as _run_summary_pipeline

from .bridge import (
    TranscriptBundle,
    extract_transcript_bundle_from_payload,
    write_consolidated_bundle,
    write_json_payload,
    write_transcript_input,
)
from .stage_state import (
    STAGE_STATUS_BLOCKED,
    STAGE_STATUS_SKIPPED,
    STAGE_STATUS_SUCCESS,
    STAGE_STATUS_WARNING,
    make_stage_status,
)

from .config import LiteConfig
from .pdf_matching.api_embedder import APIEmbeddingClient
from .pdf_matching.runner import run_pdf_matching_api

log = logging.getLogger(__name__)


# ── Audio helpers (re-implemented inline so no subprocess dependency) ─────────

def _estimate_rms(audio: np.ndarray) -> float:
    if len(audio) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio.astype(np.float32)))))


def _audio_stats(audio: np.ndarray) -> tuple[float, float]:
    if len(audio) == 0:
        return 0.0, 0.0
    return _estimate_rms(audio), float(np.max(np.abs(audio)))


async def _record_audio(
    seconds: float,
    cfg: TranscriptionConfig,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Record microphone audio, returning (raw_audio, enhanced_audio, chunk_count)."""
    from .transcription.audio import MicrophoneAudioSource

    mic = MicrophoneAudioSource(
        sample_rate=cfg.sample_rate,
        channels=cfg.channels,
        chunk_duration_ms=cfg.chunk_duration_ms,
    )
    await mic.start()
    print(f"[lite_synphonia] 请开始说话，正在录音 {seconds:.1f} 秒...")

    chunks: list[bytes] = []
    enhanced_chunks: list[np.ndarray] = []
    start = time.monotonic()
    while time.monotonic() - start < seconds:
        chunk = await asyncio.wait_for(mic.read_chunk(), timeout=2.0)
        if chunk:
            chunks.append(chunk)
            enhanced_chunks.append(
                enhance_audio(np.frombuffer(chunk, dtype=np.float32), cfg)
            )

    await mic.stop()
    if not chunks:
        return (
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
            0,
        )
    raw = np.concatenate([np.frombuffer(c, dtype=np.float32) for c in chunks])
    enhanced = np.concatenate(enhanced_chunks) if enhanced_chunks else np.array([], dtype=np.float32)
    return raw, enhanced, len(chunks)


# ── TranscriptionConfig from LiteConfig ──────────────────────────────────────

def _make_transcription_config(cfg: LiteConfig) -> TranscriptionConfig:
    tc = TranscriptionConfig()
    tc.sample_rate = cfg.transcription.sample_rate
    tc.channels = cfg.transcription.channels
    tc.chunk_duration_ms = cfg.transcription.chunk_duration_ms
    tc.silence_floor = cfg.transcription.silence_floor
    tc.input_gain = cfg.transcription.input_gain
    tc.target_rms = cfg.transcription.target_rms
    tc.max_gain = cfg.transcription.max_gain
    tc.limiter_level = cfg.transcription.limiter_level
    tc.language = cfg.language
    tc.initial_prompt = cfg.initial_prompt
    tc.pre_emphasis_coeff = cfg.transcription.pre_emphasis_coeff
    return tc


# ── Transcription stage ───────────────────────────────────────────────────────

def _build_transcription_payload(
    results: list[dict[str, Any]],
    metrics: "TranscriptionMetrics",
    tc_cfg: TranscriptionConfig,
    transcription_quality: dict[str, Any],
    stage_status: dict[str, Any],
    events: list[TranscriptionEvent],
    glossary_info: dict[str, Any],
    preflight_report: dict[str, Any] | None,
) -> dict[str, Any]:
    payload = {
        "runtime": build_runtime_summary(tc_cfg),
        "metrics": metrics.to_dict(),
        "events": [e.to_dict() for e in events],
        "results": results,
        "transcription_quality": transcription_quality,
        "stage_status": stage_status,
        "glossary": glossary_info,
        "preflight": preflight_report,
        # Flag that this came from the API backend, not local whisper
        "transcription_backend": "deepgram_api",
    }
    return payload


def _char_overlap_similarity(a: str, b: str) -> float:
    """Jaccard similarity on the character sets of two strings.

    Fast and language-agnostic — works for Chinese, English, and mixed text.
    Returns a value in [0, 1]; 1.0 means identical character sets.
    """
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _is_near_duplicate(text: str, recent: list[str], window: int = 6,
                       threshold: float = 0.82) -> bool:
    """Return True when *text* is very similar to any of the last *window* segments.

    Catches Deepgram looping/hallucination artefacts where the same short phrase
    is emitted repeatedly with slightly different timestamps.
    """
    for prev in recent[-window:]:
        if _char_overlap_similarity(text, prev) >= threshold:
            return True
    return False


def _clean_segments(
    results: list[dict[str, Any]], tc_cfg: TranscriptionConfig
) -> list[dict[str, Any]]:
    """Normalise, validate, and de-duplicate transcript segments.

    Steps applied in order:
    1. Normalise Chinese characters and strip whitespace.
    2. Discard segments that fail ``is_plausible_text`` (non-linguistic noise).
    3. Discard segments shorter than 2 characters (avoids single-char spam).
    4. Discard near-duplicate segments within a rolling window to eliminate
       Deepgram hallucination loops (e.g. the same phrase repeated 50 times).
    """
    cleaned: list[dict[str, Any]] = []
    seen_texts: list[str] = []

    for r in results:
        text = normalize_text(
            r["text"], prefer_simplified_chinese=tc_cfg.prefer_simplified_chinese
        )
        text = text.strip()

        # 1. Linguistic plausibility check
        if not is_plausible_text(text):
            log.debug("[clean_segments] dropped non-plausible: %r", text[:60])
            continue

        # 2. Minimum length: at least 2 non-space characters
        if len(text.replace(" ", "")) < 2:
            log.debug("[clean_segments] dropped too-short: %r", text)
            continue

        # 3. Near-duplicate suppression (catches Deepgram looping hallucinations)
        if _is_near_duplicate(text, seen_texts):
            log.debug("[clean_segments] dropped near-duplicate: %r", text[:60])
            continue

        cleaned.append({**r, "text": text})
        seen_texts.append(text)

    removed = len(results) - len(cleaned)
    if removed:
        log.info(
            "[clean_segments] Dropped %d/%d segments (noise/duplicates).",
            removed, len(results),
        )
    return cleaned


async def _transcription_stage(
    cfg: LiteConfig,
    output_dir: Path,
) -> dict[str, Any]:
    """Record audio → send to Deepgram → return transcription payload dict."""
    tc_cfg = _make_transcription_config(cfg)
    session_id = uuid.uuid4().hex
    t_output = output_dir / "transcription"
    t_output.mkdir(parents=True, exist_ok=True)

    # Glossary
    glossary_info: dict[str, Any] = {"terms": [], "prompt": ""}
    if cfg.glossary_file:
        try:
            terms = load_glossary_file(cfg.glossary_file)
            tc_cfg.glossary_terms = terms
            tc_cfg.initial_prompt = build_initial_prompt(terms, prefix=cfg.initial_prompt)
            glossary_info = {"terms": terms, "prompt": tc_cfg.initial_prompt}
        except Exception as exc:
            log.warning("[transcription] Could not load glossary: %s", exc)

    # ── Preflight ──────────────────────────────────────────────────────────
    preflight_report: dict[str, Any] | None = None
    if cfg.preflight_seconds > 0:
        print(f"[lite_synphonia] 麦克风预检: 录制 {cfg.preflight_seconds:.1f} 秒...")
        pre_raw, _pre_enhanced, _ = await _record_audio(cfg.preflight_seconds, tc_cfg)
        # Use the RAW signal for preflight stats: we want to detect hardware-level
        # ADC saturation, not whether the software limiter is working correctly.
        # The enhanced signal will always peak near limiter_level (0.72) for normal
        # speech, which would produce spurious clipping_risk=True with any
        # threshold derived from limiter_level.
        preflight_signal = pre_raw if len(pre_raw) > 0 else _pre_enhanced
        rms, peak = _audio_stats(preflight_signal)
        preflight = analyze_preflight_audio(
            seconds=cfg.preflight_seconds,
            rms=rms,
            peak=peak,
            silence_floor=tc_cfg.silence_floor,
            limiter_level=tc_cfg.limiter_level,
        )
        preflight_report = preflight.to_dict()
        print(f"[lite_synphonia] 预检: rms={rms:.4f} peak={peak:.4f} "
              f"clipping={preflight.clipping_risk} low_signal={preflight.low_signal_risk}")

    # ── Audio capture ──────────────────────────────────────────────────────
    metrics = TranscriptionMetrics(
        mode="direct_api",
        language=cfg.language,
        requested_seconds=cfg.record_seconds,
        recorded_seconds=0.0,
        sample_count=0,
    )

    if cfg.skip_mic:
        print("[lite_synphonia] skip-mic: using synthetic audio.")
        t_arr = np.linspace(0, 3, tc_cfg.sample_rate * 3, dtype=np.float32)
        audio = (0.3 * np.sin(2 * np.pi * 440 * t_arr)).astype(np.float32)
        raw_audio = audio
        chunk_count = 1
    else:
        raw_audio, audio, chunk_count = await _record_audio(cfg.record_seconds, tc_cfg)

    # When --no-agc is set, bypass the enhancement chain and send the raw
    # samples directly to Deepgram.  This avoids AGC-induced clipping on
    # hardware that already provides a clean signal.
    if getattr(cfg, "disable_agc", False) and len(raw_audio) > 0:
        print("[lite_synphonia] --no-agc: 跳过增益控制，直接使用原始音频。")
        audio = raw_audio

    metrics.sample_count = len(audio)
    metrics.chunk_count = chunk_count
    metrics.recorded_seconds = len(audio) / tc_cfg.sample_rate if tc_cfg.sample_rate else 0.0
    rms, peak = _audio_stats(audio)
    metrics.rms = rms
    metrics.peak = peak
    print(f"[lite_synphonia] 音频: {metrics.recorded_seconds:.1f}s  rms={rms:.4f} peak={peak:.4f}")

    # ── Deepgram transcription ─────────────────────────────────────────────
    registry = get_registry()
    api_key = registry.resolve_key(cfg.transcription.provider_name)
    entry = registry.get(cfg.transcription.provider_name)
    dg_model = cfg.transcription.model or entry.model_id or "whisper-large"

    t0 = time.perf_counter()
    raw_results: list[dict[str, Any]] = []
    transcribe_calls = 0
    deepgram_debug: dict[str, Any] = {"attempts": []}
    timeout_seconds = entry.timeout_seconds if entry.timeout_seconds > 0 else 120.0
    max_retries = entry.max_retries if entry.max_retries > 0 else 3
    base_language = (cfg.transcription.language or "").strip() or None
    use_raw_fallback = (
        len(raw_audio) > 0
        and len(audio) > 0
        and not np.array_equal(raw_audio, audio)
    )
    selected_model = dg_model

    def _lang_label(language: str | None) -> str:
        return language or "auto"

    def _transcribe_once(
        *,
        attempt: str,
        samples: np.ndarray,
        model: str,
        language: str | None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        client = DeepgramTranscriptionClient(
            api_key,
            model=model,
            language=language,
            base_url=entry.base_url,
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
        )
        record: dict[str, Any] = {
            "attempt": attempt,
            "model": model,
            "language": _lang_label(language),
        }
        try:
            print(
                f"[lite_synphonia] Deepgram 转录尝试: {attempt} "
                f"(model={model}, language={_lang_label(language)})..."
            )
            segments = client.transcribe(samples, sample_rate=tc_cfg.sample_rate, language=language)
            record["request_url"] = client.last_request_url
            if client.last_response_data is not None:
                record["response"] = client.last_response_data
            record["segment_count"] = len(segments)
            deepgram_debug["attempts"].append(record)
            return segments, None
        except DeepgramCallError as exc:
            record["request_url"] = client.last_request_url
            if client.last_response_data is not None:
                record["response"] = client.last_response_data
            elif client.last_response_text:
                record["response_text"] = client.last_response_text
            record["segment_count"] = 0
            record["error"] = str(exc)
            deepgram_debug["attempts"].append(record)
            return [], str(exc)

    attempts: list[dict[str, Any]] = [
        {
            "name": "enhanced_audio",
            "samples": audio,
            "model": dg_model,
            "language": base_language,
            "notice": "",
        },
    ]
    if use_raw_fallback:
        attempts.append(
            {
                "name": "raw_audio_fallback",
                "samples": raw_audio,
                "model": dg_model,
                "language": base_language,
                "notice": "[lite_synphonia] 增强音频无结果，回退原始音频重试...",
            }
        )
    if base_language:
        attempts.append(
            {
                "name": "enhanced_audio_auto_language",
                "samples": audio,
                "model": dg_model,
                "language": None,
                "notice": (
                    "[lite_synphonia] 指定语言无结果，改用自动语言检测重试..."
                ),
            }
        )
    if dg_model.lower() != "whisper-large":
        attempts.append(
            {
                "name": "enhanced_audio_whisper_fallback",
                "samples": audio,
                "model": "whisper-large",
                "language": None,
                "notice": (
                    "[lite_synphonia] 当前模型无结果，使用 whisper-large 自动语言兜底..."
                ),
            }
        )

    last_error: str | None = None
    for idx, spec in enumerate(attempts):
        notice = str(spec.get("notice", ""))
        if idx > 0 and notice:
            print(notice)
        segments, err = _transcribe_once(
            attempt=str(spec["name"]),
            samples=spec["samples"],
            model=str(spec["model"]),
            language=spec.get("language"),
        )
        transcribe_calls += 1
        if err:
            last_error = err
        if segments:
            raw_results = segments
            selected_model = str(spec["model"])
            break

    if not raw_results and last_error:
        log.error("[transcription] Deepgram API failed across attempts: %s", last_error)
        deepgram_debug["error"] = last_error

    metrics.transcription_seconds = time.perf_counter() - t0
    metrics.raw_segments = len(raw_results)
    results = _clean_segments(raw_results, tc_cfg)
    if raw_results and not results:
        print(
            "[lite_synphonia] Deepgram 返回了文本，但在文本质量过滤后为空。"
            "可查看 deepgram_response_raw.json 进一步诊断。"
        )
    metrics.cleaned_segments = len(results)
    metrics.filtered_segments = max(0, len(raw_results) - len(results))
    metrics.emitted_segments = len(results)
    metrics.final_segments = len(results)
    metrics.transcribe_calls = transcribe_calls

    print(f"[lite_synphonia] 转录完成: {len(results)} 个片段  ({metrics.transcription_seconds:.1f}s)")
    if results:
        print(f"[lite_synphonia] 已采用转录模型: {selected_model}")
    else:
        print("[lite_synphonia] 所有转录尝试均未得到可用文本。")

    # ── Build events ───────────────────────────────────────────────────────
    events = [
        TranscriptionEvent(
            session_id=session_id,
            event_id=uuid.uuid4().hex,
            source_chunk_id="deepgram-api",
            timestamp_start=float(r["t0"]),
            timestamp_end=float(r["t1"]),
            text=r["text"],
            language=r.get("lang", cfg.language),
            confidence=float(r.get("confidence", 0.0)),
            is_final=True,
            model_name=selected_model,
            speaker_id=None,
            tokens_count=len(r["text"].split()),
        )
        for r in results
    ]

    # ── Quality check (same heuristics as merge_syn) ───────────────────────
    quality = assess_transcription_quality(
        results,
        metrics=metrics.to_dict(),
        silence_floor=tc_cfg.silence_floor,
        limiter_level=tc_cfg.limiter_level,
        glossary_terms=tc_cfg.glossary_terms,
    )
    stage_status = build_transcription_stage_status(quality, has_results=bool(results))

    # ── Write artifacts ────────────────────────────────────────────────────
    payload = _build_transcription_payload(
        results, metrics, tc_cfg, quality, stage_status,
        events, glossary_info, preflight_report,
    )
    if not results and deepgram_debug.get("attempts"):
        write_json(t_output / "deepgram_response_raw.json", deepgram_debug)
    write_json(t_output / "transcription.json", payload)
    if len(raw_audio) > 0:
        write_wav(t_output / "raw_audio.wav", raw_audio, tc_cfg.sample_rate)
    if len(audio) > 0:
        write_wav(t_output / "enhanced_audio.wav", audio, tc_cfg.sample_rate)

    print(f"[lite_synphonia] 转录 JSON 已写入: {t_output / 'transcription.json'}")
    return payload


# ── Summarization stage ───────────────────────────────────────────────────────

def _summarization_stage(
    cfg: LiteConfig,
    bundle: TranscriptBundle,
    output_dir: Path,
    transcript_quality: dict[str, Any],
) -> dict[str, Any]:
    """Call the LLM API and return the summary payload dict."""
    s_cfg = cfg.summarization
    registry = get_registry()
    entry = registry.get(s_cfg.provider_name)

    api_cfg = ApiConfig(
        base_url=entry.base_url,
        model_id=entry.model_id or s_cfg.provider_name,
        api_key_env="",         # resolved via registry
        max_new_tokens=s_cfg.max_new_tokens,
        temperature=entry.temperature if entry.temperature >= 0 else 0.1,
        timeout_seconds=entry.timeout_seconds if entry.timeout_seconds > 0 else 60.0,
        max_retries=entry.max_retries if entry.max_retries > 0 else 3,
        provider_name=s_cfg.provider_name,
    )

    transcript_path = output_dir / "input" / "transcript.txt"
    summary_output_path = output_dir / "summary" / "results.json"
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)

    write_transcript_input(transcript_path, bundle)

    # Derive window settings from transcript word count
    word_count = max(1, getattr(bundle, "word_count", len(bundle.text.split())))
    window_size = min(s_cfg.window_size, word_count)
    overlap_size = min(s_cfg.overlap_size, window_size - 1) if window_size > 1 else 0

    app_config = AppConfig(
        input_path=transcript_path,
        output_path=summary_output_path,
        incremental_read_size=max(1, s_cfg.chunk_size),
        window_size=window_size,
        overlap_size=overlap_size,
        llm_backend="api",
        encoding="utf-8",
        api=api_cfg,
    )

    print(f"[lite_synphonia] 摘要生成中 (provider={s_cfg.provider_name})...")
    try:
        _run_summary_pipeline(app_config, resume=False, demo=False)
        summary_payload = json.loads(summary_output_path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.error("[summary] API call failed: %s", exc)
        summary_payload = {
            "completed": False,
            "results": [],
            "runtime": {"error": str(exc)},
        }

    summary_warning = transcript_quality.get("decision") != "pass"
    summary_payload["stage_status"] = make_stage_status(
        "summary",
        STAGE_STATUS_WARNING if summary_warning else STAGE_STATUS_SUCCESS,
        "Summary completed (upstream quality requires review)."
        if summary_warning
        else "Summary completed successfully.",
        upstream_dependency="transcription",
        quality_decision=str(transcript_quality.get("decision", "")),
        details={"active_backend": "api", "provider": s_cfg.provider_name},
    )
    write_json_payload(summary_output_path, summary_payload)
    rounds = len(summary_payload.get("results", []))
    print(f"[lite_synphonia] 摘要完成: {rounds} 轮")
    return summary_payload


# ── PDF matching stage ────────────────────────────────────────────────────────

def _pdf_matching_stage(
    cfg: LiteConfig,
    transcription_payload: dict[str, Any],
    output_dir: Path,
    transcript_quality: dict[str, Any],
) -> dict[str, Any]:
    """Embed transcript + PDF chunks via API, then run page-matching."""
    e_cfg = cfg.embedding
    registry = get_registry()
    api_key = registry.resolve_key(e_cfg.provider_name)
    entry = registry.get(e_cfg.provider_name)

    embedder = APIEmbeddingClient(
        api_key,
        base_url=entry.base_url,
        model=e_cfg.model or entry.model_id or "embo-01",
        batch_size=e_cfg.batch_size,
        max_retries=entry.max_retries if entry.max_retries > 0 else 3,
        timeout_seconds=entry.timeout_seconds if entry.timeout_seconds > 0 else 60.0,
        provider_format=getattr(e_cfg, "provider_format", "auto"),
    )

    pdf_output = output_dir / "pdf_match" / "results.json"
    pdf_cache_dir = (
        Path(e_cfg.pdf_cache_dir).expanduser()
        if getattr(e_cfg, "pdf_cache_dir", "")
        else output_dir / ".pdf_embed_cache"
    )
    passage_prefix = getattr(e_cfg, "passage_prefix", "")
    query_prefix   = getattr(e_cfg, "query_prefix",   "")
    print(
        f"[lite_synphonia] PDF 匹配中 "
        f"(provider={e_cfg.provider_name}, format={embedder._format}, "
        f"passage_prefix={passage_prefix!r}, query_prefix={query_prefix!r})..."
    )
    try:
        result = run_pdf_matching_api(
            pdf_path=cfg.pdf_path,
            transcription_payload=transcription_payload,
            output_path=pdf_output,
            embedder=embedder,
            pdf_cache_dir=pdf_cache_dir,
            passage_prefix=passage_prefix,
            query_prefix=query_prefix,
        )
        pdf_payload = json.loads(pdf_output.read_text(encoding="utf-8"))
        timeline_len = len(result.timeline)
        print(f"[lite_synphonia] PDF 匹配完成: {timeline_len} 个时间段")
    except Exception as exc:
        log.error("[pdf_match] Failed: %s", exc)
        pdf_payload = {"error": str(exc), "timeline": [], "segment_matches": []}

    pdf_payload["stage_status"] = make_stage_status(
        "pdf_matching",
        STAGE_STATUS_SUCCESS if "error" not in pdf_payload else STAGE_STATUS_BLOCKED,
        "PDF matching completed." if "error" not in pdf_payload else f"PDF matching failed: {pdf_payload.get('error')}",
        upstream_dependency="transcription",
        details={"embedding_provider": e_cfg.provider_name},
    )
    write_json_payload(pdf_output, pdf_payload)
    return pdf_payload


# ── Quality override helper ───────────────────────────────────────────────────

def _maybe_override_quality_decision(
    quality: dict[str, Any],
    results: list[dict[str, Any]],
    confidence_threshold: float,
) -> dict[str, Any]:
    """Re-evaluate a "fail" quality decision using content-based signals.

    merge_syn's ``assess_transcription_quality`` uses a hard confidence floor of
    0.30.  This is appropriate for clean studio audio but too aggressive for
    real-lecture environments where Deepgram often returns valid Chinese text at
    confidence 0.10–0.25 (especially for accented speech or mixed code-switching).

    Override logic
    --------------
    A ``fail`` decision is *downgraded to* ``warn`` when ALL of the following
    hold:

    1. At least one segment survived ``_clean_segments`` (i.e. real text exists).
    2. The combined unique character count of the cleaned transcript exceeds 8
       (screens out single-character loops that sneak past the deduplicator).
    3. The mean confidence from Deepgram, though low, is above
       ``confidence_threshold`` (default 0.15, configurable via CLI).
    4. The content diversity ratio — unique chars / total chars — is > 0.15
       (a highly repetitive string like "嗯嗯嗯嗯" has near-zero diversity).

    The returned dict is a shallow copy with ``decision`` and a new
    ``content_override`` key explaining the adjustment.  All original keys
    (stats, checks, etc.) are preserved so merge_syn consumers still work.
    """
    if quality.get("decision") != "fail":
        return quality  # already passing or warning — nothing to override

    if not results:
        return quality  # no text at all; let the fail stand

    all_text = "".join(r.get("text", "") for r in results).replace(" ", "")
    unique_chars = len(set(all_text))
    total_chars = len(all_text)
    diversity = unique_chars / total_chars if total_chars else 0.0
    stats = quality.get("stats", {})
    mean_conf = float(stats.get("mean_confidence", 0.0)) if isinstance(stats, dict) else 0.0

    reasons: list[str] = []
    if total_chars < 8:
        reasons.append(f"total_chars={total_chars} < 8 (too little text)")
    if diversity <= 0.15:
        reasons.append(f"diversity={diversity:.2f} ≤ 0.15 (repetitive content)")
    if mean_conf < confidence_threshold:
        reasons.append(
            f"mean_confidence={mean_conf:.3f} < threshold={confidence_threshold:.3f}"
        )

    if reasons:
        # Content is not good enough to override — keep the fail
        log.info(
            "[quality_override] Keeping fail decision.  Reasons: %s",
            "; ".join(reasons),
        )
        return quality

    # All checks passed — downgrade to warn so downstream can proceed
    overridden = dict(quality)
    overridden["decision"] = "warn"
    overridden["content_override"] = {
        "original_decision": "fail",
        "total_chars": total_chars,
        "unique_chars": unique_chars,
        "diversity": round(diversity, 3),
        "mean_confidence": round(mean_conf, 4),
        "confidence_threshold_used": confidence_threshold,
        "note": (
            "Downgraded from 'fail' to 'warn': Deepgram confidence is below the "
            "merge_syn hard floor but the transcript contains plausible, diverse "
            "content.  Downstream stages will run with a quality warning."
        ),
    }
    log.info(
        "[quality_override] Overriding 'fail' → 'warn'  "
        "(chars=%d, unique=%d, diversity=%.2f, mean_conf=%.4f)",
        total_chars, unique_chars, diversity, mean_conf,
    )
    print(
        f"[lite_synphonia] ℹ 质量评估: Deepgram 置信度偏低 "
        f"(mean={mean_conf:.3f})，但内容多样性通过检验 "
        f"(chars={total_chars}, diversity={diversity:.2f})。"
        "决策从 'fail' 修正为 'warn'，下游继续运行。"
    )
    return overridden


# ── Main pipeline ─────────────────────────────────────────────────────────────

async def run_pipeline(cfg: LiteConfig) -> int:
    """Execute the full LiteSynphonia pipeline and return an exit code."""
    run_started_at = datetime.now(timezone.utc)
    output_dir = cfg.output_path
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("  LiteSynphonia — All-API Pipeline")
    print(f"{'=' * 60}")
    print(f"  Output dir : {output_dir}")
    print(f"  Transcription : Deepgram ({cfg.transcription.provider_name})")
    print(f"  Summarization : LLM API ({cfg.summarization.provider_name})")
    if cfg.pdf_path:
        print(f"  PDF matching  : Embedding API ({cfg.embedding.provider_name})")
    print()

    # ── Stage 1: Transcription ─────────────────────────────────────────────
    try:
        transcription_payload = await _transcription_stage(cfg, output_dir)
    except Exception as exc:
        log.error("[pipeline] Transcription stage failed: %s", exc)
        return 1

    bundle = extract_transcript_bundle_from_payload(
        transcription_payload, source_label="deepgram_api"
    )
    transcript_quality = transcription_payload.get("transcription_quality", {})

    # ── Content-based quality override ────────────────────────────────────
    # Before applying the hard gate, try to downgrade "fail" → "warn" when the
    # transcript contains real, diverse content despite low Deepgram confidence.
    cleaned_results = transcription_payload.get("results", [])
    transcript_quality = _maybe_override_quality_decision(
        transcript_quality,
        cleaned_results,
        confidence_threshold=cfg.quality_confidence_threshold,
    )
    # Propagate the (possibly updated) quality back into the payload so it is
    # serialised correctly in the final merged_results.json.
    transcription_payload["transcription_quality"] = transcript_quality

    # Warn loudly when peak audio was near the hard limiter ceiling — this is
    # the primary cause of degraded Deepgram confidence.
    metrics_dict = transcription_payload.get("metrics", {})
    recorded_peak = float(metrics_dict.get("peak", 0.0)) if isinstance(metrics_dict, dict) else 0.0
    if recorded_peak >= 0.85:
        print(
            f"[lite_synphonia] ⚠ 音频峰值 {recorded_peak:.3f} 接近限幅器上限 "
            f"({cfg.transcription.limiter_level:.2f})。"
            "建议降低麦克风增益，或减小 --target-rms / --limiter-level，"
            "以避免音频失真拉低 Deepgram 置信度。"
        )

    t_decision = transcript_quality.get("decision", "pass")

    if t_decision == "fail" and not cfg.allow_low_quality_transcript:
        print("[lite_synphonia] ⚠ 转录质量不合格，下游阶段已跳过。"
              "  使用 --allow-low-quality-transcript 强制继续。")
        reason = str(transcript_quality.get("reason", "")).strip()
        if reason:
            print(f"[lite_synphonia] 质量原因: {reason}")
        stats = transcript_quality.get("stats", {})
        if isinstance(stats, dict):
            print(
                "[lite_synphonia] 质量指标: "
                f"segments={stats.get('segments', 0)} "
                f"chars={stats.get('chars', 0)} "
                f"words={stats.get('words', 0)} "
                f"mean_confidence={stats.get('mean_confidence', 0.0)}"
            )
        checks = transcript_quality.get("checks", [])
        if isinstance(checks, list):
            failed_checks = [
                str(item.get("name", ""))
                for item in checks
                if isinstance(item, dict) and not bool(item.get("pass", False))
            ]
            if failed_checks:
                print(
                    "[lite_synphonia] 未通过检查: "
                    + ", ".join(name for name in failed_checks if name)
                )
        _write_consolidated(
            output_dir, cfg, bundle, transcription_payload,
            _blocked_payload("summary", "Low-quality transcript blocked summarization."),
            _blocked_payload("pdf_matching", "Low-quality transcript blocked PDF matching."),
            run_started_at,
        )
        # Knowledge base ingestion is intentionally skipped on quality-fail runs.
        return 0

    # ── Stage 2: Summarization ─────────────────────────────────────────────
    if cfg.skip_summary:
        print("[lite_synphonia] 已按配置跳过内置摘要阶段。")
        summary_payload = _skipped_payload("summary", "Summary stage skipped by config.")
    elif bundle.unit_count == 0:
        print("[lite_synphonia] 没有文字内容，跳过摘要。")
        summary_payload = _skipped_payload("summary", "No transcript content.")
    else:
        try:
            summary_payload = _summarization_stage(cfg, bundle, output_dir, transcript_quality)
        except Exception as exc:
            log.error("[pipeline] Summarization stage failed: %s", exc)
            summary_payload = _blocked_payload("summary", str(exc))

    # ── Stage 3: PDF matching (optional) ──────────────────────────────────
    if cfg.skip_pdf_matching:
        print("[lite_synphonia] 已按配置跳过内置 PDF 匹配阶段。")
        pdf_payload = _skipped_payload("pdf_matching", "PDF matching stage skipped by config.")
    elif cfg.pdf_path:
        try:
            pdf_payload = _pdf_matching_stage(
                cfg, transcription_payload, output_dir, transcript_quality
            )
        except Exception as exc:
            log.error("[pipeline] PDF matching failed: %s", exc)
            pdf_payload = _blocked_payload("pdf_matching", str(exc))
    else:
        pdf_payload = _skipped_payload("pdf_matching", "No PDF path provided.")

    # ── Stage 4: Consolidated output ──────────────────────────────────────
    merged_path, interface_payload = _write_consolidated(
        output_dir, cfg, bundle, transcription_payload,
        summary_payload, pdf_payload, run_started_at,
    )

    # ── Stage 5: Knowledge base ingestion (optional) ──────────────────────
    _knowledge_base_stage(cfg, interface_payload, run_started_at)

    print(f"\n[lite_synphonia] ✓ 全部完成  →  {merged_path}")
    return 0


# ── Interface output (cross-module contract) ──────────────────────────────────
#
# Schema version 1.0
# Required fields  : activity_id, start_time, end_time, transcript_text,
#                    summary_text, keywords, ppt_present, ppt_file_path / ppt_id
# Optional fields  : transcript_meta, summary_meta, matched_slides,
#                    ppt_text_excerpt
#
# Consumers (knowledge-base, front-end) MUST tolerate absent optional fields.
# Optional fields are omitted entirely (not set to null) when unavailable so
# presence/absence is unambiguous.

_INTERFACE_SCHEMA_VERSION = "1.0"

# Single CJK character matcher (used for fullmatch validation)
_CJK_CHAR_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")

# Single-char function / connective characters used as phrase delimiters.
# Only unambiguously function-only characters belong here — content characters
# that happen to appear in function words (e.g. "经" in "神经") must NOT be
# included or they will break technical terms.
_ZH_SPLIT_CHARS: str = "的了在是有和也就都与或而但即对从于为以其么若把被让给"

# Phrase-level stopwords: discard candidates that are purely functional
_ZH_STOPWORDS: frozenset[str] = frozenset({
    "这个", "那个", "一个", "一些", "可以", "没有", "我们", "他们", "你们",
    "因为", "所以", "但是", "然后", "如果", "虽然", "已经", "非常", "一样",
    "这些", "那些", "什么", "怎么", "为什么", "知道", "觉得", "需要", "应该",
    "就是", "还是", "只是", "而且", "不是", "对于", "关于", "通过", "可能",
    "这样", "那样", "进行", "方面", "情况", "时候", "地方", "问题", "方式",
    "这里", "那里", "以及", "同时", "一下", "一种", "一定", "其中", "其他",
})

# Splits on non-CJK runs AND on common function characters
_PHRASE_SPLIT_RE = re.compile(
    r"[^\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+"
    r"|[" + re.escape(_ZH_SPLIT_CHARS) + r"]+"
)


def _extract_keywords(text: str, top_n: int = 10) -> list[str]:
    """Extract top-N keywords from Chinese text without an external tokenizer.

    Strategy: split on function characters ("的","在","是",...) to obtain
    phrase-level candidates like "深度学习" and "卷积神经网络", then rank by
    frequency.  For longer phrases, 2-3 char sub-grams are also emitted so
    individual technical terms nested inside longer strings are captured.
    A deduplication pass then removes shorter grams that are substrings of
    an already-retained keyword.
    """
    if not text:
        return []

    candidates: list[str] = []
    for phrase in _PHRASE_SPLIT_RE.split(text):
        phrase = phrase.strip()
        if len(phrase) < 2:
            continue
        # Accept only pure-CJK phrases
        if not all(_CJK_CHAR_RE.fullmatch(c) for c in phrase):
            continue
        # Direct phrase candidates only — no sub-gram decomposition.
        #
        # Window gram approaches (bigrams, sliding n-grams) inflate the
        # frequency of common component characters ("学习", "神经") above
        # the compound terms they belong to, causing the compound to be
        # suppressed.  In real multi-segment lecture transcripts, compound
        # terms like "卷积神经网络" appear multiple times across different
        # phrases, accumulating phrase-level frequency naturally.  Single-
        # occurrence terms in synthetic tests are an acceptable trade-off.
        if len(phrase) <= 6 and phrase not in _ZH_STOPWORDS:
            candidates.append(phrase)
        elif len(phrase) <= 8:
            # Accept slightly longer phrases (7-8 chars) only when they look
            # like a proper technical term — no function-char characters remain
            # after the split, and the phrase is entirely composed of CJK chars
            # (already guaranteed by the fullmatch check above).
            if phrase not in _ZH_STOPWORDS:
                candidates.append(phrase)

    freq: dict[str, int] = {}
    for c in candidates:
        freq[c] = freq.get(c, 0) + 1

    # Sort: frequency desc; at equal frequency prefer LONGER terms (more
    # specific compound terms over shorter common components).
    ranked = sorted(freq.items(), key=lambda kv: (-kv[1], -len(kv[0])))
    return [word for word, _ in ranked[:top_n]]


def build_interface_payload(
    *,
    activity_id: str,
    transcription_payload: dict[str, Any],
    summary_payload: dict[str, Any],
    pdf_payload: dict[str, Any],
    pdf_path: str,
    run_started_at: datetime,
) -> dict[str, Any]:
    """Build the standardised cross-module interface payload.

    Required fields are always present.  Optional fields are included only
    when the underlying data is actually available — they are never set to
    null.  This lets consumers distinguish "not recorded" from "recorded but
    empty" without additional sentinel values.

    Schema
    ------
    {
      "schema_version": "1.0",
      "activity_id":    str,          # required
      "created_at_utc": ISO-8601,     # required

      "transcription": {
        "start_time":      float,     # required — first segment t0 (seconds)
        "end_time":        float,     # required — last  segment t1 (seconds)
        "transcript_text": str,       # required — full joined transcript
        "transcript_meta": { ... }    # optional
      },

      "summary": {
        "summary_text": str,          # required
        "keywords":     [str, ...],   # required (may be empty list)
        "summary_meta": { ... }       # optional
      },

      "ppt": {
        "ppt_present":      bool,     # required
        "ppt_file_path":    str,      # required when ppt_present=true
        "ppt_id":           str,      # required when ppt_present=true (reserved)
        "matched_slides":   [ ... ],  # optional
        "ppt_text_excerpt": [ ... ]   # optional
      }
    }
    """
    # ── activity_id (auto-generate UUID if caller did not supply one) ──────
    aid = activity_id.strip() or uuid.uuid4().hex

    # ── Transcription ──────────────────────────────────────────────────────
    t_results: list[dict[str, Any]] = transcription_payload.get("results") or []
    start_time = float(t_results[0]["t0"])  if t_results else 0.0
    end_time   = float(t_results[-1]["t1"]) if t_results else 0.0
    transcript_text = " ".join(
        str(r.get("text", "")).strip() for r in t_results if r.get("text")
    ).strip()

    transcription_section: dict[str, Any] = {
        "start_time":      round(start_time, 3),
        "end_time":        round(end_time, 3),
        "transcript_text": transcript_text,
    }
    # transcript_meta — optional
    if t_results:
        metrics  = transcription_payload.get("metrics", {})
        quality  = transcription_payload.get("transcription_quality", {})
        runtime  = transcription_payload.get("runtime", {})
        t_meta: dict[str, Any] = {
            "segment_count":     len(t_results),
            "mean_confidence":   round(
                float((quality.get("stats") or {}).get("mean_confidence", 0.0)), 4
            ),
            "recorded_seconds":  round(float(metrics.get("recorded_seconds", 0.0)), 2),
            "language":          str(runtime.get("language", "")),
        }
        transcription_section["transcript_meta"] = t_meta

    # ── Summary ────────────────────────────────────────────────────────────
    s_results: list[dict[str, Any]] = summary_payload.get("results") or []
    # Collect text from all summary rounds — field name varies by provider
    summary_parts: list[str] = []
    for r in s_results:
        if not isinstance(r, dict):
            continue
        text = (
            r.get("summary") or r.get("text") or r.get("output") or
            r.get("content") or r.get("response") or ""
        )
        if text:
            summary_parts.append(str(text).strip())
    summary_text = "\n\n".join(summary_parts)

    # keywords — try explicit field first, fall back to frequency extraction
    keyword_set: list[str] = []
    for r in s_results:
        if isinstance(r, dict) and isinstance(r.get("keywords"), list):
            keyword_set.extend(str(k) for k in r["keywords"] if k)
    if not keyword_set and summary_text:
        keyword_set = _extract_keywords(summary_text)
    # Also scan transcript for domain terms not in summary
    if transcript_text and len(keyword_set) < 5:
        extra = _extract_keywords(transcript_text, top_n=5)
        seen = set(keyword_set)
        for kw in extra:
            if kw not in seen:
                keyword_set.append(kw)
                seen.add(kw)

    summary_section: dict[str, Any] = {
        "summary_text": summary_text,
        "keywords":     keyword_set[:15],    # cap at 15
    }
    # summary_meta — optional
    s_runtime = summary_payload.get("runtime", {})
    if s_results or isinstance(s_runtime, dict):
        s_meta: dict[str, Any] = {"rounds": len(s_results)}
        if isinstance(s_runtime, dict):
            if s_runtime.get("provider"):
                s_meta["provider"] = str(s_runtime["provider"])
            if s_runtime.get("model_id"):
                s_meta["model"] = str(s_runtime["model_id"])
        summary_section["summary_meta"] = s_meta

    # ── PPT / PDF ──────────────────────────────────────────────────────────
    ppt_present = bool(pdf_path and pdf_path.strip())
    ppt_section: dict[str, Any] = {"ppt_present": ppt_present}

    if ppt_present:
        ppt_section["ppt_file_path"] = pdf_path
        # ppt_id — reserved for future asset-management integration
        ppt_section["ppt_id"] = ""

        # matched_slides — optional (from timeline)
        timeline: list[dict[str, Any]] = pdf_payload.get("timeline") or []
        if timeline:
            matched_slides = [
                {
                    "slide_index": int(entry.get("page_index", 0)),
                    "start_time":  round(float(entry.get("start_time", 0.0)), 3),
                    "end_time":    round(float(entry.get("end_time",   0.0)), 3),
                    "confidence":  round(float(entry.get("confidence", 0.0)), 4),
                }
                for entry in timeline
            ]
            ppt_section["matched_slides"] = matched_slides

        # ppt_text_excerpt — optional (preview text for each matched page)
        pages: list[dict[str, Any]] = pdf_payload.get("pages") or []
        if pages and timeline:
            matched_idxs = {int(e.get("page_index", -1)) for e in timeline}
            excerpts = [
                {
                    "slide_index":  p["page_index"],
                    "text_preview": str(p.get("text_preview", ""))[:200],
                }
                for p in pages
                if int(p.get("page_index", -1)) in matched_idxs
                and p.get("text_preview")
            ]
            if excerpts:
                ppt_section["ppt_text_excerpt"] = excerpts
    else:
        ppt_section["ppt_file_path"] = ""
        ppt_section["ppt_id"] = ""

    return {
        "schema_version": _INTERFACE_SCHEMA_VERSION,
        "activity_id":    aid,
        "created_at_utc": run_started_at.isoformat(),
        "transcription":  transcription_section,
        "summary":        summary_section,
        "ppt":            ppt_section,
    }


def _write_interface_output(
    output_dir: Path,
    payload: dict[str, Any],
) -> Path:
    """Write interface_output.json alongside merged_results.json."""
    dest = output_dir / "interface_output.json"
    dest.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return dest


def _write_consolidated(
    output_dir: Path,
    cfg: LiteConfig,
    bundle: TranscriptBundle,
    transcription_payload: dict[str, Any],
    summary_payload: dict[str, Any],
    pdf_payload: dict[str, Any],
    run_started_at: datetime,
) -> tuple[Path, dict[str, Any]]:
    """Write merged_results.json and interface_output.json.

    Returns a tuple of (merged_results_path, interface_payload) so the caller
    can forward the interface payload to downstream consumers such as the
    knowledge base ingestion stage.
    """
    # ── 1. Standard merged_results.json (full internal format) ───────────
    merged_path = output_dir / "merged_results.json"
    stage_status = {
        "transcription": transcription_payload.get("stage_status", {}),
        "summary": summary_payload.get("stage_status", {}),
        "pdf_matching": pdf_payload.get("stage_status", {}),
    }
    write_consolidated_bundle(
        merged_path,
        source={
            "tool": "lite_synphonia",
            "version": "0.1.0",
            "run_started_at_utc": run_started_at.isoformat(),
            "transcription_provider": cfg.transcription.provider_name,
            "summarization_provider": cfg.summarization.provider_name,
            "embedding_provider": cfg.embedding.provider_name if cfg.pdf_path else "",
            "pdf_path": cfg.pdf_path,
            "activity_id": cfg.activity_id or "",
        },
        transcript=bundle,
        transcription_payload=transcription_payload,
        summary_payload=summary_payload,
        pdf_matching_payload=pdf_payload,
        stage_status=stage_status,
    )

    # ── 2. interface_output.json (cross-module contract) ──────────────────
    interface = build_interface_payload(
        activity_id=cfg.activity_id,
        transcription_payload=transcription_payload,
        summary_payload=summary_payload,
        pdf_payload=pdf_payload,
        pdf_path=cfg.pdf_path,
        run_started_at=run_started_at,
    )
    iface_path = _write_interface_output(output_dir, interface)
    print(f"[lite_synphonia] ✓ 接口输出  →  {iface_path}")

    return merged_path, interface


def _blocked_payload(stage: str, reason: str) -> dict[str, Any]:
    return {
        "stage_status": make_stage_status(stage, STAGE_STATUS_BLOCKED, reason),
        "completed": False,
        "results": [],
    }


def _skipped_payload(stage: str, reason: str) -> dict[str, Any]:
    return {
        "stage_status": make_stage_status(stage, STAGE_STATUS_SKIPPED, reason),
        "completed": False,
        "results": [],
    }


# ── Knowledge base ingestion stage ───────────────────────────────────────────

def _knowledge_base_stage(
    cfg: "LiteConfig",
    interface_payload: dict[str, Any],
    run_started_at: datetime,
) -> None:
    """Ingest the completed activity into the knowledge base.

    This is the activity-coordinator handoff described in the knowledge base
    boundary document.  It adapts the cross-module ``interface_output.json``
    payload into the flat ``ActivityIngestRecord.from_dict()`` format required
    by the knowledge base service, then calls ``ingest_completed_activity``.

    Failures are logged and printed as warnings — a KB ingestion failure must
    never abort or fail the pipeline itself.
    """
    if not cfg.knowledge_base_workspace:
        return

    from .knowledge_base.service import KnowledgeBaseService

    # ── Adapt interface_output format → ActivityIngestRecord flat format ──
    from datetime import timedelta

    t_section   = interface_payload.get("transcription") or {}
    s_section   = interface_payload.get("summary")       or {}
    ppt_section = interface_payload.get("ppt")           or {}

    # start_time / end_time: the interface stores elapsed audio seconds;
    # convert to absolute ISO-8601 datetimes using the run wall-clock time.
    t0_secs = float(t_section.get("start_time") or 0.0)
    t1_secs = float(t_section.get("end_time")   or 0.0)
    start_dt = (run_started_at + timedelta(seconds=t0_secs)).isoformat()
    end_dt   = (run_started_at + timedelta(seconds=t1_secs)).isoformat()

    record_dict: dict[str, Any] = {
        "activity_id":    interface_payload.get("activity_id", ""),
        "start_time":     start_dt,
        "end_time":       end_dt,
        "transcript_text": t_section.get("transcript_text", ""),
        "summary_text":   s_section.get("summary_text", ""),
        "keywords":       s_section.get("keywords") or [],
        "ppt_present":    bool(ppt_section.get("ppt_present", False)),
    }

    # ppt_file_path / ppt_id — omit when empty so the schema stays clean
    ppt_file_path = str(ppt_section.get("ppt_file_path") or "").strip()
    ppt_id        = str(ppt_section.get("ppt_id")        or "").strip()
    if ppt_file_path:
        record_dict["ppt_file_path"] = ppt_file_path
    if ppt_id:
        record_dict["ppt_id"] = ppt_id

    # Optional metadata fields — pass through when present
    if "transcript_meta" in t_section:
        record_dict["transcript_meta"] = t_section["transcript_meta"]
    if "summary_meta" in s_section:
        record_dict["summary_meta"] = s_section["summary_meta"]

    matched_slides = ppt_section.get("matched_slides")
    if matched_slides:
        record_dict["matched_slides"] = matched_slides

    # ppt_text_excerpt: interface stores a list of per-slide dicts; the KB
    # schema expects a plain string — serialise as a JSON snippet.
    ppt_text_excerpt = ppt_section.get("ppt_text_excerpt")
    if ppt_text_excerpt:
        if isinstance(ppt_text_excerpt, list):
            import json as _json
            record_dict["ppt_text_excerpt"] = _json.dumps(
                ppt_text_excerpt, ensure_ascii=False
            )
        else:
            record_dict["ppt_text_excerpt"] = str(ppt_text_excerpt)

    try:
        service = KnowledgeBaseService(cfg.knowledge_base_workspace)
        result  = service.ingest_completed_activity(record_dict)
        print(
            f"[lite_synphonia] ✓ 知识库已存储  activity_id={result.get('activity_id', '')} "
            f"→  {result.get('workspace', '')}"
        )
    except Exception as exc:
        log.warning("[kb] Failed to ingest activity into knowledge base: %s", exc)
        print(f"[lite_synphonia] ⚠ 知识库存储失败: {exc}")

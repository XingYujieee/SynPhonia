"""CLI entry for standalone PDF page matching.

Usage:
  python3 -m lite_synphonia pdf-match \
      --pdf-path slides.pdf \
      --transcription-json transcription.payload.json \
      --output-json pdf_match/results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..provider_registry import get_registry
from .api_embedder import APIEmbeddingClient
from .runner import run_pdf_matching_api


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m lite_synphonia pdf-match",
        description="Run transcript-to-PDF page matching as a standalone command.",
    )
    parser.add_argument(
        "--pdf-path",
        required=True,
        help="Path to the source PDF file.",
    )
    parser.add_argument(
        "--transcription-json",
        required=True,
        help="Path to a JSON payload containing transcription results (results[].text/t0/t1).",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Destination path for PDF matching output JSON.",
    )
    parser.add_argument(
        "--embedding-provider",
        default="siliconflow-embed",
        help="Provider name registered in providers.json. Default: siliconflow-embed.",
    )
    parser.add_argument(
        "--embedding-model",
        default="",
        help="Override embedding model id. Defaults to provider model_id.",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=32,
        help="Max texts per embedding API call.",
    )
    parser.add_argument(
        "--embedding-format",
        choices=["auto", "minimax", "openai"],
        default="auto",
        help="Embedding API wire format.",
    )
    parser.add_argument(
        "--embedding-passage-prefix",
        default="",
        help="Text prepended to PDF chunks before embedding.",
    )
    parser.add_argument(
        "--embedding-query-prefix",
        default="",
        help="Text prepended to transcript queries before embedding.",
    )
    parser.add_argument(
        "--pdf-cache-dir",
        default="",
        help=(
            "Embedding cache directory. Default: <workspace>/.pdf_embed_cache. "
            "Use none/off/disable to turn off cache."
        ),
    )
    return parser


def _resolve_cache_dir(raw: str, output_path: Path) -> Path | None:
    normalized = str(raw or "").strip()
    if not normalized:
        # output_json is usually workspace/pdf_match/results.json
        workspace_root = output_path.parent.parent
        return workspace_root / ".pdf_embed_cache"
    if normalized.lower() in {"none", "off", "disable", "disabled"}:
        return None
    return Path(normalized).expanduser().resolve()


def run_pdf_match_command(argv: list[str]) -> int:
    args = _build_parser().parse_args(argv)

    pdf_path = Path(args.pdf_path).expanduser().resolve()
    transcription_path = Path(args.transcription_json).expanduser().resolve()
    output_path = Path(args.output_json).expanduser().resolve()
    cache_dir = _resolve_cache_dir(args.pdf_cache_dir, output_path)

    try:
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")
        if not transcription_path.exists():
            raise FileNotFoundError(f"转录 payload 不存在: {transcription_path}")

        transcription_payload = json.loads(
            transcription_path.read_text(encoding="utf-8"),
        )

        registry = get_registry()
        provider_entry = registry.get(args.embedding_provider)
        api_key = registry.resolve_key(args.embedding_provider)
        model_id = (
            str(args.embedding_model).strip()
            or str(provider_entry.model_id).strip()
            or "BAAI/bge-large-zh-v1.5"
        )

        embedder = APIEmbeddingClient(
            api_key=api_key,
            base_url=provider_entry.base_url,
            model=model_id,
            batch_size=max(1, int(args.embedding_batch_size)),
            max_retries=max(1, int(provider_entry.max_retries or 3)),
            timeout_seconds=max(1.0, float(provider_entry.timeout_seconds or 60.0)),
            provider_format=args.embedding_format,
        )

        result = run_pdf_matching_api(
            pdf_path=pdf_path,
            transcription_payload=transcription_payload,
            output_path=output_path,
            embedder=embedder,
            pdf_cache_dir=cache_dir,
            passage_prefix=args.embedding_passage_prefix,
            query_prefix=args.embedding_query_prefix,
        )

        print(
            json.dumps(
                {
                    "ok": True,
                    "timeline_count": len(result.timeline),
                    "segment_count": len(result.segment_matches),
                    "output_json": str(output_path),
                },
                ensure_ascii=False,
            ),
        )
        return 0
    except Exception as exc:  # pragma: no cover - CLI error path
        print(f"[pdf-match] {exc}", file=sys.stderr)
        return 1

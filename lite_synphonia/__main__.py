"""Entry point: python3 -m lite_synphonia [options]

Two sub-surfaces:

  python3 -m lite_synphonia providers ...   → manage API keys (shared registry)
  python3 -m lite_synphonia audio-test ...  → monitor microphone levels in terminal
  python3 -m lite_synphonia summary-window  → summarize one transcript window
  python3 -m lite_synphonia [pipeline opts] → run the pipeline
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


# ── providers sub-command ─────────────────────────────────────────────────────

def _run_providers(argv: list[str]) -> int:
    """Run local provider management CLI."""
    from .providers_cli import run_providers_command

    return run_providers_command(argv)


def _run_audio_test(argv: list[str]) -> int:
    """Run terminal microphone diagnostics."""
    from .audio_test import run_audio_test_command

    return run_audio_test_command(argv)


def _run_summary_window(argv: list[str]) -> int:
    """Summarize a single transcript window."""
    from .summary_window import run_summary_window_command

    return run_summary_window_command(argv)


def _run_pdf_match(argv: list[str]) -> int:
    """Run standalone transcript-to-PDF page matching."""
    from .pdf_matching.cli import run_pdf_match_command

    return run_pdf_match_command(argv)


# ── pipeline argument parser ──────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python3 -m lite_synphonia",
        description=(
            "LiteSynphonia — zero-local-model lecture pipeline.\n"
            "All AI operations (transcription, summarization, embeddings) "
            "are performed via API calls; no local model files are required.\n\n"
            "Manage API keys:\n"
            "  python3 -m lite_synphonia providers list\n"
            "  python3 -m lite_synphonia providers add deepgram \\\n"
            "      --base-url https://api.deepgram.com \\\n"
            "      --model whisper-large --api-key <key> --service transcription\n"
            "  python3 -m lite_synphonia providers add minimax \\\n"
            "      --base-url https://api.minimaxi.chat/v1 \\\n"
            "      --model MiniMax-Text-01 --api-key <key> --service summarization\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── recording ─────────────────────────────────────────────────────────
    p.add_argument("--seconds", type=float, default=8.0,
                   help="Recording duration in seconds.")
    p.add_argument("--language", type=str, default="zh",
                   help="Whisper-style language code (e.g. zh, en, ja, auto). Default: zh.")
    p.add_argument("--skip-mic", action="store_true",
                   help="Use synthetic audio instead of microphone (for testing).")
    p.add_argument("--preflight-seconds", type=float, default=0.0,
                   help="Optional microphone preflight duration before the main recording.")

    # ── audio enhancement ─────────────────────────────────────────────────
    p.add_argument("--input-gain", type=float, default=1.0,
                   help="Minimum gain applied to quiet microphone audio.")
    p.add_argument("--target-rms", type=float, default=0.03,
                   help="Target RMS for automatic gain control. "
                        "Lower values (0.02–0.04) reduce clipping risk; "
                        "raise to 0.06 only if the mic signal is very quiet. "
                        "Default: 0.03.")
    p.add_argument("--max-gain", type=float, default=2.0,
                   help="Maximum automatic gain during enhancement. "
                        "Keep at 2.0 or below to avoid amplifying noise into distortion. "
                        "Default: 2.0.")
    p.add_argument("--limiter-level", type=float, default=0.72,
                   help="Peak limiter ceiling after enhancement (0–1). "
                        "0.72 gives ~3 dB headroom before digital full-scale, "
                        "preventing the hard clipping at 0.88 that degrades "
                        "Deepgram confidence. Default: 0.72.")
    p.add_argument("--no-agc", action="store_true",
                   help="Bypass the AGC/limiter enhancement chain entirely. "
                        "Use when hardware already provides a clean, well-levelled "
                        "signal, or to diagnose whether enhancement is causing problems.")
    p.add_argument("--pre-emphasis", type=float, default=0.97,
                   help="Pre-emphasis filter coefficient α (y[n] = x[n] - α·x[n-1]). "
                        "Boosts high-frequency consonants (zh/ch/sh/s) that ASR relies on. "
                        "Standard value is 0.97 (Kaldi/ESPnet default). "
                        "Set to 0.0 to disable. Default: 0.97.")

    # ── transcription ─────────────────────────────────────────────────────
    p.add_argument("--transcription-provider", type=str, default="deepgram",
                   help="Provider name for Deepgram STT (from registry). Default: deepgram.")
    p.add_argument("--transcription-model", type=str, default="whisper-large",
                   help="Deepgram model name. Default: whisper-large.")
    p.add_argument("--transcription-language", type=str, default="",
                   help="BCP-47 language for Deepgram (e.g. zh-CN, en-US). "
                        "Use 'auto' to disable forced language and let Deepgram decide. "
                        "Default: map from --language.")

    # ── summarization ─────────────────────────────────────────────────────
    p.add_argument("--summary-provider", type=str, default="minimax",
                   help="Provider name for the LLM summarization API (from registry). "
                        "Default: minimax.")
    p.add_argument("--summary-window-size", type=int, default=200,
                   help="Words per summarization window.")
    p.add_argument("--summary-overlap-size", type=int, default=60,
                   help="Overlap words between windows.")
    p.add_argument("--summary-chunk-size", type=int, default=1200,
                   help="Incremental reader chunk size (chars).")
    p.add_argument("--summary-max-new-tokens", type=int, default=384,
                   help="Max tokens the LLM may generate per call.")
    p.add_argument("--skip-summary", action="store_true",
                   help="Skip the built-in summarization stage.")

    # ── PDF matching ──────────────────────────────────────────────────────
    p.add_argument("--pdf-path", type=str, default="",
                   help="Optional PDF slide file for transcript-to-page matching.")
    p.add_argument("--embedding-provider", type=str, default="minimax-embed",
                   help="Provider name for text embeddings used in PDF matching. "
                        "Must support /v1/embeddings (e.g. minimax-embed, openai-embed). "
                        "Default: minimax-embed.")
    p.add_argument("--embedding-model", type=str, default="embo-01",
                   help="Embedding model ID. Default: embo-01 (MiniMax).")
    p.add_argument("--embedding-batch-size", type=int, default=32,
                   help="Max texts per embedding API call.")
    p.add_argument(
        "--embedding-format",
        type=str,
        default="auto",
        choices=["auto", "minimax", "openai"],
        help=(
            "Wire format for the embedding API. "
            "'auto' (default) detects from the provider base URL: "
            "URLs containing 'minimaxi' → MiniMax format "
            "(request key 'texts', response key 'vectors'); "
            "all others → OpenAI format "
            "(request key 'input', response key 'data[].embedding'). "
            "Use 'minimax' or 'openai' to force a specific format."
        ),
    )
    p.add_argument(
        "--embedding-passage-prefix",
        type=str,
        default="",
        help=(
            "Text prepended to each PDF chunk before embedding. "
            "E5-family models require 'passage: '; BGE-M3 and OpenAI models "
            "use plain text (default: empty string, i.e. no prefix). "
            "Example: --embedding-passage-prefix 'passage: '"
        ),
    )
    p.add_argument(
        "--embedding-query-prefix",
        type=str,
        default="",
        help=(
            "Text prepended to each transcript segment before embedding. "
            "E5-family models require 'query: '; BGE-M3 and OpenAI models "
            "use plain text (default: empty string). "
            "Example: --embedding-query-prefix 'query: '"
        ),
    )
    p.add_argument(
        "--pdf-cache-dir",
        type=str,
        default="",
        help=(
            "Directory for caching PDF chunk embeddings across runs. "
            "The cache is keyed by PDF content hash + model name, so it "
            "invalidates automatically when the PDF or model changes. "
            "Defaults to <output-dir>/.pdf_embed_cache. "
            "Pass 'none' to disable caching entirely."
        ),
    )
    p.add_argument("--skip-pdf-matching", action="store_true",
                   help="Skip the built-in PDF matching stage.")

    # ── output ────────────────────────────────────────────────────────────
    p.add_argument("--output-dir", type=str, default="",
                   help="Output directory. Defaults to lite_synphonia_output/.")
    p.add_argument("--activity-id", type=str, default="",
                   help="Unique identifier for this recording session. "
                        "Written to interface_output.json so downstream systems "
                        "(knowledge-base, front-end) can correlate this output "
                        "with the originating activity. "
                        "Defaults to a freshly generated UUID4 when not supplied.")
    p.add_argument("--initial-prompt", type=str, default="",
                   help="Optional hint to bias Deepgram recognition (e.g. domain terms).")
    p.add_argument("--glossary-file", type=str, default="",
                   help="Optional glossary file (one term per line).")
    p.add_argument("--allow-low-quality-transcript", action="store_true",
                   help="Allow downstream stages to run even when transcript quality is fail.")
    p.add_argument("--quality-confidence-threshold", type=float, default=0.15,
                   help="Minimum mean Deepgram confidence required for the content-based "
                        "quality override to promote a 'fail' decision to 'warn'. "
                        "Set to 0.0 to let any non-empty transcript through the content "
                        "check; set to 0.3 to restore the original strict behaviour. "
                        "Default: 0.15.")

    # ── knowledge base ────────────────────────────────────────────────────
    p.add_argument(
        "--knowledge-base-workspace",
        type=str,
        default="",
        help=(
            "Filesystem path to the knowledge base workspace directory. "
            "When provided, each completed pipeline run automatically ingests "
            "the activity record into the knowledge base after all other stages "
            "finish.  Omit (or leave empty) to skip knowledge base ingestion."
        ),
    )

    return p


def _build_config(args: argparse.Namespace):
    from .config import (
        LiteConfig,
        TranscriptionAPIConfig,
        SummarizationAPIConfig,
        EmbeddingAPIConfig,
    )
    from .transcription.deepgram_client import _whisper_lang_to_deepgram

    # Map --language to Deepgram BCP-47 unless user overrides.
    # Empty string means "do not force language" (auto detect).
    raw_dg_language = (args.transcription_language or "").strip()
    if raw_dg_language:
        dg_language = "" if raw_dg_language.lower() in {"auto", "none", "null"} else raw_dg_language
    else:
        mapped = _whisper_lang_to_deepgram(args.language)
        dg_language = mapped if mapped is not None else ""

    return LiteConfig(
        record_seconds=args.seconds,
        language=args.language,
        output_dir=args.output_dir,
        pdf_path=args.pdf_path,
        initial_prompt=args.initial_prompt,
        glossary_file=args.glossary_file,
        preflight_seconds=args.preflight_seconds,
        activity_id=args.activity_id,
        allow_low_quality_transcript=args.allow_low_quality_transcript,
        skip_mic=args.skip_mic,
        skip_summary=args.skip_summary,
        skip_pdf_matching=args.skip_pdf_matching,
        quality_confidence_threshold=args.quality_confidence_threshold,
        disable_agc=args.no_agc,
        knowledge_base_workspace=args.knowledge_base_workspace,
        transcription=TranscriptionAPIConfig(
            provider_name=args.transcription_provider,
            model=args.transcription_model,
            language=dg_language,
            input_gain=args.input_gain,
            target_rms=args.target_rms,
            max_gain=args.max_gain,
            limiter_level=args.limiter_level,
            pre_emphasis_coeff=args.pre_emphasis,
        ),
        summarization=SummarizationAPIConfig(
            provider_name=args.summary_provider,
            window_size=args.summary_window_size,
            overlap_size=args.summary_overlap_size,
            chunk_size=args.summary_chunk_size,
            max_new_tokens=args.summary_max_new_tokens,
        ),
        embedding=EmbeddingAPIConfig(
            provider_name=args.embedding_provider,
            model=args.embedding_model,
            batch_size=args.embedding_batch_size,
            provider_format=args.embedding_format,
            passage_prefix=args.embedding_passage_prefix,
            query_prefix=args.embedding_query_prefix,
            pdf_cache_dir=(
                "" if args.pdf_cache_dir.lower() in {"none", "off", "disable", "disabled"}
                else args.pdf_cache_dir
            ),
        ),
    )


# ── Entry point ───────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    raw = list(argv) if argv is not None else sys.argv[1:]

    # Intercept the providers sub-command before the main parser
    if raw and raw[0] == "providers":
        return _run_providers(raw[1:])
    if raw and raw[0] == "audio-test":
        return _run_audio_test(raw[1:])
    if raw and raw[0] == "summary-window":
        return _run_summary_window(raw[1:])
    if raw and raw[0] == "pdf-match":
        return _run_pdf_match(raw[1:])

    args = _build_parser().parse_args(raw)

    from .pipeline import run_pipeline
    cfg = _build_config(args)

    try:
        return asyncio.run(run_pipeline(cfg))
    except KeyboardInterrupt:
        print("\n[lite_synphonia] 已中断。")
        return 130


if __name__ == "__main__":
    sys.exit(main())

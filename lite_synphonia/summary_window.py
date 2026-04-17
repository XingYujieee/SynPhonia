from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .provider_registry import get_registry
from .summarization.api_client import ApiCallError, ApiLLMClient
from .summarization.config import ApiConfig
from .summarization.heuristic_summarizer import summarize_text
from .summarization.parser import parse_model_response
from .summarization.prompt import build_summarization_prompt


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m lite_synphonia summary-window",
        description="Summarize a single transcript window and emit JSON to stdout.",
    )
    parser.add_argument("--text", type=str, default="", help="Inline transcript window text.")
    parser.add_argument(
        "--input-file",
        type=str,
        default="",
        help="Optional UTF-8 text file path. Used when --text is empty.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="deepseek",
        help="Registered summarization provider name.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="auto",
        choices=["auto", "zh", "en"],
        help="Prompt language selection.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=384,
        help="Upper bound for generated tokens.",
    )
    parser.add_argument(
        "--previous-summary",
        type=str,
        default="",
        help="Previous summary text for deduplication.",
    )
    return parser


def _load_text(args: argparse.Namespace) -> str:
    inline_text = str(args.text or "").strip()
    if inline_text:
        return inline_text

    input_file = str(args.input_file or "").strip()
    if not input_file:
        raise ValueError("Either --text or --input-file must be provided.")

    return Path(input_file).read_text(encoding="utf-8").strip()


def _build_client(provider_name: str, max_new_tokens: int) -> ApiLLMClient:
    registry = get_registry()
    entry = registry.get(provider_name)
    api_config = ApiConfig(
        base_url=entry.base_url,
        model_id=entry.model_id or provider_name,
        api_key_env="",
        max_new_tokens=max_new_tokens,
        temperature=entry.temperature if entry.temperature >= 0 else 0.1,
        timeout_seconds=entry.timeout_seconds if entry.timeout_seconds > 0 else 60.0,
        max_retries=entry.max_retries if entry.max_retries > 0 else 3,
        provider_name=provider_name,
    )
    client = ApiLLMClient(api_config, provider_name=provider_name)
    client.ensure_ready()
    return client


def run_summary_window_command(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        text = _load_text(args)
        if not text:
            raise ValueError("Transcript window text is empty.")

        client = _build_client(args.provider, args.max_new_tokens)
        previous_summary = str(args.previous_summary or "").strip()
        prompt = build_summarization_prompt(
            text, language=args.language, previous_summary=previous_summary,
        )
        raw_response = client.generate_summary(prompt)
        parsed = parse_model_response(raw_response)

        if parsed.status in {"success", "success_recovered"}:
            payload = {
                "ok": True,
                "status": parsed.status,
                "keywords": parsed.keywords,
                "summary": parsed.summary,
                "raw_response": raw_response,
                "error_message": parsed.error_message,
                "provider": args.provider,
            }
        else:
            heuristic = summarize_text(text, keyword_limit=5)
            payload = {
                "ok": True,
                "status": "fallback_heuristic",
                "keywords": heuristic.keywords,
                "summary": heuristic.summary,
                "raw_response": raw_response,
                "error_message": parsed.error_message,
                "provider": args.provider,
            }

        sys.stdout.write(json.dumps(payload, ensure_ascii=False))
        sys.stdout.flush()
        return 0
    except (ApiCallError, ValueError, FileNotFoundError, KeyError) as exc:
        sys.stderr.write(str(exc))
        sys.stderr.flush()
        return 1


if __name__ == "__main__":
    raise SystemExit(run_summary_window_command())

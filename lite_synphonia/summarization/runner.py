from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from .api_client import ApiLLMClient
from .config import AppConfig
from .heuristic_summarizer import summarize_text
from .parser import parse_model_response
from .prompt import build_consolidation_prompt, build_summarization_prompt
from .storage import write_results_snapshot
from .window import build_window, can_form_window, slide_window


@dataclass
class RoundResult:
    round_number: int
    window_start_index: int
    window_end_index: int
    input_word_count: int
    input_text: str
    keywords: list[str]
    summary: str
    status: str
    raw_response: str
    error_message: str = ""
    generation_seconds: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "round_number": self.round_number,
            "window_start_index": self.window_start_index,
            "window_end_index": self.window_end_index,
            "input_word_count": self.input_word_count,
            "input_text": self.input_text,
            "keywords": self.keywords,
            "summary": self.summary,
            "status": self.status,
            "raw_response": self.raw_response,
            "error_message": self.error_message,
            "generation_seconds": round(self.generation_seconds, 3),
        }


def run_pipeline(
    config: AppConfig,
    resume: bool = False,
    demo: bool = False,
    demo_words_per_second: float = 2.0,
) -> list[RoundResult]:
    _ = (resume, demo, demo_words_per_second)
    if config.api is None:
        raise ValueError("AppConfig.api must be set for API summarization.")

    input_path = Path(config.input_path)
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    text = input_path.read_text(encoding=config.encoding) if input_path.exists() else ""
    words = text.split()

    client = ApiLLMClient(config.api, provider_name=config.api.provider_name or None)
    client.ensure_ready()

    results: list[RoundResult] = []
    window_start = 0
    round_number = 1
    step_size = max(1, config.window_size - config.overlap_size)

    while can_form_window(words, window_start, config.window_size):
        w = build_window(words, window_start, config.window_size)
        prompt = build_summarization_prompt(w.text)

        started = time.perf_counter()
        raw_response = ""
        parsed = None
        try:
            raw_response = client.generate_summary(prompt)
            parsed = parse_model_response(raw_response)
        except Exception as exc:
            parsed = None
            raw_response = f"ERROR: {exc}"

        elapsed = time.perf_counter() - started
        if parsed is not None and parsed.status in {"success", "success_recovered"}:
            keywords = parsed.keywords
            summary = parsed.summary
            status = parsed.status
            err = parsed.error_message
        else:
            heuristic = summarize_text(w.text, keyword_limit=5)
            keywords = heuristic.keywords
            summary = heuristic.summary
            status = "fallback_heuristic"
            err = parsed.error_message if parsed else "API generation failed."

        results.append(
            RoundResult(
                round_number=round_number,
                window_start_index=w.start_index,
                window_end_index=w.end_index,
                input_word_count=w.word_count,
                input_text=w.text,
                keywords=keywords,
                summary=summary,
                status=status,
                raw_response=raw_response,
                error_message=err,
                generation_seconds=elapsed,
            )
        )
        round_number += 1
        next_start = slide_window(window_start, config.window_size, config.overlap_size)
        if next_start <= window_start:
            next_start = window_start + step_size
        window_start = next_start

    consolidated_summary: dict[str, object] | None = None
    successful = [r.summary for r in results if r.summary.strip()]
    if len(successful) >= 2:
        try:
            consolidation_prompt = build_consolidation_prompt(successful)
            consolidation_raw = client.generate_summary(consolidation_prompt)
            parsed_final = parse_model_response(consolidation_raw)
            if parsed_final.status in {"success", "success_recovered"}:
                consolidated_summary = {
                    "keywords": parsed_final.keywords,
                    "summary": parsed_final.summary,
                    "status": parsed_final.status,
                    "raw_response": consolidation_raw,
                }
        except Exception:
            consolidated_summary = None

    runtime = {
        "active_backend": "api",
        "provider_name": config.api.provider_name,
        "model_id": config.api.model_id,
        "round_count": len(results),
        "input_word_count": len(words),
    }

    write_results_snapshot(
        output_path=output_path,
        config=config,
        results=[r.to_dict() for r in results],
        completed=True,
        runtime=runtime,
        consolidated_summary=consolidated_summary,
    )
    return results

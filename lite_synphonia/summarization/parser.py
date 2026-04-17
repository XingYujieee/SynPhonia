from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .heuristic_summarizer import normalize_keywords


@dataclass
class ParsedSummary:
    keywords: list[str]
    summary: str
    status: str
    error_message: str = ""


def parse_model_response(raw_response: str) -> ParsedSummary:
    if not raw_response.strip():
        return ParsedSummary([], "", "empty_response", "Model returned empty text.")

    candidate = _strip_code_fences(raw_response).strip()

    for text_candidate in _candidate_json_blocks(candidate):
        try:
            payload = json.loads(text_candidate)
        except json.JSONDecodeError:
            continue

        parsed = _normalize_payload(payload)
        if parsed.status == "success":
            return parsed

    fallback = _parse_labeled_text(candidate)
    if fallback.status == "success":
        return fallback

    recovered = _recover_partial_json(candidate)
    if recovered.status == "success_recovered":
        return recovered

    return ParsedSummary(
        [],
        "",
        "invalid_format",
        "Could not parse a valid keywords/summary structure from the model output.",
    )


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if not lines:
            return stripped

        remaining_lines = lines[1:]
        if remaining_lines and remaining_lines[-1].strip() == "```":
            remaining_lines = remaining_lines[:-1]
        return "\n".join(remaining_lines)
    return stripped


def _candidate_json_blocks(text: str) -> list[str]:
    candidates = [text]
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidates.append(text[first_brace : last_brace + 1])
    return candidates


def _normalize_payload(payload: object) -> ParsedSummary:
    if not isinstance(payload, dict):
        return ParsedSummary([], "", "invalid_format", "JSON payload is not an object.")

    keywords_value = payload.get("keywords")
    summary_value = payload.get("summary")

    if isinstance(keywords_value, str):
        keywords = [item.strip() for item in keywords_value.split(",") if item.strip()]
    elif isinstance(keywords_value, list):
        keywords = [str(item).strip() for item in keywords_value if str(item).strip()]
    else:
        keywords = []

    keywords = normalize_keywords(keywords)[:8]
    summary = " ".join(str(summary_value).split()) if summary_value is not None else ""

    if not keywords:
        return ParsedSummary([], "", "invalid_format", "Missing or empty keywords field.")
    if not summary:
        return ParsedSummary([], "", "invalid_format", "Missing or empty summary field.")

    return ParsedSummary(keywords, summary, "success")


def _parse_labeled_text(text: str) -> ParsedSummary:
    keyword_match = re.search(r"keywords?\s*:\s*(.+)", text, re.IGNORECASE)
    summary_match = re.search(r"summary\s*:\s*(.+)", text, re.IGNORECASE | re.DOTALL)

    if not keyword_match or not summary_match:
        return ParsedSummary([], "", "invalid_format")

    keywords = [
        item.strip(" -")
        for item in keyword_match.group(1).split(",")
        if item.strip(" -")
    ]
    keywords = normalize_keywords(keywords)[:8]
    summary = " ".join(summary_match.group(1).split()).strip()

    if not keywords or not summary:
        return ParsedSummary([], "", "invalid_format")

    return ParsedSummary(keywords, summary, "success")


def _recover_partial_json(text: str) -> ParsedSummary:
    keywords = _extract_partial_keywords(text)
    summary = _extract_partial_summary(text)

    if not keywords or not summary:
        return ParsedSummary([], "", "invalid_format")

    return ParsedSummary(
        keywords,
        summary,
        "success_recovered",
        "Recovered a usable summary from truncated JSON-like model output.",
    )


def _extract_partial_keywords(text: str) -> list[str]:
    match = re.search(r'"keywords"\s*:\s*\[(?P<body>.*?)(?:\]|\n\s*"summary"|$)', text, re.DOTALL)
    if not match:
        return []

    body = match.group("body")
    candidates = re.findall(r'"((?:\\.|[^"\\])*)"', body)
    keywords = [_decode_json_like_string(item) for item in candidates]
    return normalize_keywords(keywords)[:8]


def _extract_partial_summary(text: str) -> str:
    match = re.search(r'"summary"\s*:\s*"(?P<body>(?:\\.|[^"\\])*)', text, re.DOTALL)
    if not match:
        return ""

    summary = _decode_json_like_string(match.group("body"))
    return _trim_recovered_summary(summary)


def _decode_json_like_string(text: str) -> str:
    try:
        decoded = json.loads(f'"{text}"')
    except json.JSONDecodeError:
        decoded = (
            text.replace('\\"', '"')
            .replace("\\n", " ")
            .replace("\\r", " ")
            .replace("\\t", " ")
            .replace("\\/", "/")
        )
    return " ".join(str(decoded).split()).strip()


def _trim_recovered_summary(summary: str) -> str:
    cleaned = summary.strip().strip("`")
    if not cleaned:
        return ""
    if cleaned.endswith((".", "!", "?", "。", "！", "？")):
        return cleaned
    return cleaned.rstrip(",;:，；：") + "."

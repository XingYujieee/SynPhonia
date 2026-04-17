"""Transcription quality checks and stage-status builder."""

from __future__ import annotations

import math
from typing import Any

from ..stage_state import (
    STAGE_STATUS_BLOCKED,
    STAGE_STATUS_SUCCESS,
    STAGE_STATUS_WARNING,
    make_stage_status,
)


def _mean_confidence(results: list[dict[str, Any]]) -> float:
    vals = [float(r.get("confidence", 0.0) or 0.0) for r in results]
    return sum(vals) / len(vals) if vals else 0.0


def assess_transcription_quality(
    results: list[dict[str, Any]],
    *,
    metrics: dict[str, Any],
    silence_floor: float,
    limiter_level: float,
    glossary_terms: list[str] | None = None,
) -> dict[str, Any]:
    text = " ".join(str(r.get("text", "")).strip() for r in results).strip()
    total_chars = len(text)
    total_words = len(text.split()) if text else 0
    confidence = _mean_confidence(results)

    checks: list[dict[str, Any]] = []

    has_content = bool(results) and total_chars >= 8
    checks.append({"name": "has_content", "pass": has_content, "value": total_chars})

    conf_pass = confidence >= 0.45
    checks.append({"name": "confidence", "pass": conf_pass, "value": round(confidence, 4)})

    rms = float(metrics.get("rms", 0.0) or 0.0)
    audio_ok = rms >= max(1e-6, silence_floor * 1.2)
    checks.append({"name": "audio_level", "pass": audio_ok, "value": round(rms, 6)})

    clipped = float(metrics.get("peak", 0.0) or 0.0) >= max(0.1, limiter_level * 1.02)
    checks.append({"name": "clipping", "pass": not clipped, "value": bool(clipped)})

    glossary = glossary_terms or []
    glossary_hits = 0
    if glossary and text:
        lowered = text.lower()
        glossary_hits = sum(1 for t in glossary if t.lower() in lowered)
    checks.append({"name": "glossary_hits", "pass": True, "value": glossary_hits})

    pass_count = sum(1 for c in checks if c["pass"])
    score = pass_count / len(checks) if checks else 0.0

    if not has_content:
        decision = "fail"
        reason = "No usable transcript content was produced."
    elif confidence < 0.3:
        decision = "fail"
        reason = "Model confidence is too low."
    elif score < 0.6:
        decision = "warn"
        reason = "Transcript quality is unstable and needs review."
    elif confidence < 0.5:
        decision = "warn"
        reason = "Confidence is acceptable but below ideal range."
    else:
        decision = "pass"
        reason = "Transcript quality passed heuristic checks."

    return {
        "decision": decision,
        "reason": reason,
        "score": round(score, 4),
        "stats": {
            "segments": len(results),
            "chars": total_chars,
            "words": total_words,
            "mean_confidence": round(confidence, 4),
            "rms": round(rms, 6),
            "peak": round(float(metrics.get("peak", 0.0) or 0.0), 6),
        },
        "checks": checks,
    }


def build_transcription_stage_status(
    quality: dict[str, Any],
    *,
    has_results: bool,
) -> dict[str, Any]:
    decision = str(quality.get("decision", "pass"))
    if decision == "fail" or not has_results:
        status = STAGE_STATUS_BLOCKED
    elif decision == "warn":
        status = STAGE_STATUS_WARNING
    else:
        status = STAGE_STATUS_SUCCESS

    return make_stage_status(
        "transcription",
        status,
        str(quality.get("reason", "Transcription completed.")),
        details={
            "decision": decision,
            "score": quality.get("score"),
        },
    )

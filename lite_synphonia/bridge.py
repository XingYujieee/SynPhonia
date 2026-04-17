"""Bridge helpers for pipeline data exchange and consolidated output."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class TranscriptUnit:
    unit_id: int
    text: str
    t0: float
    t1: float
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "text": self.text,
            "t0": self.t0,
            "t1": self.t1,
            "source": self.source,
        }


@dataclass
class TranscriptBundle:
    units: list[TranscriptUnit] = field(default_factory=list)
    source_label: str = ""

    @property
    def unit_count(self) -> int:
        return len(self.units)

    @property
    def text(self) -> str:
        return "\n".join(u.text for u in self.units if u.text)

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_label": self.source_label,
            "unit_count": self.unit_count,
            "word_count": self.word_count,
            "text": self.text,
            "units": [u.to_dict() for u in self.units],
        }


def extract_transcript_bundle_from_payload(
    payload: dict[str, Any],
    source_label: str = "",
) -> TranscriptBundle:
    results = payload.get("results", []) if isinstance(payload, dict) else []
    units: list[TranscriptUnit] = []
    for i, r in enumerate(results):
        if not isinstance(r, dict):
            continue
        text = str(r.get("text", "")).strip()
        if not text:
            continue
        t0 = float(r.get("t0", r.get("start", 0.0)) or 0.0)
        t1 = float(r.get("t1", r.get("end", t0)) or t0)
        if t1 < t0:
            t1 = t0
        units.append(TranscriptUnit(i, text, t0, t1, source_label))
    return TranscriptBundle(units=units, source_label=source_label)


def write_transcript_input(path: str | Path, bundle: TranscriptBundle) -> None:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(bundle.text, encoding="utf-8")


def write_json_payload(path: str | Path, payload: dict[str, Any]) -> None:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_consolidated_bundle(
    path: str | Path,
    *,
    source: dict[str, Any],
    transcript: TranscriptBundle,
    transcription_payload: dict[str, Any],
    summary_payload: dict[str, Any],
    pdf_matching_payload: dict[str, Any],
    stage_status: dict[str, Any],
) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "stage_status": stage_status,
        "transcript": transcript.to_dict(),
        "transcription": transcription_payload,
        "summary": summary_payload,
        "pdf_matching": pdf_matching_payload,
    }
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return dest

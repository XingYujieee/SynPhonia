"""Datamodels for transcription telemetry."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class TranscriptionEvent:
    session_id: str
    event_id: str
    source_chunk_id: str
    timestamp_start: float
    timestamp_end: float
    text: str
    language: str
    confidence: float
    is_final: bool
    model_name: str
    speaker_id: str | None
    tokens_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TranscriptionMetrics:
    mode: str
    language: str
    requested_seconds: float
    recorded_seconds: float
    sample_count: int
    chunk_count: int = 0
    rms: float = 0.0
    peak: float = 0.0
    transcription_seconds: float = 0.0
    raw_segments: int = 0
    cleaned_segments: int = 0
    filtered_segments: int = 0
    emitted_segments: int = 0
    final_segments: int = 0
    transcribe_calls: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

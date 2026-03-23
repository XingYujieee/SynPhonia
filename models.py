"""Minimal shared models for the standalone audio test."""

from __future__ import annotations

import enum
from dataclasses import asdict, dataclass, field
from typing import Any

from pydantic import BaseModel


class SegmentStatus(enum.Enum):
    PARTIAL = "partial"
    FINAL = "final"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class TranscriptionSegment:
    segment_id: str
    text: str
    start_time: float
    end_time: float
    status: SegmentStatus
    is_final: bool = False
    language: str = "en"
    confidence: float = 1.0
    cumulative_word_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class TranscriptionEvent(BaseModel):
    session_id: str
    event_id: str
    event_type: str = "transcription"
    source_chunk_id: str
    timestamp_start: float
    timestamp_end: float
    text: str
    language: str = "zh"
    confidence: float = 0.0
    is_final: bool
    model_name: str
    speaker_id: str | None = None
    tokens_count: int

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


@dataclass(slots=True)
class TranscriptionMetrics:
    mode: str
    language: str
    requested_seconds: float
    recorded_seconds: float
    sample_count: int
    chunk_count: int = 0
    rms: float = 0.0
    peak: float = 0.0
    transcribe_calls: int = 0
    raw_segments: int = 0
    cleaned_segments: int = 0
    filtered_segments: int = 0
    emitted_segments: int = 0
    final_segments: int = 0
    transcription_seconds: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

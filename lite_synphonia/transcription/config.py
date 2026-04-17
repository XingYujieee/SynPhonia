"""Configuration objects for transcription runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TranscriptionConfig:
    sample_rate: int = 16_000
    channels: int = 1
    chunk_duration_ms: int = 500
    silence_floor: float = 0.003

    # Audio enhancement chain defaults — kept in sync with LiteConfig /
    # TranscriptionAPIConfig so that standalone use of TranscriptionConfig
    # produces the same conservative behaviour.
    input_gain: float = 1.0
    target_rms: float = 0.03    # bidirectional AGC target (was 0.06)
    max_gain: float = 2.0       # amplification cap (was 3.0)
    limiter_level: float = 0.72 # soft-knee ceiling (was 0.88)
    # First-order pre-emphasis coefficient α for y[n] = x[n] - α·x[n-1].
    # 0.97 is the ASR-standard value (Kaldi / ESPnet default).
    # Set to 0.0 to disable pre-emphasis.
    pre_emphasis_coeff: float = 0.97

    language: str = "zh"
    prefer_simplified_chinese: bool = True
    initial_prompt: str = ""
    glossary_terms: list[str] = field(default_factory=list)


def build_runtime_summary(cfg: TranscriptionConfig) -> dict[str, Any]:
    return {
        "sample_rate": cfg.sample_rate,
        "channels": cfg.channels,
        "chunk_duration_ms": cfg.chunk_duration_ms,
        "silence_floor": cfg.silence_floor,
        "input_gain": cfg.input_gain,
        "target_rms": cfg.target_rms,
        "max_gain": cfg.max_gain,
        "limiter_level": cfg.limiter_level,
        "pre_emphasis_coeff": cfg.pre_emphasis_coeff,
        "language": cfg.language,
        "has_initial_prompt": bool(cfg.initial_prompt.strip()),
        "glossary_size": len(cfg.glossary_terms),
    }

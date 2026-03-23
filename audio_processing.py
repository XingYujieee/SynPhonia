"""Audio preprocessing helpers for noisy microphone conditions."""

from __future__ import annotations

import numpy as np

from config import TranscriptionConfig


def estimate_rms(samples: np.ndarray) -> float:
    if len(samples) == 0:
        return 0.0
    audio = np.ascontiguousarray(samples, dtype=np.float32)
    return float(np.sqrt(np.mean(np.square(audio))) + 1e-8)


def is_active_audio(samples: np.ndarray, cfg: TranscriptionConfig) -> bool:
    return estimate_rms(samples) >= cfg.silence_floor


def enhance_audio(samples: np.ndarray, cfg: TranscriptionConfig) -> np.ndarray:
    if len(samples) == 0:
        return samples.astype(np.float32, copy=False)

    audio = np.ascontiguousarray(samples, dtype=np.float32)
    rms = estimate_rms(audio)

    gain = cfg.input_gain
    if rms >= cfg.silence_floor:
        gain = min(cfg.max_gain, max(cfg.input_gain, cfg.target_rms / rms))

    audio = audio * gain
    return np.clip(audio, -cfg.limiter_level, cfg.limiter_level).astype(np.float32, copy=False)

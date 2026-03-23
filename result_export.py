"""Helpers for exporting audio samples and transcription results."""

from __future__ import annotations

import json
import wave
from pathlib import Path
from typing import Any

import numpy as np


def ensure_output_dir(path: str | Path | None) -> Path | None:
    if not path:
        return None
    output_dir = Path(path).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def write_wav(path: str | Path, audio: np.ndarray, sample_rate: int) -> None:
    samples = np.ascontiguousarray(audio, dtype=np.float32)
    pcm16 = np.clip(samples, -1.0, 1.0)
    pcm16 = (pcm16 * 32767.0).astype(np.int16, copy=False)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.tobytes())


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

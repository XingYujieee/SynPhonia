"""Standalone config for the audio-to-text test harness."""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

_AUDIO_TEST_ROOT = Path(__file__).resolve().parent
_PROJECT_ROOT = _AUDIO_TEST_ROOT.parent
_VENDOR_WHISPER = _PROJECT_ROOT / "vendor" / "whisper"
_SYNPHONIE_VENDOR_WHISPER = _PROJECT_ROOT / "SynPhonie" / "vendor" / "whisper"
_LOCAL_MODELS = _AUDIO_TEST_ROOT / "models"
_LOCAL_VENDOR_WHISPER = _AUDIO_TEST_ROOT / "vendor" / "whisper"
_MODEL_FILENAMES = {
    "tiny": "ggml-tiny.bin",
    "base": "ggml-base.bin",
    "small": "ggml-small.bin",
}
_WHISPER_LIB_NAMES = {
    "Darwin": "libwhisper.dylib",
    "Linux": "libwhisper.so",
    "Windows": "whisper.dll",
}


def _candidate_whisper_dirs() -> tuple[Path, ...]:
    env_dirs: list[Path] = []
    for env_name in ("SYNPHONIE_WHISPER_SEARCH_DIR", "WHISPER_CPP_DIR"):
        raw = os.environ.get(env_name, "").strip()
        if raw:
            env_dirs.append(Path(raw).expanduser())

    candidates = [
        _LOCAL_VENDOR_WHISPER,
        _LOCAL_MODELS,
        _VENDOR_WHISPER,
        _SYNPHONIE_VENDOR_WHISPER,
        _AUDIO_TEST_ROOT,
        _PROJECT_ROOT,
        *env_dirs,
    ]

    deduped: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        if resolved in seen:
            continue
        deduped.append(resolved)
        seen.add(resolved)
    return tuple(deduped)


def _default_whisper_library_path() -> str:
    system = platform.system()
    lib_name = _WHISPER_LIB_NAMES.get(system)
    if lib_name is None:
        raise OSError(f"Unsupported platform for Whisper library: {system}")

    candidates: list[Path] = []
    for root in _candidate_whisper_dirs():
        if root.name == "whisper":
            candidates.append(root / lib_name)
        else:
            candidates.append(root / lib_name)
            candidates.append(root / "whisper" / lib_name)
            candidates.append(root / "vendor" / "whisper" / lib_name)

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str((_LOCAL_VENDOR_WHISPER / lib_name).resolve(strict=False))


def _iter_model_candidates(model_names: Iterable[str]) -> list[Path]:
    candidates: list[Path] = []
    for model_name in model_names:
        filename = _MODEL_FILENAMES[model_name]
        for root in _candidate_whisper_dirs():
            if root.name == "models":
                candidates.append(root / filename)
            else:
                candidates.append(root / filename)
                candidates.append(root / "models" / filename)
                candidates.append(root / "vendor" / "whisper" / filename)
    return candidates


def list_available_models() -> dict[str, str]:
    available: dict[str, str] = {}
    for model_name in _MODEL_FILENAMES:
        for candidate in _iter_model_candidates((model_name,)):
            if candidate.exists():
                available[model_name] = str(candidate)
                break
    return available


def resolve_model_path(model_name: str = "auto") -> str:
    model_name = model_name.strip().lower()
    if model_name == "auto":
        preferred_order = ("small", "base", "tiny")
    else:
        if model_name not in _MODEL_FILENAMES:
            supported = ", ".join(sorted(_MODEL_FILENAMES))
            raise ValueError(
                f"Unsupported model '{model_name}'. Supported values: {supported}, auto"
            )
        preferred_order = (model_name,)

    candidates = _iter_model_candidates(preferred_order)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    fallback_name = preferred_order[0]
    return str((_LOCAL_MODELS / _MODEL_FILENAMES[fallback_name]).resolve(strict=False))


def build_runtime_summary(
    cfg: "TranscriptionConfig",
) -> dict[str, str | int | float | bool]:
    return {
        "library_path": cfg.library_path,
        "model_path": cfg.model_path,
        "backend": cfg.backend,
        "language": cfg.language,
        "sample_rate": cfg.sample_rate,
        "channels": cfg.channels,
        "chunk_duration_ms": cfg.chunk_duration_ms,
        "accumulation_seconds": cfg.accumulation_seconds,
        "prefer_simplified_chinese": cfg.prefer_simplified_chinese,
    }


@dataclass
class TranscriptionConfig:
    library_path: str = field(
        default_factory=lambda: os.environ.get(
            "SYNPHONIE_WHISPER_LIBRARY_PATH",
            _default_whisper_library_path(),
        ),
    )
    model_path: str = field(
        default_factory=lambda: os.environ.get(
            "SYNPHONIE_WHISPER_MODEL_PATH",
            resolve_model_path("auto"),
        ),
    )
    backend: str = field(
        default_factory=lambda: (
            os.environ.get(
                "SYNPHONIE_WHISPER_BACKEND",
                "auto",
            )
            .strip()
            .lower()
        ),
    )
    language: str = "zh"
    sample_rate: int = 16000
    channels: int = 1
    chunk_duration_ms: int = 500
    accumulation_seconds: float = 4.0
    initial_prompt: str = ""
    prefer_simplified_chinese: bool = True
    input_gain: float = 1.4
    target_rms: float = 0.05
    max_gain: float = 3.0
    silence_floor: float = 0.003
    limiter_level: float = 0.92

    def available_models(self) -> dict[str, str]:
        return list_available_models()

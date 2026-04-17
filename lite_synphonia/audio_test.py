"""Terminal diagnostics for microphone capture."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
from typing import Sequence

import numpy as np

from .transcription.audio import MicrophoneAudioSource
from .transcription.audio_processing import enhance_audio
from .transcription.config import TranscriptionConfig
from .transcription.preflight import analyze_preflight_audio


def _estimate_rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio.astype(np.float32)))))


def _estimate_peak(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.max(np.abs(audio.astype(np.float32))))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m lite_synphonia audio-test",
        description="Monitor microphone chunks in terminal and print audio levels.",
    )
    parser.add_argument("--seconds", type=float, default=8.0, help="Monitor duration in seconds.")
    parser.add_argument(
        "--chunk-ms",
        type=int,
        default=500,
        help="Chunk size in milliseconds. Default: 500.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Audio sample rate. Default: 16000.",
    )
    parser.add_argument(
        "--silence-floor",
        type=float,
        default=0.003,
        help="Silence floor used by the backend enhancement chain.",
    )
    return parser


def _render_level_bar(value: float, *, width: int = 20) -> str:
    clipped = max(0.0, min(1.0, value))
    filled = min(width, int(round(clipped * width)))
    return "█" * filled + "·" * (width - filled)


async def _monitor_audio(args: argparse.Namespace) -> int:
    try:
        import sounddevice as sd  # type: ignore
    except Exception as exc:
        print(f"[audio-test] 无法导入 sounddevice: {exc}")
        print("[audio-test] 请先安装依赖后再试。")
        return 1

    cfg = TranscriptionConfig(
        sample_rate=max(8000, int(args.sample_rate)),
        channels=1,
        chunk_duration_ms=max(100, int(args.chunk_ms)),
        silence_floor=max(0.0, float(args.silence_floor)),
    )

    try:
        default_input, _ = sd.default.device
    except Exception:
        default_input = None

    try:
        device_info = (
            sd.query_devices(default_input, "input")
            if default_input is not None and default_input >= 0
            else None
        )
    except Exception:
        device_info = None

    if device_info:
        print(
            f"[audio-test] 当前默认输入设备: {device_info.get('name', 'unknown')} "
            f"(index={default_input})",
        )
    else:
        print("[audio-test] 未能识别默认输入设备，将直接尝试采集。")

    print(
        f"[audio-test] 开始监听 {float(args.seconds):.1f}s, "
        f"sample_rate={cfg.sample_rate}, chunk={cfg.chunk_duration_ms}ms",
    )
    print("[audio-test] 你现在可以直接说话。")

    source = MicrophoneAudioSource(
        sample_rate=cfg.sample_rate,
        channels=cfg.channels,
        chunk_duration_ms=cfg.chunk_duration_ms,
    )

    raw_chunks: list[np.ndarray] = []
    enhanced_chunks: list[np.ndarray] = []
    chunk_count = max(1, int(round(float(args.seconds) * 1000 / cfg.chunk_duration_ms)))

    try:
        await source.start()
        for chunk_index in range(chunk_count):
            chunk_bytes = await source.read_chunk()
            raw_audio = np.frombuffer(chunk_bytes, dtype=np.float32).copy()
            enhanced_audio = enhance_audio(raw_audio, cfg)

            raw_chunks.append(raw_audio)
            enhanced_chunks.append(enhanced_audio)

            raw_rms = _estimate_rms(raw_audio)
            raw_peak = _estimate_peak(raw_audio)
            enhanced_rms = _estimate_rms(enhanced_audio)
            enhanced_peak = _estimate_peak(enhanced_audio)

            print(
                f"[chunk {chunk_index + 1:02d}/{chunk_count:02d}] "
                f"raw_rms={raw_rms:.4f} raw_peak={raw_peak:.4f} "
                f"enhanced_rms={enhanced_rms:.4f} enhanced_peak={enhanced_peak:.4f}",
            )
            print(
                f"  raw      {_render_level_bar(raw_peak)}  "
                f"enhanced {_render_level_bar(enhanced_peak)}",
            )
    finally:
        await source.stop()

    merged_raw = np.concatenate(raw_chunks) if raw_chunks else np.zeros((0,), dtype=np.float32)
    merged_enhanced = (
        np.concatenate(enhanced_chunks) if enhanced_chunks else np.zeros((0,), dtype=np.float32)
    )

    raw_report = analyze_preflight_audio(
        seconds=float(args.seconds),
        rms=_estimate_rms(merged_raw),
        peak=_estimate_peak(merged_raw),
        silence_floor=cfg.silence_floor,
        limiter_level=cfg.limiter_level,
    )
    enhanced_report = analyze_preflight_audio(
        seconds=float(args.seconds),
        rms=_estimate_rms(merged_enhanced),
        peak=_estimate_peak(merged_enhanced),
        silence_floor=cfg.silence_floor,
        limiter_level=cfg.limiter_level,
    )

    print("\n[audio-test] 原始音频诊断:")
    print(asdict(raw_report))
    print("[audio-test] 增强后音频诊断:")
    print(asdict(enhanced_report))

    return 0


def run_audio_test_command(argv: Sequence[str]) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv))
    try:
        return asyncio.run(_monitor_audio(args))
    except KeyboardInterrupt:
        print("\n[audio-test] 已中断。")
        return 130

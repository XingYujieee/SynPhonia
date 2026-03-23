#!/usr/bin/env python3
"""Standalone microphone-to-text test runner."""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
import uuid
from pathlib import Path

import numpy as np

from audio_processing import enhance_audio
from config import TranscriptionConfig, build_runtime_summary, list_available_models, resolve_model_path
from models import TranscriptionEvent, TranscriptionMetrics, TranscriptionSegment
from result_export import ensure_output_dir, write_json, write_wav
from text_postprocess import is_plausible_text, normalize_text
from whisper_binding import WhisperLib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def _clean_results(results: list[dict], cfg: TranscriptionConfig) -> list[dict]:
    cleaned_results: list[dict] = []
    for result in results:
        text = normalize_text(
            result["text"],
            prefer_simplified_chinese=cfg.prefer_simplified_chinese,
        )
        if not is_plausible_text(text):
            continue
        cleaned = dict(result)
        cleaned["text"] = text
        cleaned_results.append(cleaned)
    return cleaned_results


def _normalize_results(results: list[dict], cfg: TranscriptionConfig) -> list[dict]:
    normalized_results: list[dict] = []
    for result in results:
        text = normalize_text(
            result["text"],
            prefer_simplified_chinese=cfg.prefer_simplified_chinese,
        )
        if not text or not is_plausible_text(text):
            continue
        normalized = dict(result)
        normalized["text"] = text
        normalized_results.append(normalized)
    return normalized_results


def _merge_result_text(results: list[dict]) -> str:
    merged: list[str] = []
    last_text = ""
    for result in results:
        text = result["text"].strip()
        if not text:
            continue
        if text == last_text or text in merged:
            continue
        merged.append(text)
        last_text = text
    return "".join(merged)


def _audio_stats(audio: np.ndarray) -> tuple[float, float]:
    if len(audio) == 0:
        return 0.0, 0.0
    rms = float(np.sqrt(np.mean(np.square(audio))))
    peak = float(np.max(np.abs(audio)))
    return rms, peak


def _print_metrics(metrics: TranscriptionMetrics) -> None:
    print("\n运行指标:")
    print(f"mode={metrics.mode}, language={metrics.language}")
    print(
        f"recorded={metrics.recorded_seconds:.2f}s, samples={metrics.sample_count}, "
        f"chunks={metrics.chunk_count}, rms={metrics.rms:.4f}, peak={metrics.peak:.4f}"
    )
    print(
        f"transcribe_calls={metrics.transcribe_calls}, raw_segments={metrics.raw_segments}, "
        f"cleaned_segments={metrics.cleaned_segments}, filtered_segments={metrics.filtered_segments}, "
        f"emitted_segments={metrics.emitted_segments}, final_segments={metrics.final_segments}, "
        f"transcribe_time={metrics.transcription_seconds:.3f}s"
    )
    for note in metrics.notes:
        print(f"note={note}")


def _serialize_results(results: list[dict]) -> list[dict]:
    serialized: list[dict] = []
    for result in results:
        serialized.append(
            {
                "text": result["text"],
                "t0": result["t0"],
                "t1": result["t1"],
                "lang": result.get("lang", ""),
            }
        )
    return serialized


def _model_name_from_path(model_path: str) -> str:
    stem = Path(model_path).stem
    return stem.removeprefix("ggml-")


def _token_count(text: str) -> int:
    if not text.strip():
        return 0
    pieces = text.split()
    if pieces:
        return len(pieces)
    return len(text.strip())


def _result_to_event(
    result: dict,
    *,
    session_id: str,
    source_chunk_id: str,
    model_name: str,
    is_final: bool = True,
) -> TranscriptionEvent:
    return TranscriptionEvent(
        session_id=session_id,
        event_id=uuid.uuid4().hex,
        source_chunk_id=source_chunk_id,
        timestamp_start=float(result["t0"]),
        timestamp_end=float(result["t1"]),
        text=result["text"],
        language=result.get("lang", "zh"),
        confidence=float(result.get("confidence", 0.0)),
        is_final=is_final,
        model_name=model_name,
        speaker_id=result.get("speaker_id"),
        tokens_count=_token_count(result["text"]),
    )


def _segment_to_event(
    segment: TranscriptionSegment,
    *,
    session_id: str,
    model_name: str,
) -> TranscriptionEvent:
    return TranscriptionEvent(
        session_id=session_id,
        event_id=uuid.uuid4().hex,
        source_chunk_id=segment.segment_id,
        timestamp_start=float(segment.start_time),
        timestamp_end=float(segment.end_time),
        text=segment.text,
        language=segment.language,
        confidence=float(segment.confidence),
        is_final=segment.is_final,
        model_name=model_name,
        speaker_id=None,
        tokens_count=_token_count(segment.text),
    )


def _export_run_artifacts(
    output_dir: str | None,
    *,
    cfg: TranscriptionConfig,
    metrics: TranscriptionMetrics,
    events: list[TranscriptionEvent] | None = None,
    results: list[dict] | None = None,
    raw_audio: np.ndarray | None = None,
    enhanced_audio: np.ndarray | None = None,
    extra: dict | None = None,
) -> None:
    destination = ensure_output_dir(output_dir)
    if destination is None:
        return

    if raw_audio is not None and len(raw_audio) > 0:
        write_wav(destination / "raw_audio.wav", raw_audio, cfg.sample_rate)
    if enhanced_audio is not None and len(enhanced_audio) > 0:
        write_wav(destination / "enhanced_audio.wav", enhanced_audio, cfg.sample_rate)

    payload = {
        "runtime": build_runtime_summary(cfg),
        "metrics": metrics.to_dict(),
        "events": [event.to_dict() for event in (events or [])],
        "results": _serialize_results(results or []),
    }
    if extra:
        payload.update(extra)
    write_json(destination / "transcription.json", payload)
    write_json(destination / "events.json", {"events": [event.to_dict() for event in (events or [])]})
    print(f"\n导出结果目录: {destination}")


def _transcribe_once(
    lib: WhisperLib,
    audio: np.ndarray,
    cfg: TranscriptionConfig,
    language: str,
) -> tuple[list[dict], dict[str, float | int]]:
    started_at = time.perf_counter()
    raw_results = lib.transcribe(audio, language=language, initial_prompt=cfg.initial_prompt)
    logging.info("Whisper raw segments (%s): %d", language, len(raw_results))
    cleaned_results = _clean_results(raw_results, cfg)
    logging.info("Whisper cleaned segments (%s): %d", language, len(cleaned_results))
    stats: dict[str, float | int] = {
        "transcribe_calls": 1,
        "raw_segments": len(raw_results),
        "cleaned_segments": len(cleaned_results),
        "filtered_segments": max(0, len(raw_results) - len(cleaned_results)),
        "transcription_seconds": time.perf_counter() - started_at,
    }
    if cleaned_results:
        return cleaned_results, stats
    if raw_results:
        logging.info(
            "Raw transcription produced %d segments for %s, but all were filtered; falling back to normalized raw text.",
            len(raw_results),
            language,
        )
        normalized = _normalize_results(raw_results, cfg)
        stats["cleaned_segments"] = len(normalized)
        return normalized, stats
    return [], stats


def _transcribe_audio(
    audio: np.ndarray,
    cfg: TranscriptionConfig,
    language: str,
    metrics: TranscriptionMetrics,
) -> list[dict]:
    lib = WhisperLib(cfg.library_path)
    try:
        lib.init_model(cfg.model_path, backend=cfg.backend)
        results, stats = _transcribe_once(lib, audio, cfg, language)
        metrics.transcribe_calls += int(stats["transcribe_calls"])
        metrics.raw_segments += int(stats["raw_segments"])
        metrics.cleaned_segments += int(stats["cleaned_segments"])
        metrics.filtered_segments += int(stats["filtered_segments"])
        metrics.transcription_seconds += float(stats["transcription_seconds"])
        metrics.emitted_segments += len(results)
        metrics.final_segments += len(results)
        return results
    finally:
        lib.close()


def _print_header(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


async def _record_audio(seconds: float, cfg: TranscriptionConfig) -> tuple[np.ndarray, np.ndarray, int]:
    from audio import MicrophoneAudioSource

    mic = MicrophoneAudioSource(
        sample_rate=cfg.sample_rate,
        channels=cfg.channels,
        chunk_duration_ms=cfg.chunk_duration_ms,
    )
    await mic.start()
    print(f"请开始说话，正在录音 {seconds:.1f} 秒...")

    chunks: list[bytes] = []
    enhanced_chunks: list[np.ndarray] = []
    start = time.monotonic()
    while time.monotonic() - start < seconds:
        chunk = await asyncio.wait_for(mic.read_chunk(), timeout=2.0)
        if chunk:
            chunks.append(chunk)
            enhanced_chunks.append(enhance_audio(np.frombuffer(chunk, dtype=np.float32), cfg))

    await mic.stop()
    if not chunks:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32), 0
    raw_audio = np.concatenate([np.frombuffer(chunk, dtype=np.float32) for chunk in chunks])
    enhanced_audio = (
        np.concatenate(enhanced_chunks) if enhanced_chunks else np.array([], dtype=np.float32)
    )
    return raw_audio, enhanced_audio, len(chunks)


def _build_metrics(mode: str, language: str, requested_seconds: float) -> TranscriptionMetrics:
    return TranscriptionMetrics(
        mode=mode,
        language=language,
        requested_seconds=requested_seconds,
        recorded_seconds=0.0,
        sample_count=0,
    )


async def _run_direct_mode(
    seconds: float,
    cfg: TranscriptionConfig,
    language: str,
    output_dir: str | None,
    compare_models: list[str] | None = None,
) -> int:
    _print_header("Direct Mode: Record Then Transcribe")
    session_id = uuid.uuid4().hex
    metrics = _build_metrics("direct", language, seconds)
    raw_audio, audio, chunk_count = await _record_audio(seconds, cfg)
    metrics.chunk_count = chunk_count
    metrics.sample_count = len(audio)
    metrics.recorded_seconds = len(audio) / cfg.sample_rate if cfg.sample_rate else 0.0
    print(f"采集到 {len(audio)} 个采样点，约 {metrics.recorded_seconds:.1f} 秒")
    rms, peak = _audio_stats(audio)
    metrics.rms = rms
    metrics.peak = peak
    print(f"音频能量: rms={rms:.4f}, peak={peak:.4f}")

    if compare_models:
        _run_model_comparison(audio, cfg, language, compare_models)

    results = _transcribe_audio(audio, cfg, language, metrics)
    model_name = _model_name_from_path(cfg.model_path)
    events = [
        _result_to_event(
            result,
            session_id=session_id,
            source_chunk_id="direct-chunk-00001",
            model_name=model_name,
            is_final=True,
        )
        for result in results
    ]

    if not results:
        print("没有识别到文本。")
        _print_metrics(metrics)
        _export_run_artifacts(
            output_dir,
            cfg=cfg,
            metrics=metrics,
            events=[],
            results=[],
            raw_audio=raw_audio,
            enhanced_audio=audio,
        )
        return 0

    print("\n转录结果:")
    for result in results:
        print(f"[{result['t0']:.1f}s - {result['t1']:.1f}s] ({result['lang']}) {result['text']}")
    _print_metrics(metrics)
    _export_run_artifacts(
        output_dir,
        cfg=cfg,
        metrics=metrics,
        events=events,
        results=results,
        raw_audio=raw_audio,
        enhanced_audio=audio,
    )
    return 0


async def _run_stream_mode(seconds: float, cfg: TranscriptionConfig, output_dir: str | None) -> int:
    from audio import MicrophoneAudioSource
    from transcription_engine import WhisperTranscriptionEngine

    _print_header("Stream Mode: Incremental Transcription")
    mic = MicrophoneAudioSource(
        sample_rate=cfg.sample_rate,
        channels=cfg.channels,
        chunk_duration_ms=cfg.chunk_duration_ms,
    )
    engine = WhisperTranscriptionEngine(cfg)
    session_id = uuid.uuid4().hex
    model_name = _model_name_from_path(cfg.model_path)
    events: list[TranscriptionEvent] = []
    recorded_chunks: list[bytes] = []
    raw_chunks: list[bytes] = []

    class RecordingAudioSource:
        sample_rate = mic.sample_rate
        channels = mic.channels

        async def read_chunk(self) -> bytes:
            chunk = await mic.read_chunk()
            if chunk:
                raw_chunks.append(chunk)
                enhanced = enhance_audio(np.frombuffer(chunk, dtype=np.float32), cfg).tobytes()
                recorded_chunks.append(enhanced)
                return enhanced
            return chunk

    await mic.start()
    await engine.start(RecordingAudioSource())
    print(
        f"请开始说话，实时转录进行中，持续 {seconds:.1f} 秒"
        f" (accumulation={cfg.accumulation_seconds:.1f}s)..."
    )

    async def stop_later() -> None:
        await asyncio.sleep(seconds)
        await mic.stop()
        await engine.stop()

    stopper = asyncio.create_task(stop_later())
    try:
        emitted = 0
        final_texts: list[str] = []
        async for segment in engine.segments():
            emitted += 1
            events.append(_segment_to_event(segment, session_id=session_id, model_name=model_name))
            if segment.is_final and segment.text:
                final_texts.append(segment.text)
            print(
                f"[{segment.start_time:.1f}s - {segment.end_time:.1f}s] "
                f"{segment.status.value.upper():7s} {segment.text}"
            )
    finally:
        await stopper

    full_audio = (
        np.concatenate([np.frombuffer(chunk, dtype=np.float32) for chunk in recorded_chunks])
        if recorded_chunks
        else np.array([], dtype=np.float32)
    )
    raw_audio = (
        np.concatenate([np.frombuffer(chunk, dtype=np.float32) for chunk in raw_chunks])
        if raw_chunks
        else np.array([], dtype=np.float32)
    )
    rms, peak = _audio_stats(full_audio)
    print(f"音频能量: rms={rms:.4f}, peak={peak:.4f}")
    metrics = engine.metrics
    metrics.requested_seconds = seconds
    metrics.rms = rms
    metrics.peak = peak
    final_results = (
        _transcribe_audio(full_audio, cfg, cfg.language, metrics) if len(full_audio) > 0 else []
    )
    final_events = [
        _result_to_event(
            result,
            session_id=session_id,
            source_chunk_id="stream-final-chunk-00001",
            model_name=model_name,
            is_final=True,
        )
        for result in final_results
    ]
    final_text = _merge_result_text(final_results)

    if final_text:
        print("\n最终转录结果:")
        print(final_text)
    elif emitted == 0:
        print("没有识别到文本。")
    elif final_texts:
        print("\n最终转录结果:")
        print(" ".join(final_texts))
    else:
        print("\n本次只产生了实时片段，没有稳定的最终文本。")

    _print_metrics(metrics)
    _export_run_artifacts(
        output_dir,
        cfg=cfg,
        metrics=metrics,
        events=events + final_events,
        results=final_results,
        raw_audio=raw_audio,
        enhanced_audio=full_audio,
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone audio-to-text test")
    parser.add_argument("--seconds", type=float, default=10.0, help="recording duration")
    parser.add_argument("--language", type=str, default="zh", help="Whisper language")
    parser.add_argument(
        "--model",
        choices=("auto", "tiny", "base", "small"),
        default="auto",
        help="Whisper model preset to resolve locally",
    )
    parser.add_argument("--model-path", type=str, default="", help="explicit model path override")
    parser.add_argument("--library-path", type=str, default="", help="explicit whisper library path override")
    parser.add_argument("--backend", choices=("auto", "cpu", "gpu"), default="auto", help="backend override")
    parser.add_argument("--output-dir", type=str, default="", help="write JSON and WAV artifacts into this directory")
    parser.add_argument(
        "--compare-models",
        type=str,
        default="",
        help="comma-separated model presets to run on the same captured audio",
    )
    parser.add_argument("--list-models", action="store_true", help="list discovered local models and exit")
    parser.add_argument("--log-level", type=str, default="INFO", help="logging level")
    parser.add_argument(
        "--mode",
        choices=("direct", "stream"),
        default="direct",
        help="direct: finish recording then transcribe; stream: incremental transcription",
    )
    parser.add_argument(
        "--skip-mic",
        action="store_true",
        help="use synthetic sine wave instead of microphone",
    )
    return parser


async def _run_skip_mic(cfg: TranscriptionConfig, language: str, output_dir: str | None) -> int:
    _print_header("Synthetic Audio Test")
    session_id = uuid.uuid4().hex
    t = np.linspace(0, 3, cfg.sample_rate * 3, dtype=np.float32)
    audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    metrics = _build_metrics("skip-mic", language, 3.0)
    metrics.sample_count = len(audio)
    metrics.recorded_seconds = len(audio) / cfg.sample_rate
    metrics.chunk_count = 1
    metrics.rms, metrics.peak = _audio_stats(audio)
    results = _transcribe_audio(audio, cfg, language, metrics)
    model_name = _model_name_from_path(cfg.model_path)
    events = [
        _result_to_event(
            result,
            session_id=session_id,
            source_chunk_id="synthetic-chunk-00001",
            model_name=model_name,
            is_final=True,
        )
        for result in results
    ]

    print(f"合成音频推理完成，返回 {len(results)} 个 segments")
    for result in results:
        print(f"[{result['t0']:.1f}s - {result['t1']:.1f}s] ({result['lang']}) {result['text']}")
    _print_metrics(metrics)
    _export_run_artifacts(
        output_dir,
        cfg=cfg,
        metrics=metrics,
        events=events,
        results=results,
        enhanced_audio=audio,
    )
    return 0


def _build_config_from_args(args: argparse.Namespace) -> TranscriptionConfig:
    cfg = TranscriptionConfig(language=args.language)
    if args.model_path:
        cfg.model_path = str(Path(args.model_path).expanduser().resolve())
    else:
        cfg.model_path = resolve_model_path(args.model)
    if args.library_path:
        cfg.library_path = str(Path(args.library_path).expanduser().resolve())
    cfg.backend = args.backend
    return cfg


def _run_model_comparison(audio: np.ndarray, cfg: TranscriptionConfig, language: str, models: list[str]) -> None:
    if len(audio) == 0:
        print("没有可用于比较的音频。")
        return

    print("\n模型对比:")
    for model_name in models:
        compare_cfg = TranscriptionConfig(
            library_path=cfg.library_path,
            model_path=resolve_model_path(model_name),
            backend=cfg.backend,
            language=language,
        )
        metrics = _build_metrics(f"compare:{model_name}", language, len(audio) / cfg.sample_rate)
        metrics.sample_count = len(audio)
        metrics.recorded_seconds = len(audio) / cfg.sample_rate
        metrics.rms, metrics.peak = _audio_stats(audio)
        try:
            results = _transcribe_audio(audio, compare_cfg, language, metrics)
            merged = _merge_result_text(results)
            print(f"- {model_name}: segments={len(results)}, time={metrics.transcription_seconds:.3f}s, text={merged}")
        except Exception as exc:
            print(f"- {model_name}: failed: {exc}")


async def main() -> int:
    args = _build_parser().parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper(), logging.INFO))

    if args.list_models:
        available = list_available_models()
        if not available:
            print("未发现可用模型。")
            return 0
        print("Discovered Whisper models:")
        for name, path in available.items():
            print(f"- {name}: {path}")
        return 0

    cfg = _build_config_from_args(args)
    logging.info("Runtime config: %s", build_runtime_summary(cfg))

    if args.skip_mic:
        return await _run_skip_mic(cfg, args.language, args.output_dir or None)
    if args.mode == "stream":
        return await _run_stream_mode(args.seconds, cfg, args.output_dir or None)
    compare_models = [item.strip().lower() for item in args.compare_models.split(",") if item.strip()]
    return await _run_direct_mode(
        args.seconds,
        cfg,
        args.language,
        args.output_dir or None,
        compare_models=compare_models,
    )


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

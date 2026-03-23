"""Standalone streaming transcription engine."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from typing import Protocol

import numpy as np

from audio_processing import is_active_audio
from config import TranscriptionConfig
from models import SegmentStatus, TranscriptionMetrics, TranscriptionSegment
from text_postprocess import (
    is_duplicate_or_extension,
    merge_incremental_text,
    is_plausible_text,
    normalize_text,
)
from whisper_binding import WhisperLib

log = logging.getLogger(__name__)


class AudioSource(Protocol):
    async def read_chunk(self) -> bytes: ...

    @property
    def sample_rate(self) -> int: ...

    @property
    def channels(self) -> int: ...


class WhisperTranscriptionEngine:
    def __init__(self, config: TranscriptionConfig) -> None:
        self._config = config
        self._whisper = WhisperLib(config.library_path)
        self._audio: AudioSource | None = None
        self._running = False
        self._seg_counter = 0
        self._cumulative_words = 0
        self._last_partial_text = ""
        self._last_final_text = ""
        self._metrics = TranscriptionMetrics(
            mode="stream",
            language=config.language,
            requested_seconds=0.0,
            recorded_seconds=0.0,
            sample_count=0,
        )

    async def start(self, audio_source: AudioSource) -> None:
        self._audio = audio_source
        self._whisper.init_model(
            self._config.model_path,
            backend=self._config.backend,
        )
        self._running = True

    async def stop(self) -> None:
        self._running = False

    @property
    def metrics(self) -> TranscriptionMetrics:
        return self._metrics

    async def segments(self) -> AsyncIterator[TranscriptionSegment]:
        assert self._audio is not None
        loop = asyncio.get_running_loop()
        sr = self._config.sample_rate
        acc_samples = int(self._config.accumulation_seconds * sr)

        audio_buf = np.array([], dtype=np.float32)
        time_offset = 0.0
        try:
            while self._running:
                chunk = await self._audio.read_chunk()
                if not chunk:
                    break

                samples = np.frombuffer(chunk, dtype=np.float32)
                self._metrics.chunk_count += 1
                self._metrics.sample_count += len(samples)
                self._metrics.recorded_seconds = self._metrics.sample_count / sr
                audio_buf = np.concatenate([audio_buf, samples])

                if len(audio_buf) < acc_samples:
                    continue
                if not is_active_audio(audio_buf, self._config):
                    self._metrics.notes.append("Skipped low-energy window in stream mode")
                    audio_buf = audio_buf[acc_samples:]
                    time_offset += self._config.accumulation_seconds
                    self._last_partial_text = ""
                    continue

                started_at = time.perf_counter()
                results = await loop.run_in_executor(
                    None,
                    self._whisper.transcribe,
                    audio_buf,
                    self._config.language,
                    self._config.initial_prompt,
                )
                self._metrics.transcribe_calls += 1
                self._metrics.transcription_seconds += time.perf_counter() - started_at
                self._metrics.raw_segments += len(results)

                if len(audio_buf) >= 2 * acc_samples:
                    for result in results:
                        if result["t1"] > self._config.accumulation_seconds:
                            continue
                        segment = self._make_segment(result, time_offset, is_final=True)
                        if segment is not None:
                            self._metrics.cleaned_segments += 1
                            self._metrics.emitted_segments += 1
                            self._metrics.final_segments += 1
                            yield segment
                        else:
                            self._metrics.filtered_segments += 1
                    audio_buf = audio_buf[acc_samples:]
                    time_offset += self._config.accumulation_seconds
                    self._last_partial_text = ""
                else:
                    if not results:
                        continue
                    segment = self._make_segment(results[-1], time_offset, is_final=False)
                    if segment is not None:
                        self._metrics.cleaned_segments += 1
                        self._metrics.emitted_segments += 1
                        yield segment
                    else:
                        self._metrics.filtered_segments += 1

            if len(audio_buf) > 0:
                if not is_active_audio(audio_buf, self._config):
                    self._metrics.notes.append("Skipped final low-energy window in stream mode")
                    return
                started_at = time.perf_counter()
                results = await loop.run_in_executor(
                    None,
                    self._whisper.transcribe,
                    audio_buf,
                    self._config.language,
                    self._config.initial_prompt,
                )
                self._metrics.transcribe_calls += 1
                self._metrics.transcription_seconds += time.perf_counter() - started_at
                self._metrics.raw_segments += len(results)
                for result in results:
                    segment = self._make_segment(result, time_offset, is_final=True)
                    if segment is not None:
                        self._metrics.cleaned_segments += 1
                        self._metrics.emitted_segments += 1
                        self._metrics.final_segments += 1
                        yield segment
                    else:
                        self._metrics.filtered_segments += 1
        finally:
            self._whisper.close()

    def _make_segment(
        self,
        result: dict,
        time_offset: float,
        *,
        is_final: bool,
    ) -> TranscriptionSegment | None:
        self._seg_counter += 1
        text = normalize_text(
            result["text"],
            prefer_simplified_chinese=self._config.prefer_simplified_chinese,
        )
        if not is_plausible_text(text):
            return None

        if is_final:
            if is_duplicate_or_extension(text, self._last_final_text):
                return None
            self._last_final_text = merge_incremental_text(self._last_final_text, text)
        else:
            if is_duplicate_or_extension(text, self._last_partial_text):
                return None
            if self._last_final_text and text.startswith(self._last_final_text):
                return None
            self._last_partial_text = merge_incremental_text(self._last_partial_text, text)

        if is_final:
            self._cumulative_words += len(text.split())
        return TranscriptionSegment(
            segment_id=f"seg-{self._seg_counter:05d}",
            text=text,
            start_time=time_offset + result["t0"],
            end_time=time_offset + result["t1"],
            status=SegmentStatus.FINAL if is_final else SegmentStatus.PARTIAL,
            is_final=is_final,
            language=result.get("lang", self._config.language),
            cumulative_word_count=self._cumulative_words,
        )

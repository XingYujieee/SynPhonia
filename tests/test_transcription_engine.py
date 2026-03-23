from __future__ import annotations

import asyncio
import unittest
from unittest.mock import patch

import numpy as np

from config import TranscriptionConfig
from models import SegmentStatus
from transcription_engine import WhisperTranscriptionEngine


class FakeWhisperLib:
    def __init__(self, library_path: str = "") -> None:
        self._calls = 0

    def init_model(self, model_path: str, *, backend: str = "auto") -> None:
        return None

    def transcribe(self, samples: np.ndarray, language: str = "auto", initial_prompt: str = "") -> list[dict]:
        self._calls += 1
        if self._calls == 1:
            return [{"text": "你好", "t0": 0.0, "t1": 0.8, "lang": "zh"}]
        return [{"text": "你好世界", "t0": 0.0, "t1": 1.0, "lang": "zh"}]

    def close(self) -> None:
        return None


class FakeAudioSource:
    def __init__(self, chunks: list[np.ndarray], sample_rate: int) -> None:
        self._chunks = [
            np.ascontiguousarray(chunk, dtype=np.float32).tobytes()
            for chunk in chunks
        ]
        self.sample_rate = sample_rate
        self.channels = 1

    async def read_chunk(self) -> bytes:
        await asyncio.sleep(0)
        if self._chunks:
            return self._chunks.pop(0)
        return b""


class TranscriptionEngineTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_engine_emits_partial_then_final(self) -> None:
        cfg = TranscriptionConfig(
            sample_rate=4,
            chunk_duration_ms=500,
            accumulation_seconds=1.0,
            silence_floor=0.001,
        )
        chunks = [
            np.array([0.2, 0.2], dtype=np.float32),
            np.array([0.2, 0.2], dtype=np.float32),
            np.array([0.2, 0.2], dtype=np.float32),
        ]

        with patch("transcription_engine.WhisperLib", FakeWhisperLib):
            engine = WhisperTranscriptionEngine(cfg)
            await engine.start(FakeAudioSource(chunks, sample_rate=cfg.sample_rate))
            segments = [segment async for segment in engine.segments()]

        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0].status, SegmentStatus.PARTIAL)
        self.assertEqual(segments[0].text, "你好")
        self.assertEqual(segments[1].status, SegmentStatus.FINAL)
        self.assertEqual(segments[1].text, "你好世界")
        self.assertGreaterEqual(engine.metrics.transcribe_calls, 2)


if __name__ == "__main__":
    unittest.main()

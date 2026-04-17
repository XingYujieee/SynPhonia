"""Async microphone capture based on sounddevice."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

log = logging.getLogger(__name__)


class MicrophoneAudioSource:
    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        channels: int = 1,
        chunk_duration_ms: int = 500,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_duration_ms = chunk_duration_ms
        self._frames_per_chunk = max(1, int(sample_rate * chunk_duration_ms / 1000))
        self._queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=32)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stream: Any = None

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        try:
            import sounddevice as sd  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Microphone capture requires the 'sounddevice' package. "
                "Install it or run with --skip-mic."
            ) from exc

        def _callback(indata, frames, _time, status) -> None:
            if status:
                log.warning("sounddevice status: %s", status)
            if self._loop is None:
                return
            data = indata.astype("float32", copy=False).tobytes()
            self._loop.call_soon_threadsafe(self._push, data)

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            blocksize=self._frames_per_chunk,
            callback=_callback,
        )
        self._stream.start()

    def _push(self, data: bytes) -> None:
        if self._queue.full():
            try:
                _ = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        self._queue.put_nowait(data)

    async def read_chunk(self) -> bytes:
        return await self._queue.get()

    async def stop(self) -> None:
        stream = self._stream
        self._stream = None
        if stream is None:
            return
        stream.stop()
        stream.close()

"""Audio source implementation using sounddevice."""

from __future__ import annotations

import asyncio
import logging

import numpy as np
import sounddevice as sd

log = logging.getLogger(__name__)


class MicrophoneAudioSource:
    """Captures audio from the system microphone via sounddevice."""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_duration_ms: int = 500,
        queue_maxsize: int = 100,
    ) -> None:
        self._sample_rate = sample_rate
        self._channels = channels
        self._chunk_samples = int(sample_rate * chunk_duration_ms / 1000)
        self._queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=queue_maxsize)
        self._stream: sd.InputStream | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def channels(self) -> int:
        return self._channels

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=self._channels,
            dtype="float32",
            blocksize=self._chunk_samples,
            callback=self._audio_callback,
        )
        self._stream.start()
        log.info(
            "Microphone started: %d Hz, %d ch, %d samples/chunk",
            self._sample_rate,
            self._channels,
            self._chunk_samples,
        )

    async def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        await self._queue.put(b"")
        log.info("Microphone stopped.")

    async def read_chunk(self) -> bytes:
        return await self._queue.get()

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: dict,
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            log.warning("sounddevice status: %s", status)
        pcm_bytes = indata[:, 0].tobytes()
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._enqueue, pcm_bytes)

    def _enqueue(self, data: bytes) -> None:
        try:
            self._queue.put_nowait(data)
        except asyncio.QueueFull:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            self._queue.put_nowait(data)

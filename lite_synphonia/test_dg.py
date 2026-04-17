import sys
from pathlib import Path
from provider_registry import get_registry
from transcription.deepgram_client import DeepgramTranscriptionClient, _float_audio_to_wav_bytes
import numpy as np
import wave
import json

reg = get_registry()
api_key = reg.resolve_key("deepgram")
entry = reg.get("deepgram")

client = DeepgramTranscriptionClient(
    api_key,
    model=entry.model_id or "whisper-large",
    language="zh-CN",
    base_url=entry.base_url
)

with wave.open("../lite_synphonia_output/transcription/enhanced_audio.wav", "rb") as wf:
    params = wf.getparams()
    frames = wf.readframes(params.nframes)
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    sr = params.framerate

print(f"Loaded audio: {len(audio)} samples, {sr} Hz")
payload = _float_audio_to_wav_bytes(audio, sample_rate=sr)
data = client._post_audio(payload)
print(json.dumps(data, indent=2, ensure_ascii=False))

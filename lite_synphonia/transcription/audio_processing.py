"""Audio enhancement utilities for microphone capture.

Processing chain (applied in order)
------------------------------------
1. DC offset removal   — subtract signal mean to eliminate hardware bias
2. Input gain          — fixed scalar multiplier (cfg.input_gain)
3. Bidirectional AGC   — scale RMS toward cfg.target_rms, with gain bounded
                         to [1/max_gain, max_gain]; attenuates loud signals
                         as well as boosting quiet ones
4. Pre-emphasis        — first-order high-pass y[n] = x[n] - α·x[n-1]
                         boosts consonants that ASR relies on (zh/ch/sh/s)
5. Noise gate          — zero out chunks whose RMS is below cfg.silence_floor
                         so that background hiss is not amplified and sent
                         to the STT API as speech
6. Peak limiter        — hard ceiling at cfg.limiter_level with a soft knee
                         to avoid rectangular clipping artefacts
7. Final clip          — ensure the signal stays within [-1, 1]
"""

from __future__ import annotations

import numpy as np

from .config import TranscriptionConfig


# ── Low-level helpers ─────────────────────────────────────────────────────────

def _estimate_rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio.astype(np.float32)))))


def _remove_dc_offset(audio: np.ndarray) -> np.ndarray:
    """Subtract the mean to remove any DC bias introduced by the hardware."""
    return audio - np.mean(audio)


def _apply_pre_emphasis(audio: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    """First-order high-pass filter:  y[n] = x[n] - coeff * x[n-1]

    Boosts high-frequency speech components (sibilants, fricatives, plosives)
    that are attenuated by microphone capsules and room acoustics.  A coefficient
    of 0.97 is the standard value used in Kaldi, ESPnet, and most ASR toolkits.

    Setting coeff=0.0 disables the filter entirely.
    """
    if len(audio) < 2 or coeff <= 0.0:
        return audio.copy()
    out = audio.copy()
    out[1:] = audio[1:] - coeff * audio[:-1]
    return out


def _bidirectional_agc(
    audio: np.ndarray,
    target_rms: float,
    max_gain: float,
) -> np.ndarray:
    """Scale audio so its RMS approaches *target_rms*.

    Unlike the previous implementation, this applies gain in both directions:
    - Quiet audio (rms < target_rms) is amplified up to *max_gain*.
    - Loud audio (rms > target_rms) is attenuated toward the target.

    Amplification is capped at *max_gain* to avoid over-amplifying pure noise.
    Attenuation has no floor — a very loud signal is always reduced toward
    target_rms rather than being hard-clipped later by the limiter.
    """
    rms = _estimate_rms(audio)
    if rms <= 1e-8:
        return audio.copy()

    gain = float(target_rms) / rms
    # Cap boost at max_gain; allow unlimited attenuation so loud signals
    # are scaled down before reaching the limiter.
    gain = min(gain, float(max_gain))
    return audio * gain


def _soft_knee_limiter(audio: np.ndarray, ceiling: float) -> np.ndarray:
    """Apply a soft-knee limiter with ceiling at *ceiling*.

    Below 85 % of the ceiling the signal passes untouched.
    Between 85 % and 100 % a smooth cubic curve compresses towards the ceiling.
    Above the ceiling the signal is hard-clipped.

    A soft knee avoids the rectangular waveshaping artefacts (inter-modulation
    distortion) that a hard limiter creates, which are the primary cause of
    degraded ASR confidence scores.
    """
    ceil = float(np.clip(ceiling, 0.05, 0.99))
    knee_start = ceil * 0.85
    abs_audio = np.abs(audio)

    in_knee = (abs_audio > knee_start) & (abs_audio <= ceil)
    above_ceil = abs_audio > ceil

    out = audio.copy()

    # Soft knee region: smooth cubic interpolation toward ceiling
    if np.any(in_knee):
        t = (abs_audio[in_knee] - knee_start) / (ceil - knee_start)   # [0, 1]
        # Hermite smooth step: 3t² - 2t³  (S-curve from 0 to 1)
        smooth = t * t * (3.0 - 2.0 * t)
        compressed = knee_start + smooth * (ceil - knee_start)
        out[in_knee] = np.sign(audio[in_knee]) * compressed

    # Hard ceiling above knee
    if np.any(above_ceil):
        out[above_ceil] = np.sign(audio[above_ceil]) * ceil

    return out


# ── Public API ────────────────────────────────────────────────────────────────

def enhance_audio(audio: np.ndarray, cfg: TranscriptionConfig) -> np.ndarray:
    """Apply the full enhancement chain to a single audio chunk.

    Returns a float32 array of the same length, or a zero-filled array of the
    same length if the chunk is below the noise gate threshold.

    Parameters
    ----------
    audio:
        Raw float32 PCM samples, typically one 500 ms microphone chunk.
    cfg:
        TranscriptionConfig carrying all tuning parameters.
    """
    if audio.size == 0:
        return audio.astype(np.float32)

    out = audio.astype(np.float32, copy=True)

    # 1. Remove DC offset
    out = _remove_dc_offset(out)

    # 2. Fixed input gain
    input_gain = max(0.0, float(cfg.input_gain))
    if input_gain != 1.0:
        out = out * input_gain

    # 3. Noise gate — must run BEFORE AGC and pre-emphasis.
    #
    # The gate must classify silence vs. speech on the raw (post-DC, post-gain)
    # level, for two reasons:
    #   a) AGC can lift a background-noise chunk (RMS 0.002) above the floor
    #      before the gate ever sees it, making the gate useless.
    #   b) Pre-emphasis attenuates low frequencies by up to −20 dB; a real
    #      speech chunk with RMS 0.008 becomes ~0.002 after filtering and
    #      would be wrongly classified as silence.
    # Checking here on the unmodified signal gives a reliable speech/silence
    # decision before either amplification or spectral shaping.
    silence_floor = max(0.0, float(cfg.silence_floor))
    if _estimate_rms(out) < silence_floor:
        return np.zeros(len(out), dtype=np.float32)

    # 4. Bidirectional AGC (attenuates loud, boosts quiet) — only on speech
    out = _bidirectional_agc(out, cfg.target_rms, cfg.max_gain)

    # 5. Pre-emphasis (high-frequency boost for ASR)
    pre_coeff = float(getattr(cfg, "pre_emphasis_coeff", 0.97))
    out = _apply_pre_emphasis(out, pre_coeff)

    # 6. Soft-knee limiter
    limiter = float(cfg.limiter_level)
    out = _soft_knee_limiter(out, limiter)

    # 7. Final hard clip (safety net only; limiter should prevent reaching here)
    return np.clip(out, -1.0, 1.0)

"""Microphone preflight diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class PreflightReport:
    seconds: float
    rms: float
    peak: float
    silence_floor: float
    limiter_level: float
    clipping_risk: bool
    low_signal_risk: bool
    recommendation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def analyze_preflight_audio(
    *,
    seconds: float,
    rms: float,
    peak: float,
    silence_floor: float,
    limiter_level: float,
) -> PreflightReport:
    # Clipping is detected on the RAW (pre-enhancement) signal and means the
    # hardware ADC is saturating — peak very close to digital full-scale (1.0).
    # The old threshold (limiter_level * 0.98) caused false positives because
    # after we lowered limiter_level to 0.72 any well-limited recording would
    # fire the alarm at peak >= 0.706.  We now use 0.95 which reflects actual
    # ADC saturation regardless of what the software limiter is set to.
    clipping = peak >= 0.95
    low_signal = rms < max(1e-6, silence_floor * 2.0)

    if clipping and low_signal:
        rec = "信号不稳定：请调整麦克风距离并降低系统增益。"
    elif clipping:
        rec = "存在削波风险：请降低麦克风输入增益。"
    elif low_signal:
        rec = "音量偏低：请靠近麦克风或提高输入音量。"
    else:
        rec = "预检通过。"

    return PreflightReport(
        seconds=float(seconds),
        rms=float(rms),
        peak=float(peak),
        silence_floor=float(silence_floor),
        limiter_level=float(limiter_level),
        clipping_risk=clipping,
        low_signal_risk=low_signal,
        recommendation=rec,
    )

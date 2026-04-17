"""Text cleanup helpers for transcript segments."""

from __future__ import annotations

import re

try:
    from opencc import OpenCC
except ImportError:  # pragma: no cover - optional at import time
    OpenCC = None


_WS_RE = re.compile(r"\s+")
_PUNCT_ONLY_RE = re.compile(r"^[\W_]+$", re.UNICODE)
_SIMPLIFIER = OpenCC("t2s") if OpenCC is not None else None


def normalize_text(text: str, *, prefer_simplified_chinese: bool = True) -> str:
    cleaned = _WS_RE.sub(" ", (text or "").strip())
    if prefer_simplified_chinese and cleaned and _SIMPLIFIER is not None:
        cleaned = _SIMPLIFIER.convert(cleaned)
    return cleaned


def is_plausible_text(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 2:
        return False
    if _PUNCT_ONLY_RE.match(t):
        return False
    return True

"""Helpers for making Whisper output cleaner for Chinese subtitles."""

from __future__ import annotations

import re

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_ASCII_WORD_RE = re.compile(r"[A-Za-z]{2,}")
_SUSPICIOUS_PATTERNS = (
    re.compile(r"^\(?\s*字幕\s*[:：].*\)?$"),
    re.compile(r"^\(?\s*字幕制作\s*[:：].*\)?$"),
    re.compile(r"^\(?\s*字幕署名\s*[:：].*\)?$"),
    re.compile(r"^\(?\s*subtitles?\s*[:：].*\)?$", re.IGNORECASE),
    re.compile(r"^\(?\s*(语音转写|实时转录|字幕|噪声猜测)\s*\)?$"),
)

try:
    from opencc import OpenCC  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    OpenCC = None

_OPENCC = OpenCC("t2s") if OpenCC is not None else None


def normalize_text(text: str, *, prefer_simplified_chinese: bool = True) -> str:
    cleaned = " ".join(text.replace("\u3000", " ").split()).strip()
    cleaned = re.sub(r"\s*([，。！？；：])\s*", r"\1", cleaned)
    cleaned = re.sub(r"([\u4e00-\u9fff])\s+([\u4e00-\u9fff])", r"\1\2", cleaned)
    if prefer_simplified_chinese and _OPENCC is not None and cleaned:
        cleaned = _OPENCC.convert(cleaned)
    return cleaned


def is_plausible_text(text: str) -> bool:
    if not text:
        return False
    if all(not ch.isalnum() and not _CJK_RE.search(ch) for ch in text):
        return False
    if any(pattern.match(text) for pattern in _SUSPICIOUS_PATTERNS):
        return False
    if _ASCII_WORD_RE.search(text) and "字幕" in text and len(text) <= 24:
        return False
    cjk_count = len(_CJK_RE.findall(text))
    ascii_word_count = len(_ASCII_WORD_RE.findall(text))
    if cjk_count == 0 and len(text) < 2:
        return ascii_word_count > 0 and len(text) >= 2
    if cjk_count > 0 and cjk_count < 2 and len(text) < 4:
        return False
    if cjk_count < 2 and ascii_word_count == 0 and len(text) < 3:
        return False
    if any(token in text for token in ("字幕制作", "字幕署名", "语音转写", "噪声猜测")):
        return False
    return True


def is_duplicate_or_extension(current: str, previous: str) -> bool:
    if not previous:
        return False
    return current == previous or current.startswith(previous)


def merge_incremental_text(previous: str, current: str) -> str:
    if not previous:
        return current
    if not current or current == previous:
        return previous
    if current.startswith(previous):
        return current
    if previous.startswith(current):
        return previous

    max_overlap = min(len(previous), len(current))
    for overlap in range(max_overlap, 0, -1):
        if previous[-overlap:] == current[:overlap]:
            return previous + current[overlap:]
    return current

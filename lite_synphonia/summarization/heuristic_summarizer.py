from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+_-]*")

_STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "this", "that", "it", "we", "you", "i",
    "uh", "um", "okay", "ok", "yeah", "嗯", "啊", "就是", "那个", "然后",
}


@dataclass
class HeuristicSummary:
    keywords: list[str]
    summary: str


def normalize_keywords(keywords: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in keywords:
        k = " ".join(str(raw).split()).strip(" ,;:，；：")
        if len(k) < 2:
            continue
        identity = k.lower()
        if identity in seen:
            continue
        seen.add(identity)
        out.append(k)
    return out


def summarize_text(text: str, keyword_limit: int = 5) -> HeuristicSummary:
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return HeuristicSummary(keywords=["summary"], summary="")

    tokens = [t.lower() for t in _WORD_RE.findall(cleaned)]
    tokens = [t for t in tokens if len(t) >= 2 and t not in _STOPWORDS]
    counter = Counter(tokens)
    keywords = [k for k, _ in counter.most_common(max(1, keyword_limit))]
    keywords = normalize_keywords(keywords)

    # simple sentence fallback
    parts = re.split(r"(?<=[。！？.!?])\s+", cleaned)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) >= 2:
        summary = " ".join(parts[:2])
    else:
        summary = cleaned[:220]
    return HeuristicSummary(keywords=keywords or ["lecture"], summary=summary)

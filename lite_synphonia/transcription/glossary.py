"""Glossary loading and prompt construction."""

from __future__ import annotations

from pathlib import Path


def load_glossary_file(path: str) -> list[str]:
    src = Path(path).expanduser()
    terms: list[str] = []
    for line in src.read_text(encoding="utf-8").splitlines():
        t = line.strip()
        if not t or t.startswith("#"):
            continue
        terms.append(t)
    # keep order, deduplicate
    return list(dict.fromkeys(terms))


def build_initial_prompt(terms: list[str], *, prefix: str = "") -> str:
    terms = [t.strip() for t in terms if t.strip()]
    if not terms:
        return prefix.strip()
    glossary_part = "术语参考：" + "；".join(terms)
    if prefix.strip():
        return prefix.strip() + "\n" + glossary_part
    return glossary_part

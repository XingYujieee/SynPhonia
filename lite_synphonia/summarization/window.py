from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WindowSlice:
    start_index: int
    end_index: int
    words: list[str]

    @property
    def text(self) -> str:
        return " ".join(self.words)

    @property
    def word_count(self) -> int:
        return len(self.words)


def can_form_window(words: list[str], start_index: int, window_size: int) -> bool:
    return start_index + window_size <= len(words)


def build_window(words: list[str], start_index: int, window_size: int) -> WindowSlice:
    if not can_form_window(words, start_index, window_size):
        raise ValueError("Not enough words to form a full window.")

    end_index = start_index + window_size - 1
    return WindowSlice(
        start_index=start_index,
        end_index=end_index,
        words=words[start_index : start_index + window_size],
    )


def slide_window(start_index: int, window_size: int, overlap_size: int) -> int:
    return start_index + (window_size - overlap_size)

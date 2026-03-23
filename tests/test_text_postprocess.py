from __future__ import annotations

import unittest

from text_postprocess import (
    is_duplicate_or_extension,
    is_plausible_text,
    merge_incremental_text,
    normalize_text,
)


class TextPostprocessTests(unittest.TestCase):
    def test_normalize_text_removes_cjk_spacing(self) -> None:
        self.assertEqual(normalize_text("你 好 ， 世 界 "), "你好，世界")

    def test_plausible_text_filters_subtitle_noise(self) -> None:
        self.assertFalse(is_plausible_text("字幕：测试"))
        self.assertFalse(is_plausible_text("(字幕制作:贝尔)"))
        self.assertTrue(is_plausible_text("今天的天气不错"))

    def test_duplicate_or_extension(self) -> None:
        self.assertTrue(is_duplicate_or_extension("你好世界", "你好"))
        self.assertFalse(is_duplicate_or_extension("世界你好", "你好"))

    def test_merge_incremental_text_uses_overlap(self) -> None:
        self.assertEqual(merge_incremental_text("今天天气", "天气不错"), "今天天气不错")
        self.assertEqual(merge_incremental_text("你好", "你好"), "你好")


if __name__ == "__main__":
    unittest.main()

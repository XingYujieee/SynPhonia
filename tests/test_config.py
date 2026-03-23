from __future__ import annotations

import platform
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import config


class ConfigTests(unittest.TestCase):
    def test_resolve_model_path_prefers_local_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            models_dir = root / "models"
            models_dir.mkdir(parents=True)
            model_path = models_dir / "ggml-small.bin"
            model_path.write_bytes(b"model")

            with patch.object(config, "_candidate_whisper_dirs", return_value=(models_dir,)):
                self.assertEqual(config.resolve_model_path("auto"), str(model_path))
                self.assertEqual(config.resolve_model_path("small"), str(model_path))

    def test_list_available_models_returns_only_present_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            models_dir = root / "models"
            models_dir.mkdir(parents=True)
            small_path = models_dir / "ggml-small.bin"
            base_path = models_dir / "ggml-base.bin"
            small_path.write_bytes(b"small")
            base_path.write_bytes(b"base")

            with patch.object(config, "_candidate_whisper_dirs", return_value=(models_dir,)):
                available = config.list_available_models()

            self.assertEqual(available["small"], str(small_path))
            self.assertEqual(available["base"], str(base_path))
            self.assertNotIn("tiny", available)

    def test_default_library_path_uses_local_vendor_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            whisper_dir = root / "vendor" / "whisper"
            whisper_dir.mkdir(parents=True)
            library_name = config._WHISPER_LIB_NAMES[platform.system()]
            library_path = whisper_dir / library_name
            library_path.write_bytes(b"lib")

            with patch.object(config, "_candidate_whisper_dirs", return_value=(whisper_dir,)):
                self.assertEqual(config._default_whisper_library_path(), str(library_path))


if __name__ == "__main__":
    unittest.main()

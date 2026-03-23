"""ctypes bindings for whisper.cpp shared library."""

from __future__ import annotations

import ctypes as ct
import logging
import os
import platform
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)
_ALLOW_UNSAFE_BACKEND_INIT = "SYNPHONIE_WHISPER_ALLOW_UNSAFE_BACKEND_INIT"
_DLL_DIRECTORIES: list[object] = []

_WhisperContextPtr = ct.c_void_p

WHISPER_SAMPLING_GREEDY = 0


class WhisperFullParams(ct.Structure):
    _fields_ = [
        ("strategy", ct.c_int),
        ("n_threads", ct.c_int),
        ("n_max_text_ctx", ct.c_int),
        ("offset_ms", ct.c_int),
        ("duration_ms", ct.c_int),
        ("translate", ct.c_bool),
        ("no_context", ct.c_bool),
        ("no_timestamps", ct.c_bool),
        ("single_segment", ct.c_bool),
        ("print_special", ct.c_bool),
        ("print_progress", ct.c_bool),
        ("print_realtime", ct.c_bool),
        ("print_timestamps", ct.c_bool),
        ("token_timestamps", ct.c_bool),
        ("thold_pt", ct.c_float),
        ("thold_ptsum", ct.c_float),
        ("max_len", ct.c_int),
        ("split_on_word", ct.c_bool),
        ("max_tokens", ct.c_int),
        ("debug_mode", ct.c_bool),
        ("audio_ctx", ct.c_int),
        ("tdrz_enable", ct.c_bool),
        ("suppress_regex", ct.c_char_p),
        ("initial_prompt", ct.c_char_p),
        ("carry_initial_prompt", ct.c_bool),
        ("_pad0", ct.c_char * 3),
        ("prompt_tokens", ct.POINTER(ct.c_int32)),
        ("prompt_n_tokens", ct.c_int),
        ("language", ct.c_char_p),
        ("detect_language", ct.c_bool),
        ("suppress_blank", ct.c_bool),
        ("suppress_nst", ct.c_bool),
        ("_pad1", ct.c_char),
        ("temperature", ct.c_float),
        ("max_initial_ts", ct.c_float),
        ("length_penalty", ct.c_float),
        ("temperature_inc", ct.c_float),
        ("entropy_thold", ct.c_float),
        ("logprob_thold", ct.c_float),
        ("no_speech_thold", ct.c_float),
        ("greedy_best_of", ct.c_int),
        ("beam_search_beam_size", ct.c_int),
        ("beam_search_patience", ct.c_float),
        ("new_segment_callback", ct.c_void_p),
        ("new_segment_callback_user_data", ct.c_void_p),
        ("progress_callback", ct.c_void_p),
        ("progress_callback_user_data", ct.c_void_p),
        ("encoder_begin_callback", ct.c_void_p),
        ("encoder_begin_callback_user_data", ct.c_void_p),
        ("abort_callback", ct.c_void_p),
        ("abort_callback_user_data", ct.c_void_p),
        ("logits_filter_callback", ct.c_void_p),
        ("logits_filter_callback_user_data", ct.c_void_p),
        ("grammar_rules", ct.c_void_p),
        ("n_grammar_rules", ct.c_size_t),
        ("i_start_rule", ct.c_size_t),
        ("grammar_penalty", ct.c_float),
        ("vad", ct.c_bool),
        ("_pad2", ct.c_char * 7),
        ("vad_model_path", ct.c_char_p),
        ("vad_threshold", ct.c_float),
        ("vad_min_speech_duration_ms", ct.c_int),
        ("vad_min_silence_duration_ms", ct.c_int),
        ("vad_max_speech_duration_s", ct.c_float),
        ("vad_speech_pad_ms", ct.c_int),
        ("vad_samples_overlap", ct.c_float),
    ]


class WhisperContextParams(ct.Structure):
    _fields_ = [
        ("use_gpu", ct.c_bool),
        ("flash_attn", ct.c_bool),
        ("gpu_device", ct.c_int),
        ("dtw_token_timestamps", ct.c_int),
    ]


_LIB_NAMES = {
    "Darwin": "libwhisper.dylib",
    "Linux": "libwhisper.so",
    "Windows": "whisper.dll",
}


def _load_library(explicit_path: str = "") -> ct.CDLL:
    if explicit_path:
        if platform.system() == "Windows":
            lib_dir = str(Path(explicit_path).resolve().parent)
            if hasattr(os, "add_dll_directory"):
                _DLL_DIRECTORIES.append(os.add_dll_directory(lib_dir))
        return ct.CDLL(explicit_path)
    name = _LIB_NAMES.get(platform.system())
    if name is None:
        raise OSError(f"Unsupported platform: {platform.system()}")
    return ct.CDLL(name)


def _has_symbol(lib: ct.CDLL, name: str) -> bool:
    try:
        getattr(lib, name)
    except AttributeError:
        return False
    return True


def _bind_functions(lib: ct.CDLL) -> None:
    lib.whisper_init_from_file.argtypes = [ct.c_char_p]
    lib.whisper_init_from_file.restype = _WhisperContextPtr

    if _has_symbol(lib, "whisper_context_default_params"):
        lib.whisper_context_default_params.restype = WhisperContextParams

    if _has_symbol(lib, "whisper_init_from_file_with_params"):
        lib.whisper_init_from_file_with_params.argtypes = [
            ct.c_char_p,
            WhisperContextParams,
        ]
        lib.whisper_init_from_file_with_params.restype = _WhisperContextPtr

    lib.whisper_free.argtypes = [_WhisperContextPtr]
    lib.whisper_free.restype = None

    lib.whisper_full_default_params.argtypes = [ct.c_int]
    lib.whisper_full_default_params.restype = WhisperFullParams

    lib.whisper_full.argtypes = [
        _WhisperContextPtr,
        WhisperFullParams,
        ct.POINTER(ct.c_float),
        ct.c_int,
    ]
    lib.whisper_full.restype = ct.c_int

    lib.whisper_full_n_segments.argtypes = [_WhisperContextPtr]
    lib.whisper_full_n_segments.restype = ct.c_int

    lib.whisper_full_get_segment_text.argtypes = [_WhisperContextPtr, ct.c_int]
    lib.whisper_full_get_segment_text.restype = ct.c_char_p

    lib.whisper_full_get_segment_t0.argtypes = [_WhisperContextPtr, ct.c_int]
    lib.whisper_full_get_segment_t0.restype = ct.c_int64

    lib.whisper_full_get_segment_t1.argtypes = [_WhisperContextPtr, ct.c_int]
    lib.whisper_full_get_segment_t1.restype = ct.c_int64

    lib.whisper_full_lang_id.argtypes = [_WhisperContextPtr]
    lib.whisper_full_lang_id.restype = ct.c_int

    lib.whisper_lang_str.argtypes = [ct.c_int]
    lib.whisper_lang_str.restype = ct.c_char_p


class WhisperLib:
    def __init__(self, library_path: str = "") -> None:
        self._lib = _load_library(library_path)
        _bind_functions(self._lib)
        self._ctx: _WhisperContextPtr | None = None
        self._ctx_params: WhisperContextParams | None = None

    def _supports_context_params(self) -> bool:
        return (
            _has_symbol(self._lib, "whisper_context_default_params")
            and _has_symbol(self._lib, "whisper_init_from_file_with_params")
        )

    def _init_with_backend(self, model_path: str, *, backend: str) -> _WhisperContextPtr:
        if not self._supports_context_params():
            raise RuntimeError(
                "This whisper library does not expose context params; backend selection is unavailable.",
            )

        params = self._lib.whisper_context_default_params()
        params.use_gpu = backend == "gpu"
        if not params.use_gpu:
            params.flash_attn = False
            params.gpu_device = 0

        self._ctx_params = params
        ctx = self._lib.whisper_init_from_file_with_params(
            model_path.encode("utf-8"),
            self._ctx_params,
        )
        if not ctx:
            raise RuntimeError(
                f"Failed to load whisper model with backend={backend}: {model_path}",
            )
        return ctx

    def init_model(self, model_path: str, *, backend: str = "auto") -> None:
        backend = backend.strip().lower()
        self._ctx = None
        self._ctx_params = None

        if backend not in {"auto", "cpu", "gpu"}:
            raise ValueError(f"Unsupported Whisper backend: {backend}")

        system = platform.system()
        if backend in {"cpu", "gpu"}:
            if system == "Darwin" and os.environ.get(_ALLOW_UNSAFE_BACKEND_INIT, "") != "1":
                raise RuntimeError(
                    "The bundled macOS Whisper library does not yet support safe explicit backend selection. "
                    "Replace the vendor library, or set SYNPHONIE_WHISPER_ALLOW_UNSAFE_BACKEND_INIT=1 to try anyway."
                )
            self._ctx = self._init_with_backend(model_path, backend=backend)
            log.info("Whisper model loaded with backend=%s: %s", backend, model_path)
            return

        self._ctx = self._lib.whisper_init_from_file(model_path.encode("utf-8"))
        if not self._ctx:
            raise RuntimeError(f"Failed to load whisper model: {model_path}")
        log.info("Whisper model loaded with default backend: %s", model_path)

    def transcribe(
        self,
        samples: np.ndarray,
        language: str = "auto",
        initial_prompt: str = "",
    ) -> list[dict[str, Any]]:
        if self._ctx is None:
            raise RuntimeError("Model not loaded - call init_model() first")

        params = self._lib.whisper_full_default_params(WHISPER_SAMPLING_GREEDY)
        params.print_realtime = False
        params.print_progress = False
        params.print_timestamps = False
        params.print_special = False
        params.single_segment = False
        params.no_context = True
        params.translate = False
        params.suppress_blank = True

        if language != "auto":
            params.language = language.encode("utf-8")
            params.detect_language = False
        else:
            params.language = b"auto"
            params.detect_language = True

        if initial_prompt:
            params.initial_prompt = initial_prompt.encode("utf-8")
            params.carry_initial_prompt = True

        samples = np.ascontiguousarray(samples, dtype=np.float32)
        samples_ptr = samples.ctypes.data_as(ct.POINTER(ct.c_float))

        ret = self._lib.whisper_full(self._ctx, params, samples_ptr, len(samples))
        if ret != 0:
            raise RuntimeError(f"whisper_full failed with code {ret}")

        n_seg = self._lib.whisper_full_n_segments(self._ctx)
        lang_id = self._lib.whisper_full_lang_id(self._ctx)
        lang_str_raw = self._lib.whisper_lang_str(lang_id)
        lang_str = lang_str_raw.decode("utf-8") if lang_str_raw else "unknown"

        results: list[dict[str, Any]] = []
        for i in range(n_seg):
            text_raw = self._lib.whisper_full_get_segment_text(self._ctx, i)
            t0_cs = self._lib.whisper_full_get_segment_t0(self._ctx, i)
            t1_cs = self._lib.whisper_full_get_segment_t1(self._ctx, i)
            text = text_raw.decode("utf-8").strip() if text_raw else ""
            if text:
                results.append(
                    {
                        "text": text,
                        "t0": t0_cs / 100.0,
                        "t1": t1_cs / 100.0,
                        "lang": lang_str,
                    }
                )
        return results

    def close(self) -> None:
        if self._ctx:
            self._lib.whisper_free(self._ctx)
            self._ctx = None
            self._ctx_params = None
            log.info("Whisper context freed.")

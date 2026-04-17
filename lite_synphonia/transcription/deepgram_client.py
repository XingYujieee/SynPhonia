"""Deepgram Speech-to-Text client for LiteSynphonia."""

from __future__ import annotations

import io
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

_RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})
_MIN_UTTERANCE_SECONDS: float = 0.08

_DEFAULT_PARAMS: dict[str, str] = {
    "punctuate": "true",
    "smart_format": "true",
    "utterances": "true",
    # 1.2 s pause threshold before Deepgram starts a new utterance.
    # The previous 0.8 s was too short for lecture speech, splitting natural
    # thinking pauses mid-sentence and reducing per-utterance context.
    "utt_split": "1.2",
    # Convert spoken numbers/dates/times to digit form (e.g. 三十 → 30).
    # Improves downstream PDF matching when slides contain numeric content.
    "numerals": "true",
    # Strip disfluencies (uh, um, er, 那个, 就是) to keep the transcript clean.
    "filler_words": "false",
}

_LANG_MAP: dict[str, str] = {
    "zh": "zh-CN",
    "zh-cn": "zh-CN",
    "zh-tw": "zh-TW",
    "en": "en-US",
    "en-us": "en-US",
    "en-gb": "en-GB",
    "ja": "ja",
    "ko": "ko",
    "fr": "fr",
    "de": "de",
    "es": "es",
    "pt": "pt",
    "ru": "ru",
    "it": "it",
    "nl": "nl",
    "ar": "ar",
    "hi": "hi",
}


class DeepgramCallError(RuntimeError):
    """Raised when Deepgram API call fails permanently."""


def _whisper_lang_to_deepgram(language: str) -> str | None:
    key = (language or "").strip().lower()
    if not key:
        return None
    if key in _LANG_MAP:
        return _LANG_MAP[key]
    if "-" in key:
        return language
    return key


class DeepgramTranscriptionClient:
    def __init__(
        self,
        api_key: str,
        *,
        model: str = "whisper-large",
        language: str | None = "zh-CN",
        base_url: str = "https://api.deepgram.com",
        max_retries: int = 3,
        timeout_seconds: float = 120.0,
    ) -> None:
        if not api_key or not api_key.strip():
            raise DeepgramCallError("Deepgram API key must not be empty.")
        self._api_key = api_key.strip()
        self._model = model
        self._language = language
        self._base_url = base_url.rstrip("/")
        self._max_retries = max(1, int(max_retries))
        self._timeout = max(5.0, float(timeout_seconds))
        self._last_request_url = ""
        self._last_response_text = ""
        self._last_response_data: dict[str, Any] | None = None

    @property
    def last_request_url(self) -> str:
        return self._last_request_url

    @property
    def last_response_text(self) -> str:
        return self._last_response_text

    @property
    def last_response_data(self) -> dict[str, Any] | None:
        return self._last_response_data

    def transcribe(
        self,
        samples: np.ndarray,
        *,
        sample_rate: int = 16000,
        language: str | None = None,
    ) -> list[dict[str, Any]]:
        if samples.size == 0:
            self._last_response_text = ""
            self._last_response_data = {}
            return []

        effective_language = language or self._language
        wav_bytes = _samples_to_wav(samples, sample_rate)
        response_text = self._post_with_retry(wav_bytes, effective_language)
        self._last_response_text = response_text
        try:
            data = json.loads(response_text)
        except json.JSONDecodeError as exc:
            raise DeepgramCallError(
                f"Deepgram returned non-JSON response: {response_text[:300]}"
            ) from exc

        self._last_response_data = data
        return _parse_response(data, effective_language or "zh-CN")

    def _build_url(self, language: str | None) -> str:
        params = dict(_DEFAULT_PARAMS)
        params["model"] = self._model
        if language:
            params["language"] = language
        return f"{self._base_url}/v1/listen?{urllib.parse.urlencode(params)}"

    def _post_with_retry(self, wav_bytes: bytes, language: str | None) -> str:
        url = self._build_url(language)
        self._last_request_url = url
        headers = {
            "Authorization": f"Token {self._api_key}",
            "Content-Type": "audio/wav",
            "Accept": "application/json",
        }
        req = urllib.request.Request(url, data=wav_bytes, headers=headers, method="POST")

        last_error: Exception | None = None
        for attempt in range(max(1, self._max_retries)):
            if attempt > 0:
                wait = 2 ** (attempt - 1)
                log.warning(
                    "[deepgram] Retry %d/%d in %ds — previous error: %s",
                    attempt,
                    self._max_retries - 1,
                    wait,
                    last_error,
                )
                time.sleep(wait)

            try:
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                    return resp.read().decode("utf-8")
            except urllib.error.HTTPError as exc:
                body = ""
                try:
                    body = exc.read().decode("utf-8", errors="replace")
                except Exception:
                    pass
                if exc.code in _RETRYABLE_STATUS_CODES:
                    last_error = Exception(f"HTTP {exc.code}: {body[:200]}")
                    continue
                raise DeepgramCallError(
                    f"Deepgram STT API returned HTTP {exc.code}.\n"
                    f"URL: {url}\n"
                    f"Response: {body[:500]}"
                ) from exc
            except urllib.error.URLError as exc:
                last_error = exc
                continue
            except TimeoutError as exc:
                last_error = exc
                continue

        raise DeepgramCallError(
            f"Deepgram API failed after {self._max_retries} attempts. "
            f"Last error: {last_error}"
        )


def _samples_to_wav(samples: np.ndarray, sample_rate: int) -> bytes:
    samples_f32 = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    samples_i16 = (samples_f32 * 32767.0).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples_i16.tobytes())
    return buf.getvalue()


def _parse_response(data: dict[str, Any], language: str) -> list[dict[str, Any]]:
    if "error" in data:
        err = data["error"]
        raise DeepgramCallError(
            f"Deepgram API error: {err.get('message', str(err))}"
        )

    results = data.get("results", {})

    utterances = results.get("utterances") or []
    if utterances:
        segments = _utterances_to_segments(utterances, language)
        if segments:
            return segments

    words: list[dict[str, Any]] = []
    try:
        words = results["channels"][0]["alternatives"][0].get("words") or []
    except (KeyError, IndexError, TypeError):
        words = []
    if words:
        segments = _words_to_segments(words, language)
        if segments:
            return segments

    transcript = ""
    confidence = 0.0
    try:
        alt = results["channels"][0]["alternatives"][0]
        transcript = str(alt.get("transcript", "")).strip()
        confidence = float(alt.get("confidence", 0.0) or 0.0)
    except (KeyError, IndexError, TypeError, ValueError):
        transcript = ""
        confidence = 0.0
    if transcript:
        duration = _estimate_duration(data)
        return [
            {
                "text": transcript,
                "t0": 0.0,
                "t1": duration,
                "lang": language,
                "confidence": confidence,
            }
        ]

    return []


def _utterances_to_segments(
    utterances: list[dict[str, Any]], language: str
) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for utt in utterances:
        text = (utt.get("transcript") or "").strip()
        start = float(utt.get("start", 0.0))
        end = float(utt.get("end", start))
        if not text:
            continue
        if end - start < _MIN_UTTERANCE_SECONDS:
            continue
        segments.append(
            {
                "text": text,
                "t0": start,
                "t1": end,
                "lang": language,
                "confidence": float(utt.get("confidence", 0.0) or 0.0),
            }
        )
    return segments


def _words_to_segments(words: list[dict[str, Any]], language: str) -> list[dict[str, Any]]:
    if not words:
        return []

    segments: list[dict[str, Any]] = []
    chunk_words: list[str] = []
    chunk_conf: list[float] = []
    chunk_start = float(words[0].get("start", 0.0))
    prev_end = chunk_start

    split_chars = frozenset("。！？.!?")
    max_words_per_chunk = 50
    # Match utt_split: only break a word-level chunk on silences ≥ 1.2 s so
    # the word-fallback path produces segments that mirror utterance boundaries.
    silence_split_seconds = 1.2

    def _flush(end: float) -> None:
        text = " ".join(chunk_words).strip()
        if text and end > chunk_start:
            conf = float(sum(chunk_conf) / len(chunk_conf)) if chunk_conf else 0.0
            segments.append(
                {
                    "text": text,
                    "t0": chunk_start,
                    "t1": end,
                    "lang": language,
                    "confidence": conf,
                }
            )

    for word in words:
        w_text = (word.get("punctuated_word") or word.get("word") or "").strip()
        w_start = float(word.get("start", prev_end))
        w_end = float(word.get("end", w_start + 0.1))
        w_conf = float(word.get("confidence", 0.0) or 0.0)

        if chunk_words and (w_start - prev_end) >= silence_split_seconds:
            _flush(prev_end)
            chunk_words = []
            chunk_conf = []
            chunk_start = w_start

        chunk_words.append(w_text)
        chunk_conf.append(w_conf)
        prev_end = w_end

        if (w_text and w_text[-1] in split_chars) or len(chunk_words) >= max_words_per_chunk:
            _flush(w_end)
            chunk_words = []
            chunk_conf = []
            chunk_start = w_end

    if chunk_words:
        _flush(prev_end)

    return segments


def _estimate_duration(data: dict[str, Any]) -> float:
    try:
        return float(data["metadata"]["duration"])
    except (KeyError, TypeError, ValueError):
        return 0.0

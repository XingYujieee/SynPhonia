from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from typing import Any

from .config import ApiConfig

log = logging.getLogger(__name__)

_RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})

_SYSTEM_MESSAGE = (
    "You are a careful lecture summarization assistant. "
    "Always follow the requested output schema exactly."
)


class ApiCallError(RuntimeError):
    pass


class ApiLLMClient:
    def __init__(
        self,
        api_config: ApiConfig,
        provider_name: str | None = None,
    ) -> None:
        self._cfg = api_config
        self._provider_name = provider_name
        self._api_key: str | None = None
        self._active_base_url: str = api_config.base_url
        self._active_model_id: str = api_config.model_id
        self._active_max_new_tokens: int = api_config.max_new_tokens
        self._active_temperature: float = api_config.temperature
        self._active_timeout: float = api_config.timeout_seconds
        self._active_max_retries: int = api_config.max_retries
        self._ready = False
        self._backend_warning = ""

    @property
    def resolved_backend(self) -> str:
        return "api"

    @property
    def backend_warning(self) -> str:
        return self._backend_warning

    def ensure_ready(self) -> None:
        if self._provider_name:
            self._resolve_from_registry(self._provider_name)
        else:
            self._resolve_from_env()
        self._ready = True

    def generate_summary(self, prompt_text: str) -> str:
        if not self._ready:
            self.ensure_ready()

        messages: list[dict[str, str]] = [
            {"role": "system", "content": _SYSTEM_MESSAGE},
            {"role": "user", "content": prompt_text},
        ]

        payload: dict[str, Any] = {
            "model": self._active_model_id,
            "messages": messages,
            "max_tokens": self._active_max_new_tokens,
            "temperature": self._active_temperature,
            "stream": False,
        }

        last_error: Exception | None = None
        for attempt in range(max(1, self._active_max_retries)):
            if attempt > 0:
                wait = 2 ** (attempt - 1)
                log.warning(
                    "[api_client] Retry %d/%d in %ds (previous error: %s)",
                    attempt,
                    self._active_max_retries - 1,
                    wait,
                    last_error,
                )
                time.sleep(wait)

            try:
                response_text = self._post(payload)
                return self._extract_content(response_text)
            except ApiCallError:
                raise
            except _RetryableError as exc:
                last_error = exc
                continue

        raise ApiCallError(
            f"API call failed after {self._active_max_retries} attempts. "
            f"Last error: {last_error}"
        )

    def _resolve_from_registry(self, provider_name: str) -> None:
        from ..provider_registry import get_registry

        try:
            registry = get_registry()
            entry = registry.get(provider_name)
            key = registry.resolve_key(provider_name)
        except (KeyError, RuntimeError) as exc:
            raise ApiCallError(
                f"Cannot resolve provider '{provider_name}' from registry: {exc}"
            ) from exc

        self._api_key = key
        self._active_base_url = entry.base_url
        self._active_model_id = entry.model_id or self._active_model_id
        if entry.temperature >= 0:
            self._active_temperature = entry.temperature
        if entry.timeout_seconds > 0:
            self._active_timeout = entry.timeout_seconds
        if entry.max_retries > 0:
            self._active_max_retries = entry.max_retries

    def _resolve_from_env(self) -> None:
        key = os.environ.get(self._cfg.api_key_env, "").strip() if self._cfg.api_key_env else ""
        if key:
            self._api_key = key
            return
        key = os.environ.get("MINIMAX_API_KEY", "").strip()
        if key:
            self._api_key = key
            return
        raise ApiCallError(
            f"API key environment variable '{self._cfg.api_key_env}' is not set or empty."
        )

    def _post(self, payload: dict[str, Any]) -> str:
        url = self._active_base_url.rstrip("/") + "/chat/completions"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {self._api_key}",
        }

        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self._active_timeout) as resp:
                return resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            status = exc.code
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except Exception:
                detail = "(could not read response body)"
            if status in _RETRYABLE_STATUS_CODES:
                raise _RetryableError(f"HTTP {status}: {detail}") from exc
            raise ApiCallError(f"HTTP {status} from {url}.\nResponse: {detail}") from exc
        except urllib.error.URLError as exc:
            raise _RetryableError(f"Network error: {exc.reason}") from exc
        except TimeoutError as exc:
            raise _RetryableError(
                f"Request timed out after {self._active_timeout}s"
            ) from exc

    def _extract_content(self, response_text: str) -> str:
        try:
            data = json.loads(response_text)
        except json.JSONDecodeError as exc:
            raise ApiCallError(
                f"API returned non-JSON response. Raw: {response_text[:200]}"
            ) from exc

        if "error" in data:
            err = data["error"]
            code = err.get("code") or err.get("type", "unknown")
            msg = err.get("message", str(err))
            raise ApiCallError(f"API error [{code}]: {msg}")

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ApiCallError(
                f"Unexpected API response structure. Raw: {response_text[:400]}"
            ) from exc

        if not isinstance(content, str):
            raise ApiCallError(
                f"API message content is not a string. Got: {type(content).__name__}"
            )

        return content.strip()


class _RetryableError(Exception):
    pass

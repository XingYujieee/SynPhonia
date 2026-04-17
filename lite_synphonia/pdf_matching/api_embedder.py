"""API-based text embedding client for LiteSynphonia.

Drop-in replacement for ``merge_syn.pdf_matching.embedding_client.EmbeddingClient``
that calls a ``/v1/embeddings`` endpoint instead of loading a local
sentence-transformer model.

Provider formats
----------------
Two wire formats are supported and selected automatically from ``base_url``,
or can be forced via the ``provider_format`` constructor argument.

**minimax** (auto-detected when ``"minimaxi"`` appears in the base URL)

  Request body::

      {
        "model": "embo-01",
        "texts": ["text1", "text2"],   ← "texts", NOT "input"
        "type": "db"                   ← "db" for passages, "query" for queries
      }

  Response::

      {
        "vectors": [[0.12, ...], [0.34, ...]],   ← flat list of float lists
        "base_resp": {"status_code": 0, "status_msg": "success"}
      }

**openai** (default for all other base URLs)

  Request body::

      {
        "model": "text-embedding-3-small",
        "input": ["text1", "text2"]    ← "input", NOT "texts"
      }

  Response::

      {
        "data": [
          {"index": 0, "embedding": [0.12, ...]},
          {"index": 1, "embedding": [0.34, ...]}
        ]
      }

All returned vectors are L2-normalised float32 numpy arrays, matching the
contract expected by ``merge_syn.pdf_matching.scorer``.

Batching
--------
Large lists of texts are split into ``batch_size`` chunks and sent in separate
requests to respect per-request token limits.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any, Literal

import numpy as np

log = logging.getLogger(__name__)

_RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})

# Canonical input-type tokens for both providers.
# MiniMax uses "db" / "query"; OpenAI does not use this field at all.
_INPUT_TYPE_MAP: dict[str, str] = {
    "passage": "db",
    "db": "db",
    "document": "db",
    "query": "query",
}

ProviderFormat = Literal["auto", "minimax", "openai"]


class APIEmbeddingError(RuntimeError):
    """Raised when an API embedding call fails permanently."""


def _detect_format(base_url: str) -> Literal["minimax", "openai"]:
    """Infer the wire format from the base URL.

    MiniMax's endpoint contains ``minimaxi`` or ``minimax``.  Everything else
    is assumed to follow the OpenAI schema.
    """
    lower = base_url.lower()
    if "minimaxi" in lower or "minimax" in lower:
        return "minimax"
    return "openai"


class APIEmbeddingClient:
    """Text embedding client that supports both MiniMax and OpenAI wire formats.

    This class replicates the interface of
    ``merge_syn.pdf_matching.embedding_client.EmbeddingClient`` so the
    PDF-matching pipeline can use it without modification.

    Parameters
    ----------
    api_key:
        Bearer token for the embedding endpoint.
    base_url:
        Root URL of the API (e.g. ``https://api.minimaxi.chat/v1``).
        ``/embeddings`` is appended automatically.
    model:
        Model identifier sent in the request body.
    batch_size:
        Max number of texts per API call.  Defaults to 32.
    max_retries:
        Retry budget for transient HTTP errors.
    timeout_seconds:
        Per-request HTTP timeout.
    provider_format:
        ``"auto"`` (default) — inferred from *base_url*.
        ``"minimax"`` — force MiniMax request/response format.
        ``"openai"``  — force OpenAI request/response format.
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.minimaxi.chat/v1",
        model: str = "embo-01",
        batch_size: int = 32,
        max_retries: int = 3,
        timeout_seconds: float = 60.0,
        provider_format: ProviderFormat = "auto",
    ) -> None:
        if not api_key or not api_key.strip():
            raise APIEmbeddingError("Embedding API key must not be empty.")
        self._api_key = api_key.strip()
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._batch_size = batch_size
        self._max_retries = max_retries
        self._timeout = timeout_seconds
        self._dim: int | None = None  # auto-detected from first response

        if provider_format == "auto":
            self._format: Literal["minimax", "openai"] = _detect_format(base_url)
        else:
            self._format = provider_format  # type: ignore[assignment]

        log.info(
            "[api_embedder] Initialised: format=%s model=%s base_url=%s",
            self._format, self._model, self._base_url,
        )

    # ── Public interface (matches EmbeddingClient) ─────────────────────────

    @property
    def embedding_dim(self) -> int | None:
        """Return the detected embedding dimensionality, or None if not yet known."""
        return self._dim

    def encode_texts(
        self,
        texts: list[str],
        input_type: str = "passage",
    ) -> np.ndarray:
        """Embed *texts* and return a normalised float32 array of shape (N, D).

        Parameters
        ----------
        texts:
            List of strings to embed.
        input_type:
            ``"passage"`` (default) for document chunks, ``"query"`` for search
            queries.  Mapped to ``"db"`` / ``"query"`` for MiniMax; ignored by
            OpenAI.

        Returns
        -------
        numpy.ndarray
            Shape ``(len(texts), D)``, dtype float32, L2-normalised rows.
        """
        if not texts:
            d = self._dim or 1536
            return np.zeros((0, d), dtype=np.float32)

        mm_type = _INPUT_TYPE_MAP.get(input_type.lower(), "db")
        embeddings: list[np.ndarray] = []
        total_batches = (len(texts) + self._batch_size - 1) // self._batch_size

        for batch_idx, batch_start in enumerate(range(0, len(texts), self._batch_size)):
            batch = texts[batch_start: batch_start + self._batch_size]
            batch_embs = self._embed_batch(batch, mm_type)
            embeddings.append(batch_embs)
            if total_batches > 1:
                log.debug(
                    "[api_embedder] Batch %d/%d embedded (%d texts, format=%s)",
                    batch_idx + 1, total_batches, len(batch), self._format,
                )

        result = np.vstack(embeddings).astype(np.float32)
        return _l2_normalise(result)

    # ── Internal helpers ───────────────────────────────────────────────────

    def _build_payload(self, texts: list[str], mm_type: str) -> dict[str, Any]:
        """Build the request body according to the detected provider format.

        MiniMax uses ``texts`` as the array key and requires the ``type``
        field.  OpenAI uses ``input`` and ignores ``type``.
        """
        if self._format == "minimax":
            return {
                "model": self._model,
                "texts": texts,   # ← MiniMax requires "texts", not "input"
                "type": mm_type,  # ← "db" | "query"
            }
        # OpenAI (and compatible providers)
        return {
            "model": self._model,
            "input": texts,       # ← OpenAI requires "input", not "texts"
        }

    def _embed_batch(self, texts: list[str], mm_type: str) -> np.ndarray:
        """Call the API for a single batch of texts."""
        payload = self._build_payload(texts, mm_type)
        response_text = self._post_with_retry(payload)
        return self._parse_response(response_text, expected_count=len(texts))

    def _post_with_retry(self, payload: dict[str, Any]) -> str:
        url = self._base_url + "/embeddings"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json",
        }
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")

        last_error: Exception | None = None
        for attempt in range(max(1, self._max_retries)):
            if attempt > 0:
                wait = 2 ** (attempt - 1)
                log.warning(
                    "[api_embedder] Retry %d/%d in %ds — error: %s",
                    attempt, self._max_retries - 1, wait, last_error,
                )
                time.sleep(wait)
            try:
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                    return resp.read().decode("utf-8")
            except urllib.error.HTTPError as exc:
                body_txt = ""
                try:
                    body_txt = exc.read().decode("utf-8", errors="replace")
                except Exception:
                    pass
                log.debug("[api_embedder] HTTP %d from %s — body: %s",
                          exc.code, url, body_txt[:300])
                if exc.code in _RETRYABLE_STATUS_CODES:
                    last_error = Exception(f"HTTP {exc.code}: {body_txt[:200]}")
                    continue
                raise APIEmbeddingError(
                    f"Embedding API returned HTTP {exc.code}.\n"
                    f"URL: {url}\n"
                    f"Request keys: {list(payload.keys())}\n"
                    f"Response: {body_txt[:500]}"
                ) from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                last_error = exc
                continue

        raise APIEmbeddingError(
            f"Embedding API failed after {self._max_retries} attempts. "
            f"Last error: {last_error}"
        )

    def _parse_response(
        self, response_text: str, expected_count: int = 0
    ) -> np.ndarray:
        """Parse the embedding response into a numpy array.

        Handles both formats transparently:

        * **MiniMax** — top-level ``"vectors"`` key holding a list of float lists,
          plus ``"base_resp"`` for error signalling.
        * **OpenAI** — ``"data"`` key holding a list of objects each with an
          ``"embedding"`` field and an ``"index"`` field.

        The method tries the format that matches ``self._format`` first, then
        falls back to the other one so that providers that advertise one format
        but respond with another still work.
        """
        try:
            data = json.loads(response_text)
        except json.JSONDecodeError as exc:
            raise APIEmbeddingError(
                f"Embedding API returned non-JSON response.\n"
                f"First 300 chars: {response_text[:300]}"
            ) from exc

        # ── Error signals ──────────────────────────────────────────────────

        # MiniMax error: {"base_resp": {"status_code": <non-zero>, "status_msg": "..."}}
        if "base_resp" in data:
            br = data["base_resp"]
            if isinstance(br, dict) and br.get("status_code", 0) != 0:
                raise APIEmbeddingError(
                    f"MiniMax embedding error {br.get('status_code')}: "
                    f"{br.get('status_msg', str(br))}"
                )

        # OpenAI-style error: {"error": {"message": "..."}}
        if "error" in data:
            err = data["error"]
            msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            raise APIEmbeddingError(f"Embedding API error: {msg}")

        # ── Vector extraction ──────────────────────────────────────────────

        vectors: list[list[float]] | None = None

        # Path 1 — MiniMax: {"vectors": [[...], [...]]}
        if "vectors" in data:
            raw = data["vectors"]
            if isinstance(raw, list) and raw:
                vectors = raw
                log.debug("[api_embedder] Parsed MiniMax 'vectors' format (%d rows).",
                          len(vectors))

        # Path 2 — OpenAI: {"data": [{"index": N, "embedding": [...]}, ...]}
        if vectors is None and "data" in data:
            try:
                items = sorted(data["data"], key=lambda x: x["index"])
                vectors = [item["embedding"] for item in items]
                log.debug("[api_embedder] Parsed OpenAI 'data' format (%d rows).",
                          len(vectors))
            except (KeyError, TypeError) as exc:
                raise APIEmbeddingError(
                    f"Malformed OpenAI-format response — could not extract embeddings.\n"
                    f"Response (first 400 chars): {response_text[:400]}"
                ) from exc

        if vectors is None:
            raise APIEmbeddingError(
                f"Unrecognised embedding response: neither 'vectors' (MiniMax) "
                f"nor 'data' (OpenAI) key found.\n"
                f"Response keys: {list(data.keys())}\n"
                f"Response (first 400 chars): {response_text[:400]}"
            )

        if expected_count and len(vectors) != expected_count:
            log.warning(
                "[api_embedder] Expected %d vectors but received %d.",
                expected_count, len(vectors),
            )

        arr = np.array(vectors, dtype=np.float32)
        if self._dim is None and arr.size > 0:
            self._dim = arr.shape[1]
            log.info(
                "[api_embedder] Detected embedding dim=%d (model=%s, format=%s)",
                self._dim, self._model, self._format,
            )
        return arr


def _l2_normalise(arr: np.ndarray) -> np.ndarray:
    """L2-normalise each row of *arr* (safe for zero vectors)."""
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return arr / norms

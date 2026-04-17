"""Unified configuration for LiteSynphonia (all-API edition).

Everything that required a local path or device selection in MergeSyn
is replaced by a provider name that is looked up in the shared provider
registry (~/.config/mergesyn/providers.json).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TranscriptionAPIConfig:
    """Settings for the Deepgram STT backend."""

    # Name of the registered transcription provider (e.g. "deepgram")
    provider_name: str = "deepgram"
    # Deepgram model.  whisper-large is now the default for better robustness
    # on noisy/mixed lecture audio.
    model: str = "whisper-large"
    # BCP-47 language code sent to Deepgram.
    language: str = "zh-CN"

    # ── audio capture ────────────────────────────────────────────────────
    sample_rate: int = 16_000
    channels: int = 1
    chunk_duration_ms: int = 500        # size of each microphone read chunk
    silence_floor: float = 0.003        # below this RMS = silence

    # ── audio enhancement ────────────────────────────────────────────────
    # Processing chain: DC removal → input gain → bidirectional AGC →
    #   pre-emphasis → noise gate → soft-knee limiter → hard clip.
    #
    # Conservative defaults to avoid clipping that degrades Deepgram confidence.
    input_gain: float = 1.0
    target_rms: float = 0.03     # AGC target; bidirectional (attenuates loud too)
    max_gain: float = 2.0        # maximum AGC gain in either direction
    limiter_level: float = 0.72  # soft-knee ceiling (~3 dB below full scale)
    # First-order pre-emphasis coefficient α for y[n] = x[n] - α·x[n-1].
    # 0.97 is the ASR-standard value (Kaldi/ESPnet default).
    # Set to 0.0 to disable pre-emphasis entirely.
    pre_emphasis_coeff: float = 0.97


@dataclass
class SummarizationAPIConfig:
    """Settings for the LLM summarization backend (OpenAI-compatible)."""

    # Name of the registered summarization provider (e.g. "minimax")
    provider_name: str = "minimax"
    # Words per summarization window
    window_size: int = 200
    # Overlap between consecutive windows
    overlap_size: int = 60
    # Incremental reader chunk size (chars)
    chunk_size: int = 1200
    # Max tokens the model may generate per call
    max_new_tokens: int = 384


@dataclass
class EmbeddingAPIConfig:
    """Settings for the API-based text embedding backend (PDF matching)."""

    # Name of the registered provider to use for embeddings.
    # The provider must expose a /v1/embeddings endpoint.
    # MiniMax "embo-01" and OpenAI "text-embedding-3-small" both work.
    provider_name: str = "minimax-embed"
    # Embedding model name.
    model: str = "embo-01"
    # Embedding vector dimensionality (auto-detected after first call).
    dim: int = 0            # 0 = auto-detect from first response
    # Max texts per batch POST (some APIs have per-request limits).
    batch_size: int = 32
    # Wire format: "auto" (detect from base_url), "minimax", or "openai".
    # MiniMax uses {"texts": [...], "type": "db"} + {"vectors": [...]}.
    # OpenAI uses {"input": [...]} + {"data": [{"embedding": [...]}]}.
    provider_format: str = "auto"
    # Text prefixes prepended before embedding.
    # E5 family (multilingual-e5-large, etc.) requires "passage: " / "query: ".
    # BGE-M3 and OpenAI models use plain text — leave both as "".
    passage_prefix: str = ""
    query_prefix: str = ""
    # Directory for caching PDF chunk embeddings on disk.  Embeddings are keyed
    # by SHA-256 of the PDF file content + model name, so the cache is
    # automatically invalidated when the PDF or model changes.
    # Set to "" to disable caching.
    pdf_cache_dir: str = ""


@dataclass
class LiteConfig:
    """Top-level configuration for a LiteSynphonia run."""

    transcription: TranscriptionAPIConfig = field(
        default_factory=TranscriptionAPIConfig
    )
    summarization: SummarizationAPIConfig = field(
        default_factory=SummarizationAPIConfig
    )
    embedding: EmbeddingAPIConfig = field(
        default_factory=EmbeddingAPIConfig
    )

    # ── pipeline ─────────────────────────────────────────────────────────
    # Unique identifier for this recording session.  Passed through to the
    # interface_output.json so downstream systems (knowledge-base, front-end)
    # can correlate the output with the originating activity.
    # Empty string means "auto-generate a UUID4 at run time".
    activity_id: str = ""
    record_seconds: float = 8.0
    language: str = "zh"            # Whisper-style language code (mapped to BCP-47 internally)
    output_dir: str = ""
    pdf_path: str = ""
    initial_prompt: str = ""
    glossary_file: str = ""
    preflight_seconds: float = 0.0
    allow_low_quality_transcript: bool = False
    skip_mic: bool = False          # use synthetic audio (for testing)
    skip_summary: bool = False
    skip_pdf_matching: bool = False
    # Quality gate: if Deepgram mean_confidence falls below this value the
    # pipeline normally blocks downstream stages.  Setting to 0.0 disables the
    # confidence gate entirely (content-based checks still run).
    quality_confidence_threshold: float = 0.15
    # When True the AGC/limiter enhancement chain is bypassed entirely and raw
    # microphone samples are sent directly to Deepgram.  Useful for environments
    # where the hardware already provides a clean, well-levelled signal.
    disable_agc: bool = False

    # ── Knowledge base ────────────────────────────────────────────────────
    # Filesystem path to the knowledge base workspace directory.
    # When non-empty, each completed pipeline run automatically ingests the
    # activity record into the knowledge base after all other stages finish.
    # Leave empty (the default) to skip knowledge base ingestion entirely.
    knowledge_base_workspace: str = ""

    @property
    def output_path(self) -> Path:
        if self.output_dir:
            return Path(self.output_dir).expanduser()
        return Path("lite_synphonia_output")

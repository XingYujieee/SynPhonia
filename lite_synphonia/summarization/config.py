from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ApiConfig:
    base_url: str
    model_id: str
    api_key_env: str = ""
    max_new_tokens: int = 384
    temperature: float = 0.1
    timeout_seconds: float = 60.0
    max_retries: int = 3
    provider_name: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "base_url": self.base_url,
            "model_id": self.model_id,
            "api_key_env": self.api_key_env,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "provider_name": self.provider_name,
        }


@dataclass
class AppConfig:
    input_path: str | Path
    output_path: str | Path
    incremental_read_size: int = 1200
    window_size: int = 200
    overlap_size: int = 60
    llm_backend: str = "api"
    encoding: str = "utf-8"
    api: ApiConfig | None = None

    @property
    def step_size(self) -> int:
        return max(1, self.window_size - self.overlap_size)

    def to_dict(self) -> dict[str, object]:
        return {
            "input_path": str(self.input_path),
            "output_path": str(self.output_path),
            "incremental_read_size": self.incremental_read_size,
            "window_size": self.window_size,
            "overlap_size": self.overlap_size,
            "step_size": self.step_size,
            "llm_backend": self.llm_backend,
            "encoding": self.encoding,
            "api": self.api.to_dict() if self.api else {},
        }

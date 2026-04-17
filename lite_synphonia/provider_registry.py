"""Provider registry for LiteSynphonia.

This module stores provider definitions (base URL, model, API key, retry
settings) in a local JSON file. It intentionally supports reading the old
MergeSyn location as a fallback for migration.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_DEFAULT_PATH = Path.home() / ".config" / "lite_synphonia" / "providers.json"
_LEGACY_PATH = Path.home() / ".config" / "mergesyn" / "providers.json"


@dataclass
class ProviderEntry:
    name: str
    base_url: str
    model_id: str = ""
    api_key: str = ""
    services: list[str] = field(default_factory=list)
    timeout_seconds: float = 60.0
    max_retries: int = 3
    temperature: float = 0.1
    created_at_utc: str = ""
    updated_at_utc: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProviderEntry":
        return cls(
            name=str(data.get("name", "")).strip(),
            base_url=str(data.get("base_url", "")).strip(),
            model_id=str(data.get("model_id", data.get("model", ""))).strip(),
            api_key=str(data.get("api_key", "")).strip(),
            services=[str(s).strip() for s in data.get("services", []) if str(s).strip()],
            timeout_seconds=float(data.get("timeout_seconds", 60.0) or 60.0),
            max_retries=int(data.get("max_retries", 3) or 3),
            temperature=float(data.get("temperature", 0.1) or 0.1),
            created_at_utc=str(data.get("created_at_utc", "")),
            updated_at_utc=str(data.get("updated_at_utc", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def public_dict(self) -> dict[str, Any]:
        d = self.to_dict()
        key = d.get("api_key", "")
        if key:
            d["api_key"] = f"{key[:4]}...{key[-4:]}" if len(key) >= 10 else "***"
        return d


class ProviderRegistry:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or _DEFAULT_PATH
        self._providers: dict[str, ProviderEntry] = {}
        self._load()

    def _load(self) -> None:
        selected = self.path
        if not selected.exists() and _LEGACY_PATH.exists():
            selected = _LEGACY_PATH

        if not selected.exists():
            self._providers = {}
            return

        raw = json.loads(selected.read_text(encoding="utf-8"))
        providers_raw = raw.get("providers") if isinstance(raw, dict) else raw
        providers: dict[str, ProviderEntry] = {}
        for item in providers_raw or []:
            if not isinstance(item, dict):
                continue
            entry = ProviderEntry.from_dict(item)
            if entry.name:
                providers[entry.name] = entry
        self._providers = providers

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "providers": [self._providers[k].to_dict() for k in sorted(self._providers)],
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def list(self) -> list[ProviderEntry]:
        return [self._providers[k] for k in sorted(self._providers)]

    def get(self, name: str) -> ProviderEntry:
        key = name.strip()
        if key not in self._providers:
            raise KeyError(
                f"Provider '{name}' not found. Use `python3 -m lite_synphonia providers list` first."
            )
        return self._providers[key]

    def resolve_key(self, name: str) -> str:
        entry = self.get(name)
        if not entry.api_key:
            raise KeyError(f"Provider '{name}' has no api_key configured.")
        return entry.api_key

    def upsert(
        self,
        *,
        name: str,
        base_url: str,
        model_id: str = "",
        api_key: str = "",
        services: list[str] | None = None,
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
        temperature: float = 0.1,
    ) -> ProviderEntry:
        now = datetime.now(timezone.utc).isoformat()
        key = name.strip()
        if not key:
            raise ValueError("Provider name must not be empty.")
        if not base_url.strip():
            raise ValueError("base_url must not be empty.")

        prev = self._providers.get(key)
        created_at = prev.created_at_utc if prev else now
        entry = ProviderEntry(
            name=key,
            base_url=base_url.strip().rstrip("/"),
            model_id=model_id.strip(),
            api_key=api_key.strip(),
            services=sorted({s.strip() for s in (services or []) if s.strip()}),
            timeout_seconds=max(1.0, float(timeout_seconds)),
            max_retries=max(1, int(max_retries)),
            temperature=float(temperature),
            created_at_utc=created_at,
            updated_at_utc=now,
        )
        self._providers[key] = entry
        self.save()
        return entry

    def remove(self, name: str) -> bool:
        key = name.strip()
        if key not in self._providers:
            return False
        del self._providers[key]
        self.save()
        return True


_REGISTRY: ProviderRegistry | None = None


def get_registry() -> ProviderRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = ProviderRegistry()
    return _REGISTRY

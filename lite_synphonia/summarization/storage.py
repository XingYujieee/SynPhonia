from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .config import AppConfig


def write_results_snapshot(
    output_path: str | Path,
    config: AppConfig,
    results: list[dict[str, object]],
    completed: bool,
    runtime: dict[str, object] | None = None,
    consolidated_summary: dict[str, object] | None = None,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, object] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed": completed,
        "config": config.to_dict(),
        "results": results,
    }
    if consolidated_summary:
        payload["consolidated_summary"] = consolidated_summary
    if runtime:
        payload["runtime"] = runtime

    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

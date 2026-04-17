from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


STAGE_STATUS_SUCCESS = "success"
STAGE_STATUS_WARNING = "warning"
STAGE_STATUS_BLOCKED = "blocked"
STAGE_STATUS_SKIPPED = "skipped"


@dataclass(frozen=True)
class StageStatus:
    stage: str
    status: str
    reason: str
    upstream_dependency: str = ""
    quality_decision: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "stage": self.stage,
            "status": self.status,
            "reason": self.reason,
        }
        if self.upstream_dependency:
            payload["upstream_dependency"] = self.upstream_dependency
        if self.quality_decision:
            payload["quality_decision"] = self.quality_decision
        if self.details:
            payload["details"] = self.details
        return payload


def make_stage_status(
    stage: str,
    status: str,
    reason: str,
    *,
    upstream_dependency: str = "",
    quality_decision: str = "",
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return StageStatus(
        stage=stage,
        status=status,
        reason=reason,
        upstream_dependency=upstream_dependency,
        quality_decision=quality_decision,
        details=details or {},
    ).to_dict()

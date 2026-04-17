from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


def _normalize_keywords(raw_keywords: Any) -> tuple[str, ...]:
    if raw_keywords is None:
        return ()

    if isinstance(raw_keywords, str):
        items = [item.strip() for item in raw_keywords.split(",")]
    else:
        items = [str(item).strip() for item in raw_keywords]

    seen: set[str] = set()
    normalized: list[str] = []
    for item in items:
        if not item:
            continue
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(item)
    return tuple(normalized)


def _parse_datetime(value: Any, field_name: str) -> datetime:
    text = str(value).strip()
    if not text:
        raise ValueError(f"Field '{field_name}' is required.")
    try:
        return datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(
            f"Field '{field_name}' must be an ISO formatted datetime string."
        ) from exc


def _resolve_optional_path(value: Any, *, base_dir: Path | None = None) -> str | None:
    text = str(value).strip() if value is not None else ""
    if not text:
        return None

    path = Path(text)
    if base_dir is not None and not path.is_absolute():
        path = (base_dir / path).resolve()
    return str(path)


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, set, dict)):
        return not value
    return not str(value).strip()


def canonical_relation_id(activity_a: str, activity_b: str) -> str:
    left, right = sorted((activity_a, activity_b))
    return f"{left}__{right}"


class InputValidationError(ValueError):
    def __init__(
        self,
        *,
        missing_fields: list[str] | tuple[str, ...] = (),
        invalid_fields: dict[str, str] | None = None,
    ) -> None:
        self.missing_fields = tuple(dict.fromkeys(missing_fields))
        self.invalid_fields = dict(invalid_fields or {})

        detail_parts: list[str] = []
        if self.missing_fields:
            detail_parts.append(
                "missing fields: " + ", ".join(self.missing_fields)
            )
        if self.invalid_fields:
            detail_parts.append(
                "invalid fields: "
                + "; ".join(
                    f"{field} ({message})"
                    for field, message in sorted(self.invalid_fields.items())
                )
            )

        message = "Invalid activity record"
        if detail_parts:
            message = f"{message}: {'; '.join(detail_parts)}"
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class ActivityIngestRecord:
    activity_id: str
    start_time: datetime
    end_time: datetime
    transcript_text: str
    summary_text: str
    summary_of_summary: str
    keywords: tuple[str, ...]
    keywords_of_keywords: tuple[str, ...]
    ppt_present: bool
    activity_intro: str
    activity_name: str
    activity_dir: str | None = None
    transcript_file_path: str | None = None
    summary_file_path: str | None = None
    ppt_file_path: str | None = None
    ppt_id: str | None = None
    transcript_meta: dict[str, Any] = field(default_factory=dict)
    summary_meta: dict[str, Any] = field(default_factory=dict)
    matched_slides: tuple[dict[str, Any], ...] = ()
    ppt_text_excerpt: str | None = None
    scene_type: str | None = None

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
        *,
        base_dir: Path | None = None,
    ) -> "ActivityIngestRecord":
        missing_fields: list[str] = []
        invalid_fields: dict[str, str] = {}

        required_text_fields = (
            "activity_id",
            "start_time",
            "end_time",
            "transcript_text",
            "summary_text",
            "summary_of_summary",
            "activity_intro",
            "activity_name",
        )
        for field_name in required_text_fields:
            if _is_missing_value(payload.get(field_name)):
                missing_fields.append(field_name)

        keywords = _normalize_keywords(payload.get("keywords"))
        if not keywords:
            missing_fields.append("keywords")

        keywords_of_keywords = _normalize_keywords(payload.get("keywords_of_keywords"))
        if not keywords_of_keywords:
            missing_fields.append("keywords_of_keywords")

        start_time: datetime | None = None
        end_time: datetime | None = None
        if "start_time" not in missing_fields:
            try:
                start_time = _parse_datetime(payload.get("start_time"), "start_time")
            except ValueError as exc:
                invalid_fields["start_time"] = str(exc)
        if "end_time" not in missing_fields:
            try:
                end_time = _parse_datetime(payload.get("end_time"), "end_time")
            except ValueError as exc:
                invalid_fields["end_time"] = str(exc)
        if start_time is not None and end_time is not None and end_time < start_time:
            invalid_fields["end_time"] = (
                "Field 'end_time' must not be earlier than 'start_time'."
            )

        ppt_present = bool(payload.get("ppt_present", False))
        ppt_file_path = _resolve_optional_path(
            payload.get("ppt_file_path"),
            base_dir=base_dir,
        )
        ppt_id = str(payload.get("ppt_id", "")).strip() or None
        if ppt_present and not (ppt_file_path or ppt_id):
            invalid_fields["ppt_present"] = (
                "When 'ppt_present' is true, either 'ppt_file_path' or 'ppt_id' must be provided."
            )

        if missing_fields or invalid_fields:
            raise InputValidationError(
                missing_fields=missing_fields,
                invalid_fields=invalid_fields,
            )

        transcript_meta = dict(payload.get("transcript_meta") or {})
        summary_meta = dict(payload.get("summary_meta") or {})
        matched_slides = tuple(payload.get("matched_slides") or ())
        ppt_text_excerpt = str(payload.get("ppt_text_excerpt", "")).strip() or None
        scene_type = str(payload.get("scene_type", "")).strip() or None
        activity_dir = _resolve_optional_path(payload.get("activity_dir"), base_dir=base_dir)
        transcript_file_path = _resolve_optional_path(
            payload.get("transcript_file_path"),
            base_dir=base_dir,
        )
        summary_file_path = _resolve_optional_path(
            payload.get("summary_file_path"),
            base_dir=base_dir,
        )

        return cls(
            activity_id=str(payload["activity_id"]).strip(),
            start_time=start_time,
            end_time=end_time,
            transcript_text=str(payload["transcript_text"]).strip(),
            summary_text=str(payload["summary_text"]).strip(),
            summary_of_summary=str(payload["summary_of_summary"]).strip(),
            keywords=keywords,
            keywords_of_keywords=keywords_of_keywords,
            ppt_present=ppt_present,
            activity_intro=str(payload["activity_intro"]).strip(),
            activity_name=str(payload["activity_name"]).strip(),
            activity_dir=activity_dir,
            transcript_file_path=transcript_file_path,
            summary_file_path=summary_file_path,
            ppt_file_path=ppt_file_path,
            ppt_id=ppt_id,
            transcript_meta=transcript_meta,
            summary_meta=summary_meta,
            matched_slides=matched_slides,
            ppt_text_excerpt=ppt_text_excerpt,
            scene_type=scene_type,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "activity_id": self.activity_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "transcript_text": self.transcript_text,
            "summary_text": self.summary_text,
            "summary_of_summary": self.summary_of_summary,
            "keywords": list(self.keywords),
            "keywords_of_keywords": list(self.keywords_of_keywords),
            "ppt_present": self.ppt_present,
            "activity_intro": self.activity_intro,
            "activity_name": self.activity_name,
            "activity_dir": self.activity_dir,
            "transcript_file_path": self.transcript_file_path,
            "summary_file_path": self.summary_file_path,
            "ppt_file_path": self.ppt_file_path,
            "ppt_id": self.ppt_id,
            "transcript_meta": self.transcript_meta,
            "summary_meta": self.summary_meta,
            "matched_slides": list(self.matched_slides),
            "ppt_text_excerpt": self.ppt_text_excerpt,
            "scene_type": self.scene_type,
        }


@dataclass(frozen=True, slots=True)
class StoredActivityRecord:
    activity_id: str
    start_time: datetime
    end_time: datetime
    transcript_text: str
    summary_text: str
    summary_of_summary: str
    keywords: tuple[str, ...]
    keywords_of_keywords: tuple[str, ...]
    ppt_present: bool
    activity_intro: str
    activity_name: str
    activity_dir: str | None
    transcript_file_path: str | None
    summary_file_path: str | None
    ppt_file_path: str | None
    ppt_id: str | None
    transcript_meta: dict[str, Any]
    summary_meta: dict[str, Any]
    matched_slides: tuple[dict[str, Any], ...]
    ppt_text_excerpt: str | None
    scene_type: str | None
    metadata_path: Path | None = None

    @property
    def start_date(self) -> str:
        return self.start_time.date().isoformat()

    @property
    def duration_minutes(self) -> int:
        return max(0, int((self.end_time - self.start_time).total_seconds() // 60))

    @property
    def title(self) -> str:
        if self.activity_name.strip():
            return self.activity_name
        if self.keywords:
            return self.keywords[0]
        summary = self.summary_text.strip()
        if not summary:
            return self.activity_id
        return summary[:24] + ("..." if len(summary) > 24 else "")

    @property
    def transcript_artifact_path(self) -> Path | None:
        return Path(self.transcript_file_path) if self.transcript_file_path else None

    @property
    def summary_artifact_path(self) -> Path | None:
        return Path(self.summary_file_path) if self.summary_file_path else None

    def to_metadata_dict(self) -> dict[str, Any]:
        return {
            "activity_id": self.activity_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "transcript_text": self.transcript_text,
            "summary_text": self.summary_text,
            "summary_of_summary": self.summary_of_summary,
            "keywords": list(self.keywords),
            "keywords_of_keywords": list(self.keywords_of_keywords),
            "ppt_present": self.ppt_present,
            "activity_intro": self.activity_intro,
            "activity_name": self.activity_name,
            "activity_dir": self.activity_dir,
            "transcript_file_path": self.transcript_file_path,
            "summary_file_path": self.summary_file_path,
            "ppt_file_path": self.ppt_file_path,
            "ppt_id": self.ppt_id,
            "transcript_meta": self.transcript_meta,
            "summary_meta": self.summary_meta,
            "matched_slides": list(self.matched_slides),
            "ppt_text_excerpt": self.ppt_text_excerpt,
            "scene_type": self.scene_type,
        }


@dataclass(frozen=True, slots=True)
class RelationEdge:
    relation_id: str
    source_activity_id: str
    target_activity_id: str
    strength: str
    state: str
    reasons: tuple[str, ...]
    source_type: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "relation_id": self.relation_id,
            "source_activity_id": self.source_activity_id,
            "target_activity_id": self.target_activity_id,
            "strength": self.strength,
            "state": self.state,
            "reasons": list(self.reasons),
            "source_type": self.source_type,
        }


@dataclass(frozen=True, slots=True)
class RelationOverride:
    relation_id: str
    activity_a: str
    activity_b: str
    action: str
    strength: str | None
    reason: str
    edited_at: datetime

    @classmethod
    def create(
        cls,
        *,
        activity_a: str,
        activity_b: str,
        action: str,
        strength: str | None = None,
        reason: str = "",
        edited_at: datetime | None = None,
    ) -> "RelationOverride":
        return cls(
            relation_id=canonical_relation_id(activity_a, activity_b),
            activity_a=activity_a,
            activity_b=activity_b,
            action=action,
            strength=strength,
            reason=reason.strip(),
            edited_at=edited_at or datetime.now(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "relation_id": self.relation_id,
            "activity_a": self.activity_a,
            "activity_b": self.activity_b,
            "action": self.action,
            "strength": self.strength,
            "reason": self.reason,
            "edited_at": self.edited_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RelationOverride":
        return cls(
            relation_id=str(payload["relation_id"]),
            activity_a=str(payload["activity_a"]),
            activity_b=str(payload["activity_b"]),
            action=str(payload["action"]),
            strength=str(payload.get("strength")) if payload.get("strength") else None,
            reason=str(payload.get("reason", "")),
            edited_at=_parse_datetime(payload.get("edited_at"), "edited_at"),
        )

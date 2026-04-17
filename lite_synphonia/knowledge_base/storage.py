from __future__ import annotations

import json
from pathlib import Path

from .schemas import (
    ActivityIngestRecord,
    RelationOverride,
    StoredActivityRecord,
)


class KnowledgeBaseStore:
    def __init__(self, workspace: str | Path) -> None:
        self.workspace = Path(workspace)
        self.records_dir = self.workspace / "records"
        self.overrides_path = self.workspace / "relation_overrides.json"
        self.ensure_layout()

    def ensure_layout(self) -> None:
        self.records_dir.mkdir(parents=True, exist_ok=True)
        if not self.overrides_path.exists():
            self.overrides_path.write_text("[]\n", encoding="utf-8")

    def ingest(self, record: ActivityIngestRecord) -> StoredActivityRecord:
        record_dir = self.records_dir / record.activity_id
        record_dir.mkdir(parents=True, exist_ok=True)

        metadata_path = record_dir / "record.json"

        transcript_file_path = record.transcript_file_path
        if transcript_file_path is None:
            fallback_path = record_dir / "transcript.txt"
            fallback_path.write_text(record.transcript_text, encoding="utf-8")
            transcript_file_path = str(fallback_path)

        summary_file_path = record.summary_file_path
        if summary_file_path is None:
            fallback_path = record_dir / "summary.txt"
            fallback_path.write_text(record.summary_text, encoding="utf-8")
            summary_file_path = str(fallback_path)

        activity_dir = record.activity_dir or str(record_dir)

        stored = StoredActivityRecord(
            activity_id=record.activity_id,
            start_time=record.start_time,
            end_time=record.end_time,
            transcript_text=record.transcript_text,
            summary_text=record.summary_text,
            summary_of_summary=record.summary_of_summary,
            keywords=record.keywords,
            keywords_of_keywords=record.keywords_of_keywords,
            ppt_present=record.ppt_present,
            activity_intro=record.activity_intro,
            activity_name=record.activity_name,
            activity_dir=activity_dir,
            transcript_file_path=transcript_file_path,
            summary_file_path=summary_file_path,
            ppt_file_path=record.ppt_file_path,
            ppt_id=record.ppt_id,
            transcript_meta=record.transcript_meta,
            summary_meta=record.summary_meta,
            matched_slides=record.matched_slides,
            ppt_text_excerpt=record.ppt_text_excerpt,
            scene_type=record.scene_type,
            metadata_path=metadata_path,
        )

        metadata_path.write_text(
            json.dumps(stored.to_metadata_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return stored

    def load_records(self) -> list[StoredActivityRecord]:
        records: list[StoredActivityRecord] = []
        for metadata_path in sorted(self.records_dir.glob("*/record.json")):
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            record_dir = metadata_path.parent
            records.append(
                self._load_record_from_payload(
                    payload,
                    metadata_path=metadata_path,
                    record_dir=record_dir,
                )
            )
        return sorted(records, key=lambda item: (item.start_time, item.activity_id))

    def save_override(self, override: RelationOverride) -> None:
        overrides = {item.relation_id: item for item in self.load_overrides()}
        overrides[override.relation_id] = override
        self.overrides_path.write_text(
            json.dumps(
                [item.to_dict() for item in sorted(overrides.values(), key=lambda it: it.relation_id)],
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def load_overrides(self) -> list[RelationOverride]:
        payload = json.loads(self.overrides_path.read_text(encoding="utf-8"))
        return [RelationOverride.from_dict(item) for item in payload]

    def clear(self) -> None:
        if self.records_dir.exists():
            for child in self.records_dir.iterdir():
                if child.is_dir():
                    for file_path in child.iterdir():
                        file_path.unlink()
                    child.rmdir()
        self.overrides_path.write_text("[]\n", encoding="utf-8")

    def _load_record_from_payload(
        self,
        payload: dict[str, object],
        *,
        metadata_path: Path,
        record_dir: Path,
    ) -> StoredActivityRecord:
        transcript_text = str(payload.get("transcript_text", "")).strip()
        transcript_file_path = _path_or_none(payload.get("transcript_file_path"))
        summary_file_path = _path_or_none(payload.get("summary_file_path"))

        legacy_transcript_path = _path_or_none(payload.get("transcript_artifact_path"))
        legacy_summary_path = _path_or_none(payload.get("summary_artifact_path"))

        if not transcript_text and legacy_transcript_path:
            legacy_path = Path(legacy_transcript_path)
            if legacy_path.exists():
                transcript_text = legacy_path.read_text(encoding="utf-8")

        summary_text = str(payload.get("summary_text", "")).strip()
        if not summary_text and legacy_summary_path:
            legacy_path = Path(legacy_summary_path)
            if legacy_path.exists():
                summary_text = legacy_path.read_text(encoding="utf-8")

        if transcript_file_path is None:
            transcript_file_path = legacy_transcript_path
        if summary_file_path is None:
            summary_file_path = legacy_summary_path

        activity_name = str(payload.get("activity_name", "")).strip()
        if not activity_name:
            activity_name = _derive_legacy_title(
                activity_id=str(payload["activity_id"]),
                summary_text=summary_text,
                keywords=payload.get("keywords"),
            )

        activity_intro = str(payload.get("activity_intro", "")).strip()
        if not activity_intro:
            activity_intro = _short_text(summary_text, limit=120)

        keywords = tuple(str(item) for item in payload.get("keywords", []))
        keywords_of_keywords = tuple(
            str(item) for item in payload.get("keywords_of_keywords", []) or keywords[:2]
        )
        summary_of_summary = str(payload.get("summary_of_summary", "")).strip()
        if not summary_of_summary:
            summary_of_summary = _short_text(summary_text, limit=120)

        activity_dir = _path_or_none(payload.get("activity_dir"))
        if activity_dir is None:
            if transcript_file_path:
                activity_dir = str(Path(transcript_file_path).parent)
            elif summary_file_path:
                activity_dir = str(Path(summary_file_path).parent)
            else:
                activity_dir = str(record_dir)

        return StoredActivityRecord(
            activity_id=str(payload["activity_id"]),
            start_time=_parse_iso(payload["start_time"]),
            end_time=_parse_iso(payload["end_time"]),
            transcript_text=transcript_text,
            summary_text=summary_text,
            summary_of_summary=summary_of_summary,
            keywords=keywords,
            keywords_of_keywords=keywords_of_keywords,
            ppt_present=bool(payload.get("ppt_present", False)),
            activity_intro=activity_intro,
            activity_name=activity_name,
            activity_dir=activity_dir,
            transcript_file_path=transcript_file_path,
            summary_file_path=summary_file_path,
            ppt_file_path=_path_or_none(payload.get("ppt_file_path")),
            ppt_id=str(payload["ppt_id"]) if payload.get("ppt_id") else None,
            transcript_meta=dict(payload.get("transcript_meta") or {}),
            summary_meta=dict(payload.get("summary_meta") or {}),
            matched_slides=tuple(payload.get("matched_slides") or ()),
            ppt_text_excerpt=str(payload.get("ppt_text_excerpt"))
            if payload.get("ppt_text_excerpt")
            else None,
            scene_type=str(payload.get("scene_type"))
            if payload.get("scene_type")
            else None,
            metadata_path=metadata_path,
        )


def _parse_iso(value: str) -> object:
    from datetime import datetime

    return datetime.fromisoformat(str(value))


def _path_or_none(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _short_text(text: str, *, limit: int) -> str:
    cleaned = " ".join(str(text).split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + "..."


def _derive_legacy_title(
    *,
    activity_id: str,
    summary_text: str,
    keywords: object,
) -> str:
    keyword_list = [str(item).strip() for item in (keywords or []) if str(item).strip()]
    if keyword_list:
        return keyword_list[0]
    if summary_text.strip():
        return _short_text(summary_text, limit=24)
    return activity_id

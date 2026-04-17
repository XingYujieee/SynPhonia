from __future__ import annotations

from pathlib import Path
from typing import Any

from knowledge_base.relations import build_content_lines, build_relation_edges
from knowledge_base.schemas import (
    ActivityIngestRecord,
    InputValidationError,
    RelationOverride,
)
from knowledge_base.search import search_in_page, search_records
from knowledge_base.storage import KnowledgeBaseStore
from knowledge_base.views import (
    TOP_LEVEL_NAVIGATION,
    build_activity_catalog,
    build_detail_panel,
    build_file_lookup_view,
    build_history_view,
    build_relation_map_view,
    build_timeline_calendar_view,
    build_timeline_line_view,
)


class KnowledgeBaseService:
    def __init__(self, workspace: str | Path) -> None:
        self.store = KnowledgeBaseStore(workspace)

    @property
    def workspace(self) -> Path:
        return self.store.workspace

    def reset(self) -> None:
        self.store.clear()

    def ingest_completed_activity(
        self,
        record: ActivityIngestRecord | dict[str, Any],
        *,
        base_dir: Path | None = None,
    ) -> dict[str, Any]:
        try:
            ingest_record = (
                record
                if isinstance(record, ActivityIngestRecord)
                else ActivityIngestRecord.from_dict(record, base_dir=base_dir)
            )
        except InputValidationError as exc:
            activity_id = None
            if isinstance(record, dict):
                activity_id = str(record.get("activity_id", "")).strip() or None
            return {
                "activity_id": activity_id,
                "status": "invalid_input",
                "missing_fields": list(exc.missing_fields),
                "invalid_fields": dict(exc.invalid_fields),
                "workspace": str(self.workspace),
                "message": str(exc),
            }

        stored = self.store.ingest(ingest_record)
        return {
            "activity_id": stored.activity_id,
            "status": "stored",
            "workspace": str(self.workspace),
        }

    def ingest_many(
        self,
        records: list[ActivityIngestRecord | dict[str, Any]],
        *,
        base_dir: Path | None = None,
    ) -> list[dict[str, Any]]:
        return [
            self.ingest_completed_activity(record, base_dir=base_dir)
            for record in records
        ]

    def set_relation_state(
        self,
        *,
        activity_a: str,
        activity_b: str,
        action: str,
        strength: str | None = None,
        reason: str = "",
    ) -> dict[str, Any]:
        override = RelationOverride.create(
            activity_a=activity_a,
            activity_b=activity_b,
            action=action,
            strength=strength,
            reason=reason,
        )
        self.store.save_override(override)
        return override.to_dict()

    def export_core_data(self, selected_activity_id: str | None = None) -> dict[str, Any]:
        records, relations, content_lines = self._load_graph_context()
        activities = build_activity_catalog(records, relations, content_lines)
        selected_activity = next(
            (item for item in activities if item["activity_id"] == selected_activity_id),
            None,
        )
        return {
            "activities": activities,
            "selected_activity": selected_activity,
            "content_lines": content_lines,
            "counts": {
                "activity_count": len(records),
                "content_line_count": len(content_lines),
                "attachment_count": sum(1 for record in records if record.ppt_present),
            },
        }

    def export_graph_view(self) -> dict[str, Any]:
        records, relations, _ = self._load_graph_context()
        return build_relation_map_view(records, relations)

    def export_view_bundle(self, selected_activity_id: str | None = None) -> dict[str, Any]:
        records, relations, content_lines = self._load_graph_context()

        selected_record = None
        if selected_activity_id is not None:
            selected_record = next(
                (record for record in records if record.activity_id == selected_activity_id),
                None,
            )

        return {
            "navigation": list(TOP_LEVEL_NAVIGATION),
            "history": build_history_view(records, relations, content_lines),
            "relation_map": build_relation_map_view(records, relations),
            "timeline_calendar": build_timeline_calendar_view(records),
            "timeline_line_view": build_timeline_line_view(content_lines),
            "file_lookup": build_file_lookup_view(records),
            "detail_panel": build_detail_panel(selected_record, relations, content_lines)
            if selected_record is not None
            else None,
        }

    def export_all_views(self, selected_activity_id: str | None = None) -> dict[str, Any]:
        return {
            "core_data": self.export_core_data(selected_activity_id=selected_activity_id),
            "graph_view": self.export_graph_view(),
            "legacy_view_bundle": self.export_view_bundle(
                selected_activity_id=selected_activity_id
            ),
        }

    def search(self, query: str) -> dict[str, Any]:
        records = self.store.load_records()
        return {
            "query": query,
            "results": search_records(records, query),
        }

    def search_current_page(self, text: str, query: str) -> dict[str, Any]:
        return search_in_page(text, query)

    def _load_graph_context(self) -> tuple[list[Any], list[Any], list[dict[str, object]]]:
        records = self.store.load_records()
        overrides = self.store.load_overrides()
        relations = build_relation_edges(records, overrides)
        content_lines = build_content_lines(records, relations)
        return records, relations, content_lines

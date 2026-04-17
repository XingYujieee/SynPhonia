from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .schemas import RelationEdge, StoredActivityRecord


TOP_LEVEL_NAVIGATION = (
    "历史记录",
    "关联地图",
    "记录时间线",
    "文件查找",
)

HISTORY_CARD_LABELS = (
    "全部记录",
    "内容主线",
    "带附件的记录",
    "待确认关联",
)


def build_activity_catalog(
    records: list[StoredActivityRecord],
    relations: list[RelationEdge],
    content_lines: list[dict[str, object]],
) -> list[dict[str, object]]:
    relations_by_activity = _relations_by_activity(relations)
    content_line_by_activity = _content_line_by_activity(content_lines)
    return [
        build_activity_record(
            record,
            relations_by_activity.get(record.activity_id, []),
            content_line_by_activity.get(record.activity_id),
        )
        for record in records
    ]


def build_history_view(
    records: list[StoredActivityRecord],
    relations: list[RelationEdge],
    content_lines: list[dict[str, object]],
) -> dict[str, object]:
    pending_relations = [item for item in relations if item.state == "pending"]
    attachment_records = [record for record in records if record.ppt_present]

    return {
        "navigation": list(TOP_LEVEL_NAVIGATION),
        "statistics_cards": [
            {
                "label": HISTORY_CARD_LABELS[0],
                "count": len(records),
                "target_view": "full_record_list",
            },
            {
                "label": HISTORY_CARD_LABELS[1],
                "count": len(content_lines),
                "target_view": "content_lines",
            },
            {
                "label": HISTORY_CARD_LABELS[2],
                "count": len(attachment_records),
                "target_view": "attachment_records",
            },
            {
                "label": HISTORY_CARD_LABELS[3],
                "count": len(pending_relations),
                "target_view": "pending_relations",
            },
        ],
        "full_record_list": [build_record_list_item(record) for record in records],
        "content_lines": content_lines,
        "attachment_records": [build_record_list_item(record) for record in attachment_records],
        "pending_relations": [build_relation_detail(item) for item in pending_relations],
    }


def build_relation_map_view(
    records: list[StoredActivityRecord],
    relations: list[RelationEdge],
) -> dict[str, object]:
    nodes = [
        {
            "node_id": record.activity_id,
            "node_type": "activity",
            "title": record.title,
            "activity_name": record.activity_name,
            "activity_intro": record.activity_intro,
            "summary_of_summary": record.summary_of_summary,
            "scene_type": record.scene_type,
            "start_time": record.start_time.isoformat(),
            "keywords": list(record.keywords),
            "keywords_of_keywords": list(record.keywords_of_keywords),
            "has_ppt": record.ppt_present,
        }
        for record in records
    ]
    edges = [build_relation_detail(item) for item in relations if item.state != "removed"]
    return {"nodes": nodes, "edges": edges}


def build_timeline_calendar_view(records: list[StoredActivityRecord]) -> dict[str, object]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[record.start_date].append(
            {
                "activity_id": record.activity_id,
                "title": record.title,
                "activity_name": record.activity_name,
                "start_time": record.start_time.strftime("%H:%M"),
                "end_time": record.end_time.strftime("%H:%M"),
                "summary": record.summary_of_summary,
            }
        )
    return {
        "mode": "chronological_calendar",
        "dates": [
            {"date": date, "entries": entries}
            for date, entries in sorted(grouped.items(), key=lambda item: item[0])
        ],
    }


def build_timeline_line_view(content_lines: list[dict[str, object]]) -> dict[str, object]:
    return {"mode": "content_line_timeline", "content_lines": content_lines}


def build_file_lookup_view(records: list[StoredActivityRecord]) -> dict[str, object]:
    return {
        "activity_groups": [
            {
                "activity_id": record.activity_id,
                "title": record.title,
                "activity_name": record.activity_name,
                "files": [
                    _build_transcript_file_entry(record),
                    _build_summary_file_entry(record),
                    _build_ppt_file_entry(record),
                ],
            }
            for record in records
        ]
    }


def build_detail_panel(
    record: StoredActivityRecord,
    relations: list[RelationEdge],
    content_lines: list[dict[str, object]],
) -> dict[str, object]:
    related = [
        build_relation_detail(item)
        for item in relations
        if item.source_activity_id == record.activity_id or item.target_activity_id == record.activity_id
    ]
    current_line = next(
        (
            line
            for line in content_lines
            if any(activity["activity_id"] == record.activity_id for activity in line["activities"])
        ),
        None,
    )

    detail = build_activity_record(record, related, current_line)
    detail.update(
        {
            "summary": record.summary_text,
            "summary_of_summary": record.summary_of_summary,
            "keywords": list(record.keywords),
            "keywords_of_keywords": list(record.keywords_of_keywords),
            "transcript_preview": record.transcript_text[:600],
        }
    )
    return detail


def build_activity_record(
    record: StoredActivityRecord,
    relations: list[dict[str, object]] | list[RelationEdge],
    content_line: dict[str, object] | None,
) -> dict[str, object]:
    normalized_relations = [
        item if isinstance(item, dict) else build_relation_detail(item)
        for item in relations
    ]
    return {
        "activity_id": record.activity_id,
        "title": record.title,
        "activity_name": record.activity_name,
        "activity_intro": record.activity_intro,
        "scene_type": record.scene_type,
        "start_time": record.start_time.isoformat(),
        "end_time": record.end_time.isoformat(),
        "duration_minutes": record.duration_minutes,
        "transcript_text": record.transcript_text,
        "summary_text": record.summary_text,
        "summary_of_summary": record.summary_of_summary,
        "keywords": list(record.keywords),
        "keywords_of_keywords": list(record.keywords_of_keywords),
        "ppt_present": record.ppt_present,
        "activity_dir": record.activity_dir,
        "transcript_file_path": record.transcript_file_path,
        "summary_file_path": record.summary_file_path,
        "ppt_file_path": record.ppt_file_path,
        "ppt_id": record.ppt_id,
        "ppt_text_excerpt": record.ppt_text_excerpt,
        "matched_slides": list(record.matched_slides),
        "transcript_meta": dict(record.transcript_meta),
        "summary_meta": dict(record.summary_meta),
        "relations": normalized_relations,
        "content_line": content_line,
        "files": [
            _build_transcript_file_entry(record),
            _build_summary_file_entry(record),
            _build_ppt_file_entry(record),
        ],
    }


def build_record_list_item(record: StoredActivityRecord) -> dict[str, object]:
    return {
        "activity_id": record.activity_id,
        "title": record.title,
        "activity_name": record.activity_name,
        "activity_intro": record.activity_intro,
        "summary": record.summary_text,
        "summary_of_summary": record.summary_of_summary,
        "keywords": list(record.keywords),
        "keywords_of_keywords": list(record.keywords_of_keywords),
        "start_time": record.start_time.isoformat(),
        "end_time": record.end_time.isoformat(),
        "has_ppt_attachment": record.ppt_present,
        "scene_type": record.scene_type,
    }


def build_relation_detail(relation: RelationEdge) -> dict[str, object]:
    return {
        "relation_id": relation.relation_id,
        "source_activity_id": relation.source_activity_id,
        "target_activity_id": relation.target_activity_id,
        "strength": relation.strength,
        "state": relation.state,
        "reasons": list(relation.reasons),
        "source_type": relation.source_type,
    }


def _build_transcript_file_entry(record: StoredActivityRecord) -> dict[str, object]:
    path_text = record.transcript_file_path or ""
    return {
        "file_type": "transcript_text",
        "label": "转录文本",
        "path": path_text,
        "exists": _path_exists(path_text),
        "preview_mode": "inline_text",
        "preview_text": record.transcript_text,
    }


def _build_summary_file_entry(record: StoredActivityRecord) -> dict[str, object]:
    path_text = record.summary_file_path or ""
    return {
        "file_type": "summary_text",
        "label": "总结文本",
        "path": path_text,
        "exists": _path_exists(path_text),
        "preview_mode": "inline_text",
        "preview_text": record.summary_text,
    }


def _build_ppt_file_entry(record: StoredActivityRecord) -> dict[str, object]:
    path_text = record.ppt_file_path or ""
    return {
        "file_type": "ppt",
        "label": "PPT",
        "path": path_text,
        "exists": _path_exists(path_text),
        "preview_mode": "external_only",
        "preview_text": record.ppt_text_excerpt,
        "ppt_id": record.ppt_id,
    }


def _path_exists(path_text: str) -> bool:
    return bool(path_text) and Path(path_text).exists()


def _relations_by_activity(
    relations: list[RelationEdge],
) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for relation in relations:
        detail = build_relation_detail(relation)
        grouped[relation.source_activity_id].append(detail)
        grouped[relation.target_activity_id].append(detail)
    return grouped


def _content_line_by_activity(
    content_lines: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    mapping: dict[str, dict[str, object]] = {}
    for line in content_lines:
        for activity in line["activities"]:
            activity_id = str(activity["activity_id"])
            mapping[activity_id] = line
    return mapping

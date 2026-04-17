from __future__ import annotations

import re
from itertools import combinations

from knowledge_base.schemas import (
    RelationEdge,
    RelationOverride,
    StoredActivityRecord,
    canonical_relation_id,
)

_LATIN_PATTERN = re.compile(r"[A-Za-z0-9]{3,}")
_CJK_BLOCK_PATTERN = re.compile(r"[\u4e00-\u9fff]{2,}")

_STOPWORDS = {
    "about",
    "after",
    "also",
    "and",
    "been",
    "class",
    "classroom",
    "during",
    "follow",
    "followup",
    "from",
    "have",
    "into",
    "meeting",
    "notes",
    "same",
    "session",
    "slide",
    "slides",
    "students",
    "teacher",
    "team",
    "that",
    "the",
    "their",
    "them",
    "then",
    "this",
    "used",
    "using",
    "week",
    "with",
    "老师",
    "学生",
    "活动",
    "内容",
    "记录",
    "讨论",
    "进行",
}


def build_relation_edges(
    records: list[StoredActivityRecord],
    overrides: list[RelationOverride],
) -> list[RelationEdge]:
    generated: dict[str, RelationEdge] = {}

    for left, right in combinations(records, 2):
        edge = _build_rule_edge(left, right)
        if edge is not None:
            generated[edge.relation_id] = edge

    for override in overrides:
        generated[override.relation_id] = _apply_override(
            generated.get(override.relation_id),
            override,
        )

    return sorted(
        generated.values(),
        key=lambda item: (
            item.state != "linked",
            _strength_rank(item.strength),
            item.source_activity_id,
            item.target_activity_id,
        ),
    )


def build_content_lines(
    records: list[StoredActivityRecord],
    relations: list[RelationEdge],
) -> list[dict[str, object]]:
    records_by_id = {record.activity_id: record for record in records}
    adjacency = {record.activity_id: set() for record in records}

    for relation in relations:
        if relation.state != "linked":
            continue
        if relation.strength not in {"strong", "medium"}:
            continue
        adjacency[relation.source_activity_id].add(relation.target_activity_id)
        adjacency[relation.target_activity_id].add(relation.source_activity_id)

    visited: set[str] = set()
    lines: list[dict[str, object]] = []
    line_number = 1

    for record in records:
        if record.activity_id in visited:
            continue

        stack = [record.activity_id]
        component: list[str] = []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            stack.extend(sorted(adjacency[current] - visited))

        component_records = sorted(
            (records_by_id[item] for item in component),
            key=lambda item: (item.start_time, item.activity_id),
        )
        lines.append(
            {
                "content_line_id": f"line-{line_number:03d}",
                "title": component_records[0].title,
                "activity_count": len(component_records),
                "activities": [_activity_line_item(item) for item in component_records],
            }
        )
        line_number += 1

    return lines


def _build_rule_edge(
    left: StoredActivityRecord,
    right: StoredActivityRecord,
) -> RelationEdge | None:
    relation_id = canonical_relation_id(left.activity_id, right.activity_id)
    reasons: list[str] = []
    score = 0.0

    keyword_overlap = _keyword_overlap(left, right)
    same_ppt_reference = False
    if keyword_overlap >= 2:
        score += 4.0
        reasons.append(f"Detected {keyword_overlap} overlapping keywords.")
    elif keyword_overlap == 1:
        score += 2.0
        reasons.append("Detected 1 overlapping keyword.")

    text_overlap = _text_overlap(left, right)
    if text_overlap >= 6:
        score += 3.0
        reasons.append(f"Detected {text_overlap} overlapping summary/text terms.")
    elif text_overlap >= 3:
        score += 1.5
        reasons.append(f"Detected {text_overlap} overlapping summary/text terms.")

    if left.ppt_present and right.ppt_present:
        if left.ppt_file_path and right.ppt_file_path and left.ppt_file_path == right.ppt_file_path:
            same_ppt_reference = True
            score += 3.0
            reasons.append("Both activities reference the same local PPT path.")
        elif left.ppt_id and right.ppt_id and left.ppt_id == right.ppt_id:
            same_ppt_reference = True
            score += 3.0
            reasons.append("Both activities reference the same PPT id.")

    day_gap = abs((right.start_time.date() - left.start_time.date()).days)
    if day_gap <= 7:
        score += 0.5
        reasons.append("Activities occurred within 7 days.")

    if left.scene_type and right.scene_type and left.scene_type == right.scene_type:
        score += 0.5
        reasons.append("Activities share the same scene type.")
    elif left.scene_type and right.scene_type and left.scene_type != right.scene_type:
        if keyword_overlap == 0 and not same_ppt_reference:
            return None
        score -= 1.0
        reasons.append("Activities belong to different scene types.")

    if score >= 7.0:
        strength = "strong"
        state = "linked"
    elif score >= 3.0:
        strength = "medium"
        state = "pending"
    elif score >= 1.5:
        strength = "weak"
        state = "visible"
    else:
        return None

    return RelationEdge(
        relation_id=relation_id,
        source_activity_id=left.activity_id,
        target_activity_id=right.activity_id,
        strength=strength,
        state=state,
        reasons=tuple(reasons),
        source_type="rule",
    )


def _apply_override(
    existing: RelationEdge | None,
    override: RelationOverride,
) -> RelationEdge:
    reasons = list(existing.reasons if existing else ())
    if override.reason:
        reasons.append(f"User note: {override.reason}")

    if override.action == "remove":
        strength = override.strength or (existing.strength if existing else "weak")
        state = "removed"
    elif override.action == "pending":
        strength = override.strength or (existing.strength if existing else "medium")
        state = "pending"
    else:
        strength = override.strength or (existing.strength if existing else "medium")
        state = "linked"

    return RelationEdge(
        relation_id=override.relation_id,
        source_activity_id=override.activity_a,
        target_activity_id=override.activity_b,
        strength=strength,
        state=state,
        reasons=tuple(reasons or ("User edited this relation.",)),
        source_type="user_override",
    )


def _keyword_overlap(left: StoredActivityRecord, right: StoredActivityRecord) -> int:
    left_terms = {_normalize_keyword(item) for item in left.keywords}
    right_terms = {_normalize_keyword(item) for item in right.keywords}
    return len({item for item in left_terms & right_terms if item})


def _text_overlap(left: StoredActivityRecord, right: StoredActivityRecord) -> int:
    left_terms = _extract_terms(left.summary_text)
    right_terms = _extract_terms(right.summary_text)
    return len(left_terms & right_terms)


def _normalize_keyword(text: str) -> str:
    return "".join(str(text).casefold().split())


def _extract_terms(text: str) -> set[str]:
    lowered = text.casefold()
    terms = {token for token in _LATIN_PATTERN.findall(lowered) if token not in _STOPWORDS}

    for block in _CJK_BLOCK_PATTERN.findall(text):
        cleaned = "".join(block.split())
        if len(cleaned) < 2:
            continue
        if len(cleaned) <= 4 and cleaned not in _STOPWORDS:
            terms.add(cleaned)
        for size in (3, 4):
            if len(cleaned) < size:
                continue
            for index in range(len(cleaned) - size + 1):
                token = cleaned[index : index + size]
                if token not in _STOPWORDS:
                    terms.add(token)
    return terms


def _strength_rank(strength: str) -> int:
    return {"strong": 0, "medium": 1, "weak": 2}.get(strength, 3)


def _activity_line_item(record: StoredActivityRecord) -> dict[str, object]:
    return {
        "activity_id": record.activity_id,
        "title": record.title,
        "date": record.start_date,
        "start_time": record.start_time.isoformat(),
        "end_time": record.end_time.isoformat(),
        "summary": record.summary_of_summary,
        "keywords": list(record.keywords),
    }

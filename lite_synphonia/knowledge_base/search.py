from __future__ import annotations

from .schemas import StoredActivityRecord


def normalize_search_text(text: str) -> str:
    return "".join(text.casefold().split())


def split_query_terms(query: str) -> list[str]:
    pieces = [normalize_search_text(piece) for piece in query.strip().split()]
    terms = [piece for piece in pieces if piece]
    return terms if terms else ([normalize_search_text(query)] if normalize_search_text(query) else [])


def search_records(
    records: list[StoredActivityRecord],
    query: str,
) -> list[dict[str, object]]:
    terms = split_query_terms(query)
    if not terms:
        return []

    matches: list[dict[str, object]] = []
    for record in records:
        fields = _record_search_fields(record)
        normalized_fields = {
            field_name: normalize_search_text(field_value)
            for field_name, field_value in fields.items()
        }
        matched_fields: set[str] = set()
        if all(
            any(term in value for value in normalized_fields.values())
            for term in terms
        ):
            for term in terms:
                for field_name, field_value in normalized_fields.items():
                    if term in field_value:
                        matched_fields.add(field_name)
            matches.append(
                {
                    "activity_id": record.activity_id,
                    "title": record.title,
                    "matched_fields": sorted(matched_fields),
                    "summary": record.summary_text,
                    "keywords": list(record.keywords),
                    "start_time": record.start_time.isoformat(),
                    "end_time": record.end_time.isoformat(),
                }
            )
    return matches


def search_in_page(text: str, query: str) -> dict[str, object]:
    normalized_text = normalize_search_text(text)
    terms = split_query_terms(query)
    if not normalized_text or not terms:
        return {"query_terms": terms, "all_terms_matched": False, "matches": []}

    matches: list[dict[str, object]] = []
    all_terms_matched = True
    for term in terms:
        index = normalized_text.find(term)
        if index < 0:
            all_terms_matched = False
            continue
        matches.append({"term": term, "normalized_index": index})

    return {
        "query_terms": terms,
        "all_terms_matched": all_terms_matched,
        "matches": matches,
    }


def _record_search_fields(record: StoredActivityRecord) -> dict[str, str]:
    return {
        "activity_id": record.activity_id,
        "summary_text": record.summary_text,
        "transcript_text": record.transcript_text,
        "keywords": " ".join(record.keywords),
        "ppt_file_path": record.ppt_file_path or "",
        "transcript_path": str(record.transcript_artifact_path),
        "summary_path": str(record.summary_artifact_path),
        "date": record.start_date,
        "start_time": record.start_time.isoformat(),
        "end_time": record.end_time.isoformat(),
    }

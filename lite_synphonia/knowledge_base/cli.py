from __future__ import annotations

import argparse
import json
from pathlib import Path

from .service import KnowledgeBaseService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the standalone knowledge base module.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest-file", help="Ingest activity records from a JSON file.")
    ingest_parser.add_argument("--workspace", required=True, help="Knowledge base workspace path.")
    ingest_parser.add_argument("--activities", required=True, help="Path to a JSON file containing activity records.")
    ingest_parser.add_argument("--reset", action="store_true", help="Clear the workspace before ingestion.")

    export_data_parser = subparsers.add_parser(
        "export-data",
        help="Export V2 core data and graph data as JSON.",
    )
    export_data_parser.add_argument("--workspace", required=True, help="Knowledge base workspace path.")
    export_data_parser.add_argument("--output", required=True, help="Path to the exported JSON file.")
    export_data_parser.add_argument(
        "--selected-activity",
        help="Optional activity id for selected activity export.",
    )

    export_parser = subparsers.add_parser("export-views", help="Export the legacy view bundle as JSON.")
    export_parser.add_argument("--workspace", required=True, help="Knowledge base workspace path.")
    export_parser.add_argument("--output", required=True, help="Path to the exported JSON file.")
    export_parser.add_argument("--selected-activity", help="Optional activity id for detail panel export.")

    relation_parser = subparsers.add_parser("set-relation", help="Apply a user relation override.")
    relation_parser.add_argument("--workspace", required=True, help="Knowledge base workspace path.")
    relation_parser.add_argument("--activity-a", required=True, help="The first activity id.")
    relation_parser.add_argument("--activity-b", required=True, help="The second activity id.")
    relation_parser.add_argument(
        "--action",
        required=True,
        choices=("confirm", "pending", "remove"),
        help="The relation override action.",
    )
    relation_parser.add_argument(
        "--strength",
        choices=("strong", "medium", "weak"),
        help="Optional relation strength override.",
    )
    relation_parser.add_argument("--reason", default="", help="Optional user note.")

    search_parser = subparsers.add_parser("search", help="Search records using the wide search scope.")
    search_parser.add_argument("--workspace", required=True, help="Knowledge base workspace path.")
    search_parser.add_argument("--query", required=True, help="Search query text.")

    demo_parser = subparsers.add_parser(
        "demo",
        help="Ingest a JSON file and export V2 data plus the legacy compatibility bundle.",
    )
    demo_parser.add_argument("--workspace", required=True, help="Knowledge base workspace path.")
    demo_parser.add_argument("--activities", required=True, help="Path to a JSON file containing activity records.")
    demo_parser.add_argument("--output", required=True, help="Path to the exported JSON file.")
    demo_parser.add_argument("--reset", action="store_true", help="Clear the workspace before ingestion.")
    demo_parser.add_argument("--selected-activity", help="Optional activity id for selected export.")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = str(args.command)

    if command == "ingest-file":
        service = KnowledgeBaseService(args.workspace)
        if args.reset:
            service.reset()
        activities_path = Path(args.activities)
        payload = json.loads(activities_path.read_text(encoding="utf-8"))
        results = service.ingest_many(list(payload), base_dir=activities_path.parent)
        print(json.dumps({"ingested": results}, ensure_ascii=False, indent=2))
        return 0

    if command == "export-data":
        service = KnowledgeBaseService(args.workspace)
        export_payload = {
            "core_data": service.export_core_data(selected_activity_id=args.selected_activity),
            "graph_view": service.export_graph_view(),
        }
        _write_json(Path(args.output), export_payload)
        print(json.dumps({"exported_to": str(args.output)}, ensure_ascii=False, indent=2))
        return 0

    if command == "export-views":
        service = KnowledgeBaseService(args.workspace)
        bundle = service.export_view_bundle(selected_activity_id=args.selected_activity)
        _write_json(Path(args.output), bundle)
        print(json.dumps({"exported_to": str(args.output)}, ensure_ascii=False, indent=2))
        return 0

    if command == "set-relation":
        service = KnowledgeBaseService(args.workspace)
        result = service.set_relation_state(
            activity_a=args.activity_a,
            activity_b=args.activity_b,
            action=args.action,
            strength=args.strength,
            reason=args.reason,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if command == "search":
        service = KnowledgeBaseService(args.workspace)
        result = service.search(args.query)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if command == "demo":
        service = KnowledgeBaseService(args.workspace)
        if args.reset:
            service.reset()
        activities_path = Path(args.activities)
        payload = json.loads(activities_path.read_text(encoding="utf-8"))
        ingest_results = service.ingest_many(list(payload), base_dir=activities_path.parent)
        export_payload = service.export_all_views(selected_activity_id=args.selected_activity)
        _write_json(Path(args.output), export_payload)
        print(
            json.dumps(
                {
                    "workspace": str(service.workspace),
                    "exported_to": str(args.output),
                    "record_count": len(export_payload["core_data"]["activities"]),
                    "stored_count": sum(1 for item in ingest_results if item["status"] == "stored"),
                    "invalid_count": sum(
                        1 for item in ingest_results if item["status"] == "invalid_input"
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    raise ValueError(f"Unsupported command: {command}")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())

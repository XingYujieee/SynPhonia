from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from uuid import uuid4

from knowledge_base.service import KnowledgeBaseService


def _write_text_file(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)


def _sample_records(base_dir: Path) -> list[dict[str, object]]:
    activity_root = base_dir / "activities"
    assets_dir = base_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    for file_name in ("classroom_week01.pptx", "classroom_week02.pptx", "meeting_budget_review.pptx"):
        (assets_dir / file_name).write_text("placeholder", encoding="utf-8")

    classroom_001_transcript = (
        "The teacher introduced concept mapping, overlap detection, and a note taking routine "
        "based on the week one slides."
    )
    classroom_001_summary = (
        "The class introduced concept mapping and overlap detection with the week one slide deck."
    )
    classroom_002_transcript = (
        "The teacher revisited concept mapping and overlap detection on the same week one slides "
        "and added a reflection checklist."
    )
    classroom_002_summary = (
        "The second classroom session revisited concept mapping and overlap detection and added "
        "a reflection checklist."
    )
    classroom_003_transcript = (
        "The teacher opened a checkpoint class about overlap detection, reflection, and concept "
        "review using a feedback checklist."
    )
    classroom_003_summary = (
        "This checkpoint session revisited overlap detection, reflection, and concept review "
        "with a feedback checklist."
    )
    meeting_001_transcript = (
        "The meeting focused on budget review, risk tracking, and the procurement timeline."
    )
    meeting_001_summary = (
        "The meeting covered budget review, cost pressure, and a shared risk register."
    )
    meeting_002_transcript = (
        "The follow-up meeting returned to budget review, risk register updates, and supplier timing."
    )
    meeting_002_summary = (
        "The follow-up meeting revisited budget review and the risk register while aligning cost control."
    )

    records = [
        {
            "activity_id": "activity-classroom-001",
            "start_time": "2026-04-01T09:00:00",
            "end_time": "2026-04-01T10:15:00",
            "transcript_text": classroom_001_transcript,
            "summary_text": classroom_001_summary,
            "summary_of_summary": "Introduced concept mapping and overlap detection.",
            "keywords": ["concept mapping", "overlap detection", "week one slides"],
            "keywords_of_keywords": ["learning design", "review loop"],
            "ppt_present": True,
            "ppt_file_path": str(assets_dir / "classroom_week01.pptx"),
            "activity_intro": "Week 1 classroom activity about concept mapping and note structure.",
            "activity_name": "Week 1 concept workshop",
            "activity_dir": str(activity_root / "activity-classroom-001"),
            "transcript_file_path": _write_text_file(
                activity_root / "activity-classroom-001" / "transcript.txt",
                classroom_001_transcript,
            ),
            "summary_file_path": _write_text_file(
                activity_root / "activity-classroom-001" / "summary.txt",
                classroom_001_summary,
            ),
            "scene_type": "classroom",
        },
        {
            "activity_id": "activity-classroom-002",
            "start_time": "2026-04-03T09:00:00",
            "end_time": "2026-04-03T10:05:00",
            "transcript_text": classroom_002_transcript,
            "summary_text": classroom_002_summary,
            "summary_of_summary": "Revisited concept mapping and added a reflection checklist.",
            "keywords": ["concept mapping", "overlap detection", "reflection checklist"],
            "keywords_of_keywords": ["review loop", "assignment preparation"],
            "ppt_present": True,
            "ppt_file_path": str(assets_dir / "classroom_week01.pptx"),
            "activity_intro": "Follow-up classroom activity that extends the first workshop.",
            "activity_name": "Week 1 checklist follow-up",
            "activity_dir": str(activity_root / "activity-classroom-002"),
            "transcript_file_path": _write_text_file(
                activity_root / "activity-classroom-002" / "transcript.txt",
                classroom_002_transcript,
            ),
            "summary_file_path": _write_text_file(
                activity_root / "activity-classroom-002" / "summary.txt",
                classroom_002_summary,
            ),
            "scene_type": "classroom",
        },
        {
            "activity_id": "activity-classroom-003",
            "start_time": "2026-04-10T09:00:00",
            "end_time": "2026-04-10T09:50:00",
            "transcript_text": classroom_003_transcript,
            "summary_text": classroom_003_summary,
            "summary_of_summary": "Checkpoint class on overlap detection and review.",
            "keywords": ["overlap detection", "feedback checklist", "concept review"],
            "keywords_of_keywords": ["review loop", "checkpoint session"],
            "ppt_present": True,
            "ppt_file_path": str(assets_dir / "classroom_week02.pptx"),
            "activity_intro": "Checkpoint classroom activity that keeps the same review theme.",
            "activity_name": "Week 2 checkpoint class",
            "activity_dir": str(activity_root / "activity-classroom-003"),
            "transcript_file_path": _write_text_file(
                activity_root / "activity-classroom-003" / "transcript.txt",
                classroom_003_transcript,
            ),
            "summary_file_path": _write_text_file(
                activity_root / "activity-classroom-003" / "summary.txt",
                classroom_003_summary,
            ),
            "scene_type": "classroom",
            "ppt_text_excerpt": "Slides focus on overlap detection and checklist prompts.",
        },
        {
            "activity_id": "activity-meeting-001",
            "start_time": "2026-04-02T14:00:00",
            "end_time": "2026-04-02T15:00:00",
            "transcript_text": meeting_001_transcript,
            "summary_text": meeting_001_summary,
            "summary_of_summary": "Budget review and risk tracking kickoff.",
            "keywords": ["budget review", "risk register", "procurement timeline"],
            "keywords_of_keywords": ["cost control", "project planning"],
            "ppt_present": False,
            "activity_intro": "Project meeting about budget pressure and procurement planning.",
            "activity_name": "Budget planning kickoff",
            "activity_dir": str(activity_root / "activity-meeting-001"),
            "transcript_file_path": _write_text_file(
                activity_root / "activity-meeting-001" / "transcript.txt",
                meeting_001_transcript,
            ),
            "summary_file_path": _write_text_file(
                activity_root / "activity-meeting-001" / "summary.txt",
                meeting_001_summary,
            ),
            "scene_type": "meeting",
        },
        {
            "activity_id": "activity-meeting-002",
            "start_time": "2026-04-09T14:00:00",
            "end_time": "2026-04-09T15:10:00",
            "transcript_text": meeting_002_transcript,
            "summary_text": meeting_002_summary,
            "summary_of_summary": "Follow-up budget review with updated risk actions.",
            "keywords": ["budget review", "risk register", "cost control"],
            "keywords_of_keywords": ["project planning", "supplier timing"],
            "ppt_present": True,
            "ppt_file_path": str(assets_dir / "meeting_budget_review.pptx"),
            "activity_intro": "Follow-up project meeting that updates cost control and supplier timing.",
            "activity_name": "Budget review follow-up",
            "activity_dir": str(activity_root / "activity-meeting-002"),
            "transcript_file_path": _write_text_file(
                activity_root / "activity-meeting-002" / "transcript.txt",
                meeting_002_transcript,
            ),
            "summary_file_path": _write_text_file(
                activity_root / "activity-meeting-002" / "summary.txt",
                meeting_002_summary,
            ),
            "scene_type": "meeting",
            "ppt_text_excerpt": "Slides summarize budget review decisions and supplier timing.",
        },
    ]
    return records


class KnowledgeBaseServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.test_root = (
            Path.cwd() / ".tmp_test_runs" / "knowledge_base_tests" / uuid4().hex
        )
        self.workspace = self.test_root / "kb_workspace"
        self.test_root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        if self.test_root.exists():
            shutil.rmtree(self.test_root)

    def test_ingestion_creates_v2_core_records_and_keeps_external_file_refs(self) -> None:
        service = KnowledgeBaseService(self.workspace)
        service.ingest_many(_sample_records(self.test_root))

        core_data = service.export_core_data(selected_activity_id="activity-classroom-001")
        self.assertEqual(core_data["counts"]["activity_count"], 5)
        self.assertEqual(core_data["selected_activity"]["activity_name"], "Week 1 concept workshop")
        self.assertEqual(
            core_data["selected_activity"]["summary_of_summary"],
            "Introduced concept mapping and overlap detection.",
        )
        self.assertTrue(
            Path(core_data["selected_activity"]["transcript_file_path"]).exists()
        )
        self.assertEqual(
            Path(core_data["selected_activity"]["transcript_file_path"]).parent.name,
            "activity-classroom-001",
        )

        legacy_bundle = service.export_view_bundle(selected_activity_id="activity-classroom-001")
        self.assertEqual(legacy_bundle["history"]["statistics_cards"][0]["label"], "全部记录")
        self.assertEqual(legacy_bundle["detail_panel"]["activity_id"], "activity-classroom-001")

    def test_missing_required_fields_return_invalid_input_instead_of_storing(self) -> None:
        service = KnowledgeBaseService(self.workspace)
        result = service.ingest_completed_activity(
            {
                "activity_id": "invalid-activity",
                "start_time": "2026-04-05T10:00:00",
                "end_time": "2026-04-05T10:45:00",
                "transcript_text": "Transcript exists.",
                "summary_text": "Summary exists.",
                "keywords": ["one keyword"],
                "ppt_present": False,
            }
        )

        self.assertEqual(result["status"], "invalid_input")
        self.assertIn("summary_of_summary", result["missing_fields"])
        self.assertIn("keywords_of_keywords", result["missing_fields"])
        self.assertIn("activity_intro", result["missing_fields"])
        self.assertIn("activity_name", result["missing_fields"])
        self.assertEqual(service.export_core_data()["counts"]["activity_count"], 0)

    def test_missing_text_file_paths_fall_back_to_workspace_files(self) -> None:
        service = KnowledgeBaseService(self.workspace)
        result = service.ingest_completed_activity(
            {
                "activity_id": "minimal-activity",
                "start_time": "2026-04-05T10:00:00",
                "end_time": "2026-04-05T10:45:00",
                "transcript_text": "Transcript body for the minimal activity.",
                "summary_text": "Summary body for the minimal activity.",
                "summary_of_summary": "Minimal activity summary.",
                "keywords": ["minimal topic", "activity note"],
                "keywords_of_keywords": ["minimal", "note"],
                "ppt_present": False,
                "activity_intro": "A minimal activity used to test fallback file creation.",
                "activity_name": "Minimal activity",
            }
        )

        self.assertEqual(result["status"], "stored")
        selected = service.export_core_data(selected_activity_id="minimal-activity")[
            "selected_activity"
        ]
        self.assertTrue(Path(selected["transcript_file_path"]).exists())
        self.assertTrue(Path(selected["summary_file_path"]).exists())
        self.assertEqual(
            Path(selected["transcript_file_path"]).parent,
            self.workspace / "records" / "minimal-activity",
        )

    def test_relations_and_content_lines_are_built_for_v2_records(self) -> None:
        service = KnowledgeBaseService(self.workspace)
        service.ingest_many(_sample_records(self.test_root))

        graph_view = service.export_graph_view()
        strong_ids = {
            item["relation_id"]
            for item in graph_view["edges"]
            if item["strength"] == "strong" and item["state"] == "linked"
        }
        pending_ids = {
            item["relation_id"]
            for item in graph_view["edges"]
            if item["state"] == "pending"
        }
        self.assertIn("activity-classroom-001__activity-classroom-002", strong_ids)
        self.assertIn("activity-classroom-002__activity-classroom-003", pending_ids)

        core_data = service.export_core_data()
        self.assertGreaterEqual(len(core_data["content_lines"]), 3)

    def test_search_uses_new_fields_and_ignores_spaces(self) -> None:
        service = KnowledgeBaseService(self.workspace)
        service.ingest_many(_sample_records(self.test_root))

        result = service.search("budget   review   risk")
        matched_ids = {item["activity_id"] for item in result["results"]}
        self.assertEqual(matched_ids, {"activity-meeting-001", "activity-meeting-002"})

        result_by_name = service.search("checkpoint class")
        matched_name_ids = {item["activity_id"] for item in result_by_name["results"]}
        self.assertEqual(matched_name_ids, {"activity-classroom-003"})

        page_result = service.search_current_page(
            "concept mapping overlap detection reflection checklist",
            "overlap detection reflection",
        )
        self.assertTrue(page_result["all_terms_matched"])

    def test_manual_relation_override_changes_relation_state(self) -> None:
        service = KnowledgeBaseService(self.workspace)
        service.ingest_many(_sample_records(self.test_root))

        service.set_relation_state(
            activity_a="activity-classroom-002",
            activity_b="activity-classroom-003",
            action="confirm",
            strength="medium",
            reason="Manual confirmation for the same teaching thread.",
        )

        graph_view = service.export_graph_view()
        relation = next(
            item
            for item in graph_view["edges"]
            if item["relation_id"] == "activity-classroom-002__activity-classroom-003"
        )
        self.assertEqual(relation["state"], "linked")
        self.assertEqual(relation["source_type"], "user_override")

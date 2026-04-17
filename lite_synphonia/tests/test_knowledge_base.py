"""Tests for the knowledge_base subpackage.

Run from the project root directory (the directory that contains this
lite_synphonia/ package):

    python -m unittest tests.test_knowledge_base -v

Or discover all tests:

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from uuid import uuid4

from knowledge_base.service import KnowledgeBaseService


def _sample_records(base_dir: Path) -> list[dict[str, object]]:
    return [
        {
            "activity_id": "activity-classroom-001",
            "start_time": "2026-04-01T09:00:00",
            "end_time": "2026-04-01T10:15:00",
            "transcript_text": "老师说明延迟预算、增量读取和窗口大小之间的关系。",
            "summary_text": "活动重点是延迟预算、增量读取和窗口大小。",
            "keywords": ["延迟预算", "增量读取", "窗口大小"],
            "ppt_present": True,
            "ppt_file_path": str(base_dir / "classroom_week01.pptx"),
            "scene_type": "classroom",
        },
        {
            "activity_id": "activity-classroom-002",
            "start_time": "2026-04-03T09:00:00",
            "end_time": "2026-04-03T10:00:00",
            "transcript_text": "老师继续讨论窗口大小、overlap 和延迟预算。",
            "summary_text": "活动继续讲解 overlap、窗口大小和延迟预算。",
            "keywords": ["延迟预算", "窗口大小", "overlap"],
            "ppt_present": True,
            "ppt_file_path": str(base_dir / "classroom_week01.pptx"),
            "scene_type": "classroom",
        },
        {
            "activity_id": "activity-classroom-003",
            "start_time": "2026-04-10T09:00:00",
            "end_time": "2026-04-10T09:30:00",
            "transcript_text": "老师发布课程通知，同时提到总结质量和窗口大小。",
            "summary_text": "本次以课程通知为主，同时轻度回顾窗口大小和总结质量。",
            "keywords": ["课程通知", "窗口大小", "总结质量"],
            "ppt_present": True,
            "ppt_file_path": str(base_dir / "classroom_week02.pptx"),
            "scene_type": "classroom",
            "ppt_text_excerpt": "通知页和参数回顾页。",
        },
        {
            "activity_id": "activity-meeting-001",
            "start_time": "2026-04-02T14:00:00",
            "end_time": "2026-04-02T15:00:00",
            "transcript_text": "会议讨论采购预算和成本控制。",
            "summary_text": "会议讨论采购预算和成本控制。",
            "keywords": ["采购预算", "成本控制"],
            "ppt_present": False,
            "scene_type": "meeting",
        },
        {
            "activity_id": "activity-meeting-002",
            "start_time": "2026-04-09T14:00:00",
            "end_time": "2026-04-09T15:10:00",
            "transcript_text": "会议继续讨论采购预算、供应商计划和成本控制。",
            "summary_text": "活动继续沿着采购预算和成本控制主线推进。",
            "keywords": ["采购预算", "成本控制", "供应商计划"],
            "ppt_present": True,
            "ppt_file_path": str(base_dir / "meeting_review.pptx"),
            "scene_type": "meeting",
        },
    ]


class KnowledgeBaseServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.test_root = (
            Path.cwd() / ".tmp_test_runs" / "knowledge_base_tests" / uuid4().hex
        )
        self.workspace = self.test_root / "kb_workspace"
        self.assets_dir = self.test_root / "assets"
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        for file_name in ("classroom_week01.pptx", "classroom_week02.pptx", "meeting_review.pptx"):
            (self.assets_dir / file_name).write_text("placeholder", encoding="utf-8")

    def tearDown(self) -> None:
        if self.test_root.exists():
            shutil.rmtree(self.test_root)

    def test_ingestion_creates_records_and_file_views(self) -> None:
        service = KnowledgeBaseService(self.workspace)
        service.ingest_many(_sample_records(self.assets_dir))

        bundle = service.export_view_bundle(selected_activity_id="activity-classroom-001")
        self.assertEqual(len(bundle["history"]["full_record_list"]), 5)
        self.assertEqual(bundle["history"]["statistics_cards"][0]["label"], "全部记录")
        self.assertEqual(len(bundle["file_lookup"]["activity_groups"]), 5)
        self.assertEqual(bundle["detail_panel"]["activity_id"], "activity-classroom-001")

        transcript_path = (
            self.workspace
            / "records"
            / "activity-classroom-001"
            / "transcript.txt"
        )
        self.assertTrue(transcript_path.exists())

    def test_missing_optional_fields_do_not_break_ingestion(self) -> None:
        service = KnowledgeBaseService(self.workspace)
        service.ingest_completed_activity(
            {
                "activity_id": "minimal-activity",
                "start_time": "2026-04-05T10:00:00",
                "end_time": "2026-04-05T10:45:00",
                "transcript_text": "这是一次最小输入活动。",
                "summary_text": "这是一次最小输入活动的总结。",
                "keywords": ["最小输入", "活动"],
                "ppt_present": False,
            }
        )

        bundle = service.export_view_bundle(selected_activity_id="minimal-activity")
        self.assertEqual(bundle["detail_panel"]["activity_id"], "minimal-activity")
        self.assertEqual(bundle["history"]["statistics_cards"][2]["count"], 0)

    def test_relations_and_content_lines_are_built(self) -> None:
        service = KnowledgeBaseService(self.workspace)
        service.ingest_many(_sample_records(self.assets_dir))

        bundle = service.export_view_bundle()
        edges = bundle["relation_map"]["edges"]

        strong_ids = {
            item["relation_id"]
            for item in edges
            if item["strength"] == "strong" and item["state"] == "linked"
        }
        pending_ids = {
            item["relation_id"]
            for item in edges
            if item["state"] == "pending"
        }
        self.assertIn("activity-classroom-001__activity-classroom-002", strong_ids)
        self.assertIn("activity-classroom-002__activity-classroom-003", pending_ids)
        self.assertGreaterEqual(len(bundle["timeline_line_view"]["content_lines"]), 3)

    def test_search_ignores_spaces_and_uses_wide_scope(self) -> None:
        service = KnowledgeBaseService(self.workspace)
        service.ingest_many(_sample_records(self.assets_dir))

        result = service.search("采购  预算   成本控制")
        matched_ids = {item["activity_id"] for item in result["results"]}
        self.assertEqual(matched_ids, {"activity-meeting-001", "activity-meeting-002"})

        page_result = service.search_current_page("窗口 大小 与 延迟预算", "窗口大小 延迟预算")
        self.assertTrue(page_result["all_terms_matched"])

    def test_manual_relation_override_changes_relation_state(self) -> None:
        service = KnowledgeBaseService(self.workspace)
        service.ingest_many(_sample_records(self.assets_dir))

        service.set_relation_state(
            activity_a="activity-classroom-002",
            activity_b="activity-classroom-003",
            action="confirm",
            strength="medium",
            reason="用户确认属于同一主线。",
        )

        bundle = service.export_view_bundle()
        relation = next(
            item
            for item in bundle["relation_map"]["edges"]
            if item["relation_id"] == "activity-classroom-002__activity-classroom-003"
        )
        self.assertEqual(relation["state"], "linked")
        self.assertEqual(relation["source_type"], "user_override")


if __name__ == "__main__":
    unittest.main()

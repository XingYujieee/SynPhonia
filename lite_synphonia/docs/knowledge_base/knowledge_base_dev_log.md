# Knowledge Base Development Log

## Entry 001

Date: 2026-04-14
Task: Establish the standalone knowledge base module and its validation scaffolding

- What changed:
  - Defined the independent implementation area for the knowledge base under `knowledge_base/`.
  - Added interface and boundary documentation so future integration can target explicit inputs and outputs rather than historical module internals.
  - Added a frontend-free validation approach based on sample activities, exported view bundles, CLI inspection, and automated tests.
- Related modules:
  - `knowledge_base/`
  - `knowledge_base_samples/`
  - `tests/knowledge_base/`
  - `docs/knowledgeBase/`
- Unfinished tail:
  - Real integration with a future activity coordinator remains outside the current task scope.
  - No production frontend consumes the new outputs yet.
- Future optimization and technical debt:
  - If the knowledge base later grows large, dynamic scanning of per-activity files may need to be replaced by a lightweight index cache.
  - Search is intentionally predictable and lightweight in v1; semantic retrieval is reserved for a later phase.

## Entry 002

Date: 2026-04-14
Task: Implement the first runnable knowledge base core and the frontend-free validation flow

- What changed:
  - Implemented the standalone knowledge base package with unified activity input validation, per-activity storage, rule-based relation generation, content-line construction, search helpers, and view-bundle export logic.
  - Added a CLI entrypoint that supports JSON ingestion, view export, wide-scope search, and user relation overrides without requiring any frontend code.
  - Added sample activity data and placeholder local PPT assets so the module can be validated through stable, repeatable example inputs.
  - Added automated tests covering ingestion, optional-field tolerance, relation and content-line behavior, wide search, and manual relation override persistence.
  - Refined the relation rules to reduce cross-scene false positives and keep lighter continuation links in the pending-confirmation path.
- Related modules:
  - `knowledge_base/`
  - `knowledge_base_samples/`
  - `tests/knowledge_base/test_service.py`
  - `docs/knowledgeBase/README.md`
  - `docs/knowledgeBase/knowledge_base_interface.md`
  - `docs/knowledgeBase/knowledge_base_boundary.md`
  - `docs/knowledgeBase/knowledge_base_dev_log.md`
- Unfinished tail:
  - The future upper activity coordinator is still outside the current implementation scope, so real ingestion currently relies on direct service calls or CLI input files.
  - No production frontend consumes the exported view bundle yet.
- Future optimization and technical debt:
  - The current relation rules are intentionally lightweight and interpretable, but further domain-specific tuning may be needed once real activity data replaces the sample dataset.
  - PPT handling in v1 is limited to local-path inspection plus external-open compatibility; in-app PPT rendering remains intentionally out of scope.

## Entry 003

Date: 2026-04-15
Task: Integrate knowledge base into the lite_synphonia pipeline

- What changed:
  - Added `knowledge_base_workspace` field to `LiteConfig` (`config.py`) so callers can configure a KB workspace path. When the field is empty the pipeline skips KB ingestion, preserving full backward compatibility.
  - Added `--knowledge-base-workspace` CLI argument to `__main__.py`, wired to `LiteConfig.knowledge_base_workspace`.
  - Added `_knowledge_base_stage()` function in `pipeline.py` — this is the activity-coordinator handoff described in `knowledge_base_boundary.md`. The function: (a) adds `knowledge_base/` to `sys.path` at runtime so the standalone package is importable without restructuring the repository layout; (b) maps the existing `interface_output.json` schema (nested sections, float audio seconds) to the flat `ActivityIngestRecord.from_dict()` format expected by the KB service, converting elapsed-second timestamps to absolute ISO-8601 datetimes; (c) calls `KnowledgeBaseService.ingest_completed_activity()`; (d) treats any KB failure as a non-fatal warning so a KB error cannot abort the pipeline.
  - Modified `_write_consolidated()` to return `tuple[Path, dict[str, Any]]` (merged path plus the interface payload) so `run_pipeline()` can forward the payload to `_knowledge_base_stage()` without rebuilding it.
  - `_knowledge_base_stage()` is invoked as Stage 5 of `run_pipeline()` after the consolidated output is written. It is intentionally not called on quality-fail early-exit paths.
- Related modules:
  - `config.py`
  - `pipeline.py`
  - `__main__.py`
  - `knowledge_base/knowledge_base/service.py` (consumer, no changes)
- Unfinished tail:
  - The `sys.path` insertion in `_knowledge_base_stage()` is an integration convenience. A future packaging step (e.g. installing `knowledge_base` as an editable package or moving it under `lite_synphonia/`) would make the path manipulation unnecessary.
  - Knowledge base ingestion is still not triggered automatically when the pipeline is called programmatically with blocked/skipped payloads; a future coordinator layer may want finer control over when ingestion fires.
- Future optimization and technical debt:
  - `ppt_text_excerpt` from the pipeline is a list of per-slide dicts; the KB schema accepts a plain string. The bridge serialises the list as a JSON string for now. A future KB schema version could accept the structured list directly.

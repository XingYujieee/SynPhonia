# Knowledge Base Module

Documentation for the `knowledge_base` subpackage.

Current documentation set:

- `knowledge_base_task.md`: task definition and functional target
- `knowledge_base_interface.md`: input/output contracts, field ownership, and consumer views
- `knowledge_base_boundary.md`: boundary statement between the knowledge base and the other pipeline modules
- `knowledge_base_dev_log.md`: task-by-task development log

All paths below are relative to the **project root directory** (the directory
that contains `knowledge_base/`, `tests/`, `pipeline.py`, etc.).

Current implementation locations:

- code: `knowledge_base/`
- sample activity inputs: `knowledge_base/samples/`
- tests: `tests/test_knowledge_base.py`
- docs: `docs/knowledge_base/`

---

## Running the CLI

### Demo — ingest samples and export a full view bundle

```bash
python -m knowledge_base demo \
  --workspace .tmp_test_runs/kb_demo \
  --activities knowledge_base/samples/sample_activities.json \
  --output .tmp_test_runs/kb_demo/views.json \
  --reset \
  --selected-activity activity-classroom-002
```

Then inspect:

- `.tmp_test_runs/kb_demo/views.json` — the exported view bundle
- `.tmp_test_runs/kb_demo/records/` — per-activity storage files

### Other CLI commands

```bash
# Search across all stored records
python -m knowledge_base search \
  --workspace .tmp_test_runs/kb_demo \
  --query "采购预算 成本控制"

# Apply a manual relation override
python -m knowledge_base set-relation \
  --workspace .tmp_test_runs/kb_demo \
  --activity-a activity-classroom-002 \
  --activity-b activity-classroom-003 \
  --action confirm --strength medium

# Export view bundle only (workspace already populated)
python -m knowledge_base export-views \
  --workspace .tmp_test_runs/kb_demo \
  --output .tmp_test_runs/kb_demo/views.json \
  --selected-activity activity-classroom-002
```

---

## Running the tests

```bash
python -m unittest tests.test_knowledge_base -v
```

Or discover and run all project tests at once:

```bash
python -m unittest discover -s tests -v
```

---

## Pipeline integration

Pass `--knowledge-base-workspace` to the main pipeline to enable automatic
ingestion after each completed run:

```bash
python -m lite_synphonia \
    --seconds 300 \
    --transcription-provider deepgram \
    --summary-provider deepseek \
    --knowledge-base-workspace ~/my_kb_workspace
```

# Knowledge Base Boundary Statement

## 1. Boundary Goal

This statement records the agreed separation between the standalone knowledge base module and the legacy project modules already present in this repository.

## 2. Legacy Areas In The Repository

The repository already contains historical code for:

- transcript and summarization flow under `src/cncld/`
- legacy compatibility runner under `src/runner.py`
- application frontend work under `app_frontend/`

These areas are kept as historical context and reference material.

They are not the primary implementation area for the knowledge base task.

## 3. Knowledge Base Implementation Area

The standalone knowledge base implementation area is:

- `knowledge_base/`

Related supporting areas are:

- `knowledge_base_samples/`
- `tests/knowledge_base/`
- `docs/knowledgeBase/`

Knowledge base work should remain centered in these areas.

## 4. What The Knowledge Base Must Not Do

The knowledge base module must not:

- modify transcript-generation internals to pull data directly
- modify summary-generation internals to pull data directly
- embed frontend rendering logic
- assume ownership of PPT matching behavior
- assume cloud file-hosting responsibilities

## 5. How Integration Must Work

Future integration must happen through a coordination layer.

That coordination layer is expected to:

- receive the user-confirmed activity end signal
- collect the finalized transcript, summary, and PPT facts
- assemble a unified activity record
- call the knowledge base ingestion interface

The knowledge base should receive the completed record, not raw module state.

## 6. Why This Boundary Is Important

This separation ensures that:

- legacy modules remain readable as historical versions
- the knowledge base can be tested independently
- future real module assembly can target explicit interfaces instead of hidden assumptions
- frontend can later consume stable view data without reassembling low-level facts


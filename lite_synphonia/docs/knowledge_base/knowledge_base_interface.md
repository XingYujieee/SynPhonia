# Knowledge Base Interface Contract

## 1. Purpose

This document defines the current interface boundary for the standalone knowledge base module.

It focuses on:

- what the knowledge base receives
- where each input field is expected to come from
- what the knowledge base stores and derives internally
- what outputs it exposes for later consumers
- which modules are expected to call which interfaces

## 2. Module Position

The knowledge base is an independent module.

It is not the owner of:

- transcript generation
- summary generation
- PPT matching
- frontend rendering

It consumes one completed activity record from an upper coordination layer.

## 3. Ingestion Input Contract

### 3.1 Required Fields

The current required ingestion fields are:

- `activity_id`: unique identifier for one completed activity
- `start_time`: ISO datetime for activity start
- `end_time`: ISO datetime for activity end
- `transcript_text`: final transcript text for the activity
- `summary_text`: final summary text for the activity
- `keywords`: final keyword list
- `ppt_present`: whether a PPT exists for the activity
- `ppt_file_path` or `ppt_id`: required when `ppt_present=true`

### 3.2 Optional Reserved Fields

The current optional fields are:

- `transcript_meta`
- `summary_meta`
- `matched_slides`
- `ppt_text_excerpt`
- `scene_type`

These fields may be absent without breaking ingestion.

## 4. Input Ownership

The knowledge base does not fetch these values by itself.

Expected field ownership:

- transcript module:
  - `activity_id`
  - `start_time`
  - `end_time`
  - `transcript_text`
- summary module:
  - `summary_text`
  - `keywords`
  - `summary_meta` if available
- PPT matching module:
  - `ppt_present`
  - `ppt_file_path` or `ppt_id`
  - `matched_slides` if available
  - `ppt_text_excerpt` if available
- upper activity coordinator:
  - assembles the unified record
  - decides when the activity is complete
  - triggers `ingest_completed_activity(...)`

## 5. Internal Storage Behavior

On successful ingestion, the knowledge base currently persists:

- one metadata file per activity record
- one local transcript text artifact per activity
- one local summary text artifact per activity
- one relation override file for user edits

Important note:

- transcript and summary are persisted as local artifacts by the knowledge base itself
- PPT remains an external local-machine file reference
- PPT preview in v1 means local path inspection plus external-open compatibility, not in-app slide rendering

## 6. Output Contract

The knowledge base exposes view-level outputs intended for later frontend consumption.

Current output bundle includes:

- `navigation`
- `history`
- `relation_map`
- `timeline_calendar`
- `timeline_line_view`
- `file_lookup`
- `detail_panel`

### 6.1 History Output

The history output currently includes:

- four finalized statistic-entry cards
- full record list
- content-line grouped data
- attachment-record view
- pending-relation view

### 6.2 Relation Map Output

The relation map output currently includes:

- activity nodes
- activity-to-activity edges
- strength labels
- relation states
- human-readable relation reasons

### 6.3 Timeline Outputs

The timeline outputs currently include:

- chronological calendar-style data
- content-line ordered data

### 6.4 File Lookup Output

The file lookup output currently includes, grouped by activity:

- transcript text artifact path and inline preview
- summary text artifact path and inline preview
- PPT local path and external-preview metadata

### 6.5 Detail Output

The detail output currently includes:

- summary
- keywords
- relation information
- current content-line context
- file entries for the selected activity
- transcript preview

## 7. Search Contract

Current search behavior:

- multi-keyword search in a single query
- extra spaces ignored
- wide search scope across transcript, summary, keywords, file paths, and time strings
- page-local search helper for the currently visible text area

## 8. Consumers

Expected future consumers:

- application frontend
- CLI validation tools
- test suite
- upper coordination layer

Consumers should not read the storage internals directly when a service output is available.

## 9. Current Validation Entry Points

Current non-frontend validation entry points:

- `python -m knowledge_base.cli ingest-file`
- `python -m knowledge_base.cli export-views`
- `python -m knowledge_base.cli search`
- `python -m knowledge_base.cli set-relation`
- `python -m knowledge_base.cli demo`


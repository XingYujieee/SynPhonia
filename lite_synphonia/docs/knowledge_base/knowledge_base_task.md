# Knowledge Base Development Task

## 1. Document Purpose

This document defines the implementation requirements for the **Knowledge Base** module.

Its purpose is to give Codex a clear, executable description of:

- what the first version of the knowledge base should do
- what it should not do
- how it fits into the current project
- what data it receives from other modules
- what views it must provide to the frontend
- what interaction rules it must support
- what boundaries must remain clean for future maintenance and extension

This document is written for implementation, not for discussion.

---

## 2. Product Context

The current project centers around one complete activity flow:

1. a user starts one activity
2. the system generates transcript text for that activity
3. the system generates summary content for that activity
4. optional PPT-related information may also be available
5. after the activity ends, the system should automatically preserve the activity as a knowledge base record

The knowledge base is **not** a general-purpose chatbot or a large external document platform.

The first version is an **activity-based knowledge base**.

Its job is to help the user:

- review past activities
- understand how multiple activities connect to each other
- see main content lines across time
- preview files related to each activity
- search across their accumulated records

The application is **not only for learning scenarios**. It may also be used in **offline meeting scenarios**.

Therefore:

- names must remain generic and user-friendly
- the design must not assume only classroom usage
- the module must remain usable for both study and meeting records

---

## 3. Core Positioning of the First Version

The first version of the knowledge base has two parallel goals:

1. **Activity accumulation and review**
2. **Activity relation display**

However, implementation priority should be:

1. reliably preserve each activity record
2. reliably display historical records
3. then build and present activity relations on top of that

The first version should be:

- lightweight
- fast
- clear in data flow
- easy to maintain
- easy to extend later

The first version should **not** depend on a heavy reasoning chain or complex graph inference pipeline.

---

## 4. Architectural Principles

### 4.1 Basic Unit

The first version uses **activity** as the basic unit.

That means:

- one completed activity becomes one knowledge base record
- knowledge graph nodes in v1 are activity nodes
- UI does not yet show theme nodes
- theme-node support may be reserved internally for future extension

Do **not** make “theme” the primary visible unit in v1.

### 4.2 Lightweight Application Layer

Because the project will later use a remote LLM vendor API, the application layer should remain lightweight and efficient.

Therefore the first version must **not** use a mandatory “rule prefilter + model review” pipeline in its main flow.

Instead, the first version must use:

**rule-based relation judgment + user fallback editing**

This means:

- the system uses lightweight rules to compute candidate relations between activities
- highly related records can be auto-linked
- uncertain relations should be placed into “pending confirmation”
- users must be able to edit relations manually in the relation map

The first version should avoid adding another mandatory LLM call into the activity-end storage flow.

### 4.3 Clean Module Boundaries

The knowledge base must remain separated from:

- transcript generation internals
- summarization internals
- PPT matching internals
- frontend rendering internals

The knowledge base should consume one unified activity record, not pull internal state from multiple modules on its own.

---

## 5. What the First Version Must Do

The first version must support the following capabilities.

### 5.1 Automatically ingest one completed activity

After an activity ends, the system must create one knowledge base record for it.

This ingestion should happen automatically.

The user should not need to manually save the activity into the knowledge base.

### 5.2 Preserve historical records

The system must preserve enough information so that users can later:

- browse all activity records
- search them
- open one activity and inspect its content
- see activity time information
- preview its related files

### 5.3 Build activity-to-activity relations

The system must build basic relations between activities using lightweight rules.

This is not a deep semantic reasoning system.

The output should be sufficient for:

- relation map display
- content-line display
- pending-confirmation display

### 5.4 Support user editing of relations

The user must be able to edit activity relations from the relation map view.

The first version must support at least:

- confirm a suggested relation
- mark a relation as pending
- remove or cancel a relation
- modify relation choices in a basic way through the relation editing UI

### 5.5 Support record search

The knowledge base must support:

- multi-keyword search in a single query
- in-page text search on the currently open view
- ignoring user-entered spaces during search

### 5.6 Support file preview by activity

The knowledge base must support file preview and file path display.

It must not implement download handling.

Files should be organized by activity.

---

## 6. What the First Version Must Not Do

The first version must not implement the following:

1. general-purpose QA assistant behavior
2. complex semantic graph reasoning
3. theme nodes shown directly in the UI
4. model-review relation judgment as a required main-path step
5. file hosting or file download service
6. incremental storage during an ongoing activity
7. full database-backed file management
8. heavy semantic retrieval infrastructure

Also, the first version must not assume that PPT files are hosted by the application.

PPT files originate from the user’s local machine.
Transcript and summary text are also stored locally.

The application must not be designed as a cloud file storage system.

---

## 7. Frontend Top-Level Navigation

The knowledge base frontend must provide the following top-level navigation items.

These names are already finalized and should be used directly.

1. **历史记录**
2. **关联地图**
3. **记录时间线**
4. **文件查找**

These labels must remain user-friendly and generic enough for both study and meeting scenarios.

### 7.1 历史记录

This page is for reviewing all accumulated activity records.

This is the main entry point for users who want to look back at past content.

### 7.2 关联地图

This page shows activity relations visually.

It should present an activity-based knowledge graph / relation map.

This page must support relation editing.

### 7.3 记录时间线

This page shows records ordered in time.

It must support two viewing modes:

- by chronological calendar view
- by content-line timeline view

### 7.4 文件查找

This page is for finding files related to each activity.

It must provide:

- file preview
- local file path display
- activity-based file organization

It must **not** provide file download functionality.

---

## 8. History Page Functional Definition

The history page must include four statistic-entry cards.

These cards are already finalized and should be used directly.

1. **全部记录**
2. **内容主线**
3. **带附件的记录**
4. **待确认关联**

These are not just passive statistics. They are clickable entry points into filtered or grouped views.

### 8.1 全部记录

This card shows the total number of stored activity records.

When clicked, it should display the full activity list view.

The full list view should:

- show every stored activity record
- show each record’s basic summary information
- allow clicking one record to open its details

### 8.2 内容主线

This card shows the number of formed content lines.

When clicked, it should display a main-line view.

The main-line view should:

- group activities by content line
- show each group as a line-like sequence
- order activities within the line by time
- allow clicking an activity to inspect details

Important:

A content line is not a complex dependency graph.

In v1, a content line means:

> a set of activities judged to belong to the same continuing content thread, ordered by time

### 8.3 带附件的记录

This card shows the number of records with PPT or file-related attachment information.

When clicked, it should display the records that contain attachment/file information.

This view should:

- show which activities contain files
- organize them by activity
- allow the user to preview files
- allow the user to see local file paths

It must not provide download actions.

### 8.4 待确认关联

This card shows the number of relations that require manual confirmation.

When clicked, it should display the records or relation entries that are still pending user confirmation.

This view should:

- show the activity pair or activity relation candidate
- show why the relation is pending
- allow the user to edit or confirm it

---

## 9. Relation Map Functional Definition

The relation map is the main visual relation view.

The v1 relation map should behave like an activity relation graph.

### 9.1 Node Type

Nodes in v1 must be **activity nodes**.

Do not show theme nodes in the first version UI.

### 9.2 Edge Meaning

Edges represent relations between activities.

The system should support at least three qualitative strengths:

- strong relation
- medium relation
- weak relation

An uncertain relation may be shown as pending confirmation.

### 9.3 Editing Behavior

Users must be able to interact with the relation map.

The first version must support relation editing from this page.

At minimum the user should be able to:

- confirm a suggested relation
- mark a relation as pending
- remove a relation
- inspect why the relation exists

### 9.4 Interaction Expectations

Clicking an activity node should:

- highlight that activity as the current selection
- update the detail panel
- update the related file preview and main-line context

### 9.5 Internal Design Constraint

Even though theme nodes are not shown in the UI, the internal structure should allow future extension toward:

- activity nodes
- theme nodes
- activity-theme edges

However, Codex must not implement the visible theme graph in v1.

---

## 10. Timeline Functional Definition

The timeline page must support **two modes**.

### 10.1 Chronological Time View

This is a calendar-style view.

Its purpose is to show **when activities happened**.

Each activity should display at least:

- date
- start time
- end time

Example style:

- one date cell or grouped date area
- entries such as `14:00 - 15:15`

This mode is meant to answer:

- what happened on which day
- when an activity started and ended

### 10.2 Content-Line Time View

This is a line / axis style view.

Its purpose is to show the order of activities within the same content line.

Each content line should:

- contain activities judged to belong to the same thread
- display those activities in chronological order
- show only basic information for each activity

This mode is meant to answer:

- how one thread evolved across multiple activities

### 10.3 Chain Connection Rule

The chain connection rule is already finalized.

A content line is formed by:

1. grouping activities into the same line
2. sorting activities in that line by start time

This must be used consistently in the frontend and data output.

---

## 11. File Lookup Functional Definition

The file page is already defined more narrowly.

### 11.1 No Download Support

The application must not provide file download functionality.

Reason:

- PPT is originally uploaded from the user’s own machine
- transcript and summary text are stored locally
- the project does not host PPT in a backend storage system

### 11.2 What It Must Provide

The file page must provide:

- file preview
- local file path display
- grouping by activity

### 11.3 File Types in Scope

The first version must support preview grouping for:

- PPT
- transcript text
- summary text

### 11.4 UI Expectation

Users should be able to:

- browse files under one activity
- open a preview of the file content or a preview summary
- see where the file is located on the local machine

The frontend should treat this as a “find and inspect” view, not a storage management panel.

---

## 12. Search Functional Definition

The first version knowledge base must support the following search behavior.

### 12.1 Multi-keyword Search

A single search input must support multiple keywords in one query.

The search behavior should treat the query as containing several search terms.

### 12.2 In-page Search

The system must support searching within the text of the currently opened page or current visible content area.

This should help users quickly locate content inside a detailed view.

### 12.3 Ignore User-entered Spaces

Search must ignore user-entered spaces.

This means:

- unnecessary spaces in input should not prevent matches
- matching logic should normalize spaces before comparison

The first version does not need complex semantic search.

It should focus on predictable, usable search behavior.

---

## 13. Data Flow and Ingestion

### 13.1 Ingestion Trigger

Knowledge base ingestion must happen **after each activity ends**.

Do not implement:

- incremental ingestion during the activity
- user-triggered “save to knowledge base” as the default path

### 13.2 Unified Activity Record

The knowledge base must not pull data directly from transcript internals, summary internals, or PPT internals.

Instead, one upper coordination layer should assemble a unified activity record and pass it to the knowledge base.

This keeps module boundaries clean.

### 13.3 Why This Boundary Matters

This design prevents:

- knowledge base depending on multiple internal module states
- frontend being forced to assemble low-level data itself
- tight coupling between storage logic and pipeline internals

Codex should preserve this boundary.

---

## 14. Input Data Requirements for V1

### 14.1 Required Fields in the Current Phase

The following fields are mandatory in the first version.

#### From Transcript Module

- `activity_id`
- `start_time`
- `end_time`
- `transcript_text`

#### From Summary Module

- `summary_text`
- `keywords`

#### From PPT Real-time Matching Module

- `ppt_present`
- `ppt_file_path` or `ppt_id` when a PPT exists

These required fields must be enough for the first version to run.

### 14.2 Optional Reserved Fields

The following are reserved enhancement fields.

They must be considered optional in the current phase.

#### Transcript Module

- `transcript_meta`

#### Summary Module

- `summary_meta`

#### PPT Module

- `matched_slides`
- `ppt_text_excerpt`

These fields:

- may be included in the schema
- must not be required in the current version
- must not block ingestion when absent

The knowledge base and frontend must function correctly when these fields are missing.

---

## 15. Module Responsibility Split

This section is important for team coordination.

### 15.1 Transcript Module Responsibility

The transcript module is responsible for transcript facts only.

It must provide:

- activity identity
- activity start and end time
- final transcript text

It is not responsible for:

- relation judgment
- knowledge graph generation
- storage presentation logic

### 15.2 Summary Module Responsibility

The summary module is responsible for final summary facts only.

It must provide:

- final summary text
- final keyword list

It is not responsible for:

- knowledge base storage
- relation map logic
- frontend grouping logic

### 15.3 PPT Real-time Matching Module Responsibility

The PPT module is responsible for whether a PPT exists and where it is located.

In the current phase it must provide:

- whether a PPT is present
- PPT file path or ID when available

It is not responsible for:

- graph relation logic
- content-line logic
- file hosting

### 15.4 Frontend Responsibility

The frontend should consume the knowledge base’s unified output.

The frontend should not directly assemble knowledge base data from all low-level modules.

Frontend responsibilities include:

- display historical records
- display relation map
- display timeline views
- display file preview/path views
- support search and relation editing interactions

### 15.5 Knowledge Base Module Responsibility

The knowledge base module is responsible for:

- storing activity records
- generating lightweight indexes
- computing relation candidates using rules
- exposing data for frontend views
- supporting user relation edits

---

## 16. Relation Judgment Requirements

This section defines the v1 relation logic.

### 16.1 Chosen Strategy

The first version must use:

**rule-based relation judgment + user fallback editing**

Do not implement model review as part of the mandatory main path.

### 16.2 Rule Judgment Output

The rule system should produce relation candidates categorized into at least:

- strong
- medium
- weak
- pending confirmation where appropriate

### 16.3 Expected System Behavior

- strong relation: may be auto-linked
- medium / uncertain relation: should appear in pending confirmation
- weak relation: may remain visible but should not dominate primary views
- user must be able to edit relation outcomes

### 16.4 Implementation Constraint

Keep the rule logic interpretable and lightweight.

Do not build a heavy reasoning pipeline here.

The rule set should remain simple enough that future developers can understand:

- why two activities were related
- why a relation became pending
- why one activity entered a content line

---

## 17. Scene Compatibility

The project is not limited to study scenarios.

It may also be used for offline meetings.

Therefore the implementation should:

- remain generic in labels and data handling
- avoid classroom-only assumptions in the UI
- reserve the possibility of a `scene_type` or equivalent field internally

The first version does not need to visibly separate scene types in the UI.

But the structure should be compatible with that future extension.

---

## 18. Interaction Requirements Summary

The first version must support these key interactions.

### 18.1 Record Selection

Users must be able to click a record and inspect:

- summary
- keywords
- relation information
- timeline context
- file preview/path information

### 18.2 Relation Map Interaction

Users must be able to:

- click nodes
- inspect relation context
- edit relations from the map context

### 18.3 Timeline Switching

Users must be able to switch between:

- chronological calendar view
- content-line time view

### 18.4 File Inspection

Users must be able to:

- inspect files grouped by activity
- preview files
- read local file paths

### 18.5 Search Interaction

Users must be able to:

- search with multiple keywords
- search the current page content
- search without being affected by extra spaces in input

---

## 19. Suggested Outputs From the Knowledge Base Layer

Codex should design the knowledge base output so the frontend can consume at least these view-level data structures:

1. full record list data
2. content-line grouped data
3. pending relation list data
4. file-by-activity view data
5. relation graph data
6. timeline calendar data
7. timeline line-view data
8. detail panel data for a selected activity

These do not need to be separate APIs in the first implementation, but the structure should clearly support these uses.

---

## 20. Implementation Guidance for Codex

Codex should implement this module in a way that keeps the system maintainable.

### 20.1 Keep the Knowledge Base Separate

Do not mix knowledge base logic directly into transcript or summary pipeline internals.

### 20.2 Keep the Main Path Lightweight

The first version should not perform extra mandatory LLM calls during activity-end ingestion.

### 20.3 Prefer Clear Data Structures

Prefer explicit, readable, debuggable structures over abstract frameworks.

### 20.4 Make Optional Fields Truly Optional

The first version must not fail because enhancement fields are missing.

### 20.5 Prioritize View Usability

The first version should first make the user able to:

- find records
- understand record relations
- inspect files
- understand the timeline

before trying to implement more advanced intelligence.

---

## 21. Definition of Done

The knowledge base v1 can be considered implemented only if all of the following are true.

1. Each completed activity is automatically ingested after the activity ends.
2. Historical records can be displayed reliably.
3. The history page supports the four finalized statistic-entry cards.
4. The relation map displays activity-based relations.
5. The timeline page supports both chronological and content-line views.
6. The file page supports preview and local path display without downloads.
7. Search supports multi-keyword input, in-page search, and ignores spaces.
8. Relation candidates are produced through lightweight rule-based logic.
9. Users can edit relations in the relation map.
10. The knowledge base can run using only the required v1 fields.
11. Missing optional fields do not break the system.
12. Module boundaries remain clean and maintainable.

---

## 22. Final Reminder

This document describes the **first version** of the knowledge base.

It should be treated as a focused implementation target, not as a long-term all-in-one design.

The correct implementation strategy is:

- make activity storage reliable
- make historical review usable
- make relations editable and understandable
- keep the system lightweight
- leave clear extension points for future theme nodes and model-enhanced reasoning


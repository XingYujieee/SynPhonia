# LiteSynphonia

A zero-local-model lecture and meeting processing pipeline.  All AI operations — transcription, summarisation, and embedding — are performed via remote API calls.  No PyTorch, no Whisper binary, and no local model files are required.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [API Key Configuration](#api-key-configuration)
4. [Function Reference](#function-reference)
   - [1. Transcription](#1-transcription)
   - [2. Summarisation](#2-summarisation)
   - [3. PDF Matching](#3-pdf-matching)
   - [4. Output](#4-output)
5. [Running the Pipeline](#running-the-pipeline)
6. [Full CLI Reference](#full-cli-reference)
7. [Output Interface Specification](#output-interface-specification)

---

## Overview

```
Microphone → Audio Enhancement → Deepgram STT ─────────────┐
                                                            ↓
                                             Transcript Quality Gate
                                                            ↓
                                        DeepSeek / LLM API (Summary)
                                                            ↓
                             PDF Slides → Embedding API (BGE-M3) → Page Matching
                                                            ↓
                                        merged_results.json + interface_output.json
```

Each stage is independently configurable.  Different providers can be used for each stage (e.g. Deepgram for transcription, DeepSeek for summarisation, SiliconFlow/BGE-M3 for embeddings).

---

## Prerequisites

- Python 3.10 or later
- The `merge_syn` package installed (provides shared audio, quality, and bridge utilities)
- API keys for your chosen providers (see below)

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## API Key Configuration

All providers are managed through a shared registry stored at:

```
~/.config/lite_synphonia/providers.json
```

Use the `providers` sub-command to register each API key.  You only need to do this once per provider.

### Register a Provider

```bash
python3 -m lite_synphonia providers add <name> \
    --base-url  <endpoint-url> \
    --model     <model-id> \
    --api-key   <your-api-key> \
    --service   <transcription|summarization|embedding>
```

### Recommended Provider Setup

#### Transcription — Deepgram

```bash
python3 -m lite_synphonia providers add deepgram \
    --base-url  https://api.deepgram.com \
    --model     whisper-large \
    --api-key   YOUR_DEEPGRAM_API_KEY \
    --service   transcription
```

Models available on Deepgram: `whisper-large` (recommended for Chinese), `nova-2-general`, `nova-2`.
`whisper-large` is more robust at lower audio signal levels; `nova-2-general` is faster but requires cleaner audio (RMS ≥ 0.015).

#### Summarisation — DeepSeek

```bash
python3 -m lite_synphonia providers add deepseek \
    --base-url  https://api.deepseek.com/v1 \
    --model     deepseek-chat \
    --api-key   YOUR_DEEPSEEK_API_KEY \
    --service   summarization
```

Any OpenAI-compatible LLM endpoint can be used here (OpenAI, MiniMax, Qwen, etc.).

#### Embedding — SiliconFlow / BGE

```bash
python3 -m lite_synphonia providers add siliconflow-embed \
    --base-url  https://api.siliconflow.cn/v1 \
    --model     BAAI/bge-large-zh-v1.5 \
    --api-key   YOUR_SILICONFLOW_API_KEY \
    --service   embedding
```

BGE-family models (e.g. `BAAI/bge-large-zh-v1.5`, `BAAI/bge-m3`) support Chinese, English, and mixed-language content and do not require input prefixes.  For E5-family models, pass `--embedding-passage-prefix "passage: "` and `--embedding-query-prefix "query: "` at runtime (see CLI reference).

### List Registered Providers

```bash
python3 -m lite_synphonia providers list
```

---

## Function Reference

### 1. Transcription

**Purpose:** Capture microphone audio, enhance it, and convert it to a timestamped transcript using Deepgram's Speech-to-Text API.

#### Audio Enhancement Pipeline

Raw microphone samples pass through the following chain before being sent to Deepgram:

| Step | Operation | Default | Effect |
|---|---|---|---|
| 1 | DC offset removal | always on | Removes hardware bias that AGC would amplify |
| 2 | Input gain | `1.0` | Fixed scalar pre-multiplier |
| 3 | Noise gate | threshold `0.003` | Zeroes chunks whose RMS is below the silence floor, preventing background noise from being amplified and hallucinated by Deepgram |
| 4 | Bidirectional AGC | target `0.03`, max `2.0×` | Attenuates loud signals AND amplifies quiet ones toward the target RMS; the previous one-directional AGC only amplified, causing loud audio to clip |
| 5 | Pre-emphasis | `α = 0.97` | High-pass filter `y[n] = x[n] − 0.97·x[n−1]`; boosts consonants (zh/ch/sh/s) that ASR models rely on; standard in Kaldi/ESPnet |
| 6 | Soft-knee limiter | ceiling `0.72` | Applies a Hermite curve in the 85–100% zone before the ceiling instead of hard clipping; eliminates inter-modulation distortion that degrades confidence |
| 7 | Hard clip | `[-1, 1]` | Safety net only |

Pass `--no-agc` to bypass steps 2–6 entirely when your hardware already provides a clean signal.

#### Deepgram Transcription

The enhanced audio is encoded as 16-bit WAV at 16 kHz and sent to Deepgram's `/v1/listen` endpoint with the following parameters:

```
punctuate=true   smart_format=true   utterances=true
utt_split=1.2    numerals=true       filler_words=false
```

`utt_split=1.2` uses a 1.2-second pause as the utterance boundary, which matches the natural thinking pauses in lecture speech.  `filler_words=false` removes disfluencies (uh, um, 那个, 就是) from the transcript.

#### Fallback Chain

If the primary model returns no segments, the pipeline automatically retries in order:

1. Enhanced audio, primary model, specified language
2. Raw audio (pre-enhancement), primary model, specified language
3. Enhanced audio, primary model, language auto-detect
4. Enhanced audio, `whisper-large`, language auto-detect (only if primary is not whisper-large)

The model that actually produced the transcript is recorded in `transcription.json` under `runtime.selected_model`.

#### Quality Gate

After transcription, a quality assessment runs:

- `decision: pass` — mean confidence ≥ threshold and content present → pipeline continues
- `decision: warn` — confidence below the merge_syn hard floor but content passes a diversity check → downstream stages run with a warning flag
- `decision: fail` — no usable content → downstream stages are blocked (use `--allow-low-quality-transcript` to override)

The quality confidence threshold defaults to `0.15` and is tunable with `--quality-confidence-threshold`.

#### Output

```
<output-dir>/transcription/
    transcription.json          Full payload (metrics, segments, quality, preflight)
    raw_audio.wav               Pre-enhancement recording
    enhanced_audio.wav          Post-enhancement recording
    deepgram_response_raw.json  Present only when all attempts return empty results
```

---

### 2. Summarisation

**Purpose:** Condense the transcript into a concise summary using an OpenAI-compatible LLM API.

#### Windowing

For long transcripts that exceed the LLM's effective context window, the transcript is processed in overlapping windows:

| Parameter | Default | Description |
|---|---|---|
| `--summary-window-size` | 200 | Words per window |
| `--summary-overlap-size` | 60 | Overlap words between consecutive windows |
| `--summary-chunk-size` | 1200 | Characters read per incremental pass |
| `--summary-max-new-tokens` | 384 | Maximum tokens the LLM may generate per call |

When the full transcript fits within a single window (≤ 200 words), the LLM processes it in one pass with no boundary stitching.  This is the ideal case and the most common for lecture segments of 5–10 minutes.

#### Providers

Any OpenAI-compatible endpoint works.  Register it with `--service summarization`.  The model must support chat-completion (`/v1/chat/completions`).

Note: DeepSeek and other providers that offer only chat completions cannot be used for embedding.  Keep the summarisation and embedding providers separate.

#### Output

```
<output-dir>/
    input/transcript.txt        Flat transcript fed to the LLM
    summary/results.json        Summary rounds, text, and runtime metadata
```

---

### 3. PDF Matching

**Purpose:** Given a PDF slide deck, assign each transcript segment to the slide page the speaker was most likely discussing at that moment.

#### Architecture

Matching is performed in two phases:

**Phase A — PDF Preprocessing (once per unique PDF)**

1. Extract text from each page and split into overlapping chunks (`pdf_reader`)
2. Check the on-disk embedding cache (key: `sha256(pdf_bytes)[:16]_<model>.npz`)
3. If cache miss: embed all chunks in a single batch API call; write cache
4. If cache hit: load embeddings directly — no API call

**Phase B — Query Matching (per run)**

1. Collect all unique segment texts from the transcript
2. Embed them in a single batch API call (all segments at once — previously one call per segment)
3. The matching loop uses only local numpy dot products — no further API calls
4. A Viterbi smoother enforces global monotonic ordering (lecture slides only advance)

This architecture reduces API calls during matching from O(N segments) to O(1) regardless of transcript length.

#### Embedding Prefix Configuration

| Model family | passage prefix | query prefix |
|---|---|---|
| BGE-M3, OpenAI | `""` (default) | `""` (default) |
| multilingual-E5 | `"passage: "` | `"query: "` |

Set prefixes via `--embedding-passage-prefix` and `--embedding-query-prefix`.

#### PDF Embedding Cache

By default, chunk embeddings are cached to `<output-dir>/.pdf_embed_cache/`.  The cache key encodes both the PDF content (SHA-256) and the model name, so it invalidates automatically when either changes.  On a cache hit, the PDF embedding stage is skipped entirely.

Disable the cache:
```bash
--pdf-cache-dir none
```

Use a shared cache directory across multiple runs:
```bash
--pdf-cache-dir ~/.cache/lite_synphonia/pdf_embeddings
```

#### Output

```
<output-dir>/pdf_match/results.json
```

The `timeline` array contains page-level time ranges:

```json
[
  { "page_index": 0, "start_time": 0.0,  "end_time": 43.2, "confidence": 0.81 },
  { "page_index": 2, "start_time": 43.2, "end_time": 90.0, "confidence": 0.76 }
]
```

The `segment_matches` array gives per-segment assignments with raw page scores, smoothed page, and the reason for any state-machine decision (e.g. `page_switch_confirmed`, `stay_current_page`, `global_monotonic_smoothing`).

---

### 4. Output

**Purpose:** Consolidate all stage results into a standardised output that downstream systems (knowledge base, front-end) can consume without parsing internal pipeline structures.

#### Files Written

| File | Description |
|---|---|
| `merged_results.json` | Full internal payload: all stage payloads, metrics, quality data, stage statuses |
| `interface_output.json` | Standardised cross-module contract (schema version 1.0) |
| `transcription/transcription.json` | Transcription stage detail |
| `summary/results.json` | Summarisation stage detail |
| `pdf_match/results.json` | PDF matching stage detail |

#### `interface_output.json` — Schema Version 1.0

This file is the stable interface consumed by the knowledge-base module and front-end.  All downstream consumers must handle missing optional fields gracefully.

```json
{
  "schema_version": "1.0",
  "activity_id":    "a3f9c1d2e4b5f6a7b8c9d0e1f2a3b4c5",
  "created_at_utc": "2026-04-15T12:34:56.789+00:00",

  "transcription": {
    "start_time":      0.0,
    "end_time":        300.0,
    "transcript_text": "深度学习是机器学习的重要分支 ...",

    "transcript_meta": {
      "segment_count":     30,
      "mean_confidence":   0.7286,
      "recorded_seconds":  300.0,
      "language":          "zh"
    }
  },

  "summary": {
    "summary_text": "本次讲座介绍了深度学习的基础理论 ...",
    "keywords":     ["深度学习", "神经网络", "卷积神经网络", "机器学习", "反向传播"],

    "summary_meta": {
      "rounds":    1,
      "provider":  "deepseek"
    }
  },

  "ppt": {
    "ppt_present":    true,
    "ppt_file_path":  "/path/to/slides.pdf",
    "ppt_id":         "",

    "matched_slides": [
      { "slide_index": 0, "start_time": 0.0,   "end_time": 43.2, "confidence": 0.81 },
      { "slide_index": 2, "start_time": 43.2,  "end_time": 90.0, "confidence": 0.76 }
    ],

    "ppt_text_excerpt": [
      { "slide_index": 0, "text_preview": "深度学习入门..." },
      { "slide_index": 2, "text_preview": "卷积神经网络..." }
    ]
  }
}
```

#### Required vs Optional Fields

| Field | Required | Present when |
|---|---|---|
| `schema_version` | ✓ | Always |
| `activity_id` | ✓ | Always (auto-generated UUID4 if not supplied) |
| `created_at_utc` | ✓ | Always |
| `transcription.start_time` | ✓ | Always |
| `transcription.end_time` | ✓ | Always |
| `transcription.transcript_text` | ✓ | Always (empty string if no content) |
| `summary.summary_text` | ✓ | Always (empty string if skipped) |
| `summary.keywords` | ✓ | Always (empty list if skipped) |
| `ppt.ppt_present` | ✓ | Always |
| `ppt.ppt_file_path` | ✓ | Always (empty string if no PDF) |
| `ppt.ppt_id` | ✓ | Always (reserved for asset management, currently empty) |
| `transcription.transcript_meta` | optional | When transcription produced content |
| `summary.summary_meta` | optional | When summarisation ran |
| `ppt.matched_slides` | optional | When PDF matching ran and produced a timeline |
| `ppt.ppt_text_excerpt` | optional | When PDF matching ran and page text is available |

**Consumer contract:** The knowledge-base module and front-end must function correctly when any optional field is absent.  Do not assume optional fields are present.

---

## Running the Pipeline

### Minimal Run (transcript + summary only)

```bash
python3 -m lite_synphonia \
    --seconds 300 \
    --transcription-provider deepgram \
    --transcription-model    whisper-large \
    --summary-provider       deepseek \
    --output-dir             ./output
```

### Full Run (transcript + summary + PDF matching)

```bash
python3 -m lite_synphonia \
    --seconds              300 \
    --transcription-provider deepgram \
    --transcription-model    whisper-large \
    --summary-provider       deepseek \
    --embedding-provider     siliconflow-embed \
    --embedding-model        BAAI/bge-large-zh-v1.5 \
    --pdf-path               slides.pdf \
    --output-dir             ./output \
    --activity-id            lecture-2026-04-15-001
```

### Full Run with Knowledge Base Ingestion (recommended final validation)

```bash
python3 -m lite_synphonia \
    --seconds                  300 \
    --transcription-provider   deepgram \
    --summary-provider         deepseek \
    --embedding-provider       siliconflow-embed \
    --embedding-model          BAAI/bge-large-zh-v1.5 \
    --embedding-format         openai \
    --pdf-path                 ./test_source/test.pdf \
    --activity-id              lecture-2026-04-15-001 \
    --knowledge-base-workspace ./.tmp_test_runs/kb_workspace \
    --output-dir               ./.tmp_test_runs/pipeline_run_final
```

### With Preflight Microphone Check

```bash
python3 -m lite_synphonia \
    --preflight-seconds 3 \
    --seconds           300 \
    --transcription-provider deepgram \
    --summary-provider       deepseek \
    --output-dir             ./output
```

The preflight records 3 seconds before the main recording begins.  It detects hardware ADC saturation (peak ≥ 0.95) and low signal (RMS < 2× silence floor) against the raw (pre-enhancement) signal, so the check correctly reflects hardware conditions rather than the software limiter level.

### Using a Domain Glossary

Create a plain-text file with one term per line:

```text
深度学习
卷积神经网络
反向传播算法
梯度下降
```

Then pass it to bias Deepgram recognition:

```bash
python3 -m lite_synphonia \
    --glossary-file  glossary.txt \
    --seconds        300 \
    --transcription-provider deepgram \
    --summary-provider deepseek \
    --output-dir ./output
```

---

## Full CLI Reference

### Recording

| Flag | Default | Description |
|---|---|---|
| `--seconds` | `8.0` | Recording duration in seconds |
| `--language` | `zh` | Whisper-style language code (`zh`, `en`, `ja`, `auto`, …) |
| `--skip-mic` | off | Use a synthetic 440 Hz tone instead of microphone (testing) |
| `--preflight-seconds` | `0.0` | Microphone preflight duration; `0` disables |

### Audio Enhancement

| Flag | Default | Description |
|---|---|---|
| `--input-gain` | `1.0` | Fixed scalar pre-multiplier before AGC |
| `--target-rms` | `0.03` | AGC target RMS; raise to `0.06` only for very quiet microphones |
| `--max-gain` | `2.0` | Maximum AGC amplification factor |
| `--limiter-level` | `0.72` | Soft-knee limiter ceiling (0–1) |
| `--no-agc` | off | Bypass all enhancement; send raw samples to Deepgram |
| `--pre-emphasis` | `0.97` | Pre-emphasis coefficient α; `0.0` disables |

### Transcription

| Flag | Default | Description |
|---|---|---|
| `--transcription-provider` | `deepgram` | Provider name from registry |
| `--transcription-model` | `whisper-large` | Deepgram model (`whisper-large`, `nova-2-general`, …) |
| `--transcription-language` | *(auto from `--language`)* | BCP-47 override, e.g. `zh-CN`; `auto` disables forced language |
| `--initial-prompt` | `""` | Hint text to bias Deepgram recognition |
| `--glossary-file` | `""` | Path to plain-text glossary file (one term per line) |
| `--allow-low-quality-transcript` | off | Force downstream stages even when quality gate fails |
| `--quality-confidence-threshold` | `0.15` | Minimum confidence for content-based quality override |

### Summarisation

| Flag | Default | Description |
|---|---|---|
| `--summary-provider` | `minimax` | Provider name from registry |
| `--summary-window-size` | `200` | Words per summarisation window |
| `--summary-overlap-size` | `60` | Overlap words between windows |
| `--summary-chunk-size` | `1200` | Characters per incremental read |
| `--summary-max-new-tokens` | `384` | Maximum tokens generated per LLM call |

### PDF Matching

| Flag | Default | Description |
|---|---|---|
| `--pdf-path` | `""` | Path to slide PDF; omit to skip matching |
| `--embedding-provider` | `minimax-embed` | Provider name from registry |
| `--embedding-model` | `embo-01` | Embedding model ID |
| `--embedding-batch-size` | `32` | Texts per API batch call |
| `--embedding-format` | `auto` | Wire format: `auto` (detect from URL), `minimax`, or `openai` |
| `--embedding-passage-prefix` | `""` | Prefix for PDF chunk texts (empty for BGE-M3; `"passage: "` for E5) |
| `--embedding-query-prefix` | `""` | Prefix for query texts (empty for BGE-M3; `"query: "` for E5) |
| `--pdf-cache-dir` | *(output-dir/.pdf_embed_cache)* | Cache directory for chunk embeddings; `none` to disable |

### Output

| Flag | Default | Description |
|---|---|---|
| `--output-dir` | `lite_synphonia_output/` | Root output directory |
| `--activity-id` | *(auto UUID4)* | Session identifier written to `interface_output.json` |

### Knowledge Base

| Flag | Default | Description |
|---|---|---|
| `--knowledge-base-workspace` | `""` | When provided, ingests `interface_output.json` into the workspace after the pipeline finishes |

---

## Supported Embedding Providers

| Provider | Base URL | Model | Format |
|---|---|---|---|
| SiliconFlow (BGE) | `https://api.siliconflow.cn/v1` | `BAAI/bge-large-zh-v1.5` (or `BAAI/bge-m3`) | `openai` (auto) |
| MiniMax | `https://api.minimaxi.com/v1` | `embo-01` | `minimax` (auto) |
| OpenAI | `https://api.openai.com/v1` | `text-embedding-3-small` | `openai` (auto) |

The `--embedding-format auto` default detects MiniMax from the URL and uses its native `texts`/`vectors` schema.  All other URLs default to the OpenAI `input`/`data[].embedding` schema.

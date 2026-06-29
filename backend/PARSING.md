# Multimodal document parsing

## Overview

Upload flow:

1. API saves file → `documents.status=queued` + `parse_jobs.status=pending`
2. **Parse worker** claims jobs from PostgreSQL (`FOR UPDATE SKIP LOCKED`)
3. Routes by file type → MinerU cloud / Python / ASR / VLM / ffmpeg
4. Table-aware chunking → Elasticsearch index

Frontend polls document list; upload returns immediately.

## Parse job lease and generation

Long-running parses use a **lease + generation** model to avoid races when jobs are reclaimed, reparsed, or run on multiple workers.

| Concept | Purpose |
|---------|---------|
| `active_parse_generation` (document) | Bumps on reparse/stale reclaim; invalidates in-flight work |
| `lease_token` + `lease_expires_at` (job) | Worker must renew heartbeat to stay owner |
| `active_job_id` (document) | Points to the single legitimate running job |

**Claim**: sets `running`, issues lease, skips pending jobs if another live `running` exists for the same document.

**Stale reclaim**: cancels job, bumps generation, deletes ES vectors with `parse_generation < active`, optionally enqueues a new `pending` job (`PARSE_JOB_STALE_AUTO_RETRY=true`).

**Pipeline**: checks ownership at stage boundaries; only the owner may write to Elasticsearch (`parse_generation` + `job_id` in chunk metadata).

**Deploy upgrade**: schema changes are applied automatically via Alembic on API/worker startup (`upgrade head`). Set `AUTO_DB_MIGRATE=false` to disable.

If upgrading from a database that already has the latest schema (e.g. previously used `create_all` or manual SQL), run once:

```bash
cd backend && alembic stamp head
```


## Document user isolation

Each document belongs to the uploading user (`documents.user_id`). Non-admin users can only list, access, and retrieve their own documents; admins see all.

| Layer | Behavior |
|-------|----------|
| API / DB | `DocumentService` filters by `user_id`; admin skips filter |
| Elasticsearch | Chunk metadata includes `user_id`; retrieval applies ES filter for non-admins |
| Chat | `RAGService` sets per-request retrieval context from JWT user |

**Deploy upgrade** (existing DB with legacy documents):

1. If schema is already current, run `alembic stamp head` (see above).
2. **Reparse recommended** for documents indexed before `user_id` migration so ES chunks get `metadata.user_id` (otherwise non-admins cannot retrieve legacy chunks).



| Variable | Default | Purpose |
|----------|---------|---------|
| `PARSE_JOB_LEASE_TTL_SEC` | `120` | Lease duration |
| `PARSE_JOB_HEARTBEAT_SEC` | `30` | Worker heartbeat interval |
| `PARSE_JOB_STALE_GRACE_SEC` | `60` | Extra wait after lease expiry before reclaim |
| `PARSE_JOB_STALE_AUTO_RETRY` | `true` | Enqueue new pending job after stale reclaim |

## Start worker

```bash
cd backend
python -m app.worker.parse_worker
```

### Docker

Full stack (from repo root):

```bash
cp backend/.env.docker.example backend/.env
docker compose up -d --build
```

Backend only (API + worker + postgres + elasticsearch + mailpit):

```bash
cd backend
cp .env.docker.example .env
docker compose up -d --build
```

Parse worker image includes **ffmpeg** for video frame extraction and audio splitting. Uploads are shared between `api` and `parse-worker` via the `uploads` volume.

| Container | Role |
|-----------|------|
| `api` | FastAPI + Alembic auto-migrate on start |
| `parse-worker` | Document parsing pipeline + ffmpeg |
| `postgres` | Metadata + LangGraph checkpoint |
| `elasticsearch` | Vector / BM25 index |
| `mailpit` | Dev SMTP capture (`SMTP_HOST=mailpit`, UI :8025) |

Not containerized (configure via `.env`): MinerU cloud, LLM/Embedding/ASR/VLM APIs, OpenViking (in-process SDK + volume).

## Configuration

Requires **Python 3.12** for local dev and parse worker (`backend/.python-version`).

See `.env.example` and [`chunking_config.yaml`](chunking_config.yaml):

| Variable | Purpose |
|----------|---------|
| `MINERU_API_TOKEN` | MinerU cloud Bearer token ([mineru.net](https://mineru.net/apiManage/docs)) |
| `CHUNKING_CONFIG_PATH` | YAML file for per-content-type chunk strategies |
| `INDEX_DELETE_RETRY_ATTEMPTS` | Retries when deleting old ES index batches |
| `VIDEO_FRAME_MODE` | `scene` (keyframe) or `interval` |
| `VIDEO_SCENE_THRESHOLD` | ffmpeg scene detection threshold (default 0.3) |
| `VIDEO_FRAME_BUDGET` | Max frames after dedupe (default 96) |
| `VIDEO_MIN_INTERVAL_SEC` | Uniform coverage minimum interval (default 60s) |
| `VIDEO_VLM_ENABLED` | Video frame VLM summary (default **false**) |
| `RETRIEVAL_ASR_SEGMENT_EXPAND_ENABLED` | Expand hits to same ASR segment chunks |
| `PARSE_JOB_STALE_TIMEOUT_SEC` | Reclaim stale running jobs (default 7200) |

## File routing

| Types | Parser |
|-------|--------|
| PDF, DOCX/PPTX/XLSX, MinerU images | MinerU cloud; Python fallback on failure |
| `.doc` / `.xls` / `.ppt` | **Python only** (MinerU unsupported): `office-oxide` primary; fallback `xlrd` (`.xls`) or LibreOffice (`.doc`/`.ppt`) |
| TXT, MD | Python loaders only |
| Audio | ASR |
| Video | ffmpeg audio → ASR; scene keyframes → MinerU OCR + VLM |
| Image | MinerU OCR + VLM summary |

## Office page numbers

Each indexed chunk from Python Office loaders may include:

| Format | `page_number` meaning |
|--------|----------------------|
| DOCX | Word page (page-break detection via python-docx) |
| PPT/PPTX | Slide number |
| XLS/XLSX | Sheet index |
| Legacy DOC/XLS/PPT | Sheet/slide when IR available; otherwise `1` |

**MinerU path** reads `content_list.json` from the result zip when available; each page becomes a Document with `page_number = page_idx + 1`. Falls back to whole-document `full.md` without page metadata when JSON is missing.

Chunking propagates `page_number`, `sheet_name`, and `slide_index` into every derived chunk.

Retrieval can expand coarse hits to all chunks on the same `(document_id, page_number)` before rerank (`RETRIEVAL_PAGE_EXPAND_ENABLED`).

## Table chunking

Strategies are configurable in `chunking_config.yaml` (defaults):

- **Row split + header repeat** for Markdown/HTML tables
- **Caption absorption**: short paragraph before a table becomes `table_caption`
- **Complex HTML**: tables with `rowspan`/`colspan` stay atomic (not row-split)
- **No-separator Markdown tables**: first row treated as header

Industry practice ([Twig](https://www.twig.so/dev/rag-scenarios-and-solutions/chunking/table-splitting), [AI/TLDR](https://ai-tldr.dev/learn/rag/chunking-and-ingestion/chunk-code-tables-markdown/)): structure-first splitting, never break mid-row, repeat headers in continuations.

## ASR segments

Audio transcription prefers `verbose_json` with `start_sec` / `end_sec`. Long segments are sub-split while preserving time metadata.

### Proactive splitting

When `ASR_PROACTIVE_SPLIT_ENABLED=true` (default), audio longer than `ASR_PROACTIVE_SPLIT_MIN_DURATION_SEC` (default 600s) or larger than `ASR_PROACTIVE_MAX_FILE_MB` (default 25MB) is **split with ffmpeg before** calling the ASR API. Each part uses `verbose_json`; segment timestamps are merged with the part offset.

| Variable | Default | Purpose |
|----------|---------|---------|
| `ASR_PROACTIVE_SPLIT_ENABLED` | `true` | Enable proactive ffmpeg split |
| `ASR_PROACTIVE_SPLIT_MIN_DURATION_SEC` | `600` | Split when duration exceeds this |
| `ASR_PROACTIVE_SEGMENT_SEC` | `600` | Part length for proactive split |
| `ASR_PROACTIVE_MAX_FILE_MB` | `25` | Split when file size exceeds this |
| `ASR_FALLBACK_SEGMENT_SEC` | `120` | Part length for failure fallback (plain text) |

If proactive split returns no segments, the pipeline falls back to whole-file `verbose_json` → plain text → 120s fallback split.

## Video processing

Flow: **extract audio → ASR (full length) → frame planner → OCR per frame**.

Frame planner combines three sources:

1. **Uniform anchors** — spread across full duration (`VIDEO_MIN_INTERVAL_SEC`)
2. **ASR anchors** — one frame near each speech segment midpoint (`VIDEO_ASR_MERGE_GAP_SEC` merges nearby segments)
3. **Scene gap fill** — ffmpeg scene detect in long gaps without frames (`VIDEO_SCENE_GAP_SEC`)

Frames are extracted with batched ffmpeg: timestamps within `VIDEO_EXTRACT_BATCH_WINDOW_SEC` (default 120s) share one `-ss` seek and a `select` filter; batches run in parallel up to `VIDEO_FRAME_CONCURRENCY`. Scene gap probes use the same concurrency. phash dedupe removes duplicate slides. Default **`VIDEO_VLM_ENABLED=false`**: only MinerU OCR runs on frames; when enabled, VLM runs only if OCR text is below `VIDEO_VLM_MIN_OCR_CHARS`.

| Variable | Default | Purpose |
|----------|---------|---------|
| `VIDEO_EXTRACT_BATCH_WINDOW_SEC` | `120` | Max span grouped into one ffmpeg decode |
| `VIDEO_EXTRACT_SELECT_MARGIN_SEC` | `0.5` | Half-width of each `between(t,...)` window |
| `VIDEO_EXTRACT_FFMPEG_THREADS` | `1` | Per-process ffmpeg thread cap when running batches in parallel |

Frame metadata includes `timestamp_sec`, `frame_source`, and when aligned: `asr_segment_index`, `asr_start_sec`, `asr_end_sec`. Retrieval can expand coarse hits to all chunks in the same ASR segment (`RETRIEVAL_ASR_SEGMENT_EXPAND_ENABLED`).

Legacy `VIDEO_FRAME_MODE` / `VIDEO_MAX_FRAMES` are kept for compatibility but the planner path uses `VIDEO_FRAME_BUDGET`.

### Citation timestamps and playback

Indexed chunks carry time metadata (`timestamp_sec` for frames, `start_sec` / `end_sec` for ASR). Chat `sources` and `GET /api/documents/{id}/chunks/{index}` expose these fields plus `file_type` and `content_type`.

Stream the original media for in-app seek:

```
GET /api/documents/{doc_id}/file
```

Supports `Authorization: Bearer` or `?access_token=` (for `<video src>`). Returns `FileResponse` with HTTP Range support. Only `video` and `audio` documents are allowed.

The chat UI shows a video player in the citation drawer when `file_type=video`, lists all cited time points from the same video in the current answer, and seeks on click.

## Manual re-parse

`POST /api/documents/{doc_id}/reparse` clears ES vectors, resets document status, and enqueues a new job. **Running jobs are marked `cancelled`** so in-flight workers skip completion.

## Reliability

- Write-then-delete indexing with `index_batch` and retry on old-batch cleanup
- Stale job reclaim, atomic failure updates, delete/reparse race handling

## Dependencies

- **ffmpeg**, **httpx**, **filetype**, **Pillow**, **ImageHash**
- **openpyxl**, **python-pptx**, **python-docx**, **office-oxide** (Python 3.12) for Office parsing
- **xlrd** optional fallback for `.xls` when office-oxide fails
- **LibreOffice** (`soffice`) fallback for legacy `.doc`/`.ppt` when office-oxide fails
- **PyYAML** for chunking config

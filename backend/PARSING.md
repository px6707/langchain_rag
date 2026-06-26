# Multimodal document parsing

## Overview

Upload flow:

1. API saves file → `documents.status=queued` + `parse_jobs.status=pending`
2. **Parse worker** claims jobs from PostgreSQL (`FOR UPDATE SKIP LOCKED`)
3. Routes by file type → MinerU cloud / Python / ASR / VLM / ffmpeg
4. Table-aware chunking → Elasticsearch index

Frontend polls document list; upload returns immediately.

## Start worker

```bash
cd backend
python -m app.worker.parse_worker
```

Docker Compose (includes ffmpeg):

```bash
docker compose up parse-worker
```

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
| `VIDEO_MAX_FRAMES` | Max frames per video |
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

## Video keyframes

Default `VIDEO_FRAME_MODE=scene` uses ffmpeg `select=gt(scene,THRESH)` to extract keyframes with real `pts_time` timestamps. Falls back to fixed-interval mode if no frames are detected. phash dedupe and concurrent OCR+VLM processing still apply.

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

# Hybrid retrieval + cloud rerank

## Overview

Document RAG uses a two-stage pipeline:

1. **Coarse retrieval (Elasticsearch)**: vector + BM25 hybrid with RRF fusion (`langchain-elasticsearch` `DenseVectorStrategy`)
2. **Rerank (cloud API)**: Jina/Cohere/vLLM-compatible `POST /v1/rerank` via LangChain `ContextualCompressionRetriever`

`RETRIEVAL_SCORE_THRESHOLD` applies to **rerank relevance scores** when rerank is enabled.

## Configuration

```env
RETRIEVAL_HYBRID_ENABLED=true
RETRIEVAL_RRF_ENABLED=true
RETRIEVAL_FETCH_K=20
RETRIEVAL_SCORE_THRESHOLD=0.7

RERANK_ENABLED=true
RERANK_API_BASE=https://api.example.com/v1
RERANK_API_KEY=sk-xxx
RERANK_MODEL=your-rerank-model
RERANK_TOP_N=4
```

If `RERANK_API_BASE` / `RERANK_API_KEY` are empty, they fall back to `EMBEDDING_API_BASE` / `EMBEDDING_API_KEY`.

Set `RERANK_ENABLED=false` to skip rerank (uses `RERANK_TOP_N` as ES `k`).

## Index migration

Hybrid search requires BM25 on the `text` field. If you enabled hybrid on an older pure-vector index:

1. Delete the ES index (e.g. `rag_documents`)
2. Re-upload documents via the UI/API

If hybrid search fails at runtime, the service **falls back to vector-only** search and logs the error.

## Citation format `[document_id#chunk_index]`

Each indexed chunk stores `document_id` and `chunk_index` in ES metadata. Retrieved context is injected as:

```text
[550e8400-e29b-41d4-a716-446655440000#2] report.pdf
chunk text...
```

The chat model is instructed to cite facts using the same `[document_id#chunk_index]` markers. The frontend renders these as clickable links that open the chunk in a drawer (`GET /api/documents/{doc_id}/chunks/{chunk_index}`).

**Legacy indexes** without `chunk_index` need re-upload after enabling this feature.

## Grounding validation

After each answer (when retrieval ran), the small LLM judges whether factual claims are supported by retrieved chunks. Results are sent as SSE `grounding` events and shown as badges in the UI (answer text is not modified).

```env
GROUNDING_ENABLED=true
GROUNDING_MIN_SUPPORTED_RATIO=0.8
GROUNDING_FAIL_RATIO=0.5
GROUNDING_MAX_CLAIMS=8
```

Status mapping:

- `supported_ratio >= GROUNDING_MIN_SUPPORTED_RATIO` → supported (green)
- `>= GROUNDING_FAIL_RATIO` → partial (orange)
- otherwise → not_supported (red)

Uses the shared small LLM (`SMALL_LLM_*` or fallback to main `LLM_*`). See also retrieval routing and query rewrite, which use the same `get_small_llm()`.

## API format

Rerank endpoint expects:

```http
POST {base}/v1/rerank
Authorization: Bearer {api_key}
Content-Type: application/json

{
  "model": "your-rerank-model",
  "query": "user question",
  "documents": ["chunk text", "..."],
  "top_n": 4
}
```

Response (Jina/Cohere style):

```json
{
  "results": [
    { "index": 0, "relevance_score": 0.92 }
  ]
}
```

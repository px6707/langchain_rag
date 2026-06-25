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

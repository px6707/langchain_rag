import logging
from copy import deepcopy
from typing import Any, Sequence

import requests
from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document
from pydantic import ConfigDict, PrivateAttr

from app.config import settings

logger = logging.getLogger(__name__)


def _resolve_rerank_api_base() -> str:
    return (settings.rerank_api_base or settings.embedding_api_base).rstrip("/")


def _resolve_rerank_api_key() -> str:
    return settings.rerank_api_key or settings.embedding_api_key


def _build_rerank_url(api_base: str) -> str:
    if api_base.endswith("/v1"):
        return f"{api_base}/rerank"
    return f"{api_base}/v1/rerank"


class HttpRerankCompressor(BaseDocumentCompressor):
    """Rerank documents via Jina/Cohere/vLLM-compatible POST /rerank API."""

    top_n: int = 4
    model: str
    api_base: str
    api_key: str

    _session: requests.Session = PrivateAttr()

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    def model_post_init(self, __context: Any) -> None:
        session = requests.Session()
        session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
        )
        self._session = session

    def _call_rerank_api(
        self,
        query: str,
        documents: Sequence[Document],
    ) -> list[dict[str, Any]]:
        if not documents:
            return []

        payload = {
            "model": self.model,
            "query": query,
            "documents": [doc.page_content for doc in documents],
            "top_n": self.top_n,
        }
        url = _build_rerank_url(self.api_base)
        response = self._session.post(url, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()

        if "results" in data:
            return data["results"]

        # Cohere v2 style: { "data": [ { "index", "relevance_score" } ] }
        if "data" in data:
            return data["data"]

        detail = data.get("detail") or data.get("message") or data
        raise RuntimeError(f"Unexpected rerank response: {detail}")

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:
        results = self._call_rerank_api(query, documents)
        compressed: list[Document] = []
        for item in results:
            index = item.get("index")
            if index is None or index >= len(documents):
                continue
            score = item.get("relevance_score")
            if score is None:
                score = item.get("score")
            doc = documents[index]
            doc_copy = Document(doc.page_content, metadata=deepcopy(doc.metadata))
            if score is not None:
                doc_copy.metadata["rerank_score"] = float(score)
            compressed.append(doc_copy)
        return compressed


def get_rerank_compressor() -> HttpRerankCompressor | None:
    if not settings.rerank_enabled:
        return None
    if not settings.rerank_model:
        logger.warning("RERANK_ENABLED=true but RERANK_MODEL is empty; skipping rerank")
        return None

    api_key = _resolve_rerank_api_key()
    if not api_key:
        logger.warning("Rerank API key missing; skipping rerank")
        return None

    return HttpRerankCompressor(
        model=settings.rerank_model,
        api_base=_resolve_rerank_api_base(),
        api_key=api_key,
        top_n=settings.rerank_top_n,
    )

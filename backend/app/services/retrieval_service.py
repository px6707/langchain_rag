import logging

from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_core.documents import Document

from app.config import settings
from app.schemas import SourceInfo
from app.services.rerank_service import get_rerank_compressor
from app.services.vector_store_service import get_vector_store

logger = logging.getLogger(__name__)


def _doc_score(doc: Document) -> float | None:
    score = doc.metadata.get("rerank_score")
    if score is None:
        score = doc.metadata.get("relevance_score")
    if score is None:
        return None
    try:
        return float(score)
    except (TypeError, ValueError):
        return None


def _build_sources_and_context(docs: list[Document]) -> tuple[list[SourceInfo], str | None]:
    if not docs:
        return [], None

    sources: list[SourceInfo] = []
    context_parts: list[str] = []
    for doc in docs:
        filename = doc.metadata.get("filename", "unknown")
        score = _doc_score(doc)
        sources.append(
            SourceInfo(
                filename=filename,
                content=doc.page_content[:200],
                score=score,
            )
        )
        score_text = f" | 相关度: {score:.3f}" if score is not None else ""
        context_parts.append(f"[来源: {filename}{score_text}]\n{doc.page_content}")

    context = "\n\n---\n\n".join(context_parts)
    return sources, context


def _filter_by_threshold(docs: list[Document]) -> list[Document]:
    if not docs:
        return []

    scored = [(doc, _doc_score(doc)) for doc in docs]
    if any(score is not None for _, score in scored):
        return [
            doc
            for doc, score in scored
            if score is not None and score >= settings.retrieval_score_threshold
        ]
    return docs


def _invoke_retriever(query: str, *, use_hybrid: bool) -> list[Document]:
    store = get_vector_store(use_hybrid=use_hybrid)
    fetch_k = settings.retrieval_fetch_k if settings.rerank_enabled else settings.rerank_top_n
    base_retriever = store.as_retriever(search_kwargs={"k": fetch_k})

    compressor = get_rerank_compressor()
    if compressor:
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever,
        )
    else:
        retriever = base_retriever

    return list(retriever.invoke(query))


def search_relevant_docs(query: str) -> tuple[list[SourceInfo], str | None]:
    if not query.strip():
        return [], None

    docs: list[Document] = []
    if settings.retrieval_hybrid_enabled:
        try:
            docs = _invoke_retriever(query, use_hybrid=True)
        except Exception:
            logger.exception("Hybrid retrieval failed; falling back to vector-only search")
            docs = _invoke_retriever(query, use_hybrid=False)
    else:
        docs = _invoke_retriever(query, use_hybrid=False)

    docs = _filter_by_threshold(docs)
    return _build_sources_and_context(docs)

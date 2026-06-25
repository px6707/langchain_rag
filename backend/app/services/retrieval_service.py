import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_core.documents import Document

from app.config import settings
from app.schemas import SourceInfo
from app.schemas.retrieval import RetrievalPlan
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


def _fetch_k() -> int:
    return settings.retrieval_fetch_k if settings.rerank_enabled else settings.rerank_top_n


def _retrieve_raw(query: str, *, use_hybrid: bool) -> list[Document]:
    store = get_vector_store(use_hybrid=use_hybrid)
    base_retriever = store.as_retriever(search_kwargs={"k": _fetch_k()})
    return list(base_retriever.invoke(query))


def _apply_rerank(query: str, docs: list[Document]) -> list[Document]:
    if not docs:
        return docs
    compressor = get_rerank_compressor()
    if compressor:
        return list(compressor.compress_documents(docs, query))
    return docs


def _invoke_retriever(query: str, *, use_hybrid: bool) -> list[Document]:
    docs = _retrieve_raw(query, use_hybrid=use_hybrid)
    return _apply_rerank(query, docs)


def _dedupe_documents(docs: list[Document]) -> list[Document]:
    best: dict[tuple[str, str], Document] = {}
    for doc in docs:
        filename = str(doc.metadata.get("filename", ""))
        key = (doc.page_content, filename)
        score = _doc_score(doc)
        existing = best.get(key)
        if existing is None:
            best[key] = doc
            continue
        existing_score = _doc_score(existing)
        if score is not None and (existing_score is None or score > existing_score):
            best[key] = doc
    return list(best.values())


def _retrieve_with_hybrid_fallback(query: str, *, use_hybrid: bool) -> list[Document]:
    if use_hybrid:
        try:
            return _retrieve_raw(query, use_hybrid=True)
        except Exception:
            logger.exception("Hybrid retrieval failed; falling back to vector-only search")
            return _retrieve_raw(query, use_hybrid=False)
    return _retrieve_raw(query, use_hybrid=False)


def _retrieve_multi_queries(queries: list[str], *, use_hybrid: bool) -> list[Document]:
    unique_queries = list(dict.fromkeys(q.strip() for q in queries if q.strip()))
    if not unique_queries:
        return []

    if len(unique_queries) == 1:
        return _retrieve_with_hybrid_fallback(unique_queries[0], use_hybrid=use_hybrid)

    all_docs: list[Document] = []
    with ThreadPoolExecutor(max_workers=min(len(unique_queries), 4)) as executor:
        futures = {
            executor.submit(_retrieve_with_hybrid_fallback, query, use_hybrid=use_hybrid): query
            for query in unique_queries
        }
        for future in as_completed(futures):
            all_docs.extend(future.result())

    return _dedupe_documents(all_docs)


def _retrieve_hyde(query: str, hyde_doc: str, *, use_hybrid: bool) -> list[Document]:
    fetch_k = _fetch_k()
    vector_store = get_vector_store(use_hybrid=False)
    hyde_docs = vector_store.similarity_search(hyde_doc, k=fetch_k)

    if use_hybrid:
        query_docs = _retrieve_with_hybrid_fallback(query, use_hybrid=True)
        return _dedupe_documents(hyde_docs + query_docs)

    return hyde_docs


def _execute_retrieval(plan: RetrievalPlan, *, use_hybrid: bool) -> list[Document]:
    query = plan.standalone_query.strip()
    if not query:
        return []

    if plan.strategy == "none":
        return _retrieve_with_hybrid_fallback(query, use_hybrid=use_hybrid)

    if plan.strategy == "multi_query":
        queries = [query, *plan.extra_queries]
        docs = _retrieve_multi_queries(queries, use_hybrid=use_hybrid)
        return _apply_rerank(query, docs)

    if plan.strategy == "hyde" and plan.hyde_document:
        docs = _retrieve_hyde(query, plan.hyde_document, use_hybrid=use_hybrid)
        return _apply_rerank(query, docs)

    if plan.strategy == "decompose":
        sub_queries = plan.extra_queries or [query]
        docs = _retrieve_multi_queries(sub_queries, use_hybrid=use_hybrid)
        return _apply_rerank(query, docs)

    return _retrieve_with_hybrid_fallback(query, use_hybrid=use_hybrid)


def search_with_plan(plan: RetrievalPlan) -> tuple[list[SourceInfo], str | None]:
    if plan.action == "skip" or not plan.standalone_query.strip():
        return [], None

    use_hybrid = settings.retrieval_hybrid_enabled
    docs = _execute_retrieval(plan, use_hybrid=use_hybrid)
    docs = _filter_by_threshold(docs)
    return _build_sources_and_context(docs)


def search_relevant_docs(query: str) -> tuple[list[SourceInfo], str | None]:
    if not query.strip():
        return [], None

    plan = RetrievalPlan(
        action="retrieve",
        strategy="none",
        standalone_query=query,
        reason="direct query",
    )
    return search_with_plan(plan)

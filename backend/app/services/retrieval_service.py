import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field

from app.config import settings
from app.schemas import SourceInfo
from app.schemas.retrieval import RetrievalPlan
from app.services.llm_service import get_router_llm
from app.services.rerank_service import get_rerank_compressor
from app.services.retrieval_fusion import rrf_fuse
from app.services.retrieval_validator import normalize_plan
from app.services.vector_store_service import get_vector_store

logger = logging.getLogger(__name__)


class FallbackQueries(BaseModel):
    queries: list[str] = Field(default_factory=list)


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


def _per_query_k(n_lists: int) -> int:
    return max(settings.retrieval_per_query_k_min, _fetch_k() // max(n_lists, 1))


def _retrieve_raw(query: str, *, use_hybrid: bool, k: int | None = None) -> list[Document]:
    fetch_k = k if k is not None else _fetch_k()
    store = get_vector_store(use_hybrid=use_hybrid)
    base_retriever = store.as_retriever(search_kwargs={"k": fetch_k})
    return list(base_retriever.invoke(query))


def _retrieve_with_hybrid_fallback(query: str, *, use_hybrid: bool, k: int | None = None) -> list[Document]:
    if use_hybrid:
        try:
            return _retrieve_raw(query, use_hybrid=True, k=k)
        except Exception:
            logger.exception("Hybrid retrieval failed; falling back to vector-only search")
            return _retrieve_raw(query, use_hybrid=False, k=k)
    return _retrieve_raw(query, use_hybrid=False, k=k)


def _apply_rerank(query: str, docs: list[Document]) -> list[Document]:
    if not docs:
        return docs
    compressor = get_rerank_compressor()
    if compressor:
        return list(compressor.compress_documents(docs, query))
    return docs


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


def _unique_queries(queries: list[str]) -> list[str]:
    return list(dict.fromkeys(q.strip() for q in queries if q.strip()))


def _collect_query_lists(queries: list[str], *, use_hybrid: bool) -> list[list[Document]]:
    unique = _unique_queries(queries)
    if not unique:
        return []

    per_k = _per_query_k(len(unique))
    if len(unique) == 1:
        return [_retrieve_with_hybrid_fallback(unique[0], use_hybrid=use_hybrid, k=per_k)]

    lists: list[list[Document]] = [[] for _ in unique]
    with ThreadPoolExecutor(max_workers=min(len(unique), 4)) as executor:
        future_to_idx = {
            executor.submit(_retrieve_with_hybrid_fallback, q, use_hybrid=use_hybrid, k=per_k): idx
            for idx, q in enumerate(unique)
        }
        for future in as_completed(future_to_idx):
            lists[future_to_idx[future]] = future.result()
    return lists


def _collect_hyde_lists(query: str, hyde_doc: str, *, use_hybrid: bool) -> list[list[Document]]:
    per_k = _per_query_k(3 if use_hybrid else 2)
    vector_store = get_vector_store(use_hybrid=False)
    lists: list[list[Document]] = [
        vector_store.similarity_search(hyde_doc, k=per_k),
        vector_store.similarity_search(query, k=per_k),
    ]
    if use_hybrid:
        lists.append(_retrieve_with_hybrid_fallback(query, use_hybrid=True, k=per_k))
    return lists


def _collect_lists(plan: RetrievalPlan, *, use_hybrid: bool) -> tuple[list[list[Document]], list[float]]:
    query = plan.standalone_query.strip()
    if not query:
        return [], []

    if plan.strategy == "hyde" and plan.hyde_document:
        lists = _collect_hyde_lists(query, plan.hyde_document, use_hybrid=use_hybrid)
        weights = [1.0, settings.retrieval_rrf_parent_weight, settings.retrieval_rrf_parent_weight]
        if not use_hybrid:
            weights = weights[:2]
        return lists, weights

    if plan.strategy in ("multi_query", "decompose"):
        queries = _unique_queries([query, *plan.extra_queries])
        lists = _collect_query_lists(queries, use_hybrid=use_hybrid)
        weights = [
            settings.retrieval_rrf_parent_weight if idx == 0 else 1.0 for idx in range(len(lists))
        ]
        return lists, weights

    lists = [_retrieve_with_hybrid_fallback(query, use_hybrid=use_hybrid, k=_fetch_k())]
    return lists, [settings.retrieval_rrf_parent_weight]


def _run_unified_pipeline(plan: RetrievalPlan, *, use_hybrid: bool) -> list[Document]:
    query = plan.standalone_query.strip()
    if not query:
        return []

    lists, weights = _collect_lists(plan, use_hybrid=use_hybrid)
    if not lists:
        return []

    if len(lists) == 1:
        fused = lists[0]
    else:
        fused = rrf_fuse(lists, list_weights=weights)

    docs = _dedupe_documents(fused)
    return _apply_rerank(query, docs)


def _fallback_extra_queries(plan: RetrievalPlan, *, llm: BaseChatModel | None = None) -> list[str]:
    model = llm or get_router_llm()
    structured = model.with_structured_output(FallbackQueries)
    prompt = (
        f"为以下检索问题生成 2 条不同表述的同义 query，用于扩大召回。\n"
        f"问题：{plan.standalone_query}"
    )
    try:
        raw = structured.invoke(
            [
                {"role": "system", "content": "输出 JSON，queries 为 2 条中文检索 query。"},
                {"role": "user", "content": prompt},
            ]
        )
        result = FallbackQueries.model_validate(raw)
        extras = [q.strip() for q in result.queries if q.strip()]
        if extras:
            return extras[:2]
    except Exception:
        logger.exception("Failed to generate fallback extra queries")

    return [plan.standalone_query]


def search_with_plan(
    plan: RetrievalPlan,
    *,
    llm: BaseChatModel | None = None,
) -> tuple[list[SourceInfo], str | None]:
    if plan.action == "skip" or not plan.standalone_query.strip():
        return [], None

    plan = normalize_plan(plan)
    use_hybrid = settings.retrieval_hybrid_enabled
    docs = _run_unified_pipeline(plan, use_hybrid=use_hybrid)
    docs = _filter_by_threshold(docs)

    if (
        not docs
        and plan.strategy == "none"
        and settings.retrieval_empty_fallback_enabled
    ):
        logger.info("Empty retrieval with none; upgrading to multi_query")
        upgraded = plan.model_copy(
            update={
                "strategy": "multi_query",
                "extra_queries": _fallback_extra_queries(plan, llm=llm),
                "reason": f"{plan.reason}; empty fallback",
            }
        )
        docs = _run_unified_pipeline(upgraded, use_hybrid=use_hybrid)
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

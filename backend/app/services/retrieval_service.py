import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field

from app.config import settings
from app.observability.retrieval_trace import RetrievalTrace
from app.observability.stage_trace import trace_stage
from app.parsing.time_metadata import extract_time_metadata
from app.schemas import SourceInfo
from app.schemas.retrieval import RetrievalPlan
from app.services.llm_service import get_small_llm
from app.services.rerank_service import get_rerank_compressor
from app.services.retrieval_fusion import rrf_fuse
from app.services.retrieval_validator import normalize_plan
from app.services.vector_store_service import (
    bm25_search,
    fetch_chunks_by_asr_segment,
    fetch_chunks_by_page,
    filter_documents_by_user,
    get_vector_store,
    user_es_term,
)
from app.services.retrieval_context import get_retrieval_user_filter

logger = logging.getLogger(__name__)


class FallbackQueries(BaseModel):
    queries: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class _TierConfig:
    plan: RetrievalPlan
    threshold: float
    use_hybrid: bool
    k_multiplier: float


@dataclass
class _PipelineStats:
    queries: list[str]
    hits_before_rerank: int
    hits_after_rerank: int
    page_expand_added: int
    asr_expand_added: int


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


def _doc_ref_id(doc: Document) -> tuple[str, int, str]:
    document_id = str(doc.metadata.get("document_id", "unknown"))
    chunk_index = doc.metadata.get("chunk_index")
    if chunk_index is None:
        chunk_index = 0
    else:
        chunk_index = int(chunk_index)
    ref_id = f"{document_id}#{chunk_index}"
    return document_id, chunk_index, ref_id


def _build_sources_and_context(docs: list[Document]) -> tuple[list[SourceInfo], str | None]:
    if not docs:
        return [], None

    sources: list[SourceInfo] = []
    context_parts: list[str] = []
    for doc in docs:
        filename = str(doc.metadata.get("filename", "unknown"))
        document_id, chunk_index, ref_id = _doc_ref_id(doc)
        score = _doc_score(doc)
        sources.append(
            SourceInfo(
                document_id=document_id,
                chunk_index=chunk_index,
                ref_id=ref_id,
                filename=filename,
                content=doc.page_content[:200],
                score=score,
                **extract_time_metadata(doc.metadata),
            )
        )
        score_text = f" | 相关度: {score:.3f}" if score is not None else ""
        context_parts.append(f"[{ref_id}] {filename}{score_text}\n{doc.page_content}")

    context = "\n\n---\n\n".join(context_parts)
    return sources, context


def _filter_by_threshold(docs: list[Document], threshold: float | None = None) -> list[Document]:
    if not docs:
        return []

    cutoff = settings.retrieval_score_threshold if threshold is None else threshold
    scored = [(doc, _doc_score(doc)) for doc in docs]
    if any(score is not None for _, score in scored):
        return [
            doc
            for doc, score in scored
            if score is not None and score >= cutoff
        ]
    return docs


def _fetch_k(*, k_multiplier: float = 1.0) -> int:
    base = settings.retrieval_fetch_k if settings.rerank_enabled else settings.rerank_top_n
    return max(settings.retrieval_per_query_k_min, int(base * k_multiplier))


def _per_query_k(n_lists: int, *, k_multiplier: float = 1.0) -> int:
    return max(settings.retrieval_per_query_k_min, _fetch_k(k_multiplier=k_multiplier) // max(n_lists, 1))


def _retrieve_raw(
    query: str,
    *,
    use_hybrid: bool,
    k: int | None = None,
) -> list[Document]:
    fetch_k = k if k is not None else _fetch_k()
    store = get_vector_store(use_hybrid=use_hybrid)
    search_kwargs: dict = {"k": fetch_k}
    user_id = get_retrieval_user_filter()
    if user_id is not None:
        search_kwargs["filter"] = user_es_term(user_id)
    base_retriever = store.as_retriever(search_kwargs=search_kwargs)
    docs = list(base_retriever.invoke(query))
    return filter_documents_by_user(docs)


def _retrieve_with_hybrid_fallback(
    query: str,
    *,
    use_hybrid: bool,
    k: int | None = None,
) -> list[Document]:
    if use_hybrid:
        try:
            return _retrieve_raw(query, use_hybrid=True, k=k)
        except Exception:
            logger.exception("Hybrid retrieval failed; falling back to vector-only search")
            return _retrieve_raw(query, use_hybrid=False, k=k)
    return _retrieve_raw(query, use_hybrid=False, k=k)


@trace_stage("rag_rerank")
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


def _page_number_from_metadata(metadata: dict) -> int | None:
    value = metadata.get("page_number")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _chunk_dedupe_key(doc: Document) -> tuple[str, int, int]:
    document_id = str(doc.metadata.get("document_id", ""))
    chunk_index = doc.metadata.get("chunk_index")
    page_number = _page_number_from_metadata(doc.metadata)
    return (document_id, page_number if page_number is not None else -1, int(chunk_index or 0))


@trace_stage("rag_page_expand")
def _expand_same_page_chunks(docs: list[Document]) -> list[Document]:
    if not docs or not settings.retrieval_page_expand_enabled:
        return docs

    page_scores: dict[tuple[str, int], float] = {}
    for doc in docs:
        page_number = _page_number_from_metadata(doc.metadata)
        document_id = str(doc.metadata.get("document_id", ""))
        if page_number is None or not document_id:
            continue
        score = _doc_score(doc)
        if score is None:
            continue
        key = (document_id, page_number)
        existing = page_scores.get(key)
        if existing is None or score > existing:
            page_scores[key] = score

    if not page_scores:
        return docs

    merged: dict[tuple[str, int, int], Document] = {}
    for doc in docs:
        merged[_chunk_dedupe_key(doc)] = doc

    for (document_id, page_number), trigger_score in page_scores.items():
        siblings = fetch_chunks_by_page(
            document_id,
            page_number,
            max_chunks=settings.retrieval_page_expand_max_chunks,
        )
        for sibling in siblings:
            key = _chunk_dedupe_key(sibling)
            existing = merged.get(key)
            if existing is None:
                sibling.metadata = {**sibling.metadata, "relevance_score": trigger_score}
                merged[key] = sibling
                continue
            existing_score = _doc_score(existing)
            if existing_score is None or trigger_score > existing_score:
                sibling.metadata = {**existing.metadata, "relevance_score": trigger_score}
                merged[key] = sibling

    expanded = list(merged.values())
    expanded.sort(
        key=lambda doc: (
            str(doc.metadata.get("document_id", "")),
            _page_number_from_metadata(doc.metadata) if _page_number_from_metadata(doc.metadata) is not None else -1,
            int(doc.metadata.get("chunk_index") or 0),
        )
    )
    return expanded


def _asr_segment_index_from_metadata(metadata: dict) -> int | None:
    for key in ("asr_segment_index", "segment_index"):
        value = metadata.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _asr_segment_dedupe_key(doc: Document) -> tuple[str, int, int]:
    document_id = str(doc.metadata.get("document_id", ""))
    segment_index = _asr_segment_index_from_metadata(doc.metadata)
    chunk_index = int(doc.metadata.get("chunk_index") or 0)
    return (document_id, segment_index if segment_index is not None else -1, chunk_index)


@trace_stage("rag_asr_expand")
def _expand_same_asr_segment_chunks(docs: list[Document]) -> list[Document]:
    if not docs or not settings.retrieval_asr_segment_expand_enabled:
        return docs

    segment_scores: dict[tuple[str, int], float] = {}
    for doc in docs:
        segment_index = _asr_segment_index_from_metadata(doc.metadata)
        document_id = str(doc.metadata.get("document_id", ""))
        if segment_index is None or not document_id:
            continue
        score = _doc_score(doc)
        if score is None:
            continue
        key = (document_id, segment_index)
        existing = segment_scores.get(key)
        if existing is None or score > existing:
            segment_scores[key] = score

    if not segment_scores:
        return docs

    merged: dict[tuple[str, int, int], Document] = {}
    for doc in docs:
        merged[_asr_segment_dedupe_key(doc)] = doc

    for (document_id, segment_index), trigger_score in segment_scores.items():
        siblings = fetch_chunks_by_asr_segment(
            document_id,
            segment_index,
            max_chunks=settings.retrieval_asr_segment_expand_max_chunks,
        )
        for sibling in siblings:
            key = _asr_segment_dedupe_key(sibling)
            existing = merged.get(key)
            if existing is None:
                sibling.metadata = {**sibling.metadata, "relevance_score": trigger_score}
                merged[key] = sibling
                continue
            existing_score = _doc_score(existing)
            if existing_score is None or trigger_score > existing_score:
                sibling.metadata = {**existing.metadata, "relevance_score": trigger_score}
                merged[key] = sibling

    expanded = list(merged.values())
    expanded.sort(
        key=lambda doc: (
            str(doc.metadata.get("document_id", "")),
            _asr_segment_index_from_metadata(doc.metadata)
            if _asr_segment_index_from_metadata(doc.metadata) is not None
            else -1,
            int(doc.metadata.get("chunk_index") or 0),
        )
    )
    return expanded


def _unique_queries(queries: list[str]) -> list[str]:
    return list(dict.fromkeys(q.strip() for q in queries if q.strip()))


def _collect_query_lists(
    queries: list[str],
    *,
    use_hybrid: bool,
    k_multiplier: float = 1.0,
) -> list[list[Document]]:
    unique = _unique_queries(queries)
    if not unique:
        return []

    per_k = _per_query_k(len(unique), k_multiplier=k_multiplier)
    if len(unique) == 1:
        return [_retrieve_with_hybrid_fallback(unique[0], use_hybrid=use_hybrid, k=per_k)]

    lists: list[list[Document]] = [[] for _ in unique]
    with ThreadPoolExecutor(max_workers=min(len(unique), 4)) as executor:
        future_to_idx = {
            executor.submit(
                _retrieve_with_hybrid_fallback, q, use_hybrid=use_hybrid, k=per_k
            ): idx
            for idx, q in enumerate(unique)
        }
        for future in as_completed(future_to_idx):
            lists[future_to_idx[future]] = future.result()
    return lists


def _collect_hyde_lists(
    plan: RetrievalPlan,
    *,
    use_hybrid: bool,
    k_multiplier: float = 1.0,
) -> tuple[list[list[Document]], list[float]]:
    query = plan.standalone_query.strip()
    hyde_doc = (plan.hyde_document or "").strip()

    channel_count = 1 + (1 if use_hybrid else 0)
    if plan.hyde_vector_enabled and hyde_doc:
        channel_count += 1
    per_k = _per_query_k(channel_count, k_multiplier=k_multiplier)

    vector_store = get_vector_store(use_hybrid=False)
    lists: list[list[Document]] = []
    weights: list[float] = []

    if plan.hyde_vector_enabled and hyde_doc:
        lists.append(vector_store.similarity_search(hyde_doc, k=per_k))
        weights.append(1.0)

    lists.append(vector_store.similarity_search(query, k=per_k))
    weights.append(settings.retrieval_rrf_parent_weight)

    if use_hybrid:
        lists.append(bm25_search(query, k=per_k))
        weights.append(settings.retrieval_rrf_parent_weight)

    return lists, weights


def _collect_lists(
    plan: RetrievalPlan,
    *,
    use_hybrid: bool,
    k_multiplier: float = 1.0,
) -> tuple[list[list[Document]], list[float]]:
    query = plan.standalone_query.strip()
    if not query:
        return [], []

    if plan.strategy == "hyde" and plan.hyde_document:
        return _collect_hyde_lists(plan, use_hybrid=use_hybrid, k_multiplier=k_multiplier)

    if plan.strategy in ("multi_query", "decompose"):
        queries = _unique_queries([query, *plan.extra_queries])
        lists = _collect_query_lists(queries, use_hybrid=use_hybrid, k_multiplier=k_multiplier)
        weights = [
            settings.retrieval_rrf_parent_weight if idx == 0 else 1.0 for idx in range(len(lists))
        ]
        return lists, weights

    lists = [
        _retrieve_with_hybrid_fallback(
            query,
            use_hybrid=use_hybrid,
            k=_fetch_k(k_multiplier=k_multiplier),
        )
    ]
    return lists, [settings.retrieval_rrf_parent_weight]


def _queries_from_plan(plan: RetrievalPlan) -> list[str]:
    query = plan.standalone_query.strip()
    if not query:
        return []
    if plan.strategy in ("multi_query", "decompose"):
        return _unique_queries([query, *plan.extra_queries])
    return [query]


def _run_unified_pipeline(
    plan: RetrievalPlan,
    *,
    use_hybrid: bool,
    k_multiplier: float = 1.0,
) -> tuple[list[Document], _PipelineStats]:
    query = plan.standalone_query.strip()
    queries = _queries_from_plan(plan)
    empty_stats = _PipelineStats(
        queries=queries,
        hits_before_rerank=0,
        hits_after_rerank=0,
        page_expand_added=0,
        asr_expand_added=0,
    )
    if not query:
        return [], empty_stats

    lists, weights = _collect_lists(plan, use_hybrid=use_hybrid, k_multiplier=k_multiplier)
    if not lists:
        return [], empty_stats

    fused = lists[0] if len(lists) == 1 else rrf_fuse(lists, list_weights=weights)
    docs = _dedupe_documents(fused)

    before_page = len(docs)
    docs = _expand_same_page_chunks(docs)
    page_expand_added = max(0, len(docs) - before_page)

    before_asr = len(docs)
    docs = _expand_same_asr_segment_chunks(docs)
    asr_expand_added = max(0, len(docs) - before_asr)

    hits_before_rerank = len(docs)
    docs = _apply_rerank(query, docs)
    hits_after_rerank = len(docs)

    stats = _PipelineStats(
        queries=queries,
        hits_before_rerank=hits_before_rerank,
        hits_after_rerank=hits_after_rerank,
        page_expand_added=page_expand_added,
        asr_expand_added=asr_expand_added,
    )
    return docs, stats


def _fallback_extra_queries(plan: RetrievalPlan, *, llm: BaseChatModel | None = None) -> list[str]:
    model = llm or get_small_llm()
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


def _build_tier_configs(plan: RetrievalPlan, *, llm: BaseChatModel | None) -> list[_TierConfig]:
    base_threshold = settings.retrieval_score_threshold
    lowered = base_threshold * settings.retrieval_fallback_threshold_ratio
    use_hybrid = settings.retrieval_hybrid_enabled

    multi_plan = plan.model_copy(
        update={
            "strategy": "multi_query",
            "extra_queries": _fallback_extra_queries(plan, llm=llm),
            "reason": f"{plan.reason}; tier1 multi_query",
        }
    )

    return [
        _TierConfig(plan=plan, threshold=base_threshold, use_hybrid=use_hybrid, k_multiplier=1.0),
        _TierConfig(plan=multi_plan, threshold=base_threshold, use_hybrid=use_hybrid, k_multiplier=1.0),
        _TierConfig(plan=multi_plan, threshold=lowered, use_hybrid=use_hybrid, k_multiplier=1.0),
        _TierConfig(
            plan=multi_plan,
            threshold=lowered,
            use_hybrid=True,
            k_multiplier=settings.retrieval_fallback_k_multiplier,
        ),
    ]


def _tiered_search(
    plan: RetrievalPlan,
    *,
    llm: BaseChatModel | None,
) -> tuple[list[Document], int | None, _PipelineStats | None]:
    if not settings.retrieval_empty_fallback_enabled:
        docs, stats = _run_unified_pipeline(plan, use_hybrid=settings.retrieval_hybrid_enabled)
        return _filter_by_threshold(docs), 0, stats

    tiers = _build_tier_configs(plan, llm=llm)[: settings.retrieval_fallback_max_tiers]
    for idx, tier in enumerate(tiers):
        docs, stats = _run_unified_pipeline(
            tier.plan,
            use_hybrid=tier.use_hybrid,
            k_multiplier=tier.k_multiplier,
        )
        filtered = _filter_by_threshold(docs, tier.threshold)
        if filtered:
            if idx > 0:
                logger.info("Tiered fallback succeeded at tier %s", idx)
            return filtered, idx, stats

    return [], None, None


@trace_stage("rag_es_search")
def search_with_plan(
    plan: RetrievalPlan,
    *,
    llm: BaseChatModel | None = None,
) -> tuple[list[SourceInfo], str | None, list[Document], RetrievalTrace | None]:
    if plan.action == "skip" or not plan.standalone_query.strip():
        return [], None, [], None

    plan = normalize_plan(plan)
    docs, tier, stats = _tiered_search(plan, llm=llm)
    sources, context = _build_sources_and_context(docs)

    retrieval_trace: RetrievalTrace | None = None
    if stats is not None:
        retrieval_trace = RetrievalTrace(
            queries=stats.queries,
            tier=tier,
            hits_before_rerank=stats.hits_before_rerank,
            hits_after_rerank=stats.hits_after_rerank,
            page_expand_added=stats.page_expand_added,
            asr_expand_added=stats.asr_expand_added,
        )

    return sources, context, docs, retrieval_trace


def search_relevant_docs(
    query: str,
) -> tuple[list[SourceInfo], str | None, list[Document], RetrievalTrace | None]:
    if not query.strip():
        return [], None, [], None

    plan = RetrievalPlan(
        action="retrieve",
        strategy="none",
        standalone_query=query,
        reason="direct query",
    )
    return search_with_plan(plan)

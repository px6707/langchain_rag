from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from app.config import settings
from app.schemas.retrieval import RetrievalPlan
from app.services.rerank_service import HttpRerankCompressor, get_rerank_compressor
from app.services.retrieval_service import (
    _PipelineStats,
    _build_sources_and_context,
    _dedupe_documents,
    _filter_by_threshold,
    _per_query_k,
    _run_unified_pipeline,
    _tiered_search,
    search_relevant_docs,
    search_with_plan,
)
from app.services.vector_store_service import clear_vector_store_cache, get_vector_store


def test_http_rerank_compressor_orders_by_score():
    compressor = HttpRerankCompressor(
        model="test-rerank",
        api_base="https://api.example.com/v1",
        api_key="test-key",
        top_n=2,
    )
    docs = [
        Document("first", metadata={"filename": "a.pdf"}),
        Document("second", metadata={"filename": "b.pdf"}),
        Document("third", metadata={"filename": "c.pdf"}),
    ]
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "results": [
            {"index": 2, "relevance_score": 0.95},
            {"index": 0, "relevance_score": 0.81},
        ]
    }
    compressor._session = MagicMock()
    compressor._session.post.return_value = mock_response

    compressed = compressor.compress_documents(docs, "query")

    assert len(compressed) == 2
    assert compressed[0].page_content == "third"


def test_filter_by_threshold_custom():
    docs = [Document("high", metadata={"rerank_score": 0.9})]
    filtered = _filter_by_threshold(docs, threshold=0.95)
    assert filtered == []


def test_per_query_k_respects_min():
    with (
        patch.object(settings, "retrieval_fetch_k", 20),
        patch.object(settings, "rerank_enabled", True),
        patch.object(settings, "retrieval_per_query_k_min", 5),
    ):
        assert _per_query_k(4) == 5


def test_none_strategy_applies_rerank():
    doc = Document("answer", metadata={"filename": "f.pdf"})
    mock_store = MagicMock()
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [doc]
    mock_store.as_retriever.return_value = mock_retriever
    mock_compressor = MagicMock()
    mock_compressor.compress_documents.return_value = [
        Document("answer", metadata={"filename": "f.pdf", "rerank_score": 0.9})
    ]

    plan = RetrievalPlan(action="retrieve", strategy="none", standalone_query="question")

    with (
        patch("app.services.retrieval_service.get_vector_store", return_value=mock_store),
        patch("app.services.retrieval_service.get_rerank_compressor", return_value=mock_compressor),
        patch.object(settings, "retrieval_hybrid_enabled", False),
    ):
        docs, _stats = _run_unified_pipeline(plan, use_hybrid=False)

    mock_compressor.compress_documents.assert_called_once()
    assert docs[0].metadata["rerank_score"] == pytest.approx(0.9)


def test_decompose_includes_standalone_and_subqueries():
    doc_main = Document("main", metadata={"filename": "a.pdf"})
    doc_sub = Document("sub", metadata={"filename": "b.pdf"})
    mock_store = MagicMock()
    mock_retriever = MagicMock()
    mock_retriever.invoke.side_effect = [[doc_main], [doc_sub]]
    mock_store.as_retriever.return_value = mock_retriever

    plan = RetrievalPlan(
        action="retrieve",
        strategy="decompose",
        standalone_query="main question",
        extra_queries=["sub question"],
    )

    with (
        patch("app.services.retrieval_service.get_vector_store", return_value=mock_store),
        patch("app.services.retrieval_service.get_rerank_compressor", return_value=None),
        patch.object(settings, "retrieval_hybrid_enabled", False),
    ):
        docs, _stats = _run_unified_pipeline(plan, use_hybrid=False)

    assert mock_retriever.invoke.call_count == 2
    assert len(docs) == 2


def test_hyde_uses_bm25_channel():
    hyde_doc = Document("hyde hit", metadata={"filename": "h.pdf"})
    query_vec_doc = Document("query vec", metadata={"filename": "v.pdf"})
    bm25_doc = Document("bm25 hit", metadata={"filename": "b.pdf"})
    mock_vector_store = MagicMock()
    mock_vector_store.similarity_search.side_effect = [[hyde_doc], [query_vec_doc]]

    plan = RetrievalPlan(
        action="retrieve",
        strategy="hyde",
        standalone_query="concept question",
        hyde_document="A hypothetical answer about the concept with enough length.",
        hyde_vector_enabled=True,
    )

    with (
        patch("app.services.retrieval_service.get_vector_store", return_value=mock_vector_store),
        patch("app.services.retrieval_service.bm25_search", return_value=[bm25_doc]) as mock_bm25,
        patch("app.services.retrieval_service.get_rerank_compressor", return_value=None),
        patch.object(settings, "retrieval_hybrid_enabled", True),
    ):
        docs, _stats = _run_unified_pipeline(plan, use_hybrid=True)

    mock_bm25.assert_called_once()
    assert len(docs) == 3


def test_tiered_search_escalates_tiers():
    plan = RetrievalPlan(action="retrieve", strategy="none", standalone_query="q")
    good_doc = [Document("hit", metadata={"filename": "a.pdf", "rerank_score": 0.9})]

    empty_stats = _PipelineStats(
        queries=["q"],
        hits_before_rerank=0,
        hits_after_rerank=0,
        page_expand_added=0,
        asr_expand_added=0,
    )
    good_stats = _PipelineStats(
        queries=["q"],
        hits_before_rerank=1,
        hits_after_rerank=1,
        page_expand_added=0,
        asr_expand_added=0,
    )

    with (
        patch(
            "app.services.retrieval_service._run_unified_pipeline",
            side_effect=[([], empty_stats), (good_doc, good_stats)],
        ),
        patch("app.services.retrieval_service._fallback_extra_queries", return_value=["alt"]),
        patch.object(settings, "retrieval_empty_fallback_enabled", True),
        patch.object(settings, "retrieval_fallback_max_tiers", 2),
        patch.object(settings, "retrieval_score_threshold", 0.7),
        patch.object(settings, "retrieval_fallback_threshold_ratio", 0.5),
    ):
        docs, tier, stats = _tiered_search(plan, llm=MagicMock())

    assert len(docs) == 1
    assert tier == 1
    assert stats is good_stats


def test_hybrid_strategy_config():
    clear_vector_store_cache()
    with (
        patch.object(settings, "retrieval_hybrid_enabled", True),
        patch.object(settings, "retrieval_rrf_enabled", True),
        patch.object(settings, "es_url", "http://localhost:9200"),
        patch.object(settings, "es_index", "test_hybrid_index"),
    ):
        store = get_vector_store(use_hybrid=True)
        strategy = store._store.retrieval_strategy
        assert strategy.hybrid is True
    clear_vector_store_cache()


@pytest.mark.asyncio
async def test_retrieval_middleware_turn_cache():
    from app.agent.middleware import retrieval as retrieval_middleware

    retrieval_middleware._turn_cache.set(None)
    llm = MagicMock()
    middleware = retrieval_middleware.RetrievalMiddleware(llm)
    handler = AsyncMock(return_value=MagicMock())
    skip_plan = RetrievalPlan(action="skip", reason="cached")
    request = MagicMock(messages=[HumanMessage(content="hello", id="msg-1")], system_message=None)

    with (
        patch("app.agent.middleware.retrieval.plan_retrieval", return_value=skip_plan) as mock_plan,
        patch("app.agent.middleware.retrieval.search_with_plan") as mock_search,
    ):
        await middleware.awrap_model_call(request, handler)
        await middleware.awrap_model_call(request, handler)

    assert mock_plan.call_count == 1
    mock_search.assert_not_called()

    retrieval_middleware._turn_cache.set(None)

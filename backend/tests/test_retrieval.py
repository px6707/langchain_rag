from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from app.config import settings
from app.schemas.retrieval import RetrievalPlan
from app.services.rerank_service import HttpRerankCompressor, get_rerank_compressor
from app.services.retrieval_service import (
    _build_sources_and_context,
    _dedupe_documents,
    _filter_by_threshold,
    _per_query_k,
    _run_unified_pipeline,
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
    assert compressed[0].metadata["rerank_score"] == pytest.approx(0.95)
    assert compressed[1].metadata["rerank_score"] == pytest.approx(0.81)


def test_filter_by_rerank_score_threshold():
    docs = [
        Document("low", metadata={"rerank_score": 0.5}),
        Document("high", metadata={"rerank_score": 0.9}),
    ]
    with patch.object(settings, "retrieval_score_threshold", 0.7):
        filtered = _filter_by_threshold(docs)
    assert len(filtered) == 1
    assert filtered[0].page_content == "high"


def test_build_sources_includes_score():
    docs = [Document("content body", metadata={"filename": "doc.pdf", "rerank_score": 0.88})]
    sources, context = _build_sources_and_context(docs)
    assert len(sources) == 1
    assert sources[0].score == pytest.approx(0.88)
    assert "0.880" in context


def test_get_rerank_compressor_disabled():
    with patch.object(settings, "rerank_enabled", False):
        assert get_rerank_compressor() is None


def test_get_rerank_compressor_requires_model():
    with (
        patch.object(settings, "rerank_enabled", True),
        patch.object(settings, "rerank_model", ""),
        patch.object(settings, "rerank_api_key", "key"),
    ):
        assert get_rerank_compressor() is None


def test_per_query_k_respects_min():
    with (
        patch.object(settings, "retrieval_fetch_k", 20),
        patch.object(settings, "rerank_enabled", True),
        patch.object(settings, "retrieval_per_query_k_min", 5),
    ):
        assert _per_query_k(4) == 5
        assert _per_query_k(1) == 20


def test_search_relevant_docs_with_mock_retriever():
    doc = Document("answer text", metadata={"filename": "f.pdf", "rerank_score": 0.91})
    mock_store = MagicMock()
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [doc]
    mock_store.as_retriever.return_value = mock_retriever

    with (
        patch("app.services.retrieval_service.get_vector_store", return_value=mock_store),
        patch("app.services.retrieval_service.get_rerank_compressor", return_value=None),
        patch.object(settings, "retrieval_hybrid_enabled", False),
        patch.object(settings, "retrieval_score_threshold", 0.7),
    ):
        sources, context = search_relevant_docs("question")

    assert len(sources) == 1
    assert sources[0].filename == "f.pdf"
    assert context is not None


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
        docs = _run_unified_pipeline(plan, use_hybrid=False)

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
        patch.object(settings, "retrieval_fetch_k", 20),
        patch.object(settings, "retrieval_per_query_k_min", 5),
    ):
        docs = _run_unified_pipeline(plan, use_hybrid=False)

    assert mock_retriever.invoke.call_count == 2
    assert len(docs) == 2


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
        assert strategy.rrf is True
    clear_vector_store_cache()


def test_dedupe_documents_keeps_higher_score():
    docs = [
        Document("same content", metadata={"filename": "a.pdf", "rerank_score": 0.5}),
        Document("same content", metadata={"filename": "a.pdf", "rerank_score": 0.9}),
        Document("other", metadata={"filename": "b.pdf"}),
    ]
    deduped = _dedupe_documents(docs)
    assert len(deduped) == 2
    by_content = {d.page_content: d for d in deduped}
    assert by_content["same content"].metadata["rerank_score"] == pytest.approx(0.9)


def test_search_with_plan_skip():
    plan = RetrievalPlan(action="skip", reason="闲聊")
    sources, context = search_with_plan(plan)
    assert sources == []
    assert context is None


def test_search_with_plan_hyde_three_channels():
    hyde_doc = Document("hyde hit", metadata={"filename": "h.pdf"})
    query_vec_doc = Document("query vec", metadata={"filename": "v.pdf"})
    hybrid_doc = Document("hybrid hit", metadata={"filename": "q.pdf"})
    mock_vector_store = MagicMock()
    mock_vector_store.similarity_search.side_effect = [[hyde_doc], [query_vec_doc]]
    mock_hybrid_store = MagicMock()
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [hybrid_doc]
    mock_hybrid_store.as_retriever.return_value = mock_retriever

    plan = RetrievalPlan(
        action="retrieve",
        strategy="hyde",
        standalone_query="concept question",
        hyde_document="A hypothetical answer about the concept with enough length.",
    )

    def fake_get_store(*, use_hybrid: bool):
        return mock_hybrid_store if use_hybrid else mock_vector_store

    with (
        patch("app.services.retrieval_service.get_vector_store", side_effect=fake_get_store),
        patch("app.services.retrieval_service.get_rerank_compressor", return_value=None),
        patch.object(settings, "retrieval_hybrid_enabled", True),
        patch.object(settings, "retrieval_score_threshold", 0.0),
    ):
        sources, _ = search_with_plan(plan)

    assert mock_vector_store.similarity_search.call_count == 2
    assert mock_retriever.invoke.call_count == 1
    assert len(sources) == 3


def test_search_with_plan_empty_fallback():
    mock_store = MagicMock()
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = []
    mock_store.as_retriever.return_value = mock_retriever

    plan = RetrievalPlan(action="retrieve", strategy="none", standalone_query="question")

    with (
        patch("app.services.retrieval_service.get_vector_store", return_value=mock_store),
        patch("app.services.retrieval_service.get_rerank_compressor", return_value=None),
        patch("app.services.retrieval_service._fallback_extra_queries", return_value=["alt1"]),
        patch.object(settings, "retrieval_hybrid_enabled", False),
        patch.object(settings, "retrieval_empty_fallback_enabled", True),
        patch.object(settings, "retrieval_score_threshold", 0.7),
    ):
        search_with_plan(plan)

    assert mock_retriever.invoke.call_count >= 2


@pytest.mark.asyncio
async def test_retrieval_middleware_turn_cache():
    from app.agent.middleware import retrieval as retrieval_middleware

    retrieval_middleware._turn_cache.set(None)
    retrieval_middleware._pending_sources.set(None)

    llm = MagicMock()
    middleware = retrieval_middleware.RetrievalMiddleware(llm)
    handler = AsyncMock(return_value=MagicMock())

    skip_plan = RetrievalPlan(action="skip", reason="cached turn")
    request = MagicMock()
    request.messages = [HumanMessage(content="hello", id="msg-1")]
    request.system_message = None

    with (
        patch("app.agent.middleware.retrieval.plan_retrieval", return_value=skip_plan) as mock_plan,
        patch("app.agent.middleware.retrieval.search_with_plan") as mock_search,
    ):
        await middleware.awrap_model_call(request, handler)
        await middleware.awrap_model_call(request, handler)

    assert mock_plan.call_count == 1
    mock_search.assert_not_called()
    assert handler.call_count == 2

    retrieval_middleware._turn_cache.set(None)


def test_get_router_llm_fallback():
    from app.services.llm_service import get_router_llm

    with (
        patch.object(settings, "router_llm_model", ""),
        patch.object(settings, "router_llm_api_base", ""),
        patch.object(settings, "router_llm_api_key", ""),
        patch.object(settings, "llm_model", "main-model"),
        patch.object(settings, "llm_api_base", "https://main.example/v1"),
        patch.object(settings, "llm_api_key", "main-key"),
    ):
        llm = get_router_llm()
    assert llm.model_name == "main-model"

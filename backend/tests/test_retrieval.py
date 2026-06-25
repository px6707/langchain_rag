from copy import deepcopy
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from app.config import settings
from app.services.rerank_service import HttpRerankCompressor, get_rerank_compressor
from app.services.retrieval_service import _build_sources_and_context, _filter_by_threshold, search_relevant_docs
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
    assert "answer text" in context


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

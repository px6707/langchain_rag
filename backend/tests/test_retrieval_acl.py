from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from app.services.retrieval_context import reset_retrieval_user_context, set_retrieval_user_context
from app.services.retrieval_service import _retrieve_raw
from app.services.vector_store_service import bm25_search, filter_documents_by_user, user_es_term


def test_user_es_term_shape():
    user_id = "550e8400-e29b-41d4-a716-446655440000"
    assert user_es_term(user_id) == {
        "term": {"metadata.user_id.keyword": user_id},
    }


def test_filter_documents_by_user():
    token = set_retrieval_user_context("user-a", is_admin=False)
    try:
        docs = [
            Document("a", metadata={"user_id": "user-a"}),
            Document("b", metadata={"user_id": "user-b"}),
        ]
        filtered = filter_documents_by_user(docs)
        assert len(filtered) == 1
        assert filtered[0].page_content == "a"
    finally:
        reset_retrieval_user_context(token)


def test_filter_documents_admin_sees_all():
    token = set_retrieval_user_context("user-a", is_admin=True)
    try:
        docs = [
            Document("a", metadata={"user_id": "user-a"}),
            Document("b", metadata={"user_id": "user-b"}),
        ]
        assert len(filter_documents_by_user(docs)) == 2
    finally:
        reset_retrieval_user_context(token)


def test_retrieve_raw_applies_user_filter():
    token = set_retrieval_user_context("user-a", is_admin=False)
    mock_store = MagicMock()
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [Document("x", metadata={"user_id": "user-a"})]
    mock_store.as_retriever.return_value = mock_retriever
    try:
        with patch("app.services.retrieval_service.get_vector_store", return_value=mock_store):
            docs = _retrieve_raw("query", use_hybrid=False, k=3)
        kwargs = mock_store.as_retriever.call_args.kwargs["search_kwargs"]
        assert kwargs["filter"] == user_es_term("user-a")
        assert len(docs) == 1
    finally:
        reset_retrieval_user_context(token)


def test_bm25_search_adds_user_filter_for_non_admin():
    token = set_retrieval_user_context("user-a", is_admin=False)
    mock_client = MagicMock()
    mock_client.search.return_value = {"hits": {"hits": []}}
    mock_store = MagicMock(client=mock_client)
    try:
        with patch("app.services.vector_store_service.get_vector_store", return_value=mock_store):
            bm25_search("hello", k=2)
        query = mock_client.search.call_args.kwargs["query"]
        assert "filter" in query["bool"]
    finally:
        reset_retrieval_user_context(token)

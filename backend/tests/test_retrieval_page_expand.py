from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from app.config import settings
from app.services.retrieval_service import _expand_same_page_chunks


def test_expand_same_page_chunks_adds_siblings():
    hit = Document(
        "row chunk",
        metadata={
            "document_id": "doc-1",
            "page_number": 3,
            "chunk_index": 5,
            "relevance_score": 0.9,
            "filename": "report.pdf",
        },
    )
    sibling_a = Document(
        "other row",
        metadata={"document_id": "doc-1", "page_number": 3, "chunk_index": 6, "filename": "report.pdf"},
    )
    sibling_b = Document(
        "caption",
        metadata={"document_id": "doc-1", "page_number": 3, "chunk_index": 4, "filename": "report.pdf"},
    )

    with (
        patch.object(settings, "retrieval_page_expand_enabled", True),
        patch(
            "app.services.retrieval_service.fetch_chunks_by_page",
            return_value=[sibling_b, hit, sibling_a],
        ) as mock_fetch,
    ):
        expanded = _expand_same_page_chunks([hit])

    mock_fetch.assert_called_once_with("doc-1", 3, max_chunks=settings.retrieval_page_expand_max_chunks)
    chunk_indices = [doc.metadata["chunk_index"] for doc in expanded]
    assert chunk_indices == [4, 5, 6]
    assert expanded[1].metadata.get("relevance_score") == 0.9


def test_expand_skips_without_page_number():
    doc = Document("text", metadata={"document_id": "doc-1", "chunk_index": 0})
    with patch.object(settings, "retrieval_page_expand_enabled", True):
        assert _expand_same_page_chunks([doc]) == [doc]


def test_expand_disabled():
    doc = Document(
        "text",
        metadata={"document_id": "doc-1", "page_number": 1, "chunk_index": 0, "relevance_score": 0.5},
    )
    with (
        patch.object(settings, "retrieval_page_expand_enabled", False),
        patch("app.services.retrieval_service.fetch_chunks_by_page") as mock_fetch,
    ):
        assert _expand_same_page_chunks([doc]) == [doc]
    mock_fetch.assert_not_called()

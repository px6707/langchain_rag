import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document
from sqlalchemy.exc import InvalidRequestError

from app.models import Document as DocumentModel
from app.parsing.indexer import DocumentDeletedError, index_documents


def test_index_documents_adds_then_deletes_old_batches():
    doc_id = uuid.uuid4()
    document = DocumentModel(
        id=doc_id,
        filename="notes.txt",
        file_path="/tmp/notes.txt",
        file_type="txt",
    )
    raw_docs = [Document(page_content="hello world", metadata={})]
    session = AsyncMock()
    mock_store = MagicMock()
    call_order: list[str] = []

    def track_add(chunks: list[Document]) -> None:
        call_order.append("add")
        assert all("index_batch" in c.metadata for c in chunks)

    def track_delete_except(document_id: str, keep_batch: str) -> None:
        call_order.append("delete_except")

    mock_store.add_documents.side_effect = track_add

    with (
        patch("app.parsing.indexer.get_vector_store", return_value=mock_store),
        patch("app.parsing.indexer.delete_document_vectors_except_batch_with_retry", side_effect=track_delete_except),
        patch("app.parsing.indexer.chunk_documents", return_value=raw_docs),
        patch("app.parsing.indexer.asyncio.to_thread", side_effect=lambda fn, *args, **kwargs: fn(*args, **kwargs)),
    ):
        count = asyncio.run(index_documents(session, document, raw_docs))

    assert count == 1
    assert call_order == ["add", "delete_except"]
    assert document.status == "completed"
    assert document.chunk_count == 1
    session.commit.assert_awaited()


def test_index_documents_add_failure_skips_delete():
    doc_id = uuid.uuid4()
    document = DocumentModel(
        id=doc_id,
        filename="notes.txt",
        file_path="/tmp/notes.txt",
        file_type="txt",
    )
    raw_docs = [Document(page_content="hello world", metadata={})]
    session = AsyncMock()
    mock_store = MagicMock()
    mock_store.add_documents.side_effect = RuntimeError("ES down")

    with (
        patch("app.parsing.indexer.get_vector_store", return_value=mock_store),
        patch("app.parsing.indexer.delete_document_vectors_except_batch_with_retry") as mock_delete,
        patch("app.parsing.indexer.chunk_documents", return_value=raw_docs),
        patch("app.parsing.indexer.asyncio.to_thread", side_effect=lambda fn, *args, **kwargs: fn(*args, **kwargs)),
    ):
        with pytest.raises(RuntimeError, match="ES down"):
            asyncio.run(index_documents(session, document, raw_docs))

    mock_delete.assert_not_called()


def test_index_documents_delete_retry_failure_sets_warning():
    doc_id = uuid.uuid4()
    document = DocumentModel(
        id=doc_id,
        filename="notes.txt",
        file_path="/tmp/notes.txt",
        file_type="txt",
    )
    raw_docs = [Document(page_content="hello world", metadata={})]
    session = AsyncMock()
    mock_store = MagicMock()

    with (
        patch("app.parsing.indexer.get_vector_store", return_value=mock_store),
        patch(
            "app.parsing.indexer.delete_document_vectors_except_batch_with_retry",
            side_effect=RuntimeError("delete failed"),
        ),
        patch("app.parsing.indexer.chunk_documents", return_value=raw_docs),
        patch("app.parsing.indexer.asyncio.to_thread", side_effect=lambda fn, *args, **kwargs: fn(*args, **kwargs)),
    ):
        count = asyncio.run(index_documents(session, document, raw_docs))

    assert count == 1
    assert "reparse" in (document.error_message or "")


def test_index_documents_raises_when_document_deleted():
    doc_id = uuid.uuid4()
    document = DocumentModel(
        id=doc_id,
        filename="notes.txt",
        file_path="/tmp/notes.txt",
        file_type="txt",
    )
    raw_docs = [Document(page_content="hello world", metadata={})]
    session = AsyncMock()
    session.refresh.side_effect = InvalidRequestError("deleted")

    with patch("app.parsing.indexer.chunk_documents", return_value=raw_docs):
        with pytest.raises(DocumentDeletedError):
            asyncio.run(index_documents(session, document, raw_docs))

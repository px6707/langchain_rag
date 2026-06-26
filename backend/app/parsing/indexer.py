import asyncio
import logging
import uuid

from langchain_core.documents import Document
from sqlalchemy.exc import InvalidRequestError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Document as DocumentModel
from app.parsing.chunking import chunk_documents
from app.services.vector_store_service import (
    delete_document_vectors_except_batch_with_retry,
    get_vector_store,
)

logger = logging.getLogger(__name__)


class DocumentDeletedError(Exception):
    pass


def assign_chunk_indices(chunks: list[Document]) -> list[Document]:
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
    return chunks


async def _ensure_document_exists(session: AsyncSession, document: DocumentModel) -> None:
    try:
        await session.refresh(document)
    except InvalidRequestError as exc:
        raise DocumentDeletedError("Document was deleted during parsing") from exc


async def index_documents(
    session: AsyncSession,
    document: DocumentModel,
    raw_docs: list[Document],
) -> int:
    for doc in raw_docs:
        doc.metadata.setdefault("document_id", str(document.id))
        doc.metadata.setdefault("filename", document.filename)

    chunks = chunk_documents(raw_docs)
    if not chunks:
        raise ValueError("No content extracted from document")

    chunks = assign_chunk_indices(chunks)
    index_batch = str(uuid.uuid4())
    for chunk in chunks:
        chunk.metadata["index_batch"] = index_batch

    await _ensure_document_exists(session, document)

    vector_store = get_vector_store()
    await asyncio.to_thread(vector_store.add_documents, chunks)
    cleanup_warning: str | None = None
    try:
        await asyncio.to_thread(
            delete_document_vectors_except_batch_with_retry,
            str(document.id),
            index_batch,
        )
    except Exception:
        cleanup_warning = "Indexed with duplicate batches possible; consider reparse"
        logger.error("Old index batch cleanup failed for document_id=%s", document.id)

    document.chunk_count = len(chunks)
    document.status = "completed"
    document.parse_stage = None
    document.error_message = cleanup_warning
    await session.commit()
    return len(chunks)

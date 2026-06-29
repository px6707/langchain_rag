import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

from app.models import Document
from app.services.document_service import DocumentService


def test_reparse_document_resets_status_and_enqueues():
    doc_id = uuid.uuid4()
    user_id = uuid.uuid4()
    doc = Document(
        id=doc_id,
        user_id=user_id,
        filename="a.pdf",
        file_path="/tmp/a.pdf",
        file_type="pdf",
        status="failed",
        parse_stage=None,
        error_message="old error",
        chunk_count=5,
    )
    session = AsyncMock()
    service = DocumentService(session)

    with (
        patch.object(service, "get_document", new_callable=AsyncMock, return_value=doc),
        patch("app.services.document_service.os.path.exists", return_value=True),
        patch("app.services.document_service.delete_document_vectors") as mock_delete,
        patch("app.services.document_service.enqueue_reparse", new_callable=AsyncMock) as mock_enqueue,
    ):
        result = asyncio.run(
            service.reparse_document(doc_id, user_id=user_id, is_admin=False)
        )

    assert result is doc
    assert doc.status == "queued"
    assert doc.parse_stage == "queued"
    assert doc.error_message is None
    assert doc.chunk_count == 0
    mock_delete.assert_called_once_with(str(doc_id))
    mock_enqueue.assert_awaited_once()

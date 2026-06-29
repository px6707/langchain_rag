import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

from app.models import Document
from app.services.document_service import DocumentService


def _doc(doc_id: uuid.UUID, user_id: uuid.UUID) -> Document:
    return Document(
        id=doc_id,
        user_id=user_id,
        filename="a.pdf",
        file_path="/tmp/a.pdf",
        file_type="pdf",
    )


def test_scoped_query_filters_non_admin():
    user_id = uuid.uuid4()
    service = DocumentService(AsyncMock())
    stmt = service._scoped_query(user_id, is_admin=False)
    assert "user_id" in str(stmt).lower()


def test_get_document_returns_none_when_not_in_scope():
    session = AsyncMock()
    session.execute = AsyncMock(
        return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=None))
    )
    service = DocumentService(session)
    missing = asyncio.run(
        service.get_document(uuid.uuid4(), user_id=uuid.uuid4(), is_admin=False)
    )
    assert missing is None


def test_get_document_allows_admin():
    owner_id = uuid.uuid4()
    admin_id = uuid.uuid4()
    doc_id = uuid.uuid4()
    doc = _doc(doc_id, owner_id)
    session = AsyncMock()
    session.execute = AsyncMock(
        return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=doc))
    )
    service = DocumentService(session)

    found = asyncio.run(service.get_document(doc_id, user_id=admin_id, is_admin=True))
    assert found is doc


def test_list_documents_scopes_non_admin():
    user_id = uuid.uuid4()
    session = AsyncMock()
    session.execute = AsyncMock(
        side_effect=[
            MagicMock(scalar_one=MagicMock(return_value=1)),
            MagicMock(scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[])))),
        ]
    )
    service = DocumentService(session)
    items, total = asyncio.run(service.list_documents(user_id=user_id, is_admin=False))
    assert total == 1
    assert items == []
    first_stmt = session.execute.await_args_list[0].args[0]
    assert "user_id" in str(first_stmt).lower()


def test_delete_document_requires_ownership():
    owner_id = uuid.uuid4()
    other_id = uuid.uuid4()
    doc_id = uuid.uuid4()
    session = AsyncMock()
    service = DocumentService(session)

    with patch.object(service, "get_document", new_callable=AsyncMock, return_value=None):
        deleted = asyncio.run(
            service.delete_document(doc_id, user_id=other_id, is_admin=False)
        )
    assert deleted is False

    doc = _doc(doc_id, owner_id)
    with (
        patch.object(service, "get_document", new_callable=AsyncMock, return_value=doc),
        patch("app.services.document_service.os.path.exists", return_value=False),
        patch("app.services.document_service.delete_document_vectors"),
    ):
        deleted = asyncio.run(
            service.delete_document(doc_id, user_id=owner_id, is_admin=False)
        )
    assert deleted is True

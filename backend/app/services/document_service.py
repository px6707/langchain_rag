import logging
import os
import uuid
from pathlib import Path

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models import Document, ParseJob
from app.parsing.file_validation import validate_upload
from app.parsing.router import get_file_type_label
from app.services.parse_job_service import enqueue_parse_job, enqueue_reparse
from app.services.vector_store_service import delete_document_vectors, get_vector_store

logger = logging.getLogger(__name__)


class DocumentService:
    def __init__(self, db: AsyncSession):
        self.db = db

    def _scoped_query(self, user_id: uuid.UUID, *, is_admin: bool):
        query = select(Document)
        if not is_admin:
            query = query.where(Document.user_id == user_id)
        return query

    async def save_upload(
        self,
        filename: str,
        content: bytes,
        *,
        user_id: uuid.UUID,
    ) -> Document:
        upload_dir = Path(settings.upload_dir)
        upload_dir.mkdir(parents=True, exist_ok=True)

        validate_upload(filename, content)

        ext = Path(filename).suffix.lower()
        file_type = get_file_type_label(filename)
        doc_id = uuid.uuid4()
        safe_filename = f"{doc_id}{ext}"
        file_path = upload_dir / safe_filename
        file_path.write_bytes(content)

        doc = Document(
            id=doc_id,
            user_id=user_id,
            filename=filename,
            file_path=str(file_path),
            file_type=file_type,
            status="queued",
            parse_stage="queued",
        )
        self.db.add(doc)
        await self.db.commit()
        await self.db.refresh(doc)
        await enqueue_parse_job(self.db, doc.id)
        return doc

    async def reparse_document(
        self,
        doc_id: uuid.UUID,
        *,
        user_id: uuid.UUID,
        is_admin: bool = False,
    ) -> Document | None:
        doc = await self.get_document(doc_id, user_id=user_id, is_admin=is_admin)
        if not doc:
            return None
        if not os.path.exists(doc.file_path):
            raise ValueError("Document file no longer exists on disk")

        delete_document_vectors(str(doc_id))
        doc.status = "queued"
        doc.parse_stage = "queued"
        doc.error_message = None
        doc.chunk_count = 0
        await self.db.commit()
        await self.db.refresh(doc)
        await enqueue_reparse(self.db, doc.id)
        return doc

    async def list_documents(
        self,
        *,
        user_id: uuid.UUID,
        is_admin: bool = False,
        skip: int = 0,
        limit: int = 20,
    ) -> tuple[list[Document], int]:
        count_stmt = select(func.count()).select_from(Document)
        if not is_admin:
            count_stmt = count_stmt.where(Document.user_id == user_id)
        count_result = await self.db.execute(count_stmt)
        total = count_result.scalar_one()

        result = await self.db.execute(
            self._scoped_query(user_id, is_admin=is_admin)
            .order_by(Document.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        items = list(result.scalars().all())
        return items, total

    async def get_document(
        self,
        doc_id: uuid.UUID,
        *,
        user_id: uuid.UUID,
        is_admin: bool = False,
    ) -> Document | None:
        result = await self.db.execute(
            self._scoped_query(user_id, is_admin=is_admin).where(Document.id == doc_id)
        )
        return result.scalar_one_or_none()

    async def delete_document(
        self,
        doc_id: uuid.UUID,
        *,
        user_id: uuid.UUID,
        is_admin: bool = False,
    ) -> bool:
        doc = await self.get_document(doc_id, user_id=user_id, is_admin=is_admin)
        if not doc:
            return False

        delete_document_vectors(str(doc_id))

        if os.path.exists(doc.file_path):
            os.remove(doc.file_path)

        await self.db.execute(delete(ParseJob).where(ParseJob.document_id == doc_id))
        await self.db.execute(delete(Document).where(Document.id == doc_id))
        await self.db.commit()
        return True

import os
import uuid
from pathlib import Path

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document as LCDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models import Document
from app.services.vector_store_service import get_vector_store

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}


def _get_file_type(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return "pdf"
    if ext in {".txt", ".md"}:
        return ext.lstrip(".")
    if ext == ".docx":
        return "docx"
    raise ValueError(f"Unsupported file type: {ext}")


def _load_document(file_path: str, file_type: str) -> list[LCDocument]:
    if file_type == "pdf":
        loader = PyPDFLoader(file_path)
    elif file_type in {"txt", "md"}:
        loader = TextLoader(file_path, encoding="utf-8")
    elif file_type == "docx":
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
    return loader.load()


def _split_documents(docs: list[LCDocument]) -> list[LCDocument]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return splitter.split_documents(docs)


async def process_document_task(doc_id: uuid.UUID) -> None:
    from app.database import async_session

    async with async_session() as session:
        result = await session.execute(select(Document).where(Document.id == doc_id))
        doc = result.scalar_one_or_none()
        if not doc:
            return

        try:
            raw_docs = _load_document(doc.file_path, doc.file_type)
            for d in raw_docs:
                d.metadata["document_id"] = str(doc.id)
                d.metadata["filename"] = doc.filename

            chunks = _split_documents(raw_docs)
            if not chunks:
                raise ValueError("No content extracted from document")

            vector_store = get_vector_store()
            vector_store.add_documents(chunks)

            doc.chunk_count = len(chunks)
            doc.status = "completed"
            doc.error_message = None
        except Exception as e:
            doc.status = "failed"
            doc.error_message = str(e)

        await session.commit()


class DocumentService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def save_upload(self, filename: str, content: bytes) -> Document:
        upload_dir = Path(settings.upload_dir)
        upload_dir.mkdir(parents=True, exist_ok=True)

        file_type = _get_file_type(filename)
        doc_id = uuid.uuid4()
        safe_filename = f"{doc_id}{Path(filename).suffix}"
        file_path = upload_dir / safe_filename

        file_path.write_bytes(content)

        doc = Document(
            id=doc_id,
            filename=filename,
            file_path=str(file_path),
            file_type=file_type,
            status="processing",
        )
        self.db.add(doc)
        await self.db.commit()
        await self.db.refresh(doc)
        return doc

    async def list_documents(self, skip: int = 0, limit: int = 20) -> tuple[list[Document], int]:
        count_result = await self.db.execute(select(func.count()).select_from(Document))
        total = count_result.scalar_one()

        result = await self.db.execute(
            select(Document).order_by(Document.created_at.desc()).offset(skip).limit(limit)
        )
        items = list(result.scalars().all())
        return items, total

    async def get_document(self, doc_id: uuid.UUID) -> Document | None:
        result = await self.db.execute(select(Document).where(Document.id == doc_id))
        return result.scalar_one_or_none()

    async def delete_document(self, doc_id: uuid.UUID) -> bool:
        doc = await self.get_document(doc_id)
        if not doc:
            return False

        vector_store = get_vector_store()
        vector_store.delete(filter={"term": {"metadata.document_id.keyword": str(doc_id)}})

        if os.path.exists(doc.file_path):
            os.remove(doc.file_path)

        await self.db.execute(delete(Document).where(Document.id == doc_id))
        await self.db.commit()
        return True

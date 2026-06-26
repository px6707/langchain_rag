from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import get_current_user
from app.config import settings
from app.database import get_db
from app.models import User
from app.parsing.router import ALLOWED_EXTENSIONS
from app.schemas import DocumentChunkResponse, DocumentListResponse, DocumentResponse
from app.services.document_service import DocumentService
from app.services.vector_store_service import get_chunk_by_ref

router = APIRouter(
    prefix="/api/documents",
    tags=["documents"],
    dependencies=[Depends(get_current_user)],
)


@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile,
    db: AsyncSession = Depends(get_db),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    content = await file.read()
    max_bytes = settings.parse_max_file_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"File too large (max {settings.parse_max_file_mb}MB)",
        )

    service = DocumentService(db)
    try:
        doc = await service.save_upload(file.filename, content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return doc


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    service = DocumentService(db)
    items, total = await service.list_documents(skip=skip, limit=limit)
    return DocumentListResponse(items=items, total=total)


@router.get("/{doc_id}/chunks/{chunk_index}", response_model=DocumentChunkResponse)
async def get_document_chunk(doc_id: UUID, chunk_index: int):
    if chunk_index < 0:
        raise HTTPException(status_code=400, detail="chunk_index must be non-negative")

    doc = get_chunk_by_ref(str(doc_id), chunk_index)
    if doc is None:
        raise HTTPException(status_code=404, detail="Chunk not found")

    document_id = str(doc.metadata.get("document_id", doc_id))
    filename = str(doc.metadata.get("filename", "unknown"))
    ref_id = f"{document_id}#{chunk_index}"
    return DocumentChunkResponse(
        document_id=document_id,
        chunk_index=chunk_index,
        ref_id=ref_id,
        filename=filename,
        content=doc.page_content,
    )


@router.post("/{doc_id}/reparse", response_model=DocumentResponse)
async def reparse_document(doc_id: UUID, db: AsyncSession = Depends(get_db)):
    service = DocumentService(db)
    try:
        doc = await service.reparse_document(doc_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@router.delete("/{doc_id}")
async def delete_document(doc_id: UUID, db: AsyncSession = Depends(get_db)):
    service = DocumentService(db)
    deleted = await service.delete_document(doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"message": "Document deleted"}

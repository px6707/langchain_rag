from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import get_current_user
from app.database import get_db
from app.models import User
from app.schemas import DocumentChunkResponse, DocumentListResponse, DocumentResponse
from app.services.document_service import ALLOWED_EXTENSIONS, DocumentService, process_document_task
from app.services.vector_store_service import get_chunk_by_ref

router = APIRouter(
    prefix="/api/documents",
    tags=["documents"],
    dependencies=[Depends(get_current_user)],
)


@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile,
    db: AsyncSession = Depends(get_db),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    content = await file.read()
    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 20MB)")

    service = DocumentService(db)
    doc = await service.save_upload(file.filename, content)
    background_tasks.add_task(process_document_task, doc.id)
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


@router.delete("/{doc_id}")
async def delete_document(doc_id: UUID, db: AsyncSession = Depends(get_db)):
    service = DocumentService(db)
    deleted = await service.delete_document(doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"message": "Document deleted"}

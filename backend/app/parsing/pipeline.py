import asyncio
import logging

from langchain_core.documents import Document
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models import Document as DocumentModel, ParseJob
from app.parsing.asr_service import transcribe_audio_segments
from app.parsing.indexer import index_documents
from app.parsing.mineru_client import MinerUError, parse_file_to_documents
from app.parsing.python_loaders import load_with_python
from app.parsing.router import classify_file, supports_python_fallback
from app.parsing.video_service import cleanup_temp_dir, extract_video_assets
from app.parsing.vlm_service import summarize_image

logger = logging.getLogger(__name__)


async def _update_stage(session: AsyncSession, document: DocumentModel, stage: str) -> None:
    document.status = "processing"
    document.parse_stage = stage
    await session.commit()


async def _parse_with_mineru(
    session: AsyncSession,
    document: DocumentModel,
    *,
    allow_fallback: bool,
) -> list[Document]:
    await _update_stage(session, document, "mineru")
    try:
        raw_docs, batch_id = await asyncio.to_thread(
            parse_file_to_documents,
            document.file_path,
            data_id=str(document.id),
        )
        job_result = await session.execute(
            select(ParseJob).where(ParseJob.document_id == document.id).order_by(ParseJob.created_at.desc())
        )
        job = job_result.scalar_one_or_none()
        if job:
            job.external_task_id = batch_id
            await session.commit()
        return raw_docs
    except MinerUError:
        if allow_fallback and supports_python_fallback(document.filename):
            logger.warning("MinerU failed for %s, falling back to Python loader", document.filename)
            await _update_stage(session, document, "python_fallback")
            return await asyncio.to_thread(load_with_python, document.file_path, document.file_type)
        raise


async def _parse_python(session: AsyncSession, document: DocumentModel) -> list[Document]:
    await _update_stage(session, document, "python")
    return await asyncio.to_thread(load_with_python, document.file_path, document.file_type)


async def _parse_asr(session: AsyncSession, document: DocumentModel, audio_path: str | None = None) -> list[Document]:
    await _update_stage(session, document, "asr")
    path = audio_path or document.file_path
    segments = await asyncio.to_thread(transcribe_audio_segments, path)
    docs = [
        Document(
            page_content=seg.text,
            metadata={
                "content_type": "audio_transcript",
                "parse_source": "asr",
                "segment_index": i,
                "start_sec": seg.start_sec,
                "end_sec": seg.end_sec,
            },
        )
        for i, seg in enumerate(segments)
        if seg.text.strip()
    ]
    if not docs:
        raise ValueError("ASR returned empty transcript")
    return docs


async def _parse_image(session: AsyncSession, document: DocumentModel) -> list[Document]:
    docs: list[Document] = []
    try:
        ocr_docs, _ = await asyncio.to_thread(
            parse_file_to_documents,
            document.file_path,
            data_id=str(document.id),
        )
        for d in ocr_docs:
            d.metadata["parse_source"] = "mineru_ocr"
        for d in ocr_docs:
            d.metadata["content_type"] = "ocr"
        docs.extend(ocr_docs)
    except MinerUError as exc:
        logger.warning("MinerU OCR failed for image %s: %s", document.filename, exc)

    await _update_stage(session, document, "vlm")
    try:
        summary = await asyncio.to_thread(summarize_image, document.file_path)
        if summary:
            docs.append(
                Document(
                    page_content=summary,
                    metadata={"content_type": "vlm_summary", "parse_source": "vlm"},
                )
            )
    except Exception as exc:
        logger.warning("VLM summary failed for image %s: %s", document.filename, exc)
        if not docs:
            raise
    return docs


async def _process_video_frame(
    document: DocumentModel,
    idx: int,
    timestamp: float,
    frame_path: str,
) -> list[Document]:
    frame_docs: list[Document] = []
    try:
        frame_docs, _ = await asyncio.to_thread(
            parse_file_to_documents,
            frame_path,
            data_id=f"{document.id}-f{int(timestamp)}",
        )
        for d in frame_docs:
            d.metadata["parse_source"] = "mineru_frame"
        for d in frame_docs:
            d.metadata.update({
                "content_type": "frame_ocr",
                "frame_index": idx,
                "timestamp_sec": timestamp,
            })
    except MinerUError as exc:
        logger.warning("MinerU frame OCR failed at %ss: %s", timestamp, exc)

    try:
        summary = await asyncio.to_thread(summarize_image, frame_path)
        if summary:
            frame_docs.append(
                Document(
                    page_content=summary,
                    metadata={
                        "content_type": "frame_vlm_summary",
                        "frame_index": idx,
                        "timestamp_sec": timestamp,
                        "parse_source": "vlm",
                    },
                )
            )
    except Exception as exc:
        logger.warning("VLM frame summary failed at %ss: %s", timestamp, exc)

    return frame_docs


async def _parse_video(session: AsyncSession, document: DocumentModel) -> list[Document]:
    await _update_stage(session, document, "video_extract")
    assets = await asyncio.to_thread(extract_video_assets, document.file_path)
    docs: list[Document] = []
    try:
        if assets.audio_path:
            docs.extend(await _parse_asr(session, document, assets.audio_path))

        if assets.frame_paths:
            await _update_stage(session, document, "frame_processing")
            semaphore = asyncio.Semaphore(settings.video_frame_concurrency)

            async def process_with_limit(idx: int, timestamp: float, frame_path: str) -> list[Document]:
                async with semaphore:
                    return await _process_video_frame(document, idx, timestamp, frame_path)

            frame_results = await asyncio.gather(
                *[
                    process_with_limit(idx, timestamp, frame_path)
                    for idx, (timestamp, frame_path) in enumerate(assets.frame_paths)
                ]
            )
            for frame_docs in frame_results:
                docs.extend(frame_docs)
    finally:
        cleanup_temp_dir(assets.temp_dir)

    if not docs:
        raise ValueError("No content extracted from video")
    return docs


async def run_parse_pipeline(session: AsyncSession, document: DocumentModel) -> int:
    strategy = classify_file(document.filename)
    if strategy == "python":
        raw_docs = await _parse_python(session, document)
    elif strategy == "mineru":
        raw_docs = await _parse_with_mineru(session, document, allow_fallback=True)
    elif strategy == "asr":
        raw_docs = await _parse_asr(session, document)
    elif strategy == "image":
        raw_docs = await _parse_image(session, document)
    elif strategy == "video":
        raw_docs = await _parse_video(session, document)
    else:
        raise ValueError(f"Unknown parse strategy: {strategy}")

    await _update_stage(session, document, "chunk")
    if not raw_docs:
        raise ValueError("No content extracted from document")

    count = await index_documents(session, document, raw_docs)
    return count

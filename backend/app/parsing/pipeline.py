import asyncio
import logging

from langchain_core.documents import Document
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models import Document as DocumentModel, ParseJob
from app.parsing.asr_service import TranscriptSegment, transcribe_audio_segments
from app.parsing.indexer import index_documents
from app.parsing.mineru_client import MinerUError, parse_file_to_documents
from app.parsing.parse_execution import JobSupersededError, ParseExecutionContext
from app.parsing.python_loaders import load_with_python
from app.parsing.router import classify_file, supports_python_fallback
from app.parsing.frame_planner import PlannedFrame
from app.parsing.video_service import (
    cleanup_temp_dir,
    extract_planned_frames,
    extract_video_audio,
)
from app.parsing.vlm_service import summarize_image

logger = logging.getLogger(__name__)


async def _update_stage(
    session: AsyncSession,
    document: DocumentModel,
    stage: str,
    ctx: ParseExecutionContext,
) -> None:
    await ctx.abort_if_lost(session)
    document.status = "processing"
    document.parse_stage = stage
    await session.commit()


async def _parse_with_mineru(
    session: AsyncSession,
    document: DocumentModel,
    ctx: ParseExecutionContext,
    *,
    allow_fallback: bool,
) -> list[Document]:
    await _update_stage(session, document, "mineru", ctx)
    try:
        raw_docs, batch_id = await asyncio.to_thread(
            parse_file_to_documents,
            document.file_path,
            data_id=str(document.id),
        )
        job_result = await session.execute(select(ParseJob).where(ParseJob.id == ctx.job_id))
        job = job_result.scalar_one_or_none()
        if job:
            job.external_task_id = batch_id
            await session.commit()
        await ctx.abort_if_lost(session)
        return raw_docs
    except MinerUError:
        if allow_fallback and supports_python_fallback(document.filename):
            logger.warning("MinerU failed for %s, falling back to Python loader", document.filename)
            await _update_stage(session, document, "python_fallback", ctx)
            return await asyncio.to_thread(load_with_python, document.file_path, document.file_type)
        raise


async def _parse_python(
    session: AsyncSession,
    document: DocumentModel,
    ctx: ParseExecutionContext,
) -> list[Document]:
    await _update_stage(session, document, "python", ctx)
    return await asyncio.to_thread(load_with_python, document.file_path, document.file_type)


async def _parse_asr(
    session: AsyncSession,
    document: DocumentModel,
    ctx: ParseExecutionContext,
    audio_path: str | None = None,
) -> list[Document]:
    await _update_stage(session, document, "asr", ctx)
    path = audio_path or document.file_path
    segments = await asyncio.to_thread(transcribe_audio_segments, path)
    return _segments_to_documents(segments)


def _segments_to_documents(segments: list[TranscriptSegment]) -> list[Document]:
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


def _segments_to_documents_optional(segments: list[TranscriptSegment]) -> list[Document]:
    try:
        return _segments_to_documents(segments)
    except ValueError:
        return []


async def _parse_image(
    session: AsyncSession,
    document: DocumentModel,
    ctx: ParseExecutionContext,
) -> list[Document]:
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

    await _update_stage(session, document, "vlm", ctx)
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
    plan: PlannedFrame,
    frame_path: str,
) -> list[Document]:
    timestamp = plan.timestamp_sec
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
            meta = {
                "content_type": "frame_ocr",
                "frame_index": idx,
                "timestamp_sec": timestamp,
                "frame_source": plan.source,
            }
            if plan.asr_segment_index is not None:
                meta["asr_segment_index"] = plan.asr_segment_index
                meta["asr_start_sec"] = plan.asr_start_sec
                meta["asr_end_sec"] = plan.asr_end_sec
            d.metadata.update(meta)
    except MinerUError as exc:
        logger.warning("MinerU frame OCR failed at %ss: %s", timestamp, exc)

    if settings.video_vlm_enabled:
        ocr_chars = sum(len(d.page_content) for d in frame_docs)
        if ocr_chars < settings.video_vlm_min_ocr_chars:
            try:
                summary = await asyncio.to_thread(summarize_image, frame_path)
                if summary:
                    vlm_meta = {
                        "content_type": "frame_vlm_summary",
                        "frame_index": idx,
                        "timestamp_sec": timestamp,
                        "parse_source": "vlm",
                        "frame_source": plan.source,
                    }
                    if plan.asr_segment_index is not None:
                        vlm_meta["asr_segment_index"] = plan.asr_segment_index
                        vlm_meta["asr_start_sec"] = plan.asr_start_sec
                        vlm_meta["asr_end_sec"] = plan.asr_end_sec
                    frame_docs.append(
                        Document(page_content=summary, metadata=vlm_meta)
                    )
            except Exception as exc:
                logger.warning("VLM frame summary failed at %ss: %s", timestamp, exc)

    return frame_docs


async def _parse_video(
    session: AsyncSession,
    document: DocumentModel,
    ctx: ParseExecutionContext,
) -> list[Document]:
    await _update_stage(session, document, "video_extract", ctx)
    audio_path, temp_dir = await asyncio.to_thread(extract_video_audio, document.file_path)
    docs: list[Document] = []
    segments: list[TranscriptSegment] = []
    try:
        if audio_path:
            segments = await asyncio.to_thread(transcribe_audio_segments, audio_path)
            docs.extend(_segments_to_documents_optional(segments))
            await ctx.abort_if_lost(session)

        await _update_stage(session, document, "frame_plan", ctx)
        frame_plans = await asyncio.to_thread(
            extract_planned_frames,
            document.file_path,
            temp_dir,
            segments,
        )
        await ctx.abort_if_lost(session)

        if frame_plans:
            await _update_stage(session, document, "frame_processing", ctx)
            semaphore = asyncio.Semaphore(settings.video_frame_concurrency)
            batch_size = max(1, settings.video_frame_concurrency)

            async def process_with_limit(
                idx: int,
                plan: PlannedFrame,
                frame_path: str,
            ) -> list[Document]:
                async with semaphore:
                    return await _process_video_frame(document, idx, plan, frame_path)

            for batch_start in range(0, len(frame_plans), batch_size):
                await ctx.abort_if_lost(session)
                batch = frame_plans[batch_start:batch_start + batch_size]
                frame_results = await asyncio.gather(
                    *[
                        process_with_limit(idx, plan, frame_path)
                        for idx, (plan, frame_path) in enumerate(
                            batch,
                            start=batch_start,
                        )
                    ]
                )
                for frame_docs in frame_results:
                    docs.extend(frame_docs)
    finally:
        cleanup_temp_dir(temp_dir)

    if not docs:
        raise ValueError("No content extracted from video")
    return docs


async def run_parse_pipeline(
    session: AsyncSession,
    document: DocumentModel,
    ctx: ParseExecutionContext,
) -> int:
    strategy = classify_file(document.filename)
    if strategy == "python":
        raw_docs = await _parse_python(session, document, ctx)
    elif strategy == "mineru":
        raw_docs = await _parse_with_mineru(session, document, ctx, allow_fallback=True)
    elif strategy == "asr":
        raw_docs = await _parse_asr(session, document, ctx)
    elif strategy == "image":
        raw_docs = await _parse_image(session, document, ctx)
    elif strategy == "video":
        raw_docs = await _parse_video(session, document, ctx)
    else:
        raise ValueError(f"Unknown parse strategy: {strategy}")

    await _update_stage(session, document, "chunk", ctx)
    if not raw_docs:
        raise ValueError("No content extracted from document")

    count = await index_documents(session, document, raw_docs, ctx)
    return count

import logging
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models import Document, ParseJob
from app.services.vector_store_service import delete_document_vectors

logger = logging.getLogger(__name__)


async def enqueue_parse_job(session: AsyncSession, document_id: uuid.UUID) -> ParseJob:
    job = ParseJob(document_id=document_id, status="pending")
    session.add(job)
    await session.commit()
    await session.refresh(job)
    return job


async def document_exists(session: AsyncSession, document_id: uuid.UUID) -> bool:
    result = await session.execute(select(Document.id).where(Document.id == document_id))
    return result.scalar_one_or_none() is not None


async def is_job_running(session: AsyncSession, job_id: uuid.UUID) -> bool:
    result = await session.execute(select(ParseJob.status).where(ParseJob.id == job_id))
    status = result.scalar_one_or_none()
    return status == "running"


async def enqueue_reparse(session: AsyncSession, document_id: uuid.UUID) -> ParseJob:
    await session.execute(
        update(ParseJob)
        .where(ParseJob.document_id == document_id, ParseJob.status == "running")
        .values(status="cancelled", error_message="Superseded by reparse")
    )
    await session.execute(
        delete(ParseJob).where(
            ParseJob.document_id == document_id,
            ParseJob.status.in_(["pending", "failed"]),
        )
    )
    job = ParseJob(document_id=document_id, status="pending")
    session.add(job)
    await session.commit()
    await session.refresh(job)
    return job


async def claim_next_job(session: AsyncSession) -> ParseJob | None:
    result = await session.execute(
        select(ParseJob)
        .where(ParseJob.status == "pending")
        .order_by(ParseJob.created_at.asc())
        .with_for_update(skip_locked=True)
        .limit(1)
    )
    job = result.scalar_one_or_none()
    if job is None:
        return None

    job.status = "running"
    job.attempts += 1
    job.started_at = datetime.now(timezone.utc)
    await session.commit()
    await session.refresh(job)
    return job


async def complete_job(session: AsyncSession, job: ParseJob) -> None:
    job.status = "completed"
    job.stage = "index"
    job.finished_at = datetime.now(timezone.utc)
    job.error_message = None
    await session.commit()


async def fail_job(session: AsyncSession, job: ParseJob, error: str, *, retry: bool) -> None:
    job.error_message = error
    if retry:
        job.status = "pending"
        job.stage = None
        job.started_at = None
        job.finished_at = None
    else:
        job.status = "failed"
        job.finished_at = datetime.now(timezone.utc)
    await session.commit()


async def finish_job_failure(
    session: AsyncSession,
    job: ParseJob,
    document: Document,
    error: str,
    *,
    retry: bool,
) -> None:
    document.error_message = error
    job.error_message = error
    if retry:
        document.status = "queued"
        document.parse_stage = "queued"
        job.status = "pending"
        job.stage = None
        job.started_at = None
        job.finished_at = None
    else:
        document.status = "failed"
        document.parse_stage = None
        job.status = "failed"
        job.finished_at = datetime.now(timezone.utc)
    await session.commit()


async def reclaim_stale_jobs(session: AsyncSession) -> int:
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=settings.parse_job_stale_timeout_sec)
    result = await session.execute(
        select(ParseJob).where(
            ParseJob.status == "running",
            ParseJob.started_at.isnot(None),
            ParseJob.started_at < cutoff,
        )
    )
    jobs = list(result.scalars().all())
    if not jobs:
        return 0

    for job in jobs:
        doc_result = await session.execute(select(Document).where(Document.id == job.document_id))
        document = doc_result.scalar_one_or_none()

        job.status = "pending"
        job.stage = None
        job.started_at = None

        if document is not None:
            document.status = "queued"
            document.parse_stage = "queued"
            document.error_message = None

        try:
            delete_document_vectors(str(job.document_id))
        except Exception:
            logger.exception(
                "Failed to delete vectors while reclaiming stale job: job_id=%s document_id=%s",
                job.id,
                job.document_id,
            )

        logger.warning(
            "Reclaimed stale parse job: job_id=%s document_id=%s",
            job.id,
            job.document_id,
        )

    await session.commit()
    return len(jobs)

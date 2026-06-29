import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, exists, or_, select, update
from sqlalchemy.orm import aliased
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models import Document, ParseJob
from app.parsing.parse_execution import ParseExecutionContext, get_worker_id
from app.services.vector_store_service import delete_document_vectors_before_generation

logger = logging.getLogger(__name__)


@dataclass
class ClaimedParseJob:
    job: ParseJob
    ctx: ParseExecutionContext


def _lease_cutoff() -> datetime:
    return datetime.now(timezone.utc)


async def _get_document(session: AsyncSession, document_id: uuid.UUID) -> Document | None:
    result = await session.execute(select(Document).where(Document.id == document_id))
    return result.scalar_one_or_none()


def _clear_lease_fields(job: ParseJob) -> None:
    job.lease_token = None
    job.lease_expires_at = None
    job.worker_id = None


async def _create_pending_job(
    session: AsyncSession,
    document_id: uuid.UUID,
    parse_generation: int,
) -> ParseJob:
    job = ParseJob(
        document_id=document_id,
        status="pending",
        parse_generation=parse_generation,
    )
    session.add(job)
    return job


async def enqueue_parse_job(session: AsyncSession, document_id: uuid.UUID) -> ParseJob:
    document = await _get_document(session, document_id)
    if document is None:
        raise ValueError(f"Document not found: {document_id}")

    job = await _create_pending_job(session, document_id, document.active_parse_generation)
    await session.commit()
    await session.refresh(job)
    return job


async def document_exists(session: AsyncSession, document_id: uuid.UUID) -> bool:
    result = await session.execute(select(Document.id).where(Document.id == document_id))
    return result.scalar_one_or_none() is not None


async def enqueue_reparse(session: AsyncSession, document_id: uuid.UUID) -> ParseJob:
    document = await _get_document(session, document_id)
    if document is None:
        raise ValueError(f"Document not found: {document_id}")

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

    document.active_parse_generation += 1
    document.active_job_id = None
    document.status = "queued"
    document.parse_stage = "queued"
    document.error_message = None
    document.chunk_count = 0

    delete_document_vectors_before_generation(str(document_id), document.active_parse_generation)

    job = await _create_pending_job(session, document_id, document.active_parse_generation)
    await session.commit()
    await session.refresh(job)
    return job


async def claim_next_job(session: AsyncSession) -> ClaimedParseJob | None:
    now = _lease_cutoff()
    running_job = aliased(ParseJob)
    live_running = exists(
        select(running_job.id).where(
            running_job.document_id == ParseJob.document_id,
            running_job.status == "running",
            running_job.lease_expires_at.isnot(None),
            running_job.lease_expires_at > now,
        )
    )

    result = await session.execute(
        select(ParseJob)
        .where(ParseJob.status == "pending", ~live_running)
        .order_by(ParseJob.created_at.asc())
        .with_for_update(skip_locked=True)
        .limit(1)
    )
    job = result.scalar_one_or_none()
    if job is None:
        return None

    document = await _get_document(session, job.document_id)
    if document is None:
        await fail_job(session, job, "Document not found", retry=False)
        return None

    if job.parse_generation != document.active_parse_generation:
        job.status = "cancelled"
        job.error_message = "Stale pending job generation"
        await session.commit()
        return None

    worker_id = get_worker_id()
    lease_token = uuid.uuid4()
    lease_expires_at = now + timedelta(seconds=settings.parse_job_lease_ttl_sec)

    job.status = "running"
    job.attempts += 1
    job.started_at = now
    job.lease_token = lease_token
    job.lease_expires_at = lease_expires_at
    job.worker_id = worker_id
    job.stage = "claimed"
    job.error_message = None

    document.active_job_id = job.id
    document.status = "processing"
    document.parse_stage = "claimed"

    await session.commit()
    await session.refresh(job)

    ctx = ParseExecutionContext.from_job(job, worker_id)
    return ClaimedParseJob(job=job, ctx=ctx)


async def complete_job(session: AsyncSession, job: ParseJob, ctx: ParseExecutionContext) -> None:
    if not await ctx.is_still_owner(session):
        return
    job.status = "completed"
    job.stage = "index"
    job.finished_at = datetime.now(timezone.utc)
    job.error_message = None
    _clear_lease_fields(job)
    await session.commit()


async def fail_job(session: AsyncSession, job: ParseJob, error: str, *, retry: bool) -> None:
    job.error_message = error
    if retry:
        job.status = "pending"
        job.stage = None
        job.started_at = None
        job.finished_at = None
        _clear_lease_fields(job)
    else:
        job.status = "failed"
        job.finished_at = datetime.now(timezone.utc)
        _clear_lease_fields(job)
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
        document.active_job_id = None
        job.status = "pending"
        _clear_lease_fields(job)
    else:
        document.status = "failed"
        document.parse_stage = None
        document.active_job_id = None
        job.status = "failed"
        job.finished_at = datetime.now(timezone.utc)
        _clear_lease_fields(job)
    await session.commit()


async def reclaim_stale_jobs(session: AsyncSession) -> int:
    now = _lease_cutoff()
    grace_cutoff = now - timedelta(seconds=settings.parse_job_stale_grace_sec)
    started_cutoff = now - timedelta(seconds=settings.parse_job_stale_timeout_sec)

    result = await session.execute(
        select(ParseJob).where(
            ParseJob.status == "running",
            or_(
                # 租约非空，且租约过期超过宽限期parse_job_stale_grace_sec
                (ParseJob.lease_expires_at.isnot(None)) & (ParseJob.lease_expires_at < grace_cutoff),
                # 租约空，且开始时间非空，且开始时间超过超时时间parse_job_stale_timeout_sec
                (ParseJob.lease_expires_at.is_(None)) & (ParseJob.started_at.isnot(None)) & (ParseJob.started_at < started_cutoff),
            ),
        )
    )
    jobs = list(result.scalars().all())
    if not jobs:
        return 0

    reclaimed = 0
    for job in jobs:
        document = await _get_document(session, job.document_id)
        if document is None:
            job.status = "cancelled"
            job.error_message = "Document not found during stale reclaim"
            _clear_lease_fields(job)
            reclaimed += 1
            continue

        job.status = "cancelled"
        job.error_message = "Lease expired / worker stale"
        _clear_lease_fields(job)

        document.active_parse_generation += 1
        document.active_job_id = None
        document.status = "queued"
        document.parse_stage = "queued"
        document.error_message = None

        try:
            delete_document_vectors_before_generation(
                str(job.document_id),
                document.active_parse_generation,
            )
        except Exception:
            logger.exception(
                "Failed to delete vectors while reclaiming stale job: job_id=%s document_id=%s",
                job.id,
                job.document_id,
            )

        if settings.parse_job_stale_auto_retry:
            await _create_pending_job(session, job.document_id, document.active_parse_generation)

        logger.warning(
            "Reclaimed stale parse job: job_id=%s document_id=%s new_generation=%s",
            job.id,
            job.document_id,
            document.active_parse_generation,
        )
        reclaimed += 1

    await session.commit()
    return reclaimed

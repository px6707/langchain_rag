import asyncio
import logging

from sqlalchemy import select

from app.config import settings
from app.db.migrate import run_migrations
from app.database import async_session
from app.models import Document, ParseJob
from app.parsing.parse_execution import JobSupersededError, ParseExecutionContext
from app.parsing.pipeline import run_parse_pipeline
from app.services.parse_job_service import (
    claim_next_job,
    complete_job,
    document_exists,
    fail_job,
    finish_job_failure,
    reclaim_stale_jobs,
)

logger = logging.getLogger(__name__)


async def _heartbeat_loop(ctx: ParseExecutionContext) -> None:
    while True:
        await asyncio.sleep(settings.parse_job_heartbeat_sec)
        async with async_session() as session:
            renewed = await ctx.renew_lease(session)
            if not renewed:
                break


async def process_one_job() -> bool:
    async with async_session() as session:
        claimed = await claim_next_job(session)
        if claimed is None:
            return False

        job = claimed.job
        ctx = claimed.ctx

        result = await session.execute(select(Document).where(Document.id == job.document_id))
        document = result.scalar_one_or_none()
        if document is None:
            await fail_job(session, job, "Document not found", retry=False)
            return True

        heartbeat_task = asyncio.create_task(_heartbeat_loop(ctx))
        try:
            job.stage = "pipeline"
            await session.commit()
            await run_parse_pipeline(session, document, ctx)
            await session.refresh(job)
            if not await document_exists(session, document.id):
                logger.warning("Document deleted during parsing: document_id=%s", document.id)
                await fail_job(session, job, "Document deleted during parsing", retry=False)
                return True
            if await ctx.is_still_owner(session):
                await complete_job(session, job, ctx)
                logger.info("Parse job completed: document_id=%s", document.id)
            else:
                logger.warning("Parse job superseded after pipeline: job_id=%s", job.id)
        except JobSupersededError:
            logger.warning("Parse job superseded during pipeline: job_id=%s", job.id)
        except Exception as exc:
            logger.exception("Parse job failed: document_id=%s", document.id)
            await session.refresh(job)
            if job.status == "cancelled":
                return True
            if await document_exists(session, document.id):
                if await ctx.is_still_owner(session):
                    retry = job.attempts < settings.parse_max_attempts
                    await finish_job_failure(session, job, document, str(exc), retry=retry)
            else:
                await fail_job(session, job, str(exc), retry=False)
        finally:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
        return True


async def worker_loop() -> None:
    logger.info("Parse worker started, poll_interval=%ss", settings.parse_worker_poll_sec)
    while True:
        async with async_session() as session:
            reclaimed = await reclaim_stale_jobs(session)
            if reclaimed:
                logger.info("Reclaimed %s stale parse job(s)", reclaimed)

        processed = await process_one_job()
        if not processed:
            await asyncio.sleep(settings.parse_worker_poll_sec)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    run_migrations()
    asyncio.run(worker_loop())


if __name__ == "__main__":
    main()

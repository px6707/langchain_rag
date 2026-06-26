import asyncio
import logging

from sqlalchemy import select

from app.config import settings
from app.database import async_session
from app.models import Document, ParseJob
from app.parsing.pipeline import run_parse_pipeline
from app.services.parse_job_service import (
    claim_next_job,
    complete_job,
    document_exists,
    fail_job,
    finish_job_failure,
    is_job_running,
    reclaim_stale_jobs,
)
from app.services.vector_store_service import delete_document_vectors

logger = logging.getLogger(__name__)


async def process_one_job() -> bool:
    async with async_session() as session:
        job = await claim_next_job(session)
        if job is None:
            return False

        result = await session.execute(select(Document).where(Document.id == job.document_id))
        document = result.scalar_one_or_none()
        if document is None:
            await fail_job(session, job, "Document not found", retry=False)
            return True

        try:
            job.stage = "pipeline"
            await session.commit()
            await run_parse_pipeline(session, document)
            await session.refresh(job)
            if not await is_job_running(session, job.id):
                logger.warning("Parse job cancelled or superseded: job_id=%s", job.id)
                delete_document_vectors(str(document.id))
                return True
            if not await document_exists(session, document.id):
                logger.warning("Document deleted during parsing: document_id=%s", document.id)
                delete_document_vectors(str(document.id))
                await fail_job(session, job, "Document deleted during parsing", retry=False)
                return True
            await complete_job(session, job)
            logger.info("Parse job completed: document_id=%s", document.id)
        except Exception as exc:
            logger.exception("Parse job failed: document_id=%s", document.id)
            await session.refresh(job)
            if job.status == "cancelled":
                delete_document_vectors(str(document.id))
                return True
            if await document_exists(session, document.id):
                retry = job.attempts < settings.parse_max_attempts
                await finish_job_failure(session, job, document, str(exc), retry=retry)
            else:
                delete_document_vectors(str(document.id))
                await fail_job(session, job, str(exc), retry=False)
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
    asyncio.run(worker_loop())


if __name__ == "__main__":
    main()

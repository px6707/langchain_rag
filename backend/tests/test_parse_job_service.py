import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from app.models import Document, ParseJob
from app.services.parse_job_service import enqueue_reparse, finish_job_failure, reclaim_stale_jobs


def test_finish_job_failure_retry():
    job = ParseJob(document_id=uuid.uuid4(), status="running", attempts=1)
    document = Document(
        id=job.document_id,
        filename="a.pdf",
        file_path="/tmp/a.pdf",
        file_type="pdf",
        status="processing",
        parse_stage="mineru",
    )
    session = AsyncMock()

    asyncio.run(finish_job_failure(session, job, document, "transient error", retry=True))

    assert document.status == "queued"
    assert document.parse_stage == "queued"
    assert document.error_message == "transient error"
    assert job.status == "pending"
    assert job.stage is None
    assert job.started_at is None
    assert job.finished_at is None
    assert job.error_message == "transient error"
    session.commit.assert_awaited_once()


def test_finish_job_failure_final():
    job = ParseJob(document_id=uuid.uuid4(), status="running", attempts=3)
    document = Document(
        id=job.document_id,
        filename="a.pdf",
        file_path="/tmp/a.pdf",
        file_type="pdf",
        status="processing",
        parse_stage="mineru",
    )
    session = AsyncMock()

    asyncio.run(finish_job_failure(session, job, document, "fatal error", retry=False))

    assert document.status == "failed"
    assert document.parse_stage is None
    assert document.error_message == "fatal error"
    assert job.status == "failed"
    assert job.finished_at is not None
    assert job.error_message == "fatal error"
    session.commit.assert_awaited_once()


def test_reclaim_stale_jobs_resets_running():
    doc_id = uuid.uuid4()
    stale_started = datetime.now(timezone.utc) - timedelta(hours=3)
    job = ParseJob(
        id=uuid.uuid4(),
        document_id=doc_id,
        status="running",
        stage="mineru",
        started_at=stale_started,
    )
    document = Document(
        id=doc_id,
        filename="a.pdf",
        file_path="/tmp/a.pdf",
        file_type="pdf",
        status="processing",
        parse_stage="mineru",
        error_message="old error",
    )

    jobs_result = MagicMock()
    jobs_result.scalars.return_value.all.return_value = [job]
    doc_result = MagicMock()
    doc_result.scalar_one_or_none.return_value = document

    session = AsyncMock()
    session.execute = AsyncMock(side_effect=[jobs_result, doc_result])

    with patch("app.services.parse_job_service.delete_document_vectors") as mock_delete:
        count = asyncio.run(reclaim_stale_jobs(session))

    assert count == 1
    assert job.status == "pending"
    assert job.stage is None
    assert job.started_at is None
    assert document.status == "queued"
    assert document.parse_stage == "queued"
    assert document.error_message is None
    mock_delete.assert_called_once_with(str(doc_id))
    session.commit.assert_awaited_once()


def test_enqueue_reparse_cancels_running_job():
    doc_id = uuid.uuid4()
    session = AsyncMock()

    asyncio.run(enqueue_reparse(session, doc_id))

    assert session.execute.await_count == 2
    session.add.assert_called_once()
    session.commit.assert_awaited_once()


def test_reclaim_stale_jobs_no_stale():
    jobs_result = MagicMock()
    jobs_result.scalars.return_value.all.return_value = []
    session = AsyncMock()
    session.execute = AsyncMock(return_value=jobs_result)

    count = asyncio.run(reclaim_stale_jobs(session))

    assert count == 0
    session.commit.assert_not_awaited()

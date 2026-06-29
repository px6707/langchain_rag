import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from app.config import settings
from app.models import Document, ParseJob
from app.services.parse_job_service import enqueue_reparse, finish_job_failure, reclaim_stale_jobs


def test_finish_job_failure_retry():
    job = ParseJob(document_id=uuid.uuid4(), status="running", attempts=1, parse_generation=1)
    document = Document(
        id=job.document_id,
        filename="a.pdf",
        file_path="/tmp/a.pdf",
        file_type="pdf",
        status="processing",
        parse_stage="mineru",
        active_parse_generation=1,
    )
    session = AsyncMock()

    asyncio.run(finish_job_failure(session, job, document, "transient error", retry=True))

    assert document.status == "queued"
    assert document.parse_stage == "queued"
    assert document.error_message == "transient error"
    assert document.active_job_id is None
    assert job.status == "pending"
    assert job.lease_token is None
    assert job.error_message == "transient error"
    session.commit.assert_awaited_once()


def test_finish_job_failure_final():
    job = ParseJob(document_id=uuid.uuid4(), status="running", attempts=3, parse_generation=1)
    document = Document(
        id=job.document_id,
        filename="a.pdf",
        file_path="/tmp/a.pdf",
        file_type="pdf",
        status="processing",
        parse_stage="mineru",
        active_parse_generation=1,
    )
    session = AsyncMock()

    asyncio.run(finish_job_failure(session, job, document, "fatal error", retry=False))

    assert document.status == "failed"
    assert document.active_job_id is None
    assert job.status == "failed"
    assert job.finished_at is not None
    session.commit.assert_awaited_once()


def test_reclaim_stale_jobs_cancels_and_enqueues_new():
    doc_id = uuid.uuid4()
    stale_started = datetime.now(timezone.utc) - timedelta(hours=3)
    job = ParseJob(
        id=uuid.uuid4(),
        document_id=doc_id,
        status="running",
        stage="mineru",
        started_at=stale_started,
        parse_generation=1,
        lease_expires_at=stale_started,
    )
    document = Document(
        id=doc_id,
        filename="a.pdf",
        file_path="/tmp/a.pdf",
        file_type="pdf",
        status="processing",
        parse_stage="mineru",
        error_message="old error",
        active_parse_generation=1,
    )

    jobs_result = MagicMock()
    jobs_result.scalars.return_value.all.return_value = [job]
    doc_result = MagicMock()
    doc_result.scalar_one_or_none.return_value = document

    session = MagicMock()
    session.execute = AsyncMock(side_effect=[jobs_result, doc_result])
    session.commit = AsyncMock()

    with patch("app.services.parse_job_service.delete_document_vectors_before_generation") as mock_delete:
        with patch.object(settings, "parse_job_stale_auto_retry", True):
            count = asyncio.run(reclaim_stale_jobs(session))

    assert count == 1
    assert job.status == "cancelled"
    assert job.lease_token is None
    assert document.status == "queued"
    assert document.active_parse_generation == 2
    assert document.active_job_id is None
    mock_delete.assert_called_once_with(str(doc_id), 2)
    session.add.assert_called_once()
    session.commit.assert_awaited_once()


def test_enqueue_reparse_bumps_generation():
    doc_id = uuid.uuid4()
    document = Document(
        id=doc_id,
        filename="a.pdf",
        file_path="/tmp/a.pdf",
        file_type="pdf",
        active_parse_generation=1,
    )
    doc_result = MagicMock()
    doc_result.scalar_one_or_none.return_value = document
    session = MagicMock()
    session.execute = AsyncMock(side_effect=[doc_result, MagicMock(), MagicMock()])
    session.commit = AsyncMock()
    session.refresh = AsyncMock()

    with patch("app.services.parse_job_service.delete_document_vectors_before_generation"):
        asyncio.run(enqueue_reparse(session, doc_id))

    assert document.active_parse_generation == 2
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

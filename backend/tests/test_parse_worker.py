import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

from app.models import Document, ParseJob
from app.parsing.parse_execution import JobSupersededError, ParseExecutionContext
from app.worker import parse_worker


def _ctx_for(doc_id: uuid.UUID, job_id: uuid.UUID) -> ParseExecutionContext:
    return ParseExecutionContext(
        job_id=job_id,
        document_id=doc_id,
        lease_token=uuid.uuid4(),
        parse_generation=1,
        worker_id="test:1",
    )


def test_process_one_job_skips_complete_when_document_deleted():
    doc_id = uuid.uuid4()
    job_id = uuid.uuid4()
    job = ParseJob(id=job_id, document_id=doc_id, status="running", attempts=1, parse_generation=1)
    ctx = _ctx_for(doc_id, job_id)
    document = Document(
        id=doc_id,
        user_id=uuid.uuid4(),
        filename="a.txt",
        file_path="/tmp/a.txt",
        file_type="txt",
        active_parse_generation=1,
    )

    session = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)

    doc_result = MagicMock()
    doc_result.scalar_one_or_none.return_value = document
    session.execute = AsyncMock(return_value=doc_result)

    from app.services.parse_job_service import ClaimedParseJob

    with (
        patch("app.worker.parse_worker.async_session", return_value=session),
        patch(
            "app.worker.parse_worker.claim_next_job",
            new_callable=AsyncMock,
            return_value=ClaimedParseJob(job=job, ctx=ctx),
        ),
        patch("app.worker.parse_worker.run_parse_pipeline", new_callable=AsyncMock),
        patch("app.worker.parse_worker._heartbeat_loop", new_callable=AsyncMock),
        patch.object(ParseExecutionContext, "is_still_owner", new_callable=AsyncMock, return_value=True),
        patch("app.worker.parse_worker.document_exists", new_callable=AsyncMock, return_value=False),
        patch("app.worker.parse_worker.fail_job", new_callable=AsyncMock) as mock_fail,
        patch("app.worker.parse_worker.complete_job", new_callable=AsyncMock) as mock_complete,
    ):
        result = asyncio.run(parse_worker.process_one_job())

    assert result is True
    mock_fail.assert_awaited_once()
    mock_complete.assert_not_awaited()


def test_process_one_job_skips_complete_when_superseded():
    doc_id = uuid.uuid4()
    job_id = uuid.uuid4()
    job = ParseJob(id=job_id, document_id=doc_id, status="running", attempts=1, parse_generation=1)
    ctx = _ctx_for(doc_id, job_id)
    document = Document(
        id=doc_id,
        user_id=uuid.uuid4(),
        filename="a.txt",
        file_path="/tmp/a.txt",
        file_type="txt",
        active_parse_generation=1,
    )

    session = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)

    doc_result = MagicMock()
    doc_result.scalar_one_or_none.return_value = document
    session.execute = AsyncMock(return_value=doc_result)

    from app.services.parse_job_service import ClaimedParseJob

    with (
        patch("app.worker.parse_worker.async_session", return_value=session),
        patch(
            "app.worker.parse_worker.claim_next_job",
            new_callable=AsyncMock,
            return_value=ClaimedParseJob(job=job, ctx=ctx),
        ),
        patch("app.worker.parse_worker.run_parse_pipeline", new_callable=AsyncMock),
        patch("app.worker.parse_worker._heartbeat_loop", new_callable=AsyncMock),
        patch.object(ParseExecutionContext, "is_still_owner", new_callable=AsyncMock, return_value=False),
        patch("app.worker.parse_worker.complete_job", new_callable=AsyncMock) as mock_complete,
    ):
        result = asyncio.run(parse_worker.process_one_job())

    assert result is True
    mock_complete.assert_not_awaited()

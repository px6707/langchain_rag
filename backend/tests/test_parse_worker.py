import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

from app.models import Document, ParseJob
from app.worker import parse_worker


def test_process_one_job_skips_complete_when_document_deleted():
    doc_id = uuid.uuid4()
    job = ParseJob(id=uuid.uuid4(), document_id=doc_id, status="running", attempts=1)
    document = Document(
        id=doc_id,
        filename="a.txt",
        file_path="/tmp/a.txt",
        file_type="txt",
    )

    session = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)

    doc_result = MagicMock()
    doc_result.scalar_one_or_none.return_value = document
    session.execute = AsyncMock(return_value=doc_result)

    with (
        patch("app.worker.parse_worker.async_session", return_value=session),
        patch("app.worker.parse_worker.claim_next_job", new_callable=AsyncMock, return_value=job),
        patch("app.worker.parse_worker.run_parse_pipeline", new_callable=AsyncMock),
        patch("app.worker.parse_worker.is_job_running", new_callable=AsyncMock, return_value=True),
        patch("app.worker.parse_worker.document_exists", new_callable=AsyncMock, return_value=False),
        patch("app.worker.parse_worker.delete_document_vectors") as mock_delete,
        patch("app.worker.parse_worker.fail_job", new_callable=AsyncMock) as mock_fail,
        patch("app.worker.parse_worker.complete_job", new_callable=AsyncMock) as mock_complete,
    ):
        result = asyncio.run(parse_worker.process_one_job())

    assert result is True
    mock_delete.assert_called_once_with(str(doc_id))
    mock_fail.assert_awaited_once()
    mock_complete.assert_not_awaited()


def test_process_one_job_skips_complete_when_job_cancelled():
    doc_id = uuid.uuid4()
    job = ParseJob(id=uuid.uuid4(), document_id=doc_id, status="cancelled", attempts=1)
    document = Document(
        id=doc_id,
        filename="a.txt",
        file_path="/tmp/a.txt",
        file_type="txt",
    )

    session = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)

    doc_result = MagicMock()
    doc_result.scalar_one_or_none.return_value = document
    session.execute = AsyncMock(return_value=doc_result)

    with (
        patch("app.worker.parse_worker.async_session", return_value=session),
        patch("app.worker.parse_worker.claim_next_job", new_callable=AsyncMock, return_value=job),
        patch("app.worker.parse_worker.run_parse_pipeline", new_callable=AsyncMock),
        patch("app.worker.parse_worker.is_job_running", new_callable=AsyncMock, return_value=False),
        patch("app.worker.parse_worker.delete_document_vectors") as mock_delete,
        patch("app.worker.parse_worker.complete_job", new_callable=AsyncMock) as mock_complete,
    ):
        result = asyncio.run(parse_worker.process_one_job())

    assert result is True
    mock_delete.assert_called_once_with(str(doc_id))
    mock_complete.assert_not_awaited()

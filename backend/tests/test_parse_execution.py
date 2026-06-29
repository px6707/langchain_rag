import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.models import Document, ParseJob
from app.parsing.parse_execution import JobSupersededError, ParseExecutionContext


def _make_ctx(
    job_id: uuid.UUID | None = None,
    document_id: uuid.UUID | None = None,
    generation: int = 1,
) -> ParseExecutionContext:
    return ParseExecutionContext(
        job_id=job_id or uuid.uuid4(),
        document_id=document_id or uuid.uuid4(),
        lease_token=uuid.uuid4(),
        parse_generation=generation,
        worker_id="test:1",
    )


def test_abort_if_lost_raises_when_not_owner():
    ctx = _make_ctx()
    session = AsyncMock()
    session.execute = AsyncMock(return_value=MagicMock(one_or_none=MagicMock(return_value=None)))

    with pytest.raises(JobSupersededError):
        asyncio.run(ctx.abort_if_lost(session))


def test_is_still_owner_all_conditions_met():
    job_id = uuid.uuid4()
    doc_id = uuid.uuid4()
    token = uuid.uuid4()
    ctx = ParseExecutionContext(
        job_id=job_id,
        document_id=doc_id,
        lease_token=token,
        parse_generation=2,
        worker_id="test:1",
    )
    now = datetime.now(timezone.utc)
    row = ("running", token, 2, job_id, 2, now + timedelta(seconds=60))
    session = AsyncMock()
    session.execute = AsyncMock(return_value=MagicMock(one_or_none=MagicMock(return_value=row)))

    assert asyncio.run(ctx.is_still_owner(session)) is True


def test_is_still_owner_false_when_status_not_running():
    ctx = _make_ctx()
    row = ("cancelled", ctx.lease_token, 1, ctx.job_id, 1, datetime.now(timezone.utc))
    session = AsyncMock()
    session.execute = AsyncMock(return_value=MagicMock(one_or_none=MagicMock(return_value=row)))

    assert asyncio.run(ctx.is_still_owner(session)) is False

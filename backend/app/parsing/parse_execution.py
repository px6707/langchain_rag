"""Parse job execution ownership (lease + generation)."""

from __future__ import annotations

import os
import socket
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models import Document, ParseJob


class JobSupersededError(Exception):
    """Raised when the worker no longer owns the parse execution."""


def get_worker_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


@dataclass
class ParseExecutionContext:
    job_id: uuid.UUID
    document_id: uuid.UUID
    lease_token: uuid.UUID
    parse_generation: int
    worker_id: str

    @classmethod
    def from_job(cls, job: ParseJob, worker_id: str) -> ParseExecutionContext:
        if job.lease_token is None:
            raise ValueError("Parse job has no lease_token")
        return cls(
            job_id=job.id,
            document_id=job.document_id,
            lease_token=job.lease_token,
            parse_generation=job.parse_generation,
            worker_id=worker_id,
        )

    async def is_still_owner(self, session: AsyncSession) -> bool:
        now = datetime.now(timezone.utc)
        result = await session.execute(
            select(ParseJob.status, ParseJob.lease_token, ParseJob.parse_generation, Document.active_job_id, Document.active_parse_generation, ParseJob.lease_expires_at)
            .join(Document, Document.id == ParseJob.document_id)
            .where(ParseJob.id == self.job_id)
        )
        row = result.one_or_none()
        if row is None:
            return False
        status, lease_token, parse_generation, active_job_id, active_generation, lease_expires_at = row
        if status != "running":
            return False
        if lease_token != self.lease_token:
            return False
        if parse_generation != self.parse_generation:
            return False
        if active_job_id != self.job_id:
            return False
        if active_generation != self.parse_generation:
            return False
        if lease_expires_at is None or lease_expires_at <= now:
            return False
        return True

    async def renew_lease(self, session: AsyncSession) -> bool:
        now = datetime.now(timezone.utc)
        new_expiry = now + timedelta(seconds=settings.parse_job_lease_ttl_sec)
        result = await session.execute(
            update(ParseJob)
            .where(
                ParseJob.id == self.job_id,
                ParseJob.lease_token == self.lease_token,
                ParseJob.status == "running",
            )
            .values(lease_expires_at=new_expiry)
            .returning(ParseJob.id)
        )
        if result.scalar_one_or_none() is None:
            return False
        await session.commit()
        return True

    async def abort_if_lost(self, session: AsyncSession) -> None:
        if not await self.is_still_owner(session):
            raise JobSupersededError(
                f"Parse execution superseded: job_id={self.job_id} generation={self.parse_generation}"
            )

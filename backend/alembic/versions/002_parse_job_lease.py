"""parse job lease and generation

Revision ID: 002
Revises: 001
Create Date: 2026-06-22

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "002"
down_revision: Union[str, Sequence[str], None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("active_parse_generation", sa.Integer(), nullable=False, server_default="1"),
    )
    op.add_column(
        "documents",
        sa.Column("active_job_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "parse_jobs",
        sa.Column("parse_generation", sa.Integer(), nullable=False, server_default="1"),
    )
    op.add_column(
        "parse_jobs",
        sa.Column("lease_token", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "parse_jobs",
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "parse_jobs",
        sa.Column("worker_id", sa.String(length=128), nullable=True),
    )
    op.create_index(
        "uq_parse_jobs_document_running",
        "parse_jobs",
        ["document_id"],
        unique=True,
        postgresql_where=sa.text("status = 'running'"),
    )


def downgrade() -> None:
    op.drop_index("uq_parse_jobs_document_running", table_name="parse_jobs")
    op.drop_column("parse_jobs", "worker_id")
    op.drop_column("parse_jobs", "lease_expires_at")
    op.drop_column("parse_jobs", "lease_token")
    op.drop_column("parse_jobs", "parse_generation")
    op.drop_column("documents", "active_job_id")
    op.drop_column("documents", "active_parse_generation")

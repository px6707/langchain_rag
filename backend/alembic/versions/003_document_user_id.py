"""document user_id isolation

Revision ID: 003
Revises: 002
Create Date: 2026-06-22

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "003"
down_revision: Union[str, Sequence[str], None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.execute(
        """
        UPDATE documents d
        SET user_id = (
            SELECT u.id FROM users u
            WHERE u.is_admin = true
            ORDER BY u.created_at ASC
            LIMIT 1
        )
        WHERE d.user_id IS NULL
        """
    )
    conn = op.get_bind()
    remaining = conn.execute(
        sa.text("SELECT COUNT(*) FROM documents WHERE user_id IS NULL")
    ).scalar_one()
    if remaining:
        raise RuntimeError(
            "Cannot backfill documents.user_id: no admin user found for legacy rows"
        )
    op.alter_column("documents", "user_id", nullable=False)
    op.create_foreign_key(
        "fk_documents_user_id",
        "documents",
        "users",
        ["user_id"],
        ["id"],
    )
    op.create_index(op.f("ix_documents_user_id"), "documents", ["user_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_documents_user_id"), table_name="documents")
    op.drop_constraint("fk_documents_user_id", "documents", type_="foreignkey")
    op.drop_column("documents", "user_id")

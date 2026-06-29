"""Per-request retrieval ACL context (user document isolation)."""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalUserContext:
    user_id: str
    is_admin: bool


_retrieval_ctx: ContextVar[RetrievalUserContext | None] = ContextVar(
    "retrieval_user_context",
    default=None,
)


def set_retrieval_user_context(user_id: str, *, is_admin: bool) -> Token:
    return _retrieval_ctx.set(RetrievalUserContext(user_id=user_id, is_admin=is_admin))


def reset_retrieval_user_context(token: Token) -> None:
    _retrieval_ctx.reset(token)


def get_retrieval_user_filter() -> str | None:
    """Return user_id for ES filter, or None when admin / unset (no filter)."""
    ctx = _retrieval_ctx.get()
    if ctx is None or ctx.is_admin:
        return None
    return ctx.user_id

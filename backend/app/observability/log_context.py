"""Structured logging context aligned with LangSmith trace ids."""

from __future__ import annotations

import logging
from contextvars import ContextVar, Token

_trace_id: ContextVar[str | None] = ContextVar("log_trace_id", default=None)
_session_id: ContextVar[str | None] = ContextVar("log_session_id", default=None)
_user_id: ContextVar[str | None] = ContextVar("log_user_id", default=None)


def init_log_context(
    *,
    trace_id: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
) -> tuple[Token, Token, Token]:
    return (
        _trace_id.set(trace_id),
        _session_id.set(session_id),
        _user_id.set(user_id),
    )


def reset_log_context(
    token_trace: Token,
    token_session: Token,
    token_user: Token,
) -> None:
    _trace_id.reset(token_trace)
    _session_id.reset(token_session)
    _user_id.reset(token_user)


def update_log_trace_id(trace_id: str | None) -> None:
    _trace_id.set(trace_id)


class TraceContextFilter(logging.Filter):
    """Attach trace/session/user ids to every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = _trace_id.get() or "-"  # type: ignore[attr-defined]
        record.session_id = _session_id.get() or "-"  # type: ignore[attr-defined]
        record.user_id = _user_id.get() or "-"  # type: ignore[attr-defined]
        return True

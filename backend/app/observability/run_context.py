"""Capture LangSmith root run id during agent execution."""

from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Any
from uuid import UUID

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult

_root_run_id: ContextVar[str | None] = ContextVar("langsmith_root_run_id", default=None)
_trace_id: ContextVar[str | None] = ContextVar("langsmith_trace_id", default=None)


def _normalize_run_id(run_id: UUID | str | None) -> str | None:
    if run_id is None:
        return None
    return str(run_id)


class RootRunCaptureHandler(BaseCallbackHandler):
    """Records the outermost chain run id (root trace) for feedback and metadata patch."""

    def __init__(self) -> None:
        self._depth = 0

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        if parent_run_id is None and _root_run_id.get() is None:
            rid = _normalize_run_id(run_id)
            if rid:
                _root_run_id.set(rid)
                _trace_id.set(rid)

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        if parent_run_id is None and _root_run_id.get() is None:
            rid = _normalize_run_id(run_id)
            if rid:
                _root_run_id.set(rid)
                _trace_id.set(rid)

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        return

    def on_chain_end(self, outputs: dict[str, Any], *, run_id: UUID, **kwargs: Any) -> None:
        return


def make_root_run_capture_handler() -> RootRunCaptureHandler:
    return RootRunCaptureHandler()


def get_captured_run_ids() -> tuple[str | None, str | None]:
    return _root_run_id.get(), _trace_id.get()


def reset_run_capture(token_run: Token, token_trace: Token) -> None:
    _root_run_id.reset(token_run)
    _trace_id.reset(token_trace)


def init_run_capture() -> tuple[Token, Token]:
    token_run = _root_run_id.set(None)
    token_trace = _trace_id.set(None)
    return token_run, token_trace

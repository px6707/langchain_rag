"""LangSmith named spans for RAG pipeline stages."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from app.config import settings
from app.observability.langsmith import is_langsmith_enabled

try:
    from langsmith import traceable as _langsmith_traceable
except ImportError:  # pragma: no cover
    _langsmith_traceable = None

F = TypeVar("F", bound=Callable[..., Any])


def trace_stage(
    name: str,
    *,
    run_type: str = "chain",
    extra_tags: list[str] | None = None,
) -> Callable[[F], F]:
    """Decorator: wrap a function in a LangSmith child span when tracing is enabled."""

    def decorator(fn: F) -> F:
        if not is_langsmith_enabled() or not settings.langsmith_stage_tracing_enabled:
            return fn
        if _langsmith_traceable is None:
            return fn
        try:
            tags = ["rag_stage", name]
            if extra_tags:
                tags.extend(extra_tags)
            return _langsmith_traceable(name=name, run_type=run_type, tags=tags)(fn)  # type: ignore[return-value]
        except Exception:
            return fn

    return decorator

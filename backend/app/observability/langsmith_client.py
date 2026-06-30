"""LangSmith SDK helpers for metadata patch and user feedback."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from app.observability.langsmith import is_langsmith_enabled

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_langsmith_client():
    from langsmith import Client

    return Client()


def patch_run_metadata(run_id: str, metadata: dict[str, Any]) -> None:
    if not is_langsmith_enabled() or not run_id:
        return
    try:
        client = get_langsmith_client()
        client.update_run(run_id, extra=metadata)
    except Exception as exc:
        logger.warning("Failed to patch LangSmith run metadata run_id=%s: %s", run_id, exc)


def submit_user_feedback(
    *,
    run_id: str,
    trace_id: str | None,
    score: float,
    comment: str | None = None,
    key: str = "user_thumbs_down",
    extra: dict[str, Any] | None = None,
) -> None:
    if not is_langsmith_enabled():
        raise RuntimeError("LangSmith tracing is not enabled")
    if not run_id:
        raise ValueError("run_id is required")

    effective_trace = trace_id or run_id
    try:
        client = get_langsmith_client()
        client.create_feedback(
            run_id=run_id,
            key=key,
            score=score,
            comment=comment,
            trace_id=effective_trace,
            extra=extra,
        )
    except Exception as exc:
        logger.exception("Failed to submit LangSmith feedback run_id=%s", run_id)
        raise RuntimeError(f"Failed to submit feedback: {exc}") from exc

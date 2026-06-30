"""Chat feedback routes — LangSmith user feedback."""

import logging

from fastapi import APIRouter, Depends, HTTPException

from app.auth.dependencies import get_current_user
from app.models import User
from app.observability.langsmith import is_langsmith_enabled
from app.observability.langsmith_client import submit_user_feedback
from app.schemas.feedback import (
    FEEDBACK_KIND_TO_KEY,
    ChatFeedbackRequest,
    ChatFeedbackResponse,
)

router = APIRouter(
    prefix="/api/chat",
    tags=["chat"],
    dependencies=[Depends(get_current_user)],
)

logger = logging.getLogger(__name__)


@router.post("/feedback", response_model=ChatFeedbackResponse)
async def submit_chat_feedback(
    request: ChatFeedbackRequest,
    current_user: User = Depends(get_current_user),
):
    if not is_langsmith_enabled():
        raise HTTPException(status_code=503, detail="LangSmith tracing is not enabled")

    user_id = str(current_user.id)
    key, score = FEEDBACK_KIND_TO_KEY[request.kind]
    extra: dict = {"user_id": user_id}
    if request.session_id:
        extra["session_id"] = request.session_id
    if request.reason:
        extra["reason"] = request.reason

    try:
        submit_user_feedback(
            run_id=request.run_id,
            trace_id=request.trace_id,
            score=score,
            comment=request.comment,
            key=key,
            extra=extra,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    logger.info(
        "chat feedback submitted: user_id=%s kind=%s reason=%s run_id=%s trace_id=%s",
        user_id,
        request.kind,
        request.reason,
        request.run_id,
        request.trace_id,
    )
    return ChatFeedbackResponse(ok=True)

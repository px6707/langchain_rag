import json
import logging

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse

from app.auth.dependencies import get_current_user
from app.models import User
from app.schemas import (
    ChatHistoryResponse,
    ChatInterruptResponse,
    ChatMessageResponse,
    ChatRequest,
    ChatResumeRequest,
    HITLRequestPayload,
    TodoItemResponse,
)
from app.services.rag_service import RAGService
from app.tools.loader import get_loaded_tools

router = APIRouter(
    prefix="/api/chat",
    tags=["chat"],
    dependencies=[Depends(get_current_user)],
)

logger = logging.getLogger(__name__)


def _sse_response(event_generator):
    return StreamingResponse(
        event_generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("")
async def chat(request: ChatRequest, current_user: User = Depends(get_current_user)):
    service = RAGService()
    user_id = str(current_user.id)
    logger.info(
        "chat start: session_id=%s user_id=%s message=%r",
        request.session_id,
        user_id,
        request.message,
    )

    async def event_generator():
        try:
            async for event in service.chat_stream(
                request.session_id,
                user_id,
                request.message,
                is_admin=current_user.is_admin,
            ):
                if event.get("type") == "error":
                    logger.error(
                        "chat stream error event: session_id=%s message=%s",
                        request.session_id,
                        event.get("message"),
                    )
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            logger.info("chat done: session_id=%s", request.session_id)
        except Exception:
            logger.exception("chat stream failed: session_id=%s user_id=%s", request.session_id, user_id)
            error_event = {"type": "error", "message": "RAG 服务错误，请查看后端日志"}
            yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"

    return _sse_response(event_generator())


@router.post("/resume")
async def resume_chat(request: ChatResumeRequest, current_user: User = Depends(get_current_user)):
    service = RAGService()
    user_id = str(current_user.id)
    decisions = [decision.model_dump(exclude_none=True) for decision in request.decisions]

    async def event_generator():
        try:
            async for event in service.resume_chat_stream(
                request.session_id,
                user_id,
                decisions,
                is_admin=current_user.is_admin,
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as e:
            error_event = {"type": "error", "message": f"RAG 服务错误: {e}"}
            yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"

    return _sse_response(event_generator())


@router.get("/interrupt", response_model=ChatInterruptResponse)
async def get_interrupt(
    session_id: str = Query(..., min_length=1),
    current_user: User = Depends(get_current_user),
):
    service = RAGService()
    payload = await service.get_pending_interrupt(session_id, str(current_user.id))
    if payload is None:
        return ChatInterruptResponse(request=None)
    return ChatInterruptResponse(request=HITLRequestPayload(**payload))


@router.get("/history", response_model=ChatHistoryResponse)
async def get_history(
    session_id: str = Query(..., min_length=1),
    current_user: User = Depends(get_current_user),
):
    service = RAGService()
    messages, todos = await service.get_history(session_id, str(current_user.id))

    response_messages = [
        ChatMessageResponse(
            id=msg["id"],
            role=msg["role"],
            content=msg["content"],
            sources=msg.get("sources"),
            grounding=msg.get("grounding"),
            tool_calls=msg.get("tool_calls"),
            created_at=msg.get("created_at"),
            run_id=msg.get("run_id"),
            trace_id=msg.get("trace_id"),
        )
        for msg in messages
    ]

    todo_items = [TodoItemResponse(**item) for item in todos] if todos else None
    return ChatHistoryResponse(messages=response_messages, todos=todo_items)


@router.get("/tools")
async def list_tools():
    tools = get_loaded_tools()
    return {"tools": [{"name": tool.name, "description": tool.description} for tool in tools]}

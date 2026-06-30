import asyncio
from contextvars import ContextVar
import logging
from typing import Annotated, Any, Literal, NotRequired

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
    OmitFromInput,
)
from langgraph.runtime import Runtime

from app.services.retrieval_router import plan_retrieval
from app.services.retrieval_service import search_with_plan
from app.observability.turn_trace import get_turn_trace

logger = logging.getLogger(__name__)

BASE_SYSTEM_APPENDIX = (
    "你是基于文档内容的问答助手。优先使用检索到的文档内容回答；"
    "若无相关信息，请诚实说明。回答简洁准确，使用中文。"
    "当提供了检索文档时，每个事实性陈述句末须附引用标记 [document_id#chunk_index]（与上下文中的标记一致）；"
    "禁止编造未出现在上下文中的引用 ID；无检索上下文时不要求引用。"
)

CITATION_CONTEXT_APPENDIX = (
    "引用格式说明：上下文每段以 [document_id#chunk_index] 开头，"
    "回答中引用该段内容时须使用相同标记，例如 [550e8400-e29b-41d4-a716-446655440000#2]。"
)

_pending_sources: ContextVar[list[dict[str, str]] | None] = ContextVar("pending_sources", default=None)
_pending_chunks: ContextVar[list[Document] | None] = ContextVar("pending_chunks", default=None)
_turn_cache: ContextVar[tuple[str, list, str | None, list[Document]] | None] = ContextVar(
    "turn_cache", default=None
)


def _merge_sources(left: dict[str, list[dict]] | None, right: dict[str, list[dict]] | None) -> dict[str, list[dict]]:
    merged = dict(left or {})
    merged.update(right or {})
    return merged


def _merge_grounding(left: dict[str, dict] | None, right: dict[str, dict] | None) -> dict[str, dict]:
    merged = dict(left or {})
    merged.update(right or {})
    return merged


def _merge_message_traces(left: dict[str, dict] | None, right: dict[str, dict] | None) -> dict[str, dict]:
    merged = dict(left or {})
    merged.update(right or {})
    return merged


class TodoItem(TypedDict):
    content: str
    status: Literal["pending", "in_progress", "completed"]


class RAGAgentState(AgentState):
    message_sources: NotRequired[Annotated[dict[str, list[dict]], _merge_sources]]
    message_grounding: NotRequired[Annotated[dict[str, dict], _merge_grounding]]
    message_traces: NotRequired[Annotated[dict[str, dict], _merge_message_traces]]
    todos: Annotated[NotRequired[list[TodoItem]], OmitFromInput]


def _get_last_user_message(messages: list) -> str | None:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            content = msg.content
            return content if isinstance(content, str) else str(content)
    return None


def _human_message_key(messages: list) -> str | None:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            if msg.id:
                return msg.id
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            return f"hash:{hash(content)}"
    return None


def _inject_context(request: ModelRequest[None], context: str | None) -> ModelRequest[None]:
    if not context:
        return request
    base = request.system_message.content if request.system_message else BASE_SYSTEM_APPENDIX
    if isinstance(base, list):
        base = str(base)
    new_system = SystemMessage(
        content=(
            f"{base}\n\n{CITATION_CONTEXT_APPENDIX}\n\n"
            f"以下是与当前问题相关的文档内容（仅供参考）：\n{context}"
        )
    )
    return request.override(system_message=new_system)


class RetrievalMiddleware(AgentMiddleware[AgentState[Any], None, Any]):
    state_schema = RAGAgentState
    tools = ()

    def __init__(self, llm: ChatOpenAI) -> None:
        self._llm = llm

    async def awrap_model_call(
        self,
        request: ModelRequest[None],
        handler,
    ) -> ModelResponse:
        query = _get_last_user_message(request.messages)
        turn_key = _human_message_key(request.messages)

        if not query:
            _pending_sources.set(None)
            _pending_chunks.set(None)
            return await handler(request)

        cached = _turn_cache.get()
        if cached and turn_key and cached[0] == turn_key:
            sources, context, docs = cached[1], cached[2], cached[3]
            turn_trace = get_turn_trace()
            if turn_trace is not None:
                turn_trace.set_chunks(docs)
        else:
            plan = await asyncio.to_thread(plan_retrieval, request.messages, llm=self._llm)
            turn_trace = get_turn_trace()
            if turn_trace is not None:
                turn_trace.set_retrieval_plan(plan)
            if plan.action == "skip":
                sources, context, docs = [], None, []
            else:
                sources, context, docs, retrieval_trace = await asyncio.to_thread(
                    search_with_plan, plan, llm=self._llm
                )
                if turn_trace is not None:
                    turn_trace.set_retrieval_trace(retrieval_trace)

            if turn_trace is not None:
                turn_trace.set_chunks(docs)

            if turn_key:
                _turn_cache.set((turn_key, sources, context, docs))

        _pending_sources.set([s.model_dump(exclude_none=True) for s in sources] if sources else None)
        _pending_chunks.set(docs if docs else None)
        request = _inject_context(request, context)
        return await handler(request)

    async def aafter_model(self, state: AgentState[Any], runtime: Runtime[None]) -> dict[str, Any] | None:
        sources = _pending_sources.get() or []
        if not sources:
            return None

        messages = state.get("messages", [])
        if not messages:
            return None

        last_msg = messages[-1]
        if isinstance(last_msg, AIMessage) and last_msg.id:
            return {"message_sources": {last_msg.id: sources}}
        return None

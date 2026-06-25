import asyncio
from contextvars import ContextVar
import logging
from typing import Annotated, Any, Literal, NotRequired

from elasticsearch import NotFoundError

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from typing_extensions import TypedDict

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
    OmitFromInput,
)
from langgraph.runtime import Runtime

from app.services.retrieval_service import search_relevant_docs

logger = logging.getLogger(__name__)

BASE_SYSTEM_APPENDIX = (
    "你是基于文档内容的问答助手。优先使用检索到的文档内容回答；"
    "若无相关信息，请诚实说明。回答简洁准确，使用中文。"
)

_pending_sources: ContextVar[list[dict[str, str]] | None] = ContextVar("pending_sources", default=None)


def _merge_sources(left: dict[str, list[dict]] | None, right: dict[str, list[dict]] | None) -> dict[str, list[dict]]:
    merged = dict(left or {})
    merged.update(right or {})
    return merged


class TodoItem(TypedDict):
    content: str
    status: Literal["pending", "in_progress", "completed"]


class RAGAgentState(AgentState):
    message_sources: NotRequired[Annotated[dict[str, list[dict]], _merge_sources]]
    todos: Annotated[NotRequired[list[TodoItem]], OmitFromInput]


def _get_last_user_message(messages: list) -> str | None:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            content = msg.content
            return content if isinstance(content, str) else str(content)
    return None


class RetrievalMiddleware(AgentMiddleware[AgentState[Any], None, Any]):
    state_schema = RAGAgentState
    tools = ()

    async def awrap_model_call(
        self,
        request: ModelRequest[None],
        handler,
    ) -> ModelResponse:
        query = _get_last_user_message(request.messages)
        sources = []
        context: str | None = None

        if query:
            sources, context = await asyncio.to_thread(search_relevant_docs, query)

        _pending_sources.set([s.model_dump(exclude_none=True) for s in sources] if sources else None)

        if context:
            base = request.system_message.content if request.system_message else BASE_SYSTEM_APPENDIX
            if isinstance(base, list):
                base = str(base)
            new_system = SystemMessage(
                content=f"{base}\n\n以下是与当前问题相关的文档内容（仅供参考）：\n{context}"
            )
            request = request.override(system_message=new_system)

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

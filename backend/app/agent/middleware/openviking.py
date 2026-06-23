from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)

from app.openviking.memory_service import find_user_memories


@dataclass(frozen=True)
class OpenVikingRequestContext:
    user_id: str
    session_id: str


_ov_request_context: ContextVar[OpenVikingRequestContext | None] = ContextVar(
    "ov_request_context",
    default=None,
)


def set_ov_request_context(user_id: str, session_id: str) -> Token:
    return _ov_request_context.set(OpenVikingRequestContext(user_id=user_id, session_id=session_id))


def reset_ov_request_context(token: Token) -> None:
    _ov_request_context.reset(token)


def get_ov_request_context() -> OpenVikingRequestContext | None:
    return _ov_request_context.get()


def _get_last_user_message(messages: list) -> str | None:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            content = msg.content
            return content if isinstance(content, str) else str(content)
    return None


class OpenVikingMemoryMiddleware(AgentMiddleware[AgentState[Any], None, Any]):
    tools = ()

    async def awrap_model_call(
        self,
        request: ModelRequest[None],
        handler,
    ) -> ModelResponse:
        ov_ctx = get_ov_request_context()
        query = _get_last_user_message(request.messages)
        memory_context: str | None = None

        if ov_ctx and query:
            memory_context = await find_user_memories(ov_ctx.user_id, query)

        if memory_context:
            base = request.system_message.content if request.system_message else ""
            if isinstance(base, list):
                base = str(base)
            new_system = SystemMessage(
                content=(
                    f"{base}\n\n"
                    "以下是与当前问题相关的用户长期记忆（含偏好与画像，仅供参考）：\n"
                    f"{memory_context}"
                )
            )
            request = request.override(system_message=new_system)

        return await handler(request)

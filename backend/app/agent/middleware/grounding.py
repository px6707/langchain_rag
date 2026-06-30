import asyncio
import logging
from typing import Any

from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from langchain.agents.middleware.types import AgentMiddleware, AgentState

from app.agent.middleware.retrieval import _pending_chunks, RAGAgentState
from app.services.grounding_service import _extract_answer_text, validate_grounding
from app.observability.turn_trace import get_turn_trace

logger = logging.getLogger(__name__)


class GroundingMiddleware(AgentMiddleware[RAGAgentState, None, Any]):
    state_schema = RAGAgentState
    tools = ()

    async def aafter_model(self, state: AgentState[Any], runtime: Runtime[None]) -> dict[str, Any] | None:
        chunks = _pending_chunks.get() or []
        if not chunks:
            return None

        messages = state.get("messages", [])
        if not messages:
            return None

        last_msg = messages[-1]
        if not isinstance(last_msg, AIMessage) or not last_msg.id:
            return None

        answer = _extract_answer_text(last_msg.content)
        if not answer:
            return None

        result = await asyncio.to_thread(validate_grounding, answer, chunks)
        turn_trace = get_turn_trace()
        if turn_trace is not None:
            turn_trace.set_grounding(result)
        if result.status == "skipped":
            return None

        return {"message_grounding": {last_msg.id: result.model_dump()}}

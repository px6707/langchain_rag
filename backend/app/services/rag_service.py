from collections.abc import AsyncIterator
import logging
from typing import Any

from langchain.agents.middleware.pii import PIIDetectionError
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from app.agent.factory import get_agent
from app.agent.hitl_utils import extract_hitl_request
from app.agent.message_utils import (
    convert_messages_to_history,
    extract_chunk_content,
    extract_grounding_from_turn,
    extract_sources_from_turn,
    extract_tool_end_from_message,
    extract_tool_starts_from_message,
)
from app.agent.middleware.openviking import reset_ov_request_context, set_ov_request_context
from app.observability.langsmith import is_langsmith_enabled
from app.openviking.session_service import sync_turn_to_openviking
from app.schemas import SourceInfo

logger = logging.getLogger(__name__)


class RAGService:
    def _agent_config(
        self,
        session_id: str,
        user_id: str,
        *,
        run_name: str = "rag_chat",
    ) -> RunnableConfig:
        config: RunnableConfig = {"configurable": {"thread_id": session_id}}
        if is_langsmith_enabled():
            config["run_name"] = run_name
            config["tags"] = ["rag", run_name]
            config["metadata"] = {"session_id": session_id, "user_id": user_id}
        return config

    @staticmethod
    def _extract_todos(state: Any) -> list[dict] | None:
        if state is None or not state.values:
            return None
        todos = state.values.get("todos")
        if not todos:
            return None
        return list(todos)

    @staticmethod
    def _extract_last_assistant_text(messages: list) -> str | None:
        for msg in reversed(messages):
            if not isinstance(msg, AIMessage):
                continue
            content = msg.content
            if isinstance(content, str):
                text = content.strip()
            elif isinstance(content, list):
                parts: list[str] = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(str(block.get("text", "")))
                    elif isinstance(block, str):
                        parts.append(block)
                text = "".join(parts).strip()
            else:
                text = str(content).strip() if content else ""
            if text:
                return text
        return None

    def _events_from_message(
        self,
        msg: Any,
        metadata: dict,
        seen_tool_ids: set[str],
    ) -> list[dict]:
        events: list[dict] = []
        node = metadata.get("langgraph_node", "")

        if node == "tools" and isinstance(msg, ToolMessage):
            events.append(extract_tool_end_from_message(msg))
            return events

        if node not in {"model", "agent"}:
            return events

        if isinstance(msg, AIMessageChunk):
            for start in extract_tool_starts_from_message(msg):
                if start["id"] in seen_tool_ids:
                    continue
                seen_tool_ids.add(start["id"])
                events.append(start)

            token = extract_chunk_content(msg)
            if token:
                events.append({"type": "token", "content": token})

        return events

    async def _run_agent_stream(
        self,
        session_id: str,
        user_id: str,
        input_data: Any,
        *,
        run_name: str = "rag_chat",
        user_message: str | None = None,
    ) -> AsyncIterator[dict]:
        agent = get_agent()
        config = self._agent_config(session_id, user_id, run_name=run_name)
        seen_tool_ids: set[str] = set()
        ov_token = set_ov_request_context(user_id, session_id)
        logger.info("agent stream start: session_id=%s user_id=%s run_name=%s", session_id, user_id, run_name)

        try:
            async for msg, metadata in agent.astream(
                input_data,
                config=config,
                stream_mode="messages",
            ):
                if not isinstance(metadata, dict):
                    continue
                for event in self._events_from_message(msg, metadata, seen_tool_ids):
                    yield event
                    if event.get("type") == "tool_end" and event.get("name") == "write_todos":
                        state = await agent.aget_state(config)
                        todos = self._extract_todos(state)
                        if todos:
                            yield {"type": "todos_update", "todos": todos}
        except PIIDetectionError as exc:
            logger.warning("PII detected: session_id=%s error=%s", session_id, exc)
            yield {"type": "error", "message": f"检测到敏感信息: {exc}"}
            return
        except Exception as exc:
            logger.exception(
                "agent stream failed: session_id=%s user_id=%s error=%s",
                session_id,
                user_id,
                exc,
            )
            yield {"type": "error", "message": f"RAG 服务错误: {exc}"}
            return
        finally:
            reset_ov_request_context(ov_token)

        state = await agent.aget_state(config)
        hitl_request = extract_hitl_request(state)
        if hitl_request:
            if user_message:
                await sync_turn_to_openviking(
                    user_id,
                    session_id,
                    user_message=user_message,
                )
            yield {"type": "hitl_request", "request": hitl_request}
            return

        todos = self._extract_todos(state)
        if todos:
            yield {"type": "todos_update", "todos": todos}

        if state and state.values:
            messages = state.values.get("messages", [])
            message_sources = state.values.get("message_sources", {})
            message_grounding = state.values.get("message_grounding", {})
            grounding = extract_grounding_from_turn(messages, message_grounding)
            if grounding:
                yield {
                    "type": "grounding",
                    "grounding": grounding.model_dump(),
                }
            sources = extract_sources_from_turn(messages, message_sources)
            yield {
                "type": "sources",
                "sources": [s.model_dump() for s in sources],
            }
            assistant_message = self._extract_last_assistant_text(messages)
            await sync_turn_to_openviking(
                user_id,
                session_id,
                user_message=user_message,
                assistant_message=assistant_message,
            )

        yield {"type": "done"}

    async def chat_stream(
        self,
        session_id: str,
        user_id: str,
        message: str,
    ) -> AsyncIterator[dict]:
        async for event in self._run_agent_stream(
            session_id,
            user_id,
            {"messages": [HumanMessage(content=message)]},
            user_message=message,
        ):
            yield event

    async def resume_chat_stream(
        self,
        session_id: str,
        user_id: str,
        decisions: list[dict],
    ) -> AsyncIterator[dict]:
        async for event in self._run_agent_stream(
            session_id,
            user_id,
            Command(resume={"decisions": decisions}),
            run_name="rag_resume",
        ):
            yield event

    async def get_pending_interrupt(self, session_id: str, user_id: str) -> dict | None:
        agent = get_agent()
        config = self._agent_config(session_id, user_id)
        state = await agent.aget_state(config)
        return extract_hitl_request(state)

    async def get_session_todos(self, session_id: str, user_id: str) -> list[dict] | None:
        agent = get_agent()
        config = self._agent_config(session_id, user_id)
        state = await agent.aget_state(config)
        return self._extract_todos(state)

    async def get_history(
        self,
        session_id: str,
        user_id: str,
    ) -> tuple[list[dict], list[dict] | None]:
        agent = get_agent()
        config = self._agent_config(session_id, user_id)
        state = await agent.aget_state(config)

        if not state or not state.values:
            return [], None

        messages = state.values.get("messages", [])
        message_sources = state.values.get("message_sources", {})
        message_grounding = state.values.get("message_grounding", {})
        history = convert_messages_to_history(messages, message_sources, message_grounding)
        todos = self._extract_todos(state)
        return history, todos

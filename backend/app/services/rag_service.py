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
from app.config import settings
from app.observability.langsmith import is_langsmith_enabled
from app.observability.langsmith_client import patch_run_metadata
from app.observability.log_context import init_log_context, reset_log_context, update_log_trace_id
from app.observability.run_context import (
    get_captured_run_ids,
    init_run_capture,
    make_root_run_capture_handler,
    reset_run_capture,
)
from app.observability.turn_trace import get_turn_trace, init_turn_trace, reset_turn_trace
from app.openviking.session_service import sync_turn_to_openviking
from app.schemas import SourceInfo
from app.services.retrieval_context import reset_retrieval_user_context, set_retrieval_user_context

logger = logging.getLogger(__name__)


class RAGService:
    @staticmethod
    def _last_assistant_message_id(messages: list) -> str | None:
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.id:
                return msg.id
        return None

    @staticmethod
    def _format_hitl_decisions(decisions: list[dict]) -> list[dict]:
        formatted: list[dict] = []
        for decision in decisions:
            entry: dict = {"decision": decision.get("type", "unknown")}
            edited = decision.get("edited_action")
            if isinstance(edited, dict) and edited.get("name"):
                entry["tool_name"] = edited["name"]
            if decision.get("message"):
                entry["message"] = decision["message"]
            formatted.append(entry)
        return formatted

    async def _persist_message_trace(
        self,
        agent,
        config: RunnableConfig,
        *,
        run_id: str,
        trace_id: str | None,
    ) -> None:
        state = await agent.aget_state(config)
        if not state or not state.values:
            return
        assistant_id = self._last_assistant_message_id(state.values.get("messages", []))
        if not assistant_id:
            return
        effective_trace = trace_id or run_id
        await agent.aupdate_state(
            config,
            {
                "message_traces": {
                    assistant_id: {"run_id": run_id, "trace_id": effective_trace},
                }
            },
        )

    async def _emit_trace_metadata(
        self,
        agent,
        config: RunnableConfig,
        *,
        run_id: str | None,
        trace_id: str | None,
        turn_metadata: dict | None,
    ) -> dict | None:
        if not run_id:
            return None
        if is_langsmith_enabled() and turn_metadata:
            patch_run_metadata(run_id, turn_metadata)
        await self._persist_message_trace(agent, config, run_id=run_id, trace_id=trace_id)
        effective_trace = trace_id or run_id
        update_log_trace_id(effective_trace)
        return {
            "type": "trace",
            "run_id": run_id,
            "trace_id": effective_trace,
        }

    def _record_tool_event(self, event: dict) -> None:
        turn_trace = get_turn_trace()
        if turn_trace is None:
            return
        event_type = event.get("type")
        if event_type == "tool_start":
            turn_trace.record_tool_start(
                str(event.get("id", "")),
                str(event.get("name", "unknown")),
                event.get("args") if isinstance(event.get("args"), str) else None,
            )
        elif event_type == "tool_end":
            output = event.get("output")
            output_str = output if isinstance(output, str) else str(output) if output is not None else None
            is_error = bool(output_str and ("error" in output_str.lower() or "exception" in output_str.lower()))
            turn_trace.record_tool_end(
                str(event.get("id", "")),
                str(event.get("name", "unknown")),
                output=output_str,
                is_error=is_error,
            )

    def _agent_config(
        self,
        session_id: str,
        user_id: str,
        *,
        run_name: str = "rag_chat",
        is_admin: bool = False,
        callbacks: list | None = None,
    ) -> RunnableConfig:
        config: RunnableConfig = {"configurable": {"thread_id": session_id}}
        if is_langsmith_enabled():
            config["run_name"] = run_name
            role_tag = "admin" if is_admin else "user"
            config["tags"] = ["rag", run_name, f"env:{settings.app_env}", role_tag]
            config["metadata"] = {
                "session_id": session_id,
                "user_id": user_id,
                "is_admin": is_admin,
                "env": settings.app_env,
            }
            if callbacks:
                config["callbacks"] = callbacks
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
        is_admin: bool = False,
        hitl_decisions: list[dict] | None = None,
    ) -> AsyncIterator[dict]:
        agent = get_agent()
        run_capture = make_root_run_capture_handler()
        config = self._agent_config(
            session_id,
            user_id,
            run_name=run_name,
            is_admin=is_admin,
            callbacks=[run_capture] if is_langsmith_enabled() else None,
        )
        seen_tool_ids: set[str] = set()
        ov_token = set_ov_request_context(user_id, session_id)
        retrieval_token = set_retrieval_user_context(user_id, is_admin=is_admin)
        turn_trace_token = init_turn_trace(
            user_id=user_id,
            session_id=session_id,
            is_admin=is_admin,
            env=settings.app_env,
        )
        if hitl_decisions:
            turn_trace = get_turn_trace()
            if turn_trace is not None:
                turn_trace.set_hitl_decisions(hitl_decisions)
        log_tokens = init_log_context(session_id=session_id, user_id=user_id)
        run_token_run, run_token_trace = init_run_capture()
        logger.info("agent stream start: session_id=%s user_id=%s run_name=%s", session_id, user_id, run_name)

        captured_run_id: str | None = None
        captured_trace_id: str | None = None
        turn_metadata: dict | None = None

        try:
            async for msg, metadata in agent.astream(
                input_data,
                config=config,
                stream_mode="messages",
            ):
                if not isinstance(metadata, dict):
                    continue
                for event in self._events_from_message(msg, metadata, seen_tool_ids):
                    self._record_tool_event(event)
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
            captured_run_id, captured_trace_id = get_captured_run_ids()
            turn_trace = get_turn_trace()
            if turn_trace is not None:
                turn_metadata = turn_trace.to_metadata()
            reset_retrieval_user_context(retrieval_token)
            reset_ov_request_context(ov_token)
            reset_turn_trace(turn_trace_token)
            reset_run_capture(run_token_run, run_token_trace)
            reset_log_context(*log_tokens)

        state = await agent.aget_state(config)
        hitl_request = extract_hitl_request(state)
        if hitl_request:
            if user_message:
                await sync_turn_to_openviking(
                    user_id,
                    session_id,
                    user_message=user_message,
                )
            trace_event = await self._emit_trace_metadata(
                agent,
                config,
                run_id=captured_run_id,
                trace_id=captured_trace_id,
                turn_metadata=turn_metadata,
            )
            if trace_event:
                yield trace_event
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

        run_id, trace_id = captured_run_id, captured_trace_id
        trace_event = await self._emit_trace_metadata(
            agent,
            config,
            run_id=run_id,
            trace_id=trace_id,
            turn_metadata=turn_metadata,
        )
        if trace_event:
            yield trace_event

        yield {"type": "done"}

    async def chat_stream(
        self,
        session_id: str,
        user_id: str,
        message: str,
        *,
        is_admin: bool = False,
    ) -> AsyncIterator[dict]:
        async for event in self._run_agent_stream(
            session_id,
            user_id,
            {"messages": [HumanMessage(content=message)]},
            user_message=message,
            is_admin=is_admin,
        ):
            yield event

    async def resume_chat_stream(
        self,
        session_id: str,
        user_id: str,
        decisions: list[dict],
        *,
        is_admin: bool = False,
    ) -> AsyncIterator[dict]:
        async for event in self._run_agent_stream(
            session_id,
            user_id,
            Command(resume={"decisions": decisions}),
            run_name="rag_resume",
            is_admin=is_admin,
            hitl_decisions=self._format_hitl_decisions(decisions),
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
        message_traces = state.values.get("message_traces", {})
        history = convert_messages_to_history(
            messages,
            message_sources,
            message_grounding,
            message_traces,
        )
        todos = self._extract_todos(state)
        return history, todos

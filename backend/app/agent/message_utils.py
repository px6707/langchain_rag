import json
import uuid
from datetime import datetime, timezone

from langchain_core.messages import AIMessage, AIMessageChunk, AnyMessage, HumanMessage, ToolMessage

from app.schemas import GroundingResultResponse, SourceInfo, ToolCallInfo


def extract_chunk_content(msg: AIMessageChunk) -> str:
    content = msg.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content) if content else ""


def serialize_tool_args(args, tool_name: str | None = None) -> str | None:
    if args is None:
        return None
    if isinstance(args, dict):
        args = redact_tool_args(tool_name, args)
    if isinstance(args, str):
        return args
    try:
        return json.dumps(args, ensure_ascii=False)
    except TypeError:
        return str(args)


def redact_tool_args(tool_name: str | None, args: dict) -> dict:
    if tool_name != "send_email" or not isinstance(args, dict):
        return args
    redacted = dict(args)
    if redacted.get("smtp_password"):
        redacted["smtp_password"] = "[REDACTED]"
    return redacted


def _redact_tool_args_string(tool_name: str | None, args: str | None) -> str | None:
    if not args or tool_name != "send_email":
        return args
    try:
        parsed = json.loads(args)
    except json.JSONDecodeError:
        return args
    if isinstance(parsed, dict):
        return serialize_tool_args(parsed, tool_name)
    return args


def extract_tool_starts_from_message(msg: AIMessage | AIMessageChunk) -> list[dict]:
    starts: list[dict] = []
    seen_ids: set[str] = set()

    for tc in getattr(msg, "tool_calls", None) or []:
        tool_id = tc.get("id")
        name = tc.get("name")
        if not tool_id or not name or tool_id in seen_ids:
            continue
        seen_ids.add(tool_id)
        starts.append({
            "type": "tool_start",
            "id": tool_id,
            "name": name,
            "args": serialize_tool_args(tc.get("args"), name),
        })

    accumulated: dict[str, dict] = {}
    for chunk in getattr(msg, "tool_call_chunks", None) or []:
        tool_id = chunk.get("id")
        if not tool_id:
            continue
        entry = accumulated.setdefault(tool_id, {"id": tool_id, "name": None, "args": ""})
        if chunk.get("name"):
            entry["name"] = chunk["name"]
        if chunk.get("args"):
            entry["args"] += chunk["args"]

    for tool_id, entry in accumulated.items():
        if tool_id in seen_ids or not entry.get("name"):
            continue
        seen_ids.add(tool_id)
        tool_name = entry["name"]
        starts.append({
            "type": "tool_start",
            "id": tool_id,
            "name": tool_name,
            "args": _redact_tool_args_string(tool_name, entry["args"] or None),
        })

    return starts


def extract_tool_end_from_message(msg: ToolMessage) -> dict:
    content = msg.content if isinstance(msg.content, str) else str(msg.content)
    return {
        "type": "tool_end",
        "id": msg.tool_call_id,
        "name": msg.name or "unknown",
        "output": content,
    }


def _tool_call_info_from_dict(data: dict) -> ToolCallInfo:
    return ToolCallInfo(
        id=data["id"],
        name=data["name"],
        args=data.get("args"),
        output=data.get("output"),
        status=data.get("status", "completed"),
    )


def _aggregate_tool_calls(messages: list[AnyMessage]) -> list[ToolCallInfo]:
    pending: dict[str, dict] = {}

    for msg in messages:
        if isinstance(msg, AIMessage):
            for tc in msg.tool_calls or []:
                tool_id = tc.get("id")
                name = tc.get("name")
                if not tool_id or not name:
                    continue
                pending[tool_id] = {
                    "id": tool_id,
                    "name": name,
                    "args": serialize_tool_args(tc.get("args"), name),
                    "output": pending.get(tool_id, {}).get("output"),
                    "status": pending.get(tool_id, {}).get("status", "completed"),
                }
        elif isinstance(msg, ToolMessage):
            tool_id = msg.tool_call_id
            if tool_id in pending:
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                pending[tool_id]["output"] = content
                pending[tool_id]["status"] = "completed"
            else:
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                pending[tool_id] = {
                    "id": tool_id,
                    "name": msg.name or "unknown",
                    "args": None,
                    "output": content,
                    "status": "completed",
                }

    return [_tool_call_info_from_dict(item) for item in pending.values()]


def extract_last_ai_content(messages: list[AnyMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            content = msg.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(block.get("text", ""))
                    elif isinstance(block, str):
                        parts.append(block)
                return "".join(parts)
            return str(content)
    return ""


def extract_sources_from_turn(
    messages: list[AnyMessage],
    message_sources: dict[str, list[dict]] | None,
) -> list[SourceInfo]:
    if not message_sources:
        return []

    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.id and msg.id in message_sources:
            return [SourceInfo(**item) for item in message_sources[msg.id]]

    return []


def extract_grounding_from_turn(
    messages: list[AnyMessage],
    message_grounding: dict[str, dict] | None,
) -> GroundingResultResponse | None:
    if not message_grounding:
        return None

    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.id and msg.id in message_grounding:
            return GroundingResultResponse(**message_grounding[msg.id])

    return None


def convert_messages_to_history(
    messages: list[AnyMessage],
    message_sources: dict[str, list[dict]] | None = None,
    message_grounding: dict[str, dict] | None = None,
    message_traces: dict[str, dict] | None = None,
) -> list[dict]:
    history: list[dict] = []
    turn_messages: list[AnyMessage] = []

    def flush_assistant_turn() -> None:
        nonlocal turn_messages
        if not turn_messages:
            return

        content = ""
        assistant_id = str(uuid.uuid4())
        sources: list[SourceInfo] | None = None
        grounding: GroundingResultResponse | None = None
        tool_calls = _aggregate_tool_calls(turn_messages)

        for msg in turn_messages:
            if isinstance(msg, AIMessage):
                if msg.id:
                    assistant_id = msg.id
                msg_content = msg.content
                if isinstance(msg_content, list):
                    text_parts = []
                    for block in msg_content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif isinstance(block, str):
                            text_parts.append(block)
                    if text_parts:
                        content = "".join(text_parts)
                elif isinstance(msg_content, str) and msg_content:
                    content = msg_content
                elif msg_content and not isinstance(msg_content, str):
                    content = str(msg_content)

                if msg.id and message_sources and msg.id in message_sources:
                    sources = [SourceInfo(**s) for s in message_sources[msg.id]]

                if msg.id and message_grounding and msg.id in message_grounding:
                    grounding = GroundingResultResponse(**message_grounding[msg.id])

        if not content and not tool_calls:
            turn_messages = []
            return

        history.append({
            "id": assistant_id,
            "role": "assistant",
            "content": content,
            "sources": sources,
            "grounding": grounding,
            "tool_calls": tool_calls or None,
            "created_at": datetime.now(timezone.utc),
            **(
                {
                    "run_id": message_traces[assistant_id]["run_id"],
                    "trace_id": message_traces[assistant_id].get("trace_id"),
                }
                if message_traces and assistant_id in message_traces
                else {}
            ),
        })
        turn_messages = []

    for msg in messages:
        if isinstance(msg, HumanMessage):
            flush_assistant_turn()
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            history.append({
                "id": msg.id or str(uuid.uuid4()),
                "role": "user",
                "content": content,
                "sources": None,
                "tool_calls": None,
                "created_at": datetime.now(timezone.utc),
            })
        elif isinstance(msg, (AIMessage, ToolMessage)):
            turn_messages.append(msg)

    flush_assistant_turn()
    return history

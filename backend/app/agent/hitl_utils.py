from typing import Any

from langgraph.types import StateSnapshot


def serialize_hitl_request(value: Any) -> dict | None:
    if value is None:
        return None

    if isinstance(value, dict):
        if "action_requests" in value:
            return {
                "action_requests": value.get("action_requests", []),
                "review_configs": value.get("review_configs", []),
            }
        if "decisions" in value:
            return None

    return None


def extract_hitl_request(state: StateSnapshot | None) -> dict | None:
    if state is None or not state.interrupts:
        return None

    for interrupt in reversed(state.interrupts):
        payload = serialize_hitl_request(interrupt.value)
        if payload:
            return payload

    return None

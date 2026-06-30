from langchain_core.messages import AIMessage, HumanMessage

from app.agent.message_utils import convert_messages_to_history


def test_convert_messages_to_history_includes_run_id():
    assistant_id = "assistant-1"
    messages = [
        HumanMessage(content="hello", id="user-1"),
        AIMessage(content="hi there", id=assistant_id),
    ]
    traces = {assistant_id: {"run_id": "run-abc", "trace_id": "trace-abc"}}

    history = convert_messages_to_history(messages, message_traces=traces)

    assistant = next(item for item in history if item["role"] == "assistant")
    assert assistant["run_id"] == "run-abc"
    assert assistant["trace_id"] == "trace-abc"

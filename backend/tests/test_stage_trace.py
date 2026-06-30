from unittest.mock import patch

from app.observability.stage_trace import trace_stage


def test_trace_stage_noop_when_disabled():
    @trace_stage("rag_test")
    def sample(x: int) -> int:
        return x + 1

    with patch("app.observability.stage_trace.is_langsmith_enabled", return_value=False):
        assert sample(1) == 2


def test_trace_stage_applies_traceable_when_enabled():
    calls: list[str] = []

    def fake_traceable(**kwargs):
        calls.append(kwargs.get("name", ""))

        def decorator(fn):
            return fn

        return decorator

    with (
        patch("app.observability.stage_trace.is_langsmith_enabled", return_value=True),
        patch("app.observability.stage_trace.settings.langsmith_stage_tracing_enabled", True),
        patch("app.observability.stage_trace._langsmith_traceable", fake_traceable),
    ):
        @trace_stage("rag_test_span")
        def sample(x: int) -> int:
            return x + 2

        assert sample(3) == 5

    assert calls == ["rag_test_span"]

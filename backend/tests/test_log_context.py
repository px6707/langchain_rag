import logging

from app.observability.log_context import TraceContextFilter, init_log_context, reset_log_context


def test_trace_context_filter_attaches_fields():
    tokens = init_log_context(trace_id="t1", session_id="s1", user_id="u1")
    try:
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="hello",
            args=(),
            exc_info=None,
        )
        assert TraceContextFilter().filter(record) is True
        assert record.trace_id == "t1"  # type: ignore[attr-defined]
        assert record.session_id == "s1"  # type: ignore[attr-defined]
        assert record.user_id == "u1"  # type: ignore[attr-defined]
    finally:
        reset_log_context(*tokens)

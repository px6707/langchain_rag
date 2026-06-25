from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.config import settings
from app.schemas.retrieval import RetrievalPlan
from app.services.retrieval_rules import rule_precheck
from app.services.retrieval_router import plan_retrieval, rewrite_query
from app.services.retrieval_validator import normalize_plan, validate_standalone_query


def _mock_router_llm(*, rewrite=None, strategy=None):
    llm = MagicMock()

    def structured_output(schema):
        mock = MagicMock()
        if schema.__name__ == "QueryRewrite":
            mock.invoke.return_value = rewrite or {"standalone_query": "resolved", "reason": "test"}
        else:
            mock.invoke.return_value = strategy or {
                "action": "retrieve",
                "strategy": "none",
                "extra_queries": [],
                "hyde_document": None,
                "reason": "test",
            }
        return mock

    llm.with_structured_output.side_effect = structured_output
    return llm


def test_rule_precheck_greeting_skip():
    plan = rule_precheck("你好")
    assert plan is not None
    assert plan.action == "skip"


def test_rule_precheck_normal_query_returns_none():
    assert rule_precheck("LangChain 的核心组件有哪些？") is None


def test_rewrite_query_single_turn_skips_llm():
    llm = MagicMock()
    messages = [HumanMessage(content="什么是 RAG？")]
    result = rewrite_query(messages, llm=llm)
    assert result == "什么是 RAG？"
    llm.with_structured_output.assert_not_called()


def test_rewrite_query_multi_turn_calls_llm():
    llm = _mock_router_llm(rewrite={"standalone_query": "LangChain 的 RAG 怎么做", "reason": "指代消解"})
    messages = [
        HumanMessage(content="LangChain 是什么"),
        AIMessage(content="LangChain 是一个框架"),
        HumanMessage(content="那 RAG 怎么做"),
    ]
    result = rewrite_query(messages, llm=llm)
    assert result == "LangChain 的 RAG 怎么做"
    llm.with_structured_output.assert_called()


def test_plan_retrieval_rule_precheck_skips_llm():
    llm = MagicMock()
    with patch.object(settings, "retrieval_routing_enabled", True):
        plan = plan_retrieval([HumanMessage(content="谢谢")], llm=llm)
    assert plan.action == "skip"
    llm.with_structured_output.assert_not_called()


def test_plan_retrieval_two_step():
    llm = _mock_router_llm(
        rewrite={"standalone_query": "完整问题", "reason": "rewrite"},
        strategy={
            "action": "retrieve",
            "strategy": "multi_query",
            "extra_queries": ["q1", "q2"],
            "hyde_document": None,
            "reason": "multi",
        },
    )
    messages = [
        HumanMessage(content="之前的问题"),
        AIMessage(content="之前的回答"),
        HumanMessage(content="它的细节"),
    ]
    with patch.object(settings, "retrieval_routing_enabled", True):
        plan = plan_retrieval(messages, llm=llm)
    assert plan.standalone_query == "完整问题"
    assert plan.strategy == "multi_query"
    assert len(plan.extra_queries) == 2


def test_validate_standalone_query_anaphora_fallback():
    result = validate_standalone_query("它是什么", "原始问题")
    assert result == "原始问题"


def test_normalize_plan_hyde_downgrade():
    plan = RetrievalPlan(
        action="retrieve",
        strategy="hyde",
        standalone_query="q",
        hyde_document="short",
        reason="x",
    )
    normalized = normalize_plan(plan)
    assert normalized.strategy == "none"
    assert normalized.hyde_document is None


def test_format_history_includes_tool_message():
    from app.services.retrieval_router import _format_history

    messages = [
        HumanMessage(content="查一下"),
        ToolMessage(content="tool output here", tool_call_id="1", name="get_current_time"),
    ]
    history = _format_history(messages)
    assert "工具[get_current_time]" in history


def test_plan_retrieval_routing_disabled():
    messages = [HumanMessage(content="文档里说了什么？")]
    with patch.object(settings, "retrieval_routing_enabled", False):
        plan = plan_retrieval(messages, llm=MagicMock())
    assert plan.action == "retrieve"
    assert plan.strategy == "none"
    assert plan.standalone_query == "文档里说了什么？"

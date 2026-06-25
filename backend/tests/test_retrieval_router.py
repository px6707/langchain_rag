from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.config import settings
from app.schemas.retrieval import RetrievalPlan
from app.services.retrieval_router import (
    _collect_recent_turns,
    expand_from_context,
    plan_retrieval,
    resolve_standalone,
    rewrite_query,
)
from app.services.retrieval_validator import normalize_plan


def _mock_router_llm(*, rewrite=None, strategy=None):
    llm = MagicMock()

    def structured_output(schema):
        mock = MagicMock()
        if schema.__name__ == "QueryRewrite":
            mock.invoke.return_value = rewrite or {
                "standalone_query": "resolved",
                "resolved_entities": ["RAG"],
                "confidence": "high",
                "reason": "test",
            }
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


def test_rewrite_query_single_turn_skips_llm():
    llm = MagicMock()
    messages = [HumanMessage(content="什么是 RAG？")]
    result = rewrite_query(messages, llm=llm)
    assert result == "什么是 RAG？"
    llm.with_structured_output.assert_not_called()


def test_resolve_standalone_calls_llm_on_multi_turn():
    llm = _mock_router_llm(
        rewrite={
            "standalone_query": "RAG 是什么",
            "resolved_entities": ["RAG"],
            "confidence": "high",
            "reason": "resolved",
        }
    )
    messages = [
        HumanMessage(content="什么是 RAG"),
        AIMessage(content="RAG 是检索增强生成"),
        HumanMessage(content="它是什么"),
    ]
    standalone, entities = resolve_standalone(messages, llm=llm)
    assert standalone == "RAG 是什么"
    assert "RAG" in entities
    llm.with_structured_output.assert_called()


def test_plan_retrieval_postcheck_overrides_skip():
    llm = _mock_router_llm(
        strategy={
            "action": "skip",
            "strategy": "none",
            "extra_queries": [],
            "hyde_document": None,
            "reason": "llm",
        }
    )
    with patch.object(settings, "retrieval_routing_enabled", True):
        plan = plan_retrieval([HumanMessage(content="文档里的合同条款是什么？")], llm=llm)
    assert plan.action == "retrieve"


def test_plan_retrieval_two_step():
    llm = _mock_router_llm(
        rewrite={
            "standalone_query": "完整问题",
            "resolved_entities": [],
            "confidence": "high",
            "reason": "rewrite",
        },
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


def test_collect_recent_turns_by_human_count():
    messages = [
        HumanMessage(content="t1"),
        AIMessage(content="a1"),
        HumanMessage(content="t2"),
        AIMessage(content="a2"),
        HumanMessage(content="t3"),
    ]
    with patch.object(settings, "retrieval_history_turns", 2):
        turns = _collect_recent_turns(messages, 2)
    assert turns[0].content == "t2"
    assert turns[-1].content == "t3"


def test_format_history_includes_tool_message():
    from app.services.retrieval_router import _format_history

    messages = [
        HumanMessage(content="查一下"),
        ToolMessage(content="tool output here", tool_call_id="1", name="get_current_time"),
    ]
    history = _format_history(messages)
    assert "工具[get_current_time]" in history


def test_expand_from_context_with_entities():
    messages = [
        HumanMessage(content="合同A的内容"),
        AIMessage(content="合同A规定了违约金"),
        HumanMessage(content="它的上限是多少"),
    ]
    expanded = expand_from_context(messages, entities=["合同A"])
    assert "合同A" in expanded


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

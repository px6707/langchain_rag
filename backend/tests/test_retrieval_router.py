from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from app.config import settings
from app.schemas.retrieval import RetrievalPlan
from app.services.retrieval_router import plan_retrieval


def _mock_llm(return_plan: RetrievalPlan) -> MagicMock:
    llm = MagicMock()
    structured = MagicMock()
    structured.invoke.return_value = return_plan
    llm.with_structured_output.return_value = structured
    return llm


def test_plan_retrieval_skip_chitchat():
    llm = _mock_llm(RetrievalPlan(action="skip", reason="问候语"))
    messages = [HumanMessage(content="你好")]

    with patch.object(settings, "retrieval_routing_enabled", True):
        plan = plan_retrieval(messages, llm=llm)

    assert plan.action == "skip"
    assert plan.reason == "问候语"


def test_plan_retrieval_skip_tool_only():
    llm = _mock_llm(RetrievalPlan(action="skip", reason="纯工具调用"))
    messages = [HumanMessage(content="现在几点了？")]

    with patch.object(settings, "retrieval_routing_enabled", True):
        plan = plan_retrieval(messages, llm=llm)

    assert plan.action == "skip"


def test_plan_retrieval_retrieve_none():
    llm = _mock_llm(
        RetrievalPlan(
            action="retrieve",
            strategy="none",
            standalone_query="LangChain 是什么？",
            reason="简单事实问句",
        )
    )
    messages = [HumanMessage(content="LangChain 是什么？")]

    with patch.object(settings, "retrieval_routing_enabled", True):
        plan = plan_retrieval(messages, llm=llm)

    assert plan.action == "retrieve"
    assert plan.strategy == "none"
    assert plan.standalone_query == "LangChain 是什么？"


def test_plan_retrieval_multi_query():
    llm = _mock_llm(
        RetrievalPlan(
            action="retrieve",
            strategy="multi_query",
            standalone_query="RAG 的优缺点",
            extra_queries=["RAG 优点", "RAG 缺点", "RAG 适用场景", "多余 query"],
            reason="多角度",
        )
    )
    messages = [HumanMessage(content="RAG 的优缺点和适用场景？")]

    with (
        patch.object(settings, "retrieval_routing_enabled", True),
        patch.object(settings, "retrieval_multi_query_count", 3),
    ):
        plan = plan_retrieval(messages, llm=llm)

    assert plan.strategy == "multi_query"
    assert len(plan.extra_queries) == 3


def test_plan_retrieval_hyde():
    llm = _mock_llm(
        RetrievalPlan(
            action="retrieve",
            strategy="hyde",
            standalone_query="向量数据库如何工作",
            hyde_document="向量数据库通过 embedding 将文本映射到高维空间...",
            reason="概念抽象",
        )
    )
    messages = [HumanMessage(content="向量数据库如何工作？")]

    with patch.object(settings, "retrieval_routing_enabled", True):
        plan = plan_retrieval(messages, llm=llm)

    assert plan.strategy == "hyde"
    assert plan.hyde_document is not None


def test_plan_retrieval_decompose():
    llm = _mock_llm(
        RetrievalPlan(
            action="retrieve",
            strategy="decompose",
            standalone_query="A 和 B 的区别",
            extra_queries=["A 是什么", "B 是什么", "A 与 B 对比"],
            reason="多部分问题",
        )
    )
    messages = [HumanMessage(content="A 和 B 有什么区别？各自限制？")]

    with (
        patch.object(settings, "retrieval_routing_enabled", True),
        patch.object(settings, "retrieval_max_sub_questions", 4),
    ):
        plan = plan_retrieval(messages, llm=llm)

    assert plan.strategy == "decompose"
    assert len(plan.extra_queries) == 3


def test_plan_retrieval_routing_disabled():
    messages = [HumanMessage(content="文档里说了什么？")]

    with patch.object(settings, "retrieval_routing_enabled", False):
        plan = plan_retrieval(messages, llm=MagicMock())

    assert plan.action == "retrieve"
    assert plan.strategy == "none"
    assert plan.standalone_query == "文档里说了什么？"


def test_plan_retrieval_fail_open_on_llm_error():
    llm = MagicMock()
    structured = MagicMock()
    structured.invoke.side_effect = RuntimeError("LLM unavailable")
    llm.with_structured_output.return_value = structured
    messages = [HumanMessage(content="查询内容")]

    with patch.object(settings, "retrieval_routing_enabled", True):
        plan = plan_retrieval(messages, llm=llm)

    assert plan.action == "retrieve"
    assert plan.strategy == "none"
    assert plan.standalone_query == "查询内容"


def test_plan_retrieval_empty_query():
    plan = plan_retrieval([], llm=MagicMock())
    assert plan.action == "skip"


def test_plan_retrieval_clears_hyde_for_non_hyde_strategy():
    llm = _mock_llm(
        RetrievalPlan(
            action="retrieve",
            strategy="none",
            standalone_query="问题",
            hyde_document="不应保留",
            reason="简单问句",
        )
    )
    messages = [HumanMessage(content="问题")]

    with patch.object(settings, "retrieval_routing_enabled", True):
        plan = plan_retrieval(messages, llm=llm)

    assert plan.hyde_document is None

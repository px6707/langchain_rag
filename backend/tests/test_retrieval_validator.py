from app.schemas.retrieval import RetrievalPlan
from app.services.retrieval_router import expand_from_context
from app.services.retrieval_validator import (
    has_anaphora,
    normalize_plan,
    suggest_strategy,
    validate_strategy_plan,
)
from langchain_core.messages import AIMessage, HumanMessage


def test_has_anaphora_chinese_and_english():
    assert has_anaphora("它是什么")
    assert has_anaphora("what about this")
    assert not has_anaphora("LangChain 的核心组件有哪些")


def test_suggest_strategy():
    assert suggest_strategy("A 和 B 的区别") == "decompose"
    assert suggest_strategy("RAG 的优缺点") == "multi_query"
    assert suggest_strategy("简单问题") is None


def test_validate_strategy_plan_decompose_downgrade():
    plan = RetrievalPlan(
        action="retrieve",
        strategy="decompose",
        standalone_query="问题",
        extra_queries=["子问题1"],
        reason="x",
    )
    updated = validate_strategy_plan(plan)
    assert updated.strategy == "multi_query"


def test_validate_strategy_plan_dedupe_extra():
    plan = RetrievalPlan(
        action="retrieve",
        strategy="multi_query",
        standalone_query="主问题",
        extra_queries=["主问题", "变体问题"],
        reason="x",
    )
    updated = validate_strategy_plan(plan)
    assert updated.extra_queries == ["变体问题"]


def test_normalize_plan_hyde_short_downgrades():
    plan = RetrievalPlan(
        action="retrieve",
        strategy="hyde",
        standalone_query="q",
        hyde_document="short",
        reason="x",
    )
    normalized = normalize_plan(plan)
    assert normalized.strategy == "none"


def test_expand_from_context_uses_ai_excerpt():
    messages = [
        HumanMessage(content="什么是 RAG"),
        AIMessage(content="RAG 是检索增强生成技术"),
        HumanMessage(content="它有什么优点"),
    ]
    expanded = expand_from_context(messages)
    assert "RAG 是检索增强生成" in expanded
    assert "它有什么优点" in expanded

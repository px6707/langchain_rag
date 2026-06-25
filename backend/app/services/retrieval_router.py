import logging

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from app.config import settings
from app.schemas.retrieval import QueryRewrite, RetrievalPlan, StrategyPlan
from app.services.llm_service import get_router_llm
from app.services.retrieval_rules import rule_precheck
from app.services.retrieval_validator import ANAPHORA_RE, normalize_plan, validate_standalone_query

logger = logging.getLogger(__name__)

REWRITE_SYSTEM_PROMPT = """你是 RAG 查询改写器。根据对话历史，将最新用户消息改写为可独立用于文档检索的自包含问题。

要求：
- standalone_query 必须完整、不得含「它」「这个」「上面」「刚才」等指代
- 结合对话与工具结果上下文解析指代
- 若用户消息本身已自包含，可原样输出
- reason 用一句话中文说明改写依据"""

STRATEGY_SYSTEM_PROMPT = """你是 RAG 检索策略路由器。根据对话与已改写 query，决定是否检索文档库并选择 query 变换策略。

## 可用工具（与文档无关的请求通常不需要检索）
{tool_names}

## 决策规则

### action=skip（不检索）
- 问候、致谢、闲聊、身份/meta 问题
- 纯工具调用意图，且与已上传文档无关
- 无实质问题

### action=retrieve（需要检索文档）
- 问题需要依据已上传文档内容回答
- 不确定是否需要文档时，优先 retrieve（strategy=none）

### strategy（仅 action=retrieve 时有效，四选一）
- none：简单事实问句，已改写 query 已足够
- multi_query：表述宽泛，需多角度同义 query；填充 extra_queries（{multi_query_count} 条左右）
- hyde：概念抽象、措辞可能与文档不一致；填充 hyde_document（假设性答案段落，100-200 字）
- decompose：多部分/对比/因果链；填充 extra_queries（子问题，最多 {max_sub_questions} 条，不含已提供的 standalone_query）

## 输出约束
- extra_queries 仅在 multi_query 或 decompose 时填充
- hyde_document 仅在 hyde 时填充，否则为 null
- reason 用一句话中文说明决策依据"""


def _get_last_user_message(messages: list) -> str | None:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            content = msg.content
            return content if isinstance(content, str) else str(content)
    return None


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _format_tool_calls(msg: AIMessage) -> str | None:
    tool_calls = getattr(msg, "tool_calls", None) or []
    if not tool_calls:
        return None
    parts: list[str] = []
    for call in tool_calls[:5]:
        if isinstance(call, dict):
            name = call.get("name", "unknown")
            args = call.get("args", {})
        else:
            name = getattr(call, "name", "unknown")
            args = getattr(call, "args", {})
        parts.append(f"{name}({args})")
    return "工具调用: " + "; ".join(parts)


def _format_message(msg: BaseMessage) -> str | None:
    max_chars = settings.retrieval_tool_context_max_chars

    if isinstance(msg, HumanMessage):
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        return f"用户: {content}"

    if isinstance(msg, AIMessage):
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        lines = [f"助手: {content}"]
        tool_summary = _format_tool_calls(msg)
        if tool_summary:
            lines.append(_truncate(tool_summary, max_chars))
        return "\n".join(lines)

    if isinstance(msg, ToolMessage):
        name = getattr(msg, "name", None) or "tool"
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        return f"工具[{name}]: {_truncate(content, max_chars)}"

    return None


def _format_history(messages: list) -> str:
    recent = messages[-settings.retrieval_history_messages :]
    lines = [line for msg in recent if (line := _format_message(msg))]
    return "\n".join(lines) if lines else "（无历史）"


def _needs_rewrite(messages: list) -> bool:
    human_count = sum(1 for msg in messages if isinstance(msg, HumanMessage))
    has_ai = any(isinstance(msg, AIMessage) for msg in messages)
    return human_count > 1 or has_ai


def _build_strategy_prompt() -> str:
    return STRATEGY_SYSTEM_PROMPT.format(
        tool_names=settings.retrieval_router_tool_names,
        multi_query_count=settings.retrieval_multi_query_count,
        max_sub_questions=settings.retrieval_max_sub_questions,
    )


def _fallback_plan(query: str, *, reason: str) -> RetrievalPlan:
    return RetrievalPlan(
        action="retrieve",
        strategy="none",
        standalone_query=query,
        reason=reason,
    )


def rewrite_query(
    messages: list,
    *,
    llm: BaseChatModel | None = None,
) -> str:
    query = _get_last_user_message(messages)
    if not query or not query.strip():
        return ""

    if not _needs_rewrite(messages):
        return query.strip()

    history = _format_history(messages)
    user_prompt = f"## 对话历史\n{history}\n\n## 最新用户消息\n{query}"
    model = llm or get_router_llm()
    structured = model.with_structured_output(QueryRewrite)

    try:
        raw = structured.invoke(
            [
                {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        )
        rewrite = QueryRewrite.model_validate(raw)
        standalone = rewrite.standalone_query.strip() or query.strip()
        logger.info("Query rewrite: %s -> %s (%s)", query, standalone, rewrite.reason)
        return standalone
    except Exception:
        logger.exception("Query rewrite failed; using original query")
        return query.strip()


def plan_strategy(
    messages: list,
    standalone_query: str,
    *,
    llm: BaseChatModel | None = None,
) -> StrategyPlan:
    history = _format_history(messages)
    user_prompt = (
        f"## 对话历史\n{history}\n\n"
        f"## 已改写 query（standalone_query）\n{standalone_query}"
    )
    model = llm or get_router_llm()
    structured = model.with_structured_output(StrategyPlan)

    raw = structured.invoke(
        [
            {"role": "system", "content": _build_strategy_prompt()},
            {"role": "user", "content": user_prompt},
        ]
    )
    strategy = StrategyPlan.model_validate(raw)

    if strategy.strategy == "multi_query":
        strategy.extra_queries = strategy.extra_queries[: settings.retrieval_multi_query_count]
    elif strategy.strategy == "decompose":
        strategy.extra_queries = strategy.extra_queries[: settings.retrieval_max_sub_questions]
    else:
        strategy.extra_queries = []

    if strategy.strategy != "hyde":
        strategy.hyde_document = None

    return strategy


def plan_retrieval(
    messages: list,
    *,
    llm: BaseChatModel | None = None,
) -> RetrievalPlan:
    query = _get_last_user_message(messages)
    if not query or not query.strip():
        return RetrievalPlan(action="skip", reason="用户消息为空")

    ruled = rule_precheck(query)
    if ruled is not None:
        logger.info("Rule precheck: action=%s reason=%s", ruled.action, ruled.reason)
        return ruled

    if not settings.retrieval_routing_enabled:
        return _fallback_plan(query, reason="检索路由已关闭")

    model = llm or get_router_llm()
    original = query.strip()
    multi_turn = _needs_rewrite(messages)

    try:
        standalone = rewrite_query(messages, llm=model)
        standalone = validate_standalone_query(standalone, original)

        if multi_turn and ANAPHORA_RE.search(original):
            standalone = rewrite_query(messages, llm=model)
            standalone = validate_standalone_query(standalone, original)

        strategy = plan_strategy(messages, standalone, llm=model)
        plan = RetrievalPlan(
            action=strategy.action,
            strategy=strategy.strategy,
            standalone_query=standalone,
            extra_queries=strategy.extra_queries,
            hyde_document=strategy.hyde_document,
            reason=strategy.reason,
        )
    except Exception:
        logger.exception("Retrieval routing failed; falling back to direct retrieval")
        return _fallback_plan(original, reason="路由异常，fail-open 直接检索")

    if plan.action == "skip":
        logger.info("Retrieval skipped: %s", plan.reason)
        return plan

    if not plan.standalone_query.strip():
        plan.standalone_query = original

    plan = normalize_plan(plan)
    logger.info(
        "Retrieval plan: action=%s strategy=%s reason=%s",
        plan.action,
        plan.strategy,
        plan.reason,
    )
    return plan

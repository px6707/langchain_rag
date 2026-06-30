import logging
from contextvars import ContextVar

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from app.config import settings
from app.observability.stage_trace import trace_stage
from app.schemas.retrieval import QueryRewrite, RetrievalPlan, StrategyPlan
from app.services.llm_service import get_small_llm
from app.services.retrieval_rules import rule_postcheck_retrieve, rule_precheck
from app.services.retrieval_validator import has_anaphora, normalize_plan, suggest_strategy

logger = logging.getLogger(__name__)

_session_entities: ContextVar[list[str] | None] = ContextVar("session_entities", default=None)

REWRITE_SYSTEM_PROMPT = """你是 RAG 查询改写器。根据对话历史，将最新用户消息改写为可独立用于文档检索的自包含问题。

要求：
- standalone_query 必须完整、不得含指代词（它/这个/上面/刚才/后者/前者/this/that 等）
- 结合对话与工具结果上下文解析指代
- resolved_entities 列出对话中的关键实体（文件名、产品名、合同名等）
- confidence=low 表示指代消解不确定
- 若用户消息本身已自包含，可原样输出且 confidence=high
- reason 用一句话中文说明改写依据"""

REWRITE_FORCE_SYSTEM_PROMPT = """你是 RAG 查询改写器。上次改写仍含指代词，本次必须输出完全自包含、不含任何指代词的检索问题。
必须结合对话历史将指代全部替换为具体实体或完整描述。"""

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
- decompose：多部分/对比/因果链；填充 extra_queries（子问题，最多 {max_sub_questions} 条，不含 standalone_query）

## 输出约束
- extra_queries 仅在 multi_query 或 decompose 时填充，且不得重复 standalone_query
- hyde_document 仅在 hyde 时填充，否则为 null
- reason 用一句话中文说明决策依据

## 示例
- 「LangChain 是什么？」→ retrieve, none
- 「RAG 有哪些优缺点和适用场景？」→ retrieve, multi_query, extra=[「RAG 优点」,「RAG 缺点」,「RAG 适用场景」]
- 「向量数据库的工作原理（文档措辞可能不同）」→ retrieve, hyde, hyde_document=假设性答案段落
- 「Elasticsearch 和 Milvus 的区别及各自限制？」→ retrieve, decompose, extra=[「Elasticsearch 是什么」,「Milvus 是什么」,「两者区别」]
- 「你好」→ skip"""


def _get_last_user_message(messages: list) -> str | None:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            content = msg.content
            return content if isinstance(content, str) else str(content)
    return None


def _get_previous_human_message(messages: list) -> str | None:
    seen_last = False
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if not seen_last:
                seen_last = True
                continue
            return content
    return None


def _get_last_ai_message(messages: list) -> str | None:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            return content
    return None


def _truncate_smart(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head_len = int(max_chars * 0.6)
    tail_len = max_chars - head_len - 10
    return f"{text[:head_len]}...[截断]...{text[-tail_len:]}"


def _truncate(text: str, max_chars: int) -> str:
    return _truncate_smart(text, max_chars)


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


def _collect_recent_turns(messages: list, max_turns: int) -> list[BaseMessage]:
    collected: list[BaseMessage] = []
    human_seen = 0
    for msg in reversed(messages):
        collected.append(msg)
        if isinstance(msg, HumanMessage):
            human_seen += 1
            if human_seen >= max_turns:
                break
    collected.reverse()
    if len(collected) > settings.retrieval_history_messages:
        collected = collected[-settings.retrieval_history_messages :]
    return collected


def _format_history(messages: list) -> str:
    recent = _collect_recent_turns(messages, settings.retrieval_history_turns)
    known_entities = _session_entities.get() or []
    lines = [line for msg in recent if (line := _format_message(msg))]
    history = "\n".join(lines) if lines else "（无历史）"
    if known_entities:
        history = f"已知实体: {', '.join(known_entities)}\n{history}"
    return history


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


def _apply_entities(standalone: str, entities: list[str]) -> str:
    if not entities or not has_anaphora(standalone):
        return standalone
    prefix = "、".join(entities[:5])
    return f"关于{prefix}：{standalone}"


def expand_from_context(messages: list, *, entities: list[str] | None = None) -> str:
    query = _get_last_user_message(messages) or ""
    ai_excerpt = (_get_last_ai_message(messages) or "")[:300]
    prev_human = _get_previous_human_message(messages)

    if ai_excerpt:
        standalone = f"「{ai_excerpt}」相关问题：{query}"
    elif prev_human:
        standalone = f"上一轮问题「{prev_human}」的后续：{query}"
    else:
        standalone = query

    return _apply_entities(standalone, entities or [])


def _invoke_rewrite(
    messages: list,
    *,
    llm: BaseChatModel,
    force: bool = False,
) -> QueryRewrite:
    query = _get_last_user_message(messages) or ""
    history = _format_history(messages)
    system = REWRITE_FORCE_SYSTEM_PROMPT if force else REWRITE_SYSTEM_PROMPT
    user_prompt = f"## 对话历史\n{history}\n\n## 最新用户消息\n{query}"
    if force:
        user_prompt += "\n\n注意：上次改写仍含指代词，本次必须完全消解。"

    structured = llm.with_structured_output(QueryRewrite)
    raw = structured.invoke(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ]
    )
    return QueryRewrite.model_validate(raw)


def rewrite_query(
    messages: list,
    *,
    llm: BaseChatModel | None = None,
) -> str:
    return resolve_standalone(messages, llm=llm)[0]


def rewrite_query_force(messages: list, *, llm: BaseChatModel | None = None) -> QueryRewrite:
    model = llm or get_small_llm()
    return _invoke_rewrite(messages, llm=model, force=True)


def resolve_standalone(
    messages: list,
    *,
    llm: BaseChatModel | None = None,
) -> tuple[str, list[str]]:
    query = _get_last_user_message(messages)
    if not query or not query.strip():
        return "", []

    if not _needs_rewrite(messages):
        return query.strip(), []

    model = llm or get_small_llm()
    entities: list[str] = []

    try:
        rewrite = _invoke_rewrite(messages, llm=model, force=False)
        standalone = rewrite.standalone_query.strip() or query.strip()
        entities = rewrite.resolved_entities

        if has_anaphora(standalone) or rewrite.confidence == "low":
            rewrite = _invoke_rewrite(messages, llm=model, force=True)
            standalone = rewrite.standalone_query.strip() or standalone
            entities = rewrite.resolved_entities or entities

        if has_anaphora(standalone):
            standalone = expand_from_context(messages, entities=entities)

        if has_anaphora(standalone) and not _get_last_ai_message(messages):
            standalone = query.strip()

        standalone = _apply_entities(standalone, entities)
        _session_entities.set(entities)
        logger.info("Query rewrite resolved: %s -> %s", query, standalone)
        return standalone, entities
    except Exception:
        logger.exception("Query rewrite failed; expanding from context")
        expanded = expand_from_context(messages, entities=entities)
        if has_anaphora(expanded) and not _get_last_ai_message(messages):
            expanded = query.strip()
        return expanded, entities


def plan_strategy(
    messages: list,
    standalone_query: str,
    *,
    llm: BaseChatModel | None = None,
) -> StrategyPlan:
    history = _format_history(messages)
    hint = suggest_strategy(standalone_query)
    hint_text = f"\n## 建议策略（可参考）：{hint}" if hint else ""
    user_prompt = (
        f"## 对话历史\n{history}\n\n"
        f"## 已改写 query（standalone_query）\n{standalone_query}"
        f"{hint_text}"
    )
    model = llm or get_small_llm()
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


def _fallback_plan(query: str, *, reason: str) -> RetrievalPlan:
    return RetrievalPlan(
        action="retrieve",
        strategy="none",
        standalone_query=query,
        reason=reason,
    )


@trace_stage("rag_retrieval_router")
def plan_retrieval(
    messages: list,
    *,
    llm: BaseChatModel | None = None,
) -> RetrievalPlan:
    query = _get_last_user_message(messages)
    if not query or not query.strip():
        return RetrievalPlan(action="skip", reason="用户消息为空")

    original = query.strip()
    precheck_postcheck_reason: str | None = None
    ruled = rule_precheck(original)
    if ruled is not None and ruled.action == "skip":
        ruled = rule_postcheck_retrieve(original, ruled)
        if ruled.action == "skip":
            logger.info("Rule precheck+postcheck skip: %s", ruled.reason)
            return ruled
        precheck_postcheck_reason = ruled.reason

    if not settings.retrieval_routing_enabled:
        return _fallback_plan(original, reason="检索路由已关闭")

    model = llm or get_small_llm()

    try:
        standalone, _ = resolve_standalone(messages, llm=model)
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

    was_skip = plan.action == "skip"
    plan = rule_postcheck_retrieve(original, plan)

    if was_skip and plan.action == "retrieve":
        postcheck_reason = plan.reason
        standalone = plan.standalone_query.strip() or original
        logger.info("Postcheck upgraded skip->retrieve; re-running strategy")
        strategy = plan_strategy(messages, standalone, llm=model)
        plan = RetrievalPlan(
            action="retrieve",
            strategy=strategy.strategy,
            standalone_query=standalone,
            extra_queries=strategy.extra_queries,
            hyde_document=strategy.hyde_document,
            reason=f"{postcheck_reason}; {strategy.reason}",
        )

    if plan.action == "skip":
        logger.info("Retrieval skipped: %s", plan.reason)
        return plan

    if not plan.standalone_query.strip():
        plan.standalone_query = original

    if precheck_postcheck_reason:
        plan = plan.model_copy(
            update={"reason": f"{precheck_postcheck_reason}; {plan.reason}"}
        )

    plan = normalize_plan(plan)
    logger.info(
        "Retrieval plan: action=%s strategy=%s reason=%s",
        plan.action,
        plan.strategy,
        plan.reason,
    )
    return plan

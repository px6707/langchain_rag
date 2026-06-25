import logging

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.config import settings
from app.schemas.retrieval import RetrievalPlan
from app.services.llm_service import get_llm

logger = logging.getLogger(__name__)

ROUTER_SYSTEM_PROMPT = """你是 RAG 检索路由器。根据对话历史与最新用户消息，决定是否检索文档库，并选择最合适的 query 变换策略。

## 可用工具（与文档无关的请求通常不需要检索）
{tool_names}

## 决策规则

### action=skip（不检索）
- 问候、致谢、闲聊、身份/meta 问题（如「你是谁」）
- 纯工具调用意图，且与已上传文档无关（如查时间、发邮件、执行脚本）
- 用户消息为空或无实质问题

### action=retrieve（需要检索文档）
- 问题需要依据已上传文档内容回答
- 不确定是否需要文档时，优先 retrieve（strategy=none）

### strategy（仅 action=retrieve 时有效，四选一）
- none：简单事实问句，standalone_query 已足够
- multi_query：表述宽泛，需多角度同义 query（优缺点、适用场景等）；填充 extra_queries（{multi_query_count} 条左右）
- hyde：概念抽象、用户措辞可能与文档不一致；填充 hyde_document（假设性答案段落，100-200 字）
- decompose：多部分/对比/因果链；填充 extra_queries（子问题，最多 {max_sub_questions} 条）

## standalone_query 要求
- 必须自包含，不得含「它」「上面那个」「这个」等指代
- 结合对话历史将指代解析为完整问题
- action=skip 时 standalone_query 可为空

## 输出约束
- extra_queries 仅在 multi_query 或 decompose 时填充，否则为空列表
- hyde_document 仅在 hyde 时填充，否则为 null
- reason 用一句话中文说明决策依据"""


def _get_last_user_message(messages: list) -> str | None:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            content = msg.content
            return content if isinstance(content, str) else str(content)
    return None


def _format_message(msg: BaseMessage) -> str | None:
    if isinstance(msg, HumanMessage):
        role = "用户"
    elif isinstance(msg, AIMessage):
        role = "助手"
    else:
        return None
    content = msg.content
    text = content if isinstance(content, str) else str(content)
    return f"{role}: {text}"


def _format_history(messages: list) -> str:
    recent = messages[-settings.retrieval_history_messages :]
    lines = [line for msg in recent if (line := _format_message(msg))]
    return "\n".join(lines) if lines else "（无历史）"


def _build_router_prompt() -> str:
    return ROUTER_SYSTEM_PROMPT.format(
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


def plan_retrieval(
    messages: list,
    *,
    llm: BaseChatModel | None = None,
) -> RetrievalPlan:
    query = _get_last_user_message(messages)
    if not query or not query.strip():
        return RetrievalPlan(action="skip", reason="用户消息为空")

    if not settings.retrieval_routing_enabled:
        return _fallback_plan(query, reason="检索路由已关闭")

    history = _format_history(messages)
    user_prompt = f"## 对话历史\n{history}\n\n## 最新用户消息\n{query}"

    model = llm or get_llm(temperature=settings.retrieval_router_temperature)
    structured = model.with_structured_output(RetrievalPlan)

    try:
        raw = structured.invoke(
            [
                {"role": "system", "content": _build_router_prompt()},
                {"role": "user", "content": user_prompt},
            ]
        )
        plan = RetrievalPlan.model_validate(raw)
    except Exception:
        logger.exception("Retrieval routing failed; falling back to direct retrieval")
        return _fallback_plan(query, reason="路由异常，fail-open 直接检索")

    if plan.action == "skip":
        logger.info("Retrieval skipped: %s", plan.reason)
        return plan

    if not plan.standalone_query.strip():
        plan.standalone_query = query

    if plan.strategy == "multi_query":
        plan.extra_queries = plan.extra_queries[: settings.retrieval_multi_query_count]
    elif plan.strategy == "decompose":
        plan.extra_queries = plan.extra_queries[: settings.retrieval_max_sub_questions]
    else:
        plan.extra_queries = []

    if plan.strategy != "hyde":
        plan.hyde_document = None

    logger.info(
        "Retrieval plan: action=%s strategy=%s reason=%s",
        plan.action,
        plan.strategy,
        plan.reason,
    )
    return plan

import re

from app.config import settings
from app.schemas.retrieval import RetrievalPlan
from app.services.retrieval_validator import has_anaphora, has_doc_intent

_GREETING_RE = re.compile(
    r"^(你好|您好|hi|hello|hey|谢谢|感谢|多谢|再见|拜拜|早上好|晚上好)[!！?？。.\s]*$",
    re.IGNORECASE,
)

_TOOL_ONLY_PATTERNS = (
    re.compile(r"^(现在)?几点(了|钟)?[?？]?$"),
    re.compile(r"^(查|看)?(一下)?时间[?？]?$"),
    re.compile(r"^发(个|一封)?邮件[?？]?$"),
    re.compile(r"^发送邮件[?？]?$"),
)


def rule_precheck(query: str) -> RetrievalPlan | None:
    text = query.strip()
    if not text:
        return RetrievalPlan(action="skip", reason="用户消息为空")

    if _GREETING_RE.match(text):
        return RetrievalPlan(action="skip", reason="规则匹配：问候语")

    if len(text) <= 20:
        for pattern in _TOOL_ONLY_PATTERNS:
            if pattern.match(text):
                return RetrievalPlan(action="skip", reason="规则匹配：纯工具意图")

    return None


def _force_retrieve_update(reason: str) -> dict:
    return {
        "action": "retrieve",
        "strategy": "none",
        "reason": reason,
    }


def rule_postcheck_retrieve(query: str, plan: RetrievalPlan) -> RetrievalPlan:
    text = query.strip()
    if plan.action != "skip":
        return plan

    if has_doc_intent(text):
        return plan.model_copy(
            update=_force_retrieve_update("规则 postcheck：文档意图强制 retrieve")
        )

    if len(text) > 15 and ("?" in text or "？" in text):
        return plan.model_copy(
            update=_force_retrieve_update("规则 postcheck：长问句强制 retrieve")
        )

    if has_anaphora(text):
        return plan.model_copy(
            update=_force_retrieve_update("规则 postcheck：指代追问强制 retrieve")
        )

    return plan

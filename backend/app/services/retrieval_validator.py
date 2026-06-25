import logging
import re

from app.config import settings
from app.schemas.retrieval import RetrievalPlan
from app.services.hyde_service import should_use_hyde

logger = logging.getLogger(__name__)

ANAPHORA_RE = re.compile(
    r"(它|这个|那个|上面|刚才|之前|上述|前述|此|后者|前者|这两种|该|此文档|"
    r"that|this|it|above|former|latter)",
    re.IGNORECASE,
)
_CONTRAST_RE = re.compile(r"(区别|对比|分别|各自|vs|versus|差异)")
_BROAD_RE = re.compile(r"(优缺点|适用场景|哪些方面|总结|概括|全面)")
_HYDE_MIN_LEN = 20


def has_anaphora(text: str) -> bool:
    return bool(ANAPHORA_RE.search(text.strip()))


def has_doc_intent(text: str) -> bool:
    keywords = [k.strip() for k in settings.retrieval_doc_intent_keywords.split(",") if k.strip()]
    lowered = text.lower()
    return any(k.lower() in lowered for k in keywords)


def suggest_strategy(query: str) -> str | None:
    text = query.strip()
    if _CONTRAST_RE.search(text):
        return "decompose"
    if _BROAD_RE.search(text):
        return "multi_query"
    return None


def _dedupe_extra_queries(standalone: str, extras: list[str]) -> list[str]:
    base = standalone.strip().lower()
    result: list[str] = []
    for q in extras:
        item = q.strip()
        if not item:
            continue
        lower = item.lower()
        if lower == base or base in lower or lower in base:
            continue
        if item not in result:
            result.append(item)
    return result


def validate_strategy_plan(plan: RetrievalPlan) -> RetrievalPlan:
    if plan.action != "retrieve":
        return plan

    standalone = plan.standalone_query.strip()
    extras = _dedupe_extra_queries(standalone, plan.extra_queries)

    if plan.strategy == "decompose":
        if len(extras) < 2 and not _CONTRAST_RE.search(standalone):
            logger.info("decompose extras insufficient; downgrading to multi_query")
            return plan.model_copy(
                update={
                    "strategy": "multi_query",
                    "extra_queries": extras,
                    "reason": f"{plan.reason}; decompose 降级为 multi_query",
                }
            )
        extras = [q for q in extras if len(q.strip()) > 5]

    if plan.strategy == "multi_query" and not extras:
        logger.info("multi_query without extras; downgrading to none")
        return plan.model_copy(
            update={
                "strategy": "none",
                "extra_queries": [],
                "reason": f"{plan.reason}; multi_query 降级为 none",
            }
        )

    return plan.model_copy(update={"extra_queries": extras})


def normalize_plan(plan: RetrievalPlan) -> RetrievalPlan:
    if plan.action != "retrieve":
        return plan

    plan = validate_strategy_plan(plan)

    if plan.strategy == "hyde":
        hyde = (plan.hyde_document or "").strip()
        query = plan.standalone_query.strip()
        if len(hyde) < _HYDE_MIN_LEN:
            logger.warning("HyDE document invalid or too short; downgrading to none")
            return plan.model_copy(
                update={
                    "strategy": "none",
                    "hyde_document": None,
                    "hyde_vector_enabled": False,
                    "extra_queries": [],
                    "reason": f"{plan.reason}; hyde 无效已降级",
                }
            )
        if not should_use_hyde(hyde, query):
            logger.warning("HyDE quality score too low; disabling hyde vector channel")
            return plan.model_copy(
                update={
                    "hyde_vector_enabled": False,
                    "reason": f"{plan.reason}; hyde 低分保留 query 通道",
                }
            )
        return plan.model_copy(update={"hyde_vector_enabled": True})

    plan = plan.model_copy(update={"hyde_document": None, "hyde_vector_enabled": False})
    if plan.strategy not in ("multi_query", "decompose"):
        plan = plan.model_copy(update={"extra_queries": []})

    return plan

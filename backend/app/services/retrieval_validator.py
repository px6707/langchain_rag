import logging
import re

from app.schemas.retrieval import RetrievalPlan

logger = logging.getLogger(__name__)

ANAPHORA_RE = re.compile(r"(它|这个|那个|上面|刚才|之前|上述|前述|此)")
_HYDE_MIN_LEN = 20


def validate_standalone_query(standalone: str, original: str) -> str:
    text = standalone.strip()
    if not text:
        return original
    if ANAPHORA_RE.search(text):
        logger.warning("standalone_query still contains anaphora, falling back to original")
        return original
    return text


def normalize_plan(plan: RetrievalPlan) -> RetrievalPlan:
    if plan.action != "retrieve":
        return plan

    if plan.strategy == "hyde":
        hyde = (plan.hyde_document or "").strip()
        if len(hyde) < _HYDE_MIN_LEN:
            logger.warning("HyDE document invalid or too short; downgrading to none")
            return plan.model_copy(
                update={
                    "strategy": "none",
                    "hyde_document": None,
                    "extra_queries": [],
                    "reason": f"{plan.reason}; hyde 无效已降级",
                }
            )

    if plan.strategy != "hyde":
        plan = plan.model_copy(update={"hyde_document": None})

    if plan.strategy not in ("multi_query", "decompose"):
        plan = plan.model_copy(update={"extra_queries": []})

    return plan

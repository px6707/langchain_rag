import logging
from typing import Any

from openviking.integrations.langchain.client import item_value, iter_result_items

from app.config import settings
from app.openviking.client import get_local_client, is_openviking_available, request_context

logger = logging.getLogger(__name__)


def _memory_category(uri: str) -> str:
    uri_lower = uri.lower()
    if "/preferences/" in uri_lower or uri_lower.endswith("/preferences"):
        return "preferences"
    if "profile" in uri_lower:
        return "profile"
    if "/entities/" in uri_lower:
        return "entities"
    if "/events/" in uri_lower:
        return "events"
    return "other"


def _format_memory_item(item: Any) -> str:
    uri = str(item_value(item, "uri", "") or "")
    abstract = str(item_value(item, "abstract", "") or "")
    overview = str(item_value(item, "overview", "") or "")
    score = item_value(item, "score")
    content = overview or abstract
    if not content:
        return ""
    score_text = f" | 相关度: {score:.3f}" if isinstance(score, (int, float)) else ""
    return f"- [{uri}{score_text}] {content.strip()}"


def format_memory_context(items: list[tuple[str, Any]]) -> str | None:
    if not items:
        return None

    grouped: dict[str, list[str]] = {
        "preferences": [],
        "profile": [],
        "entities": [],
        "events": [],
        "other": [],
    }

    for _context_type, item in items:
        uri = str(item_value(item, "uri", "") or "")
        line = _format_memory_item(item)
        if not line:
            continue
        grouped[_memory_category(uri)].append(line)

    sections: list[str] = []
    labels = {
        "profile": "用户画像",
        "preferences": "用户偏好",
        "entities": "相关实体",
        "events": "重要事件",
        "other": "其他记忆",
    }
    for key, label in labels.items():
        lines = grouped[key]
        if lines:
            sections.append(f"### {label}\n" + "\n".join(lines))

    if not sections:
        return None
    return "\n\n".join(sections)


async def find_user_memories(user_id: str, query: str) -> str | None:
    if not is_openviking_available() or not query.strip():
        return None

    client = get_local_client()
    if client is None:
        return None

    ctx = request_context(user_id)
    target_uri = f"viking://user/{user_id}/memories/"
    try:
        result = await client.service.search.find(
            query=query,
            ctx=ctx,
            target_uri=target_uri,
            limit=settings.openviking_find_limit,
            score_threshold=settings.openviking_find_score_threshold,
        )
        items = list(iter_result_items(result, context_types=("memory",)))
        return format_memory_context(items)
    except Exception:
        logger.exception(
            "OpenViking memory search failed user_id=%s query=%r",
            user_id,
            query[:200],
        )
        return None

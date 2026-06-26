import json
from collections import defaultdict

from langchain_core.documents import Document

SKIP_BLOCK_TYPES = frozenset({
    "header",
    "footer",
    "page_number",
    "page_header",
    "page_footer",
    "page_aside_text",
    "page_footnote",
    "aside_text",
})


def _normalize_blocks(raw: list | dict) -> list[dict]:
    if isinstance(raw, dict):
        for key in ("content_list", "data", "blocks", "items"):
            value = raw.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return []
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    return []


def _block_page_idx(block: dict) -> int:
    page_idx = block.get("page_idx")
    if page_idx is None:
        page_idx = block.get("page_number")
    try:
        return max(0, int(page_idx))
    except (TypeError, ValueError):
        return 0


def _join_lines(parts: list[str]) -> str:
    return "\n\n".join(part.strip() for part in parts if part and part.strip())


def _block_to_text(block: dict) -> str:
    block_type = str(block.get("type") or "text").lower()

    if block_type in SKIP_BLOCK_TYPES:
        return ""

    if block_type in {"text", "title", "ref_text"}:
        return str(block.get("text") or "").strip()

    if block_type == "equation":
        return str(block.get("text") or block.get("math_content") or "").strip()

    if block_type == "code":
        body = block.get("code_body") or block.get("code_content")
        if isinstance(body, list):
            body = "\n".join(str(line) for line in body)
        return str(body or "").strip()

    if block_type == "list":
        items = block.get("list_items") or []
        if isinstance(items, list):
            return "\n".join(f"- {str(item).strip()}" for item in items if str(item).strip())
        return ""

    if block_type in {"table", "chart"}:
        for key in ("table_body", "markdown", "md", "text", "html"):
            value = block.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        caption = block.get("table_caption")
        if isinstance(caption, list):
            caption = " ".join(str(c) for c in caption)
        body = str(block.get("table_body") or "")
        if caption and body:
            return f"{caption}\n\n{body}".strip()
        return body.strip()

    if block_type.startswith("image"):
        parts: list[str] = []
        for key in ("image_caption", "image_footnote", "text"):
            value = block.get(key)
            if isinstance(value, list):
                value = " ".join(str(v) for v in value)
            if value:
                parts.append(str(value).strip())
        return _join_lines(parts)

    for key in ("text", "markdown", "md"):
        value = block.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def parse_content_list(raw: list | dict, *, source: str = "mineru") -> list[Document]:
    blocks = _normalize_blocks(raw)
    if not blocks:
        return []

    pages: dict[int, list[str]] = defaultdict(list)
    for block in blocks:
        text = _block_to_text(block)
        if not text:
            continue
        pages[_block_page_idx(block)].append(text)

    docs: list[Document] = []
    for page_idx in sorted(pages):
        content = _join_lines(pages[page_idx])
        if not content:
            continue
        docs.append(
            Document(
                page_content=content,
                metadata={
                    "parse_source": source,
                    "page_number": page_idx + 1,
                },
            )
        )
    return docs


def parse_content_list_json(text: str, *, source: str = "mineru") -> list[Document]:
    raw = json.loads(text)
    return parse_content_list(raw, source=source)

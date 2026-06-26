import re
from html.parser import HTMLParser

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.parsing.chunking_config import get_chunking_overrides, get_strategy

TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
TABLE_SEP_RE = re.compile(r"^\s*\|[\s:\-|]+\|\s*$")
TABLE_HTML_RE = re.compile(r"<table\b[^>]*>.*?</table>", re.DOTALL | re.IGNORECASE)

PAGE_METADATA_KEYS = ("page_number", "sheet_name", "slide_index")


def _is_table_row(line: str) -> bool:
    return bool(TABLE_ROW_RE.match(line))


def _is_table_separator(line: str) -> bool:
    return bool(TABLE_SEP_RE.match(line))


def _copy_page_metadata(source: dict, target: dict) -> None:
    for key in PAGE_METADATA_KEYS:
        if key in source:
            target[key] = source[key]


class _TableHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.header_rows: list[list[str]] = []
        self.data_rows: list[list[str]] = []
        self._current_section = "data"
        self._current_row: list[str] | None = None
        self._current_cell: list[str] = []
        self.has_span_cells = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "thead":
            self._current_section = "header"
        elif tag == "tbody":
            self._current_section = "data"
        elif tag == "tr":
            self._current_row = []
        elif tag in ("td", "th"):
            self._current_cell = []
            attr_map = {k.lower(): v for k, v in attrs}
            if attr_map.get("rowspan") or attr_map.get("colspan"):
                self.has_span_cells = True

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in ("td", "th") and self._current_row is not None:
            self._current_row.append("".join(self._current_cell).strip())
            self._current_cell = []
        elif tag == "tr" and self._current_row is not None:
            if self._current_section == "header":
                self.header_rows.append(self._current_row)
            else:
                self.data_rows.append(self._current_row)
            self._current_row = None
        elif tag == "thead":
            self._current_section = "data"

    def handle_data(self, data: str) -> None:
        if self._current_row is not None:
            self._current_cell.append(data)


def _html_table_has_spans(html: str) -> bool:
    return bool(re.search(r"<t[dh][^>]*(rowspan|colspan)\s*=", html, re.IGNORECASE))


def _parse_html_table_rows(html: str) -> tuple[list[str], list[list[str]], bool]:
    parser = _TableHTMLParser()
    parser.feed(html)
    parser.close()

    if parser.header_rows:
        header = parser.header_rows[0]
        data_rows = parser.data_rows
    elif parser.data_rows:
        header = parser.data_rows[0]
        data_rows = parser.data_rows[1:]
    else:
        return [], [], parser.has_span_cells

    return header, data_rows, parser.has_span_cells


def _rows_to_markdown_table(header: list[str], data_rows: list[list[str]]) -> str:
    if not header:
        return ""
    sep = "| " + " | ".join("---" for _ in header) + " |"
    header_line = "| " + " | ".join(header) + " |"
    body = ["| " + " | ".join(row) + " |" for row in data_rows]
    return "\n".join([header_line, sep, *body])


def _chunk_html_table(html: str, rows_per_chunk: int) -> list[str]:
    if _html_table_has_spans(html):
        header, data_rows, _ = _parse_html_table_rows(html)
        if header:
            return [_rows_to_markdown_table(header, data_rows)]
        return [html.strip()] if html.strip() else []

    header, data_rows, _ = _parse_html_table_rows(html)
    if not header:
        return [html.strip()] if html.strip() else []
    if not data_rows:
        return [_rows_to_markdown_table(header, [])]

    chunks: list[str] = []
    for start in range(0, len(data_rows), rows_per_chunk):
        group = data_rows[start : start + rows_per_chunk]
        chunks.append(_rows_to_markdown_table(header, group))
    return chunks


def _split_markdown_lines(text: str) -> list[tuple[str, str]]:
    lines = text.splitlines()
    blocks: list[tuple[str, str]] = []
    current: list[str] = []

    def flush() -> None:
        nonlocal current
        if current:
            blocks.append(("text", "\n".join(current).strip()))
            current = []

    i = 0
    while i < len(lines):
        line = lines[i]
        if _is_table_row(line):
            flush()
            table_lines = [line]
            i += 1
            while i < len(lines) and (_is_table_row(lines[i]) or _is_table_separator(lines[i])):
                table_lines.append(lines[i])
                i += 1
            blocks.append(("table", "\n".join(table_lines).strip()))
            continue
        current.append(line)
        i += 1
    flush()
    return blocks


def split_text_blocks(text: str) -> list[tuple[str, str]]:
    """Split content into ('text'|'table'|'html_table', content) blocks."""
    if not text.strip():
        return []

    blocks: list[tuple[str, str]] = []
    pos = 0
    for match in TABLE_HTML_RE.finditer(text):
        before = text[pos : match.start()]
        if before.strip():
            blocks.extend(_split_markdown_lines(before))
        blocks.append(("html_table", match.group(0).strip()))
        pos = match.end()

    tail = text[pos:]
    if tail.strip():
        blocks.extend(_split_markdown_lines(tail))
    return blocks


def _absorb_table_caption(blocks: list[tuple[str, str]], max_caption_chars: int) -> list[tuple[str, str, str | None]]:
    """Return (block_type, content, caption) tuples."""
    enriched: list[tuple[str, str, str | None]] = []
    for idx, (block_type, content) in enumerate(blocks):
        caption: str | None = None
        if block_type in {"table", "html_table"} and idx > 0:
            prev_type, prev_content = blocks[idx - 1]
            if prev_type == "text" and len(prev_content) <= max_caption_chars:
                caption = prev_content.strip()
                if enriched and enriched[-1][0] == "text" and enriched[-1][1] == prev_content:
                    enriched.pop()
        enriched.append((block_type, content, caption))
    return enriched


def _chunk_table(table_md: str, rows_per_chunk: int) -> list[str]:
    lines = [ln for ln in table_md.splitlines() if ln.strip()]
    if not lines:
        return []

    header = lines[0]
    if len(lines) == 1:
        return [header]

    if len(lines) >= 2 and _is_table_separator(lines[1]):
        separator = lines[1]
        data_rows = lines[2:]
    else:
        separator = "| " + " | ".join("---" for _ in header.strip("|").split("|")) + " |"
        data_rows = lines[1:]

    if not data_rows:
        return [f"{header}\n{separator}"]

    chunks: list[str] = []
    for start in range(0, len(data_rows), rows_per_chunk):
        group = data_rows[start : start + rows_per_chunk]
        chunks.append("\n".join([header, separator, *group]))
    return chunks


def _build_table_document(
    table_chunk: str,
    base_meta: dict,
    caption: str | None,
) -> Document:
    meta = {**base_meta, "block_type": "table"}
    if caption:
        meta["table_caption"] = caption
        content = f"{caption}\n\n{table_chunk}"
    else:
        content = table_chunk
    return Document(page_content=content, metadata=meta)


def chunk_documents(raw_docs: list[Document]) -> list[Document]:
    if not raw_docs:
        return []

    overrides = get_chunking_overrides()
    chunk_size = int(overrides.get("chunk_size") or settings.chunk_size)
    chunk_overlap = int(overrides.get("chunk_overlap") or settings.chunk_overlap)
    rows_per_chunk = int(overrides.get("table_chunk_rows") or settings.table_chunk_rows)
    caption_max_chars = int(overrides.get("table_caption_max_chars") or 200)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    output: list[Document] = []

    for doc in raw_docs:
        if (
            doc.metadata.get("content_type") == "audio_transcript"
            and "segment_index" in doc.metadata
            and get_strategy("audio_transcript") == "segment_preserve"
        ):
            if len(doc.page_content) <= chunk_size:
                output.append(doc)
            else:
                sub_chunks = splitter.split_documents([doc])
                for sub_idx, sub_chunk in enumerate(sub_chunks):
                    sub_chunk.metadata = {**doc.metadata, "sub_segment_index": sub_idx}
                    output.append(sub_chunk)
            continue

        text = doc.page_content
        base_meta = dict(doc.metadata)
        blocks = _absorb_table_caption(split_text_blocks(text), caption_max_chars)

        for block_type, block_content, caption in blocks:
            if not block_content.strip():
                continue
            strategy = get_strategy(block_type)
            if block_type == "table" and strategy == "row_split_header_repeat":
                for table_chunk in _chunk_table(block_content, rows_per_chunk):
                    chunk_doc = _build_table_document(table_chunk, base_meta, caption)
                    _copy_page_metadata(doc.metadata, chunk_doc.metadata)
                    output.append(chunk_doc)
            elif block_type == "html_table" and strategy == "html_row_split":
                for table_chunk in _chunk_html_table(block_content, rows_per_chunk):
                    chunk_doc = _build_table_document(table_chunk, base_meta, caption)
                    _copy_page_metadata(doc.metadata, chunk_doc.metadata)
                    output.append(chunk_doc)
            else:
                subdoc = Document(page_content=block_content, metadata={**base_meta, "block_type": "text"})
                for split_chunk in splitter.split_documents([subdoc]):
                    _copy_page_metadata(doc.metadata, split_chunk.metadata)
                    output.append(split_chunk)
    return output

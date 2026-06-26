from langchain_core.documents import Document

from app.parsing.chunking import chunk_documents, split_text_blocks


def test_split_text_blocks_detects_table():
    text = "intro\n\n| A | B |\n| --- | --- |\n| 1 | 2 |\n\noutro"
    blocks = split_text_blocks(text)
    types = [b[0] for b in blocks]
    assert "table" in types
    assert "text" in types


def test_split_text_blocks_detects_html_table():
    text = 'intro\n<table><thead><tr><th>Name</th><th>Value</th></tr></thead><tbody><tr><td>a</td><td>1</td></tr></tbody></table>\noutro'
    blocks = split_text_blocks(text)
    types = [b[0] for b in blocks]
    assert "html_table" in types
    assert "text" in types


def test_html_table_chunks_keep_header():
    rows = "".join(f"<tr><td>row{i}</td><td>val{i}</td></tr>" for i in range(10))
    html = f"<table><tr><th>Name</th><th>Value</th></tr>{rows}</table>"
    doc = Document(page_content=f"before\n{html}\nafter")
    chunks = chunk_documents([doc])
    table_chunks = [c for c in chunks if c.metadata.get("block_type") == "table"]
    assert len(table_chunks) >= 2
    for chunk in table_chunks:
        assert "| Name | Value |" in chunk.page_content
        assert "| --- | --- |" in chunk.page_content


def test_table_chunks_keep_header():
    table = "| Name | Value |\n| --- | --- |\n" + "\n".join(
        f"| row{i} | val{i} |" for i in range(10)
    )
    doc = Document(page_content=f"before\n\n{table}\n\nafter")
    chunks = chunk_documents([doc])
    table_chunks = [c for c in chunks if c.metadata.get("block_type") == "table"]
    assert len(table_chunks) >= 2
    for chunk in table_chunks:
        assert "| Name | Value |" in chunk.page_content
        assert "| --- | --- |" in chunk.page_content


def test_table_without_separator_uses_first_row_as_header():
    table = "| Name | Value |\n| a | 1 |\n| b | 2 |"
    doc = Document(page_content=table)
    chunks = chunk_documents([doc])
    assert len(chunks) == 1
    assert "| Name | Value |" in chunks[0].page_content


def test_table_caption_is_absorbed():
    table = "| Name | Value |\n| --- | --- |\n| a | 1 |"
    doc = Document(page_content=f"Sales summary\n\n{table}")
    chunks = chunk_documents([doc])
    table_chunks = [c for c in chunks if c.metadata.get("block_type") == "table"]
    assert table_chunks
    assert table_chunks[0].metadata.get("table_caption") == "Sales summary"
    assert "Sales summary" in table_chunks[0].page_content


def test_html_table_with_rowspan_stays_atomic():
    html = '<table><tr><th>A</th><th>B</th></tr><tr><td rowspan="2">1</td><td>2</td></tr><tr><td>3</td></tr></table>'
    doc = Document(page_content=html)
    chunks = chunk_documents([doc])
    assert len(chunks) == 1
    assert "| A | B |" in chunks[0].page_content


def test_page_number_propagates_to_chunks():
    table = "| Name | Value |\n| --- | --- |\n| a | 1 |"
    doc = Document(page_content=table, metadata={"page_number": 3, "sheet_name": "Sheet1"})
    chunks = chunk_documents([doc])
    assert chunks[0].metadata.get("page_number") == 3
    assert chunks[0].metadata.get("sheet_name") == "Sheet1"

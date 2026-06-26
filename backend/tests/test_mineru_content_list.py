import io
import json
import zipfile
from unittest.mock import MagicMock, patch

import pytest

from app.parsing.mineru_client import MinerUError, parse_file_to_documents, parse_file_to_markdown
from app.parsing.mineru_content_list import parse_content_list


def test_parse_content_list_groups_by_page():
    raw = [
        {"type": "text", "text": "Page one intro", "page_idx": 0},
        {"type": "text", "text": "Page one body", "page_idx": 0},
        {"type": "text", "text": "Page two start", "page_idx": 1},
        {"type": "header", "text": "Header ignored", "page_idx": 1},
    ]
    docs = parse_content_list(raw)
    assert len(docs) == 2
    assert docs[0].metadata["page_number"] == 1
    assert "Page one intro" in docs[0].page_content
    assert "Page one body" in docs[0].page_content
    assert docs[1].metadata["page_number"] == 2
    assert "Page two start" in docs[1].page_content
    assert "Header ignored" not in docs[1].page_content


def test_parse_content_list_table_block():
    raw = [
        {
            "type": "table",
            "page_idx": 0,
            "table_body": "| A | B |\n| --- | --- |\n| 1 | 2 |",
        }
    ]
    docs = parse_content_list(raw)
    assert len(docs) == 1
    assert "| A | B |" in docs[0].page_content


def test_parse_content_list_empty():
    assert parse_content_list([]) == []


def test_parse_file_to_documents_prefers_content_list(tmp_path):
    content_list = [
        {"type": "text", "text": "First page", "page_idx": 0},
        {"type": "text", "text": "Second page", "page_idx": 1},
    ]
    md_content = "# Title\n\nfallback markdown"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("demo_content_list.json", json.dumps(content_list))
        zf.writestr("demo/full.md", md_content)
    zip_bytes = buf.getvalue()

    file_path = tmp_path / "demo.pdf"
    file_path.write_bytes(b"%PDF-1.4")

    def mock_post(url, **kwargs):
        response = MagicMock()
        response.raise_for_status = MagicMock()
        if url.endswith("/api/v4/file-urls/batch"):
            response.json.return_value = {
                "code": 0,
                "data": {"batch_id": "batch-1", "file_urls": ["https://upload.example/1"]},
            }
        return response

    def mock_put(url, **kwargs):
        response = MagicMock()
        response.status_code = 200
        return response

    def mock_get(url, **kwargs):
        response = MagicMock()
        response.raise_for_status = MagicMock()
        if "/extract-results/batch/" in url:
            response.json.return_value = {
                "code": 0,
                "data": {
                    "extract_result": [
                        {"state": "done", "full_zip_url": "https://cdn.example/out.zip"}
                    ]
                },
            }
        elif url.endswith("out.zip"):
            response.content = zip_bytes
        return response

    client = MagicMock()
    client.post.side_effect = mock_post
    client.put.side_effect = mock_put
    client.get.side_effect = mock_get
    client.__enter__ = MagicMock(return_value=client)
    client.__exit__ = MagicMock(return_value=False)

    with (
        patch("app.parsing.mineru_client.settings.mineru_api_token", "test-token"),
        patch("app.parsing.mineru_client.httpx.Client", return_value=client),
    ):
        docs, batch_id = parse_file_to_documents(str(file_path), data_id="doc-1")
        markdown, _ = parse_file_to_markdown(str(file_path), data_id="doc-1")

    assert batch_id == "batch-1"
    assert len(docs) == 2
    assert docs[0].metadata["page_number"] == 1
    assert docs[1].metadata["page_number"] == 2
    assert "First page" in markdown
    assert "Second page" in markdown


def test_parse_file_requires_token(tmp_path):
    file_path = tmp_path / "a.pdf"
    file_path.write_bytes(b"x")
    with patch("app.parsing.mineru_client.settings.mineru_api_token", ""):
        with pytest.raises(MinerUError):
            parse_file_to_documents(str(file_path), data_id="doc-1")

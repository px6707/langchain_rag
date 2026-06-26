import io
import json
import logging
import time
import zipfile
from pathlib import Path

import httpx
from langchain_core.documents import Document

from app.config import settings
from app.parsing.mineru_content_list import parse_content_list
from app.parsing.python_loaders import markdown_to_documents

logger = logging.getLogger(__name__)


class MinerUError(Exception):
    pass


class MinerUTimeoutError(MinerUError):
    pass


def _headers() -> dict[str, str]:
    if not settings.mineru_api_token:
        raise MinerUError("MINERU_API_TOKEN is not configured")
    return {
        "Authorization": f"Bearer {settings.mineru_api_token}",
        "Content-Type": "application/json",
        "Accept": "*/*",
    }


def _base_url() -> str:
    return settings.mineru_api_base.rstrip("/")


def parse_file_to_documents(file_path: str, *, data_id: str) -> tuple[list[Document], str]:
    """Upload file to MinerU cloud, poll batch result, return (documents, batch_id)."""
    path = Path(file_path)
    if not path.is_file():
        raise MinerUError(f"File not found: {file_path}")

    with httpx.Client(timeout=120.0) as client:
        batch_id, upload_url = _request_upload_urls(client, path.name, data_id)
        _upload_file(client, upload_url, file_path)
        result = _poll_batch_result(client, batch_id)
        content_list, markdown = _download_result_assets(client, result)
        if content_list is not None:
            docs = parse_content_list(content_list)
            if docs:
                return docs, batch_id
            logger.warning("MinerU content_list parsed empty for %s, falling back to markdown", file_path)
        if markdown:
            return markdown_to_documents(markdown, source="mineru"), batch_id
        raise MinerUError("MinerU result missing content_list and markdown")


def parse_file_to_markdown(file_path: str, *, data_id: str) -> tuple[str, str]:
    """Backward-compatible wrapper returning concatenated markdown text."""
    docs, batch_id = parse_file_to_documents(file_path, data_id=data_id)
    markdown = "\n\n".join(doc.page_content for doc in docs if doc.page_content.strip())
    return markdown, batch_id


def _request_upload_urls(client: httpx.Client, filename: str, data_id: str) -> tuple[str, str]:
    url = f"{_base_url()}/api/v4/file-urls/batch"
    payload = {
        "files": [{"name": filename, "data_id": data_id}],
        "model_version": settings.mineru_model_version,
    }
    response = client.post(url, headers=_headers(), json=payload)
    response.raise_for_status()
    body = response.json()
    if body.get("code") != 0:
        raise MinerUError(body.get("msg") or "Failed to request MinerU upload URLs")
    data = body.get("data") or {}
    batch_id = data.get("batch_id")
    file_urls = data.get("file_urls") or []
    if not batch_id or not file_urls:
        raise MinerUError("MinerU upload URL response missing batch_id or file_urls")
    return batch_id, file_urls[0]


def _upload_file(client: httpx.Client, upload_url: str, file_path: str) -> None:
    with open(file_path, "rb") as f:
        response = client.put(upload_url, content=f)
    if response.status_code != 200:
        raise MinerUError(f"MinerU file upload failed: HTTP {response.status_code}")


def _poll_batch_result(client: httpx.Client, batch_id: str) -> dict:
    url = f"{_base_url()}/api/v4/extract-results/batch/{batch_id}"
    deadline = time.time() + settings.mineru_poll_timeout_sec
    while time.time() < deadline:
        response = client.get(url, headers=_headers())
        response.raise_for_status()
        body = response.json()
        if body.get("code") != 0:
            raise MinerUError(body.get("msg") or "MinerU batch poll failed")
        data = body.get("data") or {}
        extract_results = data.get("extract_result") or []
        if not extract_results:
            time.sleep(settings.mineru_poll_interval_sec)
            continue
        item = extract_results[0]
        state = (item.get("state") or item.get("status") or "").lower()
        if state in {"done", "success", "completed"}:
            return item
        if state in {"failed", "error"}:
            raise MinerUError(item.get("err_msg") or item.get("message") or "MinerU parse failed")
        time.sleep(settings.mineru_poll_interval_sec)
    raise MinerUTimeoutError(f"MinerU batch {batch_id} timed out")


def _find_content_list_in_zip(zf: zipfile.ZipFile) -> list | dict | None:
    names = zf.namelist()
    candidates = [n for n in names if n.endswith("content_list.json") or n.endswith("_content_list.json")]
    candidates.sort(key=lambda n: (0 if n.endswith("_content_list.json") else 1, n))
    for name in candidates:
        try:
            return json.loads(zf.read(name).decode("utf-8", errors="replace"))
        except json.JSONDecodeError:
            logger.warning("Invalid content_list JSON in zip entry: %s", name)
    return None


def _find_markdown_in_zip(zf: zipfile.ZipFile) -> str | None:
    for name in zf.namelist():
        if name.endswith("full.md") or name.endswith("/full.md"):
            return zf.read(name).decode("utf-8", errors="replace")
    md_files = [n for n in zf.namelist() if n.endswith(".md")]
    if md_files:
        return zf.read(md_files[0]).decode("utf-8", errors="replace")
    return None


def _download_content_list_url(client: httpx.Client, result: dict) -> list | dict | None:
    for key in ("content_list_url", "content_list_json_url"):
        url = result.get(key)
        if not url:
            continue
        response = client.get(url)
        response.raise_for_status()
        return json.loads(response.text)
    embedded = result.get("content_list")
    if embedded is not None:
        return embedded
    return None


def _download_result_assets(client: httpx.Client, result: dict) -> tuple[list | dict | None, str | None]:
    content_list = _download_content_list_url(client, result)
    markdown_url = result.get("markdown_url")
    markdown: str | None = None
    if markdown_url:
        response = client.get(markdown_url)
        response.raise_for_status()
        markdown = response.text

    zip_url = result.get("full_zip_url")
    if zip_url:
        response = client.get(zip_url)
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            if content_list is None:
                content_list = _find_content_list_in_zip(zf)
            if markdown is None:
                markdown = _find_markdown_in_zip(zf)

    return content_list, markdown

import base64
import logging
from pathlib import Path

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

SUMMARY_PROMPT = (
    "请用中文简要总结这张图片的内容，包括主要对象、场景、文字信息和关键语义，"
    "便于后续文档检索。控制在 200 字以内。"
)


def _vlm_config() -> tuple[str, str, str]:
    api_base = settings.vlm_api_base or settings.llm_api_base
    api_key = settings.vlm_api_key or settings.llm_api_key
    model = settings.vlm_model or settings.llm_model
    if not api_key:
        raise ValueError("VLM API key is not configured")
    return api_base.rstrip("/"), api_key, model


def _image_media_type(path: Path) -> str:
    ext = path.suffix.lower()
    mapping = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".tiff": "image/tiff",
    }
    return mapping.get(ext, "image/jpeg")


def summarize_image(file_path: str) -> str:
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(file_path)

    api_base, api_key, model = _vlm_config()
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    media_type = _image_media_type(path)
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": SUMMARY_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{media_type};base64,{encoded}"},
                    },
                ],
            }
        ],
        "max_tokens": 512,
    }
    url = f"{api_base}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=120.0) as client:
        response = client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        body = response.json()

    choices = body.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts).strip()
    return str(content).strip()

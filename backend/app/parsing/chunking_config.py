from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from app.config import settings

DEFAULT_CONFIG: dict[str, Any] = {
    "strategies": {
        "default": "recursive_character",
        "text": "recursive_character",
        "table": "row_split_header_repeat",
        "html_table": "html_row_split",
        "audio_transcript": "segment_preserve",
    },
    "overrides": {
        "chunk_size": 500,
        "chunk_overlap": 50,
        "table_chunk_rows": 8,
        "table_caption_max_chars": 200,
    },
}


@lru_cache(maxsize=1)
def load_chunking_config() -> dict[str, Any]:
    path = Path(settings.chunking_config_path)
    if not path.is_file():
        return DEFAULT_CONFIG
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    merged = {
        "strategies": {**DEFAULT_CONFIG["strategies"], **(data.get("strategies") or {})},
        "overrides": {**DEFAULT_CONFIG["overrides"], **(data.get("overrides") or {})},
    }
    return merged


def get_strategy(block_type: str) -> str:
    config = load_chunking_config()
    strategies = config.get("strategies") or {}
    return str(strategies.get(block_type) or strategies.get("default") or "recursive_character")


def get_chunking_overrides() -> dict[str, Any]:
    config = load_chunking_config()
    return dict(config.get("overrides") or {})

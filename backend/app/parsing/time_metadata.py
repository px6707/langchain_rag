"""Extract time and media metadata from chunk metadata for API responses."""

from __future__ import annotations

from typing import Any

from app.parsing.router import get_file_type_label


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_time_metadata(metadata: dict) -> dict[str, Any]:
    """Return file_type, content_type, timestamp_sec, start_sec, end_sec from chunk metadata."""
    filename = str(metadata.get("filename", ""))
    file_type = metadata.get("file_type")
    if not file_type and filename:
        try:
            file_type = get_file_type_label(filename)
        except ValueError:
            file_type = None

    content_type = metadata.get("content_type")
    if content_type is not None:
        content_type = str(content_type)

    timestamp_sec = _optional_float(metadata.get("timestamp_sec"))
    start_sec = _optional_float(metadata.get("start_sec"))
    if start_sec is None:
        start_sec = _optional_float(metadata.get("asr_start_sec"))
    end_sec = _optional_float(metadata.get("end_sec"))
    if end_sec is None:
        end_sec = _optional_float(metadata.get("asr_end_sec"))

    result: dict[str, Any] = {}
    if file_type:
        result["file_type"] = str(file_type)
    if content_type:
        result["content_type"] = content_type
    if timestamp_sec is not None:
        result["timestamp_sec"] = timestamp_sec
    if start_sec is not None:
        result["start_sec"] = start_sec
    if end_sec is not None:
        result["end_sec"] = end_sec
    return result

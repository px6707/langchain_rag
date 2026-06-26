from pathlib import Path

import filetype

from app.parsing.router import ALLOWED_EXTENSIONS

# Extension -> acceptable MIME types (filetype guess or declared). Empty set = magic optional.
EXT_MIME_MAP: dict[str, set[str]] = {
    ".pdf": {"application/pdf"},
    ".png": {"image/png"},
    ".jpg": {"image/jpeg"},
    ".jpeg": {"image/jpeg"},
    ".jp2": {"image/jp2", "image/jpx"},
    ".webp": {"image/webp"},
    ".gif": {"image/gif"},
    ".bmp": {"image/bmp", "image/x-ms-bmp"},
    ".tiff": {"image/tiff"},
    ".doc": {"application/msword"},
    ".docx": {"application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/zip"},
    ".ppt": {"application/vnd.ms-powerpoint"},
    ".pptx": {"application/vnd.openxmlformats-officedocument.presentationml.presentation", "application/zip"},
    ".xls": {"application/vnd.ms-excel"},
    ".xlsx": {"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/zip"},
    ".txt": set(),
    ".md": set(),
    ".mp3": {"audio/mpeg", "audio/mp3"},
    ".wav": {"audio/wav", "audio/x-wav", "audio/vnd.wave"},
    ".m4a": {"audio/mp4", "audio/x-m4a", "video/mp4"},
    ".aac": {"audio/aac", "audio/x-aac"},
    ".ogg": {"audio/ogg", "application/ogg"},
    ".flac": {"audio/flac", "audio/x-flac"},
    ".wma": {"audio/x-ms-wma"},
    ".amr": {"audio/amr"},
    ".mp4": {"video/mp4", "application/mp4"},
    ".mkv": {"video/x-matroska", "video/webm"},
    ".mov": {"video/quicktime"},
    ".webm": {"video/webm"},
    ".avi": {"video/x-msvideo", "video/avi"},
    ".flv": {"video/x-flv"},
    ".wmv": {"video/x-ms-wmv"},
    ".m4v": {"video/mp4", "video/x-m4v"},
}

MAGIC_OPTIONAL_EXTENSIONS = {ext for ext, mimes in EXT_MIME_MAP.items() if not mimes}


def _is_probably_text(content: bytes) -> bool:
    if not content:
        return True
    sample = content[:8192]
    if b"\x00" in sample:
        return False
    try:
        sample.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def validate_upload(filename: str, content: bytes) -> None:
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}")

    allowed_mimes = EXT_MIME_MAP.get(ext)
    if allowed_mimes is None:
        raise ValueError(f"Unsupported file type: {ext}")

    if ext in MAGIC_OPTIONAL_EXTENSIONS:
        if _is_probably_text(content):
            return
        raise ValueError(f"File content does not look like plain text for extension {ext}")

    kind = filetype.guess(content)
    if kind is None:
        raise ValueError(f"Could not detect file type for extension {ext}")

    if kind.mime not in allowed_mimes:
        raise ValueError(
            f"File content ({kind.mime}) does not match extension {ext}. "
            f"Expected one of: {', '.join(sorted(allowed_mimes))}"
        )

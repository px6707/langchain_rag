from pathlib import Path

MINERU_EXTENSIONS = {
    ".pdf",
    ".png", ".jpg", ".jpeg", ".jp2", ".webp", ".gif", ".bmp", ".tiff",
    ".docx", ".pptx", ".xlsx",
}

LEGACY_OFFICE_EXTENSIONS = {".doc", ".xls", ".ppt"}

PYTHON_ONLY_EXTENSIONS = {".txt", ".md"}

PYTHON_FALLBACK_EXTENSIONS = {".pdf", ".docx", ".pptx", ".xlsx"}

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".wma", ".amr"}

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".flv", ".wmv", ".m4v"}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".jp2", ".webp", ".gif", ".bmp", ".tiff"}

ALLOWED_EXTENSIONS = (
    MINERU_EXTENSIONS
    | LEGACY_OFFICE_EXTENSIONS
    | PYTHON_ONLY_EXTENSIONS
    | AUDIO_EXTENSIONS
    | VIDEO_EXTENSIONS
)

ParseStrategy = str  # mineru | python | asr | video | image


def get_extension(filename: str) -> str:
    return Path(filename).suffix.lower()


def classify_file(filename: str) -> ParseStrategy:
    ext = get_extension(filename)
    if ext in VIDEO_EXTENSIONS:
        return "video"
    if ext in AUDIO_EXTENSIONS:
        return "asr"
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in PYTHON_ONLY_EXTENSIONS:
        return "python"
    if ext in LEGACY_OFFICE_EXTENSIONS:
        return "python"
    if ext in MINERU_EXTENSIONS:
        return "mineru"
    raise ValueError(f"Unsupported file type: {ext}")


def supports_python_fallback(filename: str) -> bool:
    return get_extension(filename) in PYTHON_FALLBACK_EXTENSIONS


def get_file_type_label(filename: str) -> str:
    ext = get_extension(filename)
    if ext in VIDEO_EXTENSIONS:
        return "video"
    if ext in AUDIO_EXTENSIONS:
        return "audio"
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext == ".pdf":
        return "pdf"
    if ext in {".txt", ".md"}:
        return ext.lstrip(".")
    if ext == ".docx":
        return "docx"
    if ext == ".doc":
        return "doc"
    if ext in {".ppt", ".pptx"}:
        return "pptx" if ext == ".pptx" else "ppt"
    if ext in {".xls", ".xlsx"}:
        return "xlsx" if ext == ".xlsx" else "xls"
    return ext.lstrip(".") or "unknown"

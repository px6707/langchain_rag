import pytest

from app.parsing.router import (
    ALLOWED_EXTENSIONS,
    classify_file,
    supports_python_fallback,
)


def test_classify_pdf_as_mineru():
    assert classify_file("report.pdf") == "mineru"


def test_classify_txt_as_python():
    assert classify_file("notes.txt") == "python"


def test_classify_mp4_as_video():
    assert classify_file("clip.mp4") == "video"


def test_classify_png_as_image():
    assert classify_file("scan.png") == "image"


def test_classify_mp3_as_asr():
    assert classify_file("audio.mp3") == "asr"


def test_classify_legacy_office_as_python():
    assert classify_file("legacy.doc") == "python"
    assert classify_file("legacy.xls") == "python"
    assert classify_file("legacy.ppt") == "python"


def test_classify_docx_as_mineru():
    assert classify_file("report.docx") == "mineru"


def test_python_fallback_includes_pptx_xlsx():
    assert supports_python_fallback("deck.pptx") is True
    assert supports_python_fallback("data.xlsx") is True


def test_python_fallback_excludes_legacy_office():
    assert supports_python_fallback("a.pdf") is True
    assert supports_python_fallback("a.docx") is True
    assert supports_python_fallback("a.doc") is False
    assert supports_python_fallback("a.ppt") is False
    assert supports_python_fallback("a.xls") is False


def test_allowed_extensions_include_multimodal():
    assert ".mp4" in ALLOWED_EXTENSIONS
    assert ".pptx" in ALLOWED_EXTENSIONS
    assert ".png" in ALLOWED_EXTENSIONS

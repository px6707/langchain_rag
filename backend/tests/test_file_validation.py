import io

import pytest
from PIL import Image

from app.parsing.file_validation import validate_upload

PDF_MAGIC = b"%PDF-1.4\n%fake pdf content"


def test_validate_pdf_with_correct_magic():
    validate_upload("report.pdf", PDF_MAGIC)


def test_validate_pdf_extension_mismatch():
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), (255, 0, 0)).save(buf, format="PNG")
    with pytest.raises(ValueError, match="does not match extension|Could not detect"):
        validate_upload("report.pdf", buf.getvalue())


def test_validate_txt_plain_text():
    validate_upload("notes.txt", b"hello world\nline two")


def test_validate_txt_rejects_binary():
    with pytest.raises(ValueError, match="plain text"):
        validate_upload("notes.txt", b"\x00\x01binary")

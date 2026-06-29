import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from app.routers.documents import get_document_file


def test_get_document_file_not_found():
    mock_db = AsyncMock()
    service = MagicMock()
    service.get_document = AsyncMock(return_value=None)
    with patch("app.routers.documents.DocumentService", return_value=service):
        with pytest.raises(HTTPException) as exc:
            asyncio.run(get_document_file(uuid.uuid4(), db=mock_db, _user=MagicMock()))
    assert exc.value.status_code == 404


def test_get_document_file_rejects_non_media(tmp_path):
    doc_id = uuid.uuid4()
    doc = MagicMock()
    doc.file_type = "pdf"
    doc.file_path = str(tmp_path / "file.pdf")
    doc.filename = "file.pdf"

    mock_db = AsyncMock()
    service = MagicMock()
    service.get_document = AsyncMock(return_value=doc)
    with patch("app.routers.documents.DocumentService", return_value=service):
        with pytest.raises(HTTPException) as exc:
            asyncio.run(get_document_file(doc_id, db=mock_db, _user=MagicMock()))
    assert exc.value.status_code == 400


def test_get_document_file_returns_file_response(tmp_path):
    doc_id = uuid.uuid4()
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"fake-video")

    doc = MagicMock()
    doc.file_type = "video"
    doc.file_path = str(video_path)
    doc.filename = "lecture.mp4"

    mock_db = AsyncMock()
    service = MagicMock()
    service.get_document = AsyncMock(return_value=doc)
    with patch("app.routers.documents.DocumentService", return_value=service):
        response = asyncio.run(get_document_file(doc_id, db=mock_db, _user=MagicMock()))

    assert response.path == str(video_path)
    assert "video" in response.media_type
    assert response.filename == "lecture.mp4"


def test_get_document_file_missing_on_disk(tmp_path):
    doc_id = uuid.uuid4()
    doc = MagicMock()
    doc.file_type = "video"
    doc.file_path = str(tmp_path / "missing.mp4")
    doc.filename = "missing.mp4"

    mock_db = AsyncMock()
    service = MagicMock()
    service.get_document = AsyncMock(return_value=doc)
    with patch("app.routers.documents.DocumentService", return_value=service):
        with pytest.raises(HTTPException) as exc:
            asyncio.run(get_document_file(doc_id, db=mock_db, _user=MagicMock()))
    assert exc.value.status_code == 404

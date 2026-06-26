import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.documents import Document

from app.config import settings
from app.models import Document as DocumentModel
from app.parsing.chunking import chunk_documents


def test_long_asr_segment_is_split_with_time_metadata():
    long_text = "word " * 200
    doc = Document(
        page_content=long_text.strip(),
        metadata={
            "content_type": "audio_transcript",
            "segment_index": 0,
            "start_sec": 0.0,
            "end_sec": 120.0,
        },
    )
    chunks = chunk_documents([doc])
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.metadata.get("start_sec") == 0.0
        assert chunk.metadata.get("end_sec") == 120.0
        assert "sub_segment_index" in chunk.metadata
        assert len(chunk.page_content) <= settings.chunk_size + 50


def test_short_asr_segment_not_split():
    doc = Document(
        page_content="short transcript",
        metadata={
            "content_type": "audio_transcript",
            "segment_index": 0,
            "start_sec": 0.0,
            "end_sec": 5.0,
        },
    )
    chunks = chunk_documents([doc])
    assert len(chunks) == 1
    assert "sub_segment_index" not in chunks[0].metadata

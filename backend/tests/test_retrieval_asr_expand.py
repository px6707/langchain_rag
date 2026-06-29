from unittest.mock import patch

from langchain_core.documents import Document

from app.config import settings
from app.services.retrieval_service import _expand_same_asr_segment_chunks


def test_expand_same_asr_segment_chunks_adds_siblings():
    hit = Document(
        "spoken words",
        metadata={
            "document_id": "doc-1",
            "segment_index": 2,
            "start_sec": 120.0,
            "end_sec": 180.0,
            "relevance_score": 0.88,
            "filename": "talk.mp4",
        },
    )
    frame = Document(
        "slide text",
        metadata={
            "document_id": "doc-1",
            "asr_segment_index": 2,
            "chunk_index": 9,
            "filename": "talk.mp4",
        },
    )

    with (
        patch.object(settings, "retrieval_asr_segment_expand_enabled", True),
        patch(
            "app.services.retrieval_service.fetch_chunks_by_asr_segment",
            return_value=[hit, frame],
        ) as mock_fetch,
    ):
        expanded = _expand_same_asr_segment_chunks([hit])

    mock_fetch.assert_called_once_with("doc-1", 2, max_chunks=settings.retrieval_asr_segment_expand_max_chunks)
    chunk_indices = sorted(doc.metadata.get("chunk_index", -1) for doc in expanded)
    assert 9 in chunk_indices


def test_expand_asr_segment_skips_without_index():
    doc = Document("text", metadata={"document_id": "doc-1", "chunk_index": 0})
    with patch.object(settings, "retrieval_asr_segment_expand_enabled", True):
        assert _expand_same_asr_segment_chunks([doc]) == [doc]


def test_expand_asr_segment_disabled():
    doc = Document(
        "text",
        metadata={"document_id": "doc-1", "segment_index": 1, "relevance_score": 0.5},
    )
    with (
        patch.object(settings, "retrieval_asr_segment_expand_enabled", False),
        patch("app.services.retrieval_service.fetch_chunks_by_asr_segment") as mock_fetch,
    ):
        assert _expand_same_asr_segment_chunks([doc]) == [doc]
    mock_fetch.assert_not_called()

from langchain_core.documents import Document

from app.services.retrieval_service import _build_sources_and_context


def test_build_sources_includes_time_fields():
    docs = [
        Document(
            "slide title",
            metadata={
                "document_id": "doc-1",
                "chunk_index": 3,
                "filename": "course.mp4",
                "file_type": "video",
                "content_type": "frame_ocr",
                "timestamp_sec": 42.0,
                "relevance_score": 0.91,
            },
        ),
        Document(
            "spoken words",
            metadata={
                "document_id": "doc-1",
                "chunk_index": 1,
                "filename": "course.mp4",
                "file_type": "video",
                "content_type": "audio_transcript",
                "start_sec": 30.0,
                "end_sec": 45.0,
            },
        ),
    ]
    sources, context = _build_sources_and_context(docs)
    assert len(sources) == 2
    assert sources[0].timestamp_sec == 42.0
    assert sources[0].file_type == "video"
    assert sources[0].content_type == "frame_ocr"
    assert sources[1].start_sec == 30.0
    assert sources[1].end_sec == 45.0
    assert context is not None
    assert "[doc-1#3]" in context

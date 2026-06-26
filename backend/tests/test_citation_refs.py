from langchain_core.documents import Document

from app.services.retrieval_service import _build_sources_and_context


def test_build_sources_and_context_ref_format():
    doc_id = "550e8400-e29b-41d4-a716-446655440000"
    doc = Document(
        "chunk body text",
        metadata={
            "document_id": doc_id,
            "chunk_index": 2,
            "filename": "report.pdf",
            "rerank_score": 0.88,
        },
    )

    sources, context = _build_sources_and_context([doc])

    assert len(sources) == 1
    source = sources[0]
    assert source.document_id == doc_id
    assert source.chunk_index == 2
    assert source.ref_id == f"{doc_id}#2"
    assert source.filename == "report.pdf"
    assert source.content == "chunk body text"
    assert source.score == 0.88

    assert f"[{doc_id}#2] report.pdf" in context
    assert "chunk body text" in context


def test_build_sources_and_context_fallback_chunk_index():
    doc = Document("legacy chunk", metadata={"filename": "old.pdf"})

    sources, context = _build_sources_and_context([doc])

    assert sources[0].chunk_index == 0
    assert sources[0].document_id == "unknown"
    assert "#0" in sources[0].ref_id
    assert "#0]" in context

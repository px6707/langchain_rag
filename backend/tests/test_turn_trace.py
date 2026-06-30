"""Tests for turn-level LangSmith metadata collection."""

from langchain_core.documents import Document

from app.observability.turn_trace import TurnTraceContext
from app.schemas.retrieval import RetrievalPlan
from app.services.grounding_service import GroundingResult


def test_turn_trace_to_metadata_full():
    ctx = TurnTraceContext(
        user_id="user-1",
        session_id="sess-1",
        is_admin=False,
        env="dev",
    )
    ctx.set_retrieval_plan(
        RetrievalPlan(
            action="retrieve",
            strategy="none",
            standalone_query="什么是 RAG",
            reason="需要查文档",
        )
    )
    ctx.set_chunks(
        [
            Document(
                "chunk text",
                metadata={
                    "document_id": "doc-a",
                    "chunk_index": 2,
                    "filename": "a.pdf",
                    "rerank_score": 0.91,
                },
            )
        ]
    )
    ctx.set_grounding(GroundingResult(status="supported", supported_ratio=0.9, claims=[]))

    meta = ctx.to_metadata()

    assert meta["user_id"] == "user-1"
    assert meta["session_id"] == "sess-1"
    assert meta["is_admin"] is False
    assert meta["document_ids_in_session"] == ["doc-a"]
    assert meta["retrieval_plan"]["action"] == "retrieve"
    assert meta["retrieval_plan"]["standalone_query"] == "什么是 RAG"
    assert meta["chunk_refs"][0]["ref_id"] == "doc-a#2"
    assert meta["rerank_scores"][0]["score"] == 0.91
    assert meta["grounding_status"] == "supported"
    assert meta["supported_ratio"] == 0.9


def test_turn_trace_skip_retrieval():
    ctx = TurnTraceContext(user_id="u1", session_id="s1")
    ctx.set_retrieval_plan(RetrievalPlan(action="skip", reason="闲聊"))
    ctx.set_chunks([])

    meta = ctx.to_metadata()
    assert meta["retrieval_plan"]["action"] == "skip"
    assert "document_ids_in_session" not in meta
    assert "chunk_refs" not in meta

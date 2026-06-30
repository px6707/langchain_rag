from langchain_core.documents import Document

from app.observability.retrieval_trace import RetrievalTrace
from app.observability.turn_trace import TurnTraceContext
from app.schemas.retrieval import RetrievalPlan
from app.services.grounding_service import GroundingResult


def test_turn_trace_metadata_includes_retrieval_trace_and_snippets():
    ctx = TurnTraceContext(user_id="u1", session_id="s1")
    ctx.set_retrieval_plan(
        RetrievalPlan(action="retrieve", strategy="none", standalone_query="q", reason="test")
    )
    ctx.set_retrieval_trace(
        RetrievalTrace(
            queries=["q"],
            tier=0,
            hits_before_rerank=3,
            hits_after_rerank=2,
            page_expand_added=1,
            asr_expand_added=0,
        )
    )
    ctx.set_chunks(
        [
            Document(
                "chunk body text",
                metadata={
                    "document_id": "550e8400-e29b-41d4-a716-446655440000",
                    "chunk_index": 1,
                    "filename": "a.pdf",
                    "rerank_score": 0.9,
                },
            )
        ]
    )
    ctx.set_grounding(GroundingResult(status="supported", supported_ratio=1.0, claims=[]))
    ctx.record_tool_start("t1", "get_current_time", "{}")
    ctx.record_tool_end("t1", "get_current_time", output="2024-01-01")
    ctx.set_hitl_decisions([{"decision": "approve", "tool_name": "send_email"}])

    meta = ctx.to_metadata()

    assert meta["retrieval_trace"]["tier"] == 0
    assert meta["retrieval_trace"]["hits_after_rerank"] == 2
    assert meta["chunk_snippets"][0]["ref_id"] == "550e8400-e29b-41d4-a716-446655440000#1"
    assert "chunk body" in meta["chunk_snippets"][0]["snippet"]
    assert meta["tool_trajectory"][0]["name"] == "get_current_time"
    assert meta["hitl_decisions"][0]["decision"] == "approve"

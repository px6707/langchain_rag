from unittest.mock import patch

from langchain_core.documents import Document

from app.schemas.retrieval import RetrievalPlan
from app.services.retrieval_service import _PipelineStats, search_with_plan


def test_search_with_plan_returns_retrieval_trace():
    plan = RetrievalPlan(action="retrieve", strategy="none", standalone_query="question")
    docs = [Document("hit", metadata={"filename": "a.pdf", "rerank_score": 0.9})]

    stats = _PipelineStats(
        queries=["question"],
        hits_before_rerank=1,
        hits_after_rerank=1,
        page_expand_added=0,
        asr_expand_added=0,
    )

    with (
        patch("app.services.retrieval_service._tiered_search", return_value=(docs, 0, stats)),
        patch("app.services.retrieval_service.normalize_plan", side_effect=lambda p: p),
    ):
        _sources, _context, _docs, trace = search_with_plan(plan)

    assert trace is not None
    assert trace.queries == ["question"]
    assert trace.tier == 0

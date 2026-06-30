"""Per-turn RAG trace context for LangSmith metadata."""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document

from app.config import settings
from app.schemas.retrieval import RetrievalPlan
from app.services.grounding_service import GroundingResult
from app.services.retrieval_service import _doc_ref_id, _doc_score

_turn_trace_ctx: ContextVar[TurnTraceContext | None] = ContextVar("turn_trace_context", default=None)


@dataclass
class TurnTraceContext:
    user_id: str = ""
    session_id: str = ""
    is_admin: bool = False
    env: str = "dev"
    retrieval_plan: RetrievalPlan | None = None
    chunks: list[Document] = field(default_factory=list)
    grounding: GroundingResult | None = None

    def set_retrieval_plan(self, plan: RetrievalPlan) -> None:
        self.retrieval_plan = plan

    def set_chunks(self, docs: list[Document]) -> None:
        self.chunks = list(docs)

    def set_grounding(self, result: GroundingResult) -> None:
        self.grounding = result

    def to_metadata(self) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "is_admin": self.is_admin,
            "env": self.env,
        }

        if self.retrieval_plan is not None:
            plan = self.retrieval_plan
            meta["retrieval_plan"] = {
                "action": plan.action,
                "strategy": plan.strategy,
                "reason": plan.reason,
            }
            if plan.standalone_query:
                meta["retrieval_plan"]["standalone_query"] = plan.standalone_query

        max_chunks = settings.langsmith_metadata_max_chunks
        chunk_refs: list[dict[str, Any]] = []
        rerank_scores: list[dict[str, Any]] = []
        document_ids: set[str] = set()

        for doc in self.chunks[:max_chunks]:
            document_id, chunk_index, ref_id = _doc_ref_id(doc)
            document_ids.add(document_id)
            filename = str(doc.metadata.get("filename", "unknown"))
            chunk_refs.append(
                {
                    "document_id": document_id,
                    "chunk_index": chunk_index,
                    "ref_id": ref_id,
                    "filename": filename,
                }
            )
            score = _doc_score(doc)
            if score is not None:
                rerank_scores.append({"ref_id": ref_id, "score": score})

        if document_ids:
            meta["document_ids_in_session"] = sorted(document_ids)
        if chunk_refs:
            meta["chunk_refs"] = chunk_refs
        if rerank_scores:
            meta["rerank_scores"] = rerank_scores

        if self.grounding is not None:
            meta["grounding_status"] = self.grounding.status
            meta["supported_ratio"] = self.grounding.supported_ratio

        return meta


def init_turn_trace(
    *,
    user_id: str,
    session_id: str,
    is_admin: bool,
    env: str,
) -> Token:
    ctx = TurnTraceContext(
        user_id=user_id,
        session_id=session_id,
        is_admin=is_admin,
        env=env,
    )
    return _turn_trace_ctx.set(ctx)


def get_turn_trace() -> TurnTraceContext | None:
    return _turn_trace_ctx.get()


def reset_turn_trace(token: Token) -> None:
    _turn_trace_ctx.reset(token)

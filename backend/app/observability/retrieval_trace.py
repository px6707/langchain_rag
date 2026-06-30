"""Structured retrieval observability payload for turn metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class RetrievalTrace:
    queries: list[str]
    tier: int | None
    hits_before_rerank: int
    hits_after_rerank: int
    page_expand_added: int
    asr_expand_added: int

    def to_metadata(self) -> dict:
        return asdict(self)

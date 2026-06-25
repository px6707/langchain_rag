from typing import Literal

from pydantic import BaseModel, Field


class QueryRewrite(BaseModel):
    standalone_query: str
    reason: str = ""


class StrategyPlan(BaseModel):
    action: Literal["skip", "retrieve"]
    strategy: Literal["none", "multi_query", "hyde", "decompose"] = "none"
    extra_queries: list[str] = Field(default_factory=list)
    hyde_document: str | None = None
    reason: str = ""


class RetrievalPlan(BaseModel):
    action: Literal["skip", "retrieve"]
    strategy: Literal["none", "multi_query", "hyde", "decompose"] = "none"
    standalone_query: str = ""
    extra_queries: list[str] = Field(default_factory=list)
    hyde_document: str | None = None
    reason: str = ""

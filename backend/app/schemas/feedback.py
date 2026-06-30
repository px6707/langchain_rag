from typing import Literal

from pydantic import BaseModel, Field, model_validator

FeedbackKind = Literal["thumbs_up", "thumbs_down"]

FeedbackReason = Literal[
    "retrieval_wrong",
    "hallucination",
    "tool_error",
    "too_slow",
    "other",
]

FEEDBACK_KIND_TO_KEY: dict[FeedbackKind, tuple[str, float]] = {
    "thumbs_up": ("user_thumbs_up", 1.0),
    "thumbs_down": ("user_thumbs_down", 0.0),
}


class ChatFeedbackRequest(BaseModel):
    run_id: str = Field(..., min_length=1)
    trace_id: str | None = None
    kind: FeedbackKind
    reason: FeedbackReason | None = None
    comment: str | None = Field(default=None, max_length=2000)
    session_id: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def require_reason_for_thumbs_down(self) -> "ChatFeedbackRequest":
        if self.kind == "thumbs_down" and self.reason is None:
            raise ValueError("reason is required when kind is thumbs_down")
        return self


class ChatFeedbackResponse(BaseModel):
    ok: bool = True

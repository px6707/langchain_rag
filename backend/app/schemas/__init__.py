from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=1)


class UserResponse(BaseModel):
    id: UUID
    username: str
    is_admin: bool
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


class UserCreateRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=6)
    is_admin: bool = False


class UserUpdateRequest(BaseModel):
    is_active: bool | None = None
    is_admin: bool | None = None
    password: str | None = Field(default=None, min_length=6)


class UserListResponse(BaseModel):
    items: list[UserResponse]
    total: int


class DocumentResponse(BaseModel):
    id: UUID
    filename: str
    file_type: str
    chunk_count: int
    status: str
    parse_stage: str | None = None
    error_message: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class DocumentListResponse(BaseModel):
    items: list[DocumentResponse]
    total: int


class DocumentChunkResponse(BaseModel):
    document_id: str
    chunk_index: int
    ref_id: str
    filename: str
    content: str
    file_type: str | None = None
    content_type: str | None = None
    timestamp_sec: float | None = None
    start_sec: float | None = None
    end_sec: float | None = None


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)


class HITLEditedAction(BaseModel):
    name: str
    args: dict


class HITLDecision(BaseModel):
    type: Literal["approve", "edit", "reject"]
    message: str | None = None
    edited_action: HITLEditedAction | None = None


class ChatResumeRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    decisions: list[HITLDecision] = Field(..., min_length=1)


class HITLActionRequest(BaseModel):
    name: str
    args: dict
    description: str | None = None


class HITLReviewConfig(BaseModel):
    action_name: str
    allowed_decisions: list[str]


class HITLRequestPayload(BaseModel):
    action_requests: list[HITLActionRequest]
    review_configs: list[HITLReviewConfig]


class ChatInterruptResponse(BaseModel):
    request: HITLRequestPayload | None = None


class SourceInfo(BaseModel):
    document_id: str
    chunk_index: int
    ref_id: str
    filename: str
    content: str
    score: float | None = None
    file_type: str | None = None
    content_type: str | None = None
    timestamp_sec: float | None = None
    start_sec: float | None = None
    end_sec: float | None = None


class ClaimVerdictResponse(BaseModel):
    claim: str
    supported: bool
    evidence_ref_ids: list[str]
    reason: str


class GroundingResultResponse(BaseModel):
    status: Literal["supported", "partial", "not_supported", "skipped"]
    supported_ratio: float
    claims: list[ClaimVerdictResponse]


class ToolCallInfo(BaseModel):
    id: str
    name: str
    args: str | None = None
    output: str | None = None
    status: str = "completed"


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]


class ChatMessageResponse(BaseModel):
    id: str
    role: str
    content: str
    sources: list[SourceInfo] | None = None
    grounding: GroundingResultResponse | None = None
    tool_calls: list[ToolCallInfo] | None = None
    created_at: datetime | None = None
    run_id: str | None = None
    trace_id: str | None = None


class TodoItemResponse(BaseModel):
    content: str
    status: Literal["pending", "in_progress", "completed"]


class ChatHistoryResponse(BaseModel):
    messages: list[ChatMessageResponse]
    todos: list[TodoItemResponse] | None = None


from app.schemas.feedback import (
    ChatFeedbackRequest,
    ChatFeedbackResponse,
    FeedbackKind,
    FeedbackReason,
)

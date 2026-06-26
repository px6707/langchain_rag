from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from app.config import settings
from app.services.grounding_service import (
    ClaimVerdict,
    GroundingJudgeOutput,
    GroundingResult,
    validate_grounding,
)


def _sample_chunk() -> Document:
    return Document(
        "合同期限为三年。",
        metadata={
            "document_id": "550e8400-e29b-41d4-a716-446655440000",
            "chunk_index": 1,
            "filename": "contract.pdf",
        },
    )


def test_validate_grounding_skipped_when_disabled():
    with patch.object(settings, "grounding_enabled", False):
        result = validate_grounding("答案", [_sample_chunk()])
    assert result.status == "skipped"


def test_validate_grounding_skipped_when_no_chunks():
    with patch.object(settings, "grounding_enabled", True):
        result = validate_grounding("答案", [])
    assert result.status == "skipped"


def test_validate_grounding_supported():
    judge_output = GroundingJudgeOutput(
        claims=[
            ClaimVerdict(
                claim="合同期限为三年",
                supported=True,
                evidence_ref_ids=["550e8400-e29b-41d4-a716-446655440000#1"],
                reason="与片段一致",
            ),
        ]
    )
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = judge_output
    mock_llm.with_structured_output.return_value = mock_structured

    with (
        patch.object(settings, "grounding_enabled", True),
        patch.object(settings, "grounding_min_supported_ratio", 0.8),
        patch.object(settings, "grounding_fail_ratio", 0.5),
        patch("app.services.grounding_service.get_small_llm", return_value=mock_llm),
    ):
        result = validate_grounding("合同期限为三年。[550e8400-e29b-41d4-a716-446655440000#1]", [_sample_chunk()])

    assert result.status == "supported"
    assert result.supported_ratio == 1.0
    assert len(result.claims) == 1


def test_validate_grounding_partial():
    judge_output = GroundingJudgeOutput(
        claims=[
            ClaimVerdict(claim="事实 A", supported=True, evidence_ref_ids=[], reason="ok"),
            ClaimVerdict(claim="事实 B", supported=False, evidence_ref_ids=[], reason="无依据"),
        ]
    )
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = judge_output
    mock_llm.with_structured_output.return_value = mock_structured

    with (
        patch.object(settings, "grounding_enabled", True),
        patch.object(settings, "grounding_min_supported_ratio", 0.8),
        patch.object(settings, "grounding_fail_ratio", 0.5),
        patch("app.services.grounding_service.get_small_llm", return_value=mock_llm),
    ):
        result = validate_grounding("事实 A 和事实 B", [_sample_chunk()])

    assert result.status == "partial"
    assert result.supported_ratio == 0.5


def test_validate_grounding_skipped_on_llm_error():
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.side_effect = RuntimeError("llm down")
    mock_llm.with_structured_output.return_value = mock_structured

    with (
        patch.object(settings, "grounding_enabled", True),
        patch("app.services.grounding_service.get_small_llm", return_value=mock_llm),
    ):
        result = validate_grounding("答案", [_sample_chunk()])

    assert result.status == "skipped"
    assert result.claims == []

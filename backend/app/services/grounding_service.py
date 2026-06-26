import logging
from typing import Literal

from langchain_core.documents import Document
from pydantic import BaseModel, Field

from app.config import settings
from app.services.llm_service import get_router_llm
from app.services.retrieval_service import _doc_ref_id

logger = logging.getLogger(__name__)


class ClaimVerdict(BaseModel):
    claim: str
    supported: bool
    evidence_ref_ids: list[str] = Field(default_factory=list)
    reason: str = ""


class GroundingJudgeOutput(BaseModel):
    claims: list[ClaimVerdict] = Field(default_factory=list)


class GroundingResult(BaseModel):
    status: Literal["supported", "partial", "not_supported", "skipped"]
    supported_ratio: float
    claims: list[ClaimVerdict] = Field(default_factory=list)


def _extract_answer_text(content: str | list | None) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts).strip()
    return str(content).strip()


def _format_chunks_for_judge(chunks: list[Document]) -> str:
    lines: list[str] = []
    for doc in chunks:
        _, _, ref_id = _doc_ref_id(doc)
        filename = str(doc.metadata.get("filename", "unknown"))
        lines.append(f"[{ref_id}] {filename}\n{doc.page_content}")
    return "\n\n---\n\n".join(lines)


def _map_status(supported_ratio: float) -> Literal["supported", "partial", "not_supported"]:
    if supported_ratio >= settings.grounding_min_supported_ratio:
        return "supported"
    if supported_ratio >= settings.grounding_fail_ratio:
        return "partial"
    return "not_supported"


def validate_grounding(answer: str, chunks: list[Document]) -> GroundingResult:
    if not settings.grounding_enabled:
        return GroundingResult(status="skipped", supported_ratio=0.0, claims=[])

    answer_text = answer.strip()
    if not answer_text or not chunks:
        return GroundingResult(status="skipped", supported_ratio=0.0, claims=[])

    chunks_text = _format_chunks_for_judge(chunks)
    max_claims = settings.grounding_max_claims

    prompt = (
        f"你是 RAG 答案校验助手。根据检索片段判断答案中的事实性陈述是否被支撑。\n\n"
        f"检索片段（每段以 [document_id#chunk_index] 标识）：\n{chunks_text}\n\n"
        f"助手答案：\n{answer_text}\n\n"
        f"任务：\n"
        f"1. 从答案中抽取最多 {max_claims} 条原子事实 claim（跳过寒暄、引用标记本身）。\n"
        f"2. 对每条 claim 判定 supported：仅当检索片段中有明确依据时为 true。\n"
        f"3. evidence_ref_ids 填写支撑该 claim 的引用 ID 列表（如 uuid#2），无则留空。\n"
        f"4. reason 用一句话说明判定理由。"
    )

    try:
        model = get_router_llm()
        structured = model.with_structured_output(GroundingJudgeOutput)
        raw = structured.invoke(
            [
                {
                    "role": "system",
                    "content": "输出 JSON，claims 为 ClaimVerdict 列表。",
                },
                {"role": "user", "content": prompt},
            ]
        )
        result = GroundingJudgeOutput.model_validate(raw)
        claims = result.claims[:max_claims]
    except Exception:
        logger.exception("Grounding validation failed")
        return GroundingResult(status="skipped", supported_ratio=0.0, claims=[])

    if not claims:
        return GroundingResult(status="skipped", supported_ratio=0.0, claims=[])

    supported_count = sum(1 for c in claims if c.supported)
    supported_ratio = supported_count / len(claims)
    status = _map_status(supported_ratio)
    return GroundingResult(
        status=status,
        supported_ratio=supported_ratio,
        claims=claims,
    )

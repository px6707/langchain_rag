import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from app.config import settings
from app.models import Document as DocumentModel
from app.parsing.asr_service import TranscriptSegment
from app.parsing.frame_planner import PlannedFrame
from app.parsing.pipeline import _process_video_frame


def test_process_video_frame_skips_vlm_when_disabled(tmp_path):
    frame_path = tmp_path / "frame.png"
    frame_path.write_bytes(b"png")
    document = DocumentModel(
        id=__import__("uuid").uuid4(),
        user_id=__import__("uuid").uuid4(),
        filename="demo.mp4",
        file_path="/tmp/demo.mp4",
        file_type="video",
    )
    plan = PlannedFrame(timestamp_sec=10.0, source="uniform")

    with (
        patch.object(settings, "video_vlm_enabled", False),
        patch("app.parsing.pipeline.parse_file_to_documents", return_value=([], "batch")),
        patch("app.parsing.pipeline.summarize_image") as mock_vlm,
    ):
        docs = asyncio.run(_process_video_frame(document, 0, plan, str(frame_path)))

    assert docs == []
    mock_vlm.assert_not_called()


def test_process_video_frame_calls_vlm_when_enabled_and_ocr_short(tmp_path):
    frame_path = tmp_path / "frame.png"
    frame_path.write_bytes(b"png")
    document = DocumentModel(
        id=__import__("uuid").uuid4(),
        user_id=__import__("uuid").uuid4(),
        filename="demo.mp4",
        file_path="/tmp/demo.mp4",
        file_type="video",
    )
    plan = PlannedFrame(
        timestamp_sec=10.0,
        source="asr",
        asr_segment_index=1,
        asr_start_sec=5.0,
        asr_end_sec=15.0,
    )
    from langchain_core.documents import Document

    ocr_doc = Document(page_content="hi", metadata={})

    with (
        patch.object(settings, "video_vlm_enabled", True),
        patch.object(settings, "video_vlm_min_ocr_chars", 80),
        patch("app.parsing.pipeline.parse_file_to_documents", return_value=([ocr_doc], "batch")),
        patch("app.parsing.pipeline.summarize_image", return_value="visual summary"),
    ):
        docs = asyncio.run(_process_video_frame(document, 0, plan, str(frame_path)))

    assert len(docs) == 2
    assert docs[1].metadata.get("content_type") == "frame_vlm_summary"
    assert docs[1].metadata.get("asr_segment_index") == 1

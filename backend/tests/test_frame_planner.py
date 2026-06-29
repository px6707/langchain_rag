from unittest.mock import patch

from app.config import settings
from app.parsing.asr_service import TranscriptSegment
from app.parsing.frame_planner import (
    PlannedFrame,
    _merge_nearby,
    find_scene_gap_midpoints,
    plan_frame_timestamps,
)


def test_uniform_covers_two_hour_video():
    with (
        patch.object(settings, "video_frame_budget", 96),
        patch.object(settings, "video_min_interval_sec", 60.0),
        patch.object(settings, "video_asr_anchor_enabled", False),
    ):
        plans = plan_frame_timestamps(7200.0, [])
    assert len(plans) >= 30
    assert plans[0].timestamp_sec > 0
    assert plans[-1].timestamp_sec < 7200.0
    assert all(p.source == "uniform" for p in plans)


def test_asr_anchor_merge_and_metadata():
    segments = [
        TranscriptSegment(start_sec=0.0, end_sec=30.0, text="intro"),
        TranscriptSegment(start_sec=35.0, end_sec=60.0, text="still intro"),
        TranscriptSegment(start_sec=200.0, end_sec=260.0, text="topic two"),
    ]
    with (
        patch.object(settings, "video_frame_budget", 96),
        patch.object(settings, "video_min_interval_sec", 600.0),
        patch.object(settings, "video_asr_merge_gap_sec", 90.0),
        patch.object(settings, "video_asr_anchor_enabled", True),
    ):
        plans = plan_frame_timestamps(600.0, segments)
    asr_plans = [p for p in plans if p.source == "asr"]
    assert len(asr_plans) == 2
    assert asr_plans[0].asr_segment_index == 0
    assert asr_plans[1].asr_segment_index == 2


def test_merge_nearby_prefers_asr():
    uniform = PlannedFrame(timestamp_sec=100.0, source="uniform")
    asr = PlannedFrame(
        timestamp_sec=101.0,
        source="asr",
        asr_segment_index=1,
        asr_start_sec=90.0,
        asr_end_sec=110.0,
    )
    merged = _merge_nearby([uniform, asr], merge_sec=3.0)
    assert len(merged) == 1
    assert merged[0].source == "asr"


def test_budget_caps_frame_count():
    segments = [
        TranscriptSegment(start_sec=i * 10.0, end_sec=i * 10.0 + 5.0, text=f"seg {i}")
        for i in range(50)
    ]
    with (
        patch.object(settings, "video_frame_budget", 10),
        patch.object(settings, "video_min_interval_sec", 5.0),
        patch.object(settings, "video_asr_anchor_enabled", True),
    ):
        plans = plan_frame_timestamps(600.0, segments)
    assert len(plans) <= 10


def test_find_scene_gap_midpoints():
    base = [
        PlannedFrame(timestamp_sec=100.0, source="uniform"),
        PlannedFrame(timestamp_sec=400.0, source="uniform"),
    ]
    gaps = find_scene_gap_midpoints(7200.0, base, scene_gap_sec=120.0, max_scene=5)
    assert gaps
    assert gaps[0].source == "scene"

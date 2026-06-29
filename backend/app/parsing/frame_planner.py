from dataclasses import dataclass, replace

from app.config import settings
from app.parsing.asr_service import TranscriptSegment

SOURCE_PRIORITY = {"asr": 3, "scene": 2, "uniform": 1}


@dataclass(frozen=True)
class PlannedFrame:
    timestamp_sec: float
    source: str  # uniform | asr | scene
    asr_segment_index: int | None = None
    asr_start_sec: float | None = None
    asr_end_sec: float | None = None


def _duration_from_asr(segments: list[TranscriptSegment]) -> float:
    if not segments:
        return 0.0
    return max(seg.end_sec for seg in segments)


def _uniform_plans(duration_sec: float, budget: int, min_interval: float) -> list[PlannedFrame]:
    if duration_sec <= 0 or budget <= 0:
        return []
    max_uniform = min(int(duration_sec // min_interval), int(budget * 0.4))
    if max_uniform <= 0:
        return []
    plans: list[PlannedFrame] = []
    for i in range(1, max_uniform + 1):
        ts = duration_sec * i / (max_uniform + 1)
        plans.append(PlannedFrame(timestamp_sec=ts, source="uniform"))
    return plans


def _merged_asr_groups(segments: list[TranscriptSegment], merge_gap: float) -> list[tuple[int, int, float, float]]:
    """Return (start_index, end_index, start_sec, end_sec) groups."""
    indexed = [(idx, seg) for idx, seg in enumerate(segments) if seg.text.strip()]
    if not indexed:
        return []

    groups: list[tuple[int, int, float, float]] = []
    group_start_idx, first_seg = indexed[0]
    group_end_idx = group_start_idx
    group_start = first_seg.start_sec
    group_end = first_seg.end_sec

    for idx, seg in indexed[1:]:
        if seg.start_sec - group_end <= merge_gap:
            group_end_idx = idx
            group_end = max(group_end, seg.end_sec)
        else:
            groups.append((group_start_idx, group_end_idx, group_start, group_end))
            group_start_idx = idx
            group_end_idx = idx
            group_start = seg.start_sec
            group_end = seg.end_sec
    groups.append((group_start_idx, group_end_idx, group_start, group_end))
    return groups


def _asr_plans(segments: list[TranscriptSegment], budget: int, merge_gap: float) -> list[PlannedFrame]:
    if not settings.video_asr_anchor_enabled or not segments:
        return []
    max_asr = int(budget * 0.35)
    groups = _merged_asr_groups(segments, merge_gap)
    if len(groups) > max_asr:
        groups.sort(key=lambda g: g[3] - g[2], reverse=True)
        groups = groups[:max_asr]
        groups.sort(key=lambda g: g[2])

    plans: list[PlannedFrame] = []
    for start_idx, end_idx, start_sec, end_sec in groups:
        ts = (start_sec + end_sec) / 2.0
        plans.append(
            PlannedFrame(
                timestamp_sec=ts,
                source="asr",
                asr_segment_index=start_idx,
                asr_start_sec=start_sec,
                asr_end_sec=end_sec,
            )
        )
    return plans


def _merge_nearby(plans: list[PlannedFrame], merge_sec: float) -> list[PlannedFrame]:
    if not plans:
        return []
    sorted_plans = sorted(plans, key=lambda p: p.timestamp_sec)
    merged: list[PlannedFrame] = [sorted_plans[0]]
    for plan in sorted_plans[1:]:
        last = merged[-1]
        if plan.timestamp_sec - last.timestamp_sec <= merge_sec:
            if SOURCE_PRIORITY.get(plan.source, 0) > SOURCE_PRIORITY.get(last.source, 0):
                merged[-1] = plan
            continue
        merged.append(plan)
    return merged


def _apply_budget(plans: list[PlannedFrame], budget: int) -> list[PlannedFrame]:
    if len(plans) <= budget:
        return plans
    by_priority = sorted(
        plans,
        key=lambda p: (SOURCE_PRIORITY.get(p.source, 0), -p.timestamp_sec),
        reverse=True,
    )
    kept = by_priority[:budget]
    return sorted(kept, key=lambda p: p.timestamp_sec)


def _fill_uniform_gaps(plans: list[PlannedFrame], duration_sec: float, min_interval: float) -> list[PlannedFrame]:
    if duration_sec <= 0:
        return plans
    gap_limit = min_interval * 2
    timestamps = [0.0, *[p.timestamp_sec for p in plans], duration_sec]
    extras: list[PlannedFrame] = []
    for start, end in zip(timestamps, timestamps[1:]):
        if end - start > gap_limit:
            extras.append(PlannedFrame(timestamp_sec=(start + end) / 2.0, source="uniform"))
    if not extras:
        return plans
    return _merge_nearby([*plans, *extras], settings.video_timestamp_merge_sec)


def find_scene_gap_midpoints(
    duration_sec: float,
    plans: list[PlannedFrame],
    scene_gap_sec: float,
    max_scene: int,
) -> list[PlannedFrame]:
    """Pure gap detection: return midpoint timestamps for long gaps without frames."""
    if duration_sec <= 0 or max_scene <= 0:
        return []
    points = sorted({0.0, duration_sec, *(p.timestamp_sec for p in plans)})
    gaps: list[PlannedFrame] = []
    for start, end in zip(points, points[1:]):
        if end - start <= scene_gap_sec:
            continue
        gaps.append(PlannedFrame(timestamp_sec=(start + end) / 2.0, source="scene"))
        if len(gaps) >= max_scene:
            break
    return gaps


def plan_frame_timestamps(
    duration_sec: float,
    asr_segments: list[TranscriptSegment],
) -> list[PlannedFrame]:
    if duration_sec <= 0:
        duration_sec = _duration_from_asr(asr_segments)
    if duration_sec <= 0:
        return []

    budget = settings.video_frame_budget
    min_interval = settings.video_min_interval_sec
    merge_gap = settings.video_asr_merge_gap_sec
    merge_sec = settings.video_timestamp_merge_sec

    uniform = _uniform_plans(duration_sec, budget, min_interval)
    asr = _asr_plans(asr_segments, budget, merge_gap)
    merged = _merge_nearby([*uniform, *asr], merge_sec)
    capped = _apply_budget(merged, budget)
    return _fill_uniform_gaps(capped, duration_sec, min_interval)


def merge_plans_with_scene_gaps(
    duration_sec: float,
    base_plans: list[PlannedFrame],
    scene_plans: list[PlannedFrame],
) -> list[PlannedFrame]:
    budget = settings.video_frame_budget
    merge_sec = settings.video_timestamp_merge_sec
    merged = _merge_nearby([*base_plans, *scene_plans], merge_sec)
    capped = _apply_budget(merged, budget)
    return _fill_uniform_gaps(capped, duration_sec, settings.video_min_interval_sec)


def attach_asr_to_scene_plan(plan: PlannedFrame, segments: list[TranscriptSegment]) -> PlannedFrame:
    if plan.asr_segment_index is not None or not segments:
        return plan
    for idx, seg in enumerate(segments):
        if seg.start_sec <= plan.timestamp_sec <= seg.end_sec:
            return replace(
                plan,
                asr_segment_index=idx,
                asr_start_sec=seg.start_sec,
                asr_end_sec=seg.end_sec,
            )
    return plan

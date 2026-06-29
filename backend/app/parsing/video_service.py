import logging
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import imagehash
from PIL import Image

from app.config import settings
from app.parsing.asr_service import TranscriptSegment
from app.parsing.frame_planner import (
    SOURCE_PRIORITY,
    PlannedFrame,
    attach_asr_to_scene_plan,
    find_scene_gap_midpoints,
    merge_plans_with_scene_gaps,
    plan_frame_timestamps,
)

logger = logging.getLogger(__name__)

PTS_TIME_RE = re.compile(r"pts_time:([\d.]+)")


@dataclass
class VideoExtractResult:
    audio_path: str
    frame_plans: list[tuple[PlannedFrame, str]]
    temp_dir: str


@dataclass
class _FrameBatch:
    batch_id: int
    plans: list[PlannedFrame]


def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required for audio/video parsing but was not found in PATH")


def probe_video_duration(video_path: str) -> float:
    if shutil.which("ffprobe") is None:
        return 0.0
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return max(0.0, float(result.stdout.strip()))
    except (subprocess.CalledProcessError, ValueError):
        logger.warning("ffprobe failed for %s", video_path)
        return 0.0


def _duration_from_asr(segments: list[TranscriptSegment]) -> float:
    if not segments:
        return 0.0
    return max(seg.end_sec for seg in segments)


def _ffmpeg_thread_args() -> list[str]:
    threads = settings.video_extract_ffmpeg_threads
    if threads > 0:
        return ["-threads", str(threads)]
    return []


def extract_audio_wav(video_path: str, output_path: str) -> None:
    ensure_ffmpeg()
    cmd = [
        "ffmpeg", "-y",
        *_ffmpeg_thread_args(),
        "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def extract_video_audio(video_path: str) -> tuple[str, str]:
    ensure_ffmpeg()
    temp_dir = tempfile.mkdtemp(prefix="rag_video_")
    audio_path = str(Path(temp_dir) / "audio.wav")
    try:
        extract_audio_wav(video_path, audio_path)
    except subprocess.CalledProcessError:
        return "", temp_dir
    return audio_path, temp_dir


def _extract_scene_in_range(video_path: str, start_sec: float, end_sec: float, output_path: str) -> float | None:
    if end_sec <= start_sec:
        return None
    threshold = settings.video_scene_threshold
    duration = end_sec - start_sec
    cmd = [
        "ffmpeg", "-y",
        *_ffmpeg_thread_args(),
        "-ss", str(max(0.0, start_sec)),
        "-i", video_path,
        "-t", str(duration),
        "-vf", f"select=gt(scene\\,{threshold}),showinfo",
        "-vsync", "vfr",
        "-frames:v", "1",
        output_path,
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        return None
    if not Path(output_path).is_file():
        return None
    timestamps = [float(value) for value in PTS_TIME_RE.findall(result.stderr)]
    if timestamps:
        return start_sec + timestamps[0]
    return (start_sec + end_sec) / 2.0


def cluster_plans_by_window(
    plans: list[PlannedFrame],
    window_sec: float,
) -> list[_FrameBatch]:
    if not plans:
        return []

    sorted_plans = sorted(plans, key=lambda plan: plan.timestamp_sec)
    batches: list[_FrameBatch] = []
    current: list[PlannedFrame] = [sorted_plans[0]]
    batch_start = sorted_plans[0].timestamp_sec

    for plan in sorted_plans[1:]:
        if plan.timestamp_sec - batch_start <= window_sec:
            current.append(plan)
        else:
            batches.append(_FrameBatch(batch_id=len(batches), plans=current))
            current = [plan]
            batch_start = plan.timestamp_sec

    batches.append(_FrameBatch(batch_id=len(batches), plans=current))
    return batches


def _extract_single_frame(
    video_path: str,
    temp_dir: Path,
    batch: _FrameBatch,
) -> list[tuple[PlannedFrame, str]]:
    plan = batch.plans[0]
    output_path = str(temp_dir / f"batch_{batch.batch_id:04d}_0000.png")
    cmd = [
        "ffmpeg", "-y",
        *_ffmpeg_thread_args(),
        "-ss", str(max(0.0, plan.timestamp_sec)),
        "-i", video_path,
        "-frames:v", "1",
        "-q:v", "2",
        output_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError:
        logger.warning("Failed to extract frame at %ss from %s", plan.timestamp_sec, video_path)
        return []
    if Path(output_path).is_file():
        return [(plan, output_path)]
    return []


def _extract_frame_batch(
    video_path: str,
    temp_dir: Path,
    batch: _FrameBatch,
) -> list[tuple[PlannedFrame, str]]:
    if len(batch.plans) == 1:
        return _extract_single_frame(video_path, temp_dir, batch)

    batch_start = batch.plans[0].timestamp_sec
    batch_end = batch.plans[-1].timestamp_sec
    margin = settings.video_extract_select_margin_sec
    duration = batch_end - batch_start + margin + 0.5

    rel_times = [plan.timestamp_sec - batch_start for plan in batch.plans]
    select_parts = [
        f"between(t\\,{rel:.3f}\\,{rel + margin:.3f})"
        for rel in rel_times
    ]
    select_expr = "+".join(select_parts)
    output_pattern = str(temp_dir / f"batch_{batch.batch_id:04d}_%04d.png")

    cmd = [
        "ffmpeg", "-y",
        *_ffmpeg_thread_args(),
        "-ss", str(max(0.0, batch_start)),
        "-i", video_path,
        "-t", str(duration),
        "-vf", f"select='{select_expr}',scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-vsync", "vfr",
        "-start_number", "0",
        output_pattern,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError:
        logger.warning(
            "Failed to extract frame batch %s (start=%ss) from %s",
            batch.batch_id,
            batch_start,
            video_path,
        )
        return []

    results: list[tuple[PlannedFrame, str]] = []
    for local_idx, plan in enumerate(batch.plans):
        output_path = str(temp_dir / f"batch_{batch.batch_id:04d}_{local_idx:04d}.png")
        if Path(output_path).is_file():
            results.append((plan, output_path))
        else:
            logger.warning(
                "Missing batch output for plan at %ss (batch=%s idx=%s)",
                plan.timestamp_sec,
                batch.batch_id,
                local_idx,
            )
    return results


def _run_frame_batch(
    video_path: str,
    temp_dir: Path,
    batch: _FrameBatch,
) -> list[tuple[PlannedFrame, str]]:
    return _extract_frame_batch(video_path, temp_dir, batch)


def extract_frames_at_timestamps(
    video_path: str,
    temp_dir: Path,
    plans: list[PlannedFrame],
) -> list[tuple[PlannedFrame, str]]:
    ensure_ffmpeg()
    if not plans:
        return []

    batches = cluster_plans_by_window(plans, settings.video_extract_batch_window_sec)
    results: list[tuple[PlannedFrame, str]] = []
    max_workers = max(1, settings.video_frame_concurrency)

    if len(batches) == 1:
        return _run_frame_batch(video_path, temp_dir, batches[0])

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_run_frame_batch, video_path, temp_dir, batch): batch
            for batch in batches
        }
        for future in as_completed(futures):
            try:
                batch_results = future.result()
                results.extend(batch_results)
            except Exception:
                batch = futures[future]
                logger.exception(
                    "Frame batch %s failed unexpectedly (start=%ss)",
                    batch.batch_id,
                    batch.plans[0].timestamp_sec if batch.plans else "?",
                )

    return sorted(results, key=lambda item: item[0].timestamp_sec)


@dataclass
class _SceneGapProbe:
    gap_idx: int
    start: float
    end: float
    fallback_plan: PlannedFrame


def _probe_scene_gap(
    video_path: str,
    temp_dir: Path,
    probe: _SceneGapProbe,
    asr_segments: list[TranscriptSegment],
) -> PlannedFrame:
    probe_path = str(temp_dir / f"scene_probe_{probe.gap_idx:04d}.png")
    detected_ts = _extract_scene_in_range(video_path, probe.start, probe.end, probe_path)
    if detected_ts is not None:
        return attach_asr_to_scene_plan(
            PlannedFrame(timestamp_sec=detected_ts, source="scene"),
            asr_segments,
        )
    return probe.fallback_plan


def _fill_scene_gap_frames(
    video_path: str,
    temp_dir: Path,
    duration_sec: float,
    base_plans: list[PlannedFrame],
    asr_segments: list[TranscriptSegment],
) -> list[PlannedFrame]:
    gap_plans = find_scene_gap_midpoints(
        duration_sec,
        base_plans,
        settings.video_scene_gap_sec,
        settings.video_scene_extra_max,
    )
    if not gap_plans:
        return base_plans

    probes: list[_SceneGapProbe] = []
    points = sorted({0.0, duration_sec, *(p.timestamp_sec for p in base_plans)})
    gap_idx = 0
    for start, end in zip(points, points[1:]):
        if end - start <= settings.video_scene_gap_sec:
            continue
        if gap_idx >= len(gap_plans):
            break
        probes.append(
            _SceneGapProbe(
                gap_idx=gap_idx,
                start=start,
                end=end,
                fallback_plan=gap_plans[gap_idx],
            )
        )
        gap_idx += 1

    if not probes:
        return base_plans

    scene_frames: list[tuple[int, PlannedFrame]] = []
    max_workers = max(1, min(settings.video_frame_concurrency, len(probes)))

    if len(probes) == 1:
        plan = _probe_scene_gap(video_path, temp_dir, probes[0], asr_segments)
        scene_frames.append((probes[0].gap_idx, plan))
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_probe_scene_gap, video_path, temp_dir, probe, asr_segments): probe
                for probe in probes
            }
            for future in as_completed(futures):
                probe = futures[future]
                try:
                    plan = future.result()
                    scene_frames.append((probe.gap_idx, plan))
                except Exception:
                    logger.exception(
                        "Scene gap probe failed for gap %s (%.1f-%.1fs)",
                        probe.gap_idx,
                        probe.start,
                        probe.end,
                    )
                    scene_frames.append((probe.gap_idx, probe.fallback_plan))

    ordered = [plan for _, plan in sorted(scene_frames, key=lambda item: item[0])]
    return merge_plans_with_scene_gaps(duration_sec, base_plans, ordered)


def dedupe_planned_frames(
    frames: list[tuple[PlannedFrame, str]],
    threshold: int,
) -> list[tuple[PlannedFrame, str]]:
    if not frames:
        return []

    sorted_frames = sorted(frames, key=lambda item: item[0].timestamp_sec)
    kept: list[tuple[PlannedFrame, str]] = []
    last_hash: imagehash.ImageHash | None = None

    for plan, frame_path in sorted_frames:
        try:
            with Image.open(frame_path) as img:
                frame_hash = imagehash.phash(img)
        except Exception:
            logger.warning("Failed to hash frame %s, keeping frame", frame_path)
            kept.append((plan, frame_path))
            last_hash = None
            continue

        if last_hash is not None and (last_hash - frame_hash) <= threshold:
            existing_plan, _ = kept[-1]
            if SOURCE_PRIORITY.get(plan.source, 0) > SOURCE_PRIORITY.get(existing_plan.source, 0):
                kept[-1] = (plan, frame_path)
            continue

        kept.append((plan, frame_path))
        last_hash = frame_hash

    return kept


def extract_planned_frames(
    video_path: str,
    temp_dir: str,
    asr_segments: list[TranscriptSegment],
) -> list[tuple[PlannedFrame, str]]:
    duration = probe_video_duration(video_path) or _duration_from_asr(asr_segments)
    if duration <= 0:
        return []

    base_plans = plan_frame_timestamps(duration, asr_segments)
    plans = _fill_scene_gap_frames(video_path, Path(temp_dir), duration, base_plans, asr_segments)
    frames = extract_frames_at_timestamps(video_path, Path(temp_dir), plans)
    if settings.video_dedupe_enabled:
        frames = dedupe_planned_frames(frames, settings.video_dedupe_hamming_threshold)
    return frames


def dedupe_frames(
    frames: list[tuple[float, str]],
    threshold: int,
) -> list[tuple[float, str]]:
    """Legacy helper for tests."""
    planned = [(PlannedFrame(timestamp_sec=ts, source="uniform"), path) for ts, path in frames]
    deduped = dedupe_planned_frames(planned, threshold)
    return [(plan.timestamp_sec, path) for plan, path in deduped]


def cleanup_temp_dir(temp_dir: str) -> None:
    if temp_dir and Path(temp_dir).exists():
        shutil.rmtree(temp_dir, ignore_errors=True)

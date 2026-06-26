import asyncio
import logging
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import imagehash
from PIL import Image

from app.config import settings

logger = logging.getLogger(__name__)

PTS_TIME_RE = re.compile(r"pts_time:([\d.]+)")


@dataclass
class VideoExtractResult:
    audio_path: str
    frame_paths: list[tuple[float, str]]  # (timestamp_sec, path)
    temp_dir: str


def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required for audio/video parsing but was not found in PATH")


def extract_audio_wav(video_path: str, output_path: str) -> None:
    ensure_ffmpeg()
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def _extract_interval_frames(video_path: str, temp_dir: Path) -> list[tuple[float, str]]:
    interval = settings.video_frame_interval_sec
    max_frames = settings.video_max_frames
    pattern = str(temp_dir / "frame_%04d.png")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"fps=1/{interval}",
        "-frames:v", str(max_frames),
        pattern,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError:
        logger.warning("ffmpeg interval frame extraction failed for %s", video_path)
        return []
    frames = sorted(temp_dir.glob("frame_*.png"))
    return [(idx * interval, str(frame)) for idx, frame in enumerate(frames)]


def _extract_scene_frames(video_path: str, temp_dir: Path) -> list[tuple[float, str]]:
    threshold = settings.video_scene_threshold
    max_frames = settings.video_max_frames
    pattern = str(temp_dir / "frame_%04d.png")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"select=gt(scene\\,{threshold}),showinfo",
        "-vsync", "vfr",
        "-frames:v", str(max_frames),
        pattern,
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        logger.warning("ffmpeg scene frame extraction failed for %s", video_path)
        return []

    timestamps = [float(value) for value in PTS_TIME_RE.findall(result.stderr)]
    frames = sorted(temp_dir.glob("frame_*.png"))
    if not frames:
        return []

    if len(timestamps) >= len(frames):
        return list(zip(timestamps[: len(frames)], [str(frame) for frame in frames], strict=False))
    return [(float(idx), str(frame)) for idx, frame in enumerate(frames)]


def extract_frames(video_path: str, temp_dir: Path) -> list[tuple[float, str]]:
    ensure_ffmpeg()
    mode = settings.video_frame_mode.lower()
    if mode == "scene":
        frames = _extract_scene_frames(video_path, temp_dir)
        if frames:
            return frames
        logger.info("Scene extraction returned no frames, falling back to interval mode")
    return _extract_interval_frames(video_path, temp_dir)


def dedupe_frames(
    frames: list[tuple[float, str]],
    threshold: int,
) -> list[tuple[float, str]]:
    if not frames:
        return []

    kept: list[tuple[float, str]] = []
    last_hash: imagehash.ImageHash | None = None

    for timestamp, frame_path in frames:
        try:
            with Image.open(frame_path) as img:
                frame_hash = imagehash.phash(img)
        except Exception:
            logger.warning("Failed to hash frame %s, keeping frame", frame_path)
            kept.append((timestamp, frame_path))
            last_hash = None
            continue

        if last_hash is not None and (last_hash - frame_hash) <= threshold:
            continue

        kept.append((timestamp, frame_path))
        last_hash = frame_hash

    return kept


def extract_video_assets(video_path: str) -> VideoExtractResult:
    ensure_ffmpeg()
    temp_dir = tempfile.mkdtemp(prefix="rag_video_")
    temp_path = Path(temp_dir)
    audio_path = str(temp_path / "audio.wav")
    try:
        extract_audio_wav(video_path, audio_path)
    except subprocess.CalledProcessError:
        audio_path = ""
    frame_paths = extract_frames(video_path, temp_path)
    if settings.video_dedupe_enabled:
        frame_paths = dedupe_frames(frame_paths, settings.video_dedupe_hamming_threshold)
    if not audio_path and not frame_paths:
        raise ValueError("No audio or video content extracted")
    return VideoExtractResult(
        audio_path=audio_path,
        frame_paths=frame_paths,
        temp_dir=temp_dir,
    )


def cleanup_temp_dir(temp_dir: str) -> None:
    if temp_dir and Path(temp_dir).exists():
        shutil.rmtree(temp_dir, ignore_errors=True)

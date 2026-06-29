from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from app.config import settings
from app.parsing.frame_planner import PlannedFrame
from app.parsing.video_service import (
    cluster_plans_by_window,
    dedupe_frames,
    dedupe_planned_frames,
    extract_frames_at_timestamps,
    extract_planned_frames,
    extract_video_audio,
)


def _write_png(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (32, 32), color).save(path)


def _write_pattern_png(path: Path) -> None:
    img = Image.new("RGB", (64, 64))
    pixels = img.load()
    assert pixels is not None
    for x in range(64):
        for y in range(64):
            pixels[x, y] = ((x * 7) % 256, (y * 11) % 256, (x + y) % 256)
    img.save(path)


def _fake_ffmpeg_run(cmd, **kwargs):
    output_arg = cmd[-1]
    if "%04d" in output_arg:
        prefix = output_arg.rsplit("%", 1)[0]
        for idx in range(4):
            path = Path(f"{prefix}{idx:04d}.png")
            if not path.exists():
                path.write_bytes(b"png")
                break
            path.write_bytes(b"png")
        # write two files for typical 2-frame batch
        Path(f"{prefix}0000.png").write_bytes(b"png")
        Path(f"{prefix}0001.png").write_bytes(b"png")
    else:
        Path(output_arg).write_bytes(b"png")
    return MagicMock(returncode=0)


def test_dedupe_planned_frames_skips_similar(tmp_path: Path):
    frame1 = tmp_path / "f1.png"
    frame2 = tmp_path / "f2.png"
    frame3 = tmp_path / "f3.png"
    _write_png(frame1, (255, 0, 0))
    _write_png(frame2, (255, 0, 0))
    _write_pattern_png(frame3)

    frames = [
        (PlannedFrame(timestamp_sec=0.0, source="uniform"), str(frame1)),
        (PlannedFrame(timestamp_sec=30.0, source="uniform"), str(frame2)),
        (PlannedFrame(timestamp_sec=60.0, source="asr"), str(frame3)),
    ]
    kept = dedupe_planned_frames(frames, threshold=5)

    assert len(kept) == 2
    assert kept[0][1] == str(frame1)
    assert kept[1][1] == str(frame3)


def test_dedupe_frames_legacy_wrapper(tmp_path: Path):
    frame1 = tmp_path / "f1.png"
    frame2 = tmp_path / "f2.png"
    _write_png(frame1, (0, 255, 0))
    _write_png(frame2, (0, 255, 0))
    kept = dedupe_frames([(0.0, str(frame1)), (1.0, str(frame2))], threshold=5)
    assert len(kept) == 1


def test_cluster_plans_by_window():
    plans = [
        PlannedFrame(timestamp_sec=60.0, source="uniform"),
        PlannedFrame(timestamp_sec=120.0, source="uniform"),
        PlannedFrame(timestamp_sec=300.0, source="asr"),
    ]
    batches = cluster_plans_by_window(plans, window_sec=120.0)
    assert len(batches) == 2
    assert len(batches[0].plans) == 2
    assert batches[0].plans[0].timestamp_sec == 60.0
    assert len(batches[1].plans) == 1
    assert batches[1].plans[0].timestamp_sec == 300.0


def test_extract_frames_at_timestamps_single(tmp_path: Path):
    plan = PlannedFrame(timestamp_sec=12.5, source="uniform")

    with patch("app.parsing.video_service.subprocess.run", side_effect=_fake_ffmpeg_run) as mock_run:
        results = extract_frames_at_timestamps("/tmp/video.mp4", tmp_path, [plan])

    assert len(results) == 1
    assert results[0][0].timestamp_sec == 12.5
    mock_run.assert_called_once()
    assert mock_run.call_args[0][0][-1].endswith("batch_0000_0000.png")


def test_extract_frames_batches_reduce_ffmpeg_calls(tmp_path: Path):
    plans = [
        PlannedFrame(timestamp_sec=60.0, source="uniform"),
        PlannedFrame(timestamp_sec=120.0, source="uniform"),
        PlannedFrame(timestamp_sec=300.0, source="uniform"),
    ]

    with (
        patch.object(settings, "video_extract_batch_window_sec", 120.0),
        patch.object(settings, "video_frame_concurrency", 2),
        patch("app.parsing.video_service.subprocess.run", side_effect=_fake_ffmpeg_run) as mock_run,
    ):
        results = extract_frames_at_timestamps("/tmp/video.mp4", tmp_path, plans)

    assert mock_run.call_count == 2
    assert len(results) == 3
    cmds = [call[0][0] for call in mock_run.call_args_list]
    batch_cmd = next(cmd for cmd in cmds if "select=" in " ".join(cmd))
    assert "between(t\\,60.000\\,60.500)" in batch_cmd[batch_cmd.index("-vf") + 1]


def test_extract_video_audio_failure_returns_empty_path():
    with patch(
        "app.parsing.video_service.extract_audio_wav",
        side_effect=__import__("subprocess").CalledProcessError(1, "ffmpeg"),
    ):
        audio_path, temp_dir = extract_video_audio("/tmp/video.mp4")
    assert audio_path == ""
    assert temp_dir


def test_extract_planned_frames_empty_duration():
    with (
        patch("app.parsing.video_service.probe_video_duration", return_value=0.0),
        patch("app.parsing.video_service.extract_frames_at_timestamps", return_value=[]),
    ):
        assert extract_planned_frames("/tmp/video.mp4", "/tmp", []) == []


def test_probe_video_duration_parses_output():
    from app.parsing.video_service import probe_video_duration

    class Result:
        stdout = "123.45\n"
        returncode = 0

    with (
        patch("app.parsing.video_service.shutil.which", return_value="/usr/bin/ffprobe"),
        patch("app.parsing.video_service.subprocess.run", return_value=Result()),
    ):
        assert probe_video_duration("/tmp/video.mp4") == 123.45


def test_dedupe_planned_frames_empty():
    assert dedupe_planned_frames([], threshold=5) == []

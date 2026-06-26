from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from app.parsing.video_service import dedupe_frames, extract_frames, extract_video_assets


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


def test_dedupe_frames_skips_similar(tmp_path: Path):
    frame1 = tmp_path / "f1.png"
    frame2 = tmp_path / "f2.png"
    frame3 = tmp_path / "f3.png"
    _write_png(frame1, (255, 0, 0))
    _write_png(frame2, (255, 0, 0))
    _write_pattern_png(frame3)

    frames = [(0.0, str(frame1)), (30.0, str(frame2)), (60.0, str(frame3))]
    kept = dedupe_frames(frames, threshold=5)

    assert len(kept) == 2
    assert kept[0][1] == str(frame1)
    assert kept[1][1] == str(frame3)


def test_extract_frames_returns_empty_on_ffmpeg_failure(tmp_path):
    with patch("app.parsing.video_service.subprocess.run", side_effect=__import__("subprocess").CalledProcessError(1, "ffmpeg")):
        frames = extract_frames("/tmp/video.mp4", tmp_path)
    assert frames == []


def test_extract_video_assets_audio_only():
    with (
        patch("app.parsing.video_service.extract_audio_wav"),
        patch("app.parsing.video_service.extract_frames", return_value=[]),
        patch("app.parsing.video_service.dedupe_frames", side_effect=lambda frames, threshold: frames),
    ):
        result = extract_video_assets("/tmp/audio_video.mp4")
    assert result.audio_path
    assert result.frame_paths == []


def test_extract_video_assets_raises_when_no_content():
    with (
        patch("app.parsing.video_service.extract_audio_wav", side_effect=__import__("subprocess").CalledProcessError(1, "ffmpeg")),
        patch("app.parsing.video_service.extract_frames", return_value=[]),
    ):
        with pytest.raises(ValueError, match="No audio or video content"):
            extract_video_assets("/tmp/empty.mp4")


def test_extract_scene_frames_parses_pts_time(tmp_path, monkeypatch):
    frame_path = tmp_path / "frame_0001.png"
    frame_path.write_bytes(b"png")

    class Result:
        stderr = "showinfo pts_time:12.5\n"
        returncode = 0

    def fake_run(cmd, **kwargs):
        if "-vf" in cmd and "scene" in cmd[cmd.index("-vf") + 1]:
            return Result()
        raise __import__("subprocess").CalledProcessError(1, cmd)

    monkeypatch.setattr("app.parsing.video_service.settings.video_frame_mode", "scene")
    with (
        patch("app.parsing.video_service.subprocess.run", side_effect=fake_run),
        patch("app.parsing.video_service.Path.glob", return_value=[frame_path]),
    ):
        frames = __import__("app.parsing.video_service", fromlist=["extract_frames"]).extract_frames(
            "/tmp/video.mp4", tmp_path
        )

    assert len(frames) == 1
    assert frames[0][0] == 12.5


def test_dedupe_frames_empty():
    assert dedupe_frames([], threshold=5) == []

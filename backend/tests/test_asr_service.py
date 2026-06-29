from unittest.mock import MagicMock, patch

import pytest

from app.config import settings
from app.parsing.asr_service import _segments_from_verbose, transcribe_audio, transcribe_audio_segments


def test_segments_from_verbose_json():
    body = {
        "text": "hello world",
        "segments": [
            {"start": 0.0, "end": 1.5, "text": "hello"},
            {"start": 1.5, "end": 3.0, "text": "world"},
        ],
    }
    segments = _segments_from_verbose(body)
    assert len(segments) == 2
    assert segments[0].text == "hello"
    assert segments[0].start_sec == 0.0
    assert segments[1].end_sec == 3.0


def test_segments_from_verbose_text_only():
    body = {"text": "full transcript", "duration": 10.0}
    segments = _segments_from_verbose(body)
    assert len(segments) == 1
    assert segments[0].text == "full transcript"
    assert segments[0].end_sec == 10.0


def test_transcribe_audio_segments_verbose_json():
    body = {
        "segments": [
            {"start": 0.0, "end": 2.0, "text": "segment one"},
        ],
    }
    with patch("app.parsing.asr_service._post_transcription", return_value=body):
        segments = transcribe_audio_segments("/tmp/audio.wav")
    assert len(segments) == 1
    assert segments[0].text == "segment one"


def test_transcribe_audio_segments_plain_text_fallback():
    with patch("app.parsing.asr_service._post_transcription", return_value="plain text result"):
        segments = transcribe_audio_segments("/tmp/audio.wav")
    assert len(segments) == 1
    assert segments[0].text == "plain text result"


def test_transcribe_audio_joins_segments():
    body = {
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "a"},
            {"start": 1.0, "end": 2.0, "text": "b"},
        ],
    }
    with patch("app.parsing.asr_service.transcribe_audio_segments") as mock_segments:
        mock_segments.return_value = _segments_from_verbose(body)
        assert transcribe_audio("/tmp/audio.wav") == "a b"


def test_should_proactive_split_by_duration():
    from app.parsing.asr_service import _should_proactive_split

    with (
        patch.object(settings, "asr_proactive_split_enabled", True),
        patch.object(settings, "asr_proactive_split_min_duration_sec", 600),
        patch("app.parsing.asr_service._probe_audio_duration", return_value=1200.0),
    ):
        assert _should_proactive_split("/tmp/long.wav") is True


def test_should_not_proactive_split_short_audio():
    from app.parsing.asr_service import _should_proactive_split

    with (
        patch.object(settings, "asr_proactive_split_enabled", True),
        patch("app.parsing.asr_service._probe_audio_duration", return_value=120.0),
        patch("pathlib.Path.stat") as mock_stat,
    ):
        mock_stat.return_value.st_size = 1024 * 1024
        assert _should_proactive_split("/tmp/short.wav") is False


def test_transcribe_proactive_split_merges_offsets():
    part_one = {
        "segments": [{"start": 0.0, "end": 2.0, "text": "hello"}],
    }
    part_two = {
        "segments": [{"start": 0.0, "end": 2.0, "text": "world"}],
    }

    with (
        patch.object(settings, "asr_proactive_split_enabled", True),
        patch.object(settings, "asr_proactive_segment_sec", 600),
        patch("app.parsing.asr_service._should_proactive_split", return_value=True),
        patch(
            "app.parsing.asr_service._split_audio_segments",
            return_value=[(0.0, "/tmp/part0.wav"), (600.0, "/tmp/part1.wav")],
        ),
        patch(
            "app.parsing.asr_service._post_transcription",
            side_effect=[part_one, part_two],
        ),
    ):
        segments = transcribe_audio_segments("/tmp/long.wav")

    assert len(segments) == 2
    assert segments[0].text == "hello"
    assert segments[0].start_sec == 0.0
    assert segments[1].text == "world"
    assert segments[1].start_sec == 600.0


def test_segments_from_verbose_applies_offset():
    body = {
        "segments": [{"start": 1.0, "end": 3.0, "text": "part"}],
    }
    segments = _segments_from_verbose(body, offset=600.0)
    assert segments[0].start_sec == 601.0
    assert segments[0].end_sec == 603.0

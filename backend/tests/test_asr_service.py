from unittest.mock import MagicMock, patch

import pytest

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

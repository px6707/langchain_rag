from app.parsing.time_metadata import extract_time_metadata


def test_extract_time_metadata_frame_ocr():
    meta = {
        "filename": "lecture.mp4",
        "file_type": "video",
        "content_type": "frame_ocr",
        "timestamp_sec": 125.5,
        "asr_start_sec": 120.0,
        "asr_end_sec": 180.0,
    }
    result = extract_time_metadata(meta)
    assert result["file_type"] == "video"
    assert result["content_type"] == "frame_ocr"
    assert result["timestamp_sec"] == 125.5
    assert result["start_sec"] == 120.0
    assert result["end_sec"] == 180.0


def test_extract_time_metadata_audio_transcript():
    meta = {
        "filename": "talk.mp4",
        "content_type": "audio_transcript",
        "start_sec": 60.0,
        "end_sec": 90.0,
    }
    result = extract_time_metadata(meta)
    assert result["file_type"] == "video"
    assert result["start_sec"] == 60.0
    assert result["end_sec"] == 90.0
    assert "timestamp_sec" not in result


def test_extract_time_metadata_infers_file_type_from_filename():
    meta = {"filename": "clip.webm", "start_sec": 1.0}
    result = extract_time_metadata(meta)
    assert result["file_type"] == "video"

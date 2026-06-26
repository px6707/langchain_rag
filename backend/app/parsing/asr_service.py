import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class TranscriptSegment:
    start_sec: float
    end_sec: float
    text: str


def _asr_config() -> tuple[str, str, str]:
    api_base = settings.asr_api_base or settings.llm_api_base
    api_key = settings.asr_api_key or settings.llm_api_key
    model = settings.asr_model
    if not api_key:
        raise ValueError("ASR_API_KEY is not configured")
    return api_base.rstrip("/"), api_key, model


def _post_transcription(
    file_path: str,
    *,
    response_format: str | None = None,
) -> dict | str:
    api_base, api_key, model = _asr_config()
    url = f"{api_base}/v1/audio/transcriptions"
    path = Path(file_path)
    data: dict[str, str] = {"model": model}
    if response_format:
        data["response_format"] = response_format
    headers = {"Authorization": f"Bearer {api_key}"}
    with path.open("rb") as f:
        files = {"file": (path.name, f, "application/octet-stream")}
        with httpx.Client(timeout=600.0) as client:
            response = client.post(url, headers=headers, data=data, files=files)
            response.raise_for_status()
            if response_format == "verbose_json":
                return response.json()
            return response.text


def _segments_from_verbose(body: dict) -> list[TranscriptSegment]:
    raw_segments = body.get("segments")
    if not isinstance(raw_segments, list):
        text = str(body.get("text", "")).strip()
        if not text:
            return []
        duration = float(body.get("duration") or 0.0)
        return [TranscriptSegment(start_sec=0.0, end_sec=duration, text=text)]

    segments: list[TranscriptSegment] = []
    for item in raw_segments:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        segments.append(
            TranscriptSegment(
                start_sec=float(item.get("start", 0.0)),
                end_sec=float(item.get("end", 0.0)),
                text=text,
            )
        )
    return segments


def _split_audio_segments(file_path: str, segment_sec: int) -> list[tuple[float, str]]:
    if shutil.which("ffmpeg") is None:
        logger.warning("ffmpeg not found; cannot split audio for ASR fallback")
        return [(0.0, file_path)]

    temp_dir = tempfile.mkdtemp(prefix="rag_asr_")
    temp_path = Path(temp_dir)
    pattern = str(temp_path / "part_%04d.wav")
    cmd = [
        "ffmpeg", "-y", "-i", file_path,
        "-f", "segment", "-segment_time", str(segment_sec),
        "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        pattern,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return [(0.0, file_path)]

    parts = sorted(temp_path.glob("part_*.wav"))
    result: list[tuple[float, str]] = []
    for idx, part in enumerate(parts):
        result.append((idx * segment_sec, str(part)))
    if not result:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return [(0.0, file_path)]
    return result


def _transcribe_with_fallback_segments(file_path: str) -> list[TranscriptSegment]:
    segment_sec = settings.asr_fallback_segment_sec
    parts = _split_audio_segments(file_path, segment_sec)
    segments: list[TranscriptSegment] = []
    temp_dirs: set[str] = set()

    for offset, part_path in parts:
        if part_path != file_path:
            temp_dirs.add(str(Path(part_path).parent))
        try:
            body = _post_transcription(part_path)
            if isinstance(body, dict):
                text = str(body.get("text", "")).strip()
            else:
                text = body.strip()
        except Exception:
            logger.exception("ASR fallback segment failed: %s", part_path)
            continue
        if text:
            segments.append(
                TranscriptSegment(
                    start_sec=float(offset),
                    end_sec=float(offset + segment_sec),
                    text=text,
                )
            )

    for temp_dir in temp_dirs:
        shutil.rmtree(temp_dir, ignore_errors=True)
    return segments


def transcribe_audio_segments(file_path: str) -> list[TranscriptSegment]:
    if settings.asr_use_verbose_json:
        try:
            body = _post_transcription(file_path, response_format="verbose_json")
            if isinstance(body, dict):
                segments = _segments_from_verbose(body)
                if segments:
                    return segments
        except httpx.HTTPStatusError as exc:
            logger.warning("ASR verbose_json failed (%s), falling back", exc.response.status_code)
        except Exception:
            logger.exception("ASR verbose_json failed, falling back")

    try:
        body = _post_transcription(file_path)
        if isinstance(body, dict):
            text = str(body.get("text", "")).strip()
        else:
            text = body.strip()
        if text:
            return [TranscriptSegment(start_sec=0.0, end_sec=0.0, text=text)]
    except Exception:
        logger.exception("ASR plain transcription failed, trying ffmpeg fallback")

    return _transcribe_with_fallback_segments(file_path)


def transcribe_audio(file_path: str) -> str:
    segments = transcribe_audio_segments(file_path)
    return " ".join(seg.text for seg in segments).strip()

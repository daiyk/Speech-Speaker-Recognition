"""Prepare local audio/video inputs while retaining the original media timeline."""

import json
import math
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator, Tuple

import soundfile as sf

from .ffmpeg_utils import ensure_ffmpeg

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".aac", ".wma"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm"}
MEDIA_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS


@dataclass(frozen=True)
class MediaInfo:
    duration: float
    audio_streams: Tuple[int, ...]

    @property
    def selected_stream(self) -> int:
        return self.audio_streams[0]


def validate_media_path(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Media file not found: {path}")
    if path.suffix.lower() not in MEDIA_EXTENSIONS:
        raise ValueError(f"Unsupported media extension: {path.suffix}")


def probe_media(path: Path) -> MediaInfo:
    validate_media_path(path)
    tools = ensure_ffmpeg()
    try:
        result = subprocess.run(
            [
                tools.ffprobe,
                "-v",
                "error",
                "-show_format",
                "-show_streams",
                "-of",
                "json",
                str(path.resolve()),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            stdin=subprocess.DEVNULL,
        )
        if result.returncode:
            raise ValueError(result.stderr[-2000:])
        metadata = json.loads(result.stdout)
        streams = tuple(
            int(s["index"])
            for s in metadata.get("streams", [])
            if s.get("codec_type") == "audio"
        )
        if not streams:
            raise ValueError("No audio track found")
        duration = float(metadata.get("format", {}).get("duration", 0))
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("Media has no valid positive duration")
        return MediaInfo(duration, streams)
    except (OSError, subprocess.TimeoutExpired, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"Cannot read media '{path}': {exc}") from exc


def preflight_media(path: Path) -> None:
    """Check video inputs before a CLI entry point loads speech models."""
    validate_media_path(path)
    if path.suffix.lower() in VIDEO_EXTENSIONS:
        probe_media(path)


@contextmanager
def prepared_audio(path: Path, force: bool = False) -> Iterator[Path]:
    """Yield a WAV for video (or batch input), cleaning it up even on interruption."""
    validate_media_path(path)
    if path.suffix.lower() not in VIDEO_EXTENSIONS and not force:
        yield path
        return
    info = probe_media(path)
    tools = ensure_ffmpeg()
    with TemporaryDirectory(prefix="speech-media-") as directory:
        output = Path(directory) / (path.stem + ".wav")
        # Keep stream timestamps relative to container start. first_pts inserts
        # leading silence for delayed audio; apad retains trailing video silence.
        command = [
            tools.ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-copyts",
            "-start_at_zero",
            "-i",
            str(path.resolve()),
            "-map",
            f"0:{info.selected_stream}",
            "-vn",
            "-sn",
            "-dn",
            "-af",
            "aresample=16000:async=1:first_pts=0,apad",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            "-t",
            str(info.duration),
            str(output),
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=max(120, info.duration * 2),
                stdin=subprocess.DEVNULL,
            )
            if result.returncode:
                raise ValueError(result.stderr[-2000:])
            audio = sf.info(str(output))
            if audio.frames <= 0 or audio.samplerate != 16000 or audio.channels != 1:
                raise ValueError("Decoded audio is empty or invalid")
        except (OSError, RuntimeError, subprocess.TimeoutExpired, ValueError) as exc:
            raise ValueError(f"Cannot extract audio from '{path}': {exc}") from exc
        yield output

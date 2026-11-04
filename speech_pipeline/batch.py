"""Process mixed media directories using PCM chunks and sample-accurate offsets."""

import math
from collections import Counter
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, List, Optional

import soundfile as sf

from .media_utils import MEDIA_EXTENSIONS, prepared_audio
from .models import TranscriptionResult

if TYPE_CHECKING:
    from .pipeline import SpeechPipeline


def discover_inputs(directory: Path) -> List[Path]:
    return sorted(
        (
            p
            for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in MEDIA_EXTENSIONS
        ),
        key=lambda p: p.name.casefold(),
    )


def output_names(inputs: List[Path], output_format: str) -> List[str]:
    """Preserve unique stems and disambiguate collisions, including case variants."""
    counts = Counter(p.stem.casefold() for p in inputs)
    used = set()
    names = []
    for source in inputs:
        stem = source.stem if counts[source.stem.casefold()] == 1 else source.name
        candidate = f"{stem}.{output_format}"
        index = 2
        while candidate.casefold() in used:
            candidate = f"{stem}-{index}.{output_format}"
            index += 1
        used.add(candidate.casefold())
        names.append(candidate)
    return names


def process_in_chunks(
    pipeline: "SpeechPipeline",
    source: Path,
    output: Path,
    chunk_length: float,
    output_format: str,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
) -> TranscriptionResult:
    if not math.isfinite(chunk_length) or chunk_length <= 0:
        raise ValueError("Chunk length must be a finite positive number")
    with (
        prepared_audio(source, force=True) as audio,
        TemporaryDirectory(prefix="speech-chunks-") as directory,
    ):
        chunk = Path(directory) / "chunk.wav"
        segments = []
        with sf.SoundFile(str(audio)) as stream:
            sample_rate = stream.samplerate
            frames_per_chunk = max(1, round(chunk_length * sample_rate))
            total_duration = len(stream) / sample_rate
            position = 0
            while position < len(stream):
                data = stream.read(frames_per_chunk, dtype="float32")
                sf.write(str(chunk), data, sample_rate, subtype="PCM_16")
                result = pipeline.process(
                    str(chunk),
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                )
                offset = position / sample_rate
                duration = len(data) / sample_rate
                for segment in result.segments:
                    start = max(0.0, segment.start)
                    end = min(duration, segment.end)
                    if start < end:
                        segments.append(
                            replace(segment, start=offset + start, end=offset + end)
                        )
                position += len(data)
        combined = TranscriptionResult(
            segments,
            total_duration,
            sorted({s.speaker for s in segments if s.speaker}),
        )
        pipeline._save_result(combined, str(output), output_format)
        return combined

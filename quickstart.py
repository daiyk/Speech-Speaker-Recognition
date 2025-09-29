"""Quick-start helper for the refactored speech pipeline.

The script walks through:
1. Verifying runtime prerequisites (Python modules + Hugging Face token).
2. Loading the new end-to-end pipeline that performs diarization, Whisper
   transcription, and merged caption generation.
3. Running a miniature demo that shows sentence-level speaker turns produced by
   the unified merge logic.

Run it with: ``uv run python quickstart.py``
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import suppress
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable

# Ensure the package can be imported when running the file directly.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from speech_pipeline import Config, SpeechPipeline
from speech_pipeline.models import SpeakerSegment


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def check_dependencies() -> bool:
    """Confirm that the runtime has all critical third-party packages."""

    try:
        import torch  # noqa: F401
        import faster_whisper  # noqa: F401
        import pyannote.audio  # noqa: F401
        import librosa  # noqa: F401
        import soundfile  # noqa: F401
        import pydub  # noqa: F401
        import numpy as np  # noqa: F401
    except ImportError as exc:  # pragma: no cover - user environment check
        logger.error("❌ Missing dependency: %s", exc)
        logger.info("   Install everything with: uv pip install -e .")
        return False

    logger.info("✅ Dependencies imported successfully.")
    return True


def load_configuration() -> Config | None:
    """Load and validate configuration from the environment."""

    try:
        config = Config.from_env()
        config.validate()
    except Exception as exc:  # pragma: no cover - configuration guard
        logger.error("❌ Configuration error: %s", exc)
        logger.info("   Check that HUGGINGFACE_TOKEN and OUTPUT_FORMAT are defined in your .env file.")
        return None

    logger.info("✅ Configuration loaded for Whisper '%s' and pyannote '%s'.",
                config.whisper_model, config.pyannote_model)
    return config


def bootstrap_pipeline(base_config: Config) -> SpeechPipeline | None:
    """Initialise the speech pipeline with light-weight models for demo use."""

    demo_config = Config(**vars(base_config))
    demo_config.whisper_model = os.getenv("QUICKSTART_WHISPER_MODEL", "tiny")

    try:
        pipeline = SpeechPipeline(config=demo_config)
    except Exception as exc:  # pragma: no cover - heavy initialisation guard
        logger.error("❌ Failed to load models: %s", exc)
        logger.info("   Confirm your Hugging Face token has access to %s.",
                    demo_config.pyannote_model)
        return None

    info = pipeline.get_model_info()
    logger.info("✅ Pipeline ready on '%s' (Whisper=%s, Pyannote=%s).",
                info["device"], info["whisper_model"], info["pyannote_model"])
    return pipeline


def generate_demo_audio(destination: Path) -> Path:
    """Synthesize a short demo waveform with two pseudo-speakers."""

    import numpy as np
    import soundfile as sf

    sample_rate = 16000
    seconds = 6
    t = np.linspace(0, seconds, sample_rate * seconds, endpoint=False)

    speaker_a = 0.25 * np.sin(2 * np.pi * 220 * t[: sample_rate * 3])
    speaker_b = 0.25 * np.sin(2 * np.pi * 660 * t[: sample_rate * 3])
    fade = np.linspace(0.0, 1.0, sample_rate)

    track = np.concatenate([
        speaker_a,
        fade * speaker_b[:sample_rate] + (1 - fade) * speaker_a[:sample_rate],
        speaker_b[sample_rate:],
    ])

    file_path = destination / "quickstart-demo.wav"
    sf.write(file_path, track.astype("float32"), sample_rate)

    logger.info("✅ Demo audio created at %s (%.1f seconds).", file_path, seconds)
    return file_path


def print_segment_overview(segments: Iterable[SpeakerSegment]) -> None:
    """Pretty-print merged speaker turns produced by the pipeline."""

    logger.info("\n🗒️  Merged caption preview:")
    for index, segment in enumerate(segments, start=1):
        start = segment.start
        end = segment.end
        text = segment.text or "[no transcription]"
        confidence = f"{segment.confidence:.2f}" if segment.confidence is not None else "--"
        logger.info("%2d. %s | %6.2fs → %6.2fs | conf=%s\n    %s",
                    index, segment.speaker, start, end, confidence, text)


def run_demo(pipeline: SpeechPipeline, tmp_dir: Path) -> None:
    """Execute the pipeline on generated audio and summarise the outcome."""

    demo_audio = generate_demo_audio(tmp_dir)
    output_file = tmp_dir / "quickstart-output.srt"

    result = pipeline.process(
        str(demo_audio),
        output_path=str(output_file),
        output_format="srt",
    )

    logger.info("\n✅ Transcription finished: %.2fs total, %d speakers, %d caption blocks.",
                result.total_duration, len(result.speakers), len(result.segments))
    print_segment_overview(result.segments[:5])

    logger.info("\n📄 Full SRT saved to: %s", output_file)


def main() -> None:
    logger.info("🚀 Speech Pipeline Quickstart")
    logger.info("=" * 60)

    if not check_dependencies():
        return

    config = load_configuration()
    if config is None:
        return

    pipeline = bootstrap_pipeline(config)
    if pipeline is None:
        return

    with TemporaryDirectory(prefix="speech-pipeline-quickstart-") as tmp:
        tmp_path = Path(tmp)
        run_demo(pipeline, tmp_path)

        logger.info("\n🧪 Temporary assets kept in %s while the script is running.", tmp_path)

    logger.info("\n✨ Done. Replace the demo file with your own audio and run either:")
    logger.info("   uv run speech-pipeline process your_audio.wav --output subtitles.srt")
    logger.info("or call the pipeline from Python just like in run_demo().")


if __name__ == "__main__":
    with suppress(KeyboardInterrupt):
        main()
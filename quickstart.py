"""Quick-start helper for the refactored speech pipeline.

The script now supports two modes:
1. Pipeline mode (default): verify the runtime, load configured models, and
    process a user-supplied audio file end-to-end.
2. Demo mode (``-demo``): spin up lightweight models and run against a
    synthesized six-second audio clip to showcase diarization and captions.

Run it with: ``uv run python quickstart.py <audio-file>`` or add ``-demo`` to try the
synthetic sample. Add ``--mode transcribe`` or ``--mode timestamps`` to skip speaker
diarization (no Hugging Face token or pyannote model needed).
"""

from __future__ import annotations

import argparse
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
from speech_pipeline.config import PIPELINE_MODES
from speech_pipeline.models import SpeakerSegment


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the quickstart script."""

    parser = argparse.ArgumentParser(
        description="Run the speech pipeline on your audio or a built-in demo clip.",
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="Path to an audio or video file. Required unless -demo/--demo is specified.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Destination file for the transcript. Defaults to <input>.<format>.",
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=["txt", "srt", "vtt", "json"],
        help="Override the configured output format.",
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=list(PIPELINE_MODES),
        help="Stages to run: transcribe, timestamps, or diarize (default: PIPELINE_MODE or diarize).",
    )
    parser.add_argument(
        "-demo",
        "--demo",
        action="store_true",
        help="Run the built-in six-second demo instead of your audio file.",
    )

    args_list = list(argv) if argv is not None else None
    return parser.parse_args(args_list)


def check_dependencies(need_diarization: bool) -> bool:
    """Confirm that the runtime has the third-party packages the mode needs."""

    try:
        import torch  # noqa: F401
        import faster_whisper  # noqa: F401
        if need_diarization:
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


def load_configuration(mode: str | None, output_format: str | None) -> Config | None:
    """Load and validate configuration from the environment."""

    try:
        config = Config.from_env()
        if mode:
            config.mode = mode
        if output_format:
            config.output_format = output_format
        config.validate()
    except Exception as exc:  # pragma: no cover - configuration guard
        logger.error("❌ Configuration error: %s", exc)
        logger.info("   Check PIPELINE_MODE / OUTPUT_FORMAT in your .env file "
                    "(and HUGGINGFACE_TOKEN for diarize mode).")
        return None

    if config.diarization_enabled:
        logger.info("✅ Configuration loaded: mode '%s', Whisper '%s', pyannote '%s'.",
                    config.mode, config.whisper_model, config.pyannote_model)
    else:
        logger.info("✅ Configuration loaded: mode '%s', Whisper '%s' (diarization disabled).",
                    config.mode, config.whisper_model)
    return config


def bootstrap_pipeline(base_config: Config, demo_mode: bool) -> SpeechPipeline | None:
    """Initialise the speech pipeline, optionally using lightweight demo models."""

    effective_config = Config(**vars(base_config))
    if demo_mode:
        effective_config.whisper_model = os.getenv("QUICKSTART_WHISPER_MODEL", "tiny")
        logger.info("🎯 Demo mode: forcing Whisper model '%s'.", effective_config.whisper_model)
    else:
        logger.info("🎯 Using configured Whisper model '%s'.", effective_config.whisper_model)

    try:
        pipeline = SpeechPipeline(config=effective_config)
    except Exception as exc:  # pragma: no cover - heavy initialisation guard
        logger.error("❌ Failed to load models: %s", exc)
        if effective_config.diarization_enabled:
            logger.info("   Confirm your Hugging Face token has access to %s.",
                        effective_config.pyannote_model)
        return None

    info = pipeline.get_model_info()
    logger.info("✅ Pipeline ready on '%s' (mode=%s, Whisper=%s, Pyannote=%s).",
                info["device"], info["mode"], info["whisper_model"],
                info["pyannote_model"] or "disabled")
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

    segment_list = list(segments)
    logger.info("\n🗒️  Merged caption preview:")
    if not segment_list:
        logger.info("   [no segments returned]")
        return

    for index, segment in enumerate(segment_list, start=1):
        start = segment.start
        end = segment.end
        text = segment.text or "[no transcription]"
        confidence = f"{segment.confidence:.2f}" if segment.confidence is not None else "--"
        speaker = segment.speaker or "-"
        logger.info("%2d. %s | %6.2fs → %6.2fs | conf=%s\n    %s",
                    index, speaker, start, end, confidence, text)


def summarise(pipeline: SpeechPipeline, result) -> None:
    """Log a short summary of a pipeline result."""

    if pipeline.config.diarization_enabled:
        logger.info("\n✅ Transcription finished: %.2fs total, %d speakers, %d caption blocks.",
                    result.total_duration, len(result.speakers), len(result.segments))
    else:
        logger.info("\n✅ Transcription finished (%s mode): %.2fs total, %d segments.",
                    pipeline.config.mode, result.total_duration, len(result.segments))
    print_segment_overview(result.segments[:5])


def run_demo(pipeline: SpeechPipeline, tmp_dir: Path) -> None:
    """Execute the pipeline on generated audio and summarise the outcome."""

    demo_audio = generate_demo_audio(tmp_dir)
    output_format = pipeline.config.effective_output_format()
    output_file = tmp_dir / f"quickstart-output.{output_format}"

    result = pipeline.process(
        str(demo_audio),
        output_path=str(output_file),
        output_format=output_format,
    )

    summarise(pipeline, result)
    logger.info("\n📄 Full %s saved to: %s", output_format, output_file)


def run_pipeline(pipeline: SpeechPipeline, audio_path: Path, output_path: Path, output_format: str) -> None:
    """Execute the pipeline on user audio and summarise the outcome."""

    result = pipeline.process(
        str(audio_path),
        output_path=str(output_path),
        output_format=output_format,
    )

    summarise(pipeline, result)

    logger.info("\n📄 Full %s saved to: %s", output_format, output_path)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)

    logger.info("🚀 Speech Pipeline Quickstart")
    logger.info("=" * 60)

    if args.demo:
        logger.info("🧪 Demo mode enabled: generating synthetic audio sample.")
    else:
        logger.info("🎧 Processing user-supplied audio input.")

    config = load_configuration(args.mode, args.format)
    if config is None:
        return

    if not check_dependencies(need_diarization=config.diarization_enabled):
        return

    if not args.demo:
        from speech_pipeline.media_utils import preflight_media
        if args.input is None:
            logger.error("No input provided. Pass an audio/video file or use --demo.")
            return
        try:
            preflight_media(Path(args.input).expanduser())
        except (ValueError, RuntimeError, OSError) as exc:
            logger.error("Cannot process input: %s", exc)
            return

    pipeline = bootstrap_pipeline(config, demo_mode=args.demo)
    if pipeline is None:
        return

    if args.demo:
        with TemporaryDirectory(prefix="speech-pipeline-quickstart-") as tmp:
            tmp_path = Path(tmp)
            run_demo(pipeline, tmp_path)

            logger.info("\n🧪 Temporary assets kept in %s while the script is running.", tmp_path)

        logger.info("\n✨ Demo complete. Supply an audio file instead of -demo to process your own content.")
        return

    if args.input is None:
        logger.error("❌ No input audio provided. Pass a file path or use -demo for the synthetic example.")
        return

    input_path = Path(args.input).expanduser()
    if not input_path.exists():
        logger.error("❌ Input audio not found at %s", input_path)
        return

    output_format = config.effective_output_format()
    output_path = (
        Path(args.output).expanduser()
        if args.output
        else input_path.with_suffix(f".{output_format}")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("🔊 Input audio: %s", input_path)
    logger.info("📝 Output file: %s (%s)", output_path, output_format)

    run_pipeline(pipeline, input_path, output_path, output_format)

    logger.info("\n✨ Done. Re-run with -demo to generate the quick sample, or adjust -f/-o for different outputs.")


if __name__ == "__main__":
    with suppress(KeyboardInterrupt):
        main()

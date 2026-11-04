"""Command-line interface for the speech pipeline."""

import logging
import math
import sys
from pathlib import Path
from typing import Optional
import click

from .config import Config, PIPELINE_MODES, MODE_DIARIZE


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--quiet', '-q', is_flag=True, help='Enable quiet mode')
def cli(verbose: bool, quiet: bool) -> None:
    """Speech recognition and speaker diarization pipeline."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif quiet:
        logging.getLogger().setLevel(logging.WARNING)


@cli.command()
@click.argument('input_file', type=click.Path(exists=True, path_type=Path))
@click.option('--mode', '-m',
              type=click.Choice(list(PIPELINE_MODES)),
              default=None,
              help='Stages to run: transcribe (text only), timestamps (text + '
                   'timestamps), diarize (full pipeline with speakers; needs '
                   'HUGGINGFACE_TOKEN). Default: PIPELINE_MODE from config, else diarize.')
@click.option('--output', '-o', type=click.Path(path_type=Path), help='Output file path')
@click.option('--format', '-f', 'output_format', 
              type=click.Choice(['txt', 'srt', 'vtt', 'json']), 
              default=None, help='Output format (default: txt for transcribe, srt otherwise)')
@click.option('--whisper-model', '-w', 
              type=click.Choice(['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3']),
              default=None, help='Whisper model to use')
@click.option('--pyannote-model', 
              type=str,
              help='Pyannote diarization model to use (overrides config)')
@click.option('--speaker-label-prefix', '--speaker-label',
              type=str,
              help='Prefix for speaker labels (default: value from config)')
@click.option('--min-speakers', type=int, help='Minimum number of speakers (diarize mode only)')
@click.option('--max-speakers', type=int, help='Maximum number of speakers (diarize mode only)')
@click.option('--device', type=click.Choice(['cpu', 'cuda', 'mps']), 
              help='Device to run models on')
@click.option('--config', type=click.Path(exists=True, path_type=Path), 
              help='Configuration file path')
def process(
    input_file: Path,
    mode: Optional[str],
    output: Optional[Path],
    output_format: Optional[str],
    whisper_model: Optional[str],
    pyannote_model: Optional[str],
    speaker_label_prefix: Optional[str],
    min_speakers: Optional[int],
    max_speakers: Optional[int],
    device: Optional[str],
    config: Optional[Path]
) -> None:
    """Process an audio or video file (transcription, timestamps, and/or speakers)."""
    try:
        from .media_utils import preflight_media
        preflight_media(input_file)
        # Imported here so `info`/`models`/`setup` don't pay for loading torch.
        from .pipeline import SpeechPipeline

        # Load configuration
        if config:
            pipeline_config = Config.from_env(str(config))
        else:
            pipeline_config = Config.from_env()
        
        # Override config with command-line options
        if mode:
            pipeline_config.mode = mode
        if whisper_model:
            pipeline_config.whisper_model = whisper_model
        if output_format:
            pipeline_config.output_format = output_format
        if pyannote_model:
            pipeline_config.pyannote_model = pyannote_model
        if speaker_label_prefix:
            pipeline_config.speaker_labels = speaker_label_prefix
        
        # Initialize pipeline
        logger.info("Initializing speech pipeline...")
        pipeline = SpeechPipeline(
            config=pipeline_config,
            device=device
        )
        
        # Determine output path
        resolved_format = pipeline_config.effective_output_format()
        if output is None:
            output = input_file.with_suffix(f'.{resolved_format}')
        elif output.suffix.lower() != f'.{resolved_format}':
            output = output.with_suffix(f'.{resolved_format}')
        
        # Process audio
        logger.info(f"Processing {input_file}...")
        result = pipeline.process(
            str(input_file),
            min_speakers=min_speakers if min_speakers is not None else pipeline_config.min_speakers,
            max_speakers=max_speakers if max_speakers is not None else pipeline_config.max_speakers,
            output_path=str(output),
            output_format=resolved_format
        )
        
        # Print summary
        click.echo(f"\n✅ Processing completed!")
        click.echo(f"🧭 Mode: {pipeline.config.mode}")
        click.echo(f"📁 Input: {input_file}")
        click.echo(f"📄 Output: {output}")
        click.echo(f"🤖 Whisper model: {pipeline.config.whisper_model}")
        click.echo(f"⏱️  Duration: {result.total_duration:.2f}s")
        click.echo(f"🧩 Segments: {len(result.segments)}")

        if pipeline.config.mode != MODE_DIARIZE:
            return

        click.echo(f"🎭 Pyannote model: {pipeline.config.pyannote_model}")
        click.echo(f"👥 Speakers: {len(result.speakers)}")
        
        # Show speaker statistics
        stats = result.get_speaker_stats()
        click.echo("\n📊 Speaker Statistics:")
        for speaker, data in stats.items():
            click.echo(f"  {speaker}: {data['total_time']:.1f}s ({data['percentage']:.1f}%) "
                      f"- {data['segments_count']} segments, {data['word_count']} words")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)


@cli.command()
@click.argument('input_dir', type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option('--output-dir', type=click.Path(file_okay=False, path_type=Path))
@click.option('--mode', type=click.Choice(list(PIPELINE_MODES)), default=None)
@click.option('--chunk-length', type=click.FloatRange(min=0, min_open=True), default=60.0, show_default=True)
@click.option('--min-speakers', type=int)
@click.option('--max-speakers', type=int)
@click.option('--device', type=click.Choice(['cpu', 'cuda', 'mps']))
def batch(
    input_dir: Path,
    output_dir: Optional[Path],
    mode: Optional[str],
    chunk_length: float,
    min_speakers: Optional[int],
    max_speakers: Optional[int],
    device: Optional[str],
) -> None:
    """Process supported audio/video files in INPUT_DIR (non-recursively)."""
    from .batch import discover_inputs, output_names, process_in_chunks
    from .ffmpeg_utils import ensure_ffmpeg
    from .media_utils import preflight_media

    try:
        if not math.isfinite(chunk_length):
            raise ValueError("Chunk length must be finite")
        inputs = discover_inputs(input_dir)
        if not inputs:
            raise ValueError(f"No supported audio/video files found in {input_dir}")
        config = Config.from_env()
        if mode:
            config.mode = mode
        config.output_format = 'txt' if config.mode == 'transcribe' else 'srt'
        config.validate()
        ensure_ffmpeg()
        for source in inputs:
            preflight_media(source)
        from .pipeline import SpeechPipeline
        pipeline = SpeechPipeline(config=config, device=device)
        destination = (output_dir or input_dir / 'output') / f'final_{config.output_format}'
        destination.mkdir(parents=True, exist_ok=True)
        for source, name in zip(inputs, output_names(inputs, config.output_format)):
            output = destination / name
            click.echo(f"Processing {source.name}")
            process_in_chunks(
                pipeline, source, output, chunk_length, config.output_format,
                min_speakers if min_speakers is not None else config.min_speakers,
                max_speakers if max_speakers is not None else config.max_speakers,
            )
            click.echo(f"Saved {output}")
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


@cli.command()
@click.argument('input_file', type=click.Path(exists=True, path_type=Path))
def info(input_file: Path) -> None:
    """Get information about an audio or video file."""
    try:
        from .audio_utils import AudioProcessor
        from .media_utils import VIDEO_EXTENSIONS, probe_media
        
        processor = AudioProcessor()
        
        if input_file.suffix.lower() in VIDEO_EXTENSIONS:
            metadata = probe_media(input_file)
            duration = metadata.duration
            click.echo(f"Audio tracks: {len(metadata.audio_streams)}; selected stream: {metadata.selected_stream}")
        else:
            if not processor.validate_audio_format(str(input_file)):
                raise ValueError(f"Invalid audio file: {input_file}")
            duration = processor.get_audio_duration(str(input_file))
        
        click.echo(f"📁 File: {input_file}")
        click.echo(f"⏱️  Duration: {duration:.2f}s ({duration/60:.1f} minutes)")
        click.echo(f"📊 Size: {input_file.stat().st_size / (1024*1024):.1f} MB")
        click.echo(f"✅ Format: Supported")
        
    except Exception as e:
        click.echo(f"❌ Error analyzing file: {e}")
        sys.exit(1)


@cli.command()
def models() -> None:
    """Show information about available models."""
    click.echo("🤖 Available Whisper Models:")
    whisper_models = [
        ("tiny", "~39 MB", "Fastest, lowest accuracy"),
        ("base", "~74 MB", "Good balance of speed and accuracy"),
        ("small", "~244 MB", "Better accuracy, slower"),
        ("medium", "~769 MB", "High accuracy, much slower"),
        ("large", "~1550 MB", "Best accuracy, very slow"),
        ("large-v2", "~1550 MB", "Improved large model"),
        ("large-v3", "~1550 MB", "Latest large model")
    ]
    
    for name, size, desc in whisper_models:
        click.echo(f"  {name:10} {size:10} - {desc}")
    
    click.echo("\n🎭 Pyannote Models (diarize mode only):")
    click.echo("  pyannote/speaker-diarization-3.1 - Latest speaker diarization model")
    click.echo("  (Requires Hugging Face token and license acceptance)")


@cli.command()
def setup() -> None:
    """Setup guide for first-time users."""
    click.echo("🚀 Speech Pipeline Setup Guide")
    click.echo("=" * 40)
    
    click.echo("\n1. Sync dependencies (creates .venv automatically):")
    click.echo("   uv sync")
    click.echo("   uv sync --group dev  # optional: add developer tools")

    click.echo("\n2. (diarize mode only) Get Hugging Face token:")
    click.echo("   • Skip this step if you only need --mode transcribe or --mode timestamps")
    click.echo("   • Go to https://huggingface.co/")
    click.echo("   • Create account and get access token")
    click.echo("   • Accept license at https://huggingface.co/pyannote/speaker-diarization-3.1")
    
    click.echo("\n3. Create .env file:")
    click.echo("   cp .env.example .env")
    click.echo("   # Optional: set PIPELINE_MODE (transcribe | timestamps | diarize)")
    click.echo("   # For diarize mode, add your HUGGINGFACE_TOKEN")
    
    click.echo("\n4. Test installation:")
    click.echo("   uv run speech-pipeline models")
    click.echo("   uv run speech-pipeline info your_audio_file.wav")
    
    click.echo("\n5. Process audio:")
    click.echo("   uv run speech-pipeline process input.wav --mode transcribe   # text only")
    click.echo("   uv run speech-pipeline process input.wav --mode timestamps   # subtitles, no speakers")
    click.echo("   uv run speech-pipeline process input.wav --output output.srt # full pipeline")
    click.echo("   # Or try the demo: uv run python quickstart.py --demo")
    
    click.echo("\n📚 For more info, see README.md")


def main() -> None:
    """Entry point for the CLI."""
    cli()


if __name__ == '__main__':
    main()

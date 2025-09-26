"""Command-line interface for the speech pipeline."""

import logging
import sys
from pathlib import Path
from typing import Optional
import click

from .pipeline import SpeechPipeline
from .config import Config


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
@click.option('--output', '-o', type=click.Path(path_type=Path), help='Output file path')
@click.option('--format', '-f', 'output_format', 
              type=click.Choice(['srt', 'vtt', 'json']), 
              default='srt', help='Output format')
@click.option('--whisper-model', '-w', 
              type=click.Choice(['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3']),
              default='base', help='Whisper model to use')
@click.option('--min-speakers', type=int, help='Minimum number of speakers')
@click.option('--max-speakers', type=int, help='Maximum number of speakers')
@click.option('--device', type=click.Choice(['cpu', 'cuda', 'mps']), 
              help='Device to run models on')
@click.option('--config', type=click.Path(exists=True, path_type=Path), 
              help='Configuration file path')
def process(
    input_file: Path,
    output: Optional[Path],
    output_format: str,
    whisper_model: str,
    min_speakers: Optional[int],
    max_speakers: Optional[int],
    device: Optional[str],
    config: Optional[Path]
) -> None:
    """Process an audio file with speaker diarization and speech recognition."""
    try:
        # Load configuration
        if config:
            pipeline_config = Config.from_env(str(config))
        else:
            pipeline_config = Config.from_env()
        
        # Override config with command-line options
        pipeline_config.whisper_model = whisper_model
        pipeline_config.output_format = output_format
        
        # Initialize pipeline
        logger.info("Initializing speech pipeline...")
        pipeline = SpeechPipeline(
            config=pipeline_config,
            device=device
        )
        
        # Determine output path
        if output is None:
            output = input_file.with_suffix(f'.{output_format}')
        
        # Process audio
        logger.info(f"Processing {input_file}...")
        result = pipeline.process(
            str(input_file),
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            output_path=str(output),
            output_format=output_format
        )
        
        # Print summary
        click.echo(f"\n✅ Processing completed!")
        click.echo(f"📁 Input: {input_file}")
        click.echo(f"📄 Output: {output}")
        click.echo(f"⏱️  Duration: {result.total_duration:.2f}s")
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
@click.argument('input_file', type=click.Path(exists=True, path_type=Path))
def info(input_file: Path) -> None:
    """Get information about an audio file."""
    try:
        from .audio_utils import AudioProcessor
        
        processor = AudioProcessor()
        
        if not processor.validate_audio_format(str(input_file)):
            click.echo(f"❌ Invalid audio file: {input_file}")
            sys.exit(1)
        
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
    
    click.echo("\n🎭 Pyannote Models:")
    click.echo("  pyannote/speaker-diarization-3.1 - Latest speaker diarization model")
    click.echo("  (Requires Hugging Face token and license acceptance)")


@cli.command()
def setup() -> None:
    """Setup guide for first-time users."""
    click.echo("🚀 Speech Pipeline Setup Guide")
    click.echo("=" * 40)
    
    click.echo("\n1. Install dependencies:")
    click.echo("   uv pip install -e .")
    
    click.echo("\n2. Get Hugging Face token:")
    click.echo("   • Go to https://huggingface.co/")
    click.echo("   • Create account and get access token")
    click.echo("   • Accept license at https://huggingface.co/pyannote/speaker-diarization-3.1")
    
    click.echo("\n3. Create .env file:")
    click.echo("   cp .env.example .env")
    click.echo("   # Edit .env and add your HUGGINGFACE_TOKEN")
    
    click.echo("\n4. Test installation:")
    click.echo("   speech-pipeline models")
    click.echo("   speech-pipeline info your_audio_file.wav")
    
    click.echo("\n5. Process audio:")
    click.echo("   speech-pipeline process input.wav --output output.srt")
    
    click.echo("\n📚 For more info, see README.md")


def main() -> None:
    """Entry point for the CLI."""
    cli()


if __name__ == '__main__':
    main()
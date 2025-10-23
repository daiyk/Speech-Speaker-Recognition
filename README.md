# Speech Recognition and Speaker Diarization Pipeline

A complete pipeline for speech recognition and speaker diarization using Whisper and pyannote, designed to run locally on PC or Mac.

## Features

- **Speaker Diarization**: Uses pyannote.audio to identify and separate different speakers
- **Speech Recognition**: Uses OpenAI Whisper for accurate transcription
- **Multiple Output Formats**: Supports SRT, VTT, and JSON output formats
- **Local Processing**: Runs entirely on your local machine
- **Cross-platform**: Works on both PC and Mac
- **Package Management**: Uses uv for fast and reliable dependency management

## Requirements

- Python 3.9 or higher
- uv package manager
- Sufficient RAM (8GB+ recommended for larger models)
- GPU support optional but recommended for faster processing

## Quick Start
`run_pipeline.sh` provides a convenient and easy-to-follow guided way to do large amount of audio transcription, please edit this `.sh` to fill up required variables (setp 3) and set it executable before running ".run_pipeline.sh":
```
chmod +x run_pipeline.sh
```
## Installation

1. **Install uv** (if not already installed):
   ```bash
   # On macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # On Windows
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. **Clone and setup the project**:
   ```bash
   git clone <repository-url>
   cd SpeechSpeakerRecognition
   
   # Sync dependencies (creates venv automatically)
   uv sync
   
   # Activate the virtual environment
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Configure environment variables**:
   
   Create a `.env` file in the project root:
   ```bash
   # Required
   HUGGINGFACE_TOKEN=your_token_here
   
   # Optional
   WHISPER_MODEL=base                                    # Default: base
   PYANNOTE_MODEL=pyannote/speaker-diarization-3.1      # Default: pyannote/speaker-diarization-3.1
   OUTPUT_FORMAT=srt                                     # Default: srt (options: srt, vtt, json)
   SPEAKER_LABELS=Speaker                                # Default: Speaker
   MIN_SPEAKERS=                                         # Optional: minimum number of speakers
   MAX_SPEAKERS=                                         # Optional: maximum number of speakers
   ```

   **Required Parameters:**
   - `HUGGINGFACE_TOKEN`: Get from [Hugging Face](https://huggingface.co/) → Settings → Access Tokens
   - Accept license at [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)

   **Optional Parameters:**
   - `WHISPER_MODEL`: Model size (tiny, base, small, medium, large, large-v2, large-v3)
   - `PYANNOTE_MODEL`: Diarization model identifier
   - `OUTPUT_FORMAT`: Output file format (srt, vtt, json)
   - `SPEAKER_LABELS`: Prefix for speaker labels in output
   - `MIN_SPEAKERS`: Minimum expected speakers (leave empty for auto-detection)
   - `MAX_SPEAKERS`: Maximum expected speakers (leave empty for auto-detection)

## Usage

### Command Line Interface

Process an audio file:
```bash
uv run speech-pipeline process input.wav --output output.srt
```

With custom settings:
```bash
uv run speech-pipeline process input.wav \
    --output output.srt \
    --whisper-model medium \
    --format srt \
    --min-speakers 2 \
    --max-speakers 5 \
    --device cuda
```

Available options:
- `--output, -o`: Output file path
- `--format, -f`: Output format (srt, vtt, json)
- `--whisper-model, -w`: Whisper model size
- `--min-speakers`: Minimum number of speakers
- `--max-speakers`: Maximum number of speakers
- `--device`: Device to use (cpu, cuda, mps)

Get audio file information:
```bash
uv run speech-pipeline info input.wav
```

List available models:
```bash
uv run speech-pipeline models
```

### Python API

```python
from speech_pipeline import SpeechPipeline

# Initialize pipeline
pipeline = SpeechPipeline(
    whisper_model="base",
    device="cpu"  # or "cuda", "mps"
)

# Process audio file
result = pipeline.process(
    "input.wav",
    min_speakers=2,
    max_speakers=5,
    output_path="output.srt",
    output_format="srt"
)

# Access results
print(f"Duration: {result.total_duration:.2f}s")
print(f"Speakers: {', '.join(result.speakers)}")

# Save in different formats
with open("output.srt", "w") as f:
    f.write(result.to_srt())
    
with open("output.json", "w") as f:
    f.write(result.to_json())
```

## Pipeline Process

1. **Speaker Diarization**: Identify speaker segments using pyannote
2. **Speech Recognition**: Transcribe full audio using Whisper with word timestamps
3. **Merge Results**: Assign speakers to transcribed words and merge into sentences

## Output Formats

- **SRT**: Standard subtitle format with speaker labels
- **VTT**: WebVTT format for web players
- **JSON**: Structured data with detailed timing and confidence scores

## Model Information

### Whisper Models
- `tiny`: Fastest, least accurate (~39 MB)
- `base`: Good balance (~74 MB)
- `small`: Better accuracy (~244 MB)
- `medium`: High accuracy (~769 MB)
- `large`: Best accuracy (~1550 MB)

### pyannote Models
- Uses `pyannote/speaker-diarization-3.1` by default
- Requires Hugging Face token and license acceptance

## Contributing

1. Install development dependencies: `uv sync`
2. Run tests: `uv run pytest`
3. Format code: `uv run black .`
4. Type checking: `uv run mypy speech_pipeline`

## License

MIT License - see LICENSE file for details.

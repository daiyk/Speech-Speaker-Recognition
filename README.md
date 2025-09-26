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

3. **Setup environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env file and add your Hugging Face token
   ```

4. **Get Hugging Face Token**:
   - Go to [Hugging Face](https://huggingface.co/) and create an account
   - Get your access token from Settings > Access Tokens
   - Accept the license for pyannote models at [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)

## Usage

### Command Line Interface

Process an audio file:
```bash
speech-pipeline process input.wav --output output.srt
```

With custom settings:
```bash
speech-pipeline process input.wav \
    --output output.srt \
    --whisper-model medium \
    --format srt \
    --min-speakers 2 \
    --max-speakers 5
```

### Python API

```python
from speech_pipeline import SpeechPipeline

# Initialize pipeline
pipeline = SpeechPipeline(
    whisper_model="base",
    pyannote_model="pyannote/speaker-diarization-3.1"
)

# Process audio file
result = pipeline.process("input.wav")

# Save as SRT
with open("output.srt", "w") as f:
    f.write(result.to_srt())
```

## Pipeline Process

1. **Audio Loading**: Load and preprocess audio file
2. **Speaker Diarization**: Use pyannote to identify speaker segments
3. **Audio Segmentation**: Split audio into speaker-specific segments
4. **Speech Recognition**: Use Whisper to transcribe each segment
5. **Output Generation**: Create timestamped captions with speaker labels

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

1. Install development dependencies: `uv pip install -e ".[dev]"`
2. Run tests: `pytest`
3. Format code: `black .`
4. Type checking: `mypy speech_pipeline`

## License

MIT License - see LICENSE file for details.
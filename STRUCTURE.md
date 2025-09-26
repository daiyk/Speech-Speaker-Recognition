# Project Structure

```
SpeechSpeakerRecognition/
├── speech_pipeline/                 # Main package
│   ├── __init__.py                 # Package initialization
│   ├── config.py                   # Configuration management
│   ├── models.py                   # Data models (SpeakerSegment, TranscriptionResult)
│   ├── audio_utils.py              # Audio processing utilities
│   ├── pipeline.py                 # Main pipeline implementation
│   └── cli.py                      # Command-line interface
├── examples/                       # Example scripts
│   ├── basic_usage.py              # Basic usage examples
│   └── batch_processing.py         # Batch processing example
├── tests/                          # Test files
│   ├── __init__.py
│   └── test_pipeline.py            # Unit tests
├── pyproject.toml                  # Project configuration and dependencies
├── .env.example                    # Environment variables template
├── .env                           # Your environment variables (create from .env.example)
├── .gitignore                      # Git ignore file
├── LICENSE                         # MIT license
├── README.md                       # Comprehensive documentation
├── setup.sh                       # Setup script for Unix/Linux/macOS
├── setup.bat                       # Setup script for Windows
├── quickstart.py                   # Quick test script
└── STRUCTURE.md                    # This file
```

## Key Components

### 1. Configuration System (`config.py`)
- Manages environment variables via `.env` file
- Validates model names and API tokens
- Supports both development and production settings

### 2. Audio Processing (`audio_utils.py`)
- Multi-format audio loading (WAV, MP3, M4A, FLAC, etc.)
- Audio preprocessing for Whisper
- Segment extraction and validation
- Cross-platform compatibility

### 3. Data Models (`models.py`)
- `SpeakerSegment`: Individual speaker segments with timing and text
- `TranscriptionResult`: Complete results with multiple output formats
- Export to SRT, VTT, and JSON formats
- Speaker statistics and analytics

### 4. Main Pipeline (`pipeline.py`)
- Orchestrates the complete workflow:
  1. Audio loading and validation
  2. Speaker diarization with pyannote
  3. Audio segmentation by speaker
  4. Speech recognition with Whisper
  5. Result compilation and export
- Supports GPU acceleration and multiple model sizes
- Error handling and progress tracking

### 5. Command-Line Interface (`cli.py`)
- User-friendly CLI with Click framework
- Multiple commands: `process`, `info`, `models`, `setup`
- Flexible output options and model selection
- Batch processing capabilities

## Workflow

1. **Audio Input**: Load and validate audio file
2. **Speaker Diarization**: Identify "who spoke when" using pyannote
3. **Segmentation**: Split audio into speaker-specific segments
4. **Speech Recognition**: Transcribe each segment using Whisper
5. **Output Generation**: Create timestamped captions with speaker labels

## Output Formats

### SRT (SubRip Subtitle)
```
1
00:00:00,000 --> 00:00:05,000
Speaker 1: Hello, how are you doing today?

2
00:00:05,000 --> 00:00:08,000
Speaker 2: I'm doing great, thanks for asking!
```

### VTT (WebVTT)
```
WEBVTT

00:00:00.000 --> 00:00:05.000
Speaker 1: Hello, how are you doing today?

00:00:05.000 --> 00:00:08.000
Speaker 2: I'm doing great, thanks for asking!
```

### JSON
```json
{
  "total_duration": 120.5,
  "speakers": ["Speaker 1", "Speaker 2"],
  "segments": [
    {
      "start": 0.0,
      "end": 5.0,
      "speaker": "Speaker 1",
      "text": "Hello, how are you doing today?",
      "confidence": 0.95,
      "duration": 5.0
    }
  ]
}
```

## Model Information

### Whisper Models
| Model | Size | Speed | Use Case |
|-------|------|-------|----------|
| tiny | ~39 MB | Fastest | Quick testing, real-time |
| base | ~74 MB | Fast | Good balance |
| small | ~244 MB | Medium | Better accuracy |
| medium | ~769 MB | Slow | High accuracy |
| large | ~1550 MB | Very slow | Best accuracy |

### Pyannote Models
- `pyannote/speaker-diarization-3.1`: State-of-the-art speaker diarization
- Requires Hugging Face account and license acceptance

## Usage Patterns

### Programmatic Usage
```python
from speech_pipeline import SpeechPipeline, Config

config = Config.from_env()
pipeline = SpeechPipeline(config=config)
result = pipeline.process("audio.wav")
```

### Command Line Usage
```bash
# Basic processing
speech-pipeline process audio.wav

# Advanced processing
speech-pipeline process audio.wav \
    --output output.srt \
    --whisper-model medium \
    --min-speakers 2 \
    --max-speakers 4
```

### Batch Processing
```python
# See examples/batch_processing.py for complete implementation
processor = BatchProcessor()
results = processor.process_directory("input_audio/", "output/")
```
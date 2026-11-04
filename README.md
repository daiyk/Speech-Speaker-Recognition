# Speech Recognition and Speaker Diarization Pipeline

A complete pipeline for speech recognition and speaker diarization using Whisper and pyannote, designed to run locally on PC or Mac.

## Features

- **Selectable Pipeline Modes**: Run only speech recognition, speech recognition + timestamps, or the full pipeline with speaker diarization
- **Speaker Diarization**: Uses pyannote.audio to identify and separate different speakers
- **Speech Recognition**: Uses OpenAI Whisper for accurate transcription
- **Multiple Output Formats**: Supports TXT, SRT, VTT, and JSON output formats
- **Local Processing**: Runs entirely on your local machine
- **Cross-platform**: Works on both PC and Mac
- **Package Management**: Uses uv for fast and reliable dependency management

## Requirements

- Python 3.9 or higher
- uv package manager
- Sufficient RAM (8GB+ recommended for larger models)
- GPU support optional but recommended for faster processing

## Pipeline Modes

| Mode | What runs | Output | Needs Hugging Face token? |
|------|-----------|--------|---------------------------|
| `transcribe` | Whisper speech recognition | Plain text (`txt`) | No |
| `timestamps` | Whisper speech recognition with segment timestamps | `srt` (default), `vtt`, `json`, `txt` — no speaker labels | No |
| `diarize` (default) | Whisper word timestamps + pyannote speaker diarization | `srt` (default), `vtt`, `json`, `txt` — with speaker labels | **Yes** |

In `transcribe` and `timestamps` modes the pyannote model is never loaded and `HUGGINGFACE_TOKEN`, `PYANNOTE_MODEL`, `MIN_SPEAKERS`/`MAX_SPEAKERS` are ignored, so the minimal runnable setup is just:

```bash
uv sync
uv run speech-pipeline process input.wav --mode transcribe    # -> input.txt
uv run speech-pipeline process input.wav --mode timestamps    # -> input.srt
```

Select the mode with `--mode` on the command line, or `PIPELINE_MODE` in `.env` / `run_pipeline.sh`.

## Quick Start
`run_pipeline.sh` provides a convenient and easy-to-follow guided way to do large amount of audio transcription.

**Note for Windows Users:** You need to install [Git Bash](https://gitforwindows.org/), and run the script inside Git Bash terminal.

Please first finish Step 1 and Step 2 under Installation, then edit this `.sh` to fill up required variables (step 3, including `PIPELINE_MODE`; `HUGGINGFACE_TOKEN` is only needed for `diarize`) and set it executable before running `./run_pipeline.sh`:
```
chmod +x run_pipeline.sh
```
Then simply run:
```
./run_pipeline.sh
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
   
   Create a `.env` file in the project root (or `cp .env.example .env`). Every setting is optional for `transcribe` and `timestamps` modes:
   ```bash
   PIPELINE_MODE=diarize                                 # Default: diarize (options: transcribe, timestamps, diarize)

   # Required for diarize mode only
   HUGGINGFACE_TOKEN=your_token_here
   
   # Optional
   WHISPER_MODEL=large-v3                                # Default: large-v3
   PYANNOTE_MODEL=pyannote/speaker-diarization-3.1      # Default: pyannote/speaker-diarization-3.1 (diarize only)
   OUTPUT_FORMAT=                                        # Default: txt for transcribe, srt otherwise (options: txt, srt, vtt, json)
   SPEAKER_LABELS=Speaker                                # Default: Speaker (diarize only)
   MIN_SPEAKERS=                                         # Optional: minimum number of speakers (diarize only)
   MAX_SPEAKERS=                                         # Optional: maximum number of speakers (diarize only)
   ```

   **Required Parameters (diarize mode only):**
   - `HUGGINGFACE_TOKEN`: Your Hugging Face access token. See setup instructions below. Skip the Hugging Face setup entirely if you only use `transcribe` or `timestamps`.

   #### Hugging Face Setup Guide

   **1. Get your Access Token:**
   1. Go to [Hugging Face website](https://huggingface.co/).
   2. Login or register an account.
   3. Click the top right corner profile icon -> **Access Tokens** -> **"Create new token"**.
   4. Ensure boxes under **"User Permissions"** -> **"Repositories"** are ticked:
      - "Read access to contents of all repos under your personal namespace"
      - "Read access to contents of all public gated repos you can access"
   5. Click **Create token**, copy paste the token. You only have one chance to see this token!

   **2. Accept Model License:**
   1. Go to the model card: [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1).
   2. Fill up the form "You need to agree to share your contact information to access this model" and submit.
   3. (Optional) You might also need to do the same for [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0).

   **Optional Parameters:**
   - `PIPELINE_MODE`: `transcribe`, `timestamps`, or `diarize` (see [Pipeline Modes](#pipeline-modes))
   - `WHISPER_MODEL`: Model size (tiny, base, small, medium, large, large-v2, large-v3)
   - `PYANNOTE_MODEL`: Diarization model identifier
   - `OUTPUT_FORMAT`: Output file format (txt, srt, vtt, json); `transcribe` only supports txt
   - `SPEAKER_LABELS`: Prefix for speaker labels in output
   - `MIN_SPEAKERS`: Minimum expected speakers (leave empty for auto-detection)
   - `MAX_SPEAKERS`: Maximum expected speakers (leave empty for auto-detection)

## Usage

### Command Line Interface

Process an audio file (full pipeline with speaker diarization):
```bash
uv run speech-pipeline process input.wav --output output.srt
```

Speech recognition only, or speech recognition + timestamps (no Hugging Face token needed):
```bash
uv run speech-pipeline process input.wav --mode transcribe --output output.txt
uv run speech-pipeline process input.wav --mode timestamps --output output.srt
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
- `--mode, -m`: Pipeline mode (transcribe, timestamps, diarize)
- `--output, -o`: Output file path
- `--format, -f`: Output format (txt, srt, vtt, json)
- `--whisper-model, -w`: Whisper model size
- `--min-speakers`: Minimum number of speakers (diarize only)
- `--max-speakers`: Maximum number of speakers (diarize only)
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
    whisper_model="large-v3",
    device="cpu",      # or "cuda", "mps"
    mode="diarize",    # or "transcribe" / "timestamps" (no Hugging Face token needed)
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

`transcribe` / `timestamps` modes:
1. **Speech Recognition**: Transcribe full audio using Whisper (segment timestamps)

`diarize` mode (full pipeline):
1. **Speaker Diarization**: Identify speaker segments using pyannote
2. **Speech Recognition**: Transcribe full audio using Whisper with word timestamps
3. **Merge Results**: Assign speakers to transcribed words and merge into sentences

## Output Formats

- **TXT**: Plain text transcript, one segment per line (speaker-prefixed in diarize mode)
- **SRT**: Standard subtitle format (speaker labels in diarize mode)
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
- Only used in `diarize` mode; requires Hugging Face token and license acceptance

## Contributing

1. Install development dependencies: `uv sync`
2. Run tests: `uv run pytest`
3. Format code: `uv run black .`
4. Type checking: `uv run mypy speech_pipeline`

## License

MIT License - see LICENSE file for details.

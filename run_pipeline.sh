#!/bin/bash
# ==============================================================================
# PLEASE EDIT STEP 3 TO CONFIGURE THE SCRIPT BEFORE RUNNING!!
# ==============================================================================
# Speech-Speaker-Recognition Batch Processor
#
# This script processes supported audio and video files in a specified directory. For each file,
# it splits it into smaller chunks, runs the speech-pipeline on each chunk,
# and then merges the results into a single output:
#   - diarize / timestamps mode: one time-corrected .srt file
#   - transcribe mode:           one plain-text .txt file
# ==============================================================================

set -e

# =====================================================
# 1. Make the script executable
# =====================================================
chmod +x "$0"

# =====================================================
# 2. Setup .env file
# =====================================================
# Create .env from example if it doesn't exist
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    echo "Creating .env from .env.example..."
    cp ".env.example" ".env"
fi

# Load Defaults from .env file
if [ -f ".env" ]; then
    echo "Loading default variables from .env file..."
    set -o allexport
    source ".env"
    set +o allexport
fi

# =====================================================
# 3. USER CONFIGURATION
# =====================================================
# --- EDIT THE VARIABLES BELOW ---
# These variables will override any values set in the .env file.

# 0. Pipeline mode. Can be overridden by the 6th command-line argument.
#    transcribe - speech recognition only (plain text, .txt)
#    timestamps - speech recognition + timestamps (.srt, no speaker labels)
#    diarize    - full pipeline with speaker diarization (.srt, needs HUGGINGFACE_TOKEN)
PIPELINE_MODE="${PIPELINE_MODE:-diarize}"

# 1. (Required for diarize mode only) Your Hugging Face API token, refer to README.md of this project to see the guideline
export HUGGINGFACE_TOKEN="${HUGGINGFACE_TOKEN:-}"

# 2. The absolute path to the Speech-Speaker-Recognition project directory, default to current file's directory.
SRC_DIR="$(cd "$(dirname "$0")" && pwd)"

# 3. (Required) Default input directory containing audio/video files. Can be overridden by the 1st command-line argument.
INPUT_AUDIO_DIR=""

# 4. Default output directory, default to INPUT_AUDIO_DIR/output Can be overridden by the 2nd command-line argument.
OUTPUT_DIR=""

# 5. Minimum number of speakers (diarize mode only). Can be overridden by the 3rd command-line argument.
MIN_SPEAKERS=2

# 6. Maximum number of speakers (diarize mode only). Can be overridden by the 4th command-line argument.
MAX_SPEAKERS=3

# 7. Chunk length in seconds. Can be overridden by the 5th command-line argument.
CHUNK_LENGTH=60

# --- END OF USER CONFIGURATION ---
# =====================================================

export TORCH_AUDIO_BACKEND="soundfile"

# Override config with command-line arguments (highest precedence)
INPUT_AUDIO_DIR=${1:-$INPUT_AUDIO_DIR}
OUTPUT_DIR=${2:-${OUTPUT_DIR:-${INPUT_AUDIO_DIR}/output}}
MIN_SPEAKERS=${3:-$MIN_SPEAKERS}
MAX_SPEAKERS=${4:-$MAX_SPEAKERS}
CHUNK_LENGTH=${5:-$CHUNK_LENGTH}
PIPELINE_MODE=${6:-$PIPELINE_MODE}

# =====================================================
# 4. Validate Configuration
# =====================================================
case "$PIPELINE_MODE" in
    transcribe) OUT_EXT="txt" ;;
    timestamps|diarize) OUT_EXT="srt" ;;
    *)
        echo "ERROR: Invalid PIPELINE_MODE '$PIPELINE_MODE'. Use one of: transcribe, timestamps, diarize."
        exit 1
        ;;
esac
export PIPELINE_MODE

if [ "$PIPELINE_MODE" == "diarize" ]; then
    if [ "$HUGGINGFACE_TOKEN" == "your_token_here" ] || [ "$HUGGINGFACE_TOKEN" == "your_huggingface_token_here" ] || [ -z "$HUGGINGFACE_TOKEN" ]; then
        echo "ERROR: Hugging Face token not set. Edit this script to set the HUGGINGFACE_TOKEN variable."
        echo "You must also accept the pyannote license at: https://huggingface.co/pyannote/speaker-diarization-3.1"
        echo "Or set PIPELINE_MODE to 'transcribe' or 'timestamps' to skip speaker diarization."
        exit 1
    fi
fi

if [ ! -d "$SRC_DIR" ]; then
    echo "ERROR: Project source directory not found at '$SRC_DIR'. Please edit the variable."
    exit 1
fi

if [ -z "$INPUT_AUDIO_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: $0 <path/to/input_directory> <path/to/output_directory> [min_speakers] [max_speakers] [chunk_length_in_seconds] [mode]"
    echo "Alternatively, edit the INPUT_AUDIO_DIR, OUTPUT_DIR, MIN_SPEAKERS, MAX_SPEAKERS, CHUNK_LENGTH, and PIPELINE_MODE variables inside the script."
    exit 1
fi

if [ ! -d "$INPUT_AUDIO_DIR" ]; then
    echo "ERROR: Input directory '$INPUT_AUDIO_DIR' not found."
    exit 1
fi

# Resolve caller-relative paths before changing to the project directory.
INPUT_AUDIO_DIR="$(cd "$INPUT_AUDIO_DIR" && pwd)"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
cd "$SRC_DIR"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv sync

# The Python batch command shares FFmpeg installation and media preparation
# with the CLI. It handles WAV chunks, filenames with spaces, and timing offsets.
uv run speech-pipeline batch "$INPUT_AUDIO_DIR" --output-dir "$OUTPUT_DIR" \
    --mode "$PIPELINE_MODE" --chunk-length "$CHUNK_LENGTH" \
    --min-speakers "$MIN_SPEAKERS" --max-speakers "$MAX_SPEAKERS"

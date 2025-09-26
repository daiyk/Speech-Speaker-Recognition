#!/bin/bash

# Setup script for Speech Recognition and Speaker Diarization Pipeline

set -e

echo "🚀 Setting up Speech Recognition and Speaker Diarization Pipeline"
echo "================================================================="

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv package manager not found!"
    echo "Please install uv first:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "✅ uv package manager found"

# Check Python version
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
required_version="3.9"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python $required_version or higher is required (found: $python_version)"
    exit 1
fi

echo "✅ Python $python_version found"

# Sync dependencies (creates venv automatically and installs all dependencies)
echo "📦 Syncing dependencies with uv..."
uv sync

# Install development dependencies
echo "📦 Installing development dependencies..."
uv sync --group dev

# Check for .env file
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env file and add your Hugging Face token!"
else
    echo "✅ .env file already exists"
fi

# Create directories
echo "📁 Creating necessary directories..."
mkdir -p examples/input_audio
mkdir -p examples/output
mkdir -p inputs
mkdir -p outputs

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Edit .env file and add your Hugging Face token"
echo "   - Get token from: https://huggingface.co/"
echo "   - Accept license at: https://huggingface.co/pyannote/speaker-diarization-3.1"
echo ""
echo "2. Test the installation:"
echo "   source .venv/bin/activate"
echo "   speech-pipeline models"
echo ""
echo "3. Process an audio file:"
echo "   speech-pipeline process your_audio.wav --output output.srt"
echo ""
echo "For more information, see README.md"
@echo off
REM Setup script for Speech Recognition and Speaker Diarization Pipeline

echo 🚀 Setting up Speech Recognition and Speaker Diarization Pipeline
echo =================================================================

REM Check if uv is installed
where uv >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ uv package manager not found!
    echo Please install uv first:
    echo   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    exit /b 1
)

echo ✅ uv package manager found

REM Check Python version
for /f "tokens=*" %%i in ('python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"') do set python_version=%%i
echo ✅ Python %python_version% found

REM Sync dependencies (creates venv automatically and installs all dependencies)
echo 📦 Syncing dependencies with uv...
uv sync

REM Install development dependencies
echo 📦 Installing development dependencies...
uv sync --group dev
if errorlevel 1 exit /b 1

REM Verify/install shared media tools, including ffprobe.
uv run python -m speech_pipeline.ffmpeg_utils
if errorlevel 1 exit /b 1

REM Check for .env file
if not exist ".env" (
    echo 📝 Creating .env file from template...
    copy .env.example .env
    echo ⚠️  Please edit .env file and add your Hugging Face token!
) else (
    echo ✅ .env file already exists
)

REM Create directories
echo 📁 Creating necessary directories...
mkdir examples\input_audio 2>nul
mkdir examples\output 2>nul
mkdir logs 2>nul
mkdir temp 2>nul

echo.
echo 🎉 Setup completed successfully!
echo.
echo Next steps:
echo 1. Edit .env file and add your Hugging Face token
echo    - Get token from: https://huggingface.co/
echo    - Accept license at: https://huggingface.co/pyannote/speaker-diarization-3.1
echo.
echo 2. Test the installation:
echo    .venv\Scripts\activate.bat
echo    speech-pipeline models
echo.
echo 3. Process an audio file:
echo    speech-pipeline process your_audio.wav --output output.srt
echo.
echo For more information, see README.md

pause

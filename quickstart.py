"""Quick start script for testing the pipeline."""

import os
import sys
from pathlib import Path
import tempfile
import logging

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from speech_pipeline import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_requirements():
    """Check if all requirements are installed."""
    try:
        import torch
        import faster_whisper
        import pyannote.audio
        import librosa
        import soundfile
        import pydub
        import numpy as np
        logger.info("✅ All core dependencies are available")
        return True
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        return False


def check_configuration():
    """Check if configuration is valid."""
    try:
        config = Config.from_env()
        config.validate()
        logger.info("✅ Configuration is valid")
        return True
    except Exception as e:
        logger.error(f"❌ Configuration error: {e}")
        return False


def check_models():
    """Check if models can be loaded."""
    try:
        from speech_pipeline import SpeechPipeline
        
        logger.info("🔄 Loading models (this may take a while)...")
        config = Config.from_env()
        config.whisper_model = "tiny"  # Use smallest model for testing
        
        pipeline = SpeechPipeline(config=config)
        logger.info("✅ Models loaded successfully")
        
        # Get model info
        info = pipeline.get_model_info()
        logger.info(f"📊 Whisper model: {info['whisper_model']}")
        logger.info(f"📊 Pyannote model: {info['pyannote_model']}")
        logger.info(f"📊 Device: {info['device']}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Model loading error: {e}")
        return False


def create_test_audio():
    """Create a simple test audio file."""
    try:
        import numpy as np
        import soundfile as sf
        
        # Generate 5 seconds of test audio with two different frequencies
        sample_rate = 16000
        duration = 5
        t = np.linspace(0, duration, sample_rate * duration)
        
        # Create audio with two distinct parts (simulating different speakers)
        audio1 = 0.3 * np.sin(2 * np.pi * 440 * t[:sample_rate * 2])  # 2 seconds at 440Hz
        audio2 = 0.3 * np.sin(2 * np.pi * 880 * t[:sample_rate * 3])  # 3 seconds at 880Hz
        
        test_audio = np.concatenate([audio1, audio2])
        
        test_file = "test_audio.wav"
        sf.write(test_file, test_audio, sample_rate)
        
        logger.info(f"✅ Created test audio file: {test_file}")
        return test_file
    except Exception as e:
        logger.error(f"❌ Failed to create test audio: {e}")
        return None


def run_test_pipeline(audio_file):
    """Run a test with the pipeline."""
    try:
        from speech_pipeline import SpeechPipeline
        
        logger.info("🔄 Running test transcription...")
        
        config = Config.from_env()
        config.whisper_model = "tiny"  # Use smallest model for testing
        
        pipeline = SpeechPipeline(config=config)
        
        result = pipeline.process(
            audio_file,
            output_path="test_output.srt",
            output_format="srt"
        )
        
        logger.info("✅ Test transcription completed")
        logger.info(f"📊 Duration: {result.total_duration:.2f}s")
        logger.info(f"📊 Speakers: {len(result.speakers)}")
        logger.info(f"📊 Segments: {len(result.segments)}")
        
        # Show sample output
        srt_content = result.to_srt()
        logger.info("📄 Sample SRT output:")
        logger.info(srt_content[:200] + "..." if len(srt_content) > 200 else srt_content)
        
        return True
    except Exception as e:
        logger.error(f"❌ Test pipeline failed: {e}")
        return False


def cleanup():
    """Clean up test files."""
    test_files = ["test_audio.wav", "test_output.srt"]
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)
            logger.info(f"🧹 Cleaned up {file}")


def main():
    """Run quick start tests."""
    logger.info("🚀 Speech Pipeline Quick Start Test")
    logger.info("=" * 50)
    
    all_passed = True
    
    # Check 1: Requirements
    logger.info("\n1. Checking requirements...")
    if not check_requirements():
        all_passed = False
        logger.error("Please install dependencies with: uv pip install -e .")
    
    # Check 2: Configuration
    logger.info("\n2. Checking configuration...")
    if not check_configuration():
        all_passed = False
        logger.error("Please set up your .env file with HUGGINGFACE_TOKEN")
    
    if not all_passed:
        logger.error("\n❌ Pre-flight checks failed. Please fix the issues above.")
        return
    
    # Check 3: Model loading (optional for quick test)
    logger.info("\n3. Testing model loading...")
    models_ok = check_models()
    
    # Check 4: Create test audio
    logger.info("\n4. Creating test audio...")
    test_audio = create_test_audio()
    
    if test_audio and models_ok:
        # Check 5: Run pipeline test
        logger.info("\n5. Testing pipeline...")
        run_test_pipeline(test_audio)
    
    # Cleanup
    logger.info("\n6. Cleaning up...")
    cleanup()
    
    if all_passed and models_ok:
        logger.info("\n🎉 All tests passed! Your pipeline is ready to use.")
        logger.info("\nNext steps:")
        logger.info("1. Place your audio files in the project directory")
        logger.info("2. Run: speech-pipeline process your_audio.wav")
        logger.info("3. Check the output files")
    else:
        logger.info("\n⚠️ Some tests failed, but basic setup is working.")
        logger.info("You can still use the pipeline with your own audio files.")


if __name__ == "__main__":
    main()
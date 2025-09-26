"""Example usage of the speech pipeline."""

import logging
from pathlib import Path
from speech_pipeline import SpeechPipeline, Config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Example of using the speech pipeline."""
    
    # Example 1: Basic usage with default settings
    print("Example 1: Basic usage")
    print("-" * 30)
    
    try:
        # Initialize pipeline with default config
        config = Config.from_env()
        pipeline = SpeechPipeline(config=config)
        
        # Process an audio file (replace with your audio file)
        audio_file = "example_audio.wav"  # Replace with actual file
        if Path(audio_file).exists():
            result = pipeline.process(audio_file)
            
            print(f"Processing completed!")
            print(f"Duration: {result.total_duration:.2f}s")
            print(f"Speakers found: {len(result.speakers)}")
            
            # Save as SRT
            with open("output.srt", "w") as f:
                f.write(result.to_srt())
            print("SRT file saved as output.srt")
            
        else:
            print(f"Audio file {audio_file} not found. Please provide a valid audio file.")
            
    except Exception as e:
        print(f"Error in Example 1: {e}")
    
    print("\n")
    
    # Example 2: Custom configuration
    print("Example 2: Custom configuration")
    print("-" * 30)
    
    try:
        # Create custom config
        custom_config = Config(
            whisper_model="small",  # Use more accurate model
            min_speakers=2,
            max_speakers=4,
            output_format="json"
        )
        custom_config.huggingface_token = "your_token_here"  # Set your token
        
        # Initialize pipeline
        pipeline = SpeechPipeline(config=custom_config)
        
        print("Pipeline initialized with custom config:")
        print(f"- Whisper model: {custom_config.whisper_model}")
        print(f"- Speaker range: {custom_config.min_speakers}-{custom_config.max_speakers}")
        print(f"- Output format: {custom_config.output_format}")
        
    except Exception as e:
        print(f"Error in Example 2: {e}")
    
    print("\n")
    
    # Example 3: Processing with specific parameters
    print("Example 3: Processing with parameters")
    print("-" * 30)
    
    try:
        pipeline = SpeechPipeline(whisper_model="base")
        
        audio_file = "example_audio.wav"  # Replace with actual file
        if Path(audio_file).exists():
            # Process with specific speaker constraints
            result = pipeline.process(
                audio_file,
                min_speakers=2,
                max_speakers=5,
                output_path="custom_output.vtt",
                output_format="vtt"
            )
            
            # Print speaker statistics
            stats = result.get_speaker_stats()
            print("Speaker Statistics:")
            for speaker, data in stats.items():
                print(f"  {speaker}: {data['percentage']:.1f}% of total time")
                print(f"    Segments: {data['segments_count']}")
                print(f"    Words: {data['word_count']}")
                print(f"    Avg confidence: {data['average_confidence']:.2f}")
        else:
            print(f"Audio file {audio_file} not found.")
            
    except Exception as e:
        print(f"Error in Example 3: {e}")


if __name__ == "__main__":
    main()
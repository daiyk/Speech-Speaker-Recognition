"""Advanced example with batch processing and custom output."""

import logging
from pathlib import Path
from typing import List, Optional
import json
from concurrent.futures import ThreadPoolExecutor
from speech_pipeline import SpeechPipeline, Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_single_file(pipeline: SpeechPipeline, audio_file: Path, output_dir: Path) -> dict:
    """Process a single audio file and return results."""
    try:
        logger.info(f"Processing {audio_file.name}...")
        
        # Create output path
        output_file = output_dir / f"{audio_file.stem}_transcription.json"
        
        # Process audio
        result = pipeline.process(
            str(audio_file),
            output_path=str(output_file),
            output_format="json"
        )
        
        return {
            "file": audio_file.name,
            "status": "success",
            "duration": result.total_duration,
            "speakers": len(result.speakers),
            "segments": len(result.segments),
            "output": str(output_file)
        }
        
    except Exception as e:
        logger.error(f"Failed to process {audio_file.name}: {e}")
        return {
            "file": audio_file.name,
            "status": "error",
            "error": str(e)
        }


def batch_process_audio_files(
    input_dir: str,
    output_dir: str,
    max_workers: int = 2,
    file_patterns: Optional[List[str]] = None
) -> None:
    """Process multiple audio files in parallel."""
    
    if file_patterns is None:
        file_patterns = ["*.wav", "*.mp3", "*.m4a", "*.flac"]
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find audio files
    audio_files = []
    for pattern in file_patterns:
        audio_files.extend(input_path.glob(pattern))
    
    if not audio_files:
        logger.warning(f"No audio files found in {input_dir}")
        return
    
    logger.info(f"Found {len(audio_files)} audio files to process")
    
    # Initialize pipeline
    config = Config.from_env()
    pipeline = SpeechPipeline(config=config)
    
    # Process files in parallel
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_single_file, pipeline, audio_file, output_path)
            for audio_file in audio_files
        ]
        
        for future in futures:
            results.append(future.result())
    
    # Save batch results
    batch_results = {
        "total_files": len(audio_files),
        "successful": len([r for r in results if r["status"] == "success"]),
        "failed": len([r for r in results if r["status"] == "error"]),
        "results": results
    }
    
    results_file = output_path / "batch_results.json"
    with open(results_file, "w") as f:
        json.dump(batch_results, f, indent=2)
    
    logger.info(f"Batch processing completed. Results saved to {results_file}")
    print(f"Successfully processed: {batch_results['successful']}/{batch_results['total_files']} files")


def create_summary_report(results_file: str, output_file: str) -> None:
    """Create a summary report from batch processing results."""
    
    with open(results_file, "r") as f:
        data = json.load(f)
    
    successful_results = [r for r in data["results"] if r["status"] == "success"]
    
    if not successful_results:
        print("No successful results to summarize")
        return
    
    # Calculate statistics
    total_duration = sum(r["duration"] for r in successful_results)
    total_speakers = sum(r["speakers"] for r in successful_results)
    total_segments = sum(r["segments"] for r in successful_results)
    avg_speakers = total_speakers / len(successful_results)
    avg_duration = total_duration / len(successful_results)
    
    # Create report
    report = f"""# Speech Processing Batch Report

## Summary
- **Total Files Processed**: {data['total_files']}
- **Successful**: {data['successful']}
- **Failed**: {data['failed']}
- **Success Rate**: {(data['successful']/data['total_files']*100):.1f}%

## Audio Statistics
- **Total Audio Duration**: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)
- **Average File Duration**: {avg_duration:.1f} seconds
- **Total Speaker Instances**: {total_speakers}
- **Average Speakers per File**: {avg_speakers:.1f}
- **Total Segments**: {total_segments}

## File Details

### Successful Files
"""
    
    for result in successful_results:
        report += f"- **{result['file']}**: {result['duration']:.1f}s, {result['speakers']} speakers, {result['segments']} segments\n"
    
    failed_results = [r for r in data["results"] if r["status"] == "error"]
    if failed_results:
        report += "\n### Failed Files\n"
        for result in failed_results:
            report += f"- **{result['file']}**: {result['error']}\n"
    
    # Save report
    with open(output_file, "w") as f:
        f.write(report)
    
    print(f"Summary report saved to {output_file}")


def main():
    """Advanced example demonstrating batch processing."""
    
    print("Advanced Speech Pipeline Example")
    print("=" * 40)
    
    # Example: Batch process audio files
    input_directory = "input_audio"  # Directory with audio files
    output_directory = "output_transcriptions"
    
    # Create example directories (in real use, these would contain your audio files)
    Path(input_directory).mkdir(exist_ok=True)
    Path(output_directory).mkdir(exist_ok=True)
    
    print(f"To use this example:")
    print(f"1. Place audio files in '{input_directory}' directory")
    print(f"2. Run this script")
    print(f"3. Check results in '{output_directory}' directory")
    
    # Check if there are audio files to process
    audio_files = list(Path(input_directory).glob("*.wav")) + \
                  list(Path(input_directory).glob("*.mp3")) + \
                  list(Path(input_directory).glob("*.m4a"))
    
    if audio_files:
        print(f"\nFound {len(audio_files)} audio files. Starting batch processing...")
        
        # Process files
        batch_process_audio_files(
            input_directory,
            output_directory,
            max_workers=2  # Adjust based on your system resources
        )
        
        # Create summary report
        results_file = Path(output_directory) / "batch_results.json"
        if results_file.exists():
            create_summary_report(
                str(results_file),
                str(Path(output_directory) / "summary_report.md")
            )
    else:
        print(f"\nNo audio files found in '{input_directory}'")
        print("Please add some audio files and run again.")


if __name__ == "__main__":
    main()
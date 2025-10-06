"""Main speech recognition and speaker diarization pipeline."""

import logging
import math
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


def _mps_available() -> bool:
    """Return True if PyTorch MPS backend is available."""

    return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())


def _check_ffmpeg() -> bool:
    """Check if ffmpeg is available in system PATH."""
    import subprocess
    import sys
    
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
try:
    import torch
    import whisper
    from pyannote.audio import Pipeline as PyannnotePipeline
    from pyannote.core import Annotation, Segment

    # Filter out repetitive deprecation warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    # disable tf32 to suppress pyannote warnings
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
except ImportError as e:
    raise ImportError(
        "Required dependencies not installed. Please run: "
        "uv sync"
    ) from e

from .config import Config
from .models import SpeakerSegment, TranscriptionResult
from .audio_utils import AudioProcessor


logger = logging.getLogger(__name__)


class SpeechPipeline:
    """Complete pipeline for speech recognition and speaker diarization."""
    
    def __init__(
        self,
        config: Optional[Config] = None,
        whisper_model: Optional[str] = None,
        pyannote_model: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the speech pipeline.
        
        Args:
            config: Configuration object
            whisper_model: Whisper model name override
            pyannote_model: Pyannote model name override
            device: Device to run models on ('cpu', 'cuda', 'mps')
        """
        self.config = config or Config.from_env()

        # Override model names if provided
        if whisper_model:
            self.config.whisper_model = whisper_model
        if pyannote_model:
            self.config.pyannote_model = pyannote_model

        # Validate configuration
        self.config.validate()

        # Set device
        if device is None:
            if torch.cuda.is_available():
                resolved_device = "cuda"
            elif _mps_available():
                resolved_device = "mps"
            else:
                resolved_device = "cpu"
        else:
            resolved_device = device.lower()

        if resolved_device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA device requested but torch.cuda.is_available() returned False.")
        if resolved_device == "mps" and not _mps_available():
            raise ValueError("MPS device requested but torch.backends.mps.is_available() returned False.")
        if resolved_device not in {"cpu", "cuda", "mps"}:
            raise ValueError(f"Unsupported device '{resolved_device}'. Choose from 'cpu', 'cuda', or 'mps'.")

        self.device = resolved_device
        
        logger.info(f"Using device: {self.device}")
        
        # Check ffmpeg availability (required for Whisper)
        if not _check_ffmpeg():
            error_msg = (
                "ffmpeg is not installed or not found in system PATH. "
                "Please install ffmpeg to use Whisper for audio transcription.\n"
                "Installation instructions:\n"
                "  - Ubuntu/Debian: sudo apt install ffmpeg\n"
                "  - macOS (Homebrew): brew install ffmpeg\n"
                "  - Windows: Download from https://ffmpeg.org/download.html and add to PATH"
            )
            logger.error(error_msg)
            raise RuntimeError(
                "ffmpeg not found. Please install ffmpeg to use Whisper for audio transcription. "
                "See the instructions above."
            )
        
        logger.info("ffmpeg found and available")
        
        # Initialize components
        self.audio_processor = AudioProcessor(sample_rate=self.config.sample_rate)
        self.whisper_model: Optional[Any] = None
        self.diarization_pipeline: Optional[Any] = None
        
        # Load models
        self._load_models()
    
    def _load_models(self) -> None:
        """Load Whisper and pyannote models."""
        logger.info("Loading models...")
        
        # Load OpenAI Whisper model
        try:
            logger.info(f"Loading Whisper model: {self.config.whisper_model}")
            self.whisper_model = whisper.load_model(
                self.config.whisper_model,
                device=self.device
            )
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load Whisper model: {e}")
        
        # Load pyannote diarization pipeline
        try:
            logger.info(f"Loading pyannote model: {self.config.pyannote_model}")
            self.diarization_pipeline = PyannnotePipeline.from_pretrained(
                self.config.pyannote_model,
                use_auth_token=self.config.huggingface_token
            )
            
            # Move to device if CUDA is available
            if self.diarization_pipeline is not None and hasattr(self.diarization_pipeline, "to"):
                if self.device == "cuda":
                    self.diarization_pipeline = self.diarization_pipeline.to(torch.device("cuda"))
                elif self.device == "mps":
                    self.diarization_pipeline = self.diarization_pipeline.to(torch.device("mps"))
            
            logger.info("Pyannote model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load pyannote model: {e}")
    
    def process(
        self,
        audio_path: str,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        output_path: Optional[str] = None,
        output_format: Optional[str] = None,
        diarization_output_path: Optional[str] = None,
        diarization_output_format: str = "txt",
        whisper_output_path: Optional[str] = None,
        whisper_output_format: str = "txt"
    ) -> TranscriptionResult:
        """
        Process audio file with speaker diarization and speech recognition.
        
        New Logic Flow:
        1. Perform speaker diarization on the full audio
        2. Segment the audio based on diarization results
        3. Transcribe each segment with Whisper
        4. Combine results into final output
        
        Args:
            audio_path: Path to input audio file
            min_speakers: Minimum number of speakers (optional)
            max_speakers: Maximum number of speakers (optional)
            output_path: Path to save transcription output file (optional)
            output_format: Transcription output format ('srt', 'vtt', 'json')
            diarization_output_path: Directory or base path for diarization output (if None, automatically saved next to audio)
            diarization_output_format: 'txt' (default), 'rttm', or 'json'
            whisper_output_path: Directory or base path for raw whisper transcription (if None, automatically saved next to audio)
            whisper_output_format: 'txt' (default), 'json', or 'srt'
            
        Returns:
            TranscriptionResult object
        """
        logger.info(f"Processing audio file: {audio_path}")
        
        # Validate audio file
        if not self.audio_processor.validate_audio_format(audio_path):
            raise ValueError(f"Invalid or unsupported audio file: {audio_path}")
        
        # Load audio
        total_duration = self.audio_processor.get_audio_duration(audio_path)
        
        # Step 1: Perform speaker diarization
        logger.info("Step 1/3: Performing speaker diarization...")
        diarization = self._perform_diarization(
            audio_path, min_speakers, max_speakers
        )

        # Always save diarization (default behavior) unless explicitly disabled by passing empty string
        if diarization_output_path == "":
            logger.info("Diarization saving disabled by empty path")
        else:
            # Derive default path if not provided
            if diarization_output_path is None:
                audio_file = Path(audio_path)
                diarization_output_path = str(audio_file.parent / f"{audio_file.stem}_diarization")
            try:
                logger.info(
                    "Saving diarization to %s (format=%s)",
                    diarization_output_path,
                    diarization_output_format,
                )
                self._save_diarization(
                    diarization,
                    audio_path=audio_path,
                    output_path=diarization_output_path,
                    fmt=diarization_output_format,
                )
            except Exception as e:  # pragma: no cover - non critical
                logger.warning(f"Failed to save diarization output: {e}")
        
        # Step 2: Segment audio based on diarization and transcribe each segment
        logger.info("Step 2/3: Segmenting audio and transcribing each segment...")
        transcribed_segments = self._transcribe_segments_from_diarization(
            audio_path, diarization
        )

        # Save segment-level whisper transcription (default behavior) unless explicitly disabled
        if whisper_output_path == "":
            logger.info("Whisper transcription saving disabled by empty path")
        else:
            # Derive default path if not provided
            if whisper_output_path is None:
                audio_file = Path(audio_path)
                whisper_output_path = str(audio_file.parent / f"{audio_file.stem}_whisper")
            try:
                logger.info(
                    "Saving whisper transcription to %s (format=%s)",
                    whisper_output_path,
                    whisper_output_format,
                )
                self._save_segment_transcriptions(
                    transcribed_segments,
                    output_path=whisper_output_path,
                    fmt=whisper_output_format,
                )
            except Exception as e:  # pragma: no cover - non critical
                logger.warning(f"Failed to save whisper transcription: {e}")
        
        # Step 3: Create result
        logger.info("Step 3/3: Combining results...")
        speakers = list(set(segment.speaker for segment in transcribed_segments))
        speakers.sort()  # Sort for consistent ordering
        
        result = TranscriptionResult(
            segments=transcribed_segments,
            total_duration=total_duration,
            speakers=speakers
        )
        
        # Save output if requested
        if output_path:
            self._save_result(result, output_path, output_format)
        
        logger.info(f"Processing completed. Found {len(speakers)} speakers in {total_duration:.2f}s audio")
        return result
    
    def _perform_diarization(
        self,
        audio_path: str,
        min_speakers: Optional[int],
        max_speakers: Optional[int]
    ) -> Annotation:
        """Perform speaker diarization on audio file."""
        # Set speaker constraints
        diarization_params = {}
        if min_speakers is not None:
            diarization_params["min_speakers"] = min_speakers
        if max_speakers is not None:
            diarization_params["max_speakers"] = max_speakers
        
        # Run diarization
        if self.diarization_pipeline is None:
            raise RuntimeError("Diarization pipeline not loaded")
            
        if diarization_params:
            diarization = self.diarization_pipeline(
                audio_path, **diarization_params
            )
        else:
            diarization = self.diarization_pipeline(audio_path)
        
        return diarization
    
    def _transcribe_segments_from_diarization(
        self,
        audio_path: str,
        diarization: Annotation
    ) -> List[SpeakerSegment]:
        """
        Segment audio based on diarization and transcribe each segment.
        
        Args:
            audio_path: Path to the audio file
            diarization: Diarization annotation with speaker labels
            
        Returns:
            List of SpeakerSegment objects with transcriptions
        """
        if self.whisper_model is None:
            raise RuntimeError("Whisper model not loaded")
        
        # Load the full audio once
        audio_data, sample_rate = self.audio_processor.load_audio(audio_path)
        
        # Extract segments from diarization
        diar_segments = []
        for track in diarization.itertracks(yield_label=True):
            if len(track) == 3:
                segment, _track_id, label = track  # type: ignore[misc]
            else:
                segment, _track_id = track  # type: ignore[misc]
                label = "Unknown"
            
            diar_segments.append({
                "start": float(segment.start),
                "end": float(segment.end),
                "speaker": str(label)
            })
        
        # Sort by start time
        diar_segments.sort(key=lambda x: x["start"])
        
        # Create speaker mapping
        speaker_mapping: Dict[str, str] = {}
        
        # Transcribe each segment
        transcribed_segments: List[SpeakerSegment] = []
        total_segments = len(diar_segments)
        
        for idx, seg in enumerate(diar_segments, 1):
            logger.info(f"Transcribing segment {idx}/{total_segments}: {seg['speaker']} [{seg['start']:.2f}s - {seg['end']:.2f}s]")
            
            # Extract audio segment
            audio_segment = self.audio_processor.extract_segment(
                audio_data,
                seg["start"],
                seg["end"],
                sample_rate
            )
            
            # Skip very short segments (< 0.1 seconds)
            if len(audio_segment) < sample_rate * 0.1:
                logger.debug(f"Skipping very short segment {idx}")
                continue
            
            # Transcribe this segment
            try:
                # Whisper requires audio as float32
                audio_segment = self.audio_processor.preprocess_for_whisper(audio_segment)
                
                result = self.whisper_model.transcribe(
                    audio_segment,
                    language=None,
                    word_timestamps=False,  # Not needed for segment-level transcription
                    verbose=False
                )
                
                text = result.get("text", "").strip()
                
                # Calculate confidence from segments if available
                confidence = None
                segments = result.get("segments", [])
                if segments:
                    logprobs = [s.get("avg_logprob") for s in segments if s.get("avg_logprob") is not None]
                    if logprobs:
                        confidence = sum(math.exp(lp) for lp in logprobs) / len(logprobs)
                
                # Format speaker label
                raw_label = seg["speaker"]
                if raw_label == "Unknown":
                    display_label = "Unknown"
                else:
                    if raw_label not in speaker_mapping:
                        speaker_mapping[raw_label] = f"{self.config.speaker_labels} {len(speaker_mapping) + 1}"
                    display_label = speaker_mapping[raw_label]
                
                transcribed_segments.append(
                    SpeakerSegment(
                        start=seg["start"],
                        end=seg["end"],
                        speaker=display_label,
                        text=text if text else None,
                        confidence=confidence
                    )
                )
                
            except Exception as e:
                logger.warning(f"Failed to transcribe segment {idx}: {e}")
                # Add segment without transcription
                raw_label = seg["speaker"]
                if raw_label == "Unknown":
                    display_label = "Unknown"
                else:
                    if raw_label not in speaker_mapping:
                        speaker_mapping[raw_label] = f"{self.config.speaker_labels} {len(speaker_mapping) + 1}"
                    display_label = speaker_mapping[raw_label]
                
                transcribed_segments.append(
                    SpeakerSegment(
                        start=seg["start"],
                        end=seg["end"],
                        speaker=display_label,
                        text=None,
                        confidence=None
                    )
                )
        
        return transcribed_segments


    
    def _save_result(
        self,
        result: TranscriptionResult,
        output_path: str,
        output_format: Optional[str]
    ) -> None:
        """Save transcription result to file."""
        format_name = output_format or self.config.output_format
        output_file = Path(output_path)
        
        # Ensure correct file extension
        if format_name == "srt" and output_file.suffix != ".srt":
            output_file = output_file.with_suffix(".srt")
        elif format_name == "vtt" and output_file.suffix != ".vtt":
            output_file = output_file.with_suffix(".vtt")
        elif format_name == "json" and output_file.suffix != ".json":
            output_file = output_file.with_suffix(".json")
        
        # Generate content
        if format_name == "srt":
            content = result.to_srt()
        elif format_name == "vtt":
            content = result.to_vtt()
        elif format_name == "json":
            content = result.to_json()
        else:
            raise ValueError(f"Unsupported output format: {format_name}")
        
        # Save to file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(content, encoding="utf-8")
        
        logger.info(f"Results saved to: {output_file}")
    
    def _save_segment_transcriptions(
        self,
        segments: List[SpeakerSegment],
        output_path: str,
        fmt: str = "txt"
    ) -> None:
        """Save segment-level transcriptions.

        Supported formats:
          txt   - Human-readable plain text with timestamps and speakers
          json  - JSON list of segments with timestamps and speakers
          srt   - SubRip format with speaker labels
        """
        fmt = fmt.lower()
        path = Path(output_path)
        if fmt == "txt" and path.suffix.lower() != ".txt":
            path = path.with_suffix(".txt")
        elif fmt == "json" and path.suffix.lower() != ".json":
            path = path.with_suffix(".json")
        elif fmt == "srt" and path.suffix.lower() != ".srt":
            path = path.with_suffix(".srt")

        path.parent.mkdir(parents=True, exist_ok=True)

        def _format_timestamp(seconds: float) -> str:
            """Format seconds to HH:MM:SS.sss."""
            hrs = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            secs = seconds % 60
            return f"{hrs:02}:{mins:02}:{secs:06.3f}"
        
        def _format_srt_timestamp(seconds: float) -> str:
            """Format seconds to SRT timestamp (HH:MM:SS,mmm)."""
            hrs = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            millis = int((seconds % 1) * 1000)
            return f"{hrs:02}:{mins:02}:{secs:02},{millis:03}"

        if fmt == "txt":
            lines = []
            for idx, seg in enumerate(segments, 1):
                start_time = _format_timestamp(seg.start)
                end_time = _format_timestamp(seg.end)
                text = seg.text or "[No speech detected]"
                lines.append(f"{idx}. [{start_time} --> {end_time}] {seg.speaker}")
                lines.append(f"   {text}")
                lines.append("")  # Empty line between segments
            path.write_text("\n".join(lines), encoding="utf-8")
        
        elif fmt == "json":
            import json
            json_segments = []
            for seg in segments:
                json_segments.append({
                    "start": _format_timestamp(seg.start),
                    "end": _format_timestamp(seg.end),
                    "speaker": seg.speaker,
                    "text": seg.text,
                    "confidence": seg.confidence
                })
            path.write_text(json.dumps(json_segments, ensure_ascii=False, indent=2), encoding="utf-8")
        
        elif fmt == "srt":
            lines = []
            for idx, seg in enumerate(segments, 1):
                start_time = _format_srt_timestamp(seg.start)
                end_time = _format_srt_timestamp(seg.end)
                text = seg.text or "[No speech detected]"
                lines.append(str(idx))
                lines.append(f"{start_time} --> {end_time}")
                lines.append(f"[{seg.speaker}] {text}")
                lines.append("")  # Empty line between segments
            path.write_text("\n".join(lines), encoding="utf-8")
        
        else:
            raise ValueError(f"Unsupported whisper output format: {fmt}")

        logger.info(f"Segment transcriptions saved to: {path}")
    
    def _save_diarization(
        self,
        diarization: Annotation,
        audio_path: str,
        output_path: str,
        fmt: str = "rttm"
    ) -> None:
        """Save raw diarization annotation.

        Supported formats:
          rttm  - Standard Rich Transcription Time Marked format
          json  - Simple JSON list of segments {start, end, speaker}
          txt   - Human-readable plain text
        """
        fmt = fmt.lower()
        path = Path(output_path)
        if fmt == "rttm" and path.suffix.lower() != ".rttm":
            path = path.with_suffix(".rttm")
        elif fmt == "json" and path.suffix.lower() != ".json":
            path = path.with_suffix(".json")
        elif fmt == "txt" and path.suffix.lower() != ".txt":
            path = path.with_suffix(".txt")

        path.parent.mkdir(parents=True, exist_ok=True)

        # Build segment list once
        segments: List[Dict[str, Any]] = []
        for track in diarization.itertracks(yield_label=True):
            # itertracks may yield (segment, track_id) or (segment, track_id, label)
            if len(track) == 3:
                segment, _track_id, label = track  # type: ignore[misc]
            else:
                segment, _track_id = track  # type: ignore[misc]
                label = "Unknown"
            segments.append(
                {
                    "start": float(segment.start),
                    "end": float(segment.end),
                    "duration": float(segment.duration),
                    "speaker": str(label),
                }
            )
        
        def _format_timestamp(seconds: float) -> str:
            """Format seconds to HH:MM:SS.sss for TXT output."""
            hrs = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            secs = seconds % 60
            return f"{hrs:02}:{mins:02}:{secs:06.3f}"

        if fmt == "rttm":
            # Compose RTTM lines (SPEAKER <file_id> 1 <start> <duration> <NA> <NA> <speaker> <NA> <NA>)
            file_id = Path(audio_path).stem
            lines = []
            for seg in segments:
                lines.append(
                    f"SPEAKER {file_id} 1 {seg['start']:.3f} {seg['duration']:.3f} <NA> <NA> {seg['speaker']} <NA> <NA>"
                )
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        elif fmt == "json":
            import json
            # also format timestamps for readability
            for seg in segments:
                seg['start'] = _format_timestamp(seg['start'])
                seg['end'] = _format_timestamp(seg['end'])
                seg['duration'] = _format_timestamp(seg['duration'])
                seg['speaker'] = str(seg['speaker'])
            path.write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")
        elif fmt == "txt":
            lines = []
            for seg in segments:
                lines.append(
                    f"[{_format_timestamp(seg['start'])}-{_format_timestamp(seg['end'])}] {seg['speaker']}"
                )
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        else:
            raise ValueError(f"Unsupported diarization output format: {fmt}")

        logger.info(f"Diarization saved to: {path}")
    
    def get_model_info(self) -> dict:
        """Get information about loaded models."""
        return {
            "whisper_model": self.config.whisper_model,
            "pyannote_model": self.config.pyannote_model,
            "device": self.device,
            "sample_rate": self.config.sample_rate
        }
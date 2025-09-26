"""Main speech recognition and speaker diarization pipeline."""

import logging
import tempfile
from pathlib import Path
from typing import List, Optional, Union, Any
import numpy as np

try:
    import torch
    from faster_whisper import WhisperModel
    from pyannote.audio import Pipeline as PyannnotePipeline
    from pyannote.core import Annotation
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
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self.audio_processor = AudioProcessor(sample_rate=self.config.sample_rate)
        self.whisper_model: Optional[Any] = None
        self.diarization_pipeline: Optional[Any] = None
        
        # Load models
        self._load_models()
    
    def _load_models(self) -> None:
        """Load Whisper and pyannote models."""
        logger.info("Loading models...")
        
        # Load Faster Whisper model
        try:
            logger.info(f"Loading Faster Whisper model: {self.config.whisper_model}")
            # faster-whisper uses different device specification
            device = "cuda" if self.device == "cuda" else "cpu"
            # Use int8 for faster inference (2x speed) or float32 for best accuracy
            compute_type = "int8" if device == "cpu" else "float16"
            self.whisper_model = WhisperModel(
                self.config.whisper_model,
                device=device,
                compute_type=compute_type
            )
            logger.info("Faster Whisper model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load Faster Whisper model: {e}")
        
        # Load pyannote diarization pipeline
        try:
            logger.info(f"Loading pyannote model: {self.config.pyannote_model}")
            self.diarization_pipeline = PyannnotePipeline.from_pretrained(
                self.config.pyannote_model,
                use_auth_token=self.config.huggingface_token
            )
            
            # Move to device if CUDA is available
            if self.device == "cuda" and self.diarization_pipeline is not None:
                self.diarization_pipeline = self.diarization_pipeline.to(torch.device("cuda"))
            
            logger.info("Pyannote model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load pyannote model: {e}")
    
    def process(
        self,
        audio_path: str,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        output_path: Optional[str] = None,
        output_format: Optional[str] = None
    ) -> TranscriptionResult:
        """
        Process audio file with speaker diarization and speech recognition.
        
        Args:
            audio_path: Path to input audio file
            min_speakers: Minimum number of speakers (optional)
            max_speakers: Maximum number of speakers (optional)
            output_path: Path to save output file (optional)
            output_format: Output format ('srt', 'vtt', 'json')
            
        Returns:
            TranscriptionResult object
        """
        logger.info(f"Processing audio file: {audio_path}")
        
        # Validate audio file
        if not self.audio_processor.validate_audio_format(audio_path):
            raise ValueError(f"Invalid or unsupported audio file: {audio_path}")
        
        # Load audio
        audio_data, sample_rate = self.audio_processor.load_audio(audio_path)
        total_duration = len(audio_data) / sample_rate
        
        # Perform speaker diarization
        logger.info("Performing speaker diarization...")
        diarization = self._perform_diarization(
            audio_path, min_speakers, max_speakers
        )
        
        # Extract speaker segments
        speaker_segments = self._extract_speaker_segments(
            diarization, audio_data, sample_rate
        )
        
        # Perform speech recognition on each segment
        logger.info("Performing speech recognition...")
        transcribed_segments = self._transcribe_segments(speaker_segments)
        
        # Create result
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
    
    def _extract_speaker_segments(
        self,
        diarization: Annotation,
        audio_data: Union[Any, np.ndarray],
        sample_rate: int
    ) -> List[SpeakerSegment]:
        """Extract speaker segments from diarization result."""
        
        segments = []
        
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            # Format speaker label
            speaker_label = f"{self.config.speaker_labels} {speaker}"
            
            # Extract audio segment
            audio_segment = self.audio_processor.extract_segment(
                audio_data, segment.start, segment.end, sample_rate
            )
            
            segments.append(SpeakerSegment(
                start=segment.start,
                end=segment.end,
                speaker=speaker_label,
                audio_data=audio_segment
            ))
        
        # Sort by start time
        segments.sort(key=lambda x: x.start)
        return segments
    
    def _transcribe_segments(
        self, segments: List[SpeakerSegment]
    ) -> List[SpeakerSegment]:
        """Transcribe each speaker segment using Whisper."""
        transcribed_segments = []
        
        if self.whisper_model is None:
            raise RuntimeError("Whisper model not loaded")
            
        for i, speaker_segment in enumerate(segments):
            logger.debug(f"Transcribing segment {i+1}/{len(segments)}: {speaker_segment.speaker}")
            
            try:
                # Prepare audio for Whisper
                audio_segment = self.audio_processor.preprocess_for_whisper(
                    speaker_segment.audio_data
                )
                
                # Transcribe with Faster Whisper
                whisper_segments, info = self.whisper_model.transcribe(
                    audio_segment,
                    language=None,  # Auto-detect language
                    word_timestamps=True  # Enable word-level timestamps for confidence
                )
                
                # Convert segments generator to list and extract text
                whisper_segment_list = list(whisper_segments)
                if whisper_segment_list:
                    # Combine all segments into single text
                    texts = [ws.text for ws in whisper_segment_list if ws.text is not None]
                    speaker_segment.text = " ".join(texts).strip()
                    
                    # Calculate average confidence from all words
                    all_confidences = []
                    for ws in whisper_segment_list:
                        if hasattr(ws, 'words') and ws.words:
                            for word in ws.words:
                                if hasattr(word, 'probability'):
                                    all_confidences.append(word.probability)
                    
                    if all_confidences:
                        speaker_segment.confidence = sum(all_confidences) / len(all_confidences)
                    else:
                        # Use segment-level confidence if available
                        speaker_segment.confidence = getattr(whisper_segment_list[0], 'avg_logprob', 0.8)
                else:
                    speaker_segment.text = ""
                    speaker_segment.confidence = 0.0
                
                transcribed_segments.append(speaker_segment)
                
            except Exception as e:
                logger.warning(f"Failed to transcribe segment {i+1}: {e}")
                # Add segment without transcription
                speaker_segment.text = "[Transcription failed]"
                speaker_segment.confidence = 0.0
                transcribed_segments.append(speaker_segment)
        
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
    
    def get_model_info(self) -> dict:
        """Get information about loaded models."""
        return {
            "whisper_model": self.config.whisper_model,
            "pyannote_model": self.config.pyannote_model,
            "device": self.device,
            "sample_rate": self.config.sample_rate
        }
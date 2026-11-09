"""Configuration management for the speech pipeline."""

import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from dotenv import load_dotenv


# Pipeline modes, from lightest to heaviest:
#   transcribe - speech recognition only, plain-text transcript
#   timestamps - speech recognition with segment timestamps (subtitles)
#   diarize    - full pipeline: timestamps + speaker diarization (needs Hugging Face)
MODE_TRANSCRIBE = "transcribe"
MODE_TIMESTAMPS = "timestamps"
MODE_DIARIZE = "diarize"
PIPELINE_MODES: Tuple[str, ...] = (MODE_TRANSCRIBE, MODE_TIMESTAMPS, MODE_DIARIZE)
DEFAULT_MODE = MODE_DIARIZE

# Output formats each mode can produce; the first entry is the mode's default.
MODE_OUTPUT_FORMATS: Dict[str, Tuple[str, ...]] = {
    MODE_TRANSCRIBE: ("txt",),
    MODE_TIMESTAMPS: ("srt", "vtt", "json", "txt"),
    MODE_DIARIZE: ("srt", "vtt", "json", "txt"),
}

VALID_WHISPER_MODELS = ("tiny", "base", "small", "medium", "large", "large-v2", "large-v3")


@dataclass
class Config:
    """Configuration settings for the speech pipeline."""

    # Which stages to run (see PIPELINE_MODES)
    mode: str = DEFAULT_MODE

    # API Keys and tokens
    huggingface_token: Optional[str] = None  # only needed in diarize mode
    openai_api_key: Optional[str] = None

    # Model configurations
    whisper_model: str = "large-v3"
    pyannote_model: str = "pyannote/speaker-diarization-3.1"

    # Processing settings
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None

    # Output settings (None = the mode's default format)
    output_format: Optional[str] = None
    speaker_labels: str = "Speaker"

    # Audio processing settings
    sample_rate: int = 16000
    chunk_length: float = 30.0  # seconds

    @classmethod
    def from_env(cls, env_file: Optional[str] = None) -> "Config":
        """Load configuration from environment variables."""
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()
        min_speakers_str = os.getenv("MIN_SPEAKERS")
        min_speakers = int(min_speakers_str) if min_speakers_str else None
        max_speakers_str = os.getenv("MAX_SPEAKERS")
        max_speakers = int(max_speakers_str) if max_speakers_str else None
        return cls(
            mode=(os.getenv("PIPELINE_MODE") or DEFAULT_MODE).strip().lower(),
            huggingface_token=os.getenv("HUGGINGFACE_TOKEN") or None,
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
            whisper_model=os.getenv("WHISPER_MODEL") or "large-v3",
            pyannote_model=os.getenv("PYANNOTE_MODEL") or "pyannote/speaker-diarization-3.1",
            output_format=(os.getenv("OUTPUT_FORMAT") or "").strip().lower() or None,
            speaker_labels=os.getenv("SPEAKER_LABELS") or "Speaker",
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )

    @property
    def diarization_enabled(self) -> bool:
        """True when the pipeline should run speaker diarization."""
        return self.mode == MODE_DIARIZE

    def effective_output_format(self) -> str:
        """Output format to use: the configured one, or the mode's default."""
        if self.output_format:
            return self.output_format
        return MODE_OUTPUT_FORMATS.get(self.mode, ("srt",))[0]

    def validate(self) -> None:
        """Validate configuration settings.

        Only settings needed by the selected mode are checked: the Hugging Face
        token and pyannote settings are ignored unless mode is 'diarize'.
        """
        if self.mode not in PIPELINE_MODES:
            raise ValueError(
                f"Invalid pipeline mode: {self.mode}. "
                f"Valid modes: {', '.join(PIPELINE_MODES)}"
            )

        if self.diarization_enabled and not self.huggingface_token:
            raise ValueError(
                "Hugging Face token is required for speaker diarization. Please set "
                "HUGGINGFACE_TOKEN in your .env file or environment variables, or use "
                f"mode '{MODE_TRANSCRIBE}' or '{MODE_TIMESTAMPS}' to skip diarization."
            )

        if self.whisper_model not in VALID_WHISPER_MODELS:
            raise ValueError(
                f"Invalid Whisper model: {self.whisper_model}. "
                f"Valid models: {', '.join(VALID_WHISPER_MODELS)}"
            )

        valid_formats = MODE_OUTPUT_FORMATS[self.mode]
        output_format = self.effective_output_format()
        if output_format not in valid_formats:
            hint = ""
            if self.mode == MODE_TRANSCRIBE:
                hint = (
                    f" Mode '{MODE_TRANSCRIBE}' produces plain text only; use mode "
                    f"'{MODE_TIMESTAMPS}' for subtitle/JSON output."
                )
            raise ValueError(
                f"Invalid output format '{output_format}' for mode '{self.mode}'. "
                f"Valid formats: {', '.join(valid_formats)}.{hint}"
            )

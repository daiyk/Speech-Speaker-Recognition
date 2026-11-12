"""Tests for the speech pipeline."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np

from speech_pipeline.config import Config
from speech_pipeline.models import SpeakerSegment, TranscriptionResult
from speech_pipeline.audio_utils import AudioProcessor


class TestConfig:
    """Test configuration management."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = Config()
        assert config.mode == "diarize"
        assert config.whisper_model == "large-v3"
        assert config.pyannote_model == "pyannote/speaker-diarization-3.1"
        assert config.output_format is None
        assert config.effective_output_format() == "srt"
        assert config.speaker_labels == "Speaker"
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = Config()
        
        # Should raise error without HF token
        with pytest.raises(ValueError, match="Hugging Face token is required"):
            config.validate()
        
        # Should pass with token
        config.huggingface_token = "test_token"
        config.validate()  # Should not raise
        
        # Should raise error with invalid model
        config.whisper_model = "invalid_model"
        with pytest.raises(ValueError, match="Invalid Whisper model"):
            config.validate()

    @pytest.mark.parametrize("mode", ["transcribe", "timestamps"])
    def test_no_token_needed_without_diarization(self, mode):
        """Modes without diarization must not require a Hugging Face token."""
        config = Config(mode=mode)
        config.validate()  # Should not raise
        assert not config.diarization_enabled

    def test_mode_default_output_formats(self):
        assert Config(mode="transcribe").effective_output_format() == "txt"
        assert Config(mode="timestamps").effective_output_format() == "srt"
        assert Config(mode="diarize").effective_output_format() == "srt"

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="Invalid pipeline mode"):
            Config(mode="everything").validate()

    def test_transcribe_rejects_subtitle_formats(self):
        with pytest.raises(ValueError, match="Invalid output format 'srt' for mode 'transcribe'"):
            Config(mode="transcribe", output_format="srt").validate()

    def test_from_env_mode(self, monkeypatch, tmp_path):
        """PIPELINE_MODE and an empty OUTPUT_FORMAT are read from the env file."""
        env_file = tmp_path / ".env"
        env_file.write_text("PIPELINE_MODE=Timestamps\nOUTPUT_FORMAT=\nMIN_SPEAKERS=\n")
        for key in ("PIPELINE_MODE", "OUTPUT_FORMAT", "MIN_SPEAKERS", "HUGGINGFACE_TOKEN"):
            monkeypatch.delenv(key, raising=False)
        config = Config.from_env(str(env_file))
        assert config.mode == "timestamps"
        assert config.output_format is None
        assert config.min_speakers is None
        config.validate()


class TestSpeakerSegment:
    """Test SpeakerSegment model."""
    
    def test_segment_creation(self):
        """Test creating a speaker segment."""
        segment = SpeakerSegment(
            start=0.0,
            end=5.0,
            speaker="Speaker 1",
            text="Hello world",
            confidence=0.95
        )
        
        assert segment.start == 0.0
        assert segment.end == 5.0
        assert segment.duration == 5.0
        assert segment.speaker == "Speaker 1"
        assert segment.text == "Hello world"
        assert segment.confidence == 0.95
    
    def test_segment_to_dict(self):
        """Test converting segment to dictionary."""
        segment = SpeakerSegment(
            start=1.0,
            end=3.0,
            speaker="Speaker A",
            text="Test text",
            confidence=0.8
        )
        
        expected = {
            "start": 1.0,
            "end": 3.0,
            "speaker": "Speaker A",
            "text": "Test text",
            "confidence": 0.8,
            "duration": 2.0
        }
        
        assert segment.to_dict() == expected


class TestTranscriptionResult:
    """Test TranscriptionResult model."""
    
    @pytest.fixture
    def sample_result(self):
        """Create sample transcription result."""
        segments = [
            SpeakerSegment(0.0, 2.0, "Speaker 1", "Hello there", 0.9),
            SpeakerSegment(2.0, 4.0, "Speaker 2", "How are you?", 0.85),
            SpeakerSegment(4.0, 6.0, "Speaker 1", "I'm doing well", 0.92)
        ]
        
        return TranscriptionResult(
            segments=segments,
            total_duration=6.0,
            speakers=["Speaker 1", "Speaker 2"]
        )
    
    def test_to_srt(self, sample_result):
        """Test SRT format generation."""
        srt_content = sample_result.to_srt()
        
        assert "1" in srt_content
        assert "00:00:00,000 --> 00:00:02,000" in srt_content
        assert "Speaker 1: Hello there" in srt_content
        assert "Speaker 2: How are you?" in srt_content
    
    def test_to_vtt(self, sample_result):
        """Test WebVTT format generation."""
        vtt_content = sample_result.to_vtt()
        
        assert "WEBVTT" in vtt_content
        assert "00:00:00.000 --> 00:00:02.000" in vtt_content
        assert "Speaker 1: Hello there" in vtt_content
    
    def test_to_json(self, sample_result):
        """Test JSON format generation."""
        import json
        
        json_content = sample_result.to_json()
        data = json.loads(json_content)
        
        assert data["total_duration"] == 6.0
        assert len(data["speakers"]) == 2
        assert len(data["segments"]) == 3
        assert data["segments"][0]["text"] == "Hello there"
    
    def test_to_txt(self, sample_result):
        assert sample_result.to_txt() == (
            "Speaker 1: Hello there\nSpeaker 2: How are you?\nSpeaker 1: I'm doing well\n"
        )

    def test_outputs_without_speakers(self):
        """Without diarization, captions carry no speaker prefix."""
        result = TranscriptionResult(
            segments=[
                SpeakerSegment(0.0, 1.5, None, "Hello there"),
                SpeakerSegment(1.5, 3.0, None, "How are you?"),
            ],
            total_duration=3.0,
            speakers=[],
        )
        assert result.to_txt() == "Hello there\nHow are you?\n"
        srt = result.to_srt()
        assert "00:00:00,000 --> 00:00:01,500\nHello there" in srt
        assert "None" not in srt and ":" not in srt.split("\n")[2]
        assert "00:00:01.500 --> 00:00:03.000\nHow are you?" in result.to_vtt()
        assert result.get_speaker_stats() == {}

    def test_speaker_stats(self, sample_result):
        """Test speaker statistics calculation."""
        stats = sample_result.get_speaker_stats()
        
        assert "Speaker 1" in stats
        assert "Speaker 2" in stats
        
        # Speaker 1 has 4 seconds total (segments: 0-2, 4-6)
        assert stats["Speaker 1"]["total_time"] == 4.0
        assert abs(stats["Speaker 1"]["percentage"] - 66.67) < 0.1
        assert stats["Speaker 1"]["segments_count"] == 2
        
        # Speaker 2 has 2 seconds total (segment: 2-4)
        assert stats["Speaker 2"]["total_time"] == 2.0
        assert abs(stats["Speaker 2"]["percentage"] - 33.33) < 0.1
        assert stats["Speaker 2"]["segments_count"] == 1


class TestAudioProcessor:
    """Test audio processing utilities."""
    
    def test_init(self):
        """Test AudioProcessor initialization."""
        processor = AudioProcessor(sample_rate=22050)
        assert processor.sample_rate == 22050
    
    def test_preprocess_for_whisper(self):
        """Test audio preprocessing for Whisper."""
        processor = AudioProcessor()
        
        # Test with normal audio
        audio_data = np.array([0.5, -0.3, 0.8, -0.2], dtype=np.float32)
        processed = processor.preprocess_for_whisper(audio_data)
        
        assert processed.dtype == np.float32
        assert np.max(np.abs(processed)) <= 1.0
    
    def test_extract_segment(self):
        """Test audio segment extraction."""
        processor = AudioProcessor()
        
        # Create 5 seconds of audio at 16kHz
        sample_rate = 16000
        audio_data = np.random.randn(5 * sample_rate).astype(np.float32)
        
        # Extract 1-3 second segment
        segment = processor.extract_segment(audio_data, 1.0, 3.0, sample_rate)
        
        expected_length = 2 * sample_rate  # 2 seconds
        assert len(segment) == expected_length
    
    def test_validate_audio_format(self):
        """Test audio format validation."""
        processor = AudioProcessor()
        
        # Test with non-existent file
        assert not processor.validate_audio_format("nonexistent.wav")
        
        # Test with invalid extension
        with tempfile.NamedTemporaryFile(suffix=".txt") as tmp:
            assert not processor.validate_audio_format(tmp.name)


@pytest.mark.integration
class TestSpeechPipeline:
    """Integration tests for the speech pipeline."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        config = Config()
        config.huggingface_token = "test_token"
        config.whisper_model = "base"
        return config

    @pytest.fixture(autouse=True)
    def no_real_model_download(self):
        """Keep the Windows model download out of tests that run on Windows."""
        with patch('speech_pipeline.pipeline.download_model', return_value="model-dir") as mock_download:
            yield mock_download

    def test_windows_loads_whisper_from_plain_folder(self, no_real_model_download):
        """Windows avoids the symlinked Hugging Face cache for Whisper models."""
        from speech_pipeline.pipeline import SpeechPipeline

        with patch('speech_pipeline.pipeline.sys', Mock(platform="win32")), \
             patch('speech_pipeline.pipeline.WhisperModel') as mock_whisper_cls:
            SpeechPipeline(config=Config(mode="transcribe"), device="cpu")

        args, kwargs = no_real_model_download.call_args
        assert args == ("large-v3",)
        assert Path(kwargs["output_dir"]).parts[-2:] == ("faster-whisper", "large-v3")
        assert mock_whisper_cls.call_args.args == ("model-dir",)

    def test_non_windows_loads_whisper_by_name(self, no_real_model_download):
        """Other platforms keep using the normal Hugging Face cache."""
        from speech_pipeline.pipeline import SpeechPipeline

        with patch('speech_pipeline.pipeline.sys', Mock(platform="darwin")), \
             patch('speech_pipeline.pipeline.WhisperModel') as mock_whisper_cls:
            SpeechPipeline(config=Config(mode="transcribe"), device="cpu")

        no_real_model_download.assert_not_called()
        assert mock_whisper_cls.call_args.args == ("large-v3",)

    @patch('pyannote.audio.Pipeline.from_pretrained')
    @patch('speech_pipeline.pipeline.WhisperModel')
    def test_pipeline_initialization(self, mock_whisper_cls, mock_pyannote_from_pretrained, mock_config):
        """Test pipeline initialization."""
        from speech_pipeline.pipeline import SpeechPipeline
        
        # Mock the models
        mock_whisper_instance = Mock()
        mock_whisper_instance.transcribe.return_value = ([], {})
        mock_whisper_cls.return_value = mock_whisper_instance
        mock_pyannote_from_pretrained.return_value = Mock()
        
        pipeline = SpeechPipeline(config=mock_config)
        
        assert pipeline.config == mock_config
        assert pipeline.whisper_model is not None
        assert pipeline.diarization_pipeline is not None
    
    def test_get_model_info(self, mock_config):
        """Test model information retrieval."""
        with patch('speech_pipeline.pipeline.WhisperModel') as mock_whisper_cls, \
             patch('pyannote.audio.Pipeline.from_pretrained') as mock_pyannote_from_pretrained:
            mock_whisper_instance = Mock()
            mock_whisper_instance.transcribe.return_value = ([], {})
            mock_whisper_cls.return_value = mock_whisper_instance
            mock_pyannote_from_pretrained.return_value = Mock()
            
            from speech_pipeline.pipeline import SpeechPipeline
            pipeline = SpeechPipeline(config=mock_config)
            
            info = pipeline.get_model_info()
            
            assert "whisper_model" in info
            assert "pyannote_model" in info
            assert "device" in info
            assert "sample_rate" in info
            assert info["mode"] == "diarize"

    @staticmethod
    def _fake_whisper():
        """Whisper mock returning two segments (no word timestamps)."""
        from types import SimpleNamespace

        segments = [
            SimpleNamespace(start=0.0, end=1.5, text=" Hello there.", words=None, avg_logprob=-0.1),
            SimpleNamespace(start=1.5, end=2.5, text=" How are you?", words=None, avg_logprob=-0.2),
        ]
        whisper = Mock()
        whisper.transcribe.return_value = (iter(segments), {})
        return whisper

    @staticmethod
    def _write_wav(path: Path) -> Path:
        import soundfile as sf

        sf.write(path, np.zeros(16000 * 3, dtype=np.float32), 16000)
        return path

    @pytest.mark.parametrize(
        "mode,expected_suffix,expected_text",
        [
            ("transcribe", ".txt", "Hello there.\nHow are you?\n"),
            ("timestamps", ".srt", "00:00:00,000 --> 00:00:01,500\nHello there."),
        ],
    )
    def test_modes_without_diarization(self, mode, expected_suffix, expected_text, tmp_path, monkeypatch):
        """transcribe/timestamps never touch pyannote or the Hugging Face token."""
        from speech_pipeline.pipeline import SpeechPipeline

        monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
        config = Config(mode=mode)  # no token at all
        audio = self._write_wav(tmp_path / "in.wav")

        with patch('speech_pipeline.pipeline.WhisperModel', return_value=self._fake_whisper()) as mock_whisper_cls, \
             patch('pyannote.audio.Pipeline.from_pretrained') as mock_pyannote_from_pretrained:
            pipeline = SpeechPipeline(config=config, device="cpu")
            result = pipeline.process(str(audio), min_speakers=2, output_path=str(tmp_path / "out"))

        mock_pyannote_from_pretrained.assert_not_called()
        assert pipeline.diarization_pipeline is None
        assert pipeline.get_model_info()["pyannote_model"] is None
        _, kwargs = mock_whisper_cls.return_value.transcribe.call_args
        assert kwargs["word_timestamps"] is False
        assert kwargs["vad_filter"] is True

        assert result.speakers == []
        assert [s.speaker for s in result.segments] == [None, None]
        assert [s.text for s in result.segments] == ["Hello there.", "How are you?"]

        out_file = (tmp_path / "out").with_suffix(expected_suffix)
        assert expected_text in out_file.read_text()

    def test_clip_processes_only_requested_part(self, tmp_path):
        """--start/--duration feed Whisper a clip and keep the original timeline."""
        import soundfile as sf
        from speech_pipeline.pipeline import SpeechPipeline

        audio = self._write_wav(tmp_path / "in.wav")  # 3 s
        whisper = self._fake_whisper()
        clip_lengths = []
        segments = whisper.transcribe.return_value[0]

        def transcribe(path, **kwargs):
            clip_lengths.append(sf.info(path).duration)
            return segments, {}

        whisper.transcribe.side_effect = transcribe
        with patch('speech_pipeline.pipeline.WhisperModel', return_value=whisper):
            pipeline = SpeechPipeline(config=Config(mode="timestamps"), device="cpu")
            result = pipeline.process(
                str(audio), output_path=str(tmp_path / "out"), output_format="srt",
                start=1.0, duration=1.25,
            )

        assert clip_lengths == [pytest.approx(1.25)]
        assert result.total_duration == pytest.approx(1.25)
        assert [(s.start, s.end) for s in result.segments] == [(1.0, 2.5), (2.5, 3.5)]
        assert "00:00:01,000 --> 00:00:02,500" in (tmp_path / "out.srt").read_text()

    def test_clip_duration_past_end_is_truncated(self, tmp_path):
        from speech_pipeline.media_utils import clipped_audio
        import soundfile as sf

        audio = self._write_wav(tmp_path / "in.wav")  # 3 s
        with clipped_audio(audio, start=2.0, duration=10.0) as clip:
            assert sf.info(str(clip)).duration == pytest.approx(1.0)

    @pytest.mark.parametrize("start,duration,message", [
        (5.0, None, "beyond the end"),
        (-1.0, None, "start"),
        (0.0, 0.0, "duration"),
    ])
    def test_clip_rejects_invalid_ranges(self, tmp_path, start, duration, message):
        from speech_pipeline.media_utils import clipped_audio

        audio = self._write_wav(tmp_path / "in.wav")
        with pytest.raises(ValueError, match=message):
            with clipped_audio(audio, start=start, duration=duration):
                pass

    def test_mode_override_argument(self, mock_config):
        with patch('speech_pipeline.pipeline.WhisperModel'), \
             patch('pyannote.audio.Pipeline.from_pretrained') as mock_pyannote_from_pretrained:
            from speech_pipeline.pipeline import SpeechPipeline
            pipeline = SpeechPipeline(config=mock_config, mode="timestamps", device="cpu")
        assert pipeline.mode == "timestamps"
        mock_pyannote_from_pretrained.assert_not_called()


if __name__ == "__main__":
    # Run tests with: python -m pytest tests/test_pipeline.py -v
    pytest.main([__file__, "-v"])

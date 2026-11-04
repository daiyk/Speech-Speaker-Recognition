"""Media tests use real decoders and tiny fixtures, never downloaded speech models."""

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import soundfile as sf
from click.testing import CliRunner
from pyannote.core import Annotation, Segment

from speech_pipeline import Config, SpeechPipeline
from speech_pipeline import media_utils as media
from speech_pipeline.audio_utils import AudioProcessor
from speech_pipeline.batch import discover_inputs, output_names, process_in_chunks
from speech_pipeline.cli import cli
from speech_pipeline.ffmpeg_utils import FFmpegTools
from speech_pipeline.models import SpeakerSegment, TranscriptionResult


@pytest.fixture
def tools(monkeypatch):
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        pytest.skip("Real media fixtures need installed FFmpeg and FFprobe")
    found = FFmpegTools(ffmpeg, ffprobe)
    monkeypatch.setattr(media, "ensure_ffmpeg", lambda: found)
    return found


def encode(tools, path, args):
    subprocess.run(
        [tools.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
        + args
        + [str(path)],
        check=True,
        capture_output=True,
        timeout=30,
    )
    return path


@pytest.fixture
def video(tools, tmp_path):
    return encode(
        tools,
        tmp_path / "meeting ünicode space.MKV",
        [
            "-f",
            "lavfi",
            "-i",
            "color=black:s=32x32:r=10:d=3",
            "-itsoffset",
            "1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=1",
            "-map",
            "0:v",
            "-map",
            "1:a",
            "-c:v",
            "mpeg4",
            "-c:a",
            "pcm_s16le",
        ],
    )


def pipeline_stub(mode):
    pipeline = SpeechPipeline.__new__(SpeechPipeline)
    pipeline.config = Config(mode=mode)
    pipeline.audio_processor = AudioProcessor()
    pipeline.whisper_model = Mock()
    pipeline.whisper_model.transcribe.return_value = (
        [
            SimpleNamespace(
                start=1.1, end=1.8, text=" Hello.", words=[], avg_logprob=-0.1
            )
        ],
        None,
    )
    annotation = Annotation()
    annotation[Segment(1, 2)] = "speaker"
    pipeline._perform_diarization = Mock(return_value=annotation)
    return pipeline


def test_video_preserves_delayed_audio_and_trailing_silence(video):
    info = media.probe_media(video)
    assert info.duration == pytest.approx(3, abs=0.05)
    assert len(info.audio_streams) == 1
    with media.prepared_audio(video) as prepared:
        samples, sr = sf.read(str(prepared))
        assert sr == 16000 and samples.ndim == 1
        assert len(samples) / sr == pytest.approx(3, abs=0.05)
        assert np.max(np.abs(samples[: int(0.9 * sr)])) < 0.001
        assert np.max(np.abs(samples[int(1.1 * sr) : int(1.9 * sr)])) > 0.05
        assert np.max(np.abs(samples[int(2.1 * sr) :])) < 0.001
    assert not prepared.exists()


@pytest.mark.parametrize("extension", [".mp4", ".mov", ".avi", ".webm"])
def test_containers(tools, tmp_path, extension):
    codecs = (
        ["-c:v", "libvpx", "-c:a", "libopus"]
        if extension == ".webm"
        else ["-c:v", "mpeg4", "-c:a", "aac"]
    )
    source = encode(
        tools,
        tmp_path / ("input" + extension),
        [
            "-f",
            "lavfi",
            "-i",
            "color=black:s=32x32:r=10:d=1",
            "-f",
            "lavfi",
            "-i",
            "sine=duration=1",
            *codecs,
        ],
    )
    with media.prepared_audio(source) as prepared:
        assert sf.info(str(prepared)).frames > 0


def test_first_audio_track_is_selected(tools, tmp_path):
    source = encode(
        tools,
        tmp_path / "tracks.mkv",
        [
            "-f",
            "lavfi",
            "-i",
            "color=black:s=32x32:r=10:d=1",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=16000:cl=mono:d=1",
            "-f",
            "lavfi",
            "-i",
            "sine=duration=1",
            "-map",
            "0:v",
            "-map",
            "1:a",
            "-map",
            "2:a",
            "-c:v",
            "mpeg4",
            "-c:a",
            "pcm_s16le",
            "-disposition:a:0",
            "0",
            "-disposition:a:1",
            "default",
        ],
    )
    assert len(media.probe_media(source).audio_streams) == 2
    with media.prepared_audio(source) as prepared:
        assert np.max(np.abs(sf.read(str(prepared))[0])) == 0


def test_no_audio_fails_before_model_load(tools, tmp_path, monkeypatch):
    source = encode(
        tools,
        tmp_path / "silent.mp4",
        [
            "-f",
            "lavfi",
            "-i",
            "color=black:s=32x32:r=10:d=1",
            "-c:v",
            "mpeg4",
        ],
    )
    load = Mock()
    monkeypatch.setattr(SpeechPipeline, "_load_models", load)
    result = CliRunner().invoke(cli, ["process", str(source), "--mode", "timestamps"])
    assert result.exit_code == 1
    load.assert_not_called()
    with pytest.raises(ValueError, match="No audio track"):
        media.probe_media(source)


def test_corrupt_video(tools, tmp_path):
    source = tmp_path / "broken.mp4"
    source.write_bytes(b"not a video")
    with pytest.raises(ValueError, match="Cannot read media"):
        media.probe_media(source)


def test_wav_does_not_install_ffmpeg(tmp_path, monkeypatch):
    source = tmp_path / "audio.wav"
    sf.write(str(source), np.zeros(16000), 16000)
    ensure = Mock(side_effect=AssertionError("Should not need FFmpeg"))
    monkeypatch.setattr(media, "ensure_ffmpeg", ensure)
    pipeline_stub("timestamps").process(str(source))
    ensure.assert_not_called()


@pytest.mark.parametrize(
    "mode,extension",
    [("transcribe", ".txt"), ("timestamps", ".srt"), ("diarize", ".srt")],
)
def test_all_modes_and_original_export_name(video, mode, extension):
    pipeline = pipeline_stub(mode)
    output = video.with_suffix(extension)
    result = pipeline.process(
        str(video),
        output_path=str(output),
        diarization_output_path=None,
        diarization_output_format="rttm",
    )
    assert result.total_duration == pytest.approx(3, abs=0.05)
    assert result.segments[0].start == pytest.approx(1.1)
    assert "Hello." in output.read_text(encoding="utf-8")
    prepared = Path(pipeline.whisper_model.transcribe.call_args.args[0])
    assert prepared.suffix == ".wav" and not prepared.exists()
    if mode == "diarize":
        assert pipeline._perform_diarization.call_args.args[0] == str(prepared)
        export = video.parent / (video.stem + "_diarization.rttm")
        assert "SPEAKER " + video.stem in export.read_text(encoding="utf-8")
    else:
        pipeline._perform_diarization.assert_not_called()


def test_cleanup_after_inference_failure(video):
    pipeline = pipeline_stub("timestamps")
    pipeline.whisper_model.transcribe.side_effect = RuntimeError("model failed")
    with pytest.raises(RuntimeError, match="model failed"):
        pipeline.process(str(video))
    assert not Path(pipeline.whisper_model.transcribe.call_args.args[0]).exists()


def test_cleanup_after_interruption(video):
    with pytest.raises(KeyboardInterrupt):
        with media.prepared_audio(video) as prepared:
            raise KeyboardInterrupt()
    assert not prepared.exists()


def test_info_video(video):
    result = CliRunner().invoke(cli, ["info", str(video)])
    assert result.exit_code == 0, result.output
    assert "Audio tracks: 1" in result.output
    assert "3.00s" in result.output


def test_batch_exact_offsets_and_final_partial_chunk(video, tmp_path):
    pipeline = pipeline_stub("timestamps")
    paths = []

    def fake_process(path, **kwargs):
        paths.append(Path(path))
        duration = sf.info(path).duration
        return TranscriptionResult(
            [SpeakerSegment(0, duration, None, "Chunk")], duration, []
        )

    pipeline.process = fake_process
    output = tmp_path / "batch.srt"
    result = process_in_chunks(pipeline, video, output, 1.3, "srt")
    assert [s.start for s in result.segments] == pytest.approx([0, 1.3, 2.6])
    assert [s.end for s in result.segments] == pytest.approx([1.3, 2.6, 3])
    assert "00:00:02,600 --> 00:00:03,000" in output.read_text()
    assert all(not p.exists() for p in paths)


def test_batch_discovery_and_output_collisions(tmp_path):
    for name in [
        "meeting.mp3",
        "meeting.mp4",
        "meeting.mp3.wav",
        "hello world.MOV",
        "ignored.txt",
    ]:
        (tmp_path / name).touch()
    (tmp_path / "nested").mkdir()
    inputs = discover_inputs(tmp_path)
    assert len(inputs) == 4
    names = output_names(inputs, "srt")
    assert len({n.casefold() for n in names}) == 4
    assert "hello world.srt" in names


@pytest.mark.parametrize(
    "mode,extension",
    [("transcribe", ".txt"), ("timestamps", ".srt"), ("diarize", ".srt")],
)
def test_cli_video_output_name(video, monkeypatch, mode, extension):
    pipeline = pipeline_stub(mode)
    monkeypatch.setattr(Config, "from_env", classmethod(lambda cls: pipeline.config))
    monkeypatch.setattr(
        "speech_pipeline.pipeline.SpeechPipeline", Mock(return_value=pipeline)
    )
    result = CliRunner().invoke(cli, ["process", str(video), "--mode", mode])
    assert result.exit_code == 0, result.output
    assert "Hello." in video.with_suffix(extension).read_text(encoding="utf-8")


def test_batch_cli_mixed_files(video, tmp_path, tools, monkeypatch):
    sf.write(str(video.with_suffix(".wav")), np.zeros(33600), 16000)
    pipeline = pipeline_stub("timestamps")
    pipeline.whisper_model.transcribe.side_effect = lambda *args, **kwargs: (
        [SimpleNamespace(start=0, end=0.1, text=" Hi.", words=[])],
        None,
    )
    factory = Mock(return_value=pipeline)
    monkeypatch.setattr("speech_pipeline.pipeline.SpeechPipeline", factory)
    monkeypatch.setattr(
        Config, "from_env", classmethod(lambda cls: Config(mode="timestamps"))
    )
    monkeypatch.setattr("speech_pipeline.ffmpeg_utils.ensure_ffmpeg", lambda: tools)
    result = CliRunner().invoke(
        cli,
        [
            "batch",
            str(tmp_path),
            "--mode",
            "timestamps",
            "--chunk-length",
            "1.3",
        ],
    )
    assert result.exit_code == 0, result.output
    factory.assert_called_once()
    outputs = list((tmp_path / "output/final_srt").glob("*.srt"))
    assert len(outputs) == 2
    assert {p.name for p in outputs} == {
        video.name + ".srt",
        video.with_suffix(".wav").name + ".srt",
    }
    assert (
        "00:00:02,600"
        in (tmp_path / "output/final_srt" / (video.name + ".srt")).read_text()
    )


def test_extraction_failure_removes_partial_file(video, monkeypatch):
    monkeypatch.setattr(media, "probe_media", lambda path: media.MediaInfo(3, (1,)))
    outputs = []

    def fail(command, **kwargs):
        path = Path(command[-1])
        path.write_bytes(b"partial output")
        outputs.append(path)
        return SimpleNamespace(returncode=1, stderr="decoder error")

    monkeypatch.setattr(media.subprocess, "run", fail)
    with pytest.raises(ValueError, match="decoder error"):
        with media.prepared_audio(video):
            pytest.fail("Invalid media must not reach inference")
    assert len(outputs) == 1 and not outputs[0].parent.exists()


def test_quickstart_preflight_before_models(tmp_path, monkeypatch):
    import quickstart

    monkeypatch.setattr(
        quickstart, "load_configuration", lambda *args: Config(mode="timestamps")
    )
    monkeypatch.setattr(quickstart, "check_dependencies", lambda **kwargs: True)
    bootstrap = Mock()
    monkeypatch.setattr(quickstart, "bootstrap_pipeline", bootstrap)
    quickstart.main([str(tmp_path / "missing.mp4"), "--mode", "timestamps"])
    bootstrap.assert_not_called()

"""Installer tests never install packages or require administrator privileges."""

import subprocess
from unittest.mock import Mock

import pytest

from speech_pipeline import ffmpeg_utils as ff


@pytest.fixture(autouse=True)
def clear_cache():
    ff.ensure_ffmpeg.cache_clear()
    yield
    ff.ensure_ffmpeg.cache_clear()


def test_existing_installation_is_reused(monkeypatch):
    monkeypatch.setattr(ff, "_find_tool", lambda name: "/tools/" + name)
    install = Mock(side_effect=AssertionError("No installation needed"))
    monkeypatch.setattr(ff, "_install_commands", install)
    assert ff.ensure_ffmpeg().ffprobe == "/tools/ffprobe"
    install.assert_not_called()


def test_missing_ffprobe_installs_and_rechecks(monkeypatch):
    installed = False

    def find(name):
        return "/tools/" + name if installed or name == "ffmpeg" else None

    def run(command, **kwargs):
        nonlocal installed
        installed = True
        return Mock(returncode=0)

    monkeypatch.setattr(ff, "_find_tool", find)
    monkeypatch.setattr(
        ff, "_install_commands", lambda: [["manager", "install", "ffmpeg"]]
    )
    installer = Mock(side_effect=run)
    monkeypatch.setattr(ff.subprocess, "run", installer)
    assert ff.ensure_ffmpeg().ffprobe == "/tools/ffprobe"
    ff.ensure_ffmpeg()
    installer.assert_called_once()


@pytest.mark.parametrize(
    "failure",
    [OSError("network unavailable"), subprocess.TimeoutExpired("install", 600)],
)
def test_install_failure_is_actionable(monkeypatch, failure):
    monkeypatch.setattr(ff, "_find_tool", lambda name: None)
    monkeypatch.setattr(
        ff, "_install_commands", lambda: [["manager", "install", "ffmpeg"]]
    )
    monkeypatch.setattr(ff.subprocess, "run", Mock(side_effect=failure))
    with pytest.raises(RuntimeError, match="Automatic FFmpeg installation failed"):
        ff.ensure_ffmpeg()


def test_permission_failure(monkeypatch):
    monkeypatch.setattr(ff, "_find_tool", lambda name: None)
    monkeypatch.setattr(
        ff, "_install_commands", lambda: [["manager", "install", "ffmpeg"]]
    )
    monkeypatch.setattr(
        ff.subprocess,
        "run",
        Mock(return_value=Mock(returncode=1, stderr="permission denied", stdout="")),
    )
    with pytest.raises(RuntimeError, match="permission denied"):
        ff.ensure_ffmpeg()


def test_installer_success_without_tools_is_not_success(monkeypatch):
    monkeypatch.setattr(ff, "_find_tool", lambda name: None)
    monkeypatch.setattr(
        ff, "_install_commands", lambda: [["manager", "install", "ffmpeg"]]
    )
    monkeypatch.setattr(ff.subprocess, "run", Mock(return_value=Mock(returncode=0)))
    with pytest.raises(RuntimeError, match="could not be verified"):
        ff.ensure_ffmpeg()


@pytest.mark.parametrize(
    "system,manager,expected",
    [
        ("Windows", "winget", "Gyan.FFmpeg"),
        ("Windows", "choco", "ffmpeg"),
        ("Darwin", "brew", "ffmpeg"),
        ("Linux", "apt-get", "ffmpeg"),
        ("Linux", "dnf", "ffmpeg"),
    ],
)
def test_platform_install_commands(monkeypatch, system, manager, expected):
    monkeypatch.setattr(ff.platform, "system", lambda: system)
    monkeypatch.setattr(
        ff.shutil, "which", lambda name: name if name == manager else None
    )
    monkeypatch.setattr(ff.os, "geteuid", lambda: 0, raising=False)
    commands = ff._install_commands()
    assert commands[-1][0] == manager
    assert expected in commands[-1]


def test_missing_package_manager(monkeypatch):
    monkeypatch.setattr(ff.platform, "system", lambda: "Windows")
    monkeypatch.setattr(ff.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="no supported package manager"):
        ff._install_commands()


def test_windows_discovery_without_path_refresh(tmp_path, monkeypatch):
    monkeypatch.setattr(ff.platform, "system", lambda: "Windows")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.delenv("PROGRAMFILES", raising=False)
    monkeypatch.setattr(ff.shutil, "which", lambda name: None)
    bin_dir = tmp_path / "Microsoft/WinGet/Packages/Gyan.FFmpeg_test/ffmpeg-8/bin"
    bin_dir.mkdir(parents=True)
    for name in ("ffmpeg", "ffprobe"):
        (bin_dir / (name + ".exe")).touch()
    monkeypatch.setattr(ff.subprocess, "run", Mock(return_value=Mock(returncode=0)))
    assert ff._find_tool("ffprobe") == str(bin_dir / "ffprobe.exe")

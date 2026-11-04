"""Discover FFmpeg tools and install them once when they are needed."""

import logging
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FFmpegTools:
    ffmpeg: str
    ffprobe: str


def _search_directories() -> List[Path]:
    """Include common install locations even before the shell PATH is refreshed."""
    if platform.system() != "Windows":
        return [Path("/opt/homebrew/bin"), Path("/usr/local/bin"), Path("/usr/bin")]
    directories = []
    for variable in ("LOCALAPPDATA", "PROGRAMFILES"):
        root = os.environ.get(variable)
        if root:
            winget = Path(root) / (
                "Microsoft/WinGet" if variable == "LOCALAPPDATA" else "WinGet"
            )
            directories.append(winget / "Links")
            packages = winget / "Packages"
            if packages.is_dir():
                for package in packages.glob("Gyan.FFmpeg_*"):
                    directories.extend(p.parent for p in package.glob("**/ffmpeg.exe"))
    choco = Path(os.environ.get("ChocolateyInstall", r"C:\ProgramData\chocolatey"))
    directories.append(choco / "bin")
    return directories


def _find_tool(name: str) -> Optional[str]:
    candidates = []
    on_path = shutil.which(name)
    if on_path:
        candidates.append(Path(on_path))
    suffix = ".exe" if platform.system() == "Windows" else ""
    candidates.extend(
        directory / (name + suffix) for directory in _search_directories()
    )
    for candidate in dict.fromkeys(candidates):
        if not candidate.is_file():
            continue
        try:
            result = subprocess.run(
                [str(candidate), "-version"],
                capture_output=True,
                timeout=10,
                stdin=subprocess.DEVNULL,
            )
            if result.returncode == 0:
                return str(candidate.absolute())
        except (OSError, subprocess.TimeoutExpired):
            continue
    return None


def _install_commands() -> List[List[str]]:
    system = platform.system()
    if system == "Windows":
        if shutil.which("winget"):
            return [
                [
                    "winget",
                    "install",
                    "--id",
                    "Gyan.FFmpeg",
                    "--exact",
                    "--source",
                    "winget",
                    "--accept-package-agreements",
                    "--accept-source-agreements",
                    "--disable-interactivity",
                ]
            ]
        if shutil.which("choco"):
            return [["choco", "install", "ffmpeg", "-y", "--no-progress"]]
    elif system == "Darwin":
        brew = shutil.which("brew")
        if not brew:
            brew = next(
                (
                    str(p / "brew")
                    for p in _search_directories()
                    if (p / "brew").is_file()
                ),
                None,
            )
        if brew:
            return [[brew, "install", "ffmpeg"]]
    elif system == "Linux":
        managers = [
            (
                "apt-get",
                [["apt-get", "update"], ["apt-get", "install", "-y", "ffmpeg"]],
            ),
            ("dnf", [["dnf", "install", "-y", "ffmpeg"]]),
            ("yum", [["yum", "install", "-y", "ffmpeg"]]),
            ("pacman", [["pacman", "-S", "--needed", "--noconfirm", "ffmpeg"]]),
            ("apk", [["apk", "add", "ffmpeg"]]),
        ]
        for manager, commands in managers:
            if shutil.which(manager):
                if os.geteuid() != 0:
                    if not shutil.which("sudo"):
                        raise RuntimeError(
                            "FFmpeg installation needs administrator privileges; install ffmpeg with your system package manager."
                        )
                    commands = [["sudo", "-n"] + command for command in commands]
                return commands
    raise RuntimeError(
        "FFmpeg/FFprobe are missing and no supported package manager was found. "
        "Install FFmpeg (including ffprobe) and add its bin directory to PATH. "
        "Automatic installation supports WinGet/Chocolatey, Homebrew, and APT/DNF/YUM/Pacman/APK."
    )


@lru_cache(maxsize=1)
def ensure_ffmpeg() -> FFmpegTools:
    """Reuse working tools, otherwise install and verify without shell interpolation."""
    ffmpeg, ffprobe = _find_tool("ffmpeg"), _find_tool("ffprobe")
    if ffmpeg and ffprobe:
        return FFmpegTools(ffmpeg, ffprobe)
    commands = _install_commands()
    logger.warning(
        "FFmpeg tools are missing; installing FFmpeg with %s...", commands[0][0]
    )
    try:
        for command in commands:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=600,
                stdin=subprocess.DEVNULL,
            )
            if result.returncode:
                # A manager can report 'already installed' while PATH is stale.
                ffmpeg, ffprobe = _find_tool("ffmpeg"), _find_tool("ffprobe")
                if ffmpeg and ffprobe:
                    return FFmpegTools(ffmpeg, ffprobe)
                raise RuntimeError((result.stderr or result.stdout)[-3000:])
    except (OSError, subprocess.TimeoutExpired, RuntimeError) as exc:
        raise RuntimeError(
            "Automatic FFmpeg installation failed. Check network access and package-manager "
            "permissions, or install ffmpeg manually and retry. " + str(exc)
        ) from exc
    ffmpeg, ffprobe = _find_tool("ffmpeg"), _find_tool("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError(
            "Installation finished but FFmpeg/FFprobe could not be verified. Add their bin directory to PATH and retry."
        )
    return FFmpegTools(ffmpeg, ffprobe)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    try:
        tools = ensure_ffmpeg()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"FFmpeg: {tools.ffmpeg}\nFFprobe: {tools.ffprobe}")


if __name__ == "__main__":
    main()

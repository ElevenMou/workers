"""FFmpeg caption burn-in renderer using generated ASS subtitles."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from services.captions.ass_generator import generate_ass_file

_FFMPEG_CAPTION_TIMEOUT_SECONDS = int(
    os.getenv("FFMPEG_CAPTION_TIMEOUT_SECONDS", os.getenv("FFMPEG_TIMEOUT_SECONDS", "1200"))
)


def _normalize_filter_path(path: str) -> str:
    return path.replace("\\", "/").replace(":", r"\:")


def _quote_filter_value(value: str) -> str:
    """Escape a value for use inside an FFmpeg filter expression.

    FFmpeg filter values need single-quote wrapping and escaping of
    the characters ``'``, ``\\``, ``;``, and ``[`` / ``]`` which have
    special meaning in filter graphs.
    """
    escaped = value.replace("\\", "\\\\").replace("'", r"\'")
    escaped = escaped.replace(";", r"\;").replace("[", r"\[").replace("]", r"\]")
    return "'" + escaped + "'"


def resolve_font_dir(
    *,
    explicit_font_dir: str | None = None,
    workspace_root: str | None = None,
) -> str | None:
    candidates = [
        explicit_font_dir,
        os.getenv("CAPTION_FONTS_DIR"),
        str(Path(workspace_root or ".") / "assets" / "fonts"),
        str(Path(workspace_root or ".") / "fonts"),
        "/usr/share/fonts",
        "/usr/local/share/fonts",
        "C:/Windows/Fonts",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        if os.path.isdir(candidate):
            return candidate
    return None


_SAFE_PATH_CHARS = re.compile(r"^[\w\s./:\-]+$")


def _validate_filter_path(path: str, label: str) -> None:
    """Reject paths containing characters that could inject FFmpeg filters."""
    if not _SAFE_PATH_CHARS.match(path):
        raise ValueError(
            f"Unsafe characters in {label} path: {path!r}. "
            "Only alphanumeric, spaces, dots, slashes, colons, and hyphens are allowed."
        )


def _build_ass_filter(ass_path: str, font_dir: str | None) -> str:
    _validate_filter_path(ass_path, "ASS subtitle")
    normalized_ass = _quote_filter_value(_normalize_filter_path(ass_path))
    if not font_dir:
        return f"ass=filename={normalized_ass}"
    _validate_filter_path(font_dir, "font directory")
    normalized_fonts = _quote_filter_value(_normalize_filter_path(font_dir))
    return f"ass=filename={normalized_ass}:fontsdir={normalized_fonts}"


def render_captions(
    video_path: str,
    transcript_json: dict[str, Any],
    preset_name: str,
    output_path: str,
    video_aspect_ratio: str = "9:16",
    *,
    preset_overrides: dict[str, Any] | None = None,
    ffmpeg_bin: str = "ffmpeg",
    use_gpu: bool = False,
    gpu_codec: str = "h264_nvenc",
    font_dir: str | None = None,
    workspace_root: str | None = None,
) -> str:
    """Generate ASS and burn captions into a video file with FFmpeg."""
    temp_dir = tempfile.mkdtemp(prefix="captions_")
    ass_path = str(Path(temp_dir) / "generated.ass")
    try:
        generate_ass_file(
            transcript_json=transcript_json,
            preset_name=preset_name,
            output_path=ass_path,
            video_aspect_ratio=video_aspect_ratio,
            overrides=preset_overrides,
        )

        font_dir_resolved = resolve_font_dir(
            explicit_font_dir=font_dir,
            workspace_root=workspace_root,
        )
        ass_filter = _build_ass_filter(ass_path, font_dir_resolved)

        command: list[str] = [
            ffmpeg_bin,
            "-y",
            "-i",
            video_path,
            "-vf",
            ass_filter,
        ]
        if use_gpu:
            command.extend(["-c:v", gpu_codec, "-preset", "p5"])
        else:
            command.extend(["-c:v", "libx264", "-preset", "medium", "-crf", "18"])
        command.extend(["-c:a", "copy", output_path])

        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=_FFMPEG_CAPTION_TIMEOUT_SECONDS,
        )
        if proc.returncode != 0:
            quoted = " ".join(shlex.quote(part) for part in command)
            stderr = (proc.stderr or "").strip()
            raise RuntimeError(
                "FFmpeg caption render failed\n"
                f"Command: {quoted}\n"
                f"Error: {stderr}"
            )
        return output_path
    finally:
        try:
            if os.path.isfile(ass_path):
                os.remove(ass_path)
            os.rmdir(temp_dir)
        except OSError:
            pass


__all__ = ["render_captions", "resolve_font_dir"]

"""FFmpeg operations used by clip generation."""

import logging
import os
import re
from pathlib import Path

import ffmpeg

from config import FFMPEG_THREADS
from services.clips.constants import (
    TITLE_LINE_HEIGHT_RATIO,
    intermediate_quality_preset,
    normalize_video_scale_mode,
)
from services.clips.models import QualityPreset

logger = logging.getLogger(__name__)
_FFMPEG_THREADS = max(1, int(FFMPEG_THREADS))
_FFMPEG_TIMEOUT_SECONDS = int(os.getenv("FFMPEG_TIMEOUT_SECONDS", "1200"))

_VALID_HEX_COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")
_VALID_COLOR_WITH_OPACITY = re.compile(
    r"^(?:#[0-9a-fA-F]{6}|[a-zA-Z]+)@[0-9.]+$"
)


def _sanitize_color(value: str, fallback: str = "#000000") -> str:
    """Validate that a color value is safe for FFmpeg filters.

    Accepts ``#RRGGBB`` and ``color@opacity`` (e.g. ``black@0.5``) which
    FFmpeg supports natively.  Prevents filter injection via user-controlled
    color parameters.
    """
    if isinstance(value, str):
        token = value.strip()
        if _VALID_HEX_COLOR.match(token) or _VALID_COLOR_WITH_OPACITY.match(token):
            return token
    logger.warning("Invalid color value %r, using fallback %s", value, fallback)
    return fallback


def safe_remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _run_ffmpeg_with_timeout(
    stream,
    *,
    capture_stderr: bool = False,
    quiet: bool = False,
    timeout: int | None = None,
) -> tuple[bytes | None, bytes | None]:
    """Run an ffmpeg stream with a subprocess timeout.

    Falls back to ``_FFMPEG_TIMEOUT_SECONDS`` when *timeout* is ``None``.
    Kills the process tree on timeout to prevent orphaned FFmpeg processes.
    """
    effective_timeout = timeout if timeout is not None else _FFMPEG_TIMEOUT_SECONDS
    import subprocess
    import signal
    import sys

    cmd = stream.compile()
    stderr_pipe = subprocess.PIPE if (capture_stderr or quiet) else None
    stdout_pipe = subprocess.PIPE if quiet else None

    # On POSIX, start a new process group so we can kill all child processes.
    popen_kwargs: dict = {}
    if sys.platform != "win32":
        popen_kwargs["preexec_fn"] = os.setsid

    process = subprocess.Popen(
        cmd, stdout=stdout_pipe, stderr=stderr_pipe, **popen_kwargs
    )
    try:
        stdout, stderr = process.communicate(timeout=effective_timeout)
    except subprocess.TimeoutExpired:
        # Kill the entire process group to prevent orphaned FFmpeg children.
        if sys.platform != "win32":
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except OSError:
                process.kill()
        else:
            process.kill()
        process.wait(timeout=10)
        raise RuntimeError(
            f"FFmpeg process timed out after {effective_timeout}s. "
            "The subprocess and all child processes were killed."
        )
    if process.returncode != 0:
        err_msg = stderr.decode("utf-8", errors="replace") if stderr else ""
        raise ffmpeg.Error("ffmpeg", stdout, stderr)
    return stdout, stderr


def _ffmpeg_thread_args() -> dict[str, int]:
    return {"threads": _FFMPEG_THREADS}


def _probe_media_duration(path: str) -> float:
    try:
        probe = ffmpeg.probe(path)
    except Exception:
        return 0.0

    try:
        return max(0.0, float(probe.get("format", {}).get("duration", 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _media_has_audio(path: str) -> bool:
    try:
        probe = ffmpeg.probe(path)
    except Exception:
        return False

    streams = probe.get("streams", [])
    return any(stream.get("codec_type") == "audio" for stream in streams)


def _resolve_caption_fonts_dir() -> str | None:
    candidates = [
        os.getenv("CAPTION_FONTS_DIR"),
        os.path.join(os.getcwd(), "assets", "fonts"),
        os.path.join(os.getcwd(), "fonts"),
        "/usr/share/fonts",
        "/usr/local/share/fonts",
        "C:/Windows/Fonts",
    ]
    for candidate in candidates:
        if candidate and os.path.isdir(candidate):
            return candidate
    return None


def _normalize_ass_filter_path(path: str) -> str:
    """Normalize filesystem path for FFmpeg ASS filename/fontsdir options."""
    return path.replace("\\", "/")


def _ass_escape(text: str) -> str:
    return (
        text.replace("\\", r"\\")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("\n", r"\N")
    )


def _ass_time(seconds: float) -> str:
    safe = max(0.0, float(seconds))
    total_cs = int(round(safe * 100.0))
    h = total_cs // 360000
    m = (total_cs % 360000) // 6000
    s = (total_cs % 6000) // 100
    cs = total_cs % 100
    return f"{h:02d}:{m:02d}:{s:02d}.{cs:02d}"


def _hex_to_ass_color(value: str, fallback: str) -> str:
    token = (value or "").strip()
    if not token:
        return fallback
    if token.startswith("&H"):
        return token.upper().rstrip("&")
    named = {
        "white": "&H00FFFFFF",
        "black": "&H00000000",
        "red": "&H000000FF",
        "green": "&H0000FF00",
        "blue": "&H00FF0000",
        "yellow": "&H0000FFFF",
    }
    if token.lower() in named:
        return named[token.lower()]
    if token.startswith("#"):
        raw = token.lstrip("#")
        if len(raw) == 6:
            r, g, b = raw[0:2], raw[2:4], raw[4:6]
            return f"&H00{b}{g}{r}".upper()
        if len(raw) == 8:
            a, r, g, b = raw[0:2], raw[2:4], raw[4:6], raw[6:8]
            return f"&H{a}{b}{g}{r}".upper()
    return fallback


def _build_title_ass(
    *,
    title_lines: list[str],
    duration_seconds: float,
    output_path: str,
    canvas_w: int,
    canvas_h: int,
    title_font_size: int,
    title_font_color: str,
    title_font_family: str,
    title_align: str,
    title_stroke_width: int,
    title_stroke_color: str,
    title_padding_x: int,
    title_text_y: int,
    title_area_x: int,
    title_area_w: int,
) -> str | None:
    if not title_lines:
        return None

    font_name = title_font_family or "Montserrat-Bold"
    bold_flag = 0 if "bold" in font_name.lower() else -1
    primary = _hex_to_ass_color(title_font_color, "&H00FFFFFF")
    outline = _hex_to_ass_color(title_stroke_color, "&H00000000")
    line_height = int(title_font_size * TITLE_LINE_HEIGHT_RATIO)

    # Middle alignment matches CSS ``top-1/2 -translate-y-1/2`` centering.
    align_tag = r"\an5" if title_align == "center" else r"\an4"

    area_x = max(0, int(title_area_x))
    area_w = max(2, min(canvas_w - area_x, int(title_area_w)))
    x = area_x + (area_w // 2) if title_align == "center" else area_x + max(0, int(title_padding_x))

    # title_text_y is the vertical center of the title bar.
    # Distribute lines evenly around that center.
    num_lines = len(title_lines)
    total_block_h = (num_lines - 1) * line_height
    first_line_y = title_text_y - total_block_h // 2

    lines: list[str] = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "ScaledBorderAndShadow: yes",
        f"PlayResX: {canvas_w}",
        f"PlayResY: {canvas_h}",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding",
        (
            "Style: Title,"
            f"{font_name},{title_font_size},{primary},{primary},"
            f"{outline},&HFF000000,{bold_flag},0,0,0,100,100,0,0,1,"
            f"{max(0, int(title_stroke_width))},0,5,0,0,0,1"
        ),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    start = _ass_time(0.0)
    end = _ass_time(max(0.1, duration_seconds))
    for idx, line in enumerate(title_lines):
        y = max(0, int(first_line_y + idx * line_height))
        tag = rf"{{{align_tag}\pos({x},{y})}}"
        lines.append(f"Dialogue: 0,{start},{end},Title,,0,0,0,,{tag}{_ass_escape(line)}")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def extract_segment(
    input_path: str,
    start: float,
    end: float,
    output_path: str,
    qp: QualityPreset,
) -> None:
    """Extract a source segment to a temporary clip."""
    start_f = float(start)
    end_f = float(end)
    if end_f <= start_f:
        raise RuntimeError(
            f"Invalid clip time range: start={start_f:.3f}, end={end_f:.3f}"
        )

    safe_start = max(0.0, start_f)
    safe_end = max(safe_start + 0.05, end_f)

    # Clamp to source duration when metadata is available.
    try:
        probe = ffmpeg.probe(input_path)
        source_duration = float(probe.get("format", {}).get("duration", 0.0))
    except Exception:
        source_duration = 0.0

    if source_duration > 0.0:
        if safe_start >= source_duration:
            fallback_start = max(0.0, source_duration - 0.5)
            logger.warning(
                "Clip start %.3fs is outside source duration %.3fs; clamping to %.3fs",
                safe_start,
                source_duration,
                fallback_start,
            )
            safe_start = fallback_start
        if safe_end > source_duration:
            logger.warning(
                "Clip end %.3fs exceeds source duration %.3fs; clamping end",
                safe_end,
                source_duration,
            )
            safe_end = source_duration
        if safe_end <= safe_start:
            safe_end = min(source_duration, safe_start + 0.5)
        if safe_end <= safe_start:
            raise RuntimeError(
                "Cannot extract clip segment after clamping to source duration "
                f"(start={safe_start:.3f}, end={safe_end:.3f}, source={source_duration:.3f})"
            )

    duration = safe_end - safe_start
    logger.info(
        "Extracting segment %.2f-%.2f (%.2fs, quality=%s/%s)",
        safe_start,
        safe_end,
        duration,
        qp.get("preset"),
        qp.get("crf"),
    )

    try:
        stream = (
            ffmpeg.input(input_path, ss=safe_start, t=duration)
            .output(
                output_path,
                vcodec="libx264",
                acodec="aac",
                crf=qp["crf"],
                preset=qp["preset"],
                **_ffmpeg_thread_args(),
                pix_fmt="yuv420p",
                ar=48000,
                audio_bitrate="256k",
                colorspace="bt709",
                color_primaries="bt709",
                color_trc="bt709",
                movflags="+faststart",
            )
            .overwrite_output()
        )
        _run_ffmpeg_with_timeout(stream, capture_stderr=True)
    except ffmpeg.Error as e:
        stderr_output = e.stderr.decode("utf-8", errors="replace") if e.stderr else ""
        logger.error(
            "FFmpeg extract failed for %s (start=%.3f end=%.3f duration=%.3f crf=%s preset=%s):\n%s",
            input_path,
            safe_start,
            safe_end,
            duration,
            qp.get("crf"),
            qp.get("preset"),
            stderr_output,
        )
        raise


def create_portrait_background(
    input_path: str,
    output_path: str,
    style: str,
    canvas_w: int,
    canvas_h: int,
    vid_w: int,
    vid_h: int,
    vid_x: int,
    vid_y: int,
    blur_strength: int,
    video_scale_mode: str,
    qp: QualityPreset,
    *,
    background_color: str = "#000000",
    background_image_path: str | None = None,
) -> None:
    """Create canvas composition with source video over background."""
    logger.info(
        "Creating canvas background (style=%s, %dx%d, blur=%d, color=%s, image=%s, mode=%s)",
        style,
        canvas_w,
        canvas_h,
        blur_strength,
        background_color,
        bool(background_image_path),
        video_scale_mode,
    )
    scale_mode = normalize_video_scale_mode(video_scale_mode)

    def _foreground_video_stream():
        stream = ffmpeg.input(input_path).filter(
            "scale",
            vid_w,
            vid_h,
            force_original_aspect_ratio="increase"
            if scale_mode == "fill"
            else "decrease",
            flags="lanczos",
        )
        if scale_mode == "fill":
            stream = stream.filter("crop", vid_w, vid_h)
        return stream

    if style == "blur":
        bg_temp = output_path + ".bg.mp4"

        (
            ffmpeg.input(input_path)
            .filter("scale", canvas_w, canvas_h, force_original_aspect_ratio="increase", flags="lanczos")
            .filter("crop", canvas_w, canvas_h)
            .filter("boxblur", blur_strength)
            .output(bg_temp, **_ffmpeg_thread_args())
            .overwrite_output()
            .run(quiet=True)
        )

        bg = ffmpeg.input(bg_temp)
        video = _foreground_video_stream()

        (
            ffmpeg.filter([bg, video], "overlay", vid_x, vid_y)
            .output(
                output_path,
                vcodec="libx264",
                acodec="aac",
                crf=qp["crf"],
                preset=qp["preset"],
                **_ffmpeg_thread_args(),
            )
            .overwrite_output()
            .run(quiet=True)
        )

        safe_remove(bg_temp)
        return

    if style == "solid_color":
        probe = ffmpeg.probe(input_path)
        duration = float(probe["format"]["duration"])

        safe_color = _sanitize_color(background_color)
        bg = ffmpeg.input(
            f"color=c={safe_color}:s={canvas_w}x{canvas_h}:d={duration}",
            f="lavfi",
        )
        video = _foreground_video_stream()

        (
            ffmpeg.filter([bg, video], "overlay", vid_x, vid_y)
            .output(
                output_path,
                vcodec="libx264",
                acodec="aac",
                crf=qp["crf"],
                preset=qp["preset"],
                **_ffmpeg_thread_args(),
                shortest=None,
            )
            .overwrite_output()
            .run(quiet=True)
        )
        return

    if style == "image" and background_image_path:
        bg_img_scaled = output_path + ".bgimg.png"

        (
            ffmpeg.input(background_image_path)
            .filter("scale", canvas_w, canvas_h, force_original_aspect_ratio="increase", flags="lanczos")
            .filter("crop", canvas_w, canvas_h)
            .output(bg_img_scaled, vframes=1)
            .overwrite_output()
            .run(quiet=True)
        )

        probe = ffmpeg.probe(input_path)
        duration = float(probe["format"]["duration"])

        bg = ffmpeg.input(bg_img_scaled, loop=1, t=duration, framerate=30)
        video = _foreground_video_stream()

        (
            ffmpeg.filter([bg, video], "overlay", vid_x, vid_y)
            .output(
                output_path,
                vcodec="libx264",
                acodec="aac",
                crf=qp["crf"],
                preset=qp["preset"],
                **_ffmpeg_thread_args(),
                shortest=None,
            )
            .overwrite_output()
            .run(quiet=True)
        )

        safe_remove(bg_img_scaled)
        return

    (
        _foreground_video_stream()
        .filter("pad", canvas_w, canvas_h, vid_x, vid_y, _sanitize_color(background_color))
        .output(
            output_path,
            vcodec="libx264",
            acodec="aac",
            crf=qp["crf"],
            preset=qp["preset"],
            **_ffmpeg_thread_args(),
        )
        .overwrite_output()
        .run(quiet=True)
    )


def add_overlays(
    video_path: str,
    audio_path: str,
    output_path: str,
    *,
    title_lines: list[str],
    title_show: bool,
    title_font_size: int,
    title_font_color: str,
    title_font_family: str,
    title_align: str,
    title_stroke_width: int,
    title_stroke_color: str,
    title_bar_enabled: bool,
    title_bar_color: str,
    title_bar_x: int,
    title_bar_w: int,
    title_padding_x: int,
    title_bar_y: int,
    title_text_y: int,
    title_bar_h: int,
    caption_ass_path: str | None,
    qp: QualityPreset,
    overlay_file_path: str | None = None,
    overlay_cfg: dict | None = None,
) -> None:
    """Draw title/text/captions in a single encode pass."""
    logger.info(
        "Adding overlays (title_show=%s, bar=%s, color=%s, lines=%d, captions=%s)",
        title_show,
        title_bar_enabled,
        title_font_color,
        len(title_lines),
        bool(caption_ass_path),
    )

    video = ffmpeg.input(video_path)
    audio = ffmpeg.input(audio_path)

    stream = video
    temp_files: list[str] = []
    if title_show and title_lines:
        if title_bar_enabled:
            stream = stream.drawbox(
                x=max(0, int(title_bar_x)),
                y=title_bar_y,
                width=max(2, int(title_bar_w)),
                height=title_bar_h,
                color=_sanitize_color(title_bar_color),
                t="fill",
            )
        try:
            probe = ffmpeg.probe(video_path)
            duration_seconds = float(probe["format"]["duration"])
            v_stream = next(
                (s for s in probe.get("streams", []) if s.get("codec_type") == "video"),
                {},
            )
            canvas_w = int(v_stream.get("width", 1080))
            canvas_h = int(v_stream.get("height", 1920))
            title_ass_path = _build_title_ass(
                title_lines=title_lines,
                duration_seconds=duration_seconds,
                output_path=f"{output_path}.title.ass",
                canvas_w=canvas_w,
                canvas_h=canvas_h,
                title_font_size=title_font_size,
                title_font_color=title_font_color,
                title_font_family=title_font_family,
                title_align=title_align,
                title_stroke_width=title_stroke_width,
                title_stroke_color=title_stroke_color,
                title_padding_x=title_padding_x,
                title_text_y=title_text_y,
                title_area_x=title_bar_x,
                title_area_w=title_bar_w,
            )
            if title_ass_path:
                temp_files.append(title_ass_path)
                normalized_title_ass = _normalize_ass_filter_path(title_ass_path)
                fonts_dir = _resolve_caption_fonts_dir()
                if fonts_dir:
                    normalized_fonts = _normalize_ass_filter_path(fonts_dir)
                    stream = stream.filter(
                        "ass",
                        filename=normalized_title_ass,
                        fontsdir=normalized_fonts,
                    )
                else:
                    stream = stream.filter("ass", filename=normalized_title_ass)
        except Exception:
            logger.exception("Failed to build title ASS; continuing without title text overlay")

    if caption_ass_path and os.path.isfile(caption_ass_path):
        normalized_ass_path = _normalize_ass_filter_path(caption_ass_path)
        fonts_dir = _resolve_caption_fonts_dir()
        if fonts_dir:
            normalized_fonts_dir = _normalize_ass_filter_path(fonts_dir)
            stream = stream.filter(
                "ass",
                filename=normalized_ass_path,
                fontsdir=normalized_fonts_dir,
            )
        else:
            stream = stream.filter("ass", filename=normalized_ass_path)

    if (
        overlay_file_path
        and os.path.isfile(overlay_file_path)
        and overlay_cfg
        and overlay_cfg.get("enabled")
    ):
        overlay_x = max(0, int(overlay_cfg.get("x", 0)))
        overlay_y = max(0, int(overlay_cfg.get("y", 0)))
        overlay_w = max(1, min(1920, int(overlay_cfg.get("widthPx", 200))))
        overlay_input = ffmpeg.input(overlay_file_path).filter("scale", overlay_w, -1)
        stream = ffmpeg.filter([stream, overlay_input], "overlay", overlay_x, overlay_y)

    try:
        overlay_stream = (
            ffmpeg.output(
                stream,
                audio.audio,
                output_path,
                vcodec="libx264",
                acodec="aac",
                crf=qp["crf"],
                preset=qp["preset"],
                **_ffmpeg_thread_args(),
                pix_fmt="yuv420p",
                audio_bitrate="256k",
                colorspace="bt709",
                color_primaries="bt709",
                color_trc="bt709",
                movflags="+faststart",
            )
            .overwrite_output()
        )
        _run_ffmpeg_with_timeout(overlay_stream, capture_stderr=True)
    except ffmpeg.Error as e:
        stderr_output = e.stderr.decode("utf-8", errors="replace") if e.stderr else ""
        logger.error("FFmpeg overlay failed:\n%s", stderr_output)
        raise
    finally:
        for temp_file in temp_files:
            safe_remove(temp_file)


def compose_clip(
    input_path: str,
    output_path: str,
    *,
    style: str,
    canvas_w: int,
    canvas_h: int,
    vid_w: int,
    vid_h: int,
    vid_x: int,
    vid_y: int,
    blur_strength: int,
    video_scale_mode: str,
    title_lines: list[str],
    title_show: bool,
    title_font_size: int,
    title_font_color: str,
    title_font_family: str,
    title_align: str,
    title_stroke_width: int,
    title_stroke_color: str,
    title_bar_enabled: bool,
    title_bar_color: str,
    title_bar_x: int,
    title_bar_w: int,
    title_padding_x: int,
    title_bar_y: int,
    title_text_y: int,
    title_bar_h: int,
    caption_ass_path: str | None,
    qp: QualityPreset,
    overlay_file_path: str | None = None,
    overlay_cfg: dict | None = None,
    background_color: str = "#000000",
    background_image_path: str | None = None,
    start_time: float | None = None,
    end_time: float | None = None,
) -> None:
    """Compose background, source video, title/captions, and overlay in one pass.

    When *start_time* and *end_time* are provided the source is seeked directly,
    eliminating the need for a separate ``extract_segment()`` call and avoiding
    an extra re-encoding pass.
    """
    logger.info(
        "Compositing clip (style=%s, overlay=%s, captions=%s, title=%s, seek=%s)",
        style,
        bool(overlay_file_path),
        bool(caption_ass_path),
        bool(title_show and title_lines),
        f"{start_time:.2f}-{end_time:.2f}" if start_time is not None else "none",
    )

    if start_time is not None and end_time is not None:
        duration_seconds = max(0.1, end_time - start_time)
        source = ffmpeg.input(input_path, ss=start_time, t=duration_seconds)
    else:
        duration_seconds = max(0.1, _probe_media_duration(input_path))
        source = ffmpeg.input(input_path)

    scale_mode = normalize_video_scale_mode(video_scale_mode)

    foreground = source.video.filter(
        "scale",
        vid_w,
        vid_h,
        force_original_aspect_ratio="increase" if scale_mode == "fill" else "decrease",
        flags="lanczos",
    )
    if scale_mode == "fill":
        foreground = foreground.filter("crop", vid_w, vid_h)

    if style == "blur":
        background = (
            source.video.filter(
                "scale", canvas_w, canvas_h,
                force_original_aspect_ratio="increase",
                flags="lanczos",
            )
            .filter("crop", canvas_w, canvas_h)
            .filter("boxblur", blur_strength)
        )
        stream = ffmpeg.filter([background, foreground], "overlay", vid_x, vid_y)
    elif style == "solid_color":
        safe_color = _sanitize_color(background_color)
        background = ffmpeg.input(
            f"color=c={safe_color}:s={canvas_w}x{canvas_h}:d={duration_seconds}",
            f="lavfi",
        ).video
        stream = ffmpeg.filter([background, foreground], "overlay", vid_x, vid_y)
    elif style == "image" and background_image_path:
        background = (
            ffmpeg.input(
                background_image_path,
                loop=1,
                t=duration_seconds,
                framerate=30,
            )
            .video.filter(
                "scale", canvas_w, canvas_h,
                force_original_aspect_ratio="increase",
                flags="lanczos",
            )
            .filter("crop", canvas_w, canvas_h)
        )
        stream = ffmpeg.filter([background, foreground], "overlay", vid_x, vid_y)
    else:
        stream = foreground.filter(
            "pad",
            canvas_w,
            canvas_h,
            vid_x,
            vid_y,
            _sanitize_color(background_color),
        )

    temp_files: list[str] = []
    if title_show and title_lines:
        if title_bar_enabled:
            stream = stream.drawbox(
                x=max(0, int(title_bar_x)),
                y=title_bar_y,
                width=max(2, int(title_bar_w)),
                height=title_bar_h,
                color=_sanitize_color(title_bar_color),
                t="fill",
            )
        try:
            title_ass_path = _build_title_ass(
                title_lines=title_lines,
                duration_seconds=duration_seconds,
                output_path=f"{output_path}.title.ass",
                canvas_w=canvas_w,
                canvas_h=canvas_h,
                title_font_size=title_font_size,
                title_font_color=title_font_color,
                title_font_family=title_font_family,
                title_align=title_align,
                title_stroke_width=title_stroke_width,
                title_stroke_color=title_stroke_color,
                title_padding_x=title_padding_x,
                title_text_y=title_text_y,
                title_area_x=title_bar_x,
                title_area_w=title_bar_w,
            )
            if title_ass_path:
                temp_files.append(title_ass_path)
                normalized_title_ass = _normalize_ass_filter_path(title_ass_path)
                fonts_dir = _resolve_caption_fonts_dir()
                if fonts_dir:
                    normalized_fonts = _normalize_ass_filter_path(fonts_dir)
                    stream = stream.filter(
                        "ass",
                        filename=normalized_title_ass,
                        fontsdir=normalized_fonts,
                    )
                else:
                    stream = stream.filter("ass", filename=normalized_title_ass)
        except Exception:
            logger.exception("Failed to build title ASS; continuing without title text overlay")

    if caption_ass_path and os.path.isfile(caption_ass_path):
        normalized_ass_path = _normalize_ass_filter_path(caption_ass_path)
        fonts_dir = _resolve_caption_fonts_dir()
        if fonts_dir:
            normalized_fonts_dir = _normalize_ass_filter_path(fonts_dir)
            stream = stream.filter(
                "ass",
                filename=normalized_ass_path,
                fontsdir=normalized_fonts_dir,
            )
        else:
            stream = stream.filter("ass", filename=normalized_ass_path)

    if (
        overlay_file_path
        and os.path.isfile(overlay_file_path)
        and overlay_cfg
        and overlay_cfg.get("enabled")
    ):
        overlay_x = max(0, int(overlay_cfg.get("x", 0)))
        overlay_y = max(0, int(overlay_cfg.get("y", 0)))
        overlay_w = max(1, min(1920, int(overlay_cfg.get("widthPx", 200))))
        overlay_input = ffmpeg.input(overlay_file_path).filter("scale", overlay_w, -1)
        stream = ffmpeg.filter([stream, overlay_input], "overlay", overlay_x, overlay_y)

    audio_stream = source.audio if _media_has_audio(input_path) else None
    if audio_stream is None:
        audio_stream = ffmpeg.input(
            "anullsrc=r=48000:cl=stereo",
            f="lavfi",
            t=duration_seconds,
        ).audio
    audio_stream = audio_stream.filter("aresample", 48000).filter(
        "aformat",
        sample_rates=48000,
        channel_layouts="stereo",
    )

    try:
        compose_stream = (
            ffmpeg.output(
                stream,
                audio_stream,
                output_path,
                vcodec="libx264",
                acodec="aac",
                crf=qp["crf"],
                preset=qp["preset"],
                **_ffmpeg_thread_args(),
                pix_fmt="yuv420p",
                audio_bitrate="256k",
                colorspace="bt709",
                color_primaries="bt709",
                color_trc="bt709",
                movflags="+faststart",
            )
            .overwrite_output()
        )
        _run_ffmpeg_with_timeout(compose_stream, capture_stderr=True)
    except ffmpeg.Error as e:
        stderr_output = e.stderr.decode("utf-8", errors="replace") if e.stderr else ""
        logger.error("FFmpeg compose failed:\n%s", stderr_output)
        raise
    finally:
        for temp_file in temp_files:
            safe_remove(temp_file)


def _prepare_intro_outro_segment(
    file_path: str,
    cfg: dict,
    canvas_w: int,
    canvas_h: int,
    output_path: str,
    qp: QualityPreset,
) -> str | None:
    """Convert an intro/outro source (image or video) to a normalised clip segment.

    Returns the path to the normalised segment on success, ``None`` on failure.
    """
    media_type = cfg.get("type", "image")
    duration = max(0.5, min(60.0, float(cfg.get("durationSeconds", 3.0))))
    try:
        if media_type == "image":
            video_stream = (
                ffmpeg.input(file_path, loop=1, t=duration, framerate=30)
                .filter("scale", canvas_w, canvas_h, force_original_aspect_ratio="decrease", flags="lanczos")
                .filter("pad", canvas_w, canvas_h, "(ow-iw)/2", "(oh-ih)/2", color="black")
                .filter("setsar", "1")
                .filter("fps", 30)
                .filter("setpts", "PTS-STARTPTS")
            )
            audio_stream = ffmpeg.input(
                "anullsrc=r=48000:cl=stereo",
                f="lavfi",
                t=duration,
            ).audio
        else:
            # Video type: trim to configured duration before scaling/padding.
            segment_input = ffmpeg.input(file_path, ss=0, t=duration)
            video_stream = (
                segment_input.video
                .filter("scale", canvas_w, canvas_h, force_original_aspect_ratio="decrease", flags="lanczos")
                .filter("pad", canvas_w, canvas_h, "(ow-iw)/2", "(oh-ih)/2", color="black")
                .filter("setsar", "1")
                .filter("fps", 30)
                .filter("setpts", "PTS-STARTPTS")
            )

            if _media_has_audio(file_path):
                audio_stream = (
                    segment_input.audio.filter("atrim", duration=duration)
                    .filter("asetpts", "PTS-STARTPTS")
                    .filter("aresample", 48000)
                    .filter("aformat", sample_rates=48000, channel_layouts="stereo")
                )
            else:
                audio_stream = ffmpeg.input(
                    "anullsrc=r=48000:cl=stereo",
                    f="lavfi",
                    t=duration,
                ).audio

        (
            ffmpeg.output(
                video_stream,
                audio_stream,
                output_path,
                vcodec="libx264",
                acodec="aac",
                crf=qp["crf"],
                preset=qp["preset"],
                **_ffmpeg_thread_args(),
                pix_fmt="yuv420p",
                audio_bitrate="256k",
                colorspace="bt709",
                color_primaries="bt709",
                color_trc="bt709",
                movflags="+faststart",
                shortest=None,
            )
            .overwrite_output()
            .run(quiet=True)
        )
        return output_path
    except Exception:
        logger.exception("Failed to prepare intro/outro segment from %s", file_path)
        safe_remove(output_path)
        return None


def concat_intro_outro(
    main_clip_path: str,
    output_path: str,
    qp: QualityPreset,
    *,
    intro_file_path: str | None = None,
    intro_cfg: dict | None = None,
    outro_file_path: str | None = None,
    outro_cfg: dict | None = None,
    canvas_w: int,
    canvas_h: int,
) -> list[str]:
    """Prepend intro and/or append outro to main clip using FFmpeg concat filter.

    Returns a list of temporary intermediate files created (caller should clean up).
    """
    intermediates: list[str] = []
    segments: list[str] = []

    segment_qp = intermediate_quality_preset(qp)

    # Prepare intro segment.
    if intro_file_path and intro_cfg and intro_cfg.get("enabled"):
        intro_norm = main_clip_path + ".intro_norm.mp4"
        result = _prepare_intro_outro_segment(
            intro_file_path, intro_cfg, canvas_w, canvas_h, intro_norm, segment_qp,
        )
        if result:
            segments.append(result)
            intermediates.append(intro_norm)

    segments.append(main_clip_path)

    # Prepare outro segment.
    if outro_file_path and outro_cfg and outro_cfg.get("enabled"):
        outro_norm = main_clip_path + ".outro_norm.mp4"
        result = _prepare_intro_outro_segment(
            outro_file_path, outro_cfg, canvas_w, canvas_h, outro_norm, segment_qp,
        )
        if result:
            segments.append(result)
            intermediates.append(outro_norm)

    if len(segments) <= 1:
        return intermediates

    logger.info(
        "Concatenating %d segments (intro=%s, outro=%s)",
        len(segments),
        bool(intro_file_path),
        bool(outro_file_path),
    )

    try:
        concat_inputs: list = []
        for segment_path in segments:
            segment_input = ffmpeg.input(segment_path)
            segment_duration = max(0.1, _probe_media_duration(segment_path))

            segment_video = (
                segment_input.video
                .filter("fps", 30)
                .filter(
                    "scale",
                    canvas_w,
                    canvas_h,
                    force_original_aspect_ratio="decrease",
                    flags="lanczos",
                )
                .filter("pad", canvas_w, canvas_h, "(ow-iw)/2", "(oh-ih)/2", color="black")
                .filter("setsar", "1")
                .filter("trim", duration=segment_duration)
                .filter("setpts", "PTS-STARTPTS")
            )

            if _media_has_audio(segment_path):
                segment_audio = (
                    segment_input.audio.filter("atrim", duration=segment_duration)
                    .filter("asetpts", "PTS-STARTPTS")
                    .filter("aresample", 48000)
                    .filter("aformat", sample_rates=48000, channel_layouts="stereo")
                )
            else:
                segment_audio = ffmpeg.input(
                    "anullsrc=r=48000:cl=stereo",
                    f="lavfi",
                    t=segment_duration,
                ).audio

            concat_inputs.extend([segment_video, segment_audio])

        concat_node = ffmpeg.concat(*concat_inputs, v=1, a=1).node
        concat_video = concat_node[0]
        concat_audio = concat_node[1]
        concat_stream = (
            ffmpeg.output(
                concat_video,
                concat_audio,
                output_path,
                vcodec="libx264",
                acodec="aac",
                crf=qp["crf"],
                preset=qp["preset"],
                **_ffmpeg_thread_args(),
                pix_fmt="yuv420p",
                audio_bitrate="256k",
                colorspace="bt709",
                color_primaries="bt709",
                color_trc="bt709",
                movflags="+faststart",
            )
            .overwrite_output()
        )
        _run_ffmpeg_with_timeout(concat_stream, capture_stderr=True)
    except ffmpeg.Error as e:
        stderr_output = e.stderr.decode("utf-8", errors="replace") if e.stderr else ""
        logger.error("FFmpeg concat failed:\n%s", stderr_output)
        raise

    return intermediates


def generate_thumbnail(
    video_path: str,
    output_path: str,
    *,
    width: int = 720,
    quality: int = 2,
) -> None:
    """Generate a high-quality thumbnail image from the clip.

    Seeks to ~10 % of the clip duration to avoid black frames and title cards.
    Scales to a consistent width with lanczos and applies JPEG quality control.
    """
    duration = _probe_media_duration(video_path)
    seek = max(0.5, duration * 0.1) if duration > 2.0 else 0.0

    try:
        (
            ffmpeg.input(video_path, ss=seek)
            .filter("scale", width, -1, flags="lanczos")
            .output(output_path, vframes=1, **{"q:v": quality})
            .overwrite_output()
            .run(capture_stderr=True)
        )
    except ffmpeg.Error as e:
        stderr_output = e.stderr.decode("utf-8", errors="replace") if e.stderr else ""
        logger.error("Thumbnail generation failed for %s:\n%s", video_path, stderr_output)
        raise

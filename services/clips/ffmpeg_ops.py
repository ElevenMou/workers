"""FFmpeg operations used by clip generation."""

import logging
import os
import re
from pathlib import Path

import ffmpeg

from services.clips.constants import TITLE_LINE_HEIGHT_RATIO, normalize_video_scale_mode
from services.clips.models import QualityPreset

logger = logging.getLogger(__name__)

_VALID_HEX_COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")


def _sanitize_color(value: str, fallback: str = "#000000") -> str:
    """Validate that a color value is a safe hex color string.

    Prevents FFmpeg filter injection via user-controlled color parameters.
    """
    if isinstance(value, str) and _VALID_HEX_COLOR.match(value):
        return value
    logger.warning("Invalid color value %r, using fallback %s", value, fallback)
    return fallback


def safe_remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


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
    primary = _hex_to_ass_color(title_font_color, "&H00FFFFFF")
    outline = _hex_to_ass_color(title_stroke_color, "&H00000000")
    line_height = int(title_font_size * TITLE_LINE_HEIGHT_RATIO)
    align_tag = r"\an8" if title_align == "center" else r"\an7"
    area_x = max(0, int(title_area_x))
    area_w = max(2, min(canvas_w - area_x, int(title_area_w)))
    x = area_x + (area_w // 2) if title_align == "center" else area_x + max(0, int(title_padding_x))

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
            f"{outline},&H80000000,-1,0,0,0,100,100,0,0,1,"
            f"{max(0, int(title_stroke_width))},0,7,0,0,0,1"
        ),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    start = _ass_time(0.0)
    end = _ass_time(max(0.1, duration_seconds))
    for idx, line in enumerate(title_lines):
        y = max(0, int(title_text_y + (idx * line_height)))
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
    duration = end - start
    logger.info("Extracting segment %.2f-%.2f (%.2fs)", start, end, duration)

    (
        ffmpeg.input(input_path, ss=start, t=duration)
        .output(
            output_path,
            vcodec="libx264",
            acodec="aac",
            crf=qp["crf"],
            preset=qp["preset"],
        )
        .overwrite_output()
        .run(quiet=True)
    )


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
        )
        if scale_mode == "fill":
            stream = stream.filter("crop", vid_w, vid_h)
        return stream

    if style == "blur":
        bg_temp = output_path + ".bg.mp4"

        (
            ffmpeg.input(input_path)
            .filter("scale", canvas_w, canvas_h, force_original_aspect_ratio="increase")
            .filter("crop", canvas_w, canvas_h)
            .filter("boxblur", blur_strength)
            .output(bg_temp)
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
            .filter("scale", canvas_w, canvas_h, force_original_aspect_ratio="increase")
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
        (
            ffmpeg.output(
                stream,
                audio.audio,
                output_path,
                vcodec="libx264",
                acodec="aac",
                crf=qp["crf"],
                preset=qp["preset"],
                audio_bitrate="192k",
                movflags="+faststart",
            )
            .overwrite_output()
            .run(capture_stderr=True)
        )
    except ffmpeg.Error as e:
        stderr_output = e.stderr.decode("utf-8", errors="replace") if e.stderr else ""
        logger.error("FFmpeg overlay failed:\n%s", stderr_output)
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
    try:
        if media_type == "image":
            duration = max(0.5, min(60.0, float(cfg.get("durationSeconds", 3.0))))
            img = (
                ffmpeg.input(file_path, loop=1, t=duration, framerate=30)
                .filter("scale", canvas_w, canvas_h, force_original_aspect_ratio="decrease")
                .filter("pad", canvas_w, canvas_h, "(ow-iw)/2", "(oh-ih)/2", color="black")
            )
            silent = ffmpeg.input(
                f"anullsrc=r=44100:cl=stereo",
                f="lavfi",
                t=duration,
            )
            (
                ffmpeg.output(
                    img,
                    silent,
                    output_path,
                    vcodec="libx264",
                    acodec="aac",
                    crf=qp["crf"],
                    preset=qp["preset"],
                    pix_fmt="yuv420p",
                    shortest=None,
                )
                .overwrite_output()
                .run(quiet=True)
            )
        else:
            # Video type — scale/pad to canvas size.
            video = (
                ffmpeg.input(file_path)
                .filter("scale", canvas_w, canvas_h, force_original_aspect_ratio="decrease")
                .filter("pad", canvas_w, canvas_h, "(ow-iw)/2", "(oh-ih)/2", color="black")
            )
            audio = ffmpeg.input(file_path).audio
            (
                ffmpeg.output(
                    video,
                    audio,
                    output_path,
                    vcodec="libx264",
                    acodec="aac",
                    crf=qp["crf"],
                    preset=qp["preset"],
                    pix_fmt="yuv420p",
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
    """Prepend intro and/or append outro to main clip using FFmpeg concat demuxer.

    Returns a list of temporary intermediate files created (caller should clean up).
    """
    import tempfile

    intermediates: list[str] = []
    segments: list[str] = []

    # Prepare intro segment.
    if intro_file_path and intro_cfg and intro_cfg.get("enabled"):
        intro_norm = main_clip_path + ".intro_norm.mp4"
        result = _prepare_intro_outro_segment(
            intro_file_path, intro_cfg, canvas_w, canvas_h, intro_norm, qp,
        )
        if result:
            segments.append(result)
            intermediates.append(intro_norm)

    segments.append(main_clip_path)

    # Prepare outro segment.
    if outro_file_path and outro_cfg and outro_cfg.get("enabled"):
        outro_norm = main_clip_path + ".outro_norm.mp4"
        result = _prepare_intro_outro_segment(
            outro_file_path, outro_cfg, canvas_w, canvas_h, outro_norm, qp,
        )
        if result:
            segments.append(result)
            intermediates.append(outro_norm)

    if len(segments) <= 1:
        # Nothing to concat — main clip is the only segment.
        return intermediates

    logger.info("Concatenating %d segments (intro=%s, outro=%s)", len(segments),
                bool(intro_file_path), bool(outro_file_path))

    # Write a concat list file.
    concat_list_path = main_clip_path + ".concat.txt"
    intermediates.append(concat_list_path)
    with open(concat_list_path, "w", encoding="utf-8") as f:
        for seg in segments:
            safe_path = seg.replace("\\", "/").replace("'", "'\\''")
            f.write(f"file '{safe_path}'\n")

    try:
        (
            ffmpeg.input(concat_list_path, f="concat", safe=0)
            .output(
                output_path,
                vcodec="libx264",
                acodec="aac",
                crf=qp["crf"],
                preset=qp["preset"],
                movflags="+faststart",
            )
            .overwrite_output()
            .run(capture_stderr=True)
        )
    except ffmpeg.Error as e:
        stderr_output = e.stderr.decode("utf-8", errors="replace") if e.stderr else ""
        logger.error("FFmpeg concat failed:\n%s", stderr_output)
        raise

    return intermediates


def generate_thumbnail(video_path: str, output_path: str) -> None:
    """Generate a thumbnail image from the clip."""
    (
        ffmpeg.input(video_path, ss=1)
        .output(output_path, vframes=1)
        .overwrite_output()
        .run(quiet=True)
    )

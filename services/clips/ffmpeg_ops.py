"""FFmpeg operations used by clip generation."""

import logging
import os
from typing import Any

import ffmpeg

from services.clips.constants import normalize_video_scale_mode
from services.clips.models import QualityPreset

logger = logging.getLogger(__name__)


def safe_remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


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

        bg = ffmpeg.input(
            f"color=c={background_color}:s={canvas_w}x{canvas_h}:d={duration}",
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
        .filter("pad", canvas_w, canvas_h, vid_x, vid_y, background_color)
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
    title_padding_x: int,
    title_bar_y: int,
    title_text_y: int,
    title_bar_h: int,
    caption_ass_path: str | None,
    qp: QualityPreset,
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
    if title_show and title_lines:
        if title_bar_enabled:
            stream = stream.drawbox(
                x=0,
                y=title_bar_y,
                width="iw",
                height=title_bar_h,
                color=title_bar_color,
                t="fill",
            )

        line_height = int(title_font_size * 1.3)
        for i, line_text in enumerate(title_lines):
            if title_align == "center":
                text_x = "(w-text_w)/2"
            else:
                text_x = str(title_padding_x)

            line_y = title_text_y + i * line_height

            dt_kwargs: dict[str, Any] = {
                "text": line_text,
                "fontsize": title_font_size,
                "fontcolor": title_font_color,
                "x": text_x,
                "y": str(line_y),
            }
            if title_font_family:
                dt_kwargs["font"] = title_font_family

            if title_stroke_width > 0:
                dt_kwargs["borderw"] = title_stroke_width
                dt_kwargs["bordercolor"] = title_stroke_color
            else:
                dt_kwargs["shadowcolor"] = "black@0.7"
                dt_kwargs["shadowx"] = 2
                dt_kwargs["shadowy"] = 2

            stream = stream.drawtext(**dt_kwargs)

    if caption_ass_path and os.path.isfile(caption_ass_path):
        escaped_path = caption_ass_path.replace("\\", "/").replace(":", "\\:")
        stream = stream.filter("subtitles", escaped_path)

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


def generate_thumbnail(video_path: str, output_path: str) -> None:
    """Generate a thumbnail image from the clip."""
    (
        ffmpeg.input(video_path, ss=1)
        .output(output_path, vframes=1)
        .overwrite_output()
        .run(quiet=True)
    )

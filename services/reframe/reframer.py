"""Video reframing — crops the source video to follow the speaker's face.

This is an optional pre-processing step that produces a reframed intermediate
video before the main compositing pipeline runs.  The reframed video replaces
the source input so that the foreground is always centered on the speaker.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from typing import Any

import cv2
import numpy as np

from services.reframe.detector import detect_faces_in_video
from services.reframe.tracker import CropKeyframe, smooth_trajectory

logger = logging.getLogger(__name__)

# Target crop aspect ratio for portrait reframe (width:height).
_DEFAULT_CROP_ASPECT = (9, 16)
# Padding around the face as a fraction of the crop box.
# 0.3 = face takes up ~70% of the crop width → comfortable headroom.
_DEFAULT_FACE_PADDING = 0.3
# Maximum allowed crop movement per frame (prevents jarring jumps).
_FFMPEG_TIMEOUT_SECONDS = int(os.getenv("FFMPEG_TIMEOUT_SECONDS", "1200"))


def _probe_video(path: str) -> dict[str, Any]:
    """Return width, height, fps, duration from ffprobe."""
    import ffmpeg as ffmpeg_lib

    probe = ffmpeg_lib.probe(path)
    v_stream = next(s for s in probe["streams"] if s["codec_type"] == "video")
    return {
        "width": int(v_stream["width"]),
        "height": int(v_stream["height"]),
        "fps": float(eval(v_stream.get("r_frame_rate", "30/1"))),
        "duration": float(probe.get("format", {}).get("duration", 0)),
    }


def _compute_crop_box(
    kf: CropKeyframe,
    src_w: int,
    src_h: int,
    crop_w: int,
    crop_h: int,
) -> tuple[int, int]:
    """Compute top-left (x, y) for a crop box centered on the keyframe."""
    cx_px = kf.center_x * src_w
    cy_px = kf.center_y * src_h

    x = int(round(cx_px - crop_w / 2))
    y = int(round(cy_px - crop_h / 2))

    # Clamp to source bounds.
    x = max(0, min(x, src_w - crop_w))
    y = max(0, min(y, src_h - crop_h))
    return x, y


def reframe_video(
    input_path: str,
    output_path: str,
    *,
    crop_aspect: tuple[int, int] = _DEFAULT_CROP_ASPECT,
    face_padding: float = _DEFAULT_FACE_PADDING,
    smoothing: float = 0.08,
    sample_fps: float = 3.0,
    output_quality: str = "high",
    start_time: float | None = None,
    end_time: float | None = None,
) -> bool:
    """Detect faces and produce a reframed (cropped) video.

    Returns ``True`` if reframing succeeded, ``False`` if it was skipped
    (e.g. no faces detected, mediapipe unavailable).

    The output is a video for the requested window cropped to follow the
    speaker. Audio is copied unchanged for the same window.
    """
    info = _probe_video(input_path)
    src_w, src_h = info["width"], info["height"]
    fps = info["fps"] or 30.0
    duration = info["duration"]

    if duration <= 0:
        logger.warning("Cannot reframe: source duration is 0")
        return False

    safe_start = max(0.0, float(start_time or 0.0))
    safe_end = duration if end_time is None else max(safe_start, float(end_time))
    safe_start = min(safe_start, duration)
    safe_end = min(max(safe_end, safe_start), duration)
    segment_duration = safe_end - safe_start

    if segment_duration <= 0:
        logger.warning(
            "Cannot reframe: window %.2f-%.2f is empty for %s",
            safe_start,
            safe_end,
            input_path,
        )
        return False

    # Step 1: Detect faces.
    detections = detect_faces_in_video(
        input_path,
        sample_fps=sample_fps,
        start_time=safe_start,
        end_time=safe_end,
    )
    if not detections:
        logger.info("No faces detected — skipping reframe for %s", input_path)
        return False

    # Step 2: Smooth trajectory.
    keyframes = smooth_trajectory(
        detections,
        segment_duration,
        fps,
        smoothing=smoothing,
    )
    if not keyframes:
        return False

    # Step 3: Compute crop dimensions.
    # The crop box should be as large as possible while maintaining the target
    # aspect ratio and fitting within the source.
    crop_ar = crop_aspect[0] / crop_aspect[1]  # e.g. 9/16 = 0.5625
    src_ar = src_w / src_h

    if src_ar > crop_ar:
        # Source is wider than crop → height-limited.
        crop_h = src_h
        crop_w = int(round(crop_h * crop_ar))
    else:
        # Source is taller than crop → width-limited.
        crop_w = src_w
        crop_h = int(round(crop_w / crop_ar))

    # Apply face padding: shrink the crop so the face has headroom.
    padding_factor = max(0.0, min(0.6, face_padding))
    crop_w = max(2, int(crop_w * (1.0 - padding_factor)))
    crop_h = max(2, int(crop_h * (1.0 - padding_factor)))

    # Ensure even dimensions (H.264).
    crop_w -= crop_w % 2
    crop_h -= crop_h % 2

    if crop_w >= src_w and crop_h >= src_h:
        logger.info("Crop box equals source size — skipping reframe")
        return False

    logger.info(
        "Reframing %s window %.2f-%.2f (%.2fs): %dx%d -> crop %dx%d (aspect %d:%d, %d keyframes)",
        input_path,
        safe_start,
        safe_end,
        segment_duration,
        src_w,
        src_h,
        crop_w,
        crop_h,
        crop_aspect[0],
        crop_aspect[1],
        len(keyframes),
    )

    # Step 4: Process frames via OpenCV → pipe to FFmpeg for encoding.
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        logger.error("Cannot open video for reframing: %s", input_path)
        return False

    start_frame = max(0, int(round(safe_start * fps)))
    end_frame = max(start_frame + 1, int(round(safe_end * fps)))
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    from config import FFMPEG_THREADS
    threads = max(1, int(FFMPEG_THREADS))

    # Quality presets matching the clip generator.
    crf_map = {"low": "23", "medium": "18", "high": "15"}
    preset_map = {"low": "fast", "medium": "medium", "high": "slow"}
    crf = crf_map.get(output_quality, "18")
    preset = preset_map.get(output_quality, "medium")

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{crop_w}x{crop_h}",
        "-r", str(fps),
        "-i", "pipe:0",
        # Copy audio from original.
        "-ss", str(safe_start),
        "-t", str(segment_duration),
        "-i", input_path,
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-c:v", "libx264",
        "-crf", crf,
        "-preset", preset,
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "256k",
        "-ar", "48000",
        "-threads", str(threads),
        "-movflags", "+faststart",
        "-colorspace", "bt709",
        "-color_primaries", "bt709",
        "-color_trc", "bt709",
        "-shortest",
        output_path,
    ]

    popen_kwargs: dict[str, Any] = {}
    if sys.platform != "win32":
        popen_kwargs["preexec_fn"] = os.setsid

    proc = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        **popen_kwargs,
    )

    frame_idx = start_frame
    processed_frames = 0
    total_frames = len(keyframes)
    try:
        while True:
            if frame_idx >= end_frame:
                break

            ret, frame = cap.read()
            if not ret:
                break

            # Look up crop position for this frame.
            kf_idx = min(max(0, frame_idx - start_frame), total_frames - 1)
            kf = keyframes[kf_idx] if kf_idx >= 0 else CropKeyframe(0, 0.5, 0.5)
            x, y = _compute_crop_box(kf, src_w, src_h, crop_w, crop_h)

            cropped = frame[y : y + crop_h, x : x + crop_w]

            # Safety: if the crop doesn't match expected size, resize.
            if cropped.shape[1] != crop_w or cropped.shape[0] != crop_h:
                cropped = cv2.resize(cropped, (crop_w, crop_h))

            proc.stdin.write(cropped.tobytes())
            frame_idx += 1
            processed_frames += 1

    except BrokenPipeError:
        logger.warning("FFmpeg pipe closed early during reframe")
    finally:
        cap.release()
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.close()

    proc.wait(timeout=_FFMPEG_TIMEOUT_SECONDS)
    if proc.returncode != 0:
        stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        logger.error("FFmpeg reframe encoding failed (rc=%d):\n%s", proc.returncode, stderr[-2000:])
        return False

    if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
        logger.error("Reframe output is missing or empty: %s", output_path)
        return False

    logger.info(
        "Reframe complete: %d frames processed for window %.2f-%.2f (%.2fs) -> %s (%.1f MB)",
        processed_frames,
        safe_start,
        safe_end,
        segment_duration,
        output_path,
        os.path.getsize(output_path) / (1024 * 1024),
    )
    return True


__all__ = ["reframe_video"]

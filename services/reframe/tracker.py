"""Trajectory smoothing for speaker reframe crop positions.

Takes a sparse list of face detections and produces a dense, smooth
crop-center trajectory suitable for frame-by-frame application.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from services.reframe.detector import FaceDetection

logger = logging.getLogger(__name__)

# Exponential moving average smoothing factor (0..1).
# Lower = smoother / slower to follow, higher = more responsive.
_DEFAULT_SMOOTHING = 0.08
# Dead-zone: ignore movements smaller than this fraction of frame width/height.
_DEFAULT_DEAD_ZONE = 0.02


@dataclass(frozen=True)
class CropKeyframe:
    """Smooth crop center at a specific timestamp (normalised 0..1 coords)."""

    timestamp: float
    center_x: float
    center_y: float


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def smooth_trajectory(
    detections: list[FaceDetection],
    duration: float,
    fps: float = 30.0,
    *,
    smoothing: float = _DEFAULT_SMOOTHING,
    dead_zone: float = _DEFAULT_DEAD_ZONE,
    center_bias: float = 0.6,
    crop_to_source_ratio: float = 1.0,
) -> list[CropKeyframe]:
    """Produce per-frame smooth crop centers from sparse face detections.

    Parameters
    ----------
    detections:
        Time-ordered face detections (may have gaps).
    duration:
        Total video duration in seconds.
    fps:
        Target output frame rate.
    smoothing:
        EMA alpha factor (0=no movement, 1=instant snap).
    dead_zone:
        Base minimum movement threshold to trigger a center update.
    center_bias:
        Blend factor toward frame center (0.0 = track face exactly,
        1.0 = ignore face and keep crop centred). Default 0.6 gives
        natural "camera operator" framing.
    crop_to_source_ratio:
        Ratio of crop size to source size (0..1). Used to scale the
        dead zone so that smaller crops don't stutter.

    Returns a list of :class:`CropKeyframe` — one per output frame.
    """
    if not detections:
        # No faces detected → center crop for entire video.
        total_frames = max(1, int(round(duration * fps)))
        return [
            CropKeyframe(timestamp=i / fps, center_x=0.5, center_y=0.5)
            for i in range(total_frames)
        ]

    total_frames = max(1, int(round(duration * fps)))

    # Build a dense raw-target array by interpolating between detections.
    raw_cx = np.full(total_frames, 0.5, dtype=np.float64)
    raw_cy = np.full(total_frames, 0.5, dtype=np.float64)
    raw_conf = np.full(total_frames, 1.0, dtype=np.float64)

    det_frames = [int(round(d.timestamp * fps)) for d in detections]
    det_cx = [d.center_x for d in detections]
    det_cy = [d.center_y for d in detections]
    det_conf = [d.confidence for d in detections]

    # Fill by linear interpolation between detection keyframes.
    for i in range(len(detections) - 1):
        f_start = max(0, det_frames[i])
        f_end = min(total_frames - 1, det_frames[i + 1])
        if f_end <= f_start:
            continue
        span = f_end - f_start
        for f in range(f_start, f_end + 1):
            t = (f - f_start) / span
            raw_cx[f] = _lerp(det_cx[i], det_cx[i + 1], t)
            raw_cy[f] = _lerp(det_cy[i], det_cy[i + 1], t)
            raw_conf[f] = _lerp(det_conf[i], det_conf[i + 1], t)

    # Extend first/last detection to edges.
    first_f = max(0, det_frames[0])
    last_f = min(total_frames - 1, det_frames[-1])
    raw_cx[:first_f] = det_cx[0]
    raw_cy[:first_f] = det_cy[0]
    raw_conf[:first_f] = det_conf[0]
    raw_cx[last_f + 1:] = det_cx[-1]
    raw_cy[last_f + 1:] = det_cy[-1]
    raw_conf[last_f + 1:] = det_conf[-1]

    # Apply center bias: blend face positions toward frame center.
    # 0.0 = track face exactly, 1.0 = always centre.
    bias = max(0.0, min(1.0, center_bias))
    if bias > 0.0:
        raw_cx = raw_cx * (1.0 - bias) + 0.5 * bias
        raw_cy = raw_cy * (1.0 - bias) + 0.5 * bias

    # Apply exponential moving average for smoothing.
    smooth_cx = np.empty(total_frames, dtype=np.float64)
    smooth_cy = np.empty(total_frames, dtype=np.float64)
    smooth_cx[0] = raw_cx[0]
    smooth_cy[0] = raw_cy[0]

    alpha = max(0.01, min(1.0, smoothing))
    # Scale dead zone by crop-to-source ratio so smaller crops don't stutter.
    dz = max(0.001, dead_zone * max(0.1, crop_to_source_ratio))

    for f in range(1, total_frames):
        dx = raw_cx[f] - smooth_cx[f - 1]
        dy = raw_cy[f] - smooth_cy[f - 1]
        if abs(dx) < dz and abs(dy) < dz:
            smooth_cx[f] = smooth_cx[f - 1]
            smooth_cy[f] = smooth_cy[f - 1]
        else:
            # Weight alpha by detection confidence — low-confidence
            # detections move the crop less.
            conf = max(0.1, min(1.0, raw_conf[f]))
            effective_alpha = alpha * conf
            smooth_cx[f] = smooth_cx[f - 1] + effective_alpha * dx
            smooth_cy[f] = smooth_cy[f - 1] + effective_alpha * dy

    # Clamp to [0, 1].
    smooth_cx = np.clip(smooth_cx, 0.0, 1.0)
    smooth_cy = np.clip(smooth_cy, 0.0, 1.0)

    keyframes = [
        CropKeyframe(
            timestamp=f / fps,
            center_x=float(smooth_cx[f]),
            center_y=float(smooth_cy[f]),
        )
        for f in range(total_frames)
    ]
    logger.info(
        "Trajectory smoothing: %d keyframes from %d detections "
        "(alpha=%.3f, center_bias=%.2f, crop_ratio=%.3f)",
        len(keyframes),
        len(detections),
        alpha,
        bias,
        crop_to_source_ratio,
    )
    return keyframes


__all__ = ["CropKeyframe", "smooth_trajectory"]

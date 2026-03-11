"""Face detection using MediaPipe for speaker reframing.

Samples frames at a configurable rate and returns face bounding-box centers
that can be used to compute a smooth crop trajectory.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_SAMPLE_FPS = 3.0  # analyse ~3 frames per second for efficiency
_MIN_DETECTION_CONFIDENCE = 0.5


@dataclass(frozen=True)
class FaceDetection:
    """A face detection at a specific timestamp."""

    timestamp: float
    # Normalized coordinates [0..1] relative to frame dimensions.
    center_x: float
    center_y: float
    width: float
    height: float
    confidence: float


def detect_faces_in_video(
    video_path: str,
    *,
    sample_fps: float = _SAMPLE_FPS,
    min_confidence: float = _MIN_DETECTION_CONFIDENCE,
    max_frames: int = 600,
) -> list[FaceDetection]:
    """Sample frames from *video_path* and detect faces.

    Returns a time-ordered list of :class:`FaceDetection` for the largest
    (closest) face in each sampled frame.  Frames with no detections are
    omitted so the caller can interpolate gaps.
    """
    try:
        import mediapipe as mp
    except ImportError:
        logger.warning(
            "mediapipe is not installed — speaker reframe is unavailable. "
            "Install with: pip install mediapipe"
        )
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Cannot open video for face detection: %s", video_path)
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_interval = max(1, int(round(fps / sample_fps)))

    detections: list[FaceDetection] = []
    frame_idx = 0
    sampled = 0

    mp_face = mp.solutions.face_detection
    with mp_face.FaceDetection(
        model_selection=1,  # 1 = full-range model (good for far faces)
        min_detection_confidence=min_confidence,
    ) as face_detection:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval != 0:
                frame_idx += 1
                continue

            if sampled >= max_frames:
                break

            timestamp = frame_idx / fps
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb)

            if results.detections:
                # Pick the largest face (by bounding-box area).
                best: Any = max(
                    results.detections,
                    key=lambda d: (
                        d.location_data.relative_bounding_box.width
                        * d.location_data.relative_bounding_box.height
                    ),
                )
                bb = best.location_data.relative_bounding_box
                cx = bb.xmin + bb.width / 2.0
                cy = bb.ymin + bb.height / 2.0
                detections.append(
                    FaceDetection(
                        timestamp=timestamp,
                        center_x=float(np.clip(cx, 0.0, 1.0)),
                        center_y=float(np.clip(cy, 0.0, 1.0)),
                        width=float(bb.width),
                        height=float(bb.height),
                        confidence=float(best.score[0]) if best.score else 0.0,
                    )
                )

            sampled += 1
            frame_idx += 1

    cap.release()
    logger.info(
        "Face detection: %d detections from %d sampled frames (%s)",
        len(detections),
        sampled,
        video_path,
    )
    return detections


__all__ = ["FaceDetection", "detect_faces_in_video"]

"""Shared caption rendering helper for task modules."""

from __future__ import annotations

import logging
import os
from typing import Any

from services.caption_renderer import extract_clip_segments, render_ass


def build_caption_ass(
    *,
    job_id: str,
    clip_id: str,
    transcript: dict[str, Any] | None,
    cap_cfg: dict[str, Any],
    normalized_caption_style: str,
    start_time: float,
    end_time: float,
    vid_y: int,
    vid_h: int,
    work_dir: str,
    logger: logging.Logger,
) -> str | None:
    """Render caption ASS file from transcript and caption config."""
    if not cap_cfg.get("show"):
        return None
    if not transcript or not transcript.get("segments"):
        logger.warning("[%s] Captions requested but no transcript available", job_id)
        return None

    logger.info("[%s] Building %s captions ...", job_id, normalized_caption_style)
    segments = extract_clip_segments(transcript, start_time, end_time)
    if not segments:
        return None

    return render_ass(
        segments,
        style=normalized_caption_style,
        animation=cap_cfg.get("animation", "none"),
        font_size=cap_cfg["fontSize"],
        font_color=cap_cfg["fontColor"],
        font_family=cap_cfg["fontFamily"],
        font_weight=cap_cfg.get("fontWeight", "bold"),
        font_case=cap_cfg.get("fontCase", "as_typed"),
        italic=bool(cap_cfg.get("italic", False)),
        underline=bool(cap_cfg.get("underline", False)),
        stroke_color=cap_cfg.get("strokeColor", "black"),
        stroke_thickness=cap_cfg.get("strokeThickness", 3),
        shadow_color=cap_cfg.get("shadowColor", "#000000AA"),
        shadow_x=cap_cfg.get("shadowX", 2),
        shadow_y=cap_cfg.get("shadowY", 2),
        shadow_blur=cap_cfg.get("shadowBlur", 2),
        highlight_color=cap_cfg["highlightColor"],
        position=cap_cfg["position"],
        lines_per_page=cap_cfg.get("linesPerPage", 1),
        max_words_per_line=cap_cfg["maxWordsPerLine"],
        vid_y=vid_y,
        vid_h=vid_h,
        output_path=os.path.join(work_dir, f"{clip_id}.ass"),
    )


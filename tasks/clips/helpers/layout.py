"""Shared helpers for loading and resolving layout settings in tasks."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

from utils.supabase_client import assert_response_ok, supabase


@dataclass
class LayoutOverrides:
    bg_style: str = "blur"
    bg_color: str = "#000000"
    bg_image_storage_path: str | None = None
    blur_strength: int = 20
    output_quality: str = "medium"
    layout_video: dict[str, Any] = field(default_factory=dict)
    layout_title: dict[str, Any] = field(default_factory=dict)
    layout_captions: dict[str, Any] = field(default_factory=dict)


def load_layout_overrides(
    *,
    user_id: str,
    layout_id: str | None,
    job_id: str,
    logger: logging.Logger,
) -> LayoutOverrides:
    """Load layout settings from Supabase (or return defaults when missing)."""
    overrides = LayoutOverrides()
    if not layout_id:
        return overrides

    logger.info("[%s] Loading layout %s ...", job_id, layout_id)
    layout_resp = (
        supabase.table("layouts")
        .select("*")
        .eq("id", layout_id)
        .eq("user_id", user_id)
        .single()
        .execute()
    )
    assert_response_ok(layout_resp, f"Failed to load layout {layout_id}")

    layout = layout_resp.data
    if not layout:
        logger.warning("[%s] Layout %s not found, using defaults", job_id, layout_id)
        return overrides

    overrides.bg_style = layout.get("background_style", "blur")
    overrides.bg_color = layout.get("background_color") or "#000000"
    overrides.bg_image_storage_path = layout.get("background_image_path")
    overrides.blur_strength = layout.get("background_blur_strength") or 20
    overrides.output_quality = layout.get("output_quality") or "medium"
    overrides.layout_video = layout.get("video") or {}
    overrides.layout_title = layout.get("title") or {}

    raw_layout_captions = layout.get("captions") or {}
    if not isinstance(raw_layout_captions, dict):
        raw_layout_captions = {}

    # Legacy cleanup: templates should persist explicit caption values only.
    overrides.layout_captions = {
        k: v for k, v in raw_layout_captions.items() if k != "preset"
    }
    return overrides


def maybe_download_layout_background_image(
    *,
    bg_style: str,
    bg_image_storage_path: str | None,
    work_dir: str,
    job_id: str,
    logger: logging.Logger,
) -> tuple[str, str | None]:
    """Download the layout background image (if configured)."""
    if bg_style != "image" or not bg_image_storage_path:
        return bg_style, None

    bg_image_path = os.path.join(work_dir, "bg_image.jpg")
    try:
        file_bytes = supabase.storage.from_("layouts").download(bg_image_storage_path)
        with open(bg_image_path, "wb") as f:
            f.write(file_bytes)
        logger.info(
            "[%s] Downloaded background image from storage: %s",
            job_id,
            bg_image_storage_path,
        )
        return bg_style, bg_image_path
    except Exception as dl_err:
        logger.warning(
            "[%s] Could not download background image, falling back to blur: %s",
            job_id,
            dl_err,
        )
        return "blur", None


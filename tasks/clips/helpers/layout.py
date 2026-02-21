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


def _normalize_layout_id(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _first_created_layout_id(*, user_id: str) -> str | None:
    first_layout_resp = (
        supabase.table("layouts")
        .select("id")
        .eq("user_id", user_id)
        .order("created_at", desc=False)
        .order("id", desc=False)
        .limit(1)
        .execute()
    )
    assert_response_ok(first_layout_resp, "Failed to load fallback layout")

    first_rows = first_layout_resp.data or []
    if not first_rows:
        return None

    first_id = _normalize_layout_id(first_rows[0].get("id"))
    if first_id:
        return first_id
    return None


def resolve_effective_layout_id(
    *,
    user_id: str,
    job_id: str,
    logger: logging.Logger,
) -> str | None:
    """Resolve the layout id to be used for clip rendering.

    Priority:
    1) User default layout (`is_default = true`)
    2) First created layout (legacy fallback for users without defaults)
    3) None -> renderer built-in defaults
    """
    logger.info("[%s] Resolving effective layout for user %s", job_id, user_id)

    try:
        default_layout_resp = (
            supabase.table("layouts")
            .select("id")
            .eq("user_id", user_id)
            .eq("is_default", True)
            .limit(1)
            .execute()
        )
        assert_response_ok(default_layout_resp, "Failed to load default layout")
        default_rows = default_layout_resp.data or []
        if default_rows:
            default_id = _normalize_layout_id(default_rows[0].get("id"))
            if default_id:
                logger.info("[%s] Using default layout %s", job_id, default_id)
                return default_id
    except Exception as exc:
        logger.warning(
            "[%s] Default layout lookup failed, trying fallback strategies: %s",
            job_id,
            exc,
        )

    first_id = _first_created_layout_id(user_id=user_id)
    if first_id:
        logger.info("[%s] Falling back to first created layout %s", job_id, first_id)
        return first_id

    logger.info("[%s] No layout available, using renderer defaults", job_id)
    return None


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

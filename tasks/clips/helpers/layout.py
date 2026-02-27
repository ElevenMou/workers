"""Shared helpers for loading and resolving layout settings in tasks."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Literal

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
    layout_intro: dict[str, Any] = field(default_factory=dict)
    layout_outro: dict[str, Any] = field(default_factory=dict)
    layout_overlay: dict[str, Any] = field(default_factory=dict)


LayoutSelectionSource = Literal["requested", "clip", "default", "first_created", "none"]


@dataclass(frozen=True)
class EffectiveLayoutSelection:
    layout_id: str | None
    source: LayoutSelectionSource

    @property
    def should_persist_to_clip(self) -> bool:
        # Only persist explicit user-requested layout selections.
        return self.source == "requested"


def _normalize_layout_id(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _resolve_user_layout_candidate(
    *,
    user_id: str,
    workspace_team_id: str | None,
    workspace_role: str,
    candidate_layout_id: Any,
    source: str,
    job_id: str,
    logger: logging.Logger,
) -> str | None:
    normalized_id = _normalize_layout_id(candidate_layout_id)
    if not normalized_id:
        return None

    try:
        layout_query = supabase.table("layouts").select("id").eq("id", normalized_id)
        if workspace_team_id:
            layout_query = layout_query.eq("team_id", workspace_team_id)
        else:
            layout_query = layout_query.eq("user_id", user_id).is_("team_id", "null")
        layout_resp = layout_query.limit(1).execute()
        assert_response_ok(layout_resp, f"Failed to validate {source} layout {normalized_id}")
    except Exception as exc:
        logger.warning(
            "[%s] Failed to validate %s layout %s: %s",
            job_id,
            source,
            normalized_id,
            exc,
        )
        return None

    rows = layout_resp.data or []
    if not rows:
        logger.warning(
            "[%s] Ignoring %s layout %s because it was not found in active workspace",
            job_id,
            source,
            normalized_id,
        )
        return None

    return normalized_id


def _first_created_layout_id(*, user_id: str, workspace_team_id: str | None) -> str | None:
    first_layout_query = supabase.table("layouts").select("id")
    if workspace_team_id:
        first_layout_query = first_layout_query.eq("team_id", workspace_team_id)
    else:
        first_layout_query = first_layout_query.eq("user_id", user_id).is_("team_id", "null")
    first_layout_resp = (
        first_layout_query
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
    workspace_team_id: str | None = None,
    workspace_role: str = "owner",
    job_id: str,
    logger: logging.Logger,
    requested_layout_id: Any = None,
    clip_layout_id: Any = None,
) -> EffectiveLayoutSelection:
    """Resolve the layout id to be used for clip rendering.

    Priority:
    1) Requested layout id from job payload (`layoutId`) if owned by user
    2) Existing clip layout id if owned by user
    3) User default layout (`is_default = true`)
    4) First created layout (legacy fallback for users without defaults)
    5) None -> renderer built-in defaults
    """
    logger.info("[%s] Resolving effective layout for user %s", job_id, user_id)

    seen_candidate_ids: set[str] = set()
    for source, candidate in (("requested", requested_layout_id), ("clip", clip_layout_id)):
        normalized_candidate = _normalize_layout_id(candidate)
        if not normalized_candidate or normalized_candidate in seen_candidate_ids:
            continue
        seen_candidate_ids.add(normalized_candidate)

        resolved_candidate = _resolve_user_layout_candidate(
            user_id=user_id,
            workspace_team_id=workspace_team_id,
            workspace_role=workspace_role,
            candidate_layout_id=normalized_candidate,
            source=source,
            job_id=job_id,
            logger=logger,
        )
        if resolved_candidate:
            logger.info(
                "[%s] Using %s layout %s",
                job_id,
                source,
                resolved_candidate,
            )
            return EffectiveLayoutSelection(layout_id=resolved_candidate, source=source)

    try:
        default_layout_query = (
            supabase.table("layouts")
            .select("id")
            .eq("is_default", True)
        )
        if workspace_team_id:
            default_layout_query = default_layout_query.eq("team_id", workspace_team_id)
        else:
            default_layout_query = default_layout_query.eq("user_id", user_id).is_(
                "team_id",
                "null",
            )
        default_layout_resp = default_layout_query.limit(1).execute()
        assert_response_ok(default_layout_resp, "Failed to load default layout")
        default_rows = default_layout_resp.data or []
        if default_rows:
            default_id = _normalize_layout_id(default_rows[0].get("id"))
            if default_id:
                logger.info("[%s] Using default layout %s", job_id, default_id)
                return EffectiveLayoutSelection(layout_id=default_id, source="default")
    except Exception as exc:
        logger.warning(
            "[%s] Default layout lookup failed, trying fallback strategies: %s",
            job_id,
            exc,
        )

    first_id = _first_created_layout_id(
        user_id=user_id,
        workspace_team_id=workspace_team_id,
    )
    if first_id:
        logger.info("[%s] Falling back to first created layout %s", job_id, first_id)
        return EffectiveLayoutSelection(layout_id=first_id, source="first_created")

    logger.info("[%s] No layout available, using renderer defaults", job_id)
    return EffectiveLayoutSelection(layout_id=None, source="none")


def load_layout_overrides(
    *,
    user_id: str,
    workspace_team_id: str | None = None,
    workspace_role: str = "owner",
    layout_id: str | None,
    job_id: str,
    logger: logging.Logger,
) -> LayoutOverrides:
    """Load layout settings from Supabase (or return defaults when missing)."""
    overrides = LayoutOverrides()
    if not layout_id:
        return overrides

    logger.info("[%s] Loading layout %s ...", job_id, layout_id)
    layout_query = supabase.table("layouts").select("*").eq("id", layout_id)
    if workspace_team_id:
        layout_query = layout_query.eq("team_id", workspace_team_id)
    else:
        layout_query = layout_query.eq("user_id", user_id).is_("team_id", "null")
    layout_resp = layout_query.single().execute()
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

    raw_intro = layout.get("intro")
    overrides.layout_intro = raw_intro if isinstance(raw_intro, dict) else {}
    raw_outro = layout.get("outro")
    overrides.layout_outro = raw_outro if isinstance(raw_outro, dict) else {}
    raw_overlay = layout.get("overlay")
    overrides.layout_overlay = raw_overlay if isinstance(raw_overlay, dict) else {}

    return overrides


def maybe_download_media_files(
    *,
    intro_cfg: dict[str, Any],
    outro_cfg: dict[str, Any],
    overlay_cfg: dict[str, Any],
    work_dir: str,
    job_id: str,
    logger: logging.Logger,
) -> tuple[str | None, str | None, str | None]:
    """Download intro, outro, and overlay files from storage.

    Returns ``(intro_path, outro_path, overlay_path)``.  Each is ``None``
    when the corresponding feature is disabled or the download fails.
    """

    def _download(cfg: dict[str, Any], label: str, local_name: str) -> str | None:
        if not cfg.get("enabled"):
            return None
        storage_path = cfg.get("storagePath")
        if not storage_path or not isinstance(storage_path, str) or not storage_path.strip():
            return None
        dest = os.path.join(work_dir, local_name)
        try:
            file_bytes = supabase.storage.from_("layouts").download(storage_path.strip())
            with open(dest, "wb") as f:
                f.write(file_bytes)
            logger.info("[%s] Downloaded %s from storage: %s", job_id, label, storage_path)
            return dest
        except Exception as exc:
            logger.warning("[%s] Could not download %s: %s", job_id, label, exc)
            return None

    intro_path = _download(intro_cfg, "intro", "intro_media" + _ext(intro_cfg))
    outro_path = _download(outro_cfg, "outro", "outro_media" + _ext(outro_cfg))
    overlay_path = _download(overlay_cfg, "overlay", "overlay_image.png")
    return intro_path, outro_path, overlay_path


def _ext(cfg: dict[str, Any]) -> str:
    """Guess a file extension from the intro/outro config."""
    storage_path = cfg.get("storagePath", "")
    if isinstance(storage_path, str) and "." in storage_path:
        return "." + storage_path.rsplit(".", 1)[-1].lower()
    return ".mp4" if cfg.get("type") == "video" else ".png"


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

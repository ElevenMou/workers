"""Runtime quality controls for clip generation."""

from __future__ import annotations

from dataclasses import dataclass

_QUALITY_LEVELS = {"low", "medium", "high"}


@dataclass(frozen=True, slots=True)
class ClipQualityControls:
    effective_source_max_height: int | None
    prefer_fresh_source_download: bool
    allow_upload_reencode: bool
    smart_cleanup_crf: int
    smart_cleanup_preset: str


def _normalize_output_quality(value: object) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in _QUALITY_LEVELS:
        return normalized
    return "medium"


def _normalize_source_max_height(value: object) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed


def resolve_quality_controls(
    *,
    output_quality: object,
    policy_source_max_height: object,
) -> ClipQualityControls:
    """Resolve deterministic runtime controls from output quality + policy."""
    normalized_quality = _normalize_output_quality(output_quality)
    normalized_policy_height = _normalize_source_max_height(policy_source_max_height)

    if normalized_quality == "high":
        return ClipQualityControls(
            effective_source_max_height=None,
            prefer_fresh_source_download=True,
            allow_upload_reencode=False,
            smart_cleanup_crf=10,
            smart_cleanup_preset="slow",
        )

    return ClipQualityControls(
        effective_source_max_height=normalized_policy_height,
        prefer_fresh_source_download=False,
        allow_upload_reencode=True,
        smart_cleanup_crf=21,
        smart_cleanup_preset="medium",
    )


__all__ = ["ClipQualityControls", "resolve_quality_controls"]

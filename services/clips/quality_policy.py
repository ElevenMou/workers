"""Tier-aware source/output quality policy for clip generation."""

from __future__ import annotations

from typing import TypedDict


class ClipQualityPolicy(TypedDict):
    source_max_height: int | None
    output_quality: str
    profile: str


_QUALITY_LEVELS = ("low", "medium", "high")
_QUALITY_RANK = {"low": 0, "medium": 1, "high": 2}


def _normalize_quality(value: str | None) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in _QUALITY_LEVELS:
        return normalized
    return "high"


def _normalize_quality_optional(value: object) -> str | None:
    normalized = str(value or "").strip().lower()
    if normalized in _QUALITY_RANK:
        return normalized
    return None


def resolve_effective_output_quality(
    *,
    template_quality: object,
    policy_override_quality: object,
) -> str:
    """Return final output quality, preventing policy-based downgrades.

    Policy may upgrade quality, but never lower a template-selected quality level.
    """
    base_quality = _normalize_quality_optional(template_quality) or "high"
    override_quality = _normalize_quality_optional(policy_override_quality)
    if not override_quality:
        return base_quality
    if _QUALITY_RANK[override_quality] > _QUALITY_RANK[base_quality]:
        return override_quality
    return base_quality


def resolve_clip_quality_policy(
    *,
    tier: str | None,
    clip_duration_seconds: float,
    requested_output_quality: str | None,
) -> ClipQualityPolicy:
    """Return deterministic source/output quality knobs for the clip pipeline."""
    del tier, clip_duration_seconds
    output_quality = _normalize_quality(requested_output_quality)
    source_max_height: int | None = 2160
    profile = "master_first_2160"

    return {
        "source_max_height": source_max_height,
        "output_quality": output_quality,
        "profile": profile,
    }

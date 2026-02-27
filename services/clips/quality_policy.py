"""Tier-aware source/output quality policy for clip generation."""

from __future__ import annotations

from typing import TypedDict


class ClipQualityPolicy(TypedDict):
    source_max_height: int | None
    output_quality: str
    profile: str


_QUALITY_LEVELS = ("low", "medium", "high")
_QUALITY_RANK = {"low": 0, "medium": 1, "high": 2}
_PREMIUM_TIERS = {"pro", "enterprise"}
_SHORT_CLIP_SECONDS = 90.0


def _normalize_quality(value: str | None) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in _QUALITY_LEVELS:
        return normalized
    return "medium"


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
    base_quality = _normalize_quality_optional(template_quality) or "medium"
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
    normalized_tier = str(tier or "").strip().lower()
    output_quality = _normalize_quality(requested_output_quality)
    source_max_height: int | None = 1080
    profile = "balanced_1080"

    # Premium short clips can use higher source quality and a stronger encode profile.
    if normalized_tier in _PREMIUM_TIERS and float(clip_duration_seconds) <= _SHORT_CLIP_SECONDS:
        source_max_height = 2160
        profile = "premium_short_clip"
        if output_quality == "low":
            output_quality = "medium"
        elif output_quality == "medium":
            output_quality = "high"

    return {
        "source_max_height": source_max_height,
        "output_quality": output_quality,
        "profile": profile,
    }

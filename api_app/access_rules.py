"""Subscription and plan rule enforcement for API routes."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from fastapi import HTTPException, status

from utils.supabase_client import assert_response_ok, supabase

FREE_MAX_CONCURRENT_JOBS = 1
_ACTIVE_JOB_STATUSES = ("queued", "processing", "retrying")
_DEFAULT_MAX_CLIP_DURATION_SECONDS = 90
_DEFAULT_PLAN_LIMITS_BY_TIER: dict[str, dict[str, int | bool | None]] = {
    "free": {
        "max_videos_per_month": 10,
        "max_clip_duration_seconds": _DEFAULT_MAX_CLIP_DURATION_SECONDS,
        "max_analysis_duration_seconds": 20 * 60,
        "priority_processing": False,
        "allow_custom_clips": False,
        "clip_retention_days": 7,
        "max_active_jobs": FREE_MAX_CONCURRENT_JOBS,
        "max_templates": 1,
        "templates_editable": False,
        "allow_social_publishing": False,
    },
    "basic": {
        "max_videos_per_month": 60,
        "max_clip_duration_seconds": _DEFAULT_MAX_CLIP_DURATION_SECONDS,
        "max_analysis_duration_seconds": 45 * 60,
        "priority_processing": False,
        "allow_custom_clips": True,
        "clip_retention_days": 30,
        "max_active_jobs": 2,
        "max_templates": 3,
        "templates_editable": True,
        "allow_social_publishing": False,
    },
    "pro": {
        "max_videos_per_month": 220,
        "max_clip_duration_seconds": _DEFAULT_MAX_CLIP_DURATION_SECONDS,
        "max_analysis_duration_seconds": 2 * 60 * 60,
        "priority_processing": True,
        "allow_custom_clips": True,
        "clip_retention_days": 90,
        "max_active_jobs": 5,
        "max_templates": 15,
        "templates_editable": True,
        "allow_social_publishing": True,
    },
    "enterprise": {
        "max_videos_per_month": 600,
        "max_clip_duration_seconds": _DEFAULT_MAX_CLIP_DURATION_SECONDS,
        "max_analysis_duration_seconds": None,
        "priority_processing": True,
        "allow_custom_clips": True,
        "clip_retention_days": 90,
        "max_active_jobs": 10,
        "max_templates": None,
        "templates_editable": True,
        "allow_social_publishing": True,
    },
}
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UserAccessContext:
    tier: str
    status: str
    interval: str | None
    max_videos_per_month: int | None
    max_clip_duration_seconds: int
    max_analysis_duration_seconds: int | None = None
    priority_processing: bool = False
    allow_custom_clips: bool = True
    clip_retention_days: int | None = None
    max_active_jobs: int | None = None
    max_templates: int | None = None
    templates_editable: bool = True
    allow_social_publishing: bool = False


def _as_int(value, fallback: int | None) -> int | None:
    if value is None:
        return fallback
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _as_bool(value, fallback: bool) -> bool:
    if isinstance(value, bool):
        return value
    return fallback


def _fallback_plan_limits(tier: str) -> dict[str, int | bool | None]:
    fallback = _DEFAULT_PLAN_LIMITS_BY_TIER.get(tier)
    if fallback:
        return dict(fallback)
    return dict(_DEFAULT_PLAN_LIMITS_BY_TIER["free"])


def _format_duration_limit(seconds: int) -> str:
    if seconds % 3600 == 0:
        hours = seconds // 3600
        suffix = "hour" if hours == 1 else "hours"
        return f"{hours} {suffix}"

    if seconds % 60 == 0:
        minutes = seconds // 60
        suffix = "minute" if minutes == 1 else "minutes"
        return f"{minutes} {suffix}"

    suffix = "second" if seconds == 1 else "seconds"
    return f"{seconds} {suffix}"


def _read_user_subscription(
    user_id: str,
    supabase_client=supabase,
) -> tuple[str, str, str | None]:
    try:
        response = (
            supabase_client.table("subscriptions")
            .select("tier, status, interval")
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        assert_response_ok(response, f"Failed to load subscription for {user_id}")
        rows = response.data or []
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning("Failed to read subscription for %s: %s", user_id, exc)
        rows = []

    if not rows:
        return "free", "active", "month"

    row = rows[0]
    interval = row.get("interval")
    return (
        str(row.get("tier") or "free"),
        str(row.get("status") or "active"),
        str(interval) if interval else "month",
    )


def _read_plan_limits(
    *,
    tier: str,
    interval: str | None,
    supabase_client=supabase,
) -> dict[str, int | bool | None]:
    fallback = _fallback_plan_limits(tier)
    plan_interval = interval or "month"

    try:
        response = (
            supabase_client.table("pricing_tiers")
            .select(
                "max_videos_per_month, max_clip_duration_seconds, max_analysis_duration_seconds, "
                "priority_processing, allow_custom_clips, clip_retention_days, max_active_jobs, "
                "max_templates, templates_editable, allow_social_publishing"
            )
            .eq("tier", tier)
            .eq("interval", plan_interval)
            .eq("active", True)
            .limit(1)
            .execute()
        )
        assert_response_ok(
            response,
            f"Failed to load pricing limits for tier={tier}, interval={plan_interval}",
        )
        rows = response.data or []
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning("Failed to read pricing limits for tier=%s: %s", tier, exc)
        rows = []

    if not rows:
        return fallback

    row = rows[0]
    return {
        "max_videos_per_month": _as_int(
            row.get("max_videos_per_month"),
            _as_int(fallback.get("max_videos_per_month"), None),
        ),
        "max_clip_duration_seconds": _as_int(
            row.get("max_clip_duration_seconds"),
            _as_int(fallback.get("max_clip_duration_seconds"), _DEFAULT_MAX_CLIP_DURATION_SECONDS),
        ),
        "max_analysis_duration_seconds": _as_int(
            row.get("max_analysis_duration_seconds"),
            _as_int(fallback.get("max_analysis_duration_seconds"), None),
        ),
        "priority_processing": _as_bool(
            row.get("priority_processing"),
            _as_bool(fallback.get("priority_processing"), False),
        ),
        "allow_custom_clips": _as_bool(
            row.get("allow_custom_clips"),
            _as_bool(fallback.get("allow_custom_clips"), tier != "free"),
        ),
        "clip_retention_days": _as_int(
            row.get("clip_retention_days"),
            _as_int(fallback.get("clip_retention_days"), None),
        ),
        "max_active_jobs": _as_int(
            row.get("max_active_jobs"),
            _as_int(fallback.get("max_active_jobs"), None),
        ),
        "max_templates": _as_int(
            row.get("max_templates"),
            _as_int(fallback.get("max_templates"), None),
        ),
        "templates_editable": _as_bool(
            row.get("templates_editable"),
            _as_bool(fallback.get("templates_editable"), tier != "free"),
        ),
        "allow_social_publishing": _as_bool(
            row.get("allow_social_publishing"),
            _as_bool(fallback.get("allow_social_publishing"), tier in {"pro", "enterprise"}),
        ),
    }


def _count_active_jobs_for_user(
    user_id: str,
    *,
    max_rows: int,
    supabase_client=supabase,
) -> int:
    try:
        response = (
            supabase_client.table("jobs")
            .select("id")
            .eq("user_id", user_id)
            .in_("status", list(_ACTIVE_JOB_STATUSES))
            .limit(max_rows)
            .execute()
        )
        assert_response_ok(response, f"Failed to load active jobs for {user_id}")
        return len(response.data or [])
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning("Failed to read active jobs for %s: %s", user_id, exc)
        return 0


def _count_monthly_videos_for_user(user_id: str, supabase_client=supabase) -> int:
    month_start = datetime.now(timezone.utc).replace(
        day=1,
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )
    month_start_iso = month_start.isoformat()

    try:
        response = (
            supabase_client.table("videos")
            .select("id")
            .eq("user_id", user_id)
            .gte("created_at", month_start_iso)
            .execute()
        )
        assert_response_ok(response, f"Failed to load monthly videos for {user_id}")
        return len(response.data or [])
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning("Failed to read monthly videos for %s: %s", user_id, exc)
        return 0


def get_user_access_context(user_id: str, supabase_client=supabase) -> UserAccessContext:
    tier, sub_status, interval = _read_user_subscription(user_id, supabase_client=supabase_client)
    plan_limits = _read_plan_limits(
        tier=tier,
        interval=interval,
        supabase_client=supabase_client,
    )

    max_clip_duration_seconds = _as_int(
        plan_limits.get("max_clip_duration_seconds"),
        _DEFAULT_MAX_CLIP_DURATION_SECONDS,
    )
    if max_clip_duration_seconds is None:
        max_clip_duration_seconds = _DEFAULT_MAX_CLIP_DURATION_SECONDS

    return UserAccessContext(
        tier=tier,
        status=sub_status,
        interval=interval,
        max_videos_per_month=_as_int(plan_limits.get("max_videos_per_month"), None),
        max_clip_duration_seconds=max_clip_duration_seconds,
        max_analysis_duration_seconds=_as_int(
            plan_limits.get("max_analysis_duration_seconds"), None
        ),
        priority_processing=_as_bool(plan_limits.get("priority_processing"), False),
        allow_custom_clips=_as_bool(plan_limits.get("allow_custom_clips"), tier != "free"),
        clip_retention_days=_as_int(plan_limits.get("clip_retention_days"), None),
        max_active_jobs=_as_int(plan_limits.get("max_active_jobs"), None),
        max_templates=_as_int(plan_limits.get("max_templates"), None),
        templates_editable=_as_bool(plan_limits.get("templates_editable"), tier != "free"),
        allow_social_publishing=_as_bool(
            plan_limits.get("allow_social_publishing"),
            tier in {"pro", "enterprise"},
        ),
    )


def enforce_processing_access_rules(user_id: str, supabase_client=supabase) -> UserAccessContext:
    """Block requests based on subscription status and plan concurrency limits."""
    context = get_user_access_context(user_id, supabase_client=supabase_client)

    if context.status == "past_due":
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="Your subscription is past due. Please update billing to continue.",
        )

    max_active_jobs = context.max_active_jobs
    if isinstance(max_active_jobs, int):
        if max_active_jobs <= 0:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Your plan does not allow background processing jobs right now.",
            )

        active_jobs = _count_active_jobs_for_user(
            user_id,
            max_rows=max_active_jobs,
            supabase_client=supabase_client,
        )
        if active_jobs >= max_active_jobs:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=(
                    f"Your plan allows up to {max_active_jobs} active processing jobs. "
                    "Wait for an active job to finish or upgrade your plan."
                ),
            )

    return context


def enforce_monthly_video_limit(user_id: str, supabase_client=supabase):
    """Enforce plan-specific monthly limit on newly analyzed videos."""
    context = get_user_access_context(user_id, supabase_client=supabase_client)
    if context.max_videos_per_month is None:
        return

    monthly_video_count = _count_monthly_videos_for_user(user_id, supabase_client=supabase_client)
    if monthly_video_count >= context.max_videos_per_month:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"Monthly video limit reached for your plan "
                f"({context.max_videos_per_month}/month)."
            ),
        )


def enforce_clip_duration_limit(
    *,
    user_id: str,
    duration_seconds: float,
    supabase_client=supabase,
):
    """Enforce plan-specific maximum clip duration."""
    context = get_user_access_context(user_id, supabase_client=supabase_client)
    if duration_seconds > float(context.max_clip_duration_seconds):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Your plan allows clips up to "
                f"{context.max_clip_duration_seconds} seconds."
            ),
        )


def enforce_custom_clip_access(*, context: UserAccessContext):
    """Block custom-clip generation for plans that disallow it."""
    if context.allow_custom_clips:
        return

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Custom clip generation is available on paid plans only.",
    )


def is_analysis_duration_allowed(
    *,
    context: UserAccessContext,
    duration_seconds: float,
) -> bool:
    max_seconds = context.max_analysis_duration_seconds
    if max_seconds is None:
        return True
    return duration_seconds <= float(max_seconds)


def enforce_analysis_duration_limit(
    *,
    context: UserAccessContext,
    duration_seconds: float,
):
    """Enforce plan-specific maximum source-video analysis length."""
    max_seconds = context.max_analysis_duration_seconds
    if max_seconds is None:
        return

    if duration_seconds <= float(max_seconds):
        return

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Your plan allows video analysis up to {_format_duration_limit(max_seconds)}.",
    )

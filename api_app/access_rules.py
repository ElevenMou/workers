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
_DEFAULT_MAX_VIDEOS_PER_MONTH_BY_TIER = {
    "free": 10,
}
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UserAccessContext:
    tier: str
    status: str
    interval: str | None
    max_videos_per_month: int | None
    max_clip_duration_seconds: int


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
) -> tuple[int | None, int]:
    fallback_max_videos = _DEFAULT_MAX_VIDEOS_PER_MONTH_BY_TIER.get(tier)
    fallback_max_clip_duration = _DEFAULT_MAX_CLIP_DURATION_SECONDS
    plan_interval = interval or "month"

    try:
        response = (
            supabase_client.table("pricing_tiers")
            .select("max_videos_per_month, max_clip_duration_seconds")
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
        return fallback_max_videos, fallback_max_clip_duration

    row = rows[0]
    max_videos_raw = row.get("max_videos_per_month")
    max_clip_duration_raw = row.get("max_clip_duration_seconds")

    max_videos = int(max_videos_raw) if max_videos_raw is not None else fallback_max_videos
    max_clip_duration = (
        int(max_clip_duration_raw)
        if max_clip_duration_raw is not None
        else fallback_max_clip_duration
    )
    return max_videos, max_clip_duration


def _count_active_jobs_for_user(user_id: str, supabase_client=supabase) -> int:
    try:
        response = (
            supabase_client.table("jobs")
            .select("id")
            .eq("user_id", user_id)
            .in_("status", list(_ACTIVE_JOB_STATUSES))
            .limit(FREE_MAX_CONCURRENT_JOBS)
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
    max_videos_per_month, max_clip_duration_seconds = _read_plan_limits(
        tier=tier,
        interval=interval,
        supabase_client=supabase_client,
    )

    return UserAccessContext(
        tier=tier,
        status=sub_status,
        interval=interval,
        max_videos_per_month=max_videos_per_month,
        max_clip_duration_seconds=max_clip_duration_seconds,
    )


def enforce_processing_access_rules(user_id: str, supabase_client=supabase) -> UserAccessContext:
    """Block requests based on subscription status and free-plan limits."""
    context = get_user_access_context(user_id, supabase_client=supabase_client)

    if context.status == "past_due":
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="Your subscription is past due. Please update billing to continue.",
        )

    if context.tier == "free":
        active_jobs = _count_active_jobs_for_user(user_id, supabase_client=supabase_client)
        if active_jobs >= FREE_MAX_CONCURRENT_JOBS:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=(
                    "Free plan allows only one active processing job at a time. "
                    "Wait for the current job to finish or upgrade your plan."
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

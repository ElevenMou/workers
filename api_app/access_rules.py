"""Subscription and plan rule enforcement for API routes."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from fastapi import HTTPException, status

from utils.supabase_client import assert_response_ok, supabase

FREE_MAX_CONCURRENT_JOBS = 1
_ACTIVE_JOB_STATUSES = ("queued", "processing", "retrying")
_MAX_CLIP_DURATION_SECONDS_BY_TIER: dict[str, int] = {
    "free": 60,
    "basic": 120,
    "pro": 300,
    "enterprise": 300,
}
_DEFAULT_MAX_CLIP_DURATION_SECONDS = _MAX_CLIP_DURATION_SECONDS_BY_TIER["pro"]
_DEFAULT_PLAN_LIMITS_BY_TIER: dict[str, dict[str, int | bool | None]] = {
    "free": {
        "max_videos_per_month": 10,
        "max_clip_duration_seconds": _MAX_CLIP_DURATION_SECONDS_BY_TIER["free"],
        "max_analysis_duration_seconds": 20 * 60,
        "priority_processing": False,
        "allow_custom_clips": False,
        "clip_retention_days": 7,
        "max_active_jobs": FREE_MAX_CONCURRENT_JOBS,
        "max_templates": 1,
        "templates_editable": False,
        "allow_social_publishing": False,
        "max_teams": 0,
        "max_team_members": 0,
    },
    "basic": {
        "max_videos_per_month": 60,
        "max_clip_duration_seconds": _MAX_CLIP_DURATION_SECONDS_BY_TIER["basic"],
        "max_analysis_duration_seconds": 45 * 60,
        "priority_processing": False,
        "allow_custom_clips": True,
        "clip_retention_days": 30,
        "max_active_jobs": 2,
        "max_templates": 3,
        "templates_editable": True,
        "allow_social_publishing": False,
        "max_teams": 0,
        "max_team_members": 0,
    },
    "pro": {
        "max_videos_per_month": 220,
        "max_clip_duration_seconds": _MAX_CLIP_DURATION_SECONDS_BY_TIER["pro"],
        "max_analysis_duration_seconds": 2 * 60 * 60,
        "priority_processing": True,
        "allow_custom_clips": True,
        "clip_retention_days": 90,
        "max_active_jobs": 5,
        "max_templates": 15,
        "templates_editable": True,
        "allow_social_publishing": True,
        "max_teams": 3,
        "max_team_members": 5,
    },
    "enterprise": {
        "max_videos_per_month": 600,
        "max_clip_duration_seconds": _MAX_CLIP_DURATION_SECONDS_BY_TIER["enterprise"],
        "max_analysis_duration_seconds": None,
        "priority_processing": True,
        "allow_custom_clips": True,
        "clip_retention_days": 90,
        "max_active_jobs": 10,
        "max_templates": None,
        "templates_editable": True,
        "allow_social_publishing": True,
        "max_teams": None,
        "max_team_members": None,
    },
}
logger = logging.getLogger(__name__)


def _default_max_clip_duration_for_tier(tier: str) -> int:
    normalized = str(tier or "").strip().lower()
    return int(_MAX_CLIP_DURATION_SECONDS_BY_TIER.get(normalized, _DEFAULT_MAX_CLIP_DURATION_SECONDS))


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
    max_teams: int | None = None
    max_team_members: int | None = None
    workspace_team_id: str | None = None
    workspace_role: str = "owner"
    billing_owner_user_id: str | None = None
    team_writable: bool = True
    charge_source: str = "owner_wallet"


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
                "max_templates, templates_editable, allow_social_publishing, "
                "max_teams, max_team_members"
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
            _as_int(
                fallback.get("max_clip_duration_seconds"),
                _default_max_clip_duration_for_tier(tier),
            ),
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
        "max_teams": _as_int(
            row.get("max_teams"),
            _as_int(fallback.get("max_teams"), None),
        ),
        "max_team_members": _as_int(
            row.get("max_team_members"),
            _as_int(fallback.get("max_team_members"), None),
        ),
    }


def _read_profile_active_team_id(
    user_id: str,
    *,
    supabase_client=supabase,
) -> str | None:
    try:
        response = (
            supabase_client.table("profiles")
            .select("active_team_id")
            .eq("id", user_id)
            .limit(1)
            .execute()
        )
        assert_response_ok(response, f"Failed to load profile workspace for {user_id}")
        rows = response.data or []
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning("Failed to read active team for %s: %s", user_id, exc)
        rows = []

    if not rows:
        return None

    active_team_id = rows[0].get("active_team_id")
    return str(active_team_id) if active_team_id else None


def _read_workspace_team_membership(
    *,
    team_id: str,
    user_id: str,
    supabase_client=supabase,
) -> tuple[str, str] | None:
    try:
        member_response = (
            supabase_client.table("team_members")
            .select("role")
            .eq("team_id", team_id)
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        assert_response_ok(
            member_response,
            f"Failed to load team membership for user={user_id} team={team_id}",
        )
        member_rows = member_response.data or []
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning(
            "Failed to read team membership for user=%s team=%s: %s",
            user_id,
            team_id,
            exc,
        )
        member_rows = []

    if not member_rows:
        return None

    role_value = member_rows[0].get("role")
    role = str(role_value) if role_value else "member"

    try:
        team_response = (
            supabase_client.table("teams")
            .select("owner_user_id")
            .eq("id", team_id)
            .limit(1)
            .execute()
        )
        assert_response_ok(team_response, f"Failed to load team owner for {team_id}")
        team_rows = team_response.data or []
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning("Failed to read team owner for %s: %s", team_id, exc)
        team_rows = []

    if not team_rows:
        return None

    owner_user_id = team_rows[0].get("owner_user_id")
    if not owner_user_id:
        return None

    return str(owner_user_id), role


def _count_active_jobs_for_owner(
    owner_user_id: str,
    *,
    max_rows: int,
    supabase_client=supabase,
) -> int:
    try:
        try:
            response = (
                supabase_client.table("jobs")
                .select("id")
                .eq("billing_owner_user_id", owner_user_id)
                .in_("status", list(_ACTIVE_JOB_STATUSES))
                .limit(max_rows)
                .execute()
            )
            assert_response_ok(response, f"Failed to load active jobs for owner={owner_user_id}")
            return len(response.data or [])
        except Exception:
            fallback_response = (
                supabase_client.table("jobs")
                .select("id")
                .eq("user_id", owner_user_id)
                .in_("status", list(_ACTIVE_JOB_STATUSES))
                .limit(max_rows)
                .execute()
            )
            assert_response_ok(
                fallback_response,
                f"Failed to load active jobs fallback for owner={owner_user_id}",
            )
            return len(fallback_response.data or [])
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning("Failed to read active jobs for owner=%s: %s", owner_user_id, exc)
        return 0


def _count_monthly_videos_for_owner(owner_user_id: str, supabase_client=supabase) -> int:
    month_start = datetime.now(timezone.utc).replace(
        day=1,
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )
    month_start_iso = month_start.isoformat()

    try:
        try:
            response = (
                supabase_client.table("videos")
                .select("id")
                .eq("billing_owner_user_id", owner_user_id)
                .gte("created_at", month_start_iso)
                .execute()
            )
            assert_response_ok(
                response,
                f"Failed to load monthly videos for owner={owner_user_id}",
            )
            return len(response.data or [])
        except Exception:
            fallback_response = (
                supabase_client.table("videos")
                .select("id")
                .eq("user_id", owner_user_id)
                .gte("created_at", month_start_iso)
                .execute()
            )
            assert_response_ok(
                fallback_response,
                f"Failed to load monthly videos fallback for owner={owner_user_id}",
            )
            return len(fallback_response.data or [])
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning("Failed to read monthly videos for owner=%s: %s", owner_user_id, exc)
        return 0


def get_user_access_context(user_id: str, supabase_client=supabase) -> UserAccessContext:
    workspace_team_id = _read_profile_active_team_id(
        user_id,
        supabase_client=supabase_client,
    )
    workspace_role = "owner"
    billing_owner_user_id = user_id
    charge_source = "owner_wallet"

    if workspace_team_id:
        membership = _read_workspace_team_membership(
            team_id=workspace_team_id,
            user_id=user_id,
            supabase_client=supabase_client,
        )
        if membership:
            billing_owner_user_id, workspace_role = membership
            charge_source = "team_wallet"
        else:
            workspace_team_id = None
            workspace_role = "owner"
            billing_owner_user_id = user_id
            charge_source = "owner_wallet"

    tier, sub_status, interval = _read_user_subscription(
        billing_owner_user_id,
        supabase_client=supabase_client,
    )
    plan_limits = _read_plan_limits(
        tier=tier,
        interval=interval,
        supabase_client=supabase_client,
    )
    team_writable = (
        workspace_team_id is None
        or tier in {"pro", "enterprise"}
    )

    max_clip_duration_seconds = _as_int(
        plan_limits.get("max_clip_duration_seconds"),
        _default_max_clip_duration_for_tier(tier),
    )
    if max_clip_duration_seconds is None:
        max_clip_duration_seconds = _default_max_clip_duration_for_tier(tier)

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
        max_teams=_as_int(plan_limits.get("max_teams"), None),
        max_team_members=_as_int(plan_limits.get("max_team_members"), None),
        workspace_team_id=workspace_team_id,
        workspace_role=workspace_role,
        billing_owner_user_id=billing_owner_user_id,
        team_writable=team_writable,
        charge_source=charge_source,
    )


def enforce_processing_access_rules(user_id: str, supabase_client=supabase) -> UserAccessContext:
    """Block requests based on subscription status and plan concurrency limits."""
    context = get_user_access_context(user_id, supabase_client=supabase_client)

    if context.status == "past_due":
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="Your subscription is past due. Please update billing to continue.",
        )

    if context.workspace_team_id and not context.team_writable:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "This team workspace is read-only because the owner is currently below the Pro tier."
            ),
        )

    max_active_jobs = context.max_active_jobs
    if isinstance(max_active_jobs, int):
        if max_active_jobs <= 0:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Your plan does not allow background processing jobs right now.",
            )

        owner_user_id = context.billing_owner_user_id or user_id
        active_jobs = _count_active_jobs_for_owner(
            owner_user_id,
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

    owner_user_id = context.billing_owner_user_id or user_id
    monthly_video_count = _count_monthly_videos_for_owner(
        owner_user_id,
        supabase_client=supabase_client,
    )
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

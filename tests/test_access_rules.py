from __future__ import annotations

import pytest
from fastapi import HTTPException

import api_app.access_rules as access_rules


def _context(
    *,
    tier: str = "basic",
    status: str = "active",
    interval: str | None = "month",
    max_videos_per_month: int | None = 60,
    max_clip_duration_seconds: int = 90,
) -> access_rules.UserAccessContext:
    return access_rules.UserAccessContext(
        tier=tier,
        status=status,
        interval=interval,
        max_videos_per_month=max_videos_per_month,
        max_clip_duration_seconds=max_clip_duration_seconds,
    )


def test_enforce_processing_rules_blocks_past_due(monkeypatch):
    monkeypatch.setattr(
        access_rules,
        "get_user_access_context",
        lambda *_args, **_kwargs: _context(status="past_due"),
    )

    with pytest.raises(HTTPException) as exc_info:
        access_rules.enforce_processing_access_rules("user-1")

    assert exc_info.value.status_code == 402
    assert "past due" in str(exc_info.value.detail).lower()


def test_enforce_processing_rules_blocks_free_parallel_jobs(monkeypatch):
    monkeypatch.setattr(
        access_rules,
        "get_user_access_context",
        lambda *_args, **_kwargs: _context(tier="free"),
    )
    monkeypatch.setattr(
        access_rules,
        "_count_active_jobs_for_user",
        lambda *_args, **_kwargs: access_rules.FREE_MAX_CONCURRENT_JOBS,
    )

    with pytest.raises(HTTPException) as exc_info:
        access_rules.enforce_processing_access_rules("user-1")

    assert exc_info.value.status_code == 429
    assert "one active processing job" in str(exc_info.value.detail).lower()


def test_enforce_monthly_video_limit_blocks_when_limit_reached(monkeypatch):
    monkeypatch.setattr(
        access_rules,
        "get_user_access_context",
        lambda *_args, **_kwargs: _context(max_videos_per_month=3),
    )
    monkeypatch.setattr(
        access_rules,
        "_count_monthly_videos_for_user",
        lambda *_args, **_kwargs: 3,
    )

    with pytest.raises(HTTPException) as exc_info:
        access_rules.enforce_monthly_video_limit("user-1")

    assert exc_info.value.status_code == 403
    assert "monthly video limit reached" in str(exc_info.value.detail).lower()


def test_enforce_clip_duration_limit_blocks_over_plan_cap(monkeypatch):
    monkeypatch.setattr(
        access_rules,
        "get_user_access_context",
        lambda *_args, **_kwargs: _context(max_clip_duration_seconds=75),
    )

    with pytest.raises(HTTPException) as exc_info:
        access_rules.enforce_clip_duration_limit(user_id="user-1", duration_seconds=80)

    assert exc_info.value.status_code == 400
    assert "clips up to 75 seconds" in str(exc_info.value.detail).lower()

"""Shared subscription lifecycle normalization for worker-side access rules."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping


def _as_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _parse_iso_timestamp(value: Any) -> datetime | None:
    text = _as_text(value)
    if not text:
        return None

    normalized = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def normalize_subscription_status(raw_status: Any) -> str:
    normalized = _as_text(raw_status)
    if normalized in {"active", "trialing", "past_due", "canceled", "paused"}:
        return normalized
    if normalized in {"unpaid", "incomplete", "incomplete_expired"}:
        return "past_due"
    return "active"


def normalize_subscription_interval(raw_interval: Any) -> str | None:
    normalized = _as_text(raw_interval)
    if normalized in {"month", "year"}:
        return normalized
    return None


def normalize_subscription_tier(raw_tier: Any) -> str:
    normalized = _as_text(raw_tier)
    if normalized in {"free", "basic", "pro", "enterprise"}:
        return normalized
    return "free"


@dataclass(frozen=True)
class EffectiveSubscriptionState:
    tier: str
    status: str
    interval: str | None
    raw_tier: str
    raw_status: str
    raw_interval: str | None
    canceled_at: str | None
    current_period_end: str | None
    is_scheduled_cancellation: bool
    is_terminal: bool


def derive_effective_subscription_state(
    row: Mapping[str, Any] | None,
    *,
    now: datetime | None = None,
) -> EffectiveSubscriptionState:
    reference_now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    raw_tier = normalize_subscription_tier(row.get("tier") if row else None)
    raw_status = normalize_subscription_status(row.get("status") if row else None)
    raw_interval = normalize_subscription_interval(row.get("interval") if row else None) or "month"
    canceled_at = _as_text(row.get("canceled_at") if row else None)
    current_period_end = _as_text(row.get("current_period_end") if row else None)

    canceled_at_dt = _parse_iso_timestamp(canceled_at)
    current_period_end_dt = _parse_iso_timestamp(current_period_end)
    has_cancellation_signal = bool(canceled_at_dt) or raw_status == "canceled"
    period_ended = (
        current_period_end_dt is not None and current_period_end_dt <= reference_now
    )
    is_terminal = raw_tier != "free" and (
        (has_cancellation_signal and period_ended)
        or (raw_status == "canceled" and current_period_end_dt is None)
    )
    is_scheduled_cancellation = (
        raw_tier != "free" and not is_terminal and bool(canceled_at_dt)
    )

    effective_tier = "free" if is_terminal else raw_tier
    effective_status = "active" if is_terminal else ("active" if is_scheduled_cancellation and raw_status == "canceled" else raw_status)
    effective_interval = "month" if effective_tier == "free" else raw_interval

    return EffectiveSubscriptionState(
        tier=effective_tier,
        status=effective_status,
        interval=effective_interval,
        raw_tier=raw_tier,
        raw_status=raw_status,
        raw_interval=raw_interval,
        canceled_at=canceled_at,
        current_period_end=current_period_end,
        is_scheduled_cancellation=is_scheduled_cancellation,
        is_terminal=is_terminal,
    )

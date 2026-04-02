import pytest

from utils import supabase_client


def test_charge_clip_generation_credits_skips_zero_amount(monkeypatch):
    owner_charge_calls: list[dict] = []
    team_charge_calls: list[dict] = []
    usage_calls: list[dict] = []

    monkeypatch.setattr(
        supabase_client,
        "_charge_credits_or_raise",
        lambda **kwargs: owner_charge_calls.append(kwargs),
    )
    monkeypatch.setattr(
        supabase_client,
        "_charge_team_credits_or_raise",
        lambda **kwargs: team_charge_calls.append(kwargs),
    )
    monkeypatch.setattr(
        supabase_client,
        "_emit_usage_event_after_charge",
        lambda **kwargs: usage_calls.append(kwargs),
    )

    supabase_client.charge_clip_generation_credits(
        user_id="user-1",
        amount=0,
        description="free first generation",
        video_id="video-1",
        clip_id="clip-1",
    )

    assert owner_charge_calls == []
    assert team_charge_calls == []
    assert usage_calls == []


def test_charge_video_analysis_credits_skips_zero_amount(monkeypatch):
    owner_charge_calls: list[dict] = []
    team_charge_calls: list[dict] = []
    usage_calls: list[dict] = []

    monkeypatch.setattr(
        supabase_client,
        "_charge_credits_or_raise",
        lambda **kwargs: owner_charge_calls.append(kwargs),
    )
    monkeypatch.setattr(
        supabase_client,
        "_charge_team_credits_or_raise",
        lambda **kwargs: team_charge_calls.append(kwargs),
    )
    monkeypatch.setattr(
        supabase_client,
        "_emit_usage_event_after_charge",
        lambda **kwargs: usage_calls.append(kwargs),
    )

    supabase_client.charge_video_analysis_credits(
        user_id="user-1",
        amount=0,
        description="no-op analysis charge",
        video_id="video-1",
    )

    assert owner_charge_calls == []
    assert team_charge_calls == []
    assert usage_calls == []


def test_charge_clip_generation_credits_rejects_negative_amount():
    with pytest.raises(RuntimeError, match="amount must be >= 0"):
        supabase_client.charge_clip_generation_credits(
            user_id="user-1",
            amount=-1,
            description="invalid",
            video_id="video-1",
            clip_id="clip-1",
        )


def test_charge_video_analysis_credits_rejects_negative_amount():
    with pytest.raises(RuntimeError, match="amount must be >= 0"):
        supabase_client.charge_video_analysis_credits(
            user_id="user-1",
            amount=-1,
            description="invalid",
            video_id="video-1",
        )


def test_charge_clip_generation_credits_skips_usage_when_owner_charge_is_deduped(monkeypatch):
    usage_calls: list[dict] = []

    monkeypatch.setattr(
        supabase_client,
        "_charge_credits_or_raise",
        lambda **kwargs: False,
    )
    monkeypatch.setattr(
        supabase_client,
        "emit_clip_generation_usage_event",
        lambda **kwargs: usage_calls.append(kwargs),
    )

    applied = supabase_client.charge_clip_generation_credits(
        user_id="user-1",
        amount=3,
        description="deduped generation",
        video_id="video-1",
        clip_id="clip-1",
    )

    assert applied is False
    assert usage_calls == []


def test_charge_video_analysis_credits_skips_usage_when_team_charge_is_deduped(monkeypatch):
    usage_calls: list[dict] = []

    monkeypatch.setattr(
        supabase_client,
        "_charge_team_credits_or_raise",
        lambda **kwargs: False,
    )
    monkeypatch.setattr(
        supabase_client,
        "emit_video_analysis_usage_event",
        lambda **kwargs: usage_calls.append(kwargs),
    )

    applied = supabase_client.charge_video_analysis_credits(
        user_id="user-1",
        amount=4,
        description="deduped team analysis",
        video_id="video-1",
        charge_source="team_wallet",
        team_id="team-1",
        billing_owner_user_id="owner-1",
    )

    assert applied is False
    assert usage_calls == []

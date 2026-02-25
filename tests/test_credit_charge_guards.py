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

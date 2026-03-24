from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from services.social import tiktok
from services.social.base import (
    PublicationContext,
    PublicationMedia,
    SocialAccountContext,
    SocialAccountTokens,
    SocialProviderError,
)


class _FakeResponse:
    def __init__(self, *, status_code: int = 200, payload: dict | None = None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text

    def json(self):
        return self._payload


def _account(refresh_token: str | None = "refresh-token") -> SocialAccountContext:
    return SocialAccountContext(
        id="account-1",
        provider="tiktok",
        external_account_id="open-id-1",
        display_name="TikTok account",
        handle="clipry",
        provider_metadata={"openId": "open-id-1"},
        scopes=["user.info.basic", "video.publish"],
        tokens=SocialAccountTokens(
            access_token="access-token",
            refresh_token=refresh_token,
            token_expires_at=None,
        ),
    )


def test_tiktok_refresh_token_returns_none_without_refresh_token():
    assert tiktok.refresh_access_token(_account(refresh_token=None)) is None


def test_tiktok_refresh_token_raises_provider_error_for_oauth_error_payload(monkeypatch):
    monkeypatch.setenv("TIKTOK_CLIENT_KEY", "client-key")
    monkeypatch.setenv("TIKTOK_CLIENT_SECRET", "client-secret")

    def _fake_request(method: str, url: str, **kwargs):
        assert method == "POST"
        assert url == "https://open.tiktokapis.com/v2/oauth/token/"
        assert kwargs["headers"]["Content-Type"] == "application/x-www-form-urlencoded"
        assert kwargs["data"]["grant_type"] == "refresh_token"
        assert kwargs["data"]["refresh_token"] == "refresh-token"
        return _FakeResponse(
            payload={
                "error": "invalid_grant",
                "error_description": "Refresh token is invalid or expired.",
                "log_id": "log-123",
            }
        )

    monkeypatch.setattr(tiktok.httpx, "request", _fake_request)

    with pytest.raises(SocialProviderError) as exc:
        tiktok.refresh_access_token(_account())

    assert exc.value.code == "tiktok_token_refresh_failed"
    assert exc.value.refresh_required is True
    assert str(exc.value) == "Refresh token is invalid or expired."
    assert exc.value.provider_payload == {
        "error": "invalid_grant",
        "error_description": "Refresh token is invalid or expired.",
        "log_id": "log-123",
    }


def test_tiktok_refresh_token_returns_rotated_tokens(monkeypatch):
    monkeypatch.setenv("TIKTOK_CLIENT_KEY", "client-key")
    monkeypatch.setenv("TIKTOK_CLIENT_SECRET", "client-secret")

    def _fake_request(method: str, url: str, **kwargs):
        assert method == "POST"
        assert url == "https://open.tiktokapis.com/v2/oauth/token/"
        return _FakeResponse(
            payload={
                "access_token": "new-access-token",
                "expires_in": 7200,
                "refresh_token": "new-refresh-token",
                "refresh_expires_in": 31536000,
                "open_id": "open-id-1",
                "scope": "user.info.basic,video.publish",
            }
        )

    monkeypatch.setattr(tiktok.httpx, "request", _fake_request)

    before = datetime.now(timezone.utc)
    result = tiktok.refresh_access_token(_account())
    after = datetime.now(timezone.utc)

    assert result is not None
    assert result.updated_access_token == "new-access-token"
    assert result.updated_refresh_token == "new-refresh-token"
    assert result.provider_payload["open_id"] == "open-id-1"
    assert result.updated_token_expires_at is not None
    assert before + timedelta(seconds=7190) <= result.updated_token_expires_at
    assert result.updated_token_expires_at <= after + timedelta(seconds=7210)


def test_tiktok_plan_file_upload_uses_floor_chunk_count_for_large_files():
    chunk_size, total_chunk_count = tiktok._plan_file_upload(80_070_528)

    assert chunk_size == 10 * 1024 * 1024
    assert total_chunk_count == 7


def test_tiktok_publish_video_init_uses_floor_chunk_count_for_large_files(tmp_path, monkeypatch):
    clip_path = tmp_path / "clip.mp4"
    clip_path.write_bytes(b"clip")
    media = PublicationMedia(
        local_path=str(clip_path),
        file_size=80_070_528,
        content_type="video/mp4",
        signed_url=None,
        width=1080,
        height=1920,
        duration_seconds=66.0,
    )
    publication = PublicationContext(
        id="publication-1",
        clip_id="clip-1",
        clip_title="Launch clip",
        caption="Launch caption",
        youtube_title=None,
        scheduled_for=datetime.now(timezone.utc),
    )
    init_calls: list[dict] = []

    monkeypatch.setattr(
        tiktok,
        "_query_creator_info",
        lambda _account: {
            "privacy_level_options": ["SELF_ONLY"],
            "max_video_post_duration_sec": 600,
        },
    )

    def _fake_init_video_publish(**kwargs):
        init_calls.append(kwargs)
        return {"data": {"publish_id": "publish-123", "upload_url": "https://upload.example.com"}}

    monkeypatch.setattr(tiktok, "_init_video_publish", _fake_init_video_publish)
    monkeypatch.setattr(tiktok, "_upload_file_chunks", lambda **_kwargs: None)
    monkeypatch.setattr(
        tiktok,
        "_authorized_post",
        lambda *args, **kwargs: {
            "error": {"code": "ok"},
            "data": {
                "status": "PUBLISH_COMPLETE",
                "publicly_available_post_id": ["post-123"],
            },
        },
    )
    monkeypatch.setattr(tiktok.time, "sleep", lambda _seconds: None)

    result = tiktok.publish_video(
        account=_account(),
        publication=publication,
        media=media,
    )

    assert result.remote_post_id == "post-123"
    assert init_calls
    assert init_calls[0]["chunk_size"] == 10 * 1024 * 1024
    assert init_calls[0]["total_chunk_count"] == 7

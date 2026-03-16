from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import api_app.routers.publishing as publishing_router
import pytest
from api_app.app import app
from api_app.auth import AuthenticatedUser, get_current_user
from services.social import meta_facebook
from services.social.base import (
    PublicationContext,
    PublicationMedia,
    SocialAccountContext,
    SocialAccountTokens,
    SocialProviderError,
)


class _FakeResponse:
    def __init__(self, *, status_code: int = 200, payload: object | None = None, text: str = ""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def _future_timestamp() -> str:
    return (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()


def _facebook_account() -> SocialAccountContext:
    return SocialAccountContext(
        id="account-1",
        provider="facebook_page",
        external_account_id="page-1",
        display_name="Page One",
        handle=None,
        provider_metadata={},
        scopes=[],
        tokens=SocialAccountTokens(access_token="page-token"),
    )


def _publication_context() -> PublicationContext:
    return PublicationContext(
        id="publication-1",
        clip_id="clip-1",
        clip_title="Launch clip",
        caption="Launch caption",
        youtube_title=None,
        scheduled_for=datetime.now(timezone.utc),
    )


def _media(tmp_path, *, width: int = 1080, height: int = 1920, duration_seconds: float = 45.0):
    clip_path = tmp_path / "publication.mp4"
    clip_bytes = b"clip-bytes"
    clip_path.write_bytes(clip_bytes)
    return PublicationMedia(
        local_path=str(clip_path),
        file_size=len(clip_bytes),
        content_type="video/mp4",
        signed_url=None,
        width=width,
        height=height,
        duration_seconds=duration_seconds,
    )


def _mock_create_publish_flow(
    monkeypatch,
    *,
    clip_duration_seconds: float,
    accounts: list[dict],
    publications: list[dict],
) -> None:
    monkeypatch.setattr(
        publishing_router,
        "_resolve_publish_access",
        lambda _user_id: SimpleNamespace(
            workspace_team_id=None,
            billing_owner_user_id=None,
            priority_processing=False,
            workspace_role="owner",
            tier="pro",
        ),
    )
    monkeypatch.setattr(
        publishing_router,
        "_load_workspace_clip",
        lambda **_kwargs: {
            "id": "clip-1",
            "video_id": "video-1",
            "title": "Clip title",
            "duration_seconds": clip_duration_seconds,
            "status": "completed",
            "storage_path": "clips/clip-1.mp4",
        },
    )
    monkeypatch.setattr(
        publishing_router,
        "_load_social_accounts_for_workspace",
        lambda **_kwargs: accounts,
    )
    monkeypatch.setattr(
        publishing_router,
        "_create_or_load_batch",
        lambda **_kwargs: (
            {
                "id": "batch-1",
                "clip_id": "clip-1",
                "publish_mode": "schedule",
                "scheduled_timezone": "UTC",
                "caption": "Caption",
                "youtube_title": None,
                "scheduled_for": _future_timestamp(),
            },
            True,
        ),
    )
    monkeypatch.setattr(publishing_router, "_load_batch_publications", lambda _batch_id: [])
    monkeypatch.setattr(
        publishing_router,
        "_assert_no_duplicate_active_publications",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        publishing_router,
        "_ensure_batch_publications",
        lambda **_kwargs: publications,
    )
    monkeypatch.setattr(
        publishing_router,
        "_ensure_dispatch_jobs_for_publications",
        lambda **_kwargs: None,
    )


def test_create_clip_publications_allows_long_facebook_page_requests(client, monkeypatch):
    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(
        id="user-id",
        email=None,
        claims={},
    )
    _mock_create_publish_flow(
        monkeypatch,
        clip_duration_seconds=61,
        accounts=[
            {
                "id": "account-1",
                "provider": "facebook_page",
                "status": "active",
            }
        ],
        publications=[
            {
                "id": "publication-1",
                "batch_id": "batch-1",
                "clip_id": "clip-1",
                "social_account_id": "account-1",
                "provider": "facebook_page",
                "status": "scheduled",
                "scheduled_for": _future_timestamp(),
                "scheduled_timezone": "UTC",
                "caption_snapshot": "Caption",
                "youtube_title_snapshot": None,
                "remote_post_id": None,
                "remote_post_url": None,
                "last_error": None,
                "attempt_count": 0,
                "queued_at": None,
                "started_at": None,
                "published_at": None,
                "failed_at": None,
                "canceled_at": None,
                "created_at": _future_timestamp(),
                "social_account": {"display_name": "Page One"},
            }
        ],
    )

    response = client.post(
        "/publishing",
        json={
            "clipId": "clip-1",
            "socialAccountIds": ["account-1"],
            "caption": "Caption",
            "mode": "schedule",
            "scheduledAt": _future_timestamp(),
            "timeZone": "UTC",
            "clientRequestId": "request-1",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["publications"][0]["provider"] == "facebook_page"
    assert payload["publications"][0]["accountDisplayName"] == "Page One"


def test_create_clip_publications_allows_non_facebook_longer_clips(client, monkeypatch):
    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(
        id="user-id",
        email=None,
        claims={},
    )
    _mock_create_publish_flow(
        monkeypatch,
        clip_duration_seconds=120,
        accounts=[
            {
                "id": "account-2",
                "provider": "instagram_business",
                "status": "active",
            }
        ],
        publications=[
            {
                "id": "publication-2",
                "batch_id": "batch-1",
                "clip_id": "clip-1",
                "social_account_id": "account-2",
                "provider": "instagram_business",
                "status": "scheduled",
                "scheduled_for": _future_timestamp(),
                "scheduled_timezone": "UTC",
                "caption_snapshot": "Caption",
                "youtube_title_snapshot": None,
                "remote_post_id": None,
                "remote_post_url": None,
                "last_error": None,
                "attempt_count": 0,
                "queued_at": None,
                "started_at": None,
                "published_at": None,
                "failed_at": None,
                "canceled_at": None,
                "created_at": _future_timestamp(),
                "social_account": {"display_name": "IG One"},
            }
        ],
    )

    response = client.post(
        "/publishing",
        json={
            "clipId": "clip-1",
            "socialAccountIds": ["account-2"],
            "caption": "Caption",
            "mode": "schedule",
            "scheduledAt": _future_timestamp(),
            "timeZone": "UTC",
            "clientRequestId": "request-2",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["publications"][0]["provider"] == "instagram_business"
    assert payload["publications"][0]["accountDisplayName"] == "IG One"


def test_facebook_publish_falls_back_to_page_video_for_longer_clips(tmp_path, monkeypatch):
    def _fake_post(url: str, **kwargs):
        assert url.endswith("/page-1/videos")
        return _FakeResponse(payload={"id": "video-123"})

    monkeypatch.setattr(meta_facebook.httpx, "post", _fake_post)

    result = meta_facebook.publish_video(
        account=_facebook_account(),
        publication=_publication_context(),
        media=_media(tmp_path, duration_seconds=61.0),
    )

    assert result.remote_post_id == "video-123"
    assert result.result_payload["publish_target"] == "facebook_page_video"


def test_facebook_publish_falls_back_to_page_video_for_non_vertical_clips(tmp_path, monkeypatch):
    def _fake_post(url: str, **kwargs):
        assert url.endswith("/page-1/videos")
        return _FakeResponse(payload={"id": "video-456"})

    monkeypatch.setattr(meta_facebook.httpx, "post", _fake_post)

    result = meta_facebook.publish_video(
        account=_facebook_account(),
        publication=_publication_context(),
        media=_media(tmp_path, width=1080, height=1080),
    )

    assert result.remote_post_id == "video-456"
    assert result.result_payload["publish_target"] == "facebook_page_video"


def test_facebook_reels_publish_happy_path(tmp_path, monkeypatch):
    media = _media(tmp_path)
    post_calls: list[tuple[str, dict]] = []
    status_payloads = iter(
        [
            _FakeResponse(
                payload={
                    "status": {
                        "uploading_phase": {"status": "complete"},
                        "processing_phase": {"status": "not_started"},
                        "publishing_phase": {"status": "not_started"},
                        "video_status": "uploaded",
                    }
                }
            ),
            _FakeResponse(
                payload={
                    "status": {
                        "uploading_phase": {"status": "complete"},
                        "processing_phase": {"status": "complete"},
                        "publishing_phase": {"status": "complete"},
                        "video_status": "published",
                    }
                }
            ),
        ]
    )

    def _fake_post(url: str, **kwargs):
        post_calls.append((url, kwargs))
        if url.endswith("/me/video_reels"):
            phase = (kwargs.get("params") or {}).get("upload_phase") or (kwargs.get("data") or {}).get(
                "upload_phase"
            )
            if phase == "start":
                return _FakeResponse(payload={"video_id": "video-123", "upload_url": "https://upload.example.com"})
            if phase == "finish":
                return _FakeResponse(payload={"success": True})
        if url == "https://upload.example.com":
            assert kwargs["headers"]["Authorization"] == "OAuth page-token"
            assert kwargs["headers"]["offset"] == "0"
            assert kwargs["headers"]["file_size"] == str(media.file_size)
            assert kwargs["content"] == b"clip-bytes"
            return _FakeResponse(payload={"success": True})
        raise AssertionError(f"Unexpected POST URL: {url}")

    def _fake_get(url: str, **kwargs):
        assert url.endswith("/video-123")
        fields = (kwargs.get("params") or {}).get("fields")
        if fields == "status":
            return next(status_payloads)
        if fields == "permalink_url":
            return _FakeResponse(payload={"permalink_url": "https://www.facebook.com/reel/video-123"})
        raise AssertionError(f"Unexpected GET fields: {fields}")

    monkeypatch.setattr(meta_facebook.httpx, "post", _fake_post)
    monkeypatch.setattr(meta_facebook.httpx, "get", _fake_get)
    monkeypatch.setattr(meta_facebook.time, "sleep", lambda _seconds: None)

    result = meta_facebook.publish_video(
        account=_facebook_account(),
        publication=_publication_context(),
        media=media,
    )

    assert result.remote_post_id == "video-123"
    assert result.remote_post_url == "https://www.facebook.com/reel/video-123"
    assert result.result_payload == {
        "platform": "facebook_page",
        "video_id": "video-123",
        "publish_target": "facebook_reels",
    }
    assert len(post_calls) == 3


def test_facebook_reels_publish_marks_oauth_errors_for_reconnect(tmp_path, monkeypatch):
    def _fake_post(url: str, **kwargs):
        if url.endswith("/me/video_reels"):
            return _FakeResponse(
                status_code=400,
                payload={
                    "error": {
                        "code": 190,
                        "type": "OAuthException",
                        "message": "Token expired.",
                    }
                },
            )
        raise AssertionError(f"Unexpected POST URL: {url}")

    monkeypatch.setattr(meta_facebook.httpx, "post", _fake_post)

    with pytest.raises(SocialProviderError) as exc:
        meta_facebook.publish_video(
            account=_facebook_account(),
            publication=_publication_context(),
            media=_media(tmp_path),
        )

    assert exc.value.code == "facebook_reels_start_failed"
    assert exc.value.refresh_required is True

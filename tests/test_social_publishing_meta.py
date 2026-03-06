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


class _FakeSupabaseResponse:
    def __init__(self, data):
        self.data = data
        self.error = None


class _FakeInsertQuery:
    def __init__(self, store: "_FakePublishingSupabase", table_name: str):
        self.store = store
        self.table_name = table_name
        self.payload = None

    def insert(self, payload):
        self.payload = payload
        return self

    def execute(self):
        if self.table_name == "clip_publish_batches":
            self.store.batch_payload = self.payload
            return _FakeSupabaseResponse([self.payload])
        if self.table_name == "clip_publications":
            self.store.publication_payloads = list(self.payload)
            return _FakeSupabaseResponse(self.payload)
        raise AssertionError(f"Unexpected insert table: {self.table_name}")


class _FakePublishingSupabase:
    def __init__(self):
        self.batch_payload = None
        self.publication_payloads = None

    def table(self, name: str):
        return _FakeInsertQuery(self, name)


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


def test_create_clip_publications_rejects_facebook_reels_over_60_seconds(client, monkeypatch):
    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(
        id="user-id",
        email=None,
        claims={},
    )
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
    monkeypatch.setattr(publishing_router, "_load_existing_batch", lambda **_kwargs: (None, []))
    monkeypatch.setattr(
        publishing_router,
        "_load_workspace_clip",
        lambda **_kwargs: {
            "id": "clip-1",
            "video_id": "video-1",
            "title": "Clip title",
            "duration_seconds": 61,
            "status": "completed",
            "storage_path": "clips/clip-1.mp4",
        },
    )
    monkeypatch.setattr(
        publishing_router,
        "_load_social_accounts_for_workspace",
        lambda **_kwargs: [
            {
                "id": "account-1",
                "provider": "facebook_page",
                "status": "active",
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

    assert response.status_code == 400
    assert "60 seconds or shorter" in response.json()["detail"]


def test_create_clip_publications_allows_non_facebook_longer_clips(client, monkeypatch):
    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(
        id="user-id",
        email=None,
        claims={},
    )
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
    monkeypatch.setattr(publishing_router, "_load_existing_batch", lambda **_kwargs: (None, []))
    monkeypatch.setattr(
        publishing_router,
        "_load_workspace_clip",
        lambda **_kwargs: {
            "id": "clip-1",
            "video_id": "video-1",
            "title": "Clip title",
            "duration_seconds": 120,
            "status": "completed",
            "storage_path": "clips/clip-1.mp4",
        },
    )
    monkeypatch.setattr(
        publishing_router,
        "_load_social_accounts_for_workspace",
        lambda **_kwargs: [
            {
                "id": "account-2",
                "provider": "instagram_business",
                "status": "active",
            }
        ],
    )
    fake_supabase = _FakePublishingSupabase()
    monkeypatch.setattr(publishing_router, "supabase", fake_supabase)

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
    assert fake_supabase.batch_payload is not None
    assert fake_supabase.publication_payloads is not None


def test_facebook_reels_publish_validates_duration(tmp_path):
    with pytest.raises(SocialProviderError) as exc:
        meta_facebook.publish_video(
            account=_facebook_account(),
            publication=_publication_context(),
            media=_media(tmp_path, duration_seconds=61.0),
        )

    assert exc.value.code == "facebook_reels_duration_exceeded"
    assert exc.value.refresh_required is False


def test_facebook_reels_publish_validates_aspect_ratio(tmp_path):
    with pytest.raises(SocialProviderError) as exc:
        meta_facebook.publish_video(
            account=_facebook_account(),
            publication=_publication_context(),
            media=_media(tmp_path, width=1080, height=1080),
        )

    assert exc.value.code == "facebook_reels_invalid_aspect_ratio"
    assert exc.value.refresh_required is False


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

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone

import pytest

from services.social import meta_instagram
from services.social.base import (
    PublicationContext,
    PublicationMedia,
    PublicationResult,
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
        return self._payload


def _instagram_account() -> SocialAccountContext:
    return SocialAccountContext(
        id="account-1",
        provider="instagram_business",
        external_account_id="account-1",
        display_name="IG One",
        handle="clipscut",
        provider_metadata={},
        scopes=[],
        tokens=SocialAccountTokens(access_token="ig-token"),
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


def _media(tmp_path) -> PublicationMedia:
    clip_path = tmp_path / "publication.mp4"
    clip_path.write_bytes(b"clip-bytes")
    return PublicationMedia(
        local_path=str(clip_path),
        file_size=clip_path.stat().st_size,
        content_type="video/mp4",
        signed_url=None,
        width=1080,
        height=1920,
        duration_seconds=45.0,
    )


def test_instagram_reels_publish_uses_video_url_container_flow(tmp_path, monkeypatch):
    media = _media(tmp_path)
    media.signed_url = "https://api.clipscut.pro/media/clips/clip-1?sig=test"
    request_calls: list[tuple[str, str, dict]] = []
    status_payloads = iter(
        [
            _FakeResponse(payload={"status_code": "IN_PROGRESS"}),
            _FakeResponse(payload={"status_code": "FINISHED"}),
        ]
    )

    def _fake_resilient_request(method: str, url: str, **kwargs):
        request_calls.append((method, url, kwargs))
        if url.endswith("/account-1/media"):
            assert kwargs["data"]["media_type"] == "REELS"
            assert kwargs["data"]["video_url"] == media.signed_url
            assert "upload_type" not in kwargs["data"]
            return _FakeResponse(payload={"id": "creation-123"})
        if url.endswith("/account-1/media_publish"):
            assert kwargs["data"]["creation_id"] == "creation-123"
            return _FakeResponse(payload={"id": "media-123"})
        if url.endswith("/media-123"):
            return _FakeResponse(payload={"permalink": "https://www.instagram.com/reel/media-123/"})
        raise AssertionError(f"Unexpected request: {method} {url}")

    def _fake_resilient_client_request(_client, method: str, url: str, **kwargs):
        assert method == "GET"
        assert url.endswith("/creation-123")
        assert kwargs["params"]["fields"] == "status_code,status"
        return next(status_payloads)

    @contextmanager
    def _fake_pooled_client(**_kwargs):
        yield object()

    monkeypatch.setattr(meta_instagram, "resilient_request", _fake_resilient_request)
    monkeypatch.setattr(meta_instagram, "resilient_client_request", _fake_resilient_client_request)
    monkeypatch.setattr(meta_instagram, "pooled_client", _fake_pooled_client)
    monkeypatch.setattr(meta_instagram.time, "sleep", lambda _seconds: None)

    result = meta_instagram.publish_reel(
        account=_instagram_account(),
        publication=_publication_context(),
        media=media,
    )

    assert isinstance(result, PublicationResult)
    assert result.remote_post_id == "media-123"
    assert result.remote_post_url == "https://www.instagram.com/reel/media-123/"
    assert result.provider_payload["upload_mode"] == "video_url"
    assert result.provider_payload["media_url_host"] == "api.clipscut.pro"
    assert request_calls[0][1].endswith("/account-1/media")


def test_instagram_reels_requires_public_media_url(tmp_path):
    media = _media(tmp_path)

    with pytest.raises(SocialProviderError) as exc:
        meta_instagram.publish_reel(
            account=_instagram_account(),
            publication=_publication_context(),
            media=media,
        )

    assert exc.value.code == "instagram_public_media_url_unavailable"
    assert exc.value.recoverable is True
    assert exc.value.provider_payload["media_url_host"] is None
    assert exc.value.provider_payload["has_local_file"] is True


def test_instagram_container_failure_includes_video_url_context(tmp_path, monkeypatch):
    media = _media(tmp_path)
    media.signed_url = "https://storage.clipscut.pro/generated-clips/clips/clip-1.mp4?sig=test"
    status_payload = {
        "status_code": "ERROR",
        "error_message": "Failed to fetch the video from the provided URL.",
    }

    def _fake_resilient_request(method: str, url: str, **kwargs):
        if url.endswith("/account-1/media"):
            return _FakeResponse(payload={"id": "creation-123"})
        raise AssertionError(f"Unexpected request: {method} {url}")

    def _fake_resilient_client_request(_client, method: str, url: str, **kwargs):
        assert method == "GET"
        assert url.endswith("/creation-123")
        return _FakeResponse(payload=status_payload)

    @contextmanager
    def _fake_pooled_client(**_kwargs):
        yield object()

    monkeypatch.setattr(meta_instagram, "resilient_request", _fake_resilient_request)
    monkeypatch.setattr(meta_instagram, "resilient_client_request", _fake_resilient_client_request)
    monkeypatch.setattr(meta_instagram, "pooled_client", _fake_pooled_client)
    monkeypatch.setattr(meta_instagram.time, "sleep", lambda _seconds: None)

    with pytest.raises(SocialProviderError) as exc:
        meta_instagram.publish_reel(
            account=_instagram_account(),
            publication=_publication_context(),
            media=media,
        )

    assert exc.value.code == "instagram_container_failed"
    assert "Failed to fetch the video from the provided URL." in str(exc.value)
    assert exc.value.provider_payload["container"]["id"] == "creation-123"
    assert exc.value.provider_payload["upload"]["source"] == "video_url"
    assert exc.value.provider_payload["upload_mode"] == "video_url"
    assert exc.value.provider_payload["media_url_host"] == "storage.clipscut.pro"

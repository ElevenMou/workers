from __future__ import annotations

from types import SimpleNamespace

import api_app.routers.publishing as publishing_router
import pytest
from api_app.auth import AuthenticatedUser
from fastapi import HTTPException


class _FakeSupabaseResponse:
    def __init__(self, data):
        self.data = data
        self.error = None


class _FakeClipPublicationQuery:
    def __init__(self, row: dict):
        self.row = row
        self.operation = "select"
        self.update_payload: dict | None = None

    def select(self, _fields: str):
        self.operation = "select"
        return self

    def update(self, payload: dict):
        self.operation = "update"
        self.update_payload = payload
        return self

    def eq(self, _field: str, _value):
        return self

    def limit(self, _value: int):
        return self

    def execute(self):
        if self.operation == "select":
            return _FakeSupabaseResponse([dict(self.row)])
        self.row.update(self.update_payload or {})
        return _FakeSupabaseResponse([dict(self.row)])


class _FakePublishingSupabase:
    def __init__(self, row: dict):
        self.clip_publications = _FakeClipPublicationQuery(row)

    def table(self, name: str):
        if name != "clip_publications":
            raise AssertionError(f"Unexpected table: {name}")
        return self.clip_publications


def _current_user() -> AuthenticatedUser:
    return AuthenticatedUser(id="user-1", email=None, claims={})


def _access_context() -> SimpleNamespace:
    return SimpleNamespace(
        workspace_team_id=None,
        billing_owner_user_id=None,
        priority_processing=False,
        workspace_role="owner",
        tier="pro",
    )


def test_cancel_clip_publication_allows_queued_and_clears_error(monkeypatch):
    publication_row = {
        "id": "publication-1",
        "batch_id": "batch-1",
        "clip_id": "clip-1",
        "social_account_id": "account-1",
        "provider": "instagram_business",
        "status": "queued",
        "scheduled_for": "2026-03-16T12:00:00+00:00",
        "scheduled_timezone": "UTC",
        "caption_snapshot": "Caption",
        "youtube_title_snapshot": None,
        "last_error": "queue was full",
        "attempt_count": 1,
        "queued_at": "2026-03-16T11:55:00+00:00",
        "started_at": None,
        "published_at": None,
        "failed_at": None,
        "canceled_at": None,
        "created_at": "2026-03-16T11:50:00+00:00",
        "social_account": {"display_name": "IG One"},
    }
    monkeypatch.setattr(publishing_router, "supabase", _FakePublishingSupabase(publication_row))
    monkeypatch.setattr(publishing_router, "_resolve_publish_access", lambda _user_id: _access_context())
    monkeypatch.setattr(
        publishing_router,
        "_load_workspace_clip",
        lambda **_kwargs: {"id": "clip-1", "status": "completed", "storage_path": "clips/clip-1.mp4"},
    )
    monkeypatch.setattr(publishing_router, "_assert_clip_storage_ready_for_publish", lambda _clip: None)

    response = publishing_router.cancel_clip_publication("publication-1", current_user=_current_user())

    assert response.publication.status == "canceled"
    assert response.publication.lastError is None
    assert response.publication.accountDisplayName == "IG One"


def test_retry_clip_publication_clears_stale_publish_artifacts(monkeypatch):
    publication_row = {
        "id": "publication-2",
        "batch_id": "batch-2",
        "clip_id": "clip-2",
        "social_account_id": "account-2",
        "provider": "facebook_page",
        "status": "failed",
        "scheduled_for": "2026-03-16T12:00:00+00:00",
        "scheduled_timezone": "UTC",
        "caption_snapshot": "Caption",
        "youtube_title_snapshot": None,
        "remote_post_id": "remote-1",
        "remote_post_url": "https://example.com/post/1",
        "provider_payload": {"upload": {"id": "upload-1"}},
        "result_payload": {"publish_target": "facebook_reels"},
        "last_error": "provider failed",
        "attempt_count": 2,
        "queued_at": "2026-03-16T11:40:00+00:00",
        "started_at": "2026-03-16T11:41:00+00:00",
        "published_at": "2026-03-16T11:42:00+00:00",
        "failed_at": "2026-03-16T11:43:00+00:00",
        "canceled_at": "2026-03-16T11:44:00+00:00",
        "created_at": "2026-03-16T11:30:00+00:00",
        "social_account": {"display_name": "Page One"},
    }
    enqueue_calls: list[dict] = []
    monkeypatch.setattr(publishing_router, "supabase", _FakePublishingSupabase(publication_row))
    monkeypatch.setattr(publishing_router, "_resolve_publish_access", lambda _user_id: _access_context())
    monkeypatch.setattr(
        publishing_router,
        "_load_workspace_clip",
        lambda **_kwargs: {
            "id": "clip-2",
            "video_id": "video-2",
            "status": "completed",
            "storage_path": "clips/clip-2.mp4",
        },
    )
    monkeypatch.setattr(
        publishing_router,
        "_enqueue_publication_job",
        lambda **kwargs: enqueue_calls.append(kwargs),
    )
    monkeypatch.setattr(publishing_router, "_assert_clip_storage_ready_for_publish", lambda _clip: None)

    response = publishing_router.retry_clip_publication("publication-2", current_user=_current_user())

    assert response.publication.status == "queued"
    assert response.publication.remotePostId is None
    assert response.publication.remotePostUrl is None
    assert response.publication.lastError is None
    assert response.publication.publishedAt is None
    assert response.publication.failedAt is None
    assert response.publication.canceledAt is None
    assert enqueue_calls
    queued_publication = enqueue_calls[0]["publication"]
    assert queued_publication["provider_payload"] == {}
    assert queued_publication["result_payload"] == {}


def test_retry_clip_publication_rejects_missing_clip_asset(monkeypatch):
    publication_row = {
        "id": "publication-3",
        "batch_id": "batch-3",
        "clip_id": "clip-3",
        "social_account_id": "account-3",
        "provider": "instagram_business",
        "status": "failed",
        "scheduled_for": "2026-03-16T12:00:00+00:00",
        "scheduled_timezone": "UTC",
        "caption_snapshot": "Caption",
        "youtube_title_snapshot": None,
        "remote_post_id": None,
        "remote_post_url": None,
        "provider_payload": {},
        "result_payload": {},
        "last_error": "materialize failed",
        "attempt_count": 1,
        "queued_at": None,
        "started_at": None,
        "published_at": None,
        "failed_at": "2026-03-16T11:43:00+00:00",
        "canceled_at": None,
        "created_at": "2026-03-16T11:30:00+00:00",
        "social_account": {"display_name": "IG Two"},
    }
    monkeypatch.setattr(publishing_router, "supabase", _FakePublishingSupabase(publication_row))
    monkeypatch.setattr(publishing_router, "_resolve_publish_access", lambda _user_id: _access_context())
    monkeypatch.setattr(
        publishing_router,
        "_load_workspace_clip",
        lambda **_kwargs: {
            "id": "clip-3",
            "video_id": "video-3",
            "status": "completed",
            "storage_path": "clips/clip-3.mp4",
        },
    )
    monkeypatch.setattr(
        publishing_router,
        "_assert_clip_storage_ready_for_publish",
        lambda _clip: (_ for _ in ()).throw(
            HTTPException(
                status_code=409,
                detail="Clip asset is missing from storage (generated-clips/clips/clip-3.mp4). Regenerate the clip before publishing.",
            )
        ),
    )

    with pytest.raises(HTTPException) as exc:
        publishing_router.retry_clip_publication("publication-3", current_user=_current_user())

    assert exc.value.status_code == 409
    assert "Regenerate the clip before publishing" in str(exc.value.detail)

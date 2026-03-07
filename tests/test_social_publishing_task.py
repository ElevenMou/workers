from __future__ import annotations

from pathlib import Path

import pytest

from services.social.base import SocialProviderError
from tasks.publishing import publish as publish_task_module


def test_publish_clip_task_fails_disconnected_account_without_provider_call(tmp_path, monkeypatch):
    updates: list[tuple[str, dict]] = []
    provider_calls: list[dict] = []
    media_loads: list[str] = []

    def _fake_create_work_dir(_folder_name: str) -> str:
        work_dir = tmp_path / "publication-workdir"
        work_dir.mkdir(exist_ok=True)
        return str(work_dir)

    monkeypatch.setattr(publish_task_module, "create_work_dir", _fake_create_work_dir)
    monkeypatch.setattr(publish_task_module, "configure_job_scope", lambda **_kwargs: None)
    monkeypatch.setattr(publish_task_module, "update_job_status", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        publish_task_module,
        "_load_publication_row",
        lambda _publication_id: {
            "id": "publication-1",
            "clip_id": "clip-1",
            "social_account_id": "account-1",
            "scheduled_for": "2026-03-07T12:00:00+00:00",
            "caption_snapshot": "Caption",
            "youtube_title_snapshot": None,
        },
    )
    monkeypatch.setattr(
        publish_task_module,
        "_load_clip_row",
        lambda _clip_id: {
            "id": "clip-1",
            "title": "Launch clip",
            "status": "completed",
            "storage_path": "generated-clips/clip-1.mp4",
        },
    )
    monkeypatch.setattr(
        publish_task_module,
        "_load_social_account",
        lambda _account_id: {
            "id": "account-1",
            "provider": "youtube_channel",
            "external_account_id": "external-1",
            "display_name": "Workspace YouTube",
            "handle": "workspace",
            "status": "disconnected",
            "provider_metadata": {},
            "scopes": [],
            "encrypted_access_token": "unused",
            "encrypted_refresh_token": None,
            "token_expires_at": None,
        },
    )
    monkeypatch.setattr(
        publish_task_module,
        "_best_effort_update_publication",
        lambda publication_id, payload: updates.append((publication_id, payload)),
    )

    def _unexpected_media_load(*_args, **_kwargs):
        media_loads.append("called")
        raise AssertionError("load_publication_media should not be called for disconnected accounts")

    def _unexpected_provider_call(**kwargs):
        provider_calls.append(kwargs)
        raise AssertionError("publish_to_provider should not be called for disconnected accounts")

    monkeypatch.setattr(publish_task_module, "load_publication_media", _unexpected_media_load)
    monkeypatch.setattr(publish_task_module, "publish_to_provider", _unexpected_provider_call)

    with pytest.raises(SocialProviderError) as exc:
        publish_task_module.publish_clip_task(
            {
                "jobId": "job-1",
                "publicationId": "publication-1",
                "clipId": "clip-1",
                "userId": "user-1",
            }
        )

    assert exc.value.code == "social_account_disconnected"
    assert str(exc.value) == (
        "The linked social account was disconnected. Reconnect the account before publishing."
    )
    assert provider_calls == []
    assert media_loads == []
    assert updates
    publication_id, payload = updates[-1]
    assert publication_id == "publication-1"
    assert payload["status"] == "failed"
    assert payload["last_error"] == str(exc.value)
    assert payload["result_payload"]["provider_error_code"] == "social_account_disconnected"
    assert not Path(tmp_path / "publication-workdir").exists()

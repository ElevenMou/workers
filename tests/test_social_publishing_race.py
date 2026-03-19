from __future__ import annotations

import importlib
import sys
import threading
import types
from datetime import datetime, timezone
from pathlib import Path

from services.social.base import (
    PublicationContext,
    PublicationResult,
    SocialAccountContext,
    SocialAccountTokens,
)


def _install_cryptography_stub() -> None:
    if "cryptography.hazmat.primitives.ciphers.aead" in sys.modules:
        return

    cryptography_module = sys.modules.setdefault("cryptography", types.ModuleType("cryptography"))
    hazmat_module = sys.modules.setdefault(
        "cryptography.hazmat",
        types.ModuleType("cryptography.hazmat"),
    )
    primitives_module = sys.modules.setdefault(
        "cryptography.hazmat.primitives",
        types.ModuleType("cryptography.hazmat.primitives"),
    )
    ciphers_module = sys.modules.setdefault(
        "cryptography.hazmat.primitives.ciphers",
        types.ModuleType("cryptography.hazmat.primitives.ciphers"),
    )
    aead_module = types.ModuleType("cryptography.hazmat.primitives.ciphers.aead")

    class _AESGCM:
        def __init__(self, *_args, **_kwargs):
            return

        def encrypt(self, _nonce, data, _aad):
            return data

        def decrypt(self, _nonce, data, _aad):
            return data

    aead_module.AESGCM = _AESGCM
    sys.modules["cryptography.hazmat.primitives.ciphers.aead"] = aead_module

    cryptography_module.hazmat = hazmat_module
    hazmat_module.primitives = primitives_module
    primitives_module.ciphers = ciphers_module
    ciphers_module.aead = aead_module


def _load_publish_task_module():
    try:
        return importlib.import_module("tasks.publishing.publish")
    except ModuleNotFoundError as exc:
        if not str(exc.name or "").startswith("cryptography"):
            raise
        _install_cryptography_stub()
        sys.modules.pop("tasks.publishing.publish", None)
        return importlib.import_module("tasks.publishing.publish")


class _FakeResponse:
    def __init__(self, data=None):
        self.data = data


class _FakeTable:
    def __init__(self, name: str, *, state: dict, state_lock: threading.Lock):
        self.name = name
        self.state = state
        self.state_lock = state_lock
        self.payload: dict | None = None
        self.filters: list[tuple[str, object]] = []

    def update(self, payload: dict):
        self.payload = dict(payload)
        return self

    def eq(self, field: str, value: object):
        self.filters.append((field, value))
        return self

    def is_(self, field: str, value: object):
        self.filters.append((field, None if value == "null" else value))
        return self

    def execute(self):
        with self.state_lock:
            if self.name == "clip_publications":
                publication_id = next(value for key, value in self.filters if key == "id")
                publication = self.state["publications"][str(publication_id)]
                expected_status = next(
                    (value for key, value in self.filters if key == "status"),
                    None,
                )
                expects_not_canceled = any(
                    key == "canceled_at" and value is None
                    for key, value in self.filters
                )
                if expected_status is not None and publication.get("status") != expected_status:
                    return _FakeResponse([])
                if expects_not_canceled and publication.get("canceled_at") is not None:
                    return _FakeResponse([])
                if self.payload:
                    publication.update(self.payload)
                return _FakeResponse([dict(publication)])

            if self.name == "clips":
                clip_id = next(value for key, value in self.filters if key == "id")
                clip = self.state["clip"]
                assert str(clip["id"]) == str(clip_id)
                if self.payload:
                    clip.update(self.payload)
                return _FakeResponse([dict(clip)])

        raise AssertionError(f"Unexpected table {self.name}")


class _FakeSupabase:
    def __init__(self, *, state: dict, state_lock: threading.Lock):
        self.state = state
        self.state_lock = state_lock

    def table(self, name: str):
        return _FakeTable(name, state=self.state, state_lock=self.state_lock)


def _make_work_dir(base_dir: Path, folder_name: str) -> str:
    work_dir = base_dir / folder_name
    work_dir.mkdir(parents=True, exist_ok=True)
    return str(work_dir)


def test_publish_clip_task_allows_parallel_same_clip_publications(tmp_path, monkeypatch):
    publish_task_module = _load_publish_task_module()
    social_media = importlib.import_module("services.social.media")

    shared_clip_path = tmp_path / "shared-cache" / "generated-clips" / "clips" / "clip-1.mp4"
    shared_clip_path.parent.mkdir(parents=True, exist_ok=True)
    shared_clip_path.write_bytes(b"video")

    publication_ids = [f"publication-{index}" for index in range(4)]
    state_lock = threading.Lock()
    state = {
        "clip": {
            "id": "clip-1",
            "video_id": "video-1",
            "title": "Launch clip",
            "status": "completed",
            "storage_path": "clips/clip-1.mp4",
            "delivery_storage_path": "clips/clip-1.mp4",
            "master_storage_path": None,
            "delivery_profile": "social_auto_h264",
            "publish_profile_used": None,
        },
        "publications": {
            publication_id: {
                "id": publication_id,
                "clip_id": "clip-1",
                "social_account_id": "account-1",
                "status": "queued",
                "scheduled_for": "2026-03-19T01:32:45+00:00",
                "caption_snapshot": f"Caption for {publication_id}",
                "youtube_title_snapshot": "Launch clip",
                "attempt_count": 0,
                "canceled_at": None,
            }
            for publication_id in publication_ids
        },
    }

    provider_media_paths: list[str] = []
    provider_publication_ids: list[str] = []
    provider_lock = threading.Lock()
    job_errors: list[Exception] = []
    start_barrier = threading.Barrier(len(publication_ids))

    monkeypatch.setattr(
        social_media,
        "resolve_generated_clip_path",
        lambda *_args, **_kwargs: str(shared_clip_path),
    )
    monkeypatch.setattr(social_media, "probe_media", lambda _path: (1080, 1920, 29.97))
    monkeypatch.setattr(social_media, "create_signed_clip_url", lambda *_args, **_kwargs: None)

    monkeypatch.setattr(
        publish_task_module,
        "create_work_dir",
        lambda folder_name: _make_work_dir(tmp_path, folder_name),
    )
    monkeypatch.setattr(publish_task_module, "configure_job_scope", lambda **_kwargs: None)
    monkeypatch.setattr(publish_task_module, "update_job_status", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(publish_task_module, "assert_response_ok", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        publish_task_module,
        "_load_publication_row",
        lambda publication_id: dict(state["publications"][str(publication_id)]),
    )
    monkeypatch.setattr(
        publish_task_module,
        "_load_clip_row",
        lambda _clip_id: dict(state["clip"]),
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
            "status": "active",
            "provider_metadata": {},
            "scopes": [],
            "encrypted_access_token": "unused",
            "encrypted_refresh_token": None,
            "token_expires_at": None,
        },
    )
    monkeypatch.setattr(
        publish_task_module,
        "_build_social_account_context",
        lambda _account: SocialAccountContext(
            id="account-1",
            provider="youtube_channel",
            external_account_id="external-1",
            display_name="Workspace YouTube",
            handle="workspace",
            provider_metadata={},
            scopes=[],
            tokens=SocialAccountTokens(access_token="token"),
        ),
    )
    monkeypatch.setattr(
        publish_task_module,
        "_build_publication_context",
        lambda publication, _clip: PublicationContext(
            id=str(publication["id"]),
            clip_id="clip-1",
            clip_title="Launch clip",
            caption=str(publication["caption_snapshot"]),
            youtube_title=str(publication["youtube_title_snapshot"]),
            scheduled_for=datetime.now(timezone.utc),
        ),
    )
    monkeypatch.setattr(publish_task_module, "_validate_media_for_provider", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(publish_task_module, "token_is_expired", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        publish_task_module,
        "_persist_social_account_token_update",
        lambda *_args, **_kwargs: None,
    )

    def _best_effort_update_publication(publication_id: str, payload: dict) -> None:
        with state_lock:
            state["publications"][str(publication_id)].update(payload)

    monkeypatch.setattr(
        publish_task_module,
        "_best_effort_update_publication",
        _best_effort_update_publication,
    )
    monkeypatch.setattr(
        publish_task_module,
        "supabase",
        _FakeSupabase(state=state, state_lock=state_lock),
    )

    def _fake_publish_to_provider(*, publication, media, **_kwargs):
        media_path = Path(media.local_path)
        assert media_path.is_file()
        with provider_lock:
            provider_publication_ids.append(publication.id)
            provider_media_paths.append(str(media_path))
        return PublicationResult(remote_post_id=f"remote-{publication.id}")

    monkeypatch.setattr(
        publish_task_module,
        "publish_to_provider",
        _fake_publish_to_provider,
    )

    def _run(publication_id: str):
        try:
            start_barrier.wait(timeout=2)
            publish_task_module.publish_clip_task(
                {
                    "jobId": f"job-{publication_id}",
                    "publicationId": publication_id,
                    "clipId": "clip-1",
                    "userId": "user-1",
                }
            )
        except Exception as exc:  # pragma: no cover - assertion path
            job_errors.append(exc)

    threads = [
        threading.Thread(target=_run, args=(publication_id,), daemon=True)
        for publication_id in publication_ids
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3)

    assert not job_errors
    assert sorted(provider_publication_ids) == sorted(publication_ids)
    assert len(set(provider_media_paths)) == len(publication_ids)
    for publication_id in publication_ids:
        publication = state["publications"][publication_id]
        assert publication["status"] == "published"
        assert publication["remote_post_id"] == f"remote-{publication_id}"

from __future__ import annotations

from dataclasses import dataclass

import api_app.access_rules as access_rules
import api_app.routers.videos as videos_router
from api_app.auth import AuthenticatedUser, get_current_user


@dataclass
class _FakeResponse:
    data: list[dict] | None = None
    error: object | None = None


class _FakeBatchSplitQuery:
    def __init__(self, store: "_FakeBatchSplitSupabase", table_name: str):
        self.store = store
        self.table_name = table_name
        self._mode = "select"
        self._payload: dict | None = None
        self._eq_filters: dict[str, object] = {}
        self._in_filters: dict[str, set[object]] = {}
        self._is_null_filters: dict[str, bool] = {}
        self._limit: int | None = None

    def select(self, *_args, **_kwargs):
        self._mode = "select"
        self._eq_filters = {}
        self._in_filters = {}
        self._is_null_filters = {}
        self._limit = None
        return self

    def upsert(self, payload, **_kwargs):
        self._mode = "upsert"
        self._payload = payload
        return self

    def eq(self, key, value):
        self._eq_filters[str(key)] = value
        return self

    def in_(self, key, values):
        self._in_filters[str(key)] = set(values)
        return self

    def is_(self, key, value):
        self._is_null_filters[str(key)] = value == "null"
        return self

    def order(self, *_args, **_kwargs):
        return self

    def limit(self, value, *_args, **_kwargs):
        try:
            self._limit = int(value)
        except (TypeError, ValueError):
            self._limit = None
        return self

    def execute(self):
        if self._mode == "select":
          rows = list(self.store.tables[self.table_name])
          for key, value in self._eq_filters.items():
              rows = [row for row in rows if row.get(key) == value]
          for key, values in self._in_filters.items():
              rows = [row for row in rows if row.get(key) in values]
          for key, expect_null in self._is_null_filters.items():
              rows = [row for row in rows if (row.get(key) is None) == expect_null]
          if self._limit is not None:
              rows = rows[: self._limit]
          return _FakeResponse(data=rows)

        if self.table_name == "videos" and self._payload is not None:
            self.store.video_upserts.append(self._payload)
            existing_index = next(
                (
                    index
                    for index, row in enumerate(self.store.tables["videos"])
                    if row.get("id") == self._payload.get("id")
                ),
                None,
            )
            if existing_index is None:
                self.store.tables["videos"].append(dict(self._payload))
            else:
                updated = dict(self.store.tables["videos"][existing_index])
                updated.update(self._payload)
                self.store.tables["videos"][existing_index] = updated
        elif self.table_name == "jobs" and self._payload is not None:
            self.store.job_upserts.append(self._payload)
            self.store.tables["jobs"].append(dict(self._payload))

        return _FakeResponse(data=[self._payload] if self._payload else [])


class _FakeBatchSplitSupabase:
    def __init__(self, *, videos: list[dict] | None = None, jobs: list[dict] | None = None):
        self.tables = {
            "videos": list(videos or []),
            "jobs": list(jobs or []),
        }
        self.video_upserts: list[dict] = []
        self.job_upserts: list[dict] = []

    def table(self, name: str):
        if name not in self.tables:
            raise AssertionError(f"Unexpected table access: {name}")
        return _FakeBatchSplitQuery(self, name)


def _basic_access_context(max_clip_duration_seconds: int = 120):
    return access_rules.UserAccessContext(
        tier="basic",
        status="active",
        interval="month",
        max_videos_per_month=60,
        max_clip_duration_seconds=max_clip_duration_seconds,
        max_analysis_duration_seconds=60 * 60,
        allow_custom_clips=True,
        max_active_jobs=2,
        priority_processing=False,
        clip_retention_days=30,
    )


def test_batch_split_video_requires_exactly_one_source(client, monkeypatch):
    from api_app.app import app

    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(
        id="user-id",
        email=None,
        claims={},
    )
    monkeypatch.setattr(videos_router, "enforce_processing_access_rules", lambda *_a, **_k: _basic_access_context())

    response = client.post(
        "/videos/batch-split",
        json={
            "videoId": "video-1",
            "url": "https://www.youtube.com/watch?v=abc123",
            "segmentLengthSeconds": 60,
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Provide exactly one of videoId or url."


def test_batch_split_video_rejects_segment_length_over_plan_limit(client, monkeypatch):
    from api_app.app import app

    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(
        id="user-id",
        email=None,
        claims={},
    )
    monkeypatch.setattr(
        videos_router,
        "enforce_processing_access_rules",
        lambda *_a, **_k: _basic_access_context(max_clip_duration_seconds=60),
    )

    response = client.post(
        "/videos/batch-split",
        json={
            "url": "https://www.youtube.com/watch?v=abc123",
            "segmentLengthSeconds": 90,
        },
    )

    assert response.status_code == 400
    assert "Allowed maximum: 60 seconds" in response.json()["detail"]


def test_batch_split_video_reuses_active_job_for_existing_video(client, monkeypatch):
    from api_app.app import app

    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(
        id="user-id",
        email=None,
        claims={},
    )
    fake_supabase = _FakeBatchSplitSupabase(
        videos=[
            {
                "id": "video-1",
                "user_id": "user-id",
                "team_id": None,
                "url": "https://www.youtube.com/watch?v=abc123",
                "duration_seconds": 180,
                "status": "completed",
            }
        ],
        jobs=[
            {
                "id": "job-existing",
                "video_id": "video-1",
                "user_id": "user-id",
                "team_id": None,
                "type": "split_video",
                "status": "processing",
            }
        ],
    )
    monkeypatch.setattr(videos_router, "supabase", fake_supabase)
    monkeypatch.setattr(videos_router, "enforce_processing_access_rules", lambda *_a, **_k: _basic_access_context())
    monkeypatch.setattr(videos_router, "has_sufficient_credits", lambda **_k: True)

    response = client.post(
        "/videos/batch-split",
        json={"videoId": "video-1", "segmentLengthSeconds": 60},
    )

    assert response.status_code == 200
    assert response.json() == {
        "jobId": "job-existing",
        "videoId": "video-1",
        "status": "processing",
    }


def test_batch_split_video_queues_new_job_from_url(client, monkeypatch):
    from api_app.app import app

    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(
        id="user-id",
        email=None,
        claims={},
    )
    fake_supabase = _FakeBatchSplitSupabase()
    enqueued_payload: dict | None = None
    enqueued_timeout_seconds: int | None = None

    def _fake_enqueue_or_fail(*, job_data, **_kwargs):
        nonlocal enqueued_payload
        nonlocal enqueued_timeout_seconds
        enqueued_payload = dict(job_data)
        enqueued_timeout_seconds = _kwargs.get("job_timeout_seconds")

    monkeypatch.setattr(videos_router, "supabase", fake_supabase)
    monkeypatch.setattr(videos_router, "enforce_processing_access_rules", lambda *_a, **_k: _basic_access_context())
    monkeypatch.setattr(videos_router, "has_sufficient_credits", lambda **_k: True)
    monkeypatch.setattr(videos_router, "enforce_monthly_video_limit", lambda *_a, **_k: None)
    monkeypatch.setattr(videos_router, "enqueue_or_fail", _fake_enqueue_or_fail)
    monkeypatch.setattr(
        videos_router,
        "_probe_credit_cost_for_url",
        lambda _url: {
            "valid_url": True,
            "duration_seconds": 210,
            "video_title": "Long source",
            "thumbnail_url": "https://image.test/thumb.jpg",
            "platform": "youtube",
            "external_id": "abc123",
            "has_captions": True,
            "has_audio": True,
            "detected_language": "es",
        },
    )

    response = client.post(
        "/videos/batch-split",
        json={
            "url": "https://www.youtube.com/watch?v=abc123",
            "segmentLengthSeconds": 60,
            "layoutId": "layout-1",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "queued"
    assert payload["videoId"]
    assert enqueued_payload is not None
    assert enqueued_payload["segmentLengthSeconds"] == 60
    assert enqueued_payload["expectedPartCount"] == 4
    assert enqueued_payload["expectedGenerationCredits"] == 12
    assert enqueued_payload["sourceDetectedLanguage"] == "es"
    assert enqueued_timeout_seconds == videos_router.VIDEO_JOB_TIMEOUT
    assert fake_supabase.video_upserts[0]["title"] == "Long source"
    assert fake_supabase.job_upserts[0]["type"] == "split_video"


def test_batch_split_video_extends_timeout_for_likely_whisper_fallback(client, monkeypatch):
    from api_app.app import app

    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(
        id="user-id",
        email=None,
        claims={},
    )
    fake_supabase = _FakeBatchSplitSupabase()
    enqueued_timeout_seconds: int | None = None

    def _fake_enqueue_or_fail(**_kwargs):
        nonlocal enqueued_timeout_seconds
        enqueued_timeout_seconds = _kwargs.get("job_timeout_seconds")

    monkeypatch.setattr(videos_router, "supabase", fake_supabase)
    monkeypatch.setattr(videos_router, "enforce_processing_access_rules", lambda *_a, **_k: _basic_access_context())
    monkeypatch.setattr(videos_router, "has_sufficient_credits", lambda **_k: True)
    monkeypatch.setattr(videos_router, "enforce_monthly_video_limit", lambda *_a, **_k: None)
    monkeypatch.setattr(videos_router, "enqueue_or_fail", _fake_enqueue_or_fail)
    monkeypatch.setattr(
        videos_router,
        "_probe_credit_cost_for_url",
        lambda _url: {
            "valid_url": True,
            "duration_seconds": 16 * 60 + 24,
            "video_title": "Audio only source",
            "thumbnail_url": "https://image.test/thumb.jpg",
            "platform": "youtube",
            "external_id": "abc123",
            "has_captions": False,
            "has_audio": True,
            "detected_language": "en",
        },
    )

    response = client.post(
        "/videos/batch-split",
        json={
            "url": "https://www.youtube.com/watch?v=abc123",
            "segmentLengthSeconds": 60,
            "layoutId": "layout-1",
        },
    )

    assert response.status_code == 200
    assert enqueued_timeout_seconds is not None
    assert enqueued_timeout_seconds > videos_router.VIDEO_JOB_TIMEOUT

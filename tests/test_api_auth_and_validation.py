from __future__ import annotations

from dataclasses import dataclass

import api_app.access_rules as access_rules
import api_app.routers.clips as clips_router
import api_app.routers.videos as videos_router
from api_app.auth import AuthenticatedUser, get_current_user


@dataclass
class _FakeResponse:
    data: list[dict] | None = None
    error: object | None = None


class _FakeClipSelectQuery:
    def select(self, *_args, **_kwargs):
        return self

    def eq(self, *_args, **_kwargs):
        return self

    def execute(self):
        return _FakeResponse(
            data=[
                {
                    "id": "clip-1",
                    "video_id": "video-1",
                    "user_id": "owner-user-id",
                }
            ]
        )


class _FakeSupabase:
    def table(self, name: str):
        assert name == "clips"
        return _FakeClipSelectQuery()


class _FakeGenerateClipsQuery:
    def __init__(self, store: "_FakeGenerateSupabase"):
        self.store = store

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, *_args, **_kwargs):
        return self

    def execute(self):
        return _FakeResponse(data=[self.store.clip_row])


class _FakeGenerateJobsQuery:
    def __init__(self, store: "_FakeGenerateSupabase"):
        self.store = store
        self._mode = "select"
        self._upsert_payload = None
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
        self._upsert_payload = payload
        self.store.upsert_payloads.append(payload)
        return self

    def eq(self, key, value):
        self._eq_filters[str(key)] = value
        return self

    def is_(self, key, value):
        self._is_null_filters[str(key)] = value == "null"
        return self

    def in_(self, key, values):
        self._in_filters[str(key)] = set(values)
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
            rows = list(self.store.jobs)

            for key, value in self._eq_filters.items():
                rows = [row for row in rows if row.get(key) == value]

            for key, values in self._in_filters.items():
                rows = [row for row in rows if row.get(key) in values]

            for key, expect_null in self._is_null_filters.items():
                rows = [row for row in rows if (row.get(key) is None) == expect_null]

            if self._limit is not None:
                rows = rows[: self._limit]

            return _FakeResponse(data=rows)
        return _FakeResponse(data=[self._upsert_payload] if self._upsert_payload else [])


class _FakeGenerateSupabase:
    def __init__(
        self,
        *,
        owner_user_id: str,
        active_jobs: list[dict] | None = None,
        prior_jobs: list[dict] | None = None,
        clip_row: dict | None = None,
    ):
        self.owner_user_id = owner_user_id
        self.clip_row = clip_row or {
            "id": "clip-1",
            "video_id": "video-1",
            "user_id": owner_user_id,
            "team_id": None,
            "billing_owner_user_id": owner_user_id,
            "start_time": 0,
            "end_time": 60,
            "status": "pending",
            "storage_path": None,
            "thumbnail_path": None,
            "ai_score": None,
            "transcript_excerpt": None,
        }
        self.jobs = [*(active_jobs or []), *(prior_jobs or [])]
        self.upsert_payloads: list[dict] = []

    def table(self, name: str):
        if name == "clips":
            return _FakeGenerateClipsQuery(self)
        if name == "jobs":
            return _FakeGenerateJobsQuery(self)
        raise AssertionError(f"Unexpected table access: {name}")


class _FakeCustomVideosQuery:
    def __init__(self, store: "_FakeCustomSupabase"):
        self.store = store
        self._mode = "select"
        self._insert_payload = None

    def select(self, *_args, **_kwargs):
        self._mode = "select"
        return self

    def insert(self, payload):
        self._mode = "insert"
        self._insert_payload = payload
        self.store.video_inserts.append(payload)
        return self

    def eq(self, *_args, **_kwargs):
        return self

    def is_(self, *_args, **_kwargs):
        return self

    def limit(self, *_args, **_kwargs):
        return self

    def execute(self):
        if self._mode == "select":
            return _FakeResponse(data=self.store.existing_videos)
        return _FakeResponse(data=[self._insert_payload] if self._insert_payload else [])


class _FakeCustomClipsQuery:
    def __init__(self, store: "_FakeCustomSupabase"):
        self.store = store
        self._insert_payload = None

    def insert(self, payload):
        self._insert_payload = payload
        self.store.clip_inserts.append(payload)
        return self

    def execute(self):
        return _FakeResponse(data=[self._insert_payload] if self._insert_payload else [])


class _FakeCustomJobsQuery:
    def __init__(self, store: "_FakeCustomSupabase"):
        self.store = store
        self._insert_payload = None

    def insert(self, payload):
        self._insert_payload = payload
        self.store.job_inserts.append(payload)
        return self

    def execute(self):
        return _FakeResponse(data=[self._insert_payload] if self._insert_payload else [])


class _FakeCustomSupabase:
    def __init__(self, *, existing_videos: list[dict] | None = None):
        self.existing_videos = existing_videos or []
        self.video_inserts: list[dict] = []
        self.clip_inserts: list[dict] = []
        self.job_inserts: list[dict] = []

    def table(self, name: str):
        if name == "videos":
            return _FakeCustomVideosQuery(self)
        if name == "clips":
            return _FakeCustomClipsQuery(self)
        if name == "jobs":
            return _FakeCustomJobsQuery(self)
        raise AssertionError(f"Unexpected table access: {name}")


def test_unauthenticated_custom_clip_returns_401(client):
    response = client.post(
        "/clips/custom",
        json={
            "url": "https://www.youtube.com/watch?v=FWkVBjcVw18",
            "startTime": 0,
            "endTime": 45,
            "title": "Hello",
        },
    )
    assert response.status_code == 401


def test_clip_layout_options_endpoint_exposes_aspect_and_scale_modes(client):
    response = client.get("/clips/layout-options")
    assert response.status_code == 200
    payload = response.json()
    assert payload["recommendedAspectRatio"] == "9:16"
    assert set(payload["aspectRatios"]) == {"9:16", "1:1", "4:5", "16:9"}
    assert set(payload["videoScaleModes"]) == {"fit", "fill"}


def test_generate_clip_rejects_non_owner(client, monkeypatch):
    from api_app.app import app

    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(
        id="different-user-id",
        email=None,
        claims={},
    )
    monkeypatch.setattr(clips_router, "supabase", _FakeSupabase())
    monkeypatch.setattr(
        clips_router,
        "has_sufficient_credits",
        lambda *, user_id, amount: True,
    )

    response = client.post(
        "/clips/generate",
        json={"clipId": "clip-1"},
    )
    assert response.status_code == 403
    assert response.json()["detail"] == "Clip does not belong to this user"


def test_custom_clip_duration_validation_returns_400(client):
    from api_app.app import app

    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(
        id="user-id",
        email=None,
        claims={},
    )

    response = client.post(
        "/clips/custom",
        json={
            "url": "https://www.youtube.com/watch?v=FWkVBjcVw18",
            "startTime": 0,
            "endTime": 8,
            "title": "Too short",
        },
    )
    assert response.status_code == 400
    assert "at least 10 seconds" in response.json()["detail"]


def test_custom_clip_rejects_free_tier_access(client, monkeypatch):
    from api_app.app import app

    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(
        id="free-user-id",
        email=None,
        claims={},
    )
    monkeypatch.setattr(
        clips_router,
        "enforce_processing_access_rules",
        lambda *_args, **_kwargs: access_rules.UserAccessContext(
            tier="free",
            status="active",
            interval="month",
            max_videos_per_month=10,
            max_clip_duration_seconds=90,
            max_analysis_duration_seconds=1200,
            allow_custom_clips=False,
            max_active_jobs=1,
        ),
    )

    response = client.post(
        "/clips/custom",
        json={
            "url": "https://www.youtube.com/watch?v=FWkVBjcVw18",
            "startTime": 0,
            "endTime": 45,
            "title": "Locked",
        },
    )

    assert response.status_code == 403
    assert "paid plans only" in response.json()["detail"].lower()


def test_credit_cost_endpoint_returns_explicit_fields(client, monkeypatch):
    from api_app.app import app

    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(
        id="user-id",
        email=None,
        claims={},
    )

    def _fake_probe(_self, _url: str):
        return {
            "duration_seconds": 120,
            "can_download": True,
            "has_captions": False,
            "has_audio": True,
        }

    monkeypatch.setattr(videos_router.VideoDownloader, "probe_url", _fake_probe)
    monkeypatch.setattr(videos_router, "whisper_ready", lambda: True)
    monkeypatch.setattr(videos_router, "get_credit_balance", lambda _user_id: 25)
    monkeypatch.setattr(
        videos_router,
        "get_user_access_context",
        lambda *_args, **_kwargs: access_rules.UserAccessContext(
            tier="basic",
            status="active",
            interval="month",
            max_videos_per_month=60,
            max_clip_duration_seconds=90,
            max_analysis_duration_seconds=45 * 60,
            allow_custom_clips=True,
            max_active_jobs=2,
        ),
    )
    monkeypatch.setattr(
        videos_router,
        "has_sufficient_credits",
        lambda *, user_id, amount: True,
    )

    response = client.post(
        "/credits/cost",
        json={
            "url": "https://www.youtube.com/watch?v=FWkVBjcVw18",
            "numClips": 8,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["valid_url"] is True
    assert payload["analysisCredits"] >= 0
    assert payload["totalCredits"] == payload["analysisCredits"]
    assert payload["requestedClipCount"] == 8
    assert payload["clipGenerationCreditsPerClip"] == 0
    assert payload["estimatedGenerationCredits"] == 0
    assert payload["estimatedTotalCredits"] == payload["analysisCredits"]
    assert payload["smartCleanupSurchargePerClip"] == 0
    assert payload["analysisDurationSeconds"] == 120
    assert payload["maxAnalysisDurationSeconds"] is not None
    assert payload["durationLimitExceeded"] is False
    assert payload["currentBalance"] == 25
    assert payload["hasEnoughCredits"] is True
    assert payload["hasEnoughCreditsForEstimatedTotal"] is True


def test_credit_cost_endpoint_waives_smart_cleanup_surcharge_for_pro_tier(client, monkeypatch):
    from api_app.app import app

    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(
        id="user-id",
        email=None,
        claims={},
    )

    def _fake_probe(_self, _url: str):
        return {
            "duration_seconds": 120,
            "can_download": True,
            "has_captions": True,
            "has_audio": True,
        }

    monkeypatch.setattr(videos_router.VideoDownloader, "probe_url", _fake_probe)
    monkeypatch.setattr(videos_router, "whisper_ready", lambda: True)
    monkeypatch.setattr(videos_router, "get_credit_balance", lambda _user_id: 100)
    monkeypatch.setattr(
        videos_router,
        "get_user_access_context",
        lambda *_args, **_kwargs: access_rules.UserAccessContext(
            tier="pro",
            status="active",
            interval="month",
            max_videos_per_month=220,
            max_clip_duration_seconds=90,
            max_analysis_duration_seconds=7200,
            allow_custom_clips=True,
            max_active_jobs=5,
        ),
    )

    response = client.post(
        "/credits/cost",
        json={
            "url": "https://www.youtube.com/watch?v=FWkVBjcVw18",
            "numClips": 4,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["clipGenerationCreditsPerClip"] == 0
    assert payload["smartCleanupSurchargePerClip"] == 0


def test_credit_cost_endpoint_flags_duration_limit_exceeded(client, monkeypatch):
    from api_app.app import app

    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(
        id="user-id",
        email=None,
        claims={},
    )

    def _fake_probe(_self, _url: str):
        return {
            "duration_seconds": 1500,
            "can_download": True,
            "has_captions": True,
            "has_audio": True,
        }

    monkeypatch.setattr(videos_router.VideoDownloader, "probe_url", _fake_probe)
    monkeypatch.setattr(videos_router, "whisper_ready", lambda: True)
    monkeypatch.setattr(videos_router, "get_credit_balance", lambda _user_id: 500)
    monkeypatch.setattr(
        videos_router,
        "get_user_access_context",
        lambda *_args, **_kwargs: access_rules.UserAccessContext(
            tier="free",
            status="active",
            interval="month",
            max_videos_per_month=10,
            max_clip_duration_seconds=90,
            max_analysis_duration_seconds=1200,
            allow_custom_clips=False,
            max_active_jobs=1,
        ),
    )

    response = client.post(
        "/credits/cost",
        json={
            "url": "https://www.youtube.com/watch?v=FWkVBjcVw18",
            "numClips": 6,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["valid_url"] is True
    assert payload["durationLimitExceeded"] is True
    assert payload["maxAnalysisDurationSeconds"] == 1200
    assert payload["hasEnoughCredits"] is False
    assert payload["hasEnoughCreditsForEstimatedTotal"] is False


def test_credit_cost_endpoint_tracks_estimated_total_balance_separately(client, monkeypatch):
    from api_app.app import app

    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(
        id="user-id",
        email=None,
        claims={},
    )

    def _fake_probe(_self, _url: str):
        return {
            "duration_seconds": 120,
            "can_download": True,
            "has_captions": False,
            "has_audio": True,
        }

    monkeypatch.setattr(videos_router.VideoDownloader, "probe_url", _fake_probe)
    monkeypatch.setattr(videos_router, "whisper_ready", lambda: True)
    monkeypatch.setattr(videos_router, "get_credit_balance", lambda _user_id: 12)
    monkeypatch.setattr(
        videos_router,
        "get_user_access_context",
        lambda *_args, **_kwargs: access_rules.UserAccessContext(
            tier="basic",
            status="active",
            interval="month",
            max_videos_per_month=60,
            max_clip_duration_seconds=90,
            max_analysis_duration_seconds=45 * 60,
            allow_custom_clips=True,
            max_active_jobs=2,
        ),
    )

    response = client.post(
        "/credits/cost",
        json={
            "url": "https://www.youtube.com/watch?v=FWkVBjcVw18",
            "numClips": 5,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["analysisCredits"] == 2
    assert payload["estimatedGenerationCredits"] == 0
    assert payload["estimatedTotalCredits"] == 2
    assert payload["hasEnoughCredits"] is True
    assert payload["hasEnoughCreditsForEstimatedTotal"] is True


def test_analyze_video_fails_early_when_no_credits(client, monkeypatch):
    from api_app.app import app

    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(
        id="user-id",
        email=None,
        claims={},
    )

    monkeypatch.setattr(
        videos_router,
        "has_sufficient_credits",
        lambda *, user_id, amount: False,
    )
    monkeypatch.setattr(videos_router, "get_credit_balance", lambda _user_id: 0)

    def _should_not_probe(_url: str):
        raise AssertionError("URL probe should not run when user has no credits")

    monkeypatch.setattr(videos_router, "_probe_credit_cost_for_url", _should_not_probe)

    response = client.post(
        "/videos/analyze",
        json={
            "url": "https://www.youtube.com/watch?v=FWkVBjcVw18",
            "numClips": 5,
        },
    )

    assert response.status_code == 402
    assert "Insufficient credits for analysis" in response.json()["detail"]


def test_analyze_video_rejects_duration_over_plan_cap(client, monkeypatch):
    from api_app.app import app

    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(
        id="user-id",
        email=None,
        claims={},
    )
    monkeypatch.setattr(
        videos_router,
        "has_sufficient_credits",
        lambda *, user_id, amount: True,
    )
    monkeypatch.setattr(
        videos_router,
        "enforce_processing_access_rules",
        lambda *_args, **_kwargs: access_rules.UserAccessContext(
            tier="free",
            status="active",
            interval="month",
            max_videos_per_month=10,
            max_clip_duration_seconds=90,
            max_analysis_duration_seconds=20 * 60,
            allow_custom_clips=False,
            max_active_jobs=1,
        ),
    )
    monkeypatch.setattr(
        videos_router,
        "_probe_credit_cost_for_url",
        lambda _url: {
            "valid_url": True,
            "analysis_credits": 8,
            "duration_seconds": 21 * 60,
        },
    )

    response = client.post(
        "/videos/analyze",
        json={
            "url": "https://www.youtube.com/watch?v=FWkVBjcVw18",
            "numClips": 5,
        },
    )

    assert response.status_code == 400
    assert "20 minutes" in response.json()["detail"].lower()


def test_analyze_video_rejects_clip_count_above_density_ratio(client, monkeypatch):
    from api_app.app import app

    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(
        id="user-id",
        email=None,
        claims={},
    )
    monkeypatch.setattr(
        videos_router,
        "has_sufficient_credits",
        lambda *, user_id, amount: True,
    )
    monkeypatch.setattr(
        videos_router,
        "enforce_processing_access_rules",
        lambda *_args, **_kwargs: access_rules.UserAccessContext(
            tier="basic",
            status="active",
            interval="month",
            max_videos_per_month=60,
            max_clip_duration_seconds=90,
            max_analysis_duration_seconds=60 * 60,
            allow_custom_clips=True,
            max_active_jobs=2,
        ),
    )
    monkeypatch.setattr(
        videos_router,
        "_probe_credit_cost_for_url",
        lambda _url: {
            "valid_url": True,
            "analysis_credits": 8,
            "duration_seconds": 8 * 60,
        },
    )

    response = client.post(
        "/videos/analyze",
        json={
            "url": "https://www.youtube.com/watch?v=FWkVBjcVw18",
            "numClips": 3,
        },
    )

    assert response.status_code == 400
    assert "1 clip every 3 minutes" in response.json()["detail"]


def test_analyze_video_queues_range_based_analysis_credits(client, monkeypatch):
    from api_app.app import app

    class _FakeAnalyzeQuery:
        def __init__(self, store: "_FakeAnalyzeSupabase", table_name: str):
            self.store = store
            self.table_name = table_name
            self._mode = "select"
            self._payload = None

        def select(self, *_args, **_kwargs):
            self._mode = "select"
            return self

        def upsert(self, payload, **_kwargs):
            self._mode = "upsert"
            self._payload = payload
            return self

        def eq(self, *_args, **_kwargs):
            return self

        def is_(self, *_args, **_kwargs):
            return self

        def limit(self, *_args, **_kwargs):
            return self

        def execute(self):
            if self._mode == "select":
                return _FakeResponse(data=[])
            if self.table_name == "videos":
                self.store.video_upserts.append(self._payload)
            elif self.table_name == "jobs":
                self.store.job_upserts.append(self._payload)
            return _FakeResponse(data=[self._payload] if self._payload else [])

    class _FakeAnalyzeSupabase:
        def __init__(self):
            self.video_upserts: list[dict] = []
            self.job_upserts: list[dict] = []

        def table(self, name: str):
            if name in {"videos", "jobs"}:
                return _FakeAnalyzeQuery(self, name)
            raise AssertionError(f"Unexpected table access: {name}")

    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(
        id="user-id",
        email=None,
        claims={},
    )

    checked_amounts: list[int] = []

    def _fake_has_sufficient_credits(*, user_id, amount, **_kwargs):
        assert user_id == "user-id"
        checked_amounts.append(int(amount))
        return True

    enqueued_payload: dict | None = None

    def _fake_enqueue_or_fail(*, job_data, **_kwargs):
        nonlocal enqueued_payload
        enqueued_payload = dict(job_data)

    monkeypatch.setattr(videos_router, "supabase", _FakeAnalyzeSupabase())
    monkeypatch.setattr(videos_router, "raise_on_error", lambda *_a, **_k: None)
    monkeypatch.setattr(videos_router, "enforce_monthly_video_limit", lambda *_a, **_k: None)
    monkeypatch.setattr(videos_router, "enqueue_or_fail", _fake_enqueue_or_fail)
    monkeypatch.setattr(videos_router, "has_sufficient_credits", _fake_has_sufficient_credits)
    monkeypatch.setattr(videos_router, "get_credit_balance", lambda _user_id: 999)
    monkeypatch.setattr(
        videos_router,
        "enforce_processing_access_rules",
        lambda *_args, **_kwargs: access_rules.UserAccessContext(
            tier="basic",
            status="active",
            interval="month",
            max_videos_per_month=60,
            max_clip_duration_seconds=120,
            max_analysis_duration_seconds=60 * 60,
            allow_custom_clips=True,
            max_active_jobs=2,
        ),
    )
    monkeypatch.setattr(
        videos_router,
        "_probe_credit_cost_for_url",
        lambda _url: {
            "valid_url": True,
            "analysis_credits": 10,
            "duration_seconds": 10 * 60,
        },
    )

    response = client.post(
        "/videos/analyze",
        json={
            "url": "https://www.youtube.com/watch?v=FWkVBjcVw18",
            "numClips": 1,
            "processingStartSeconds": 60,
            "processingEndSeconds": 180,
        },
    )

    assert response.status_code == 200
    assert enqueued_payload is not None
    assert enqueued_payload["analysisCredits"] == 2
    assert checked_amounts == [1, 2]
    assert enqueued_payload["clipLengthMinSeconds"] == 10
    assert enqueued_payload["clipLengthMaxSeconds"] == 60


def test_analyze_video_accepts_explicit_clip_length_range(client, monkeypatch):
    from api_app.app import app

    class _FakeAnalyzeQuery:
        def __init__(self, store: "_FakeAnalyzeSupabase", table_name: str):
            self.store = store
            self.table_name = table_name
            self._mode = "select"
            self._payload = None

        def select(self, *_args, **_kwargs):
            self._mode = "select"
            return self

        def upsert(self, payload, **_kwargs):
            self._mode = "upsert"
            self._payload = payload
            return self

        def eq(self, *_args, **_kwargs):
            return self

        def is_(self, *_args, **_kwargs):
            return self

        def limit(self, *_args, **_kwargs):
            return self

        def execute(self):
            if self._mode == "select":
                return _FakeResponse(data=[])
            if self.table_name == "videos":
                self.store.video_upserts.append(self._payload)
            elif self.table_name == "jobs":
                self.store.job_upserts.append(self._payload)
            return _FakeResponse(data=[self._payload] if self._payload else [])

    class _FakeAnalyzeSupabase:
        def __init__(self):
            self.video_upserts: list[dict] = []
            self.job_upserts: list[dict] = []

        def table(self, name: str):
            if name in {"videos", "jobs"}:
                return _FakeAnalyzeQuery(self, name)
            raise AssertionError(f"Unexpected table access: {name}")

    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(
        id="user-id",
        email=None,
        claims={},
    )

    enqueued_payload: dict | None = None

    def _fake_enqueue_or_fail(*, job_data, **_kwargs):
        nonlocal enqueued_payload
        enqueued_payload = dict(job_data)

    monkeypatch.setattr(videos_router, "supabase", _FakeAnalyzeSupabase())
    monkeypatch.setattr(videos_router, "raise_on_error", lambda *_a, **_k: None)
    monkeypatch.setattr(videos_router, "enforce_monthly_video_limit", lambda *_a, **_k: None)
    monkeypatch.setattr(videos_router, "enqueue_or_fail", _fake_enqueue_or_fail)
    monkeypatch.setattr(videos_router, "has_sufficient_credits", lambda **_k: True)
    monkeypatch.setattr(videos_router, "get_credit_balance", lambda _user_id: 999)
    monkeypatch.setattr(
        videos_router,
        "enforce_processing_access_rules",
        lambda *_args, **_kwargs: access_rules.UserAccessContext(
            tier="basic",
            status="active",
            interval="month",
            max_videos_per_month=60,
            max_clip_duration_seconds=120,
            max_analysis_duration_seconds=60 * 60,
            allow_custom_clips=True,
            max_active_jobs=2,
        ),
    )
    monkeypatch.setattr(
        videos_router,
        "_probe_credit_cost_for_url",
        lambda _url: {
            "valid_url": True,
            "analysis_credits": 10,
            "duration_seconds": 10 * 60,
        },
    )

    response = client.post(
        "/videos/analyze",
        json={
            "url": "https://www.youtube.com/watch?v=FWkVBjcVw18",
            "numClips": 1,
            "clipLengthMinSeconds": 60,
            "clipLengthMaxSeconds": 120,
            "processingStartSeconds": 0,
            "processingEndSeconds": 70,
        },
    )

    assert response.status_code == 200
    assert enqueued_payload is not None
    assert enqueued_payload["clipLengthMinSeconds"] == 60
    assert enqueued_payload["clipLengthMaxSeconds"] == 120
    assert enqueued_payload["clipLengthSeconds"] == 120


def test_analyze_video_rejects_unpaired_clip_range_values(client, monkeypatch):
    from api_app.app import app

    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(
        id="user-id",
        email=None,
        claims={},
    )
    monkeypatch.setattr(videos_router, "has_sufficient_credits", lambda **_k: True)
    monkeypatch.setattr(videos_router, "get_credit_balance", lambda _user_id: 999)
    monkeypatch.setattr(
        videos_router,
        "enforce_processing_access_rules",
        lambda *_args, **_kwargs: access_rules.UserAccessContext(
            tier="basic",
            status="active",
            interval="month",
            max_videos_per_month=60,
            max_clip_duration_seconds=120,
            max_analysis_duration_seconds=60 * 60,
            allow_custom_clips=True,
            max_active_jobs=2,
        ),
    )
    monkeypatch.setattr(
        videos_router,
        "_probe_credit_cost_for_url",
        lambda _url: {
            "valid_url": True,
            "analysis_credits": 10,
            "duration_seconds": 10 * 60,
        },
    )

    response = client.post(
        "/videos/analyze",
        json={
            "url": "https://www.youtube.com/watch?v=FWkVBjcVw18",
            "numClips": 1,
            "clipLengthMinSeconds": 60,
        },
    )

    assert response.status_code == 400
    assert "provided together" in response.json()["detail"]


def test_analyze_video_rejects_invalid_clip_range_order(client, monkeypatch):
    from api_app.app import app

    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(
        id="user-id",
        email=None,
        claims={},
    )
    monkeypatch.setattr(videos_router, "has_sufficient_credits", lambda **_k: True)
    monkeypatch.setattr(videos_router, "get_credit_balance", lambda _user_id: 999)
    monkeypatch.setattr(
        videos_router,
        "enforce_processing_access_rules",
        lambda *_args, **_kwargs: access_rules.UserAccessContext(
            tier="basic",
            status="active",
            interval="month",
            max_videos_per_month=60,
            max_clip_duration_seconds=120,
            max_analysis_duration_seconds=60 * 60,
            allow_custom_clips=True,
            max_active_jobs=2,
        ),
    )
    monkeypatch.setattr(
        videos_router,
        "_probe_credit_cost_for_url",
        lambda _url: {
            "valid_url": True,
            "analysis_credits": 10,
            "duration_seconds": 10 * 60,
        },
    )

    response = client.post(
        "/videos/analyze",
        json={
            "url": "https://www.youtube.com/watch?v=FWkVBjcVw18",
            "numClips": 1,
            "clipLengthMinSeconds": 90,
            "clipLengthMaxSeconds": 60,
        },
    )

    assert response.status_code == 400
    assert "cannot be greater" in response.json()["detail"]


def test_analyze_video_rejects_clip_range_above_plan_limit(client, monkeypatch):
    from api_app.app import app

    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(
        id="user-id",
        email=None,
        claims={},
    )
    monkeypatch.setattr(videos_router, "has_sufficient_credits", lambda **_k: True)
    monkeypatch.setattr(videos_router, "get_credit_balance", lambda _user_id: 999)
    monkeypatch.setattr(
        videos_router,
        "enforce_processing_access_rules",
        lambda *_args, **_kwargs: access_rules.UserAccessContext(
            tier="basic",
            status="active",
            interval="month",
            max_videos_per_month=60,
            max_clip_duration_seconds=120,
            max_analysis_duration_seconds=60 * 60,
            allow_custom_clips=True,
            max_active_jobs=2,
        ),
    )
    monkeypatch.setattr(
        videos_router,
        "_probe_credit_cost_for_url",
        lambda _url: {
            "valid_url": True,
            "analysis_credits": 10,
            "duration_seconds": 10 * 60,
        },
    )

    response = client.post(
        "/videos/analyze",
        json={
            "url": "https://www.youtube.com/watch?v=FWkVBjcVw18",
            "numClips": 1,
            "clipLengthMinSeconds": 60,
            "clipLengthMaxSeconds": 180,
        },
    )

    assert response.status_code == 400
    assert "outside your plan limits" in response.json()["detail"]


def test_analyze_video_rejects_window_shorter_than_selected_range_min(client, monkeypatch):
    from api_app.app import app

    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(
        id="user-id",
        email=None,
        claims={},
    )
    monkeypatch.setattr(videos_router, "has_sufficient_credits", lambda **_k: True)
    monkeypatch.setattr(videos_router, "get_credit_balance", lambda _user_id: 999)
    monkeypatch.setattr(
        videos_router,
        "enforce_processing_access_rules",
        lambda *_args, **_kwargs: access_rules.UserAccessContext(
            tier="basic",
            status="active",
            interval="month",
            max_videos_per_month=60,
            max_clip_duration_seconds=120,
            max_analysis_duration_seconds=60 * 60,
            allow_custom_clips=True,
            max_active_jobs=2,
        ),
    )
    monkeypatch.setattr(
        videos_router,
        "_probe_credit_cost_for_url",
        lambda _url: {
            "valid_url": True,
            "analysis_credits": 10,
            "duration_seconds": 10 * 60,
        },
    )

    response = client.post(
        "/videos/analyze",
        json={
            "url": "https://www.youtube.com/watch?v=FWkVBjcVw18",
            "numClips": 1,
            "clipLengthMinSeconds": 60,
            "clipLengthMaxSeconds": 120,
            "processingStartSeconds": 0,
            "processingEndSeconds": 50,
        },
    )

    assert response.status_code == 400
    assert "minimum clip length" in response.json()["detail"]


def test_generate_clip_fails_early_when_no_credits(client, monkeypatch):
    from api_app.app import app

    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(
        id="user-id",
        email=None,
        claims={},
    )
    monkeypatch.setattr(
        clips_router,
        "supabase",
        _FakeGenerateSupabase(
            owner_user_id="user-id",
            clip_row={
                "id": "clip-1",
                "video_id": "video-1",
                "user_id": "user-id",
                "team_id": None,
                "billing_owner_user_id": "user-id",
                "start_time": 0,
                "end_time": 60,
                "status": "pending",
                "storage_path": None,
                "thumbnail_path": None,
                "ai_score": None,
                "transcript_excerpt": None,
            },
        ),
    )
    monkeypatch.setattr(
        clips_router,
        "enforce_processing_access_rules",
        lambda *_args, **_kwargs: access_rules.UserAccessContext(
            tier="basic",
            status="active",
            interval="month",
            max_videos_per_month=60,
            max_clip_duration_seconds=90,
        ),
    )

    monkeypatch.setattr(
        clips_router,
        "has_sufficient_credits",
        lambda *, user_id, amount: False,
    )
    monkeypatch.setattr(clips_router, "get_credit_balance", lambda _user_id: 0)

    response = client.post(
        "/clips/generate",
        json={"clipId": "clip-1"},
    )

    assert response.status_code == 402
    assert "Insufficient credits for clip generation" in response.json()["detail"]


def test_generate_clip_allows_smart_cleanup_for_basic_tier_at_three_credits(client, monkeypatch):
    from api_app.app import app

    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(
        id="user-id",
        email=None,
        claims={},
    )
    monkeypatch.setattr(clips_router, "supabase", _FakeGenerateSupabase(owner_user_id="user-id"))
    monkeypatch.setattr(
        clips_router,
        "enforce_processing_access_rules",
        lambda *_args, **_kwargs: access_rules.UserAccessContext(
            tier="basic",
            status="active",
            interval="month",
            max_videos_per_month=60,
            max_clip_duration_seconds=90,
        ),
    )

    checked_amounts: list[int] = []

    def _fake_has_sufficient_credits(*, user_id, amount, **_kwargs):
        assert user_id == "user-id"
        checked_amounts.append(amount)
        return True

    enqueued_job_data: dict | None = None

    def _fake_enqueue_or_fail(*, job_data, **_kwargs):
        nonlocal enqueued_job_data
        enqueued_job_data = job_data

    monkeypatch.setattr(clips_router, "has_sufficient_credits", _fake_has_sufficient_credits)
    monkeypatch.setattr(clips_router, "enqueue_or_fail", _fake_enqueue_or_fail)

    response = client.post(
        "/clips/generate",
        json={"clipId": "clip-1", "smartCleanupEnabled": True},
    )

    assert response.status_code == 200
    assert checked_amounts[-1] == 3
    assert enqueued_job_data is not None
    assert enqueued_job_data["generationCredits"] == 3
    assert enqueued_job_data["smartCleanupEnabled"] is True


def test_custom_clip_allows_smart_cleanup_for_basic_tier_at_three_credits(client, monkeypatch):
    from api_app.app import app

    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(
        id="user-id",
        email=None,
        claims={},
    )
    fake_supabase = _FakeCustomSupabase()
    monkeypatch.setattr(clips_router, "supabase", fake_supabase)
    monkeypatch.setattr(clips_router, "enforce_monthly_video_limit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        clips_router,
        "enforce_processing_access_rules",
        lambda *_args, **_kwargs: access_rules.UserAccessContext(
            tier="basic",
            status="active",
            interval="month",
            max_videos_per_month=60,
            max_clip_duration_seconds=90,
            allow_custom_clips=True,
        ),
    )

    checked_amounts: list[int] = []

    def _fake_has_sufficient_credits(*, user_id, amount, **_kwargs):
        assert user_id == "user-id"
        checked_amounts.append(amount)
        return True

    enqueued_job_data: dict | None = None

    def _fake_enqueue_or_fail(*, job_data, **_kwargs):
        nonlocal enqueued_job_data
        enqueued_job_data = job_data

    monkeypatch.setattr(clips_router, "has_sufficient_credits", _fake_has_sufficient_credits)
    monkeypatch.setattr(clips_router, "enqueue_or_fail", _fake_enqueue_or_fail)

    response = client.post(
        "/clips/custom",
        json={
            "url": "https://www.youtube.com/watch?v=FWkVBjcVw18",
            "startTime": 0,
            "endTime": 45,
            "title": "Hello",
            "smartCleanupEnabled": True,
        },
    )

    assert response.status_code == 200
    assert checked_amounts[-1] == 3
    assert enqueued_job_data is not None
    assert enqueued_job_data["generationCredits"] == 3
    assert enqueued_job_data["smartCleanupEnabled"] is True


def test_custom_clip_requires_three_credits_without_analysis(client, monkeypatch):
    from api_app.app import app

    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(
        id="user-id",
        email=None,
        claims={},
    )

    fake_supabase = _FakeCustomSupabase()
    monkeypatch.setattr(clips_router, "supabase", fake_supabase)
    monkeypatch.setattr(clips_router, "enforce_monthly_video_limit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        clips_router,
        "enforce_processing_access_rules",
        lambda *_args, **_kwargs: access_rules.UserAccessContext(
            tier="pro",
            status="active",
            interval="month",
            max_videos_per_month=220,
            max_clip_duration_seconds=90,
            allow_custom_clips=True,
            priority_processing=True,
        ),
    )

    checked_amounts: list[int] = []

    def _fake_has_sufficient_credits(*, user_id, amount, **_kwargs):
        assert user_id == "user-id"
        checked_amounts.append(amount)
        return True

    enqueued_job_data: dict | None = None
    enqueued_queue_name: str | None = None

    def _fake_enqueue_or_fail(*, queue_name, job_data, **_kwargs):
        nonlocal enqueued_job_data
        nonlocal enqueued_queue_name
        enqueued_queue_name = queue_name
        enqueued_job_data = job_data

    monkeypatch.setattr(clips_router, "has_sufficient_credits", _fake_has_sufficient_credits)
    monkeypatch.setattr(clips_router, "enqueue_or_fail", _fake_enqueue_or_fail)

    response = client.post(
        "/clips/custom",
        json={
            "url": "https://www.youtube.com/watch?v=FWkVBjcVw18",
            "startTime": 0,
            "endTime": 45,
            "title": "Custom paid clip",
        },
    )

    assert response.status_code == 200
    assert checked_amounts[-1] == 3
    assert enqueued_queue_name == "clip-generation-priority"
    assert enqueued_job_data is not None
    assert enqueued_job_data["generationCredits"] == 3
    assert enqueued_job_data["smartCleanupEnabled"] is False


def test_custom_clip_pro_smart_cleanup_keeps_three_credit_cost(client, monkeypatch):
    from api_app.app import app

    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(
        id="user-id",
        email=None,
        claims={},
    )

    fake_supabase = _FakeCustomSupabase()
    monkeypatch.setattr(clips_router, "supabase", fake_supabase)
    monkeypatch.setattr(clips_router, "enforce_monthly_video_limit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        clips_router,
        "enforce_processing_access_rules",
        lambda *_args, **_kwargs: access_rules.UserAccessContext(
            tier="pro",
            status="active",
            interval="month",
            max_videos_per_month=220,
            max_clip_duration_seconds=90,
            allow_custom_clips=True,
            priority_processing=True,
        ),
    )

    checked_amounts: list[int] = []

    def _fake_has_sufficient_credits(*, user_id, amount, **_kwargs):
        assert user_id == "user-id"
        checked_amounts.append(amount)
        return True

    enqueued_job_data: dict | None = None

    def _fake_enqueue_or_fail(*, job_data, **_kwargs):
        nonlocal enqueued_job_data
        enqueued_job_data = job_data

    monkeypatch.setattr(clips_router, "has_sufficient_credits", _fake_has_sufficient_credits)
    monkeypatch.setattr(clips_router, "enqueue_or_fail", _fake_enqueue_or_fail)

    response = client.post(
        "/clips/custom",
        json={
            "url": "https://www.youtube.com/watch?v=FWkVBjcVw18",
            "startTime": 0,
            "endTime": 45,
            "title": "Custom cleanup clip",
            "smartCleanupEnabled": True,
        },
    )

    assert response.status_code == 200
    assert checked_amounts[-1] == 3
    assert enqueued_job_data is not None
    assert enqueued_job_data["generationCredits"] == 3
    assert enqueued_job_data["smartCleanupEnabled"] is True


def test_ai_suggested_first_generation_is_free_even_with_pro_smart_cleanup(client, monkeypatch):
    from api_app.app import app

    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(
        id="user-id",
        email=None,
        claims={},
    )

    fake_supabase = _FakeGenerateSupabase(
        owner_user_id="user-id",
        clip_row={
            "id": "clip-1",
            "video_id": "video-1",
            "user_id": "user-id",
            "team_id": None,
            "billing_owner_user_id": "user-id",
            "start_time": 0,
            "end_time": 60,
            "status": "pending",
            "storage_path": None,
            "thumbnail_path": None,
            "ai_score": 0.92,
            "transcript_excerpt": "AI suggested excerpt",
        },
    )
    monkeypatch.setattr(clips_router, "supabase", fake_supabase)
    monkeypatch.setattr(
        clips_router,
        "enforce_processing_access_rules",
        lambda *_args, **_kwargs: access_rules.UserAccessContext(
            tier="pro",
            status="active",
            interval="month",
            max_videos_per_month=220,
            max_clip_duration_seconds=90,
            priority_processing=True,
        ),
    )

    checked_amounts: list[int] = []

    def _fake_has_sufficient_credits(*, user_id, amount, **_kwargs):
        assert user_id == "user-id"
        checked_amounts.append(amount)
        return True

    enqueued_job_data: dict | None = None

    def _fake_enqueue_or_fail(*, job_data, **_kwargs):
        nonlocal enqueued_job_data
        enqueued_job_data = job_data

    monkeypatch.setattr(clips_router, "has_sufficient_credits", _fake_has_sufficient_credits)
    monkeypatch.setattr(clips_router, "enqueue_or_fail", _fake_enqueue_or_fail)

    response = client.post(
        "/clips/generate",
        json={"clipId": "clip-1", "smartCleanupEnabled": True},
    )

    assert response.status_code == 200
    assert checked_amounts == []
    assert enqueued_job_data is not None
    assert enqueued_job_data["generationCredits"] == 0
    assert enqueued_job_data["smartCleanupEnabled"] is True


def test_restart_generation_requires_credits(client, monkeypatch):
    from api_app.app import app

    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(
        id="user-id",
        email=None,
        claims={},
    )

    fake_supabase = _FakeGenerateSupabase(
        owner_user_id="user-id",
        clip_row={
            "id": "clip-1",
            "video_id": "video-1",
            "user_id": "user-id",
            "team_id": None,
            "billing_owner_user_id": "user-id",
            "start_time": 0,
            "end_time": 60,
            "status": "completed",
            "storage_path": "clips/clip-1.mp4",
            "thumbnail_path": None,
            "ai_score": 0.86,
            "transcript_excerpt": "AI suggestion",
        },
    )
    monkeypatch.setattr(clips_router, "supabase", fake_supabase)
    monkeypatch.setattr(
        clips_router,
        "enforce_processing_access_rules",
        lambda *_args, **_kwargs: access_rules.UserAccessContext(
            tier="pro",
            status="active",
            interval="month",
            max_videos_per_month=220,
            max_clip_duration_seconds=90,
            priority_processing=True,
        ),
    )

    checked_amounts: list[int] = []

    def _fake_has_sufficient_credits(*, user_id, amount, **_kwargs):
        assert user_id == "user-id"
        checked_amounts.append(amount)
        return True

    enqueued_job_data: dict | None = None

    def _fake_enqueue_or_fail(*, job_data, **_kwargs):
        nonlocal enqueued_job_data
        enqueued_job_data = job_data

    monkeypatch.setattr(clips_router, "has_sufficient_credits", _fake_has_sufficient_credits)
    monkeypatch.setattr(clips_router, "enqueue_or_fail", _fake_enqueue_or_fail)

    # Restarting an AI-suggested clip must consume credits.
    response = client.post("/clips/generate", json={"clipId": "clip-1"})

    assert response.status_code == 200
    assert checked_amounts[-1] == 3
    assert enqueued_job_data is not None
    assert enqueued_job_data["generationCredits"] == 3


def test_restart_failed_ai_suggested_generation_is_still_free(client, monkeypatch):
    from api_app.app import app

    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(
        id="user-id",
        email=None,
        claims={},
    )

    fake_supabase = _FakeGenerateSupabase(
        owner_user_id="user-id",
        prior_jobs=[
            {
                "id": "job-failed-1",
                "clip_id": "clip-1",
                "type": "generate_clip",
                "status": "failed",
                "user_id": "user-id",
                "team_id": None,
            }
        ],
        clip_row={
            "id": "clip-1",
            "video_id": "video-1",
            "user_id": "user-id",
            "team_id": None,
            "billing_owner_user_id": "user-id",
            "start_time": 0,
            "end_time": 60,
            "status": "failed",
            "storage_path": None,
            "thumbnail_path": None,
            "ai_score": 0.92,
            "transcript_excerpt": "AI suggested excerpt",
        },
    )
    monkeypatch.setattr(clips_router, "supabase", fake_supabase)
    monkeypatch.setattr(
        clips_router,
        "enforce_processing_access_rules",
        lambda *_args, **_kwargs: access_rules.UserAccessContext(
            tier="pro",
            status="active",
            interval="month",
            max_videos_per_month=220,
            max_clip_duration_seconds=90,
            priority_processing=True,
        ),
    )

    checked_amounts: list[int] = []

    def _fake_has_sufficient_credits(*, user_id, amount, **_kwargs):
        assert user_id == "user-id"
        checked_amounts.append(amount)
        return True

    enqueued_job_data: dict | None = None

    def _fake_enqueue_or_fail(*, job_data, **_kwargs):
        nonlocal enqueued_job_data
        enqueued_job_data = job_data

    monkeypatch.setattr(clips_router, "has_sufficient_credits", _fake_has_sufficient_credits)
    monkeypatch.setattr(clips_router, "enqueue_or_fail", _fake_enqueue_or_fail)

    response = client.post("/clips/generate", json={"clipId": "clip-1"})

    assert response.status_code == 200
    assert checked_amounts == []
    assert enqueued_job_data is not None
    assert enqueued_job_data["generationCredits"] == 0


def test_generate_clip_sets_three_credits_for_pro_smart_cleanup(client, monkeypatch):
    from api_app.app import app

    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(
        id="user-id",
        email=None,
        claims={},
    )

    fake_supabase = _FakeGenerateSupabase(
        owner_user_id="user-id",
        clip_row={
            "id": "clip-1",
            "video_id": "video-1",
            "user_id": "user-id",
            "team_id": None,
            "billing_owner_user_id": "user-id",
            "start_time": 0,
            "end_time": 60,
            "status": "completed",
            "storage_path": "clips/clip-1.mp4",
            "thumbnail_path": None,
            "ai_score": 0.88,
            "transcript_excerpt": "AI suggestion",
        },
    )
    monkeypatch.setattr(clips_router, "supabase", fake_supabase)
    monkeypatch.setattr(
        clips_router,
        "enforce_processing_access_rules",
        lambda *_args, **_kwargs: access_rules.UserAccessContext(
            tier="pro",
            status="active",
            interval="month",
            max_videos_per_month=220,
            max_clip_duration_seconds=90,
            priority_processing=True,
        ),
    )

    checked_amounts: list[int] = []

    def _fake_has_sufficient_credits(*, user_id, amount):
        assert user_id == "user-id"
        checked_amounts.append(amount)
        return True

    enqueued_job_data: dict | None = None
    enqueued_queue_name: str | None = None

    def _fake_enqueue_or_fail(*, queue_name, job_data, **_kwargs):
        nonlocal enqueued_job_data
        nonlocal enqueued_queue_name
        enqueued_queue_name = queue_name
        enqueued_job_data = job_data

    monkeypatch.setattr(clips_router, "has_sufficient_credits", _fake_has_sufficient_credits)
    monkeypatch.setattr(clips_router, "enqueue_or_fail", _fake_enqueue_or_fail)

    response = client.post(
        "/clips/generate",
        json={"clipId": "clip-1", "smartCleanupEnabled": True},
    )

    assert response.status_code == 200
    assert checked_amounts[-1] == 3
    assert enqueued_job_data is not None
    assert enqueued_queue_name == "clip-generation-priority"
    assert enqueued_job_data["generationCredits"] == 3
    assert enqueued_job_data["smartCleanupEnabled"] is True


def test_credit_cost_endpoint_allows_cors_preflight(client):
    response = client.options(
        "/credits/cost",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"

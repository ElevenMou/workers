from __future__ import annotations

from dataclasses import dataclass

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
            "endTime": 20,
            "title": "Too short",
        },
    )
    assert response.status_code == 400
    assert "40 and 90 seconds" in response.json()["detail"]


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
        "has_sufficient_credits",
        lambda *, user_id, amount: True,
    )

    response = client.post(
        "/credits/cost",
        json={"url": "https://www.youtube.com/watch?v=FWkVBjcVw18"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["valid_url"] is True
    assert payload["analysisCredits"] >= 0
    assert payload["totalCredits"] == payload["analysisCredits"]
    assert payload["currentBalance"] == 25
    assert payload["hasEnoughCredits"] is True


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


def test_generate_clip_fails_early_when_no_credits(client, monkeypatch):
    from api_app.app import app

    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(
        id="user-id",
        email=None,
        claims={},
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

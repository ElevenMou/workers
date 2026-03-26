from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

if "whisper" not in sys.modules:
    whisper_stub = ModuleType("whisper")
    whisper_stub.load_model = lambda *_args, **_kwargs: SimpleNamespace(
        transcribe=lambda *_a, **_k: {"text": "", "segments": [], "language": "en"}
    )
    sys.modules["whisper"] = whisper_stub

if "openai" not in sys.modules:
    openai_stub = ModuleType("openai")

    class _StubNotFoundError(Exception):
        def __init__(self, message: str = "model not found", *, status_code: int = 404):
            super().__init__(message)
            self.status_code = status_code

    openai_stub.NotFoundError = _StubNotFoundError
    openai_stub.OpenAI = lambda *_args, **_kwargs: SimpleNamespace(
        beta=SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    parse=lambda **_k: SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                message=SimpleNamespace(
                                    parsed={"clips": []},
                                    content='{"clips":[]}',
                                    refusal=None,
                                )
                            )
                        ]
                    )
                )
            )
        )
    )
    sys.modules["openai"] = openai_stub

import config as config_module
from services import ai_analyzer as ai_analyzer_module
from tasks.videos import analyze as analyze_task_module


@pytest.fixture(autouse=True)
def _reset_ai_analyzer_state():
    ai_analyzer_module.AIAnalyzer._consecutive_failures = 0
    ai_analyzer_module.AIAnalyzer._circuit_open_until = 0.0
    ai_analyzer_module.AIAnalyzer._model_resolution_cache.clear()
    yield
    ai_analyzer_module.AIAnalyzer._consecutive_failures = 0
    ai_analyzer_module.AIAnalyzer._circuit_open_until = 0.0
    ai_analyzer_module.AIAnalyzer._model_resolution_cache.clear()


class _FakeResponse:
    def __init__(self, data: Any = None, *, count: int | None = None):
        self.data = data
        self.count = count
        self.error = None


class _FakeSupabaseQuery:
    def __init__(self, supabase: "_FakeSupabase", table_name: str):
        self.supabase = supabase
        self.table_name = table_name
        self._mode = "select"
        self._payload: dict[str, Any] | None = None
        self._select_columns: str | None = None
        self._select_count: str | None = None
        self._filters: dict[str, Any] = {}

    def select(self, columns: str, **kwargs):
        self._mode = "select"
        self._select_columns = columns
        self._select_count = kwargs.get("count")
        return self

    def update(self, payload: dict[str, Any]):
        self._mode = "update"
        self._payload = dict(payload)
        return self

    def insert(self, payload: dict[str, Any]):
        self._mode = "insert"
        self._payload = dict(payload)
        return self

    def eq(self, key: str, value: Any):
        self._filters[key] = value
        return self

    def limit(self, *_args, **_kwargs):
        return self

    def lt(self, *_args, **_kwargs):
        return self

    def execute(self):
        if self.table_name == "videos" and self._mode == "select":
            row = dict(self.supabase.video_row or {})
            if not row:
                return _FakeResponse([])
            return _FakeResponse([row])

        if self.table_name == "videos" and self._mode == "update":
            payload = dict(self._payload or {})
            self.supabase.video_updates.append(payload)
            if payload.get("status") == "completed":
                self.supabase.completed_finalize_attempts += 1
                if (
                    self.supabase.fail_first_completed_finalize
                    and self.supabase.completed_finalize_attempts == 1
                ):
                    raise RuntimeError("forced finalize failure")
            return _FakeResponse({})

        if self.table_name == "clips" and self._mode == "select":
            if self._select_columns == "start_time,end_time,title":
                rows = []
                for row in self.supabase.existing_clips + self.supabase.inserted_clips:
                    rows.append(
                        {
                            "start_time": row["start_time"],
                            "end_time": row["end_time"],
                            "title": row["title"],
                        }
                    )
                return _FakeResponse(rows)

            if self._select_columns == "id" and self._select_count == "exact":
                return _FakeResponse(
                    [],
                    count=len(self.supabase.existing_clips) + len(self.supabase.inserted_clips),
                )

            return _FakeResponse([])

        if self.table_name == "clips" and self._mode == "insert":
            payload = dict(self._payload or {})
            self.supabase.inserted_clips.append(payload)
            return _FakeResponse(payload)

        if self.table_name == "team_wallet_transactions" and self._mode == "select":
            return _FakeResponse([])

        return _FakeResponse({})


class _FakeSupabase:
    def __init__(
        self,
        *,
        existing_clips: list[dict[str, Any]] | None = None,
        fail_first_completed_finalize: bool = False,
        video_row: dict[str, Any] | None = None,
    ):
        self.existing_clips = list(existing_clips or [])
        self.inserted_clips: list[dict[str, Any]] = []
        self.video_updates: list[dict[str, Any]] = []
        self.fail_first_completed_finalize = fail_first_completed_finalize
        self.completed_finalize_attempts = 0
        self.video_row = dict(video_row or {})

    def table(self, table_name: str):
        return _FakeSupabaseQuery(self, table_name)


class _FakeDownloader:
    def __init__(
        self,
        *,
        work_dir: str,
        transcript: dict[str, Any] | None,
        duration_seconds: int = 180,
        has_captions: bool = True,
        has_audio: bool = True,
    ):
        self.work_dir = work_dir
        self._transcript = transcript
        self._duration_seconds = duration_seconds
        self._has_captions = has_captions
        self._has_audio = has_audio
        self.extract_audio_calls = 0
        self.download_calls = 0
        self.download_audio_only_calls = 0
        self.probe_calls = 0

    def download(self, _url: str, video_id: str):
        self.download_calls += 1
        return {
            "path": str(Path(self.work_dir) / f"{video_id}.mp4"),
            "title": "Hardening Test Video",
            "duration": self._duration_seconds,
            "thumbnail": "https://example.com/thumb.jpg",
            "platform": "youtube",
            "external_id": "yt123",
        }

    def download_audio_only(self, _url: str, video_id: str):
        self.download_audio_only_calls += 1
        return {
            "path": str(Path(self.work_dir) / f"{video_id}.m4a"),
            "title": "Hardening Test Video",
            "duration": self._duration_seconds,
            "thumbnail": "https://example.com/thumb.jpg",
            "platform": "youtube",
            "external_id": "yt123",
        }

    def probe_url(self, _url: str):
        self.probe_calls += 1
        return {
            "duration_seconds": self._duration_seconds,
            "title": "Hardening Test Video",
            "thumbnail": "https://example.com/thumb.jpg",
            "platform": "youtube",
            "external_id": "yt123",
            "detected_language": "en",
            "has_captions": self._has_captions,
            "has_audio": self._has_audio,
        }

    def get_youtube_transcript(self, _video_id: str, *, preferred_languages=None):
        return self._transcript

    def get_provider_transcript(self, _url: str, *, preferred_languages=None):
        return {
            "transcript": None,
            "fallback_reason": "provider_caption_unavailable",
            "track_source": None,
            "track_ext": None,
            "track_language": None,
        }

    def extract_audio(self, _video_path: str):
        self.extract_audio_calls += 1
        return str(Path(self.work_dir) / "audio.wav")


def _job_data(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "jobId": "job-123",
        "videoId": "video-123",
        "userId": "user-123",
        "url": "https://www.youtube.com/watch?v=test",
        "numClips": 2,
        "analysisCredits": 3,
        "chargeSource": "owner_wallet",
    }
    data.update(overrides)
    return data


def _base_transcript() -> dict[str, Any]:
    return {
        "source": "youtube",
        "language": "en",
        "languageCode": "en",
        "text": "segment one segment two segment three",
        "segments": [
            {"id": 0, "start": 0.0, "end": 60.0, "text": "segment one"},
            {"id": 1, "start": 60.0, "end": 120.0, "text": "segment two"},
            {"id": 2, "start": 120.0, "end": 180.0, "text": "segment three"},
        ],
    }


def _install_common_patches(
    monkeypatch: pytest.MonkeyPatch,
    *,
    tmp_path: Path,
    supabase: _FakeSupabase,
    downloader: _FakeDownloader,
    analyzer_clips: list[dict[str, Any]],
    transcriber_factory: Any = None,
):
    job_updates: list[dict[str, Any]] = []
    video_status_updates: list[tuple[str, dict[str, Any]]] = []
    capture_calls: list[str] = []
    release_calls: list[str] = []
    team_charge_calls: list[dict[str, Any]] = []

    monkeypatch.setattr(analyze_task_module, "supabase", supabase)
    monkeypatch.setattr(analyze_task_module, "assert_response_ok", lambda *_a, **_k: None)
    monkeypatch.setattr(
        analyze_task_module,
        "create_work_dir",
        lambda _name: str(tmp_path.mkdir(parents=True, exist_ok=True) or tmp_path),
    )
    monkeypatch.setattr(analyze_task_module, "has_sufficient_credits", lambda **_k: True)
    monkeypatch.setattr(analyze_task_module, "get_credit_balance", lambda *_a, **_k: 999)
    monkeypatch.setattr(analyze_task_module, "get_team_wallet_balance", lambda *_a, **_k: 999)
    monkeypatch.setattr(
        analyze_task_module,
        "update_job_status",
        lambda job_id, status, progress, error=None, result_data=None: job_updates.append(
            {
                "job_id": job_id,
                "status": status,
                "progress": progress,
                "error": error,
                "result_data": result_data,
            }
        ),
    )
    monkeypatch.setattr(
        analyze_task_module,
        "update_video_status",
        lambda video_id, status, **kwargs: video_status_updates.append(
            (status, {"video_id": video_id, **kwargs})
        ),
    )

    monkeypatch.setattr(
        analyze_task_module,
        "VideoDownloader",
        lambda work_dir: downloader,
    )
    monkeypatch.setattr(
        analyze_task_module,
        "AIAnalyzer",
        lambda: SimpleNamespace(find_best_clips=lambda *_a, **_k: list(analyzer_clips)),
    )
    monkeypatch.setattr(
        analyze_task_module,
        "_is_latest_analyze_job_for_video",
        lambda **_k: True,
    )

    if transcriber_factory is None:
        transcriber_factory = lambda: SimpleNamespace(
            transcribe=lambda _audio_path, **_kwargs: {
                "source": "whisper",
                "language": "en",
                "segments": [{"start": 0.0, "end": 90.0, "text": "whisper transcript"}],
            }
        )
    monkeypatch.setattr(analyze_task_module, "Transcriber", transcriber_factory)

    reservations_by_key: dict[str, str] = {}
    captured: set[str] = set()

    def _reserve_credits(**kwargs):
        key = str(kwargs.get("reservation_key") or f"default:{kwargs['user_id']}")
        reservation_id = reservations_by_key.get(key)
        if reservation_id:
            return reservation_id
        reservation_id = f"res-{len(reservations_by_key) + 1}"
        reservations_by_key[key] = reservation_id
        return reservation_id

    def _capture_credit_reservation(*, reservation_id: str, **_kwargs):
        capture_calls.append(reservation_id)
        captured.add(reservation_id)
        return True

    def _release_credit_reservation(*, reservation_id: str):
        release_calls.append(reservation_id)
        return True

    monkeypatch.setattr(analyze_task_module, "reserve_credits", _reserve_credits)
    monkeypatch.setattr(analyze_task_module, "capture_credit_reservation", _capture_credit_reservation)
    monkeypatch.setattr(analyze_task_module, "release_credit_reservation", _release_credit_reservation)
    monkeypatch.setattr(analyze_task_module, "emit_video_analysis_usage_event", lambda **_k: None)
    monkeypatch.setattr(analyze_task_module, "has_team_wallet_charge_for_job", lambda **_k: False)
    monkeypatch.setattr(
        analyze_task_module,
        "charge_video_analysis_credits",
        lambda **kwargs: team_charge_calls.append(kwargs),
    )

    return {
        "job_updates": job_updates,
        "video_status_updates": video_status_updates,
        "capture_calls": capture_calls,
        "release_calls": release_calls,
        "team_charge_calls": team_charge_calls,
    }


def test_reanalyze_keeps_clip_count_correct_with_boundary_aligned_clip(monkeypatch, tmp_path: Path):
    existing = [{"start_time": 5.0, "end_time": 55.0, "title": "existing clip"}]
    supabase = _FakeSupabase(existing_clips=existing)
    downloader = _FakeDownloader(work_dir=str(tmp_path), transcript=_base_transcript(), duration_seconds=180)
    state = _install_common_patches(
        monkeypatch,
        tmp_path=tmp_path,
        supabase=supabase,
        downloader=downloader,
        analyzer_clips=[
            {
                "start": 120.0,
                "end": 180.0,
                "duration": 60.0,
                "text": "boundary aligned clip",
                "title": "New aligned clip",
                "score": 0.92,
                "rank": 1,
                "hook": "hook",
                "tags": ["test"],
            }
        ],
    )

    analyze_task_module.analyze_video_task(_job_data())

    assert len(supabase.inserted_clips) == 1
    inserted = supabase.inserted_clips[0]
    assert inserted["end_time"] - inserted["start_time"] == pytest.approx(60.0)

    completed_video_updates = [
        payload for payload in supabase.video_updates if payload.get("status") == "completed"
    ]
    assert completed_video_updates, "missing completed video update"
    assert "clip_count" not in completed_video_updates[-1]

    completed_jobs = [u for u in state["job_updates"] if u["status"] == "completed"]
    assert completed_jobs, "missing completed job update"
    assert completed_jobs[-1]["result_data"]["clip_count"] == 2


def test_youtube_transcript_path_does_not_init_transcriber(monkeypatch, tmp_path: Path):
    supabase = _FakeSupabase()
    downloader = _FakeDownloader(work_dir=str(tmp_path), transcript=_base_transcript(), duration_seconds=180)
    _install_common_patches(
        monkeypatch,
        tmp_path=tmp_path,
        supabase=supabase,
        downloader=downloader,
        analyzer_clips=[],
        transcriber_factory=lambda: (_ for _ in ()).throw(RuntimeError("transcriber should not be initialized")),
    )
    monkeypatch.setattr(
        analyze_task_module,
        "AIAnalyzer",
        lambda: SimpleNamespace(find_best_clips=lambda *_a, **_k: []),
    )

    analyze_task_module.analyze_video_task(_job_data())

    assert downloader.extract_audio_calls == 0
    assert downloader.download_calls == 0
    assert downloader.download_audio_only_calls == 0


def test_analysis_uses_precomputed_source_metadata_before_any_media_download(
    monkeypatch, tmp_path: Path
):
    supabase = _FakeSupabase()
    downloader = _FakeDownloader(
        work_dir=str(tmp_path),
        transcript=_base_transcript(),
        duration_seconds=180,
    )
    _install_common_patches(
        monkeypatch,
        tmp_path=tmp_path,
        supabase=supabase,
        downloader=downloader,
        analyzer_clips=[],
        transcriber_factory=lambda: (_ for _ in ()).throw(RuntimeError("transcriber should not be initialized")),
    )
    monkeypatch.setattr(
        analyze_task_module,
        "AIAnalyzer",
        lambda: SimpleNamespace(find_best_clips=lambda *_a, **_k: []),
    )

    analyze_task_module.analyze_video_task(
        _job_data(
            analysisDurationSeconds=180,
                sourceTitle="Preloaded metadata title",
                sourceThumbnailUrl="https://example.com/thumb.jpg",
                sourcePlatform="youtube",
                sourceExternalId="yt123",
                sourceDetectedLanguage="en",
                sourceHasCaptions=True,
                sourceHasAudio=True,
            )
        )

    assert downloader.probe_calls == 0
    assert downloader.download_calls == 0
    assert downloader.download_audio_only_calls == 0
    assert downloader.extract_audio_calls == 0


def test_analysis_reuses_stored_transcript_before_whisper_fallback(
    monkeypatch, tmp_path: Path
):
    stored_transcript = {
        "source": "whisper",
        "language": "en",
        "languageCode": "en",
        "segments": [
            {"start": 0.0, "end": 60.0, "text": "stored one"},
            {"start": 60.0, "end": 120.0, "text": "stored two"},
            {"start": 120.0, "end": 180.0, "text": "stored three"},
        ],
    }
    supabase = _FakeSupabase(video_row={"transcript": stored_transcript})
    downloader = _FakeDownloader(
        work_dir=str(tmp_path),
        transcript=None,
        duration_seconds=180,
        has_captions=False,
        has_audio=True,
    )
    _install_common_patches(
        monkeypatch,
        tmp_path=tmp_path,
        supabase=supabase,
        downloader=downloader,
        analyzer_clips=[],
        transcriber_factory=lambda: (_ for _ in ()).throw(
            RuntimeError("transcriber should not be initialized")
        ),
    )
    monkeypatch.setattr(
        analyze_task_module,
        "AIAnalyzer",
        lambda: SimpleNamespace(find_best_clips=lambda *_a, **_k: []),
    )

    analyze_task_module.analyze_video_task(
        _job_data(
            analysisDurationSeconds=180,
            sourceTitle="Stored transcript video",
            sourceThumbnailUrl="https://example.com/thumb.jpg",
            sourcePlatform="youtube",
            sourceExternalId="yt123",
            sourceDetectedLanguage="en",
            sourceHasCaptions=False,
            sourceHasAudio=True,
        )
    )

    assert downloader.probe_calls == 0
    assert downloader.download_calls == 0
    assert downloader.download_audio_only_calls == 0
    assert downloader.extract_audio_calls == 0


def test_analysis_whisper_fallback_disables_full_word_timestamps_by_default(
    monkeypatch, tmp_path: Path
):
    supabase = _FakeSupabase()
    downloader = _FakeDownloader(
        work_dir=str(tmp_path),
        transcript=None,
        duration_seconds=180,
        has_captions=False,
        has_audio=True,
    )
    transcribe_calls: list[dict[str, Any]] = []

    def _transcriber_factory():
        return SimpleNamespace(
            transcribe=lambda _audio_path, **kwargs: (
                transcribe_calls.append(dict(kwargs))
                or {
                    "source": "whisper",
                    "language": "en",
                    "segments": [
                        {"start": 0.0, "end": 90.0, "text": "whisper transcript"}
                    ],
                }
            )
        )

    _install_common_patches(
        monkeypatch,
        tmp_path=tmp_path,
        supabase=supabase,
        downloader=downloader,
        analyzer_clips=[],
        transcriber_factory=_transcriber_factory,
    )
    monkeypatch.setattr(
        analyze_task_module,
        "WHISPER_FULL_TRANSCRIPT_WORD_TIMESTAMPS",
        False,
    )

    analyze_task_module.analyze_video_task(
        _job_data(
            analysisDurationSeconds=180,
            sourceTitle="Whisper fallback video",
            sourceThumbnailUrl="https://example.com/thumb.jpg",
            sourcePlatform="youtube",
            sourceExternalId="yt123",
            sourceDetectedLanguage="en",
            sourceHasCaptions=False,
            sourceHasAudio=True,
        )
    )

    assert transcribe_calls == [
        {"language_hint": "en", "word_timestamps": False}
    ]
    assert downloader.download_audio_only_calls == 1
    assert downloader.extract_audio_calls == 1


def test_owner_wallet_retry_after_post_charge_failure_is_idempotent(monkeypatch, tmp_path: Path):
    supabase = _FakeSupabase(fail_first_completed_finalize=True)
    downloader = _FakeDownloader(work_dir=str(tmp_path), transcript=_base_transcript(), duration_seconds=180)
    state = _install_common_patches(
        monkeypatch,
        tmp_path=tmp_path,
        supabase=supabase,
        downloader=downloader,
        analyzer_clips=[
            {
                "start": 10.0,
                "end": 55.0,
                "duration": 45.0,
                "text": "clip",
                "title": "Retry clip",
                "score": 0.8,
                "rank": 1,
                "hook": "",
                "tags": [],
            }
        ],
    )

    charge_count = {"net": 0}
    captured_ids: set[str] = set()

    def _capture_credit_reservation(*, reservation_id: str, **_kwargs):
        if reservation_id not in captured_ids:
            captured_ids.add(reservation_id)
            charge_count["net"] += 1
        state["capture_calls"].append(reservation_id)
        return True

    monkeypatch.setattr(analyze_task_module, "capture_credit_reservation", _capture_credit_reservation)

    with pytest.raises(RuntimeError, match="forced finalize failure"):
        analyze_task_module.analyze_video_task(_job_data(jobId="job-retry"))

    analyze_task_module.analyze_video_task(_job_data(jobId="job-retry"))

    assert charge_count["net"] == 1
    assert len(state["capture_calls"]) == 2


def test_window_credit_fallback_uses_processing_window_duration(monkeypatch, tmp_path: Path):
    supabase = _FakeSupabase()
    downloader = _FakeDownloader(work_dir=str(tmp_path), transcript=_base_transcript(), duration_seconds=300)
    _install_common_patches(
        monkeypatch,
        tmp_path=tmp_path,
        supabase=supabase,
        downloader=downloader,
        analyzer_clips=[],
    )

    reserved_amounts: list[int] = []

    def _reserve_credits(**kwargs):
        reserved_amounts.append(int(kwargs["amount"]))
        return "res-window"

    monkeypatch.setattr(analyze_task_module, "reserve_credits", _reserve_credits)

    analyze_task_module.analyze_video_task(
        _job_data(
            analysisCredits=0,
            processingStartSeconds=0,
            processingEndSeconds=119,
        )
    )

    completed_video_updates = [
        payload for payload in supabase.video_updates if payload.get("status") == "completed"
    ]
    assert completed_video_updates, "missing completed video update"
    assert completed_video_updates[-1]["credits_charged"] == 2
    assert reserved_amounts == [2]


def test_team_wallet_skips_duplicate_charge_by_job_id(monkeypatch, tmp_path: Path):
    supabase = _FakeSupabase()
    downloader = _FakeDownloader(work_dir=str(tmp_path), transcript=_base_transcript(), duration_seconds=180)
    state = _install_common_patches(
        monkeypatch,
        tmp_path=tmp_path,
        supabase=supabase,
        downloader=downloader,
        analyzer_clips=[],
    )
    monkeypatch.setattr(analyze_task_module, "has_team_wallet_charge_for_job", lambda **_k: True)

    analyze_task_module.analyze_video_task(
        _job_data(
            chargeSource="team_wallet",
            workspaceTeamId="team-1",
            billingOwnerUserId="owner-1",
        )
    )

    assert state["team_charge_calls"] == []


def test_marks_failed_when_initializer_errors(monkeypatch, tmp_path: Path):
    failed_job_calls: list[dict[str, Any]] = []
    failed_video_calls: list[tuple[str, dict[str, Any]]] = []

    monkeypatch.setattr(analyze_task_module, "create_work_dir", lambda _name: str(tmp_path))
    monkeypatch.setattr(
        analyze_task_module,
        "_is_latest_analyze_job_for_video",
        lambda **_k: True,
    )
    monkeypatch.setattr(analyze_task_module, "has_sufficient_credits", lambda **_k: True)
    monkeypatch.setattr(analyze_task_module, "reserve_credits", lambda **_k: "res-1")
    monkeypatch.setattr(analyze_task_module, "capture_credit_reservation", lambda **_k: True)
    monkeypatch.setattr(analyze_task_module, "release_credit_reservation", lambda **_k: True)
    monkeypatch.setattr(analyze_task_module, "VideoDownloader", lambda **_k: (_ for _ in ()).throw(RuntimeError("init failed")))
    monkeypatch.setattr(analyze_task_module, "update_job_status", lambda *args, **kwargs: failed_job_calls.append({"args": args, "kwargs": kwargs}))
    monkeypatch.setattr(
        analyze_task_module,
        "update_video_status",
        lambda video_id, status, **kwargs: failed_video_calls.append((status, {"video_id": video_id, **kwargs})),
    )

    with pytest.raises(RuntimeError, match="init failed"):
        analyze_task_module.analyze_video_task(_job_data(jobId="job-init-fail", videoId="video-init-fail"))

    assert any(call["args"][1] == "failed" for call in failed_job_calls)
    assert any(status == "failed" for status, _ in failed_video_calls)


def test_superseded_analyze_job_exits_before_processing(monkeypatch, tmp_path: Path):
    supabase = _FakeSupabase()
    downloader = _FakeDownloader(work_dir=str(tmp_path), transcript=_base_transcript(), duration_seconds=180)
    state = _install_common_patches(
        monkeypatch,
        tmp_path=tmp_path,
        supabase=supabase,
        downloader=downloader,
        analyzer_clips=[],
    )
    monkeypatch.setattr(
        analyze_task_module,
        "_is_latest_analyze_job_for_video",
        lambda **_k: False,
    )

    analyze_task_module.analyze_video_task(_job_data(jobId="job-stale", videoId="video-stale"))

    assert downloader.probe_calls == 0
    assert downloader.download_calls == 0
    assert state["capture_calls"] == []
    failed_jobs = [u for u in state["job_updates"] if u["status"] == "failed"]
    assert failed_jobs
    assert failed_jobs[-1]["result_data"]["stage"] == "superseded"


def test_superseded_analyze_job_releases_reservation_before_saving(monkeypatch, tmp_path: Path):
    supabase = _FakeSupabase()
    downloader = _FakeDownloader(work_dir=str(tmp_path), transcript=_base_transcript(), duration_seconds=180)
    state = _install_common_patches(
        monkeypatch,
        tmp_path=tmp_path,
        supabase=supabase,
        downloader=downloader,
        analyzer_clips=[],
    )
    stale_checks = iter([True, False])
    monkeypatch.setattr(
        analyze_task_module,
        "_is_latest_analyze_job_for_video",
        lambda **_k: next(stale_checks),
    )

    analyze_task_module.analyze_video_task(_job_data(jobId="job-stale-late", videoId="video-stale-late"))

    assert state["release_calls"] == ["res-1"]
    assert state["capture_calls"] == []
    completed_jobs = [u for u in state["job_updates"] if u["status"] == "completed"]
    assert completed_jobs == []
    failed_jobs = [u for u in state["job_updates"] if u["status"] == "failed"]
    assert failed_jobs
    assert failed_jobs[-1]["result_data"]["stage"] == "superseded"


def test_analysis_transcript_is_strictly_clipped_to_processing_window(monkeypatch, tmp_path: Path):
    supabase = _FakeSupabase()
    transcript = {
        "source": "youtube",
        "language": "en",
        "segments": [
            {"id": 0, "start": 0.0, "end": 30.0, "text": "outside to inside"},
            {"id": 1, "start": 30.0, "end": 50.0, "text": "inside full"},
            {
                "id": 2,
                "start": 50.0,
                "end": 80.0,
                "text": "inside with words clipped",
                "words": [
                    {"word": "inside", "start": 52.0, "end": 56.0},
                    {"word": "with", "start": 56.0, "end": 60.0},
                    {"word": "words", "start": 60.0, "end": 64.0},
                    {"word": "clipped", "start": 64.0, "end": 74.0},
                ],
            },
            {"id": 3, "start": 80.0, "end": 100.0, "text": "outside"},
        ],
    }
    downloader = _FakeDownloader(work_dir=str(tmp_path), transcript=transcript, duration_seconds=120)
    state = _install_common_patches(
        monkeypatch,
        tmp_path=tmp_path,
        supabase=supabase,
        downloader=downloader,
        analyzer_clips=[],
    )

    captured_segments: list[dict[str, Any]] = []

    def _capture_find_best_clips(transcript_payload, **_kwargs):
        captured_segments.extend(list(transcript_payload.get("segments") or []))
        return []

    monkeypatch.setattr(
        analyze_task_module,
        "AIAnalyzer",
        lambda: SimpleNamespace(find_best_clips=_capture_find_best_clips),
    )

    analyze_task_module.analyze_video_task(
        _job_data(
            processingStartSeconds=25,
            processingEndSeconds=70,
            clipLengthMinSeconds=10,
            clipLengthMaxSeconds=60,
            clipLengthSeconds=60,
        )
    )

    assert captured_segments
    assert all(float(seg["start"]) >= 25.0 for seg in captured_segments)
    assert all(float(seg["end"]) <= 70.0 for seg in captured_segments)
    # Segment 0 overlaps only partially and has no word timing; it should be dropped.
    assert all(str(seg.get("text") or "").strip() != "outside to inside" for seg in captured_segments)

    completed_jobs = [u for u in state["job_updates"] if u["status"] == "completed"]
    assert completed_jobs
    result_data = completed_jobs[-1]["result_data"]
    assert result_data["transcript_window_stats"]["segments_dropped_partial_without_words"] >= 1


def test_strict_clip_validation_skips_unaligned_and_short_clips_without_resizing(
    monkeypatch, tmp_path: Path
):
    supabase = _FakeSupabase()
    downloader = _FakeDownloader(work_dir=str(tmp_path), transcript=_base_transcript(), duration_seconds=180)
    state = _install_common_patches(
        monkeypatch,
        tmp_path=tmp_path,
        supabase=supabase,
        downloader=downloader,
        analyzer_clips=[
            {
                "start": 60.0,
                "end": 120.0,
                "duration": 60.0,
                "text": "too short for selected range",
                "title": "Short clip",
                "score": 0.7,
                "rank": 1,
                "hook": "",
                "tags": [],
            },
            {
                "start": 31.0,
                "end": 121.0,
                "duration": 90.0,
                "text": "not boundary aligned",
                "title": "Unaligned clip",
                "score": 0.8,
                "rank": 2,
                "hook": "",
                "tags": [],
            },
            {
                "start": 0.0,
                "end": 120.0,
                "duration": 120.0,
                "text": "valid range and boundaries",
                "title": "Valid clip",
                "score": 0.9,
                "rank": 3,
                "hook": "",
                "tags": [],
            },
        ],
    )

    analyze_task_module.analyze_video_task(
        _job_data(
            clipLengthMinSeconds=80,
            clipLengthMaxSeconds=120,
            clipLengthSeconds=120,
        )
    )

    assert len(supabase.inserted_clips) == 1
    inserted = supabase.inserted_clips[0]
    assert inserted["start_time"] == pytest.approx(0.0)
    assert inserted["end_time"] == pytest.approx(120.0)
    assert inserted["end_time"] - inserted["start_time"] == pytest.approx(120.0)

    completed_jobs = [u for u in state["job_updates"] if u["status"] == "completed"]
    assert completed_jobs
    result_data = completed_jobs[-1]["result_data"]
    assert result_data["clip_validation_stats"]["accepted"] == 1
    assert result_data["clip_validation_stats"]["skipped_duration_out_of_range"] >= 1
    assert result_data["clip_validation_stats"]["skipped_not_boundary_aligned"] >= 1


def test_legacy_clip_length_payload_maps_to_deterministic_range(monkeypatch, tmp_path: Path):
    supabase = _FakeSupabase()
    downloader = _FakeDownloader(work_dir=str(tmp_path), transcript=_base_transcript(), duration_seconds=180)
    state = _install_common_patches(
        monkeypatch,
        tmp_path=tmp_path,
        supabase=supabase,
        downloader=downloader,
        analyzer_clips=[
            {
                "start": 60.0,
                "end": 110.0,
                "duration": 50.0,
                "text": "below legacy minimum",
                "title": "Too short",
                "score": 0.8,
                "rank": 1,
                "hook": "",
                "tags": [],
            },
            {
                "start": 60.0,
                "end": 120.0,
                "duration": 60.0,
                "text": "within legacy range",
                "title": "Valid legacy",
                "score": 0.9,
                "rank": 2,
                "hook": "",
                "tags": [],
            },
        ],
    )

    analyze_task_module.analyze_video_task(
        _job_data(
            clipLengthSeconds=90,
        )
    )

    assert len(supabase.inserted_clips) == 1
    inserted = supabase.inserted_clips[0]
    assert inserted["end_time"] - inserted["start_time"] == pytest.approx(60.0)

    completed_jobs = [u for u in state["job_updates"] if u["status"] == "completed"]
    assert completed_jobs
    result_data = completed_jobs[-1]["result_data"]
    assert result_data["min_clip_seconds"] == 60
    assert result_data["max_clip_seconds"] == 90


def _parsed_payload(payload: dict[str, Any]) -> ai_analyzer_module.ClipAnalysisResponse:
    return ai_analyzer_module.ClipAnalysisResponse.model_validate(payload)


class _FakeOpenAIClient:
    """Fake OpenAI client that can return parsed or text fallback responses."""

    def __init__(
        self,
        response_text: str = "",
        *,
        parsed_payload: dict[str, Any] | None = None,
        refusal: str | None = None,
    ):
        self._response_text = response_text
        self._parsed_payload = parsed_payload
        self._refusal = refusal
        self.beta = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(parse=self._parse)
            )
        )

    def _parse(self, **_kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        parsed=(
                            _parsed_payload(self._parsed_payload)
                            if self._parsed_payload is not None
                            else None
                        ),
                        content=self._response_text,
                        refusal=self._refusal,
                    )
                )
            ]
        )


class _FakeOpenAIFallbackClient:
    """Fake OpenAI client that fails on one model and succeeds on another."""

    def __init__(self, *, failing_model: str, fallback_model: str):
        self.failing_model = failing_model
        self.fallback_model = fallback_model
        self.calls: list[str] = []
        self.beta = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(parse=self._parse)
            )
        )

    def _parse(self, **kwargs):
        model = str(kwargs.get("model") or "")
        self.calls.append(model)
        if model == self.failing_model:
            raise ai_analyzer_module.NotFoundError(
                "The model `invalid-model` does not exist.",
                status_code=404,
            )
        if model != self.fallback_model:
            raise RuntimeError(f"Unexpected model used: {model}")
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        parsed=_parsed_payload(
                            {
                                "clips": [
                                    {
                                        "rank": 1,
                                        "start_time": 0.0,
                                        "end_time": 60.0,
                                        "duration": 60.0,
                                        "clip_title": "Fallback clip",
                                        "hook": "hook",
                                        "summary": "summary",
                                        "confidence_score": 0.9,
                                        "tags": [],
                                    }
                                ]
                            }
                        ),
                        content="",
                        refusal=None,
                    )
                )
            ]
        )


class _FakeOpenAIFailingClient:
    def __init__(self, error: Exception):
        self.error = error
        self.beta = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(parse=self._parse)
            )
        )

    def _parse(self, **_kwargs):
        raise self.error


def test_ai_analyzer_salvages_valid_clips_from_mixed_payload(monkeypatch):
    parsed_payload = {
        "clips": [
            {
                "rank": 2,
                "start_time": 12.0,
                "end_time": 58.0,
                "duration": 46.0,
                "clip_title": "Valid clip",
                "hook": "Hook",
                "summary": "Valid summary",
                "confidence_score": 0.91,
                "tags": ["one"],
            },
            {
                "rank": 1,
                "start_time": None,
                "end_time": 99.0,
                "duration": 30.0,
                "clip_title": "Invalid clip",
                "hook": "",
                "summary": "Invalid",
                "confidence_score": 0.4,
                "tags": [],
            },
        ]
    }

    monkeypatch.setattr(
        ai_analyzer_module,
        "OpenAI",
        lambda **_kwargs: _FakeOpenAIClient(parsed_payload=parsed_payload),
    )
    analyzer = ai_analyzer_module.AIAnalyzer()

    clips = analyzer.find_best_clips(
        transcript=_base_transcript(),
        num_clips=5,
        min_duration=40,
        max_duration=90,
    )

    assert len(clips) == 1
    assert clips[0]["title"] == "Valid clip"
    assert clips[0]["duration"] == pytest.approx(46.0)


def test_ai_analyzer_raises_clean_error_for_non_json_response(monkeypatch):
    """When model returns text instead of parsed JSON, non-JSON raises."""
    monkeypatch.setattr(
        ai_analyzer_module,
        "OpenAI",
        lambda **_kwargs: _FakeOpenAIClient("No JSON payload here"),
    )
    analyzer = ai_analyzer_module.AIAnalyzer()

    with pytest.raises(Exception):
        analyzer.find_best_clips(
            transcript=_base_transcript(),
            num_clips=2,
            min_duration=40,
            max_duration=90,
        )


def test_ai_analyzer_handles_structured_parse_response(monkeypatch):
    """Primary path: model returns a parsed structured response."""
    parsed_payload = {
        "clips": [
            {
                "rank": 1,
                "start_time": 0.0,
                "end_time": 60.0,
                "duration": 60.0,
                "clip_title": "Great insight",
                "hook": "Did you know...",
                "summary": "An amazing insight about the topic",
                "confidence_score": 0.95,
                "tags": ["educational", "insight"],
            },
            {
                "rank": 2,
                "start_time": 60.0,
                "end_time": 120.0,
                "duration": 60.0,
                "clip_title": "Key takeaway",
                "hook": "The most important thing...",
                "summary": "Summary of the key takeaway",
                "confidence_score": 0.88,
                "tags": ["takeaway"],
            },
        ]
    }

    monkeypatch.setattr(
        ai_analyzer_module,
        "OpenAI",
        lambda **_kwargs: _FakeOpenAIClient(parsed_payload=parsed_payload),
    )
    analyzer = ai_analyzer_module.AIAnalyzer()

    clips = analyzer.find_best_clips(
        transcript=_base_transcript(),
        num_clips=2,
        min_duration=40,
        max_duration=90,
    )

    assert len(clips) == 2
    assert clips[0]["title"] == "Great insight"
    assert clips[0]["score"] == pytest.approx(0.95)
    assert clips[0]["start"] == pytest.approx(0.0)
    assert clips[0]["end"] == pytest.approx(60.0)
    assert clips[1]["title"] == "Key takeaway"
    assert clips[1]["hook"] == "The most important thing..."


def test_ai_analyzer_accepts_nested_segments_payload(monkeypatch):
    response_text = json.dumps(
        {
            "output": {
                "segments": [
                    {
                        "rank": 1,
                        "start_time": 0.0,
                        "end_time": 45.0,
                        "duration": 45.0,
                        "clip_title": "Nested payload clip",
                        "summary": "Nested summary",
                        "confidence_score": 0.8,
                        "tags": [],
                    }
                ]
            }
        }
    )

    monkeypatch.setattr(
        ai_analyzer_module,
        "OpenAI",
        lambda **_kwargs: _FakeOpenAIClient(response_text=response_text),
    )
    analyzer = ai_analyzer_module.AIAnalyzer()

    clips = analyzer.find_best_clips(
        transcript=_base_transcript(),
        num_clips=1,
        min_duration=30,
        max_duration=90,
    )

    assert len(clips) == 1
    assert clips[0]["title"] == "Nested payload clip"
    assert clips[0]["duration"] == pytest.approx(45.0)


def test_ai_analyzer_falls_back_when_text_has_json(monkeypatch):
    monkeypatch.setattr(
        ai_analyzer_module,
        "OpenAI",
        lambda **_kwargs: _FakeOpenAIClient(
            response_text=json.dumps(
                {
                    "clips": [
                        {
                            "rank": 1,
                            "start_time": 0.0,
                            "end_time": 45.0,
                            "duration": 45.0,
                            "clip_title": "Recovered from text",
                            "summary": "Recovered summary",
                            "confidence_score": 0.81,
                            "tags": [],
                        }
                    ]
                }
            )
        ),
    )
    analyzer = ai_analyzer_module.AIAnalyzer()

    clips = analyzer.find_best_clips(
        transcript=_base_transcript(),
        num_clips=1,
        min_duration=30,
        max_duration=90,
    )

    assert len(clips) == 1
    assert clips[0]["title"] == "Recovered from text"


def test_ai_analyzer_logs_salvage_for_repaired_truncated_json(monkeypatch, caplog):
    payload_text = json.dumps(
        {
            "clips": [
                {
                    "rank": 1,
                    "start_time": 0.0,
                    "end_time": 45.0,
                    "duration": 45.0,
                    "clip_title": "Recovered truncated",
                    "summary": "Recovered summary",
                    "confidence_score": 0.81,
                    "tags": [],
                },
                {
                    "rank": 2,
                    "start_time": 45.0,
                    "end_time": 90.0,
                    "duration": 45.0,
                    "clip_title": "Incomplete clip",
                    "summary": "Should be dropped because payload truncates here",
                    "confidence_score": 0.62,
                    "tags": [],
                },
            ]
        }
    )
    second_clip_marker = payload_text.find('{"rank": 2')
    assert second_clip_marker > 0
    truncated_payload = payload_text[: second_clip_marker + 28]

    monkeypatch.setattr(
        ai_analyzer_module,
        "OpenAI",
        lambda **_kwargs: _FakeOpenAIClient(response_text=truncated_payload),
    )
    analyzer = ai_analyzer_module.AIAnalyzer()

    caplog.set_level(logging.WARNING)
    clips = analyzer.find_best_clips(
        transcript=_base_transcript(),
        num_clips=1,
        min_duration=30,
        max_duration=90,
    )

    assert len(clips) == 1
    assert clips[0]["title"] == "Recovered truncated"
    assert "Repaired truncated JSON response" in caplog.text
    assert "Analyzer salvaged" in caplog.text


def test_ai_analyzer_accepts_json_block_payload(monkeypatch):
    monkeypatch.setattr(
        ai_analyzer_module,
        "OpenAI",
        lambda **_kwargs: _FakeOpenAIClient(
            response_text=json.dumps(
                {
                    "result": {
                        "clips": [
                            {
                                "rank": 1,
                                "start_time": 10.0,
                                "end_time": 55.0,
                                "duration": 45.0,
                                "clip_title": "JSON block clip",
                                "summary": "JSON block summary",
                                "confidence_score": 0.83,
                                "tags": ["json"],
                            }
                        ]
                    }
                }
            )
        ),
    )
    analyzer = ai_analyzer_module.AIAnalyzer()

    clips = analyzer.find_best_clips(
        transcript=_base_transcript(),
        num_clips=1,
        min_duration=30,
        max_duration=90,
    )

    assert len(clips) == 1
    assert clips[0]["title"] == "JSON block clip"


def test_ai_analyzer_retries_with_fallback_model_when_primary_missing(monkeypatch):
    failing_model = "invalid-model"
    fallback_model = "fallback-model"
    fake_client = _FakeOpenAIFallbackClient(
        failing_model=failing_model,
        fallback_model=fallback_model,
    )

    monkeypatch.setattr(
        ai_analyzer_module,
        "OpenAI",
        lambda **_kwargs: fake_client,
    )
    monkeypatch.setattr(ai_analyzer_module, "DEFAULT_ANALYZER_MODEL", failing_model)
    monkeypatch.setenv("OPENAI_ANALYZER_FALLBACK_MODELS", fallback_model)
    ai_analyzer_module.AIAnalyzer._model_resolution_cache.clear()

    analyzer = ai_analyzer_module.AIAnalyzer()
    clips = analyzer.find_best_clips(
        transcript=_base_transcript(),
        num_clips=1,
        min_duration=40,
        max_duration=90,
    )

    assert len(clips) == 1
    assert fake_client.calls == [failing_model, fallback_model]


def test_ai_analyzer_raises_clean_error_for_refusal(monkeypatch):
    monkeypatch.setattr(
        ai_analyzer_module,
        "OpenAI",
        lambda **_kwargs: _FakeOpenAIClient(refusal="I cannot comply."),
    )
    analyzer = ai_analyzer_module.AIAnalyzer()

    with pytest.raises(RuntimeError, match="OpenAI analyzer refusal"):
        analyzer.find_best_clips(
            transcript=_base_transcript(),
            num_clips=1,
            min_duration=30,
            max_duration=90,
        )


def test_ai_analyzer_circuit_breaker_opens_after_repeated_failures(monkeypatch):
    monkeypatch.setattr(
        ai_analyzer_module,
        "OpenAI",
        lambda **_kwargs: _FakeOpenAIFailingClient(RuntimeError("boom")),
    )
    monkeypatch.setattr(ai_analyzer_module, "_CIRCUIT_BREAKER_THRESHOLD", 2)
    monkeypatch.setattr(ai_analyzer_module, "_CIRCUIT_BREAKER_COOLDOWN", 60.0)

    analyzer = ai_analyzer_module.AIAnalyzer()

    with pytest.raises(RuntimeError, match="boom"):
        analyzer.find_best_clips(
            transcript=_base_transcript(),
            num_clips=1,
            min_duration=30,
            max_duration=90,
        )

    with pytest.raises(RuntimeError, match="boom"):
        analyzer.find_best_clips(
            transcript=_base_transcript(),
            num_clips=1,
            min_duration=30,
            max_duration=90,
        )

    with pytest.raises(RuntimeError, match="circuit breaker open"):
        analyzer.find_best_clips(
            transcript=_base_transcript(),
            num_clips=1,
            min_duration=30,
            max_duration=90,
        )


def test_provider_transcript_path_without_audio_skips_whisper(monkeypatch, tmp_path: Path):
    supabase = _FakeSupabase()
    downloader = _FakeDownloader(
        work_dir=str(tmp_path),
        transcript=None,
        duration_seconds=180,
        has_captions=True,
        has_audio=False,
    )
    provider_transcript = {
        "source": "provider_captions",
        "language": "es",
        "languageCode": "es",
        "segments": [
            {"id": 0, "start": 0.0, "end": 60.0, "text": "segmento uno"},
            {"id": 1, "start": 60.0, "end": 120.0, "text": "segmento dos"},
            {"id": 2, "start": 120.0, "end": 180.0, "text": "segmento tres"},
        ],
    }
    downloader.probe_url = lambda _url: {
        "duration_seconds": 180,
        "title": "Provider Caption Test",
        "thumbnail": "https://example.com/thumb.jpg",
        "platform": "vimeo",
        "external_id": "vm123",
        "detected_language": "es",
        "has_captions": True,
        "has_audio": False,
    }
    downloader.get_provider_transcript = lambda _url, *, preferred_languages=None: {
        "transcript": provider_transcript,
        "fallback_reason": None,
        "track_source": "provider_subtitles",
        "track_ext": "vtt",
        "track_language": "es",
    }

    state = _install_common_patches(
        monkeypatch,
        tmp_path=tmp_path,
        supabase=supabase,
        downloader=downloader,
        analyzer_clips=[],
        transcriber_factory=lambda: (_ for _ in ()).throw(
            RuntimeError("transcriber should not be initialized")
        ),
    )
    monkeypatch.setattr(
        analyze_task_module,
        "AIAnalyzer",
        lambda: SimpleNamespace(find_best_clips=lambda *_a, **_k: []),
    )

    analyze_task_module.analyze_video_task(_job_data(url="https://vimeo.com/123456"))

    assert downloader.extract_audio_calls == 0
    assert downloader.download_calls == 0
    assert downloader.download_audio_only_calls == 0
    completed_jobs = [update for update in state["job_updates"] if update["status"] == "completed"]
    assert completed_jobs[-1]["result_data"]["transcript_source"] == "provider_captions"
    assert completed_jobs[-1]["result_data"]["transcript_language"] == "es"


def test_validation_backfills_past_invalid_candidates(monkeypatch, tmp_path: Path):
    supabase = _FakeSupabase()
    downloader = _FakeDownloader(work_dir=str(tmp_path), transcript=_base_transcript(), duration_seconds=180)
    state = _install_common_patches(
        monkeypatch,
        tmp_path=tmp_path,
        supabase=supabase,
        downloader=downloader,
        analyzer_clips=[
            {
                "start": 300.0,
                "end": 360.0,
                "duration": 60.0,
                "text": "outside window",
                "title": "Invalid window clip",
                "score": 0.91,
                "rank": 1,
            },
            {
                "start": 0.0,
                "end": 60.0,
                "duration": 60.0,
                "text": "segment one",
                "title": "First valid clip",
                "score": 0.93,
                "rank": 2,
            },
            {
                "start": 120.0,
                "end": 180.0,
                "duration": 60.0,
                "text": "segment three",
                "title": "Second valid clip",
                "score": 0.94,
                "rank": 3,
            },
        ],
    )

    analyze_task_module.analyze_video_task(_job_data(numClips=2))

    assert [clip["title"] for clip in supabase.inserted_clips] == [
        "First valid clip",
        "Second valid clip",
    ]
    completed_jobs = [update for update in state["job_updates"] if update["status"] == "completed"]
    clip_stats = completed_jobs[-1]["result_data"]["clip_validation_stats"]
    assert clip_stats["candidate_validated"] == 3
    assert clip_stats["accepted"] == 2
    assert clip_stats["skipped_outside_window"] == 1


def test_ai_analyzer_chunked_mode_keeps_tail_candidates(monkeypatch):
    transcript = {
        "source": "provider_captions",
        "language": "en",
        "segments": [
            {
                "id": index,
                "start": float(index * 120),
                "end": float(index * 120 + 60),
                "text": f"segment {index}",
            }
            for index in range(10)
        ],
    }

    monkeypatch.setattr(
        ai_analyzer_module.AIAnalyzer,
        "_analyze_snippets",
        lambda self, *, snippets, request_count, **_kwargs: [
            {
                "start": float(snippets[0]["start"]),
                "end": float(snippets[0]["start"]) + 60.0,
                "duration": 60.0,
                "text": f"chunk {int(snippets[0]['start'])}",
                "title": f"Chunk {int(snippets[0]['start'])}",
                "hook": "hook",
                "hook_text": "hook",
                "reasoning": "reason",
                "content_category": "education",
                "score": 0.7 if float(snippets[0]["start"]) < 900 else 0.95,
                "tags": [],
                "rank": 1,
            }
        ],
    )
    monkeypatch.setattr(
        ai_analyzer_module.AIAnalyzer,
        "_synthesize_candidates",
        lambda self, *, candidates, candidate_target, **_kwargs: list(candidates)[:candidate_target],
    )

    analyzer = ai_analyzer_module.AIAnalyzer()
    result = analyzer.find_best_clips_detailed(
        transcript=transcript,
        num_clips=2,
        min_duration=30,
        max_duration=90,
    )

    assert result["diagnostics"]["analysis_mode"] == "chunked"
    assert result["diagnostics"]["chunk_count"] > 1
    assert any(float(candidate["start"]) >= 900.0 for candidate in result["clips"])


def test_validate_env_accepts_openai_api_key_only(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "service-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    config_module.validate_env()


def test_validate_env_accepts_legacy_anthropic_api_key_only(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "service-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "legacy-key")

    config_module.validate_env()


def test_validate_env_requires_analyzer_api_key(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "service-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(SystemExit):
        config_module.validate_env(require_browser_cors=True)


def test_validate_env_rejects_localhost_only_cors_in_production(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "service-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("MINIO_ENDPOINT", "minio:9000")
    monkeypatch.setenv("MINIO_ACCESS_KEY", "minio-access")
    monkeypatch.setenv("MINIO_SECRET_KEY", "minio-secret")
    monkeypatch.setenv("MINIO_PUBLIC_ENDPOINT", "https://storage.example.com")
    monkeypatch.setenv("WORKER_INTERNAL_API_TOKEN", "internal-token")
    monkeypatch.setenv("WORKER_PUBLIC_BASE_URL", "https://api.example.com")
    monkeypatch.setenv("CADDY_DOMAIN", "api.example.com")
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
    monkeypatch.delenv("CORS_ALLOWED_ORIGIN_REGEX", raising=False)

    with pytest.raises(SystemExit):
        config_module.validate_env(require_browser_cors=True)


def test_validate_env_accepts_public_cors_origin_in_production(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "service-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("MINIO_ENDPOINT", "minio:9000")
    monkeypatch.setenv("MINIO_ACCESS_KEY", "minio-access")
    monkeypatch.setenv("MINIO_SECRET_KEY", "minio-secret")
    monkeypatch.setenv("MINIO_PUBLIC_ENDPOINT", "https://storage.example.com")
    monkeypatch.setenv("WORKER_INTERNAL_API_TOKEN", "internal-token")
    monkeypatch.setenv("WORKER_PUBLIC_BASE_URL", "https://api.example.com")
    monkeypatch.setenv("CADDY_DOMAIN", "api.example.com")
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "https://www.clipscut.pro")
    monkeypatch.delenv("CORS_ALLOWED_ORIGIN_REGEX", raising=False)

    config_module.validate_env(require_browser_cors=True)


def test_validate_env_warns_on_local_worker_public_base_url_in_production(monkeypatch, caplog):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "service-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("MINIO_ENDPOINT", "minio:9000")
    monkeypatch.setenv("MINIO_ACCESS_KEY", "minio-access")
    monkeypatch.setenv("MINIO_SECRET_KEY", "minio-secret")
    monkeypatch.setenv("MINIO_PUBLIC_ENDPOINT", "https://storage.example.com")
    monkeypatch.setenv("WORKER_INTERNAL_API_TOKEN", "internal-token")
    monkeypatch.setenv("WORKER_PUBLIC_BASE_URL", "http://localhost:8080")
    monkeypatch.setenv("CADDY_DOMAIN", "api.example.com")
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "https://www.clipscut.pro")
    monkeypatch.delenv("CORS_ALLOWED_ORIGIN_REGEX", raising=False)

    config_module.validate_env(require_browser_cors=True)

    assert any(
        "WORKER_PUBLIC_BASE_URL is not publicly reachable" in record.getMessage()
        for record in caplog.records
    )


def test_validate_env_warns_on_local_minio_public_endpoint_in_production(monkeypatch, caplog):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "service-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("MINIO_ENDPOINT", "minio:9000")
    monkeypatch.setenv("MINIO_ACCESS_KEY", "minio-access")
    monkeypatch.setenv("MINIO_SECRET_KEY", "minio-secret")
    monkeypatch.setenv("MINIO_PUBLIC_ENDPOINT", "http://localhost:9000")
    monkeypatch.setenv("WORKER_INTERNAL_API_TOKEN", "internal-token")
    monkeypatch.setenv("WORKER_PUBLIC_BASE_URL", "https://api.example.com")
    monkeypatch.setenv("CADDY_DOMAIN", "api.example.com")
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "https://www.clipscut.pro")
    monkeypatch.delenv("CORS_ALLOWED_ORIGIN_REGEX", raising=False)

    config_module.validate_env(require_browser_cors=True)

    assert any(
        "MINIO_PUBLIC_ENDPOINT is not publicly reachable" in record.getMessage()
        for record in caplog.records
    )

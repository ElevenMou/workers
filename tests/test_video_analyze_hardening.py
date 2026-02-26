from __future__ import annotations

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

if "anthropic" not in sys.modules:
    anthropic_stub = ModuleType("anthropic")
    anthropic_stub.Anthropic = lambda *_args, **_kwargs: SimpleNamespace(
        messages=SimpleNamespace(
            create=lambda **_k: SimpleNamespace(content=[SimpleNamespace(text='{"clips": []}')])
        )
    )
    sys.modules["anthropic"] = anthropic_stub

from services import ai_analyzer as ai_analyzer_module
from tasks.videos import analyze as analyze_task_module


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
    ):
        self.existing_clips = list(existing_clips or [])
        self.inserted_clips: list[dict[str, Any]] = []
        self.video_updates: list[dict[str, Any]] = []
        self.fail_first_completed_finalize = fail_first_completed_finalize
        self.completed_finalize_attempts = 0

    def table(self, table_name: str):
        return _FakeSupabaseQuery(self, table_name)


class _FakeDownloader:
    def __init__(
        self,
        *,
        work_dir: str,
        transcript: dict[str, Any] | None,
        duration_seconds: int = 180,
    ):
        self.work_dir = work_dir
        self._transcript = transcript
        self._duration_seconds = duration_seconds
        self.extract_audio_calls = 0

    def download(self, _url: str, video_id: str):
        return {
            "path": str(Path(self.work_dir) / f"{video_id}.mp4"),
            "title": "Hardening Test Video",
            "duration": self._duration_seconds,
            "thumbnail": "https://example.com/thumb.jpg",
            "platform": "youtube",
            "external_id": "yt123",
        }

    def get_youtube_transcript(self, _video_id: str):
        return self._transcript

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
    monkeypatch.setattr(analyze_task_module, "create_work_dir", lambda _name: str(tmp_path))
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

    if transcriber_factory is None:
        transcriber_factory = lambda: SimpleNamespace(
            transcribe=lambda _audio_path: {
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


class _FakeAnthropicClient:
    def __init__(self, response_text: str):
        self._response_text = response_text
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **_kwargs):
        return SimpleNamespace(content=[SimpleNamespace(text=self._response_text)])


def test_ai_analyzer_salvages_valid_clips_from_mixed_payload(monkeypatch):
    response_text = """
```json
{
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
      "tags": ["one"]
    },
    {
      "rank": "bad-rank",
      "start_time": "oops",
      "end_time": 99.0,
      "duration": 30.0,
      "clip_title": "Invalid clip",
      "hook": "",
      "summary": "Invalid",
      "confidence_score": 0.4,
      "tags": []
    }
  ]
}
```
""".strip()

    monkeypatch.setattr(
        ai_analyzer_module,
        "Anthropic",
        lambda **_kwargs: _FakeAnthropicClient(response_text),
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
    monkeypatch.setattr(
        ai_analyzer_module,
        "Anthropic",
        lambda **_kwargs: _FakeAnthropicClient("No JSON payload here"),
    )
    analyzer = ai_analyzer_module.AIAnalyzer()

    with pytest.raises(Exception):
        analyzer.find_best_clips(
            transcript=_base_transcript(),
            num_clips=2,
            min_duration=40,
            max_duration=90,
        )

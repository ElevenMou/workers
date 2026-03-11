from __future__ import annotations
from types import SimpleNamespace

import pytest

from tasks.clips.helpers import lifecycle as lifecycle_helpers


def test_generated_clips_bucket_limit_env_override(monkeypatch):
    logger = SimpleNamespace(info=lambda *_a, **_k: None, warning=lambda *_a, **_k: None)
    monkeypatch.setenv("GENERATED_CLIPS_BUCKET_LIMIT_BYTES", "52428800")
    monkeypatch.setattr(lifecycle_helpers, "_generated_clips_bucket_limit_bytes", lifecycle_helpers._BUCKET_LIMIT_UNSET)
    monkeypatch.setattr(
        lifecycle_helpers.supabase.storage,
        "get_bucket",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("bucket metadata should not be queried")),
    )

    limit = lifecycle_helpers._get_generated_clips_bucket_limit_bytes(job_id="job-env", logger=logger)

    assert limit == 52_428_800


def test_upload_clip_reports_optimization_status_callback(monkeypatch, tmp_path):
    clip_path = tmp_path / "clip.mp4"
    clip_path.write_bytes(b"x" * 64)
    statuses: list[tuple[str, dict[str, object]]] = []

    monkeypatch.setattr(
        lifecycle_helpers,
        "_get_generated_clips_bucket_limit_bytes",
        lambda **_kwargs: None,
    )

    upload_calls = {"count": 0}

    def _upload(**_kwargs):
        upload_calls["count"] += 1
        if upload_calls["count"] == 1:
            raise Exception(
                {
                    "statusCode": 400,
                    "error": "payload too large",
                    "message": "maximum allowed size exceeded",
                }
            )

    def _reencode(**kwargs):
        output_path = kwargs["output_path"]
        with open(output_path, "wb") as handle:
            handle.write(b"y" * 32)

    monkeypatch.setattr(lifecycle_helpers, "_upload_with_duplicate_replace", _upload)
    monkeypatch.setattr(lifecycle_helpers, "_reencode_clip_for_upload", _reencode)

    uploaded_size = lifecycle_helpers.upload_clip_with_replace(
        local_clip_path=str(clip_path),
        storage_path="clips/test.mp4",
        job_id="job-callback",
        logger=SimpleNamespace(info=lambda *_a, **_k: None, warning=lambda *_a, **_k: None),
        status_callback=lambda status, detail: statuses.append((status, dict(detail))),
    )

    assert upload_calls["count"] == 2
    assert uploaded_size == 32
    assert statuses
    assert statuses[0][0] == "optimizing_upload"
    assert statuses[0][1]["attempt"] == 1
    assert statuses[0][1]["reason"] == "payload_too_large"


def test_upload_clip_disallow_reencode_fails_before_preupload_optimization(
    monkeypatch,
    tmp_path,
):
    clip_path = tmp_path / "clip.mp4"
    clip_path.write_bytes(b"x" * 64)
    reencode_called = {"value": False}

    monkeypatch.setattr(
        lifecycle_helpers,
        "_get_generated_clips_bucket_limit_bytes",
        lambda **_kwargs: 32,
    )
    monkeypatch.setattr(
        lifecycle_helpers,
        "_upload_with_duplicate_replace",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("upload should not be called")),
    )
    monkeypatch.setattr(
        lifecycle_helpers,
        "_reencode_clip_for_upload",
        lambda **_kwargs: reencode_called.__setitem__("value", True),
    )

    with pytest.raises(RuntimeError, match="quality-preserving mode disables upload re-encoding"):
        lifecycle_helpers.upload_clip_with_replace(
            local_clip_path=str(clip_path),
            storage_path="clips/test.mp4",
            job_id="job-1",
            logger=SimpleNamespace(info=lambda *_a, **_k: None, warning=lambda *_a, **_k: None),
            allow_reencode=False,
        )

    assert reencode_called["value"] is False


def test_upload_clip_disallow_reencode_fails_on_payload_too_large_without_retry(
    monkeypatch,
    tmp_path,
):
    clip_path = tmp_path / "clip.mp4"
    clip_path.write_bytes(b"x" * 64)
    reencode_called = {"value": False}

    monkeypatch.setattr(
        lifecycle_helpers,
        "_get_generated_clips_bucket_limit_bytes",
        lambda **_kwargs: None,
    )

    def _raise_payload_too_large(**_kwargs):
        raise Exception(
            {
                "statusCode": 400,
                "error": "payload too large",
                "message": "maximum allowed size exceeded",
            }
        )

    monkeypatch.setattr(
        lifecycle_helpers,
        "_upload_with_duplicate_replace",
        _raise_payload_too_large,
    )
    monkeypatch.setattr(
        lifecycle_helpers,
        "_reencode_clip_for_upload",
        lambda **_kwargs: reencode_called.__setitem__("value", True),
    )

    with pytest.raises(
        RuntimeError,
        match="quality-preserving mode disables fallback re-encoding",
    ):
        lifecycle_helpers.upload_clip_with_replace(
            local_clip_path=str(clip_path),
            storage_path="clips/test.mp4",
            job_id="job-2",
            logger=SimpleNamespace(info=lambda *_a, **_k: None, warning=lambda *_a, **_k: None),
            allow_reencode=False,
        )

    assert reencode_called["value"] is False

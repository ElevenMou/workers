from __future__ import annotations

from types import SimpleNamespace

import pytest

from tasks.clips.helpers import lifecycle as lifecycle_helpers


def test_upload_clip_disallow_reencode_fails_before_preupload_optimization(
    monkeypatch,
    tmp_path,
):
    clip_path = tmp_path / "clip.mp4"
    clip_path.write_bytes(b"x" * 64)
    reencode_called = {"value": False}

    monkeypatch.setattr(lifecycle_helpers, "prefer_local_media_storage", lambda: False)
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

    monkeypatch.setattr(lifecycle_helpers, "prefer_local_media_storage", lambda: False)
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

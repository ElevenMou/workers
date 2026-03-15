from __future__ import annotations

from types import SimpleNamespace

from tasks.clips.helpers import lifecycle as lifecycle_helpers


def test_upload_clip_with_replace_uses_minio_storage_even_when_reencode_disabled(
    monkeypatch,
    tmp_path,
):
    clip_path = tmp_path / "clip.mp4"
    clip_path.write_bytes(b"x" * 64)
    uploads: list[tuple[str, str]] = []
    infos: list[str] = []

    monkeypatch.setattr(
        lifecycle_helpers,
        "store_generated_clip",
        lambda *, local_clip_path, storage_path: uploads.append(
            (local_clip_path, storage_path)
        )
        or 64,
    )

    file_size = lifecycle_helpers.upload_clip_with_replace(
        local_clip_path=str(clip_path),
        storage_path="clips/test.mp4",
        job_id="job-1",
        logger=SimpleNamespace(
            info=lambda message, *_args: infos.append(message),
            warning=lambda *_args, **_kwargs: None,
        ),
        allow_reencode=False,
    )

    assert file_size == 64
    assert uploads == [(str(clip_path), "clips/test.mp4")]
    assert infos

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import services.social.media as social_media
from services.social.base import SocialProviderError


def test_download_clip_to_path_returns_publication_local_file_and_survives_shared_removal(
    tmp_path,
    monkeypatch,
):
    shared_path = tmp_path / "shared-cache" / "generated-clips" / "clips" / "clip-1.mp4"
    shared_path.parent.mkdir(parents=True, exist_ok=True)
    shared_path.write_bytes(b"video")

    monkeypatch.setattr(
        social_media,
        "resolve_generated_clip_path",
        lambda *_args, **_kwargs: str(shared_path),
    )

    work_dir = tmp_path / "publication-workdir"
    local_path = social_media.download_clip_to_path(
        "clips/clip-1.mp4",
        work_dir=str(work_dir),
    )

    assert Path(local_path).is_file()
    assert Path(local_path).parent == work_dir
    assert Path(local_path).read_bytes() == b"video"
    assert local_path != str(shared_path)

    shared_path.unlink()

    assert Path(local_path).read_bytes() == b"video"


def test_load_publication_media_maps_probe_failures_to_recoverable_storage_error(
    tmp_path,
    monkeypatch,
):
    local_path = tmp_path / "publication-workdir" / "clip-1.mp4"
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(b"video")

    monkeypatch.setattr(
        social_media,
        "download_clip_to_path",
        lambda *_args, **_kwargs: str(local_path),
    )

    def _raise_probe_failure(*_args, **_kwargs):
        raise subprocess.CalledProcessError(1, ["ffprobe", str(local_path)])

    monkeypatch.setattr(social_media.subprocess, "run", _raise_probe_failure)

    with pytest.raises(SocialProviderError) as exc:
        social_media.load_publication_media(
            "clips/clip-1.mp4",
            work_dir=str(tmp_path / "publication-workdir"),
        )

    assert exc.value.code == "clip_storage_unavailable"
    assert exc.value.recoverable is True
    assert exc.value.provider_payload["storage_reason"] == "probe_failed"
    assert exc.value.provider_payload["storage_bucket"] == "generated-clips"
    assert exc.value.provider_payload["storage_object"] == "clips/clip-1.mp4"

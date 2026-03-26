from __future__ import annotations

from pathlib import Path

import pytest

from tasks.videos import transcript as transcript_module


def test_transcribe_clip_window_with_whisper_rejects_media_without_audio(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.setattr(transcript_module, "probe_has_audio_stream", lambda _path: False)

    with pytest.raises(RuntimeError, match="no usable audio stream"):
        transcript_module.transcribe_clip_window_with_whisper(
            media_path=str(tmp_path / "source.mp4"),
            work_dir=str(tmp_path),
            clip_id="clip-1",
            start_time=5.0,
            end_time=10.0,
            video_duration_seconds=30.0,
        )

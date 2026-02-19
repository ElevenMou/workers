"""Shared media probe helpers for task modules."""

import ffmpeg as ffmpeg_lib


def probe_video_size(video_path: str) -> tuple[int, int]:
    """Return ``(width, height)`` for a local video file."""
    probe = ffmpeg_lib.probe(video_path)
    v_stream = next(
        (s for s in probe.get("streams", []) if s.get("codec_type") == "video"),
        None,
    )
    if v_stream is None:
        raise RuntimeError(f"No video stream found in {video_path}")
    return int(v_stream["width"]), int(v_stream["height"])


def probe_has_audio_stream(video_path: str) -> bool:
    """Return ``True`` when the file exposes at least one audio stream."""
    probe = ffmpeg_lib.probe(video_path)
    return any(s.get("codec_type") == "audio" for s in probe.get("streams", []))

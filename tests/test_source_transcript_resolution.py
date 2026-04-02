from __future__ import annotations

from tasks.videos.source_transcript import resolve_source_transcript


def test_resolve_source_transcript_falls_back_to_whisper_after_provider_probe_error():
    whisper_calls: list[str | None] = []
    whisper_transcript = {
        "source": "whisper",
        "language": "en",
        "languageCode": "en",
        "segments": [
            {"id": 0, "start": 0.0, "end": 1.0, "text": "hello world"},
        ],
    }

    class _Downloader:
        def get_provider_transcript(self, _url: str, *, preferred_languages=None):
            raise RuntimeError("provider probe failed")

    def _whisper_fallback(language_hint: str | None):
        whisper_calls.append(language_hint)
        return whisper_transcript, True

    resolution = resolve_source_transcript(
        existing_transcript=None,
        downloader=_Downloader(),
        source_url="https://vimeo.com/123456",
        source_platform="vimeo",
        source_external_id=None,
        source_detected_language="en",
        source_has_audio=True,
        whisper_fallback=_whisper_fallback,
        job_id="job-123",
    )

    assert whisper_calls == ["en"]
    assert resolution.transcript == whisper_transcript
    assert resolution.is_full_transcript is True
    assert resolution.diagnostics["transcript_source"] == "whisper"
    assert resolution.diagnostics["transcript_language"] == "en"
    assert (
        resolution.diagnostics["transcript_fallback_reason"]
        == "provider_caption_probe_failed:RuntimeError"
    )

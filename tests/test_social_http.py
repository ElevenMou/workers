from __future__ import annotations

import httpx

from services.social import http as social_http


class _FakeResponse:
    def __init__(self, status_code: int = 200):
        self.status_code = status_code


def test_resilient_request_rewinds_seekable_content_between_retries(tmp_path, monkeypatch):
    media_path = tmp_path / "clip.mp4"
    media_path.write_bytes(b"abcdef")
    attempts: list[bytes] = []

    def _fake_request(_method: str, _url: str, **kwargs):
        attempts.append(kwargs["content"].read())
        if len(attempts) == 1:
            raise httpx.ConnectTimeout("temporary upload failure")
        return _FakeResponse(200)

    monkeypatch.setattr(social_http.httpx, "request", _fake_request)
    monkeypatch.setattr(social_http.time, "sleep", lambda _seconds: None)

    with open(media_path, "rb") as handle:
        response = social_http.resilient_request(
            "POST",
            "https://example.com/upload",
            content=handle,
            max_retries=1,
        )

    assert response.status_code == 200
    assert attempts == [b"abcdef", b"abcdef"]


def test_resilient_request_rewinds_multipart_file_handles_between_retries(tmp_path, monkeypatch):
    media_path = tmp_path / "clip.mp4"
    media_path.write_bytes(b"abcdef")
    attempts: list[bytes] = []

    def _fake_request(_method: str, _url: str, **kwargs):
        attempts.append(kwargs["files"]["source"][1].read())
        if len(attempts) == 1:
            raise httpx.ConnectTimeout("temporary multipart failure")
        return _FakeResponse(200)

    monkeypatch.setattr(social_http.httpx, "request", _fake_request)
    monkeypatch.setattr(social_http.time, "sleep", lambda _seconds: None)

    with open(media_path, "rb") as handle:
        response = social_http.resilient_request(
            "POST",
            "https://example.com/upload",
            files={"source": ("clip.mp4", handle, "video/mp4")},
            max_retries=1,
        )

    assert response.status_code == 200
    assert attempts == [b"abcdef", b"abcdef"]

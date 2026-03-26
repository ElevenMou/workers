from __future__ import annotations

import shutil
import subprocess

import pytest


def _rendered_compose_config() -> str:
    if shutil.which("docker") is None:
        pytest.skip("docker is unavailable")

    result = subprocess.run(
        ["docker", "compose", "config"],
        cwd=".",
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip("docker compose config is unavailable in this environment")
    return result.stdout


def test_compose_passes_cookie_source_file_to_media_services():
    config = _rendered_compose_config()
    current_service: str | None = None
    seen = {service_name: False for service_name in ("api", "video-workers", "clip-workers")}

    for line in config.splitlines():
        stripped = line.lstrip()
        indent = len(line) - len(stripped)

        if indent == 2 and stripped.endswith(":") and stripped[:-1] in seen:
            current_service = stripped[:-1]
            continue

        if (
            current_service in seen
            and indent == 6
            and stripped.startswith("YTDLP_COOKIES_SOURCE_FILE:")
        ):
            value = stripped.split(":", 1)[1].strip()
            seen[current_service] = bool(value)

    assert all(seen.values()), seen

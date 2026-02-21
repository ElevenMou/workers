"""Helpers for creating writable per-job work directories."""

from __future__ import annotations

import os
import tempfile

from config import TEMP_DIR


def create_work_dir(folder_name: str) -> str:
    """Create a per-job work directory.

    Prefer ``TEMP_DIR`` from config. If it is not writable (for example due to
    Docker volume ownership mismatch), transparently fall back to the process
    temp directory.
    """
    preferred_base = (TEMP_DIR or "").strip() or "/tmp/video_clipper"
    temp_base = tempfile.gettempdir()
    home_base = os.path.expanduser("~")

    # Keep several fallback candidates because TEMP_DIR may be a mounted path
    # with incompatible ownership/permissions inside the container.
    candidates = [
        preferred_base,
        os.path.join(temp_base, "video_clipper_runtime"),
        os.path.join(home_base, ".cache", "video_clipper"),
        os.path.join(home_base, "video_clipper_tmp"),
    ]

    # Remove duplicates while preserving order.
    seen: set[str] = set()
    unique_candidates: list[str] = []
    for base_dir in candidates:
        if base_dir and base_dir not in seen:
            seen.add(base_dir)
            unique_candidates.append(base_dir)

    for base_dir in unique_candidates:
        try:
            os.makedirs(base_dir, exist_ok=True)
            work_dir = os.path.join(base_dir, folder_name)
            os.makedirs(work_dir, exist_ok=True)
            return work_dir
        except PermissionError:
            continue

    attempted = ", ".join(repr(path) for path in unique_candidates)
    raise PermissionError(
        f"Unable to create writable work dir. Attempted bases: {attempted}"
    )

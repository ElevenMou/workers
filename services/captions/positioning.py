"""Caption positioning helpers."""

from __future__ import annotations


def _as_int(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def compute_video_anchored_margin_v(
    *,
    position: str,
    canvas_h: int,
    vid_y: int | None,
    vid_h: int | None,
    inset: int,
) -> int | None:
    """Return ASS ``margin_v`` anchored to the source video region.

    This only applies to explicit ``top`` and ``bottom`` positions.
    """
    position_key = str(position or "").strip().lower()
    if position_key not in {"top", "bottom"}:
        return None

    ch = max(1, _as_int(canvas_h, 0))
    y = _as_int(vid_y, -1)
    h = _as_int(vid_h, -1)
    if y < 0 or h <= 0:
        return None

    safe_inset = max(0, _as_int(inset, 0))
    video_top = max(0, min(y, ch))
    video_bottom = max(video_top, min(y + h, ch))

    if position_key == "top":
        return max(0, min(video_top + safe_inset, ch))

    return max(0, min((ch - video_bottom) + safe_inset, ch))


__all__ = ["compute_video_anchored_margin_v"]


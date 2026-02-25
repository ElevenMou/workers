from __future__ import annotations

from services.clips.layout import compute_video_position
from tasks.clips.helpers.captions import _overrides_from_layout


def test_compute_video_position_uses_custom_x_y_width():
    vid_w, vid_h, vid_x, vid_y = compute_video_position(
        src_w=1920,
        src_h=1080,
        width_pct=100,
        position_y="custom",
        custom_x=120,
        custom_y=340,
        custom_width=720,
        canvas_w=1080,
        canvas_h=1920,
        video_scale_mode="fit",
    )

    assert vid_w == 720
    assert vid_h == 404
    assert vid_x == 120
    assert vid_y == 340


def test_caption_overrides_build_custom_position_y():
    overrides = _overrides_from_layout(
        {
            "position": "custom",
            "customY": 250,
            "customWidth": 500,
        },
        canvas_w=1080,
        canvas_h=1920,
    )

    assert overrides["position"] == "auto"
    assert overrides["alignment"] == 8
    assert overrides["margin_v"] == 250
    assert overrides["safe_margin_y"] == 250

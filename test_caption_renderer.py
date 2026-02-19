import os
import tempfile

from services.caption_renderer import (
    CAPTION_TEMPLATE_DEFAULTS,
    compute_video_anchored_margin_v,
    get_preset,
    generate_ass_content,
    list_animation_presets,
    list_caption_presets,
    list_supported_styles,
    normalize_caption_style,
    render_ass,
)


def _sample_transcript():
    return {
        "segments": [
            {
                "start": 0.0,
                "end": 2.8,
                "text": "This is powerful caption rendering",
                "words": [
                    {"word": "This", "start": 0.0, "end": 0.5},
                    {"word": "is", "start": 0.5, "end": 0.8},
                    {"word": "powerful", "start": 0.8, "end": 1.4},
                    {"word": "caption", "start": 1.4, "end": 2.0},
                    {"word": "rendering", "start": 2.0, "end": 2.8},
                ],
            }
        ]
    }


def run():
    modes = set(list_supported_styles())
    expected = {
        "impact_bold",
        "focus_highlight",
        "clean_fade",
        "bold_box",
        "classic_fade",
        "cinematic_subtitle",
        "neon_pulse",
        "minimal_presenter",
        "elegant_lower_third",
        "dynamic_karaoke",
        "glow_edge",
        "typewriter_clean",
        "news_strip",
        "retro_arcade",
        "soft_serif",
        "contrast_max",
        "split_dual_tone",
        "quiet_doc",
    }
    assert modes == expected

    assert normalize_caption_style("animated") == "dynamic_karaoke"
    assert normalize_caption_style("classic") == "classic_fade"
    assert normalize_caption_style("split") == "elegant_lower_third"
    assert normalize_caption_style("caps") == "impact_bold"
    assert normalize_caption_style("unknown_style") == "classic_fade"
    assert normalize_caption_style("typewriter") == "typewriter_clean"
    assert normalize_caption_style("high_contrast") == "contrast_max"
    assert normalize_caption_style("documentary_quiet") == "quiet_doc"

    animations = set(list_animation_presets())
    assert animations == {"none", "fade", "slide_up", "pop", "karaoke", "scale"}
    assert get_preset("impact_bold").name == "impact_bold"

    presets = list_caption_presets()
    preset_ids = {p["id"] for p in presets}
    assert preset_ids == expected
    assert all("captions" in p for p in presets)
    assert all("style" in p for p in presets)
    assert all("presetName" in p["captions"] for p in presets)
    assert all(p["captions"].get("show") is True for p in presets)
    assert all(p["captions"].get("position") == "bottom" for p in presets)
    assert all(set(CAPTION_TEMPLATE_DEFAULTS).issubset(set(p["captions"])) for p in presets)

    # Position top/bottom should anchor against the video window (not full canvas).
    assert (
        compute_video_anchored_margin_v(
            position="top",
            canvas_h=1920,
            vid_y=420,
            vid_h=1080,
            inset=90,
        )
        == 510
    )
    assert (
        compute_video_anchored_margin_v(
            position="bottom",
            canvas_h=1920,
            vid_y=420,
            vid_h=1080,
            inset=90,
        )
        == 510
    )

    transcript = _sample_transcript()
    karaoke_ass = generate_ass_content(transcript, "dynamic_karaoke")
    assert "[Script Info]" in karaoke_ass
    assert "[V4+ Styles]" in karaoke_ass
    assert "[Events]" in karaoke_ass
    assert r"{\k" in karaoke_ass

    line_ass = generate_ass_content(
        {"segments": [{"start": 0.0, "end": 1.5, "text": "Line level fallback works"}]},
        "classic_fade",
    )
    assert "Dialogue:" in line_ass
    assert r"{\k" not in line_ass

    highlight_line_ass = generate_ass_content(
        {"segments": [{"start": 0.0, "end": 1.5, "text": "Word highlight fallback works"}]},
        "dynamic_karaoke",
    )
    assert "Dialogue:" in highlight_line_ass
    assert r"{\k" in highlight_line_ass

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = os.path.join(tmp_dir, "captions.ass")
        result_path = render_ass(
            transcript["segments"],
            style="dynamic_karaoke",
            animation="karaoke",
            output_path=out_path,
        )
        assert result_path == out_path
        assert os.path.isfile(out_path)
        with open(out_path, "r", encoding="utf-8") as handle:
            content = handle.read()
        assert "Dialogue:" in content
        assert r"{\k" in content

    print("ok: caption preset renderer")


if __name__ == "__main__":
    run()

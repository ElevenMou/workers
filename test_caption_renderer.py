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
    required_modes = {
        "accessible_box",
        "boxed",
        "calm_karaoke",
        "clean",
        "contrast",
        "karaoke_gold",
        "mrbeast",
        "podcast",
        "readable",
        "speaker_focus",
        "whisper",
    }
    assert required_modes.issubset(modes)

    assert normalize_caption_style("animated") == "karaoke_gold"
    assert normalize_caption_style("accessible") == "accessible_box"
    assert normalize_caption_style("impact_bold") == "contrast"
    assert normalize_caption_style("typewriter_clean") == "clean"
    assert normalize_caption_style("unknown_style") == CAPTION_TEMPLATE_DEFAULTS["presetName"]

    animations = set(list_animation_presets())
    assert {
        "none",
        "fade",
        "pop",
        "slide_up",
        "bounce",
        "glow",
    }.issubset(animations)
    assert get_preset("contrast").name == "contrast"

    presets = list_caption_presets()
    preset_ids = {p["id"] for p in presets}
    assert required_modes.issubset(preset_ids)
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
    karaoke_ass = generate_ass_content(transcript, "karaoke_gold")
    assert "[Script Info]" in karaoke_ass
    assert "[V4+ Styles]" in karaoke_ass
    assert "[Events]" in karaoke_ass
    assert r"{\k" in karaoke_ass

    line_ass = generate_ass_content(
        {"segments": [{"start": 0.0, "end": 1.5, "text": "Line level fallback works"}]},
        "clean",
    )
    assert "Dialogue:" in line_ass
    assert r"{\k" not in line_ass

    highlight_ass = generate_ass_content(
        transcript,
        "clean",
        overrides={
            "style": "highlight",
            "word_highlight": True,
            "max_chars_per_line": 120,
            "max_lines": 2,
        },
    )
    assert highlight_ass.count("Dialogue:") == len(transcript["segments"][0]["words"])
    assert r"{\k" not in highlight_ass
    assert "Style: HighlightWord," in highlight_ass
    assert r"{\rHighlightWord" in highlight_ass

    highlight_box_ass = generate_ass_content(
        transcript,
        "clean",
        overrides={
            "style": "highlight_box",
            "word_highlight": True,
            "max_chars_per_line": 120,
            "max_lines": 2,
        },
    )
    assert highlight_box_ass.count("Dialogue:") == len(transcript["segments"][0]["words"])
    assert "Style: HighlightBox," in highlight_box_ass
    assert r"{\rHighlightBox" in highlight_box_ass

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = os.path.join(tmp_dir, "captions.ass")
        result_path = render_ass(
            transcript["segments"],
            style="dynamic_karaoke",
            animation="none",
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

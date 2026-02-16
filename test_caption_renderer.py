import os
import tempfile

from services.caption_renderer import (
    CAPTION_TEMPLATE_DEFAULTS,
    list_animation_presets,
    list_caption_presets,
    list_supported_styles,
    normalize_caption_style,
    render_ass,
)


def _sample_segments():
    return [
        {
            "start": 0.0,
            "end": 2.0,
            "text": "Hello world from clipry",
            "words": [
                {"word": "Hello", "start": 0.0, "end": 0.5},
                {"word": "world", "start": 0.5, "end": 1.0},
                {"word": "from", "start": 1.0, "end": 1.4},
                {"word": "clipry", "start": 1.4, "end": 2.0},
            ],
        }
    ]


def run():
    modes = set(list_supported_styles())
    assert len(modes) >= 10
    assert {
        "animated",
        "static",
        "grouped",
        "word_by_word",
        "progressive",
        "two_line",
        "punctuated",
        "uppercase",
        "lowercase",
        "headline",
    } <= modes

    assert normalize_caption_style("karaoke") == "animated"
    assert normalize_caption_style("classic") == "static"
    assert normalize_caption_style("minimal") == "grouped"
    assert normalize_caption_style("focus_word") == "word_by_word"
    assert normalize_caption_style("build") == "progressive"
    assert normalize_caption_style("split") == "two_line"
    assert normalize_caption_style("sentence") == "punctuated"
    assert normalize_caption_style("caps") == "uppercase"
    assert normalize_caption_style("lower") == "lowercase"
    assert normalize_caption_style("title_case") == "headline"
    assert normalize_caption_style("unknown_style") == "static"

    animations = list_animation_presets()
    assert len(animations) >= 20
    assert "bounce" in animations
    assert "glitch_infinite_zoom" in animations

    presets = list_caption_presets()
    assert len(presets) == 20
    preset_ids = {p["id"] for p in presets}
    assert preset_ids == {
        "word_by_word_highlight",
        "karaoke_sweep",
        "pop_animation",
        "minimal_clean",
        "neon_glow",
        "typewriter_effect",
        "bounce_in",
        "slide_up",
        "center_word_focus",
        "conveyor_belt_ticker",
        "bold_punch",
        "soft_pill_highlight",
        "classic_clean_outline",
        "brand_bar",
        "energetic_pop_pulse",
        "dramatic_fade_in",
        "news_ticker",
        "two_line_stylish",
        "uppercase_bold_impact",
        "progressive_reveal",
    }
    assert all("captions" in p for p in presets)
    assert all("preset" not in p["captions"] for p in presets)
    assert all("style" in p["captions"] for p in presets)
    assert all("animation" in p["captions"] for p in presets)
    assert all(set(CAPTION_TEMPLATE_DEFAULTS).issubset(set(p["captions"])) for p in presets)
    assert all(
        p["captions"]["fontCase"] in {"as_typed", "uppercase", "lowercase"}
        for p in presets
    )
    assert all(
        p["captions"]["style"]
        in {
            "animated",
            "grouped",
            "headline",
            "lowercase",
            "progressive",
            "punctuated",
            "static",
            "two_line",
            "uppercase",
            "word_by_word",
        }
        for p in presets
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        segments = _sample_segments()
        for style in sorted(modes):
            out_path = os.path.join(tmp_dir, f"{style}.ass")
            result_path = render_ass(
                segments,
                style=style,
                animation="bounce",
                lines_per_page=2,
                font_case="as_typed",
                max_words_per_line=2,
                output_path=out_path,
            )
            assert result_path == out_path
            assert os.path.isfile(out_path)
            with open(out_path, "r", encoding="utf-8") as f:
                content = f.read()
            assert "[Events]" in content
            assert "Dialogue:" in content

    print("ok: caption renderer modes")


if __name__ == "__main__":
    run()

import os
import tempfile

from services.caption_renderer import (
    CAPTION_TEMPLATE_DEFAULTS,
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
    assert modes == {
        "clean_minimal",
        "bold_tiktok",
        "word_highlighted",
        "cinematic_lower_third",
        "block_background",
    }

    assert normalize_caption_style("animated") == "word_highlighted"
    assert normalize_caption_style("classic") == "clean_minimal"
    assert normalize_caption_style("split") == "cinematic_lower_third"
    assert normalize_caption_style("caps") == "bold_tiktok"
    assert normalize_caption_style("unknown_style") == "clean_minimal"

    animations = set(list_animation_presets())
    assert animations == {"none", "fade", "slide_up", "pop", "karaoke"}

    presets = list_caption_presets()
    preset_ids = {p["id"] for p in presets}
    assert preset_ids == modes
    assert all("captions" in p for p in presets)
    assert all("style" in p for p in presets)
    assert all("presetName" in p["captions"] for p in presets)
    assert all(set(CAPTION_TEMPLATE_DEFAULTS).issubset(set(p["captions"])) for p in presets)

    transcript = _sample_transcript()
    karaoke_ass = generate_ass_content(transcript, "word_highlighted")
    assert "[Script Info]" in karaoke_ass
    assert "[V4+ Styles]" in karaoke_ass
    assert "[Events]" in karaoke_ass
    assert r"{\k" in karaoke_ass

    line_ass = generate_ass_content(
        {"segments": [{"start": 0.0, "end": 1.5, "text": "Line level fallback works"}]},
        "clean_minimal",
    )
    assert "Dialogue:" in line_ass
    assert r"{\k" not in line_ass

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = os.path.join(tmp_dir, "captions.ass")
        result_path = render_ass(
            transcript["segments"],
            style="word_highlighted",
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

from services.captions.caption_presets import resolve_caption_preset, resolve_preset


def test_karaoke_preset_maps_colors_for_ass_fill_direction():
    preset = resolve_preset("karaoke_gold")
    # In ASS karaoke, fill sweeps Secondary -> Primary. For UI semantics
    # this means:
    # - primary should be highlight
    # - secondary should be base text color
    assert preset["primary_color"] == preset["highlight_color"]
    assert preset["secondary_color"] == "&H00FFFFFF"


def test_word_highlight_grouped_maps_colors_like_karaoke():
    preset = resolve_preset(
        "clean",
        overrides={
            "style": "grouped",
            "word_highlight": True,
            "primary_color": "&H00FFFFFF",
            "highlight_color": "&H0000D7FF",
        },
    )
    assert preset["primary_color"] == "&H0000D7FF"
    assert preset["secondary_color"] == "&H00FFFFFF"


def test_word_by_word_does_not_swap_primary_and_secondary():
    preset = resolve_preset(
        "clean",
        overrides={
            "style": "word_by_word",
            "word_highlight": True,
            "primary_color": "&H00FFFFFF",
            "highlight_color": "&H0000D7FF",
        },
    )
    assert preset["primary_color"] == "&H00FFFFFF"


def test_frontend_caption_payload_keeps_font_color_semantics_for_karaoke():
    payload = resolve_caption_preset("karaoke_gold")
    assert payload["fontColor"] == "&H00FFFFFF"
    assert payload["highlightColor"] == "&H0000D7FF"

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

if "cv2" not in sys.modules:
    cv2_stub = ModuleType("cv2")
    cv2_stub.CAP_PROP_POS_FRAMES = 1
    cv2_stub.CAP_PROP_FPS = 5
    cv2_stub.CAP_PROP_FRAME_COUNT = 7
    cv2_stub.COLOR_BGR2RGB = 0
    cv2_stub.VideoCapture = object
    cv2_stub.cvtColor = lambda frame, _code: frame
    cv2_stub.resize = lambda frame, _size: frame
    sys.modules["cv2"] = cv2_stub

if "numpy" not in sys.modules:
    sys.modules["numpy"] = ModuleType("numpy")

from services.clips import ffmpeg_ops
from services.clips.generator import ClipGenerator
from services.reframe import reframer as reframer_module


def _write_minimal_ass(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "[Script Info]",
                "ScriptType: v4.00+",
                "[V4+ Styles]",
                (
                    "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
                    "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
                    "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
                    "Alignment, MarginL, MarginR, MarginV, Encoding"
                ),
                (
                    "Style: Default,Arial,48,&H00FFFFFF,&H00000000,&H00000000,"
                    "&H00000000,0,0,0,0,100,100,0,0,1,0,0,2,10,10,10,1"
                ),
                "[Events]",
                "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
                "Dialogue: 0,0:00:00.00,0:00:05.00,Default,,0,0,0,,hello",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _compile_compose(monkeypatch, tmp_path: Path, **overrides) -> list[str]:
    compiled: list[list[str]] = []

    def _fake_run(stream, **_kwargs):
        compiled.append(stream.compile())
        return None, None

    monkeypatch.setattr(ffmpeg_ops, "_run_ffmpeg_with_timeout", _fake_run)
    monkeypatch.setattr(ffmpeg_ops, "_media_has_audio", lambda _path: True)
    monkeypatch.setattr(ffmpeg_ops, "_resolve_caption_fonts_dir", lambda: None)

    tmp_path.mkdir(parents=True, exist_ok=True)
    caption_path = tmp_path / "captions.ass"
    overlay_path = tmp_path / "overlay.png"
    background_image_path = tmp_path / "bg.png"
    _write_minimal_ass(caption_path)
    overlay_path.write_bytes(b"png")
    background_image_path.write_bytes(b"png")

    kwargs = {
        "input_path": "input.mp4",
        "output_path": str(tmp_path / "out.mp4"),
        "style": "blur",
        "canvas_w": 1080,
        "canvas_h": 1920,
        "vid_w": 1080,
        "vid_h": 1920,
        "vid_x": 0,
        "vid_y": 0,
        "blur_strength": 20,
        "video_scale_mode": "fill",
        "title_lines": ["A title line"],
        "title_show": True,
        "title_font_size": 48,
        "title_font_color": "#FFFFFF",
        "title_font_family": "Arial",
        "title_align": "center",
        "title_stroke_width": 0,
        "title_stroke_color": "#000000",
        "title_bar_enabled": True,
        "title_bar_color": "#000000",
        "title_bar_x": 0,
        "title_bar_w": 1080,
        "title_padding_x": 16,
        "title_bar_y": 190,
        "title_text_y": 330,
        "title_bar_h": 280,
        "caption_ass_path": str(caption_path),
        "qp": {"crf": 18, "preset": "medium"},
        "overlay_file_path": str(overlay_path),
        "overlay_cfg": {"enabled": True, "x": 24, "y": 32, "widthPx": 200},
        "background_color": "#000000",
        "background_image_path": str(background_image_path),
        "start_time": 10.0,
        "end_time": 15.0,
    }
    kwargs.update(overrides)

    ffmpeg_ops.compose_clip(**kwargs)

    assert compiled
    return compiled[0]


def test_compose_clip_blur_fill_compiles_with_split_and_overlays(monkeypatch, tmp_path: Path):
    cmd = _compile_compose(monkeypatch, tmp_path)
    cmd_str = " ".join(cmd)

    assert "split=2" in cmd_str
    assert "boxblur=20" in cmd_str
    assert "ass=filename=" in cmd_str


def test_compose_clip_blur_mixed_dimensions_compiles(monkeypatch, tmp_path: Path):
    cmd = _compile_compose(
        monkeypatch,
        tmp_path,
        vid_w=720,
        vid_h=1280,
        vid_x=180,
        vid_y=320,
    )
    cmd_str = " ".join(cmd)

    assert "split=2" in cmd_str
    assert "overlay=180:320" in cmd_str


def test_compose_clip_other_background_styles_compile(monkeypatch, tmp_path: Path):
    solid_cmd = _compile_compose(monkeypatch, tmp_path / "solid", style="solid_color")
    image_cmd = _compile_compose(monkeypatch, tmp_path / "image", style="image")

    assert "color=c=#000000:s=1080x1920:d=5.0" in " ".join(solid_cmd)
    assert "bg.png" in " ".join(image_cmd)


def test_generator_passes_reframe_window_and_uses_segment_relative_seek(monkeypatch, tmp_path: Path):
    layout = {
        "canvas_w": 1080,
        "canvas_h": 1920,
        "vid_w": 1080,
        "vid_h": 1920,
        "vid_x": 0,
        "vid_y": 0,
        "title_padding_x": 16,
        "title_bar_x": 0,
        "title_bar_w": 1080,
        "title_bar_y": 190,
        "title_bar_h": 280,
        "title_text_y": 330,
    }
    calls: dict[str, object] = {}

    def _fake_probe(path: str) -> tuple[int, int]:
        if path.endswith("_reframed.mp4"):
            return (140, 250)
        return (640, 360)

    def _fake_reframe(input_path: str, output_path: str, **kwargs):
        calls["reframe"] = {
            "input_path": input_path,
            "output_path": output_path,
            **kwargs,
        }
        Path(output_path).write_bytes(b"reframed")
        return True

    def _fake_compose(input_path: str, output_path: str, **kwargs):
        calls["compose"] = {
            "input_path": input_path,
            "output_path": output_path,
            **kwargs,
        }
        Path(output_path).write_bytes(b"composited")

    monkeypatch.setattr("services.clips.generator._probe_video_resolution", _fake_probe)
    monkeypatch.setattr("services.clips.generator.wrap_title", lambda *_args, **_kwargs: ["Title"])
    monkeypatch.setattr("services.clips.generator.compute_layout", lambda *_args, **_kwargs: dict(layout))
    monkeypatch.setattr("services.clips.generator.compose_clip", _fake_compose)
    monkeypatch.setattr(reframer_module, "reframe_video", _fake_reframe)

    result = ClipGenerator(work_dir=str(tmp_path)).generate(
        video_path="source.mp4",
        clip_id="clip-123",
        start_time=530.0,
        end_time=603.0,
        title="Title",
        background_style="blur",
        video_scale_mode="fill",
        reframe_enabled=True,
    )

    assert calls["reframe"]["start_time"] == 530.0
    assert calls["reframe"]["end_time"] == 603.0
    assert calls["compose"]["input_path"].endswith("clip-123_reframed.mp4")
    assert calls["compose"]["start_time"] == 0.0
    assert calls["compose"]["end_time"] == 73.0
    assert result["file_size"] == len(b"composited")

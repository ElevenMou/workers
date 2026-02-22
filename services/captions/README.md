# ASS Caption Rendering

## Core API

```python
from services.captions.renderer import render_captions

render_captions(
    video_path="input.mp4",
    transcript_json=transcript,
    preset_name="word_highlighted",
    output_path="output.mp4",
    video_aspect_ratio="9:16",
)
```

Compatibility note:
- `services.caption_renderer` remains as a facade for legacy imports.
- New code should import from `services.captions.*` modules directly.

## Preset Example

```python
{
    "font_name": "Montserrat-Bold",
    "font_size": 72,
    "primary_color": "&H00FFFFFF",
    "outline_color": "&H00000000",
    "back_color": "&H80000000",
    "alignment": 2,
    "margin_v": 80,
    "animation": {"type": "fade", "duration": 0.3},
    "word_highlight": True,
}
```

## Sample ASS Output

```ass
[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Style: Default,Montserrat-Bold,72,&H00FFFFFF,&H0000C8FF,&H00000000,&H68000000,-1,0,0,0,100,100,0,0,1,3,1,2,56,56,96,1

[Events]
Dialogue: 0,00:00:00.52,00:00:01.20,Default,,0,0,0,,{\k20}This {\k9}is {\k39}powerful
```

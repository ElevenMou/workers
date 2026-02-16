"""Animation helpers for ASS caption rendering."""

from services.captions.presets import _normalize_animation


def _animation_prefix(animation: str, start: float, end: float) -> str:
    anim = _normalize_animation(animation)
    if anim == "none":
        return ""

    duration_ms = max(160, int((end - start) * 1000))
    t1 = max(80, duration_ms // 4)
    t2 = max(t1 + 80, duration_ms // 2)
    t3 = max(t2 + 80, int(duration_ms * 0.85))

    if anim == "bounce":
        return f"{{\\t(0,{t1},\\fscx112\\fscy112)\\t({t1},{t2},\\fscx100\\fscy100)}}"
    if anim == "underline":
        return r"{\u1}"
    if anim == "box":
        return r"{\bord10\shad0\3c&H00202020&}"
    if anim == "pop":
        return f"{{\\fscx70\\fscy70\\t(0,{t1},\\fscx100\\fscy100)}}"
    if anim == "scale":
        return f"{{\\fscx120\\fscy120\\t(0,{t2},\\fscx100\\fscy100)}}"
    if anim == "slide_left":
        return f"{{\\fsp22\\t(0,{t2},\\fsp0)}}"
    if anim == "slide_up":
        return f"{{\\fscy85\\t(0,{t2},\\fscy100)}}"
    if anim == "seamless_bounce":
        return (
            f"{{\\t(0,{t1},\\fscx104\\fscy104)"
            f"\\t({t1},{t2},\\fscx100\\fscy100)"
            f"\\t({t2},{t3},\\fscx104\\fscy104)}}"
        )
    if anim == "highlighter_box_around":
        return r"{\bord8\shad0\3c&H0030C8FF&}"
    if anim == "blur_switch":
        return f"{{\\blur7\\t(0,{t2},\\blur0)}}"
    if anim == "baby_earthquake":
        return (
            f"{{\\frz2\\t(0,{t1},\\frz-2)\\t({t1},{t2},\\frz1)\\t({t2},{t3},\\frz0)}}"
        )
    if anim == "glitch_infinite_zoom":
        return (
            f"{{\\fscx104\\fscy104\\frz-1"
            f"\\t(0,{t2},\\fscx100\\fscy100\\frz1)"
            f"\\t({t2},{duration_ms},\\fscx104\\fscy104\\frz-1)}}"
        )
    if anim == "focus":
        return f"{{\\alpha&H88&\\t(0,{t2},\\alpha&H00&)}}"
    if anim == "blur_in":
        return f"{{\\blur9\\t(0,{t2},\\blur0)}}"
    if anim == "with_backdrop":
        return r"{\bord8\shad2\3c&H00000000&\4c&H64000000&}"
    if anim == "soft_landing":
        return f"{{\\fscx94\\fscy94\\t(0,{t2},\\fscx100\\fscy100)}}"
    if anim == "baby_steps":
        return (
            f"{{\\fscx92\\fscy92\\t(0,{t1},\\fscx96\\fscy96)"
            f"\\t({t1},{t2},\\fscx100\\fscy100)}}"
        )
    if anim == "grow":
        return f"{{\\fscx78\\fscy78\\t(0,{t2},\\fscx100\\fscy100)}}"
    if anim == "breathe":
        return (
            f"{{\\fscx102\\fscy102\\t(0,{t2},\\fscx98\\fscy98)"
            f"\\t({t2},{duration_ms},\\fscx102\\fscy102)}}"
        )

    # Preset-facing animations with stronger visual separation.
    if anim == "karaoke_sweep":
        return (
            f"{{\\u1\\bord4\\fscx106\\fscy106"
            f"\\t(0,{t1},\\fscx104\\fscy104)"
            f"\\t({t1},{t2},\\fscx100\\fscy100)}}"
        )
    if anim == "typewriter":
        return f"{{\\alpha&HFF&\\fsp28\\t(0,{t2},\\alpha&H00&\\fsp0)}}"
    if anim == "neon_glow":
        return f"{{\\blur14\\t(0,{t2},\\blur4)}}"
    if anim == "conveyor_left":
        return (
            f"{{\\fsp120"
            f"\\t(0,{duration_ms},\\fsp-8)"
            f"\\t({max(duration_ms - 120, 0)},{duration_ms},\\alpha&H22&)}}"
        )
    if anim == "punch":
        return (
            f"{{\\fscx150\\fscy150\\frz-5"
            f"\\t(0,{t1},\\fscx95\\fscy95\\frz4)"
            f"\\t({t1},{t2},\\fscx108\\fscy108\\frz-2)"
            f"\\t({t2},{t3},\\fscx100\\fscy100\\frz0)}}"
        )
    if anim == "pill_highlight":
        return r"{\bord16\shad0\3c&H00A6E0FF&\4c&H00000000&}"
    if anim == "brand_bar":
        return r"{\bord18\shad0\3c&H00202020&\4c&H88000000&\fsp1}"
    if anim == "pulse_pop":
        return (
            f"{{\\fscx75\\fscy75"
            f"\\t(0,{t1},\\fscx115\\fscy115)"
            f"\\t({t1},{t2},\\fscx98\\fscy98)"
            f"\\t({t2},{duration_ms},\\fscx103\\fscy103)}}"
        )
    if anim == "fade_in":
        return f"{{\\alpha&HFF&\\blur10\\t(0,{t2},\\alpha&H00&\\blur0)}}"
    if anim == "news_ticker":
        return (
            f"{{\\fsp80\\bord10\\shad0\\3c&H001A1A1A&"
            f"\\t(0,{duration_ms},\\fsp4)}}"
        )
    if anim == "impact":
        return (
            f"{{\\fscx170\\fscy170\\frz3"
            f"\\t(0,{t1},\\fscx90\\fscy90\\frz-2)"
            f"\\t({t1},{t2},\\fscx105\\fscy105\\frz1)"
            f"\\t({t2},{t3},\\fscx100\\fscy100\\frz0)}}"
        )
    if anim == "progressive_reveal":
        return (
            f"{{\\alpha&H88&\\fscx70\\fscy70"
            f"\\t(0,{t2},\\alpha&H00&\\fscx100\\fscy100)}}"
        )
    return ""


def _apply_animation(text: str, animation: str, start: float, end: float) -> str:
    normalized = _normalize_animation(animation)
    prefix = _animation_prefix(animation, start, end)
    if not prefix:
        return text
    if normalized in {"underline", "karaoke_sweep"}:
        return f"{prefix}{text}" + r"{\u0}"
    return f"{prefix}{text}"

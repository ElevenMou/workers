"""Transcript segment extraction and word-timing helpers."""

from typing import Any

def _words_from_segment(seg: dict) -> list[dict]:
    """Return a list of ``{word, start, end}`` dicts for *seg*.

    If the segment comes from Whisper it already has a ``words`` key.
    For YouTube transcripts we split the text and distribute timing
    evenly across the segment duration.
    """
    if "words" in seg and seg["words"]:
        return [
            {
                "word": w.get("word", w.get("text", "")).strip(),
                "start": float(w["start"]),
                "end": float(w["end"]),
            }
            for w in seg["words"]
            if w.get("word", w.get("text", "")).strip()
        ]

    # Estimate word timing from sentence-level timestamps
    text = seg.get("text", "")
    words = text.split()
    if not words:
        return []

    seg_start = float(seg["start"])
    seg_end = float(seg["end"])
    duration = seg_end - seg_start
    per_word = duration / len(words) if len(words) > 0 else duration

    return [
        {
            "word": w,
            "start": seg_start + i * per_word,
            "end": seg_start + (i + 1) * per_word,
        }
        for i, w in enumerate(words)
    ]


# ---------------------------------------------------------------------------
# Segment extraction for a clip time-range
# ---------------------------------------------------------------------------


def extract_clip_segments(
    transcript: dict,
    clip_start: float,
    clip_end: float,
) -> list[dict]:
    """Return only the transcript segments that overlap ``[clip_start, clip_end]``
    with timestamps shifted so ``clip_start`` becomes ``0``.

    YouTube transcripts often have overlapping and long segments.
    This function de-overlaps them and splits long segments into
    shorter sentence-level chunks so captions display cleanly.
    """

    min_duration = 0.01

    raw: list[dict] = []
    for seg in transcript.get("segments", []):
        seg_start = float(seg["start"])
        seg_end = float(seg["end"])

        # Skip segments entirely outside the clip range
        if seg_end <= clip_start or seg_start >= clip_end:
            continue

        # Clamp to clip boundaries
        clamped_start = max(seg_start, clip_start)
        clamped_end = min(seg_end, clip_end)
        if (clamped_end - clamped_start) < min_duration:
            continue

        rel_start = clamped_start - clip_start
        rel_end = clamped_end - clip_start
        new_seg: dict[str, Any] = {
            "start": rel_start,
            "end": rel_end,
            "text": str(seg.get("text", "")).strip(),
        }

        # Shift word-level timing too and drop degenerate words.
        if "words" in seg and seg["words"]:
            clipped_words = []
            for w in seg["words"]:
                ws = max(float(w["start"]), clip_start) - clip_start
                we = min(float(w["end"]), clip_end) - clip_start
                if (we - ws) < min_duration:
                    continue
                token = str(w.get("word", w.get("text", ""))).strip()
                if not token:
                    continue
                clipped_words.append({"word": token, "start": ws, "end": we})

            if clipped_words:
                clipped_words.sort(key=lambda w: (w["start"], w["end"]))
                normalized_words = []
                cursor = rel_start
                for word in clipped_words:
                    ws = max(float(word["start"]), cursor)
                    we = min(float(word["end"]), rel_end)
                    if (we - ws) < min_duration:
                        continue
                    normalized_words.append(
                        {
                            "word": word["word"],
                            "start": ws,
                            "end": we,
                        }
                    )
                    cursor = we

                if normalized_words:
                    new_seg["words"] = normalized_words
                    new_seg["text"] = " ".join(w["word"] for w in normalized_words)

        raw.append(new_seg)

    # -- De-overlap: trim each segment's end so it doesn't exceed the
    #    next segment's start. YouTube transcripts commonly overlap.
    raw.sort(key=lambda s: s["start"])
    for i in range(len(raw) - 1):
        curr = raw[i]
        nxt = raw[i + 1]
        if curr["end"] > nxt["start"]:
            curr["end"] = max(curr["start"], nxt["start"])
            if "words" in curr and curr["words"]:
                trimmed_words = []
                for w in curr["words"]:
                    ws = max(float(w["start"]), float(curr["start"]))
                    we = min(float(w["end"]), float(curr["end"]))
                    if (we - ws) < min_duration:
                        continue
                    trimmed_words.append({"word": w["word"], "start": ws, "end": we})
                curr["words"] = trimmed_words
                if trimmed_words:
                    curr["text"] = " ".join(w["word"] for w in trimmed_words)

    # -- Split long segments (no word-level timing) into ~8-word chunks
    #    so captions don't show a wall of text at once.
    segments: list[dict] = []
    for seg in raw:
        seg_start = float(seg["start"])
        seg_end = float(seg["end"])
        if (seg_end - seg_start) < min_duration:
            continue

        if "words" in seg and seg["words"]:
            # Whisper segments already have word timing.
            segments.append(seg)
            continue

        words = str(seg.get("text", "")).split()
        if not words:
            continue
        if len(words) <= 8:
            segments.append(
                {
                    "start": seg_start,
                    "end": seg_end,
                    "text": " ".join(words),
                }
            )
            continue

        # Split into chunks of ~8 words, distributing time evenly.
        chunk_size = 8
        seg_duration = seg_end - seg_start
        total_words = len(words)
        time_per_word = seg_duration / total_words if total_words > 0 else 0

        for ci in range(0, total_words, chunk_size):
            chunk_words = words[ci : ci + chunk_size]
            c_start = seg_start + ci * time_per_word
            c_end = seg_start + min(ci + chunk_size, total_words) * time_per_word
            if (c_end - c_start) < min_duration:
                continue
            segments.append(
                {
                    "start": c_start,
                    "end": c_end,
                    "text": " ".join(chunk_words),
                }
            )

    return segments


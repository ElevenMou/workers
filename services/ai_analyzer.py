import json
import logging
import os
import time
from typing import Any, Dict, List

from anthropic import Anthropic

logger = logging.getLogger(__name__)

# Default to Haiku 4.5 for best cost/quality ratio on structured extraction tasks.
# Override with CLAUDE_ANALYZER_MODEL for higher quality (e.g. claude-sonnet-4-20250514)
DEFAULT_ANALYZER_MODEL = os.getenv("CLAUDE_ANALYZER_MODEL", "claude-haiku-4-5-20241022")
# Approx 4 chars per token; cap transcript to stay within context and reduce cost
MAX_TRANSCRIPT_CHARS = int(os.getenv("CLAUDE_ANALYZER_MAX_TRANSCRIPT_CHARS", "-1"))
# Claude API call timeout
_CLAUDE_TIMEOUT_SECONDS = float(os.getenv("CLAUDE_ANALYZER_TIMEOUT_SECONDS", "120"))
# Circuit breaker settings
_CIRCUIT_BREAKER_THRESHOLD = int(os.getenv("CLAUDE_CIRCUIT_BREAKER_THRESHOLD", "3"))
_CIRCUIT_BREAKER_COOLDOWN = float(os.getenv("CLAUDE_CIRCUIT_BREAKER_COOLDOWN_SECONDS", "60"))

# ---------------------------------------------------------------------------
# Static system prompt (cacheable — identical across all calls)
# ---------------------------------------------------------------------------
_STATIC_SYSTEM_PROMPT = """\
You are a short-form clip editor. Your job is to identify the best segments from a video transcript for TikTok/Reels/Shorts.

TRANSCRIPT FORMAT: Each line is "start|end|text" where start and end are seconds.

TIMESTAMPS (strict): Use ONLY timestamps present in the transcript. No inventing or rounding. Clips = consecutive entries only. start_time = first entry start; end_time = last entry end. duration = end_time - start_time. Never exceed transcript bounds. Boundaries must match transcript entry boundaries exactly.

EDITORIAL (viral-first): Prioritize clips with strong standalone value: surprising insight, clear transformation, high emotion, conflict, actionable steps, or memorable one-liners. Strong hook in first 3-5s. Self-contained. Never cut a sentence or idea mid-way.

STRICT EXCLUSIONS: Reject segments that are:
- intros, greetings, scene-setting, "in this video/episode" setup, housekeeping
- outros, wrap-ups, "thanks for watching", "see you next time"
- engagement CTAs ("like/subscribe/follow/comment/share", "link in bio/description")
- sponsor/affiliate/promotional reads, discount-code plugs, admin announcements
- low-information filler, repetition, apologies, or technical notes

POSITIONAL BIAS: Avoid likely intro/outro zones in the first/last ~8% of the timeline unless the content is immediately high-value.

Use the submit_clips tool to return your results. Verify that duration = end_time - start_time for each clip."""

# ---------------------------------------------------------------------------
# Tool schema for structured output (guarantees valid JSON)
# ---------------------------------------------------------------------------
_CLIP_ANALYSIS_TOOL = {
    "name": "submit_clips",
    "description": "Submit the identified clip segments from the transcript analysis.",
    "input_schema": {
        "type": "object",
        "properties": {
            "clips": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "rank": {
                            "type": "integer",
                            "description": "Priority rank (1 = best)",
                        },
                        "start_time": {
                            "type": "number",
                            "description": "Start time in seconds (must match a transcript entry start)",
                        },
                        "end_time": {
                            "type": "number",
                            "description": "End time in seconds (must match a transcript entry end)",
                        },
                        "duration": {
                            "type": "number",
                            "description": "Duration in seconds (end_time - start_time)",
                        },
                        "clip_title": {
                            "type": "string",
                            "description": "Short, catchy title for the clip",
                        },
                        "hook": {
                            "type": "string",
                            "description": "The opening hook line (first 3-5 seconds)",
                        },
                        "summary": {
                            "type": "string",
                            "description": "Why this clip has viral potential",
                        },
                        "confidence_score": {
                            "type": "number",
                            "description": "Confidence score 0.0-1.0",
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Content tags for categorization",
                        },
                    },
                    "required": [
                        "rank",
                        "start_time",
                        "end_time",
                        "duration",
                        "clip_title",
                        "summary",
                        "confidence_score",
                    ],
                },
            }
        },
        "required": ["clips"],
    },
}


class AIAnalyzer:
    # Circuit breaker state (shared across instances within a worker process)
    _consecutive_failures: int = 0
    _circuit_open_until: float = 0.0

    def __init__(self):
        self.client = Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            timeout=_CLAUDE_TIMEOUT_SECONDS,
        )
        self.model = DEFAULT_ANALYZER_MODEL

    @staticmethod
    def _format_transcript_compact(snippets: List[Dict]) -> str:
        """Convert snippets to compact pipe-delimited format (~57% fewer tokens than JSON)."""
        lines: list[str] = []
        for s in snippets:
            start = f"{s['start']:.2f}"
            end = f"{s['start'] + s['duration']:.2f}"
            text = s["text"].replace("|", " ").replace("\n", " ")
            lines.append(f"{start}|{end}|{text}")
        return "\n".join(lines)

    @staticmethod
    def _truncate_snippets(snippets: List[Dict], max_chars: int) -> List[Dict]:
        """Keep snippets from the start until total JSON size ~ max_chars to limit API cost."""
        if max_chars <= 0:
            return snippets
        total = 0
        out = []
        for s in snippets:
            total += len(json.dumps(s)) + 2
            if total > max_chars:
                break
            out.append(s)
        return out

    @staticmethod
    def _extract_json_payload(response_text: str) -> str:
        """Extract a JSON object payload from raw model output (fallback for non-tool responses)."""
        text = (response_text or "").strip()
        if not text:
            raise ValueError("Analyzer returned an empty response")

        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            if lines and lines[0].strip().lower() == "json":
                lines = lines[1:]
            text = "\n".join(lines).strip()

        if text.startswith("{") and text.endswith("}"):
            return text

        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
            raise ValueError("Analyzer response did not contain a JSON object payload")
        return text[start_idx : end_idx + 1]

    @staticmethod
    def _as_float(value: Any) -> float | None:
        if isinstance(value, str):
            text = value.strip().lower()
            if not text:
                return None
            if text.endswith("s"):
                text = text[:-1].strip()
            if ":" in text:
                parts = text.split(":")
                try:
                    nums = [float(part) for part in parts]
                except ValueError:
                    nums = []
                if len(nums) == 2:
                    return nums[0] * 60.0 + nums[1]
                if len(nums) == 3:
                    return nums[0] * 3600.0 + nums[1] * 60.0 + nums[2]
            value = text
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _as_int(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _normalize_clip(
        cls,
        clip: Any,
        *,
        fallback_rank: int,
        min_duration: int,
    ) -> Dict[str, Any] | None:
        if not isinstance(clip, dict):
            return None

        start = cls._as_float(clip.get("start_time"))
        if start is None:
            start = cls._as_float(clip.get("start"))
        if start is None:
            start = cls._as_float(clip.get("startTime"))

        end = cls._as_float(clip.get("end_time"))
        if end is None:
            end = cls._as_float(clip.get("end"))
        if end is None:
            end = cls._as_float(clip.get("endTime"))
        if start is None or end is None or end <= start:
            return None

        duration = end - start

        title = str(clip.get("clip_title") or clip.get("title") or "").strip()
        summary = str(
            clip.get("summary")
            or clip.get("description")
            or clip.get("reason")
            or ""
        ).strip()

        if not title and summary:
            title = summary.split(".")[0][:80].strip()
        if not title:
            title = f"Clip {fallback_rank}"
        if not summary:
            summary = title

        hook = str(clip.get("hook") or "").strip()
        tags = clip.get("tags")
        if isinstance(tags, str):
            tags = [part.strip() for part in tags.split(",") if part.strip()]
        elif not isinstance(tags, list):
            tags = []

        raw_score = cls._as_float(clip.get("confidence_score"))
        if raw_score is None:
            raw_score = cls._as_float(clip.get("score"))
        if raw_score is None:
            score = None
        else:
            score = max(0.0, min(1.0, raw_score))

        rank = cls._as_int(clip.get("rank"))
        if rank is None:
            rank = cls._as_int(clip.get("priority"))
        if rank is None or rank <= 0:
            rank = fallback_rank

        return {
            "start": start,
            "end": end,
            "duration": duration,
            "text": summary,
            "title": title,
            "hook": hook,
            "score": score,
            "tags": tags,
            "rank": rank,
        }

    def find_best_clips(
        self,
        transcript: dict,
        num_clips: int = 5,
        min_duration: int = 10,
        max_duration: int = 60,
        extra_prompt: str | None = None,
    ) -> List[Dict]:
        """Analyze transcript using Claude to find best clips.

        Uses prompt caching, tool use for structured output, and compact
        transcript format to minimize cost while maximizing clip quality.

        Args:
            transcript: Video transcript with segments
            num_clips: Number of clips to generate (1-n)
            min_duration: Minimum clip duration in seconds
            max_duration: Maximum clip duration in seconds
            extra_prompt: Optional user-provided editorial guidance
        """

        # Format transcript for Claude
        segments = transcript.get("segments") or []
        snippets = []
        for segment in segments:
            start = self._as_float(segment.get("start"))
            end = self._as_float(segment.get("end"))
            text = str(segment.get("text") or "").strip()
            if start is None or end is None or end <= start or not text:
                continue
            snippets.append(
                {
                    "text": text,
                    "start": start,
                    "duration": end - start,
                }
            )
        if not snippets:
            raise RuntimeError("Transcript has no valid segments for AI analysis")
        snippets = self._truncate_snippets(snippets, MAX_TRANSCRIPT_CHARS)
        if not snippets:
            raise RuntimeError("Transcript snippet set is empty after truncation")
        transcript_start = min(float(item["start"]) for item in snippets)
        transcript_end = max(float(item["start"]) + float(item["duration"]) for item in snippets)
        transcript_duration = max(0.0, transcript_end - transcript_start)

        # Compact transcript format (~57% fewer tokens than json.dumps)
        transcript_text = self._format_transcript_compact(snippets)

        # System prompt: static cached block + small dynamic block
        system_blocks = [
            {
                "type": "text",
                "text": _STATIC_SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            },
            {
                "type": "text",
                "text": (
                    f"For this request: extract {num_clips} clip(s), "
                    f"each {min_duration}-{max_duration}s duration. "
                    f"Return exactly {num_clips} if possible; fewer if video is too short. "
                    f"Quality over quantity."
                ),
            },
        ]

        # User message: transcript data + optional extra prompt only
        extra_prompt_text = (extra_prompt or "").strip()
        extra_section = (
            f"\nAdditional user guidance:\n{extra_prompt_text}\n"
            if extra_prompt_text
            else ""
        )

        user_message = (
            f"Transcript span: {transcript_start:.2f}s to {transcript_end:.2f}s "
            f"({transcript_duration:.2f}s total).\n\n"
            f"{transcript_text}"
            f"{extra_section}"
        )

        # Dynamic max_tokens: ~200 tokens per clip + overhead
        max_tokens = min(4096, max(512, num_clips * 200 + 100))

        # Circuit breaker: fail fast if Claude API has been failing repeatedly
        if AIAnalyzer._consecutive_failures >= _CIRCUIT_BREAKER_THRESHOLD:
            if time.monotonic() < AIAnalyzer._circuit_open_until:
                raise RuntimeError(
                    f"Claude API circuit breaker open after {_CIRCUIT_BREAKER_THRESHOLD} "
                    f"consecutive failures. Cooldown until "
                    f"{AIAnalyzer._circuit_open_until - time.monotonic():.0f}s remaining."
                )
            # Cooldown expired — allow a single probe attempt
            logger.info("Circuit breaker cooldown expired, attempting probe request")
            AIAnalyzer._consecutive_failures = 0

        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=0,
                system=system_blocks,
                messages=[{"role": "user", "content": user_message}],
                tools=[_CLIP_ANALYSIS_TOOL],
                tool_choice={"type": "tool", "name": "submit_clips"},
            )
            # Reset circuit breaker on success
            AIAnalyzer._consecutive_failures = 0

            if not message.content:
                raise RuntimeError("Analyzer returned no content blocks")

            # Extract from tool_use block (primary path with structured output)
            tool_block = next(
                (block for block in message.content if block.type == "tool_use"),
                None,
            )

            if tool_block is not None:
                result = tool_block.input
                raw_clips = result.get("clips") if isinstance(result, dict) else None
            else:
                # Fallback: extract JSON from text (shouldn't happen with tool_choice)
                text_blocks = []
                for block in message.content:
                    block_text = getattr(block, "text", None)
                    if isinstance(block_text, str):
                        text_blocks.append(block_text)
                response_text = "\n".join(text_blocks).strip()
                if not response_text:
                    raise RuntimeError("Analyzer returned empty text content")
                payload = self._extract_json_payload(response_text)
                result = json.loads(payload)
                raw_clips = result.get("clips") if isinstance(result, dict) else None

            if not isinstance(raw_clips, list):
                raise ValueError("Analyzer JSON must include a 'clips' array")

            normalized: List[Dict[str, Any]] = []
            for idx, clip in enumerate(raw_clips, start=1):
                item = self._normalize_clip(
                    clip,
                    fallback_rank=idx,
                    min_duration=min_duration,
                )
                if item is None:
                    logger.warning("Skipping invalid analyzer clip payload at index %d", idx)
                    continue
                normalized.append(item)

            if not normalized:
                raise RuntimeError("Analyzer returned no valid clips")

            normalized.sort(key=lambda item: (int(item["rank"]), float(item["start"])))
            return normalized[: max(1, int(num_clips))]

        except Exception as e:
            AIAnalyzer._consecutive_failures += 1
            if AIAnalyzer._consecutive_failures >= _CIRCUIT_BREAKER_THRESHOLD:
                AIAnalyzer._circuit_open_until = time.monotonic() + _CIRCUIT_BREAKER_COOLDOWN
                logger.warning(
                    "Claude API circuit breaker opened after %d consecutive failures. "
                    "Cooldown: %.0fs",
                    AIAnalyzer._consecutive_failures,
                    _CIRCUIT_BREAKER_COOLDOWN,
                )
            logger.exception("Claude analyzer call failed: %s", e)
            raise

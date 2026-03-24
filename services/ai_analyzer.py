import json
import logging
import os
import time
from typing import Any, Dict, List, Literal, Sequence

from openai import NotFoundError, OpenAI
try:
    from openai import LengthFinishReasonError
except ImportError:  # older SDK versions
    LengthFinishReasonError = None  # type: ignore[assignment,misc]
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


def _get_env_text(*names: str, default: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return str(default)


def _get_env_int(*names: str, default: int) -> int:
    for name in names:
        value = os.getenv(name)
        if value is None or not str(value).strip():
            continue
        try:
            return int(str(value).strip())
        except (TypeError, ValueError):
            logger.warning("Invalid %s=%r; using default %d", name, value, default)
            break
    return int(default)


def _get_env_float(*names: str, default: float) -> float:
    for name in names:
        value = os.getenv(name)
        if value is None or not str(value).strip():
            continue
        try:
            return float(str(value).strip())
        except (TypeError, ValueError):
            logger.warning("Invalid %s=%r; using default %s", name, value, default)
            break
    return float(default)


DEFAULT_ANALYZER_MODEL = _get_env_text(
    "OPENAI_ANALYZER_MODEL",
    "CLAUDE_ANALYZER_MODEL",
    default="gpt-4.1-mini",
)
DEFAULT_ANALYZER_SYNTHESIS_MODEL = _get_env_text(
    "OPENAI_ANALYZER_SYNTHESIS_MODEL",
    default="gpt-4.1",
)
_DEFAULT_FALLBACK_MODELS = "gpt-4.1-mini,gpt-4.1"
# Approx 4 chars per token; cap transcript to stay within context and reduce cost.
MAX_TRANSCRIPT_CHARS = _get_env_int(
    "OPENAI_ANALYZER_MAX_TRANSCRIPT_CHARS",
    "CLAUDE_ANALYZER_MAX_TRANSCRIPT_CHARS",
    default=-1,
)
_ANALYZER_CHUNK_TARGET_SECONDS = _get_env_int(
    "OPENAI_ANALYZER_CHUNK_TARGET_SECONDS",
    default=480,
)
_ANALYZER_CHUNK_OVERLAP_SECONDS = _get_env_int(
    "OPENAI_ANALYZER_CHUNK_OVERLAP_SECONDS",
    default=30,
)
_ANALYZER_CHUNK_MAX_CHARS = _get_env_int(
    "OPENAI_ANALYZER_CHUNK_MAX_CHARS",
    default=12000,
)
_ANALYZER_CANDIDATES_PER_CHUNK = _get_env_int(
    "OPENAI_ANALYZER_CANDIDATES_PER_CHUNK",
    default=4,
)
_OPENAI_TIMEOUT_SECONDS = _get_env_float(
    "OPENAI_ANALYZER_TIMEOUT_SECONDS",
    "CLAUDE_ANALYZER_TIMEOUT_SECONDS",
    default=120.0,
)
_CIRCUIT_BREAKER_THRESHOLD = _get_env_int(
    "OPENAI_CIRCUIT_BREAKER_THRESHOLD",
    "CLAUDE_CIRCUIT_BREAKER_THRESHOLD",
    default=3,
)
_CIRCUIT_BREAKER_COOLDOWN = _get_env_float(
    "OPENAI_CIRCUIT_BREAKER_COOLDOWN_SECONDS",
    "CLAUDE_CIRCUIT_BREAKER_COOLDOWN_SECONDS",
    default=60.0,
)


def _parse_model_candidates(primary_model: str) -> list[str]:
    candidates: list[str] = []
    first = str(primary_model or "").strip()
    if first:
        candidates.append(first)

    fallback_models_raw = _get_env_text(
        "OPENAI_ANALYZER_FALLBACK_MODELS",
        "CLAUDE_ANALYZER_FALLBACK_MODELS",
        default=_DEFAULT_FALLBACK_MODELS,
    )
    for item in str(fallback_models_raw or "").split(","):
        model = item.strip()
        if model and model not in candidates:
            candidates.append(model)
    return candidates


def _resolve_api_key() -> str:
    return _get_env_text(
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        default="",
    )


_STATIC_SYSTEM_PROMPT = """\
You are an expert short-form clip editor. Your job is to identify the best segments from a video transcript for TikTok/Reels/Shorts.

TRANSCRIPT FORMAT: Each line is "start|end|text" where start and end are seconds.

TIMESTAMPS (strict):
- Use ONLY timestamps present in the transcript. No inventing or rounding.
- Clips = consecutive transcript entries only.
- start_time = first entry's start; end_time = last entry's end.
- duration = end_time - start_time. Never exceed transcript bounds.
- Boundaries must align with transcript entry boundaries exactly.

CONTENT ANALYSIS: Before selecting clips, identify the content type (story/narrative, educational/how-to, debate/opinion, humor/entertainment, interview/conversation) and apply type-appropriate criteria:
- Story/narrative: look for complete story arcs with setup and payoff within the clip.
- Educational/how-to: look for self-contained tips, surprising facts, or clear step-by-step value.
- Debate/opinion: look for bold claims, strong counter-arguments, or "mic drop" moments.
- Humor/entertainment: look for complete jokes, reactions, or memorable moments with punchlines.
- Interview/conversation: look for revealing answers, unexpected admissions, or powerful exchanges.

EDITORIAL (viral-first): Prioritize clips with strong standalone value:
- Surprising insight or counterintuitive claim
- Clear transformation or before/after contrast
- High emotion (joy, outrage, awe, empathy)
- Conflict, tension, or stakes
- Actionable steps or concrete advice
- Memorable one-liners or quotable phrases
The first 3-5 seconds MUST hook the viewer — identify the exact sentence or phrase that would stop someone from scrolling. The clip must be self-contained: never cut a sentence, idea, or story arc mid-way.

STRICT EXCLUSIONS: Reject segments that are:
- Intros, greetings, scene-setting, "in this video/episode" setup, housekeeping
- Outros, wrap-ups, "thanks for watching", "see you next time"
- Engagement CTAs ("like/subscribe/follow/comment/share", "link in bio/description")
- Sponsor/affiliate/promotional reads, discount-code plugs, admin announcements
- Low-information filler, repetition, apologies, or technical notes

POSITIONAL BIAS: The first and last 8% of the timeline are likely intro/outro zones. Only select from these zones if the content itself is immediately high-value (not setup or wind-down).

OVERLAP RULE: Clips must not overlap by more than 10% of the shorter clip's duration. Spread selections across the full timeline to give variety.

SCORING RUBRIC — assign confidence_score using these criteria:
- 0.90-1.00: Standalone viral hit — instantly compelling hook, emotional payoff, highly shareable even without context. Would perform well posted on its own.
- 0.70-0.89: Strong clip — good hook, mostly self-contained, clear value. Engaging for the target audience.
- 0.50-0.69: Decent content but weaker hook, or requires some outside context to fully land.
- Below 0.50: Low engagement potential — incomplete ideas, weak opening, or filler-adjacent. Avoid returning clips below 0.50.

REASONING: For each clip, you MUST provide a "reasoning" field: 1-2 sentences explaining WHY this specific segment has viral potential. Reference the hook, emotional arc, or value proposition. Think carefully before assigning the score.

Return a JSON object with a top-level "clips" array matching the requested response format. Verify duration = end_time - start_time for each clip."""

_SYNTHESIS_SYSTEM_PROMPT = """\
You are ranking candidate short-form clips selected from a longer video.

INPUT FORMAT:
- Each candidate already has exact transcript-aligned timestamps.
- Each candidate includes title, hook, summary, reasoning, and confidence from a first-pass analyzer.
- Use ONLY the candidate start/end times provided. Do not invent or adjust timestamps.

GOAL:
- Return the strongest standalone clips for TikTok/Reels/Shorts.
- Prefer clips with immediate hooks, clear payoff, strong standalone context, and variety across the video.
- Penalize duplicates, weaker rephrasings of the same moment, filler, intros/outros, sponsor reads, and clips that need outside context.

OUTPUT:
- Return a JSON object with top-level "clips".
- Preserve start_time/end_time exactly from the candidate list.
- Re-rank by overall final confidence using the same 0.00-1.00 scale.
- Include concise reasoning for why each selected candidate survives the global rerank.
"""


class ClipCandidate(BaseModel):
    model_config = ConfigDict(extra="ignore")

    rank: int | None = None
    start_time: float | None = None
    end_time: float | None = None
    duration: float | None = None
    clip_title: str | None = None
    hook: str | None = None
    hook_text: str | None = None
    summary: str | None = None
    reasoning: str | None = None
    content_category: str | None = None
    confidence_score: float | None = None
    tags: list[str] | None = None


class ClipAnalysisResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    clips: list[ClipCandidate]


class RemovableRangeCandidate(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    kind: Literal["intro", "outro", "ad"] | None = None
    start_time: float | None = Field(default=None, alias="startTime")
    end_time: float | None = Field(default=None, alias="endTime")
    confidence: float | None = None
    reason: str | None = None


class RemovableRangeResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    segments: list[RemovableRangeCandidate] = Field(default_factory=list)


class SegmentTitleCandidate(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    index: int | None = None
    title: str | None = None


class SegmentTitleResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    titles: list[SegmentTitleCandidate] = Field(default_factory=list)


class AIAnalyzer:
    # Circuit breaker state (shared across instances within a worker process).
    _consecutive_failures: int = 0
    _circuit_open_until: float = 0.0
    _model_resolution_cache: dict[str, str] = {}

    def __init__(self):
        self.client = OpenAI(
            api_key=_resolve_api_key(),
            timeout=_OPENAI_TIMEOUT_SECONDS,
        )
        configured_model = str(DEFAULT_ANALYZER_MODEL or "").strip()
        self.configured_model = configured_model if configured_model else DEFAULT_ANALYZER_MODEL
        self.model_candidates = _parse_model_candidates(self.configured_model)
        resolved_model = AIAnalyzer._model_resolution_cache.get(self.configured_model)
        if resolved_model:
            if resolved_model in self.model_candidates:
                self.model_candidates = [
                    resolved_model,
                    *[m for m in self.model_candidates if m != resolved_model],
                ]
            else:
                self.model_candidates = [resolved_model, *self.model_candidates]
        self.model = self.model_candidates[0]
        synthesis_model = str(DEFAULT_ANALYZER_SYNTHESIS_MODEL or "").strip()
        self.synthesis_configured_model = (
            synthesis_model if synthesis_model else DEFAULT_ANALYZER_SYNTHESIS_MODEL
        )
        self.synthesis_model_candidates = _parse_model_candidates(
            self.synthesis_configured_model
        )

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
        """Proportionally sample snippets when total size exceeds *max_chars*.

        Instead of keeping only the head (which loses the entire second half of
        long videos), we keep ~40% from the start, ~30% from the middle, and
        ~30% from the end so the model sees content across the full timeline.
        Gaps between sections are indicated by a sentinel snippet.
        """
        if max_chars <= 0:
            return snippets

        total = sum(len(json.dumps(s)) + 2 for s in snippets)
        if total <= max_chars:
            return snippets

        n = len(snippets)
        head_count = max(1, int(n * 0.40))
        mid_count = max(1, int(n * 0.30))
        tail_count = max(1, int(n * 0.30))

        head = snippets[:head_count]
        mid_start = max(head_count, (n - mid_count) // 2)
        mid_end = mid_start + mid_count
        mid = snippets[mid_start:mid_end]
        tail = snippets[max(mid_end, n - tail_count):]

        gap_marker = {"text": "[...transcript trimmed...]", "start": 0.0, "duration": 0.0}

        combined: List[Dict] = []
        combined.extend(head)
        if head and mid and mid[0] is not head[-1]:
            combined.append(gap_marker)
            combined.extend(mid)
        if mid and tail and tail[0] is not mid[-1]:
            combined.append(gap_marker)
            combined.extend(tail)
        elif not mid:
            if head and tail and tail[0] is not head[-1]:
                combined.append(gap_marker)
                combined.extend(tail)

        budget = max_chars
        out: List[Dict] = []
        for s in combined:
            cost = len(json.dumps(s)) + 2
            if budget - cost < 0 and out:
                break
            budget -= cost
            out.append(s)
        return out

    @staticmethod
    def _repair_truncated_json(text: str) -> str | None:
        """Attempt to repair JSON truncated by a max_tokens limit.

        The model typically produces ``{"clips": [{...}, {...]`` before
        hitting the token limit.  We try progressively simpler closings
        to recover whatever complete clip objects exist.
        """
        # Find the start of the JSON object.
        start = text.find("{")
        if start == -1:
            return None
        fragment = text[start:]

        # Strategy 1: close the array + object.
        # Try ``]}`` first (last value was complete), then ``"]}`` (string was open).
        for suffix in ("]}", '"]}'):
            candidate = fragment.rstrip().rstrip(",") + suffix
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                pass

        # Strategy 2: trim back to the last complete object in the clips
        # array, then close.  Look for ``}, {`` boundaries and try from
        # the rightmost one.
        last_obj_end = fragment.rfind("},")
        while last_obj_end > 0:
            candidate = fragment[: last_obj_end + 1] + "]}"
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                last_obj_end = fragment.rfind("},", 0, last_obj_end)

        return None

    @staticmethod
    def _extract_json_payload(response_text: str) -> tuple[str, bool]:
        """Extract a JSON object payload from raw model output."""
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
            return text, False

        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            candidate = text[start_idx : end_idx + 1]
            try:
                json.loads(candidate)
                return candidate, False
            except json.JSONDecodeError:
                pass

        # The response may be truncated (e.g. max_tokens reached).
        # Try to repair the JSON by closing open brackets.
        repaired = AIAnalyzer._repair_truncated_json(text)
        if repaired is not None:
            logger.warning(
                "Repaired truncated JSON response (%d chars → %d chars)",
                len(text),
                len(repaired),
            )
            return repaired, True

        raise ValueError("Analyzer response did not contain a JSON object payload")

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

    @staticmethod
    def _looks_like_clip_mapping(value: Any) -> bool:
        if not isinstance(value, dict):
            return False
        has_start = any(key in value for key in ("start_time", "start", "startTime"))
        has_end = any(key in value for key in ("end_time", "end", "endTime"))
        has_duration = "duration" in value
        return (has_start and has_end) or has_duration

    @classmethod
    def _extract_raw_clips(cls, value: Any, *, key_hint: str | None = None) -> list[Any] | None:
        key = str(key_hint or "").strip().lower()
        if isinstance(value, list):
            if key in {"clips", "segments", "highlights", "moments"}:
                return value
            if any(cls._looks_like_clip_mapping(item) for item in value):
                return value
            return None

        if not isinstance(value, dict):
            return None

        preferred_keys = (
            "clips",
            "segments",
            "highlights",
            "moments",
            "clip_candidates",
            "clipCandidates",
            "items",
        )
        for preferred in preferred_keys:
            child = value.get(preferred)
            extracted = cls._extract_raw_clips(child, key_hint=preferred)
            if extracted is not None:
                return extracted

        container_keys = (
            "input",
            "output",
            "result",
            "results",
            "data",
            "response",
            "payload",
            "analysis",
        )
        for container in container_keys:
            child = value.get(container)
            extracted = cls._extract_raw_clips(child, key_hint=container)
            if extracted is not None:
                return extracted

        for nested_key, child in value.items():
            if not isinstance(child, (dict, list)):
                continue
            extracted = cls._extract_raw_clips(child, key_hint=nested_key)
            if extracted is not None:
                return extracted
        return None

    @staticmethod
    def _is_model_not_found_error(exc: Exception) -> bool:
        if isinstance(exc, NotFoundError):
            return True

        status_code = getattr(exc, "status_code", None)
        text = str(exc).lower()
        if status_code in {400, 404} and "model" in text:
            if any(marker in text for marker in ("not found", "not_found", "invalid", "does not exist")):
                return True
        return False

    @staticmethod
    def _message_text(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()

        if not isinstance(content, list):
            return ""

        text_parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    text_parts.append(text)
                continue

            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    text_parts.append(text.strip())
                    continue
                if isinstance(text, dict):
                    value = text.get("value")
                    if isinstance(value, str) and value.strip():
                        text_parts.append(value.strip())
                        continue

            for attr_name in ("text", "value", "content"):
                attr_value = getattr(item, attr_name, None)
                if isinstance(attr_value, str) and attr_value.strip():
                    text_parts.append(attr_value.strip())
                    break

        return "\n".join(text_parts).strip()

    @staticmethod
    def _coerce_parsed_payload(parsed: Any) -> Any:
        if parsed is None:
            return None
        if hasattr(parsed, "model_dump"):
            return parsed.model_dump()
        if hasattr(parsed, "dict"):
            return parsed.dict()
        return parsed

    _TRANSIENT_RETRY_COUNT = 3
    _TRANSIENT_RETRY_BASE_DELAY = 2.0

    @staticmethod
    def _is_transient_error(exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None)
        if status_code in {429, 500, 502, 503, 529}:
            return True
        exc_name = type(exc).__name__.lower()
        if "timeout" in exc_name or "connection" in exc_name:
            return True
        text = str(exc).lower()
        if any(kw in text for kw in ("rate limit", "rate_limit", "timeout", "overloaded")):
            return True
        return False

    def _create_with_model_fallback(
        self,
        *,
        max_tokens: int,
        system_prompt: str,
        user_message: str,
        response_format: type[BaseModel] = ClipAnalysisResponse,
        model_candidates: list[str] | None = None,
        configured_model: str | None = None,
    ):
        last_error: Exception | None = None
        candidates = list(model_candidates or self.model_candidates)
        configured_name = str(configured_model or self.configured_model).strip() or self.configured_model
        for index, candidate_model in enumerate(candidates):
            transient_attempts = 0
            while True:
                try:
                    try:
                        completion = self.client.beta.chat.completions.parse(
                            model=candidate_model,
                            max_completion_tokens=max_tokens,
                            temperature=0,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_message},
                            ],
                            response_format=response_format,
                        )
                    except Exception as length_exc:
                        # Handle LengthFinishReasonError: the model hit
                        # max_tokens before completing the structured output.
                        # Extract the partial completion so downstream parsing
                        # can salvage whatever clips were generated.
                        if (
                            LengthFinishReasonError is not None
                            and isinstance(length_exc, LengthFinishReasonError)
                        ):
                            partial = getattr(length_exc, "completion", None)
                            if partial is not None:
                                logger.warning(
                                    "OpenAI structured output truncated at %d tokens "
                                    "(max_tokens=%d) on model '%s'. "
                                    "Attempting to salvage partial response.",
                                    getattr(
                                        getattr(partial, "usage", None),
                                        "completion_tokens",
                                        0,
                                    ),
                                    max_tokens,
                                    candidate_model,
                                )
                                completion = partial
                            else:
                                raise
                        else:
                            raise
                    if candidate_model != configured_name:
                        logger.warning(
                            "Configured OpenAI model '%s' unavailable; using fallback '%s'.",
                            configured_name,
                            candidate_model,
                        )
                        AIAnalyzer._model_resolution_cache[configured_name] = candidate_model
                    else:
                        AIAnalyzer._model_resolution_cache.pop(configured_name, None)
                    self.model = candidate_model
                    return completion
                except Exception as exc:
                    last_error = exc

                    if (
                        self._is_transient_error(exc)
                        and transient_attempts < self._TRANSIENT_RETRY_COUNT
                    ):
                        transient_attempts += 1
                        delay = self._TRANSIENT_RETRY_BASE_DELAY * (2 ** (transient_attempts - 1))
                        logger.warning(
                            "Transient API error on model '%s' (attempt %d/%d): %s. "
                            "Retrying in %.1fs ...",
                            candidate_model,
                            transient_attempts,
                            self._TRANSIENT_RETRY_COUNT,
                            exc,
                            delay,
                        )
                        time.sleep(delay)
                        continue

                    is_model_not_found = self._is_model_not_found_error(exc)
                    has_next_candidate = index < len(candidates) - 1
                    if not is_model_not_found or not has_next_candidate:
                        raise
                    logger.warning(
                        "OpenAI model '%s' not found; retrying with fallback '%s'.",
                        candidate_model,
                        candidates[index + 1],
                    )
                    break

        if last_error is not None:
            raise last_error
        raise RuntimeError("OpenAI analyzer failed without a concrete error")

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
        hook_text = str(clip.get("hook_text") or "").strip()
        reasoning = str(clip.get("reasoning") or "").strip()
        content_category = str(clip.get("content_category") or "").strip()
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
            "hook_text": hook_text,
            "reasoning": reasoning,
            "content_category": content_category,
            "score": score,
            "tags": tags,
            "rank": rank,
        }

    @staticmethod
    def _snippet_char_cost(snippet: Dict[str, Any]) -> int:
        start = float(snippet["start"])
        end = float(snippet["start"]) + float(snippet["duration"])
        text = str(snippet["text"]).replace("|", " ").replace("\n", " ")
        return len(f"{start:.2f}|{end:.2f}|{text}\n")

    @classmethod
    def _build_snippets(cls, transcript: dict) -> list[Dict[str, Any]]:
        segments = transcript.get("segments") or []
        snippets: list[Dict[str, Any]] = []
        for segment in segments:
            start = cls._as_float(segment.get("start"))
            end = cls._as_float(segment.get("end"))
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
        return snippets

    @classmethod
    def _snippets_total_chars(cls, snippets: list[Dict[str, Any]]) -> int:
        return sum(cls._snippet_char_cost(snippet) for snippet in snippets)

    @staticmethod
    def _video_context_section(
        *,
        video_title: str | None,
        video_platform: str | None,
        video_duration: float | None,
    ) -> str:
        video_context_parts: list[str] = []
        title_text = (video_title or "").strip()
        if title_text:
            video_context_parts.append(f'Video: "{title_text}"')
        platform_text = (video_platform or "").strip()
        if platform_text:
            video_context_parts.append(f"Platform: {platform_text}")
        if video_duration and video_duration > 0:
            video_context_parts.append(f"Full video duration: {video_duration / 60:.1f} min")
        return " | ".join(video_context_parts) + "\n" if video_context_parts else ""

    def _request_clip_candidates(
        self,
        *,
        max_tokens: int,
        system_prompt: str,
        user_message: str,
        model_candidates: list[str] | None = None,
        configured_model: str | None = None,
    ) -> list[Dict[str, Any]]:
        if AIAnalyzer._consecutive_failures >= _CIRCUIT_BREAKER_THRESHOLD:
            if time.monotonic() < AIAnalyzer._circuit_open_until:
                raise RuntimeError(
                    f"OpenAI API circuit breaker open after {_CIRCUIT_BREAKER_THRESHOLD} "
                    f"consecutive failures. Cooldown until "
                    f"{AIAnalyzer._circuit_open_until - time.monotonic():.0f}s remaining."
                )
            logger.info("Circuit breaker cooldown expired, attempting probe request")
            AIAnalyzer._consecutive_failures = 0

        try:
            completion = self._create_with_model_fallback(
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                user_message=user_message,
                model_candidates=model_candidates,
                        configured_model=configured_model,
                        response_format=ClipAnalysisResponse,
                    )
            AIAnalyzer._consecutive_failures = 0

            choices = list(getattr(completion, "choices", []) or [])
            if not choices:
                raise RuntimeError("Analyzer returned no choices")

            message = getattr(choices[0], "message", None)
            if message is None:
                raise RuntimeError("Analyzer returned no message")

            refusal = str(getattr(message, "refusal", "") or "").strip()
            if refusal:
                raise RuntimeError(f"OpenAI analyzer refusal: {refusal}")

            result: Any = None
            raw_clips: list[Any] | None = None
            candidate_source: str | None = None
            repaired_truncated_json = False

            parsed = self._coerce_parsed_payload(getattr(message, "parsed", None))
            if parsed is not None:
                result = parsed
                raw_clips = self._extract_raw_clips(parsed)
                if isinstance(raw_clips, list):
                    candidate_source = "parsed_response"

            if raw_clips is None:
                response_text = self._message_text(getattr(message, "content", None))
                if response_text:
                    payload, repaired_truncated_json = self._extract_json_payload(
                        response_text
                    )
                    result = json.loads(payload)
                    raw_clips = self._extract_raw_clips(result)
                    candidate_source = "text_fallback"

            if not isinstance(raw_clips, list):
                if isinstance(result, dict):
                    logger.warning(
                        "Analyzer payload missing clips-like array. Top-level keys: %s",
                        list(result.keys())[:20],
                    )
                else:
                    logger.warning(
                        "Analyzer payload missing clips-like array. Payload type: %s",
                        type(result).__name__,
                    )
                raise ValueError("Analyzer JSON must include a recognizable clips array")

            if candidate_source:
                logger.info("Analyzer clips payload accepted from %s", candidate_source)

            normalized: List[Dict[str, Any]] = []
            for idx, clip in enumerate(raw_clips, start=1):
                item = self._normalize_clip(
                    clip,
                    fallback_rank=idx,
                    min_duration=1,
                )
                if item is None:
                    logger.warning("Skipping invalid analyzer clip payload at index %d", idx)
                    continue
                normalized.append(item)

            if not normalized:
                raise RuntimeError("Analyzer returned no valid clips")

            if repaired_truncated_json:
                logger.warning(
                    "Analyzer salvaged %d/%d valid clip payload item(s) from truncated JSON "
                    "(max_tokens=%d)",
                    len(normalized),
                    len(raw_clips),
                    max_tokens,
                )

            normalized.sort(key=lambda item: (int(item["rank"]), float(item["start"])))
            return normalized

        except Exception as e:
            AIAnalyzer._consecutive_failures += 1
            if AIAnalyzer._consecutive_failures >= _CIRCUIT_BREAKER_THRESHOLD:
                AIAnalyzer._circuit_open_until = time.monotonic() + _CIRCUIT_BREAKER_COOLDOWN
                logger.warning(
                    "OpenAI API circuit breaker opened after %d consecutive failures. "
                    "Cooldown: %.0fs",
                    AIAnalyzer._consecutive_failures,
                    _CIRCUIT_BREAKER_COOLDOWN,
                )
            logger.exception("OpenAI analyzer call failed: %s", e)
            raise

    @staticmethod
    def _extract_list_payload(value: Any, keys: Sequence[str]) -> list[Any] | None:
        if isinstance(value, list):
            return value
        if not isinstance(value, dict):
            return None
        for key in keys:
            child = value.get(key)
            if isinstance(child, list):
                return child
        return None

    def _request_structured_payload(
        self,
        *,
        max_tokens: int,
        system_prompt: str,
        user_message: str,
        response_format: type[BaseModel],
        model_candidates: list[str] | None = None,
        configured_model: str | None = None,
    ) -> Any:
        if AIAnalyzer._consecutive_failures >= _CIRCUIT_BREAKER_THRESHOLD:
            if time.monotonic() < AIAnalyzer._circuit_open_until:
                raise RuntimeError(
                    f"OpenAI API circuit breaker open after {_CIRCUIT_BREAKER_THRESHOLD} "
                    f"consecutive failures. Cooldown until "
                    f"{AIAnalyzer._circuit_open_until - time.monotonic():.0f}s remaining."
                )
            logger.info("Circuit breaker cooldown expired, attempting probe request")
            AIAnalyzer._consecutive_failures = 0

        try:
            completion = self._create_with_model_fallback(
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                user_message=user_message,
                response_format=response_format,
                model_candidates=model_candidates,
                configured_model=configured_model,
            )
            AIAnalyzer._consecutive_failures = 0

            choices = list(getattr(completion, "choices", []) or [])
            if not choices:
                raise RuntimeError("Analyzer returned no choices")

            message = getattr(choices[0], "message", None)
            if message is None:
                raise RuntimeError("Analyzer returned no message")

            refusal = str(getattr(message, "refusal", "") or "").strip()
            if refusal:
                raise RuntimeError(f"OpenAI analyzer refusal: {refusal}")

            parsed = self._coerce_parsed_payload(getattr(message, "parsed", None))
            if parsed is not None:
                return parsed

            response_text = self._message_text(getattr(message, "content", None))
            if not response_text:
                raise RuntimeError("Analyzer returned an empty response")
            payload, _ = self._extract_json_payload(response_text)
            return json.loads(payload)
        except Exception as exc:
            AIAnalyzer._consecutive_failures += 1
            if AIAnalyzer._consecutive_failures >= _CIRCUIT_BREAKER_THRESHOLD:
                AIAnalyzer._circuit_open_until = time.monotonic() + _CIRCUIT_BREAKER_COOLDOWN
                logger.warning(
                    "OpenAI API circuit breaker opened after %d consecutive failures. "
                    "Cooldown: %.0fs",
                    AIAnalyzer._consecutive_failures,
                    _CIRCUIT_BREAKER_COOLDOWN,
                )
            logger.exception("OpenAI structured request failed: %s", exc)
            raise

    def _analyze_snippets(
        self,
        *,
        snippets: list[Dict[str, Any]],
        request_count: int,
        min_duration: int,
        max_duration: int,
        extra_prompt: str | None,
        video_title: str | None,
        video_platform: str | None,
        video_duration: float | None,
        model_candidates: list[str] | None = None,
        configured_model: str | None = None,
    ) -> list[Dict[str, Any]]:
        if not snippets:
            raise RuntimeError("Transcript has no valid segments for AI analysis")

        transcript_start = min(float(item["start"]) for item in snippets)
        transcript_end = max(float(item["start"]) + float(item["duration"]) for item in snippets)
        transcript_duration = max(0.0, transcript_end - transcript_start)
        transcript_text = self._format_transcript_compact(snippets)
        system_prompt = (
            f"{_STATIC_SYSTEM_PROMPT}\n\n"
            f"For this request: extract {request_count} clip(s), "
            f"each {min_duration}-{max_duration}s duration. "
            f"Return exactly {request_count} if possible; fewer if video is too short. "
            f"Quality over quantity."
        )

        extra_prompt_text = (extra_prompt or "").strip()
        extra_section = (
            f"\nAdditional user guidance:\n{extra_prompt_text}\n"
            if extra_prompt_text
            else ""
        )
        user_message = (
            f"{self._video_context_section(video_title=video_title, video_platform=video_platform, video_duration=video_duration)}"
            f"Transcript span: {transcript_start:.2f}s to {transcript_end:.2f}s "
            f"({transcript_duration:.2f}s total).\n\n"
            f"{transcript_text}"
            f"{extra_section}"
        )
        # Each clip in structured output uses ~500-600 tokens (title, hook,
        # summary, reasoning, tags, etc.). Use 600 per clip + 400 overhead.
        max_tokens = min(16384, max(2048, request_count * 600 + 400))
        clips = self._request_clip_candidates(
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            user_message=user_message,
            model_candidates=model_candidates,
            configured_model=configured_model,
        )
        return clips[: max(1, int(request_count))]

    @classmethod
    def _build_snippet_chunks(
        cls,
        snippets: list[Dict[str, Any]],
        *,
        target_seconds: int,
        overlap_seconds: int,
        max_chars: int,
    ) -> list[dict[str, Any]]:
        if not snippets:
            return []

        chunks: list[dict[str, Any]] = []
        start_index = 0
        snippet_count = len(snippets)

        while start_index < snippet_count:
            chunk_snippets: list[Dict[str, Any]] = []
            chunk_start = float(snippets[start_index]["start"])
            chunk_end = chunk_start
            chunk_chars = 0
            index = start_index

            while index < snippet_count:
                snippet = snippets[index]
                snippet_end = float(snippet["start"]) + float(snippet["duration"])
                snippet_cost = cls._snippet_char_cost(snippet)
                exceeds_time = (
                    bool(chunk_snippets)
                    and snippet_end - chunk_start > max(1, int(target_seconds))
                )
                exceeds_chars = (
                    bool(chunk_snippets)
                    and max_chars > 0
                    and chunk_chars + snippet_cost > max_chars
                )
                if exceeds_time or exceeds_chars:
                    break
                chunk_snippets.append(snippet)
                chunk_chars += snippet_cost
                chunk_end = snippet_end
                index += 1

            if not chunk_snippets:
                snippet = snippets[start_index]
                chunk_snippets = [snippet]
                chunk_end = float(snippet["start"]) + float(snippet["duration"])
                index = start_index + 1

            chunks.append(
                {
                    "index": len(chunks) + 1,
                    "start": chunk_start,
                    "end": chunk_end,
                    "snippets": chunk_snippets,
                }
            )

            if index >= snippet_count:
                break

            next_start_threshold = max(chunk_start, chunk_end - max(0, int(overlap_seconds)))
            next_start_index = index
            while (
                next_start_index > start_index
                and float(snippets[next_start_index - 1]["start"]) >= next_start_threshold
            ):
                next_start_index -= 1
            if next_start_index <= start_index:
                next_start_index = start_index + 1
            start_index = next_start_index

        return chunks

    @staticmethod
    def _build_candidate_excerpt(
        snippets: list[Dict[str, Any]],
        *,
        start: float,
        end: float,
        max_chars: int = 260,
    ) -> str:
        parts: list[str] = []
        total_chars = 0
        for snippet in snippets:
            snippet_start = float(snippet["start"])
            snippet_end = snippet_start + float(snippet["duration"])
            if snippet_end <= start or snippet_start >= end:
                continue
            text = str(snippet["text"]).strip()
            if not text:
                continue
            parts.append(text)
            total_chars += len(text)
            if total_chars >= max_chars:
                break
        excerpt = " ".join(parts).strip()
        if len(excerpt) <= max_chars:
            return excerpt
        return excerpt[: max_chars - 3].rstrip() + "..."

    @staticmethod
    def _dedupe_candidates(candidates: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        deduped: dict[tuple[float, float], Dict[str, Any]] = {}
        for candidate in candidates:
            key = (round(float(candidate["start"]), 2), round(float(candidate["end"]), 2))
            existing = deduped.get(key)
            candidate_score = float(candidate.get("score") or 0.0)
            existing_score = float(existing.get("score") or 0.0) if existing else -1.0
            if existing is None or candidate_score > existing_score:
                deduped[key] = dict(candidate)
        return sorted(
            deduped.values(),
            key=lambda item: (
                -float(item.get("score") or 0.0),
                int(item.get("rank") or 9999),
                float(item["start"]),
            ),
        )

    @classmethod
    def _candidate_time_match(
        cls,
        candidate: Dict[str, Any],
        allowed_candidates: list[Dict[str, Any]],
    ) -> Dict[str, Any] | None:
        start = float(candidate["start"])
        end = float(candidate["end"])
        exact_key = (round(start, 2), round(end, 2))
        for allowed in allowed_candidates:
            allowed_key = (
                round(float(allowed["start"]), 2),
                round(float(allowed["end"]), 2),
            )
            if allowed_key == exact_key:
                return allowed

        matches = [
            allowed
            for allowed in allowed_candidates
            if abs(float(allowed["start"]) - start) <= 0.75
            and abs(float(allowed["end"]) - end) <= 0.75
        ]
        if not matches:
            return None
        return min(
            matches,
            key=lambda allowed: abs(float(allowed["start"]) - start)
            + abs(float(allowed["end"]) - end),
        )

    def _synthesize_candidates(
        self,
        *,
        candidates: list[Dict[str, Any]],
        candidate_target: int,
        min_duration: int,
        max_duration: int,
        extra_prompt: str | None,
        video_title: str | None,
        video_platform: str | None,
        video_duration: float | None,
    ) -> list[Dict[str, Any]]:
        lines = [
            self._video_context_section(
                video_title=video_title,
                video_platform=video_platform,
                video_duration=video_duration,
            ).strip(),
            f"Return the best {candidate_target} clip(s) from this candidate shortlist.",
            (
                f"Target duration range: {min_duration}-{max_duration}s. "
                "Use only candidate timestamps exactly as provided."
            ),
            "",
            "Candidates:",
        ]
        for index, candidate in enumerate(candidates, start=1):
            tags = ", ".join(candidate.get("tags") or [])
            lines.extend(
                [
                    f"Candidate {index}:",
                    f"start_time={float(candidate['start']):.2f}",
                    f"end_time={float(candidate['end']):.2f}",
                    f"duration={float(candidate['duration']):.2f}",
                    f"title={str(candidate.get('title') or '').strip()}",
                    f"hook={str(candidate.get('hook') or candidate.get('hook_text') or '').strip()}",
                    f"summary={str(candidate.get('text') or '').strip()}",
                    f"reasoning={str(candidate.get('reasoning') or '').strip()}",
                    f"first_pass_confidence={float(candidate.get('first_pass_score') or candidate.get('score') or 0.0):.2f}",
                    f"source_chunk={int(candidate.get('source_chunk_index') or 0)}",
                    f"tags={tags}",
                    f"excerpt={str(candidate.get('excerpt') or '').strip()}",
                    "",
                ]
            )

        extra_prompt_text = (extra_prompt or "").strip()
        if extra_prompt_text:
            lines.extend(["Additional user guidance:", extra_prompt_text, ""])

        user_message = "\n".join(line for line in lines if line is not None).strip()
        max_tokens = min(30000, max(2048, candidate_target * 600 + 400))
        reranked = self._request_clip_candidates(
            max_tokens=max_tokens,
            system_prompt=_SYNTHESIS_SYSTEM_PROMPT,
            user_message=user_message,
            model_candidates=self.synthesis_model_candidates,
            configured_model=self.synthesis_configured_model,
        )

        matched: list[Dict[str, Any]] = []
        used_keys: set[tuple[float, float]] = set()
        for candidate in reranked:
            allowed = self._candidate_time_match(candidate, candidates)
            if allowed is None:
                continue
            key = (round(float(allowed["start"]), 2), round(float(allowed["end"]), 2))
            if key in used_keys:
                continue
            merged = dict(allowed)
            merged.update(
                {
                    "title": candidate.get("title") or allowed.get("title"),
                    "text": candidate.get("text") or allowed.get("text"),
                    "hook": candidate.get("hook") or allowed.get("hook"),
                    "hook_text": candidate.get("hook_text") or allowed.get("hook_text"),
                    "reasoning": candidate.get("reasoning") or allowed.get("reasoning"),
                    "content_category": candidate.get("content_category") or allowed.get("content_category"),
                    "score": candidate.get("score"),
                    "rank": len(matched) + 1,
                }
            )
            merged["start"] = float(allowed["start"])
            merged["end"] = float(allowed["end"])
            merged["duration"] = float(allowed["duration"])
            matched.append(merged)
            used_keys.add(key)
            if len(matched) >= candidate_target:
                break

        if matched:
            return matched
        raise RuntimeError("Synthesis rerank returned no valid candidates")

    def find_best_clips_detailed(
        self,
        transcript: dict,
        num_clips: int = 5,
        min_duration: int = 10,
        max_duration: int = 60,
        extra_prompt: str | None = None,
        video_title: str | None = None,
        video_platform: str | None = None,
        video_duration: float | None = None,
    ) -> Dict[str, Any]:
        """Analyze transcript and return clips plus diagnostics."""
        requested_count = max(1, int(num_clips))
        candidate_target = min(20, max(requested_count * 3, requested_count + 4))
        snippets = self._build_snippets(transcript)
        if not snippets:
            raise RuntimeError("Transcript has no valid segments for AI analysis")

        transcript_duration = max(
            0.0,
            max(float(item["start"]) + float(item["duration"]) for item in snippets)
            - min(float(item["start"]) for item in snippets),
        )
        transcript_chars = self._snippets_total_chars(snippets)
        single_pass_mode = (
            transcript_duration <= float(_ANALYZER_CHUNK_TARGET_SECONDS)
            and transcript_chars <= int(_ANALYZER_CHUNK_MAX_CHARS)
        )

        diagnostics: Dict[str, Any] = {
            "analysis_mode": "single_pass" if single_pass_mode else "chunked",
            "chunk_count": 1,
            "candidate_target": candidate_target,
            "candidate_returned": 0,
            "chunk_candidate_count": 0,
            "transcript_chars": transcript_chars,
            "transcript_duration": transcript_duration,
        }

        if single_pass_mode:
            clips = self._analyze_snippets(
                snippets=snippets,
                request_count=candidate_target,
                min_duration=min_duration,
                max_duration=max_duration,
                extra_prompt=extra_prompt,
                video_title=video_title,
                video_platform=video_platform,
                video_duration=video_duration,
            )
            diagnostics["candidate_returned"] = len(clips)
            return {"clips": clips, "diagnostics": diagnostics}

        chunks = self._build_snippet_chunks(
            snippets,
            target_seconds=_ANALYZER_CHUNK_TARGET_SECONDS,
            overlap_seconds=_ANALYZER_CHUNK_OVERLAP_SECONDS,
            max_chars=_ANALYZER_CHUNK_MAX_CHARS,
        )
        diagnostics["chunk_count"] = len(chunks)

        merged_candidates: list[Dict[str, Any]] = []
        for chunk in chunks:
            chunk_candidates = self._analyze_snippets(
                snippets=list(chunk["snippets"]),
                request_count=max(1, int(_ANALYZER_CANDIDATES_PER_CHUNK)),
                min_duration=min_duration,
                max_duration=max_duration,
                extra_prompt=extra_prompt,
                video_title=video_title,
                video_platform=video_platform,
                video_duration=video_duration,
            )
            for candidate in chunk_candidates:
                candidate_copy = dict(candidate)
                candidate_copy["first_pass_score"] = candidate.get("score")
                candidate_copy["source_chunk_index"] = int(chunk["index"])
                candidate_copy["excerpt"] = self._build_candidate_excerpt(
                    list(chunk["snippets"]),
                    start=float(candidate_copy["start"]),
                    end=float(candidate_copy["end"]),
                )
                merged_candidates.append(candidate_copy)

        if not merged_candidates:
            raise RuntimeError("Analyzer returned no valid candidates across transcript chunks")

        deduped_candidates = self._dedupe_candidates(merged_candidates)
        diagnostics["chunk_candidate_count"] = len(deduped_candidates)
        synthesis_shortlist = deduped_candidates[
            : max(candidate_target * 2, candidate_target + 8)
        ]

        try:
            final_candidates = self._synthesize_candidates(
                candidates=synthesis_shortlist,
                candidate_target=candidate_target,
                min_duration=min_duration,
                max_duration=max_duration,
                extra_prompt=extra_prompt,
                video_title=video_title,
                video_platform=video_platform,
                video_duration=video_duration,
            )
        except Exception as exc:
            logger.warning(
                "Synthesis rerank failed; falling back to first-pass candidates: %s",
                exc,
            )
            diagnostics["analysis_mode"] = "chunked_fallback"
            final_candidates = synthesis_shortlist[:candidate_target]

        diagnostics["candidate_returned"] = len(final_candidates)
        return {"clips": final_candidates, "diagnostics": diagnostics}

    def find_best_clips(
        self,
        transcript: dict,
        num_clips: int = 5,
        min_duration: int = 10,
        max_duration: int = 60,
        extra_prompt: str | None = None,
        video_title: str | None = None,
        video_platform: str | None = None,
        video_duration: float | None = None,
    ) -> List[Dict]:
        detailed = self.find_best_clips_detailed(
            transcript=transcript,
            num_clips=num_clips,
            min_duration=min_duration,
            max_duration=max_duration,
            extra_prompt=extra_prompt,
            video_title=video_title,
            video_platform=video_platform,
            video_duration=video_duration,
        )
        return list(detailed["clips"])[: max(1, int(num_clips))]

    def find_removable_ranges(
        self,
        transcript: dict,
        *,
        video_title: str | None = None,
        video_platform: str | None = None,
        video_duration: float | None = None,
    ) -> list[dict[str, Any]]:
        snippets = self._build_snippets(transcript)
        if not snippets:
            return []

        compact_snippets = self._truncate_snippets(
            snippets,
            max_chars=min(12000, max(4000, _ANALYZER_CHUNK_MAX_CHARS)),
        )
        system_prompt = """\
You are helping preprocess a long-form video before it is split into fixed-length short parts.

Return ONLY ranges that should be removed because they are:
- intro
- outro
- ad

Rules:
- Use exact transcript-aligned boundaries only.
- Keep the main content whenever uncertain.
- Mark confidence below 0.70 if there is any ambiguity.
- Ads can appear anywhere in the transcript.
- Intro/outro ranges should usually be near the start or end of the video unless the transcript clearly indicates a separate promo break.

Return a JSON object with top-level "segments". Each item must include:
- kind: "intro" | "outro" | "ad"
- startTime
- endTime
- confidence
- reason
"""
        user_message = (
            f"{self._video_context_section(video_title=video_title, video_platform=video_platform, video_duration=video_duration)}"
            "Transcript lines are formatted as start|end|text.\n"
            "Identify only removable intro/outro/ad ranges.\n\n"
            f"{self._format_transcript_compact(compact_snippets)}"
        )
        result = self._request_structured_payload(
            max_tokens=2048,
            system_prompt=system_prompt,
            user_message=user_message,
            response_format=RemovableRangeResponse,
        )
        raw_segments = self._extract_list_payload(result, ("segments", "ranges", "items")) or []
        normalized: list[dict[str, Any]] = []
        for item in raw_segments:
            if not isinstance(item, dict):
                continue
            kind = str(item.get("kind") or "").strip().lower()
            start = self._as_float(item.get("start_time"))
            if start is None:
                start = self._as_float(item.get("startTime"))
            end = self._as_float(item.get("end_time"))
            if end is None:
                end = self._as_float(item.get("endTime"))
            confidence = self._as_float(item.get("confidence"))
            if kind not in {"intro", "outro", "ad"} or start is None or end is None:
                continue
            if end <= start:
                continue
            normalized.append(
                {
                    "kind": kind,
                    "start": float(start),
                    "end": float(end),
                    "confidence": max(0.0, min(1.0, float(confidence or 0.0))),
                    "reason": str(item.get("reason") or "").strip() or None,
                }
            )
        normalized.sort(key=lambda value: (float(value["start"]), float(value["end"])))
        return normalized

    def generate_segment_titles(
        self,
        segments: list[dict[str, Any]],
        *,
        language_hint: str | None = None,
        video_title: str | None = None,
    ) -> list[str]:
        if not segments:
            return []

        lines = [
            "Generate one short title for each segment.",
            "Keep titles concise, specific, and natural for short-form video.",
            "Use the video's language for the title text.",
            "Do not include numbering, the word Part, or quotation marks.",
        ]
        language_text = str(language_hint or "").strip()
        if language_text:
            lines.append(f"Preferred language: {language_text}")
        if video_title:
            lines.append(f'Source video title: "{video_title}"')
        lines.append("")
        for segment in segments:
            lines.extend(
                [
                    f"Segment {int(segment.get('index') or 0)}:",
                    f"start={float(segment.get('start') or 0.0):.2f}",
                    f"end={float(segment.get('end') or 0.0):.2f}",
                    f"summary={str(segment.get('text') or '').strip()}",
                    "",
                ]
            )

        result = self._request_structured_payload(
            max_tokens=max(1024, min(4096, 256 * len(segments) + 400)),
            system_prompt=(
                "You are a concise short-form video titling assistant. "
                "Return a JSON object with top-level 'titles'."
            ),
            user_message="\n".join(lines).strip(),
            response_format=SegmentTitleResponse,
            model_candidates=self.synthesis_model_candidates,
            configured_model=self.synthesis_configured_model,
        )
        raw_titles = self._extract_list_payload(result, ("titles", "items", "segments")) or []
        titles_by_index: dict[int, str] = {}
        for item in raw_titles:
            if not isinstance(item, dict):
                continue
            index = self._as_int(item.get("index"))
            title = str(item.get("title") or "").strip()
            if index is None or index <= 0 or not title:
                continue
            titles_by_index[index] = title

        ordered: list[str] = []
        for fallback_index, segment in enumerate(segments, start=1):
            requested_index = self._as_int(segment.get("index")) or fallback_index
            ordered.append(titles_by_index.get(requested_index, ""))
        return ordered

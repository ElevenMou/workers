import json
import logging
import os
import time
from typing import Any, Dict, List

from openai import NotFoundError, OpenAI
from pydantic import BaseModel, ConfigDict

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
_DEFAULT_FALLBACK_MODELS = "gpt-4.1-mini,gpt-4.1"
# Approx 4 chars per token; cap transcript to stay within context and reduce cost.
MAX_TRANSCRIPT_CHARS = _get_env_int(
    "OPENAI_ANALYZER_MAX_TRANSCRIPT_CHARS",
    "CLAUDE_ANALYZER_MAX_TRANSCRIPT_CHARS",
    default=-1,
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

Return a JSON object with a top-level "clips" array that matches the requested response format. Verify that duration = end_time - start_time for each clip."""


class ClipCandidate(BaseModel):
    model_config = ConfigDict(extra="ignore")

    rank: int | None = None
    start_time: float | None = None
    end_time: float | None = None
    duration: float | None = None
    clip_title: str | None = None
    hook: str | None = None
    summary: str | None = None
    confidence_score: float | None = None
    tags: list[str] | None = None


class ClipAnalysisResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    clips: list[ClipCandidate]


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

    def _create_with_model_fallback(self, *, max_tokens: int, system_prompt: str, user_message: str):
        last_error: Exception | None = None
        for index, candidate_model in enumerate(self.model_candidates):
            try:
                completion = self.client.beta.chat.completions.parse(
                    model=candidate_model,
                    max_completion_tokens=max_tokens,
                    temperature=0,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    response_format=ClipAnalysisResponse,
                )
                if candidate_model != self.configured_model:
                    logger.warning(
                        "Configured OpenAI model '%s' unavailable; using fallback '%s'.",
                        self.configured_model,
                        candidate_model,
                    )
                    AIAnalyzer._model_resolution_cache[self.configured_model] = candidate_model
                else:
                    AIAnalyzer._model_resolution_cache.pop(self.configured_model, None)
                self.model = candidate_model
                return completion
            except Exception as exc:
                last_error = exc
                is_model_not_found = self._is_model_not_found_error(exc)
                has_next_candidate = index < len(self.model_candidates) - 1
                if not is_model_not_found or not has_next_candidate:
                    raise
                logger.warning(
                    "OpenAI model '%s' not found; retrying with fallback '%s'.",
                    candidate_model,
                    self.model_candidates[index + 1],
                )

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
        """Analyze transcript using OpenAI to find best clips."""

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

        transcript_text = self._format_transcript_compact(snippets)
        system_prompt = (
            f"{_STATIC_SYSTEM_PROMPT}\n\n"
            f"For this request: extract {num_clips} clip(s), "
            f"each {min_duration}-{max_duration}s duration. "
            f"Return exactly {num_clips} if possible; fewer if video is too short. "
            f"Quality over quantity."
        )

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

        max_tokens = min(4096, max(512, num_clips * 200 + 100))

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

            parsed = self._coerce_parsed_payload(getattr(message, "parsed", None))
            if parsed is not None:
                result = parsed
                raw_clips = self._extract_raw_clips(parsed)
                if isinstance(raw_clips, list):
                    candidate_source = "parsed_response"

            if raw_clips is None:
                response_text = self._message_text(getattr(message, "content", None))
                if response_text:
                    payload = self._extract_json_payload(response_text)
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
                    "OpenAI API circuit breaker opened after %d consecutive failures. "
                    "Cooldown: %.0fs",
                    AIAnalyzer._consecutive_failures,
                    _CIRCUIT_BREAKER_COOLDOWN,
                )
            logger.exception("OpenAI analyzer call failed: %s", e)
            raise

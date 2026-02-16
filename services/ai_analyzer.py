import json
import logging
import os
from typing import Dict, List

from anthropic import Anthropic

logger = logging.getLogger(__name__)

# Default to Sonnet for ~5x lower cost vs Opus; set CLAUDE_ANALYZER_MODEL for override
# e.g. claude-3-5-haiku-20241022 (cheapest/fastest), claude-opus-4-20250514 (highest quality)
DEFAULT_ANALYZER_MODEL = os.getenv("CLAUDE_ANALYZER_MODEL", "claude-sonnet-4-20250514")
# Approx 4 chars per token; cap transcript to stay within context and reduce cost
MAX_TRANSCRIPT_CHARS = int(os.getenv("CLAUDE_ANALYZER_MAX_TRANSCRIPT_CHARS", "-1"))


class AIAnalyzer:
    def __init__(self):
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = DEFAULT_ANALYZER_MODEL

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

    def find_best_clips(
        self,
        transcript: dict,
        num_clips: int = 5,
        min_duration: int = 60,
        max_duration: int = 90,
    ) -> List[Dict]:
        """Analyze transcript using Claude to find best clips (model configurable via CLAUDE_ANALYZER_MODEL).

        Args:
            transcript: Video transcript with segments
            num_clips: Number of clips to generate (1-10)
            min_duration: Minimum clip duration in seconds
            max_duration: Maximum clip duration in seconds
        """

        # Format transcript for Claude
        snippets = [
            {
                "text": segment["text"],
                "start": segment["start"],
                "duration": segment["end"] - segment["start"],
            }
            for segment in transcript["segments"]
        ]
        snippets = self._truncate_snippets(snippets, MAX_TRANSCRIPT_CHARS)

        system_prompt = f"""You are a short-form clip editor. Extract exactly {num_clips} clip(s) from the transcript for TikTok/Reels/Shorts.

TIMESTAMPS (strict): Use ONLY timestamps from the transcript. No inventing or rounding. Clips = consecutive entries only. start_time = first entry start; end_time = last entry (start+duration). duration = end_time - start_time. Never exceed transcript bounds.

DURATION: Clips between {min_duration}-{max_duration}s when possible. Discard or trim if over {max_duration}s. Shorter (15-50s) OK if needed.

COUNT: Return exactly {num_clips} clip(s) if possible; fewer if video too short. Quality over quantity.

EDITORIAL: High insight, strong opinions, actionable advice, emotional impact. Self-contained. No filler/intros/outros. Strong hook in first 3-5s. Rank by quality (1=best).

OUTPUT: Valid JSON only. No markdown, no comments. Timestamps in numeric seconds. Schema: {{"clips": [{{"rank", "start_time", "end_time", "duration", "clip_title", "hook", "summary", "confidence_score", "tags"}}]}} max {num_clips} items. Verify duration = end_time - start_time and duration <= {max_duration}."""

        user_message = f"""Extract {num_clips} clip(s). Use ONLY transcript timestamps. Return exactly {num_clips} if possible. Valid {min_duration}-{max_duration}s preferred; shorter OK.

Transcript (each: text, start, duration):
{json.dumps(snippets)}

Return ONLY valid JSON matching the schema."""

        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )

            # Extract JSON from response
            response_text = message.content[0].text.strip()

            # Remove markdown if present
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()

            result = json.loads(response_text)

            # Convert to our format
            clips = []
            for clip in result["clips"]:
                clips.append(
                    {
                        "start": clip["start_time"],
                        "end": clip["end_time"],
                        "duration": clip["duration"],
                        "text": clip["summary"],
                        "title": clip["clip_title"],
                        "hook": clip["hook"],
                        "score": clip["confidence_score"],
                        "tags": clip["tags"],
                        "rank": clip["rank"],
                    }
                )

            return clips

        except Exception as e:
            logger.exception("Claude analyzer call failed: %s", e)
            raise

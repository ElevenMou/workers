from anthropic import Anthropic
import os
import json
from typing import List, Dict


class AIAnalyzer:
    def __init__(self):
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def find_best_clips(
        self,
        transcript: dict,
        num_clips: int = 5,
        min_duration: int = 60,
        max_duration: int = 90,
    ) -> List[Dict]:
        """Analyze transcript using Claude Opus 4 to find best clips

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

        system_prompt = f"""You are a professional short-form content editor and viral clip strategist.
Your job is to analyze a video transcript with timestamps and extract exactly {num_clips} clip(s) optimized for short-form platforms (TikTok, Instagram Reels, YouTube Shorts).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL TIMESTAMP RULES (MUST FOLLOW STRICTLY)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. You MUST ONLY use timestamps that come directly from the transcript.
2. You MUST NOT invent, guess, extrapolate, round, or extend timestamps.
3. Each transcript entry contains:
   - start: numeric seconds (e.g., 6.685 (6 seconds and 685 milliseconds))
   - duration: numeric seconds (e.g., 8.26 (8 seconds and 26 milliseconds))
4. A clip may ONLY be formed by combining CONSECUTIVE transcript entries.
5. clip.start_time MUST equal the start of the FIRST transcript entry used.
6. clip.end_time MUST equal (start + duration) of the LAST transcript entry used.
7. clip.duration MUST equal (end_time - start_time).
8. clip.end_time MUST be ≤ total video duration.
9. You MUST NEVER extend a clip beyond transcript boundaries.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ABSOLUTE DURATION LIMIT (NON-NEGOTIABLE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- A clip's duration MUST be between {min_duration} and {max_duration} seconds when possible.
- Even if the content is strong, you MUST trim or discard clips that exceed {max_duration} seconds.
- Shorter clips (15-50s) are acceptable if no valid {min_duration}-{max_duration}s clip exists.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CLIP COUNT REQUIREMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- You MUST return exactly {num_clips} clip(s).
- If the video is too short to extract {num_clips} clips, return as many valid clips as possible.
- Prioritize quality over quantity - better to return fewer high-quality clips than force low-quality ones.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EDITORIAL RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Identify moments with high insight, strong opinions, actionable advice, or emotional impact.
- Prefer self-contained segments that make sense without external context.
- Avoid filler, intros, greetings, outros, ads, or rambling.
- Ensure each clip has a strong hook within the first 3-5 seconds.
- Rank clips by quality (1 = best).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT RULES (STRICT)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Output ONLY valid JSON.
- No explanations, no comments, no markdown.
- All timestamps MUST be in NUMERIC SECONDS (e.g., 125.5, not "00:02:05").
- All durations MUST be in NUMERIC SECONDS.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REQUIRED OUTPUT SCHEMA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{{
  "type": "object",
  "required": ["clips"],
  "properties": {{
    "clips": {{
      "type": "array",
      "minItems": 1,
      "maxItems": {num_clips},
      "items": {{
        "type": "object",
        "required": [
          "rank",
          "start_time",
          "end_time",
          "duration",
          "clip_title",
          "hook",
          "summary",
          "confidence_score",
          "tags"
        ],
        "properties": {{
          "rank": {{ "type": "integer", "minimum": 1 }},
          "start_time": {{ "type": "number" }},
          "end_time": {{ "type": "number" }},
          "duration": {{ "type": "number", "maximum": {max_duration} }},
          "clip_title": {{ "type": "string" }},
          "hook": {{ "type": "string" }},
          "summary": {{ "type": "string" }},
          "confidence_score": {{ "type": "number", "minimum": 0, "maximum": 1 }},
          "tags": {{ "type": "string" }}
        }}
      }}
    }}
  }}
}}

VALIDATION CHECKPOINT:
Before returning, verify EVERY clip:
- duration = end_time - start_time
- duration ≤ {max_duration}
- start_time and end_time exist in transcript
If ANY clip fails, DISCARD it."""

        user_message = f"""Analyze the transcript below and extract exactly {num_clips} short-form clip(s) according to the system instructions.

IMPORTANT:
- You MUST strictly follow all timestamp rules defined in the system prompt.
- Use ONLY the timestamps provided in the transcript.
- Do NOT invent, round, or extend any timestamps.
- Return exactly {num_clips} clip(s) if possible.
- If no valid {min_duration}-{max_duration} second clip exists, return the best shorter clip instead.

Transcript format:
Each entry contains:
- text
- start
- duration

Transcript:
{json.dumps(snippets)}

Return ONLY valid JSON that matches the required output structure.
- Do NOT include explanations
- Do NOT include markdown
- Do NOT include comments"""

        try:
            message = self.client.messages.create(
                model="claude-opus-4-20250514",
                max_tokens=4096,
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
            print(f"Error calling Claude API: {e}")
            raise

import logging
import os
import traceback
from datetime import datetime, timedelta

from config import TEMP_DIR, calculate_video_analysis_cost
from services.ai_analyzer import AIAnalyzer
from services.transcriber import Transcriber
from services.video_downloader import VideoDownloader
from utils.supabase_client import supabase, update_job_status, update_video_status

logger = logging.getLogger(__name__)


def analyze_video_task(job_data: dict):
    """Main task for video analysis.

    Each invocation gets its own working directory under ``TEMP_DIR`` so
    multiple workers can process different videos concurrently without
    file-path collisions.
    """
    job_id = job_data["jobId"]
    video_id = job_data["videoId"]
    user_id = job_data["userId"]
    url = job_data["url"]
    num_clips = job_data.get("numClips", 5)

    # -- Per-job working directory for isolation --------------------------
    work_dir = os.path.join(TEMP_DIR, f"analyze_{job_id}")
    os.makedirs(work_dir, exist_ok=True)

    downloader = VideoDownloader(work_dir=work_dir)
    transcriber = Transcriber()
    analyzer = AIAnalyzer()

    video_path = None
    audio_path = None

    try:
        update_job_status(job_id, "processing", 0)
        update_video_status(video_id, "downloading")

        # 1. Download video --------------------------------------------------
        logger.info("[%s] Downloading video: %s", job_id, url)
        video_data = downloader.download(url, video_id)
        video_path = video_data["path"]
        update_job_status(job_id, "processing", 20)

        # Calculate credits based on duration
        duration_seconds = int(video_data["duration"])
        credits_required = calculate_video_analysis_cost(duration_seconds)

        logger.info(
            "[%s] Video duration: %ds (~%.1f min) - credits required: %d",
            job_id,
            duration_seconds,
            duration_seconds / 60,
            credits_required,
        )

        # Check credits
        result = supabase.rpc(
            "has_credits", {"p_user_id": user_id, "p_amount": credits_required}
        ).execute()

        if not result.data:
            raise Exception(f"Insufficient credits. Required: {credits_required}")

        # Update video metadata (keep raw_video_path for later clip generation)
        update_video_status(
            video_id,
            "analyzing",
            title=video_data["title"],
            duration_seconds=duration_seconds,
            thumbnail_url=video_data["thumbnail"],
            platform=video_data["platform"],
            external_id=video_data.get("external_id"),
            raw_video_path=video_path,
            raw_video_expires_at=(datetime.now() + timedelta(hours=24)).isoformat(),
        )

        # 2. Get transcript ---------------------------------------------------
        transcript = None
        if video_data["platform"] == "youtube" and video_data.get("external_id"):
            logger.info("[%s] Attempting to get YouTube transcript …", job_id)
            transcript = downloader.get_youtube_transcript(video_data["external_id"])
            if transcript:
                logger.info("[%s] Got transcript from YouTube", job_id)
                update_job_status(job_id, "processing", 60)

        if not transcript:
            logger.info("[%s] Extracting audio for Whisper transcription …", job_id)
            audio_path = downloader.extract_audio(video_path)
            update_job_status(job_id, "processing", 40)

            logger.info("[%s] Transcribing with Whisper …", job_id)
            transcript = transcriber.transcribe(audio_path)
            transcript["source"] = "whisper"
            update_job_status(job_id, "processing", 60)

        # 3. Analyse with Claude -----------------------------------------------
        logger.info("[%s] Analysing for %d clips with Claude …", job_id, num_clips)
        clips = analyzer.find_best_clips(transcript, num_clips=num_clips)
        update_job_status(job_id, "processing", 90)

        # 4. Save results -----------------------------------------------------
        logger.info("[%s] Saving results …", job_id)
        supabase.table("videos").update(
            {"status": "completed", "transcript": transcript}
        ).eq("id", video_id).execute()

        # Insert clip suggestions (enforce duration constraints: 60-90 s)
        MIN_CLIP_SECONDS = 60
        MAX_CLIP_SECONDS = 90
        inserted_count = 0

        for clip in clips:
            start = float(clip["start"])
            end = float(clip["end"])

            # Clamp within video bounds
            end = min(end, duration_seconds)
            start = max(0.0, min(start, end))
            duration = end - start

            # Trim if too long
            if duration > MAX_CLIP_SECONDS:
                end = start + MAX_CLIP_SECONDS
                duration = end - start

            # Expand if too short
            if duration < MIN_CLIP_SECONDS:
                desired_end = min(duration_seconds, start + MIN_CLIP_SECONDS)
                if desired_end - start < MIN_CLIP_SECONDS and start > 0:
                    start = max(0.0, desired_end - MIN_CLIP_SECONDS)
                end = desired_end
                duration = end - start

            if duration < MIN_CLIP_SECONDS or duration > MAX_CLIP_SECONDS:
                logger.warning(
                    "[%s] Skipping clip outside duration bounds (%.2fs): %.2f-%.2f",
                    job_id,
                    duration,
                    start,
                    end,
                )
                continue

            supabase.table("clips").insert(
                {
                    "video_id": video_id,
                    "user_id": user_id,
                    "start_time": start,
                    "end_time": end,
                    "title": clip["title"],
                    "ai_score": clip["score"],
                    "transcript_excerpt": clip["text"][:500],
                    "status": "pending",
                }
            ).execute()
            inserted_count += 1

        # Charge credits (dynamic amount)
        supabase.rpc(
            "charge_credits",
            {
                "p_user_id": user_id,
                "p_amount": credits_required,
                "p_type": "video_analysis",
                "p_description": (
                    f"Video analysis ({duration_seconds / 60:.1f}min): "
                    f'{video_data["title"][:50]}'
                ),
                "p_video_id": video_id,
                "p_clip_id": None,
            },
        ).execute()

        supabase.table("videos").update(
            {"credits_charged": credits_required, "clip_count": inserted_count}
        ).eq("id", video_id).execute()

        update_job_status(job_id, "completed", 100, result_data={
            "clip_count": inserted_count,
            "duration_seconds": duration_seconds,
            "credits_charged": credits_required,
        })
        logger.info(
            "[%s] Video analysis completed - %d clips saved", job_id, inserted_count
        )

    except Exception as e:
        error_msg = str(e)
        logger.error("[%s] Error analysing video: %s", job_id, error_msg)
        logger.debug(traceback.format_exc())

        update_job_status(job_id, "failed", 0, error_msg)
        update_video_status(video_id, "failed", error_message=error_msg)
        raise

    finally:
        # Clean up audio (video stays - needed for clip generation)
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)
            logger.debug("[%s] Removed temp audio %s", job_id, audio_path)

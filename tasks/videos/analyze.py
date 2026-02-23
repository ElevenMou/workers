import logging
import os
import traceback
from datetime import datetime, timedelta, timezone

from config import calculate_video_analysis_cost
from services.ai_analyzer import AIAnalyzer
from services.transcriber import Transcriber
from services.video_downloader import VideoDownloader
from tasks.models.jobs import AnalyzeVideoJob
from utils.workdirs import create_work_dir
from utils.supabase_client import (
    assert_response_ok,
    charge_video_analysis_credits,
    get_credit_balance,
    get_team_wallet_balance,
    has_sufficient_credits,
    supabase,
    update_job_status,
    update_video_status,
)

logger = logging.getLogger(__name__)


def _update_analysis_job_progress(job_id: str, progress: int, stage: str):
    """Persist progress percentage plus a machine-readable stage label."""
    update_job_status(
        job_id,
        "processing",
        progress,
        result_data={"stage": stage},
    )


def _best_effort_mark_failed(*, job_id: str, video_id: str, error_msg: str):
    try:
        update_job_status(job_id, "failed", 0, error_msg)
    except Exception as exc:
        logger.warning("[%s] Failed to update job failed status: %s", job_id, exc)

    try:
        update_video_status(video_id, "failed", error_message=error_msg)
    except Exception as exc:
        logger.warning("[%s] Failed to mark video %s failed: %s", job_id, video_id, exc)


def analyze_video_task(job_data: AnalyzeVideoJob):
    """Main task for video analysis.

    Each invocation gets its own working directory under ``TEMP_DIR`` so
    multiple workers can process different videos concurrently without
    file-path collisions.
    """
    job_id = job_data["jobId"]
    video_id = job_data["videoId"]
    user_id = job_data["userId"]
    workspace_team_id = job_data.get("workspaceTeamId")
    billing_owner_user_id = job_data.get("billingOwnerUserId") or user_id
    charge_source = str(job_data.get("chargeSource") or "owner_wallet")
    url = job_data["url"]
    num_clips = job_data.get("numClips", 5)
    expected_credits = int(job_data.get("analysisCredits") or 0)

    # -- Per-job working directory for isolation --------------------------
    work_dir = create_work_dir(f"analyze_{job_id}")

    downloader = VideoDownloader(work_dir=work_dir)
    transcriber = Transcriber()
    analyzer = AIAnalyzer()

    video_path = None
    audio_path = None

    try:
        _update_analysis_job_progress(job_id, 0, "starting")

        if expected_credits > 0 and not has_sufficient_credits(
            user_id=billing_owner_user_id,
            amount=expected_credits,
            charge_source=charge_source,
            team_id=workspace_team_id,
        ):
            available = (
                get_team_wallet_balance(workspace_team_id)
                if charge_source == "team_wallet" and workspace_team_id
                else get_credit_balance(billing_owner_user_id)
            )
            raise RuntimeError(
                "Insufficient credits for analysis before processing starts: "
                f"required={expected_credits}, available={available}"
            )

        update_video_status(video_id, "downloading")

        # 1. Download video --------------------------------------------------
        _update_analysis_job_progress(job_id, 5, "downloading_video")
        logger.info("[%s] Downloading video: %s", job_id, url)
        video_data = downloader.download(url, video_id)
        video_path = video_data["path"]
        _update_analysis_job_progress(job_id, 20, "downloading_video")

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
            raw_video_expires_at=(
                datetime.now(timezone.utc) + timedelta(hours=24)
            ).isoformat(),
        )

        # 2. Get transcript ---------------------------------------------------
        transcript = None
        if video_data["platform"] == "youtube" and video_data.get("external_id"):
            _update_analysis_job_progress(job_id, 35, "fetching_source_captions")
            logger.info("[%s] Attempting to get YouTube transcript ...", job_id)
            transcript = downloader.get_youtube_transcript(video_data["external_id"])
            if transcript:
                logger.info("[%s] Got transcript from YouTube", job_id)
                _update_analysis_job_progress(job_id, 60, "fetching_source_captions")

        if not transcript:
            _update_analysis_job_progress(job_id, 35, "extracting_audio")
            logger.info("[%s] Extracting audio for Whisper transcription ...", job_id)
            audio_path = downloader.extract_audio(video_path)
            _update_analysis_job_progress(job_id, 40, "extracting_audio")

            _update_analysis_job_progress(job_id, 50, "transcribing_audio")
            logger.info("[%s] Transcribing with Whisper ...", job_id)
            transcript = transcriber.transcribe(audio_path)
            transcript["source"] = "whisper"
            _update_analysis_job_progress(job_id, 60, "transcribing_audio")

        # 3. Analyse with Claude -----------------------------------------------
        _update_analysis_job_progress(job_id, 70, "analyzing_transcript")
        logger.info("[%s] Analysing for %d clips with Claude ...", job_id, num_clips)
        clips = analyzer.find_best_clips(transcript, num_clips=num_clips)
        _update_analysis_job_progress(job_id, 90, "saving_results")

        # 4. Save results -----------------------------------------------------
        logger.info("[%s] Saving results ...", job_id)
        save_video_resp = supabase.table("videos").update(
            {"transcript": transcript}
        ).eq("id", video_id).execute()
        assert_response_ok(save_video_resp, f"Failed to save transcript for {video_id}")

        # Insert clip suggestions (enforce duration constraints: 60-90 s)
        MIN_CLIP_SECONDS = 60
        MAX_CLIP_SECONDS = 90
        inserted_count = 0
        existing_resp = (
            supabase.table("clips")
            .select("start_time,end_time,title")
            .eq("video_id", video_id)
            .execute()
        )
        assert_response_ok(existing_resp, f"Failed to load existing clips for {video_id}")
        existing_keys = {
            (
                round(float(row["start_time"]), 2),
                round(float(row["end_time"]), 2),
                (row.get("title") or "").strip().lower(),
            )
            for row in (existing_resp.data or [])
        }

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

            # ai_score is DECIMAL(3,2); clamp to 0-1 to avoid numeric overflow
            raw_score = clip.get("score")
            if raw_score is not None:
                try:
                    score = max(0.0, min(1.0, float(raw_score)))
                except (TypeError, ValueError):
                    score = None
            else:
                score = None

            clip_key = (
                round(start, 2),
                round(end, 2),
                str(clip.get("title", "")).strip().lower(),
            )
            if clip_key in existing_keys:
                logger.info(
                    "[%s] Skipping duplicate clip suggestion %.2f-%.2f '%s'",
                    job_id,
                    start,
                    end,
                    clip.get("title", ""),
                )
                continue

            clip_insert_resp = supabase.table("clips").insert(
                {
                    "video_id": video_id,
                    "user_id": user_id,
                    "team_id": workspace_team_id,
                    "billing_owner_user_id": billing_owner_user_id,
                    "start_time": start,
                    "end_time": end,
                    "title": clip["title"],
                    "ai_score": round(score, 2) if score is not None else None,
                    "transcript_excerpt": clip["text"][:500],
                    "status": "pending",
                }
            ).execute()
            assert_response_ok(
                clip_insert_resp,
                f"Failed to insert clip suggestion for {video_id} ({start}-{end})",
            )
            existing_keys.add(clip_key)
            inserted_count += 1

        # Charge credits atomically at finalization time.
        _update_analysis_job_progress(job_id, 95, "charging_credits")
        charge_video_analysis_credits(
            user_id=user_id,
            amount=credits_required,
            description=(
                f"Video analysis ({duration_seconds / 60:.1f}min): "
                f'{video_data["title"][:50]}'
            ),
            video_id=video_id,
            charge_source=charge_source,
            team_id=workspace_team_id,
            billing_owner_user_id=billing_owner_user_id,
            actor_user_id=user_id,
            job_id=job_id,
        )

        finalize_video_resp = supabase.table("videos").update(
            {
                "status": "completed",
                "credits_charged": credits_required,
                "clip_count": inserted_count,
            }
        ).eq("id", video_id).execute()
        assert_response_ok(
            finalize_video_resp,
            f"Failed to finalize video metadata for {video_id}",
        )

        _update_analysis_job_progress(job_id, 99, "finalizing")
        update_job_status(job_id, "completed", 100, result_data={
            "stage": "completed",
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

        _best_effort_mark_failed(job_id=job_id, video_id=video_id, error_msg=error_msg)
        raise

    finally:
        # Clean up audio (video stays - needed for clip generation)
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)
            logger.debug("[%s] Removed temp audio %s", job_id, audio_path)

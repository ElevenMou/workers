"""Background task for social publishing of generated clips."""

from __future__ import annotations

import logging
import json
import shutil
import traceback
from datetime import datetime, timezone

from config import YOUTUBE_SHORTS_MAX_DURATION_SECONDS
from services.social import publish_to_provider
from services.social.base import (
    PublicationContext,
    PublicationResult,
    SocialAccountContext,
    SocialAccountTokens,
    SocialProviderError,
)
from services.clips.render_profiles import publish_profile_for_provider
from services.social.crypto import (
    decrypt_text,
    encrypt_text,
    parse_token_expiry,
    token_is_expired,
)
from services.social.media import load_publication_media
from tasks.clips.helpers.lifecycle import build_progress_result_data
from tasks.models.jobs import PublishClipJob
from utils.sentry_context import configure_job_scope
from utils.supabase_client import assert_response_ok, supabase, update_job_status
from utils.workdirs import create_work_dir

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _has_retries_remaining() -> bool:
    try:
        from rq import get_current_job

        current = get_current_job()
        if current is None:
            return False
        return int(getattr(current, "retries_left", 0) or 0) > 0
    except Exception:
        return False


def _best_effort_update_publication(publication_id: str, payload: dict) -> None:
    response = (
        supabase.table("clip_publications")
        .update(payload)
        .eq("id", publication_id)
        .execute()
    )
    assert_response_ok(response, f"Failed to update publication {publication_id}")


def _load_publication_row(publication_id: str) -> dict:
    response = (
        supabase.table("clip_publications")
        .select("*")
        .eq("id", publication_id)
        .maybe_single()
        .execute()
    )
    assert_response_ok(response, f"Failed to load publication {publication_id}")
    if not response.data:
        raise SocialProviderError(
            f"Publication {publication_id} no longer exists (may have been canceled).",
            code="publication_not_found",
        )
    return response.data


def _load_clip_row(clip_id: str) -> dict:
    response = (
        supabase.table("clips")
        .select(
            "id, video_id, title, duration_seconds, status, storage_path, "
            "delivery_storage_path, master_storage_path, delivery_profile, publish_profile_used"
        )
        .eq("id", clip_id)
        .maybe_single()
        .execute()
    )
    assert_response_ok(response, f"Failed to load clip {clip_id}")
    if not response.data:
        raise SocialProviderError(
            f"Clip {clip_id} no longer exists.",
            code="clip_not_found",
        )
    return response.data


def _load_social_account(account_id: str) -> dict:
    response = (
        supabase.table("social_accounts")
        .select("*")
        .eq("id", account_id)
        .maybe_single()
        .execute()
    )
    assert_response_ok(response, f"Failed to load social account {account_id}")
    if not response.data:
        raise SocialProviderError(
            f"Social account {account_id} no longer exists. Please reconnect the account.",
            code="social_account_not_found",
            refresh_required=True,
        )
    return response.data


def _assert_social_account_ready_for_publish(account: dict) -> None:
    status = str(account.get("status") or "active")
    if status == "active":
        return

    if status == "disconnected":
        raise SocialProviderError(
            "The linked social account was disconnected. Reconnect the account before publishing.",
            code="social_account_disconnected",
        )

    if status == "refresh_required":
        raise SocialProviderError(
            "The linked social account needs to be reconnected before publishing.",
            code="social_account_refresh_required",
        )

    raise SocialProviderError(
        "The linked social account is not available for publishing.",
        code="social_account_unavailable",
    )


def _build_social_account_context(account: dict) -> SocialAccountContext:
    scopes = account.get("scopes") if isinstance(account.get("scopes"), list) else []
    provider_metadata = (
        account.get("provider_metadata")
        if isinstance(account.get("provider_metadata"), dict)
        else {}
    )
    return SocialAccountContext(
        id=str(account["id"]),
        provider=str(account["provider"]),
        external_account_id=str(account["external_account_id"]),
        display_name=str(account.get("display_name") or account["external_account_id"]),
        handle=(str(account["handle"]) if account.get("handle") else None),
        provider_metadata=provider_metadata,
        scopes=[str(item) for item in scopes],
        tokens=SocialAccountTokens(
            access_token=decrypt_text(str(account["encrypted_access_token"])),
            refresh_token=(
                decrypt_text(str(account["encrypted_refresh_token"]))
                if account.get("encrypted_refresh_token")
                else None
            ),
            token_expires_at=parse_token_expiry(account.get("token_expires_at")),
        ),
    )


def _build_publication_context(publication: dict, clip: dict) -> PublicationContext:
    scheduled_for_raw = str(publication.get("scheduled_for") or _utc_now_iso()).replace("Z", "+00:00")
    scheduled_for = datetime.fromisoformat(scheduled_for_raw)
    if scheduled_for.tzinfo is None:
        scheduled_for = scheduled_for.replace(tzinfo=timezone.utc)
    else:
        scheduled_for = scheduled_for.astimezone(timezone.utc)

    return PublicationContext(
        id=str(publication["id"]),
        clip_id=str(publication["clip_id"]),
        clip_title=(str(clip["title"]) if clip.get("title") else None),
        caption=str(publication.get("caption_snapshot") or ""),
        youtube_title=(
            str(publication["youtube_title_snapshot"])
            if publication.get("youtube_title_snapshot")
            else None
        ),
        scheduled_for=scheduled_for,
    )


def _mark_social_account_refresh_required(account_id: str, message: str) -> None:
    try:
        response = (
            supabase.table("social_accounts")
            .update(
                {
                    "status": "refresh_required",
                    "refresh_error": message,
                    "updated_at": _utc_now_iso(),
                }
            )
            .eq("id", account_id)
            .execute()
        )
        assert_response_ok(response, f"Failed to mark social account {account_id} refresh required")
    except Exception as exc:
        logger.warning("Failed to mark social account %s refresh required: %s", account_id, exc)


def _persist_social_account_token_update(account_id: str, result: PublicationResult) -> None:
    payload: dict = {
        "status": "active",
        "refresh_error": None,
        "last_used_at": _utc_now_iso(),
    }
    if result.updated_access_token:
        payload["encrypted_access_token"] = encrypt_text(result.updated_access_token)
    if result.updated_refresh_token:
        payload["encrypted_refresh_token"] = encrypt_text(result.updated_refresh_token)
    if result.updated_token_expires_at:
        payload["token_expires_at"] = result.updated_token_expires_at.astimezone(timezone.utc).isoformat()
    response = (
        supabase.table("social_accounts")
        .update(payload)
        .eq("id", account_id)
        .execute()
    )
    assert_response_ok(response, f"Failed to update social account {account_id}")


def _provider_error_payload(error: SocialProviderError | Exception) -> dict:
    if not isinstance(error, SocialProviderError):
        return {}

    payload: dict = {
        "result_payload": {"provider_error_code": error.code},
    }
    if error.provider_payload:
        payload["provider_payload"] = error.provider_payload
    return payload


def _provider_payload_preview(payload: dict | None, *, limit: int = 2000) -> str:
    if not payload:
        return "{}"
    try:
        rendered = json.dumps(payload, sort_keys=True, default=str)
    except Exception:
        rendered = str(payload)
    if len(rendered) <= limit:
        return rendered
    return f"{rendered[:limit]}..."


def _is_publication_canceled(publication: dict) -> bool:
    return (
        str(publication.get("status") or "") == "canceled"
        or publication.get("canceled_at") is not None
    )


def _complete_publication_job_as_skipped(
    *,
    job_id: str,
    publication_id: str,
    stage: str,
    detail_key: str | None = None,
    extra_result_data: dict | None = None,
) -> None:
    logger.info("[%s] Skipping publication %s during %s", job_id, publication_id, stage)
    update_job_status(
        job_id,
        "completed",
        100,
        result_data=build_progress_result_data(
            stage=stage,
            detail_key=detail_key,
            extra_result_data=extra_result_data,
        ),
    )


def _mark_publication_retrying(publication_id: str, error: SocialProviderError | Exception) -> None:
    payload = {
        "status": "queued",
        "started_at": None,
        "last_error": str(error),
        "updated_at": _utc_now_iso(),
    }
    payload.update(_provider_error_payload(error))
    _best_effort_update_publication(
        publication_id,
        payload,
    )


def _mark_publication_failed(publication_id: str, error: SocialProviderError | Exception) -> None:
    payload = {
        "status": "failed",
        "failed_at": _utc_now_iso(),
        "last_error": str(error),
        "updated_at": _utc_now_iso(),
    }
    payload.update(_provider_error_payload(error))
    _best_effort_update_publication(
        publication_id,
        payload,
    )


def _validate_media_for_provider(provider: str, media_duration: float | None) -> None:
    if provider == "youtube_channel" and media_duration and media_duration > float(YOUTUBE_SHORTS_MAX_DURATION_SECONDS):
        raise SocialProviderError(
            f"YouTube Shorts must be {YOUTUBE_SHORTS_MAX_DURATION_SECONDS} seconds or shorter.",
            code="youtube_duration_limit_exceeded",
        )
    if provider == "instagram_business" and media_duration and media_duration > 900.0:
        raise SocialProviderError(
            "Instagram Reels requires clips that are 15 minutes or shorter.",
            code="instagram_duration_limit_exceeded",
        )


def publish_clip_task(job_data: PublishClipJob) -> None:
    job_id = str(job_data["jobId"])
    publication_id = str(job_data["publicationId"])
    work_dir = create_work_dir(f"publication_{publication_id}")

    configure_job_scope(
        job_id=job_id,
        job_type="publish_clip",
        user_id=job_data.get("userId"),
        clip_id=job_data.get("clipId"),
        extra={"publication_id": publication_id},
    )

    try:
        update_job_status(
            job_id,
            "processing",
            0,
            result_data=build_progress_result_data(
                stage="starting",
                detail_key="publication_started",
            ),
        )

        publication = _load_publication_row(publication_id)
        if _is_publication_canceled(publication):
            _complete_publication_job_as_skipped(
                job_id=job_id,
                publication_id=publication_id,
                stage="canceled",
                detail_key="publication_canceled",
            )
            return

        clip = _load_clip_row(str(publication["clip_id"]))
        social_account = _load_social_account(str(publication["social_account_id"]))

        delivery_storage_path = clip.get("delivery_storage_path") or clip.get("storage_path")
        if clip.get("status") != "completed" or not delivery_storage_path:
            raise SocialProviderError(
                "The clip asset is no longer available for publishing.",
                code="clip_unavailable",
            )
        _assert_social_account_ready_for_publish(social_account)

        started_at = _utc_now_iso()
        transition_resp = (
            supabase.table("clip_publications")
            .update(
                {
                    "status": "publishing",
                    "started_at": started_at,
                    "attempt_count": int(publication.get("attempt_count") or 0) + 1,
                    "last_error": None,
                    "updated_at": started_at,
                }
            )
            .eq("id", publication_id)
            .eq("status", "queued")
            .is_("canceled_at", "null")
            .execute()
        )
        assert_response_ok(
            transition_resp,
            f"Failed to transition publication {publication_id} to publishing",
        )
        publication = _load_publication_row(publication_id)
        if _is_publication_canceled(publication):
            _complete_publication_job_as_skipped(
                job_id=job_id,
                publication_id=publication_id,
                stage="canceled",
                detail_key="publication_canceled",
            )
            return
        if str(publication.get("status") or "") != "publishing":
            _complete_publication_job_as_skipped(
                job_id=job_id,
                publication_id=publication_id,
                stage="skipped",
                detail_key="publication_skipped",
                extra_result_data={"publication_status": publication.get("status")},
            )
            return

        update_job_status(
            job_id,
            "processing",
            20,
            result_data=build_progress_result_data(
                stage="loading_media",
                detail_key="loading_clip_asset",
            ),
        )

        account_context = _build_social_account_context(social_account)
        publication_context = _build_publication_context(publication, clip)
        publish_profile = publish_profile_for_provider(account_context.provider)
        media = load_publication_media(
            str(delivery_storage_path),
            work_dir=work_dir,
        )
        _validate_media_for_provider(account_context.provider, media.duration_seconds)

        if token_is_expired(social_account.get("token_expires_at")):
            if account_context.provider == "youtube_channel":
                from services.social.youtube import refresh_access_token

                refresh_result = refresh_access_token(account_context)
            elif account_context.provider == "tiktok":
                from services.social.tiktok import refresh_access_token

                refresh_result = refresh_access_token(account_context)
            else:
                refresh_result = None

            if refresh_result is None:
                raise SocialProviderError(
                    "The linked social account needs to be reconnected before publishing.",
                    code="social_account_refresh_required",
                    refresh_required=True,
                )

            if refresh_result.updated_access_token:
                account_context.tokens.access_token = refresh_result.updated_access_token
            if refresh_result.updated_refresh_token:
                account_context.tokens.refresh_token = refresh_result.updated_refresh_token
            if refresh_result.updated_token_expires_at:
                account_context.tokens.token_expires_at = refresh_result.updated_token_expires_at
            _persist_social_account_token_update(str(social_account["id"]), refresh_result)

        update_job_status(
            job_id,
            "processing",
            55,
            result_data=build_progress_result_data(
                stage="publishing",
                detail_key="sending_to_provider",
                detail_params={"provider": account_context.provider},
            ),
        )

        result = publish_to_provider(
            provider=account_context.provider,
            account=account_context,
            publication=publication_context,
            media=media,
        )
        _persist_social_account_token_update(str(social_account["id"]), result)
        try:
            clip_update_resp = (
                supabase.table("clips")
                .update({"publish_profile_used": publish_profile})
                .eq("id", clip["id"])
                .execute()
            )
            assert_response_ok(clip_update_resp, f"Failed to update publish profile for clip {clip['id']}")
        except Exception as clip_update_exc:
            logger.warning(
                "[%s] Failed to persist publish profile for clip %s: %s",
                job_id,
                clip["id"],
                clip_update_exc,
            )

        _best_effort_update_publication(
            publication_id,
            {
                "status": "published",
                "published_at": _utc_now_iso(),
                "remote_post_id": result.remote_post_id,
                "remote_post_url": result.remote_post_url,
                "provider_payload": result.provider_payload or {},
                "result_payload": result.result_payload or {},
                "last_error": None,
                "updated_at": _utc_now_iso(),
            },
        )

        update_job_status(
            job_id,
            "completed",
            100,
            result_data=build_progress_result_data(
                stage="completed",
                detail_key="publication_completed",
                extra_result_data={
                    "remote_post_id": result.remote_post_id,
                    "remote_post_url": result.remote_post_url,
                },
            ),
        )
    except Exception as exc:
        if isinstance(exc, SocialProviderError):
            logger.error(
                "[%s] Social publication failed (code=%s recoverable=%s refresh_required=%s): %s | provider_payload=%s",
                job_id,
                exc.code,
                exc.recoverable,
                exc.refresh_required,
                exc,
                _provider_payload_preview(exc.provider_payload),
            )
        else:
            logger.error("[%s] Social publication failed: %s", job_id, exc)
        logger.debug(traceback.format_exc())

        if isinstance(exc, SocialProviderError) and exc.refresh_required:
            try:
                publication = publication if "publication" in locals() else _load_publication_row(publication_id)
                _mark_social_account_refresh_required(str(publication["social_account_id"]), str(exc))
            except Exception as mark_exc:
                logger.warning("Failed to mark account refresh required for publication %s: %s", publication_id, mark_exc)

        if _has_retries_remaining() and isinstance(exc, SocialProviderError) and exc.recoverable:
            try:
                _mark_publication_retrying(publication_id, exc)
                update_job_status(
                    job_id,
                    "retrying",
                    0,
                    str(exc),
                    result_data=build_progress_result_data(stage="retrying"),
                )
            except Exception as update_exc:
                logger.warning("[%s] Failed to mark publication retrying: %s", job_id, update_exc)
            raise

        try:
            _mark_publication_failed(publication_id, exc)
        except Exception as update_exc:
            logger.warning("[%s] Failed to mark publication failed: %s", job_id, update_exc)

        try:
            update_job_status(
                job_id,
                "failed",
                0,
                str(exc),
                result_data=build_progress_result_data(
                    stage="failed",
                    detail_key="publication_failed",
                    detail_params={"message": str(exc)},
                ),
            )
        except Exception as update_exc:
            logger.warning("[%s] Failed to mark job failed: %s", job_id, update_exc)
        raise
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

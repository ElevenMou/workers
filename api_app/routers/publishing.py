"""Clip social publishing endpoints."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, Body, Depends, HTTPException, status

from api_app.access_rules import (
    enforce_social_publishing_access,
    get_user_access_context,
)
from api_app.auth import AuthenticatedUser, get_current_user
from api_app.constants import PUBLISH_TASK_PATH
from config import YOUTUBE_SHORTS_MAX_DURATION_SECONDS
from api_app.helpers import enqueue_or_fail, raise_on_error
from api_app.models import (
    CancelClipPublicationResponse,
    ClipPublicationResponse,
    CreateClipPublicationsRequest,
    CreateClipPublicationsResponse,
    RetryClipPublicationResponse,
)
from utils.supabase_client import supabase

router = APIRouter()
_SOCIAL_PRIORITY_QUEUE = "social-publishing-priority"
_SOCIAL_STANDARD_QUEUE = "social-publishing"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_scheduled_for(payload: CreateClipPublicationsRequest) -> datetime:
    if payload.mode == "now":
        return _utc_now()
    if payload.scheduledAt is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="scheduledAt is required when mode=schedule",
        )
    scheduled_for = payload.scheduledAt
    if scheduled_for.tzinfo is None:
        scheduled_for = scheduled_for.replace(tzinfo=timezone.utc)
    else:
        scheduled_for = scheduled_for.astimezone(timezone.utc)
    if scheduled_for <= _utc_now():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Scheduled publish time must be in the future.",
        )
    return scheduled_for


def _publication_response(row: dict) -> ClipPublicationResponse:
    return ClipPublicationResponse(
        id=str(row["id"]),
        batchId=str(row["batch_id"]),
        clipId=str(row["clip_id"]),
        socialAccountId=str(row["social_account_id"]),
        provider=str(row["provider"]),
        status=str(row["status"]),
        scheduledFor=str(row["scheduled_for"]),
        scheduledTimezone=str(row["scheduled_timezone"]),
        remotePostId=(str(row["remote_post_id"]) if row.get("remote_post_id") else None),
        remotePostUrl=(str(row["remote_post_url"]) if row.get("remote_post_url") else None),
        lastError=(str(row["last_error"]) if row.get("last_error") else None),
        attemptCount=int(row.get("attempt_count") or 0),
    )


def _social_queue_for_context(priority_processing: bool) -> str:
    return _SOCIAL_PRIORITY_QUEUE if priority_processing else _SOCIAL_STANDARD_QUEUE


def _load_workspace_clip(*, clip_id: str, user_id: str, workspace_team_id: str | None) -> dict:
    query = (
        supabase.table("clips")
        .select("id, video_id, user_id, team_id, title, duration_seconds, status, storage_path")
        .eq("id", clip_id)
    )
    if workspace_team_id:
        query = query.eq("team_id", workspace_team_id)
    else:
        query = query.eq("user_id", user_id).is_("team_id", "null")
    response = query.limit(1).execute()
    raise_on_error(response, "Failed to load clip")
    rows = response.data or []
    if not rows:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Clip not found")
    return rows[0]


def _load_social_accounts_for_workspace(
    *,
    social_account_ids: list[str],
    user_id: str,
    workspace_team_id: str | None,
) -> list[dict]:
    query = (
        supabase.table("social_accounts")
        .select("id, provider, user_id, team_id, status, external_account_id")
        .in_("id", social_account_ids)
    )
    if workspace_team_id:
        query = query.eq("team_id", workspace_team_id)
    else:
        query = query.eq("user_id", user_id).is_("team_id", "null")

    response = query.execute()
    raise_on_error(response, "Failed to load social accounts")
    rows = response.data or []
    if len(rows) != len(set(social_account_ids)):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="One or more social accounts do not belong to the active workspace.",
        )
    inactive = [row for row in rows if str(row.get("status") or "") != "active"]
    if inactive:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Reconnect inactive social accounts before publishing.",
        )
    return rows


def _resolve_publish_access(user_id: str):
    context = get_user_access_context(user_id, supabase_client=supabase)
    if context.status == "past_due":
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="Your subscription is past due. Update billing before publishing clips.",
        )
    if context.workspace_team_id and not context.team_writable:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This team workspace is read-only because the owner is below the Pro tier.",
        )
    enforce_social_publishing_access(context=context)
    return context


def _validate_provider_specific_constraints(*, clip: dict, social_accounts: list[dict]) -> None:
    raw_duration = clip.get("duration_seconds")
    try:
        duration_seconds = float(raw_duration) if raw_duration is not None else None
    except (TypeError, ValueError):
        duration_seconds = None

    if duration_seconds is None:
        return

    providers = {str(row.get("provider") or "") for row in social_accounts}

    if "youtube_channel" in providers and duration_seconds > float(YOUTUBE_SHORTS_MAX_DURATION_SECONDS):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"YouTube Shorts requires clips that are {YOUTUBE_SHORTS_MAX_DURATION_SECONDS} seconds or shorter.",
        )

    if "instagram_business" in providers and duration_seconds > 900.0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Instagram Reels requires clips that are 15 minutes or shorter.",
        )


def _load_existing_batch(
    *,
    client_request_id: str,
    user_id: str,
    workspace_team_id: str | None,
) -> tuple[dict | None, list[dict]]:
    batch_resp = (
        supabase.table("clip_publish_batches")
        .select("*")
        .eq("client_request_id", client_request_id)
    )
    if workspace_team_id:
        batch_resp = batch_resp.eq("team_id", workspace_team_id)
    else:
        batch_resp = batch_resp.eq("user_id", user_id).is_("team_id", "null")
    batch_resp = batch_resp.limit(1).execute()
    raise_on_error(batch_resp, "Failed to load existing publish batch")
    batches = batch_resp.data or []
    if not batches:
        return None, []
    batch = batches[0]
    publication_resp = (
        supabase.table("clip_publications")
        .select("*")
        .eq("batch_id", batch["id"])
        .order("created_at")
        .execute()
    )
    raise_on_error(publication_resp, "Failed to load existing clip publications")
    return batch, list(publication_resp.data or [])


def _enqueue_publication_job(
    *,
    publication: dict,
    clip: dict,
    context,
    user_id: str,
) -> None:
    job_id = str(uuid4())
    job_data = {
        "jobId": job_id,
        "publicationId": publication["id"],
        "clipId": clip["id"],
        "userId": user_id,
        "workspaceTeamId": context.workspace_team_id,
        "billingOwnerUserId": context.billing_owner_user_id or user_id,
        "workspaceRole": context.workspace_role,
        "subscriptionTier": context.tier,
        "provider": publication["provider"],
    }
    job_resp = (
        supabase.table("jobs")
        .insert(
            {
                "id": job_id,
                "user_id": user_id,
                "team_id": context.workspace_team_id,
                "billing_owner_user_id": context.billing_owner_user_id or user_id,
                "video_id": clip["video_id"],
                "clip_id": clip["id"],
                "publication_id": publication["id"],
                "type": "publish_clip",
                "status": "queued",
                "input_data": job_data,
            }
        )
        .execute()
    )
    raise_on_error(job_resp, "Failed to create publish job")
    enqueue_or_fail(
        queue_name=_social_queue_for_context(context.priority_processing),
        task_path=PUBLISH_TASK_PATH,
        job_data=job_data,
        job_id=job_id,
        user_id=user_id,
        job_type="publish_clip",
        video_id=clip["video_id"],
        clip_id=clip["id"],
        publication_id=publication["id"],
    )


@router.post("/publishing", response_model=CreateClipPublicationsResponse)
def create_clip_publications(
    payload: CreateClipPublicationsRequest = Body(...),
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> CreateClipPublicationsResponse:
    user_id = current_user.id
    access_context = _resolve_publish_access(user_id)
    caption = payload.caption.strip()
    if not caption:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Caption is required.",
        )

    existing_batch, existing_publications = _load_existing_batch(
        client_request_id=payload.clientRequestId,
        user_id=user_id,
        workspace_team_id=access_context.workspace_team_id,
    )
    if existing_batch is not None:
        return CreateClipPublicationsResponse(
            batchId=str(existing_batch["id"]),
            publications=[_publication_response(row) for row in existing_publications],
        )

    clip = _load_workspace_clip(
        clip_id=payload.clipId,
        user_id=user_id,
        workspace_team_id=access_context.workspace_team_id,
    )
    if clip.get("status") != "completed" or not clip.get("storage_path"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only completed clips with generated assets can be published.",
        )

    social_accounts = _load_social_accounts_for_workspace(
        social_account_ids=payload.socialAccountIds,
        user_id=user_id,
        workspace_team_id=access_context.workspace_team_id,
    )
    youtube_selected = any(str(row.get("provider") or "") == "youtube_channel" for row in social_accounts)
    youtube_title = (payload.youtubeTitle or "").strip() or str(clip.get("title") or "").strip() or None
    if youtube_selected and not youtube_title:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="YouTube publishing requires a title.",
        )
    _validate_provider_specific_constraints(clip=clip, social_accounts=social_accounts)

    scheduled_for = _normalize_scheduled_for(payload)
    batch_id = str(uuid4())
    batch_resp = supabase.table("clip_publish_batches").insert(
        {
            "id": batch_id,
            "clip_id": clip["id"],
            "user_id": user_id,
            "team_id": access_context.workspace_team_id,
            "billing_owner_user_id": access_context.billing_owner_user_id or user_id,
            "caption": caption,
            "youtube_title": youtube_title,
            "scheduled_for": scheduled_for.isoformat(),
            "scheduled_timezone": payload.timeZone,
            "publish_mode": payload.mode,
            "client_request_id": payload.clientRequestId,
        }
    ).execute()
    raise_on_error(batch_resp, "Failed to create clip publish batch")

    publication_rows: list[dict] = []
    initial_status = "queued" if payload.mode == "now" else "scheduled"
    for social_account in social_accounts:
        publication_rows.append(
            {
                "id": str(uuid4()),
                "batch_id": batch_id,
                "clip_id": clip["id"],
                "social_account_id": social_account["id"],
                "provider": social_account["provider"],
                "status": initial_status,
                "caption_snapshot": caption,
                "youtube_title_snapshot": youtube_title,
                "scheduled_for": scheduled_for.isoformat(),
                "scheduled_timezone": payload.timeZone,
                "queued_at": _utc_now().isoformat() if payload.mode == "now" else None,
                "created_by_user_id": user_id,
            }
        )

    publications_resp = (
        supabase.table("clip_publications")
        .insert(publication_rows)
        .execute()
    )
    raise_on_error(publications_resp, "Failed to create clip publications")
    inserted_rows = list(publications_resp.data or publication_rows)

    if payload.mode == "now":
        for publication in inserted_rows:
            _enqueue_publication_job(
                publication=publication,
                clip=clip,
                context=access_context,
                user_id=user_id,
            )

    return CreateClipPublicationsResponse(
        batchId=batch_id,
        publications=[_publication_response(row) for row in inserted_rows],
    )


@router.post("/publishing/{publication_id}/cancel", response_model=CancelClipPublicationResponse)
def cancel_clip_publication(
    publication_id: str,
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> CancelClipPublicationResponse:
    access_context = _resolve_publish_access(current_user.id)
    response = (
        supabase.table("clip_publications")
        .select("*")
        .eq("id", publication_id)
        .limit(1)
        .execute()
    )
    raise_on_error(response, "Failed to load clip publication")
    rows = response.data or []
    if not rows:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Publication not found")
    publication = rows[0]
    _load_workspace_clip(
        clip_id=str(publication["clip_id"]),
        user_id=current_user.id,
        workspace_team_id=access_context.workspace_team_id,
    )
    if str(publication.get("status") or "") != "scheduled":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only scheduled publications can be canceled.",
        )
    canceled_at = _utc_now().isoformat()
    update_resp = (
        supabase.table("clip_publications")
        .update(
            {
                "status": "canceled",
                "canceled_at": canceled_at,
                "updated_at": canceled_at,
            }
        )
        .eq("id", publication_id)
        .execute()
    )
    raise_on_error(update_resp, "Failed to cancel publication")
    publication["status"] = "canceled"
    publication["canceled_at"] = canceled_at
    return CancelClipPublicationResponse(publication=_publication_response(publication))


@router.post("/publishing/{publication_id}/retry", response_model=RetryClipPublicationResponse)
def retry_clip_publication(
    publication_id: str,
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> RetryClipPublicationResponse:
    user_id = current_user.id
    access_context = _resolve_publish_access(user_id)
    response = (
        supabase.table("clip_publications")
        .select("*")
        .eq("id", publication_id)
        .limit(1)
        .execute()
    )
    raise_on_error(response, "Failed to load publication")
    rows = response.data or []
    if not rows:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Publication not found")
    publication = rows[0]
    if str(publication.get("status") or "") != "failed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only failed publications can be retried.",
        )

    clip = _load_workspace_clip(
        clip_id=str(publication["clip_id"]),
        user_id=user_id,
        workspace_team_id=access_context.workspace_team_id,
    )
    queued_at = _utc_now().isoformat()
    update_resp = (
        supabase.table("clip_publications")
        .update(
            {
                "status": "queued",
                "failed_at": None,
                "started_at": None,
                "last_error": None,
                "queued_at": queued_at,
                "updated_at": queued_at,
            }
        )
        .eq("id", publication_id)
        .execute()
    )
    raise_on_error(update_resp, "Failed to retry publication")
    publication["status"] = "queued"
    publication["queued_at"] = queued_at
    publication["failed_at"] = None
    publication["last_error"] = None

    _enqueue_publication_job(
        publication=publication,
        clip=clip,
        context=access_context,
        user_id=user_id,
    )

    return RetryClipPublicationResponse(publication=_publication_response(publication))

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
from utils.media_storage import GeneratedClipStorageError, ensure_generated_clip_available

router = APIRouter()
_SOCIAL_PRIORITY_QUEUE = "social-publishing-priority"
_SOCIAL_STANDARD_QUEUE = "social-publishing"
_ACTIVE_PUBLICATION_STATUSES = ("scheduled", "queued", "publishing")
_PUBLICATION_SELECT = "*, social_account:social_accounts(display_name)"


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


def _normalize_timestamp(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        raw = str(value).strip()
        if not raw:
            return None
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _publication_account_display_name(row: dict) -> str | None:
    social_account = row.get("social_account")
    if isinstance(social_account, dict):
        display_name = social_account.get("display_name")
        if display_name:
            return str(display_name)
    return None


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
        captionSnapshot=(str(row["caption_snapshot"]) if row.get("caption_snapshot") else None),
        youtubeTitleSnapshot=(
            str(row["youtube_title_snapshot"])
            if row.get("youtube_title_snapshot")
            else None
        ),
        queuedAt=(str(row["queued_at"]) if row.get("queued_at") else None),
        startedAt=(str(row["started_at"]) if row.get("started_at") else None),
        publishedAt=(str(row["published_at"]) if row.get("published_at") else None),
        failedAt=(str(row["failed_at"]) if row.get("failed_at") else None),
        canceledAt=(str(row["canceled_at"]) if row.get("canceled_at") else None),
        createdAt=(str(row["created_at"]) if row.get("created_at") else None),
        accountDisplayName=_publication_account_display_name(row),
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


def _assert_clip_storage_ready_for_publish(clip: dict) -> None:
    storage_path = str(clip.get("storage_path") or "").strip()
    if not storage_path:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Clip asset is missing from storage. Regenerate the clip before publishing.",
        )

    try:
        ensure_generated_clip_available(storage_path)
    except GeneratedClipStorageError as exc:
        storage_location = (
            f"{exc.bucket}/{exc.object_name}"
            if exc.object_name
            else storage_path
        )
        if exc.reason in {"invalid_storage_path", "missing_object"}:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"Clip asset is missing from storage ({storage_location}). "
                    "Regenerate the clip before publishing."
                ),
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"Clip storage is temporarily unavailable while checking {storage_location}. "
                "Please retry shortly."
            ),
        ) from exc


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
    return batch, _load_batch_publications(str(batch["id"]))


def _load_batch_publications(batch_id: str) -> list[dict]:
    publication_resp = (
        supabase.table("clip_publications")
        .select(_PUBLICATION_SELECT)
        .eq("batch_id", batch_id)
        .order("created_at")
        .execute()
    )
    raise_on_error(publication_resp, "Failed to load existing clip publications")
    return list(publication_resp.data or [])


def _assert_existing_batch_matches_request(
    *,
    batch: dict,
    publications: list[dict],
    clip_id: str,
    social_account_ids: list[str],
    caption: str,
    youtube_title: str | None,
    scheduled_for: datetime,
    mode: str,
    time_zone: str,
) -> None:
    existing_account_ids = {
        str(row["social_account_id"])
        for row in publications
        if row.get("social_account_id")
    }
    requested_account_ids = {str(value) for value in social_account_ids}
    existing_scheduled_for = _normalize_timestamp(batch.get("scheduled_for"))
    mismatch = (
        str(batch.get("clip_id")) != clip_id
        or str(batch.get("publish_mode") or "") != mode
        or str(batch.get("scheduled_timezone") or "") != time_zone
        or str(batch.get("caption") or "") != caption
        or ((str(batch.get("youtube_title")) if batch.get("youtube_title") else None) != youtube_title)
        or not existing_account_ids.issubset(requested_account_ids)
    )
    if mode == "schedule":
        mismatch = mismatch or existing_scheduled_for != scheduled_for

    if mismatch:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="clientRequestId already belongs to a different publish request.",
        )


def _create_or_load_batch(
    *,
    batch_payload: dict,
    client_request_id: str,
    user_id: str,
    workspace_team_id: str | None,
) -> tuple[dict, bool]:
    try:
        batch_resp = (
            supabase.table("clip_publish_batches")
            .insert(batch_payload)
            .execute()
        )
        raise_on_error(batch_resp, "Failed to create clip publish batch")
        batch_rows = list(batch_resp.data or [batch_payload])
        return batch_rows[0], True
    except HTTPException as exc:
        if exc.status_code != status.HTTP_409_CONFLICT:
            raise

    existing_batch, _ = _load_existing_batch(
        client_request_id=client_request_id,
        user_id=user_id,
        workspace_team_id=workspace_team_id,
    )
    if existing_batch is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="clientRequestId is already in use by another publish batch.",
        )
    return existing_batch, False


def _assert_no_duplicate_active_publications(
    *,
    clip_id: str,
    social_accounts: list[dict],
    scheduled_for: datetime,
    exclude_batch_id: str | None = None,
) -> None:
    if not social_accounts:
        return

    account_ids = [str(row["id"]) for row in social_accounts if row.get("id")]
    duplicate_resp = (
        supabase.table("clip_publications")
        .select(_PUBLICATION_SELECT)
        .eq("clip_id", clip_id)
        .eq("scheduled_for", scheduled_for.isoformat())
        .in_("social_account_id", account_ids)
        .in_("status", list(_ACTIVE_PUBLICATION_STATUSES))
        .execute()
    )
    raise_on_error(duplicate_resp, "Failed to validate active publication duplicates")
    duplicates = [
        row
        for row in list(duplicate_resp.data or [])
        if not exclude_batch_id or str(row.get("batch_id")) != exclude_batch_id
    ]
    if not duplicates:
        return

    destination_names = sorted(
        {
            _publication_account_display_name(row) or str(row["social_account_id"])
            for row in duplicates
        }
    )
    joined_names = ", ".join(destination_names)
    raise HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail=(
            "Active publications already exist for the selected clip, destination, and time slot"
            + (f": {joined_names}." if joined_names else ".")
        ),
    )


def _ensure_batch_publications(
    *,
    batch: dict,
    clip: dict,
    social_accounts: list[dict],
    caption: str,
    youtube_title: str | None,
    scheduled_for: datetime,
    user_id: str,
) -> list[dict]:
    batch_id = str(batch["id"])
    existing_rows = _load_batch_publications(batch_id)
    existing_account_ids = {
        str(row["social_account_id"])
        for row in existing_rows
        if row.get("social_account_id")
    }
    publish_mode = str(batch.get("publish_mode") or "schedule")
    missing_rows: list[dict] = []

    for social_account in social_accounts:
        social_account_id = str(social_account["id"])
        if social_account_id in existing_account_ids:
            continue
        missing_rows.append(
            {
                "id": str(uuid4()),
                "batch_id": batch_id,
                "clip_id": clip["id"],
                "social_account_id": social_account["id"],
                "provider": social_account["provider"],
                "status": "queued" if publish_mode == "now" else "scheduled",
                "caption_snapshot": caption,
                "youtube_title_snapshot": youtube_title,
                "scheduled_for": scheduled_for.isoformat(),
                "scheduled_timezone": batch["scheduled_timezone"],
                "queued_at": _utc_now().isoformat() if publish_mode == "now" else None,
                "created_by_user_id": user_id,
            }
        )

    if not missing_rows:
        return existing_rows

    try:
        publications_resp = (
            supabase.table("clip_publications")
            .insert(missing_rows)
            .execute()
        )
        raise_on_error(publications_resp, "Failed to create clip publications")
    except HTTPException as exc:
        if exc.status_code != status.HTTP_409_CONFLICT:
            raise

    return _load_batch_publications(batch_id)


def _load_latest_publish_jobs_by_publication_id(publication_ids: list[str]) -> dict[str, dict]:
    if not publication_ids:
        return {}

    jobs_resp = (
        supabase.table("jobs")
        .select("id, publication_id, status, created_at")
        .eq("type", "publish_clip")
        .in_("publication_id", publication_ids)
        .order("created_at", desc=True)
        .execute()
    )
    raise_on_error(jobs_resp, "Failed to load existing publish jobs")
    latest_by_publication_id: dict[str, dict] = {}
    for row in list(jobs_resp.data or []):
        publication_id = str(row.get("publication_id") or "")
        if publication_id and publication_id not in latest_by_publication_id:
            latest_by_publication_id[publication_id] = row
    return latest_by_publication_id


def _ensure_dispatch_jobs_for_publications(
    *,
    publications: list[dict],
    clip: dict,
    context,
    user_id: str,
) -> None:
    latest_jobs = _load_latest_publish_jobs_by_publication_id(
        [str(row["id"]) for row in publications if row.get("id")]
    )
    for publication in publications:
        if str(publication.get("status") or "") != "queued":
            continue
        latest_job = latest_jobs.get(str(publication["id"]))
        latest_status = str(latest_job.get("status") or "") if latest_job else ""
        if latest_status in {"queued", "processing", "retrying", "completed"}:
            continue
        _enqueue_publication_job(
            publication=publication,
            clip=clip,
            context=context,
            user_id=user_id,
        )


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
    _assert_clip_storage_ready_for_publish(clip)

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
    batch, created_batch = _create_or_load_batch(
        batch_payload={
            "id": str(uuid4()),
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
        },
        client_request_id=payload.clientRequestId,
        user_id=user_id,
        workspace_team_id=access_context.workspace_team_id,
    )
    batch_id = str(batch["id"])
    existing_publications = _load_batch_publications(batch_id)
    if not created_batch:
        _assert_existing_batch_matches_request(
            batch=batch,
            publications=existing_publications,
            clip_id=str(clip["id"]),
            social_account_ids=payload.socialAccountIds,
            caption=caption,
            youtube_title=youtube_title,
            scheduled_for=scheduled_for,
            mode=payload.mode,
            time_zone=payload.timeZone,
        )

    _assert_no_duplicate_active_publications(
        clip_id=str(clip["id"]),
        social_accounts=social_accounts,
        scheduled_for=scheduled_for,
        exclude_batch_id=batch_id,
    )
    publications = _ensure_batch_publications(
        batch=batch,
        clip=clip,
        social_accounts=social_accounts,
        caption=caption,
        youtube_title=youtube_title,
        scheduled_for=scheduled_for,
        user_id=user_id,
    )

    if payload.mode == "now":
        _ensure_dispatch_jobs_for_publications(
            publications=publications,
            clip=clip,
            context=access_context,
            user_id=user_id,
        )
        publications = _load_batch_publications(batch_id)

    return CreateClipPublicationsResponse(
        batchId=batch_id,
        publications=[_publication_response(row) for row in publications],
    )


@router.post("/publishing/{publication_id}/cancel", response_model=CancelClipPublicationResponse)
def cancel_clip_publication(
    publication_id: str,
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> CancelClipPublicationResponse:
    access_context = _resolve_publish_access(current_user.id)
    response = (
        supabase.table("clip_publications")
        .select(_PUBLICATION_SELECT)
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
    if str(publication.get("status") or "") not in {"scheduled", "queued"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only scheduled or queued publications can be canceled.",
        )
    canceled_at = _utc_now().isoformat()
    update_resp = (
        supabase.table("clip_publications")
        .update(
            {
                "status": "canceled",
                "canceled_at": canceled_at,
                "last_error": None,
                "updated_at": canceled_at,
            }
        )
        .eq("id", publication_id)
        .execute()
    )
    raise_on_error(update_resp, "Failed to cancel publication")
    publication["status"] = "canceled"
    publication["canceled_at"] = canceled_at
    publication["last_error"] = None
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
        .select(_PUBLICATION_SELECT)
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
    _assert_clip_storage_ready_for_publish(clip)
    queued_at = _utc_now().isoformat()
    update_resp = (
        supabase.table("clip_publications")
        .update(
            {
                "status": "queued",
                "published_at": None,
                "failed_at": None,
                "started_at": None,
                "canceled_at": None,
                "remote_post_id": None,
                "remote_post_url": None,
                "provider_payload": {},
                "result_payload": {},
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
    publication["published_at"] = None
    publication["failed_at"] = None
    publication["started_at"] = None
    publication["canceled_at"] = None
    publication["remote_post_id"] = None
    publication["remote_post_url"] = None
    publication["provider_payload"] = {}
    publication["result_payload"] = {}
    publication["last_error"] = None

    _enqueue_publication_job(
        publication=publication,
        clip=clip,
        context=access_context,
        user_id=user_id,
    )

    return RetryClipPublicationResponse(publication=_publication_response(publication))

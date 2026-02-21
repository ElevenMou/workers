"""Authentication/authorization helpers for API routes."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from config import SUPABASE_SERVICE_KEY, SUPABASE_URL

_bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class AuthenticatedUser:
    id: str
    email: str | None
    claims: dict[str, Any]


def _parse_admin_allowlist() -> set[str]:
    raw = os.getenv("WORKER_SCALE_ADMIN_USER_IDS", "")
    if not raw.strip():
        return set()
    return {item.strip() for item in raw.split(",") if item.strip()}


def _token_claims_are_admin(claims: dict[str, Any]) -> bool:
    role = str(claims.get("role") or "").lower()
    if role in {"admin", "service_role"}:
        return True

    app_metadata = claims.get("app_metadata")
    if isinstance(app_metadata, dict):
        app_role = str(app_metadata.get("role") or "").lower()
        if app_role == "admin":
            return True
        if bool(app_metadata.get("is_admin")):
            return True

    # SECURITY: user_metadata is user-writable and must NEVER be trusted for
    # authorization. Only app_metadata (set via service role) is safe.

    return False


def _fetch_user_from_token(token: str) -> dict[str, Any]:
    headers = {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {token}",
    }

    try:
        response = httpx.get(
            f"{SUPABASE_URL}/auth/v1/user",
            headers=headers,
            timeout=10.0,
        )
    except Exception as exc:  # pragma: no cover - network-level failure
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service unavailable",
        ) from exc

    if response.status_code == status.HTTP_401_UNAUTHORIZED:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired bearer token",
        )

    if response.status_code != status.HTTP_200_OK:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Authentication service error ({response.status_code})",
        )

    data = response.json()
    if not isinstance(data, dict) or not data.get("id"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid bearer token payload",
        )
    return data


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> AuthenticatedUser:
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
        )

    payload = _fetch_user_from_token(credentials.credentials)
    return AuthenticatedUser(
        id=str(payload["id"]),
        email=payload.get("email"),
        claims=payload,
    )


def require_admin_user(
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> AuthenticatedUser:
    allowlist = _parse_admin_allowlist()
    if current_user.id in allowlist or _token_claims_are_admin(current_user.claims):
        return current_user

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Admin access required",
    )


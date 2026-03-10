"""Authentication/authorization helpers for API routes."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from api_app.jwt_verifier import extract_rate_limit_user_id, verify_supabase_jwt_locally
from config import SUPABASE_SERVICE_KEY, SUPABASE_URL

try:
    import sentry_sdk as _sentry_sdk
except ImportError:  # pragma: no cover
    _sentry_sdk = None  # type: ignore[assignment]

try:
    from jwt.exceptions import PyJWTError as _PyJWTError
except ImportError:  # pragma: no cover
    _PyJWTError = None  # type: ignore[assignment,misc]

# Tuple of exceptions expected from local JWT verification failures.
_JWT_EXPECTED_ERRORS: tuple[type[BaseException], ...] = (RuntimeError, ValueError, KeyError)
if _PyJWTError is not None:
    _JWT_EXPECTED_ERRORS = (*_JWT_EXPECTED_ERRORS, _PyJWTError)

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

    token = credentials.credentials

    # Try fast local JWT verification first (avoids HTTP round-trip per request).
    try:
        claims = verify_supabase_jwt_locally(token)
        user_id = claims.get("sub") or claims.get("id")
        if user_id:
            user = AuthenticatedUser(
                id=str(user_id),
                email=claims.get("email"),
                claims=claims,
            )
            if _sentry_sdk is not None:
                _sentry_sdk.set_user({"id": user.id, "email": user.email or ""})
            return user
    except _JWT_EXPECTED_ERRORS:
        pass  # Expected failures: missing PyJWT, bad token format, missing claims
    except Exception as exc:
        # Unexpected error — log so it doesn't silently mask bugs.
        import logging
        logging.getLogger(__name__).warning(
            "Unexpected error during local JWT verification, falling through to remote: %s", exc
        )
        if _sentry_sdk is not None:
            _sentry_sdk.capture_exception(exc)

    # Fallback: verify against Supabase Auth HTTP endpoint.
    payload = _fetch_user_from_token(token)
    user = AuthenticatedUser(
        id=str(payload["id"]),
        email=payload.get("email"),
        claims=payload,
    )
    if _sentry_sdk is not None:
        _sentry_sdk.set_user({"id": user.id, "email": user.email or ""})
    return user


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


def get_user_rate_key(request: Request) -> str:
    """Extract the authenticated user ID for per-user rate limiting.

    Falls back to the client IP when the user cannot be resolved (e.g.
    unauthenticated preflight requests) so the limiter always gets a key.
    """
    try:
        auth_header = request.headers.get("authorization", "")
        if auth_header.lower().startswith("bearer "):
            token = auth_header[7:].strip()
            if token:
                user_id = extract_rate_limit_user_id(token)
                if user_id:
                    return f"user:{user_id}"
    except Exception:
        pass
    from slowapi.util import get_remote_address

    return get_remote_address(request)

"""Local Supabase JWT verification with JWKS caching."""

from __future__ import annotations

import os
import threading
import time
from typing import Any

import httpx
try:
    import jwt
except ModuleNotFoundError:  # pragma: no cover - dependency guard for local envs
    jwt = None

from config import SUPABASE_URL

_JWKS_CACHE_TTL_SECONDS = max(
    30,
    int(os.getenv("SUPABASE_JWKS_CACHE_TTL_SECONDS", "300")),
)
_SUPABASE_JWT_ISSUER = os.getenv("SUPABASE_JWT_ISSUER") or f"{SUPABASE_URL}/auth/v1"
_SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")
_JWKS_URL = f"{SUPABASE_URL}/auth/v1/.well-known/jwks.json"

_cache_lock = threading.Lock()
_jwks_cache: dict[str, Any] = {"expires_at": 0.0, "keys_by_kid": {}}


def _refresh_jwks_cache(force: bool = False) -> dict[str, Any]:
    now = time.monotonic()
    with _cache_lock:
        cached = _jwks_cache.copy()
        if not force and now < float(cached.get("expires_at", 0.0)):
            return cached

    response = httpx.get(_JWKS_URL, timeout=5.0)
    response.raise_for_status()
    body = response.json()
    keys = body.get("keys", []) if isinstance(body, dict) else []

    keys_by_kid: dict[str, Any] = {}
    for entry in keys:
        if not isinstance(entry, dict):
            continue
        kid = entry.get("kid")
        if not kid:
            continue
        keys_by_kid[str(kid)] = entry

    updated = {
        "expires_at": now + _JWKS_CACHE_TTL_SECONDS,
        "keys_by_kid": keys_by_kid,
    }
    with _cache_lock:
        _jwks_cache.clear()
        _jwks_cache.update(updated)
        return _jwks_cache.copy()


def _verify_hs256(token: str, algorithm: str) -> dict[str, Any]:
    if jwt is None:
        raise RuntimeError("PyJWT is not installed")
    if not _SUPABASE_JWT_SECRET:
        raise RuntimeError("SUPABASE_JWT_SECRET is required for HS token verification")
    return jwt.decode(
        token,
        _SUPABASE_JWT_SECRET,
        algorithms=[algorithm],
        issuer=_SUPABASE_JWT_ISSUER,
        options={"verify_aud": False},
    )


def _verify_jwks_token(token: str, algorithm: str, kid: str) -> dict[str, Any]:
    if jwt is None:
        raise RuntimeError("PyJWT is not installed")
    cache = _refresh_jwks_cache()
    jwk = cache.get("keys_by_kid", {}).get(kid)
    if not isinstance(jwk, dict):
        cache = _refresh_jwks_cache(force=True)
        jwk = cache.get("keys_by_kid", {}).get(kid)
    if not isinstance(jwk, dict):
        raise RuntimeError(f"Signing key not found for kid={kid!r}")

    signing_key = jwt.algorithms.RSAAlgorithm.from_jwk(jwk)
    return jwt.decode(
        token,
        signing_key,
        algorithms=[algorithm],
        issuer=_SUPABASE_JWT_ISSUER,
        options={"verify_aud": False},
    )


_ALLOWED_ALGORITHMS = frozenset({"HS256", "RS256", "ES256"})


def verify_supabase_jwt_locally(token: str) -> dict[str, Any]:
    """Verify token signature locally and return decoded claims."""
    if jwt is None:
        raise RuntimeError("PyJWT is not installed")
    header = jwt.get_unverified_header(token)
    algorithm = str(header.get("alg") or "RS256")

    if algorithm not in _ALLOWED_ALGORITHMS:
        raise RuntimeError(f"Unsupported JWT algorithm: {algorithm}")

    if algorithm.startswith("HS"):
        return _verify_hs256(token, algorithm)

    kid = str(header.get("kid") or "")
    if not kid:
        raise RuntimeError("Missing token kid")
    return _verify_jwks_token(token, algorithm, kid)


def extract_rate_limit_user_id(token: str) -> str | None:
    """Return a stable user identifier for rate limiting, or ``None``."""
    claims = verify_supabase_jwt_locally(token)
    user_id = claims.get("sub") or claims.get("id")
    if not user_id:
        return None
    return str(user_id)

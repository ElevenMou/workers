"""AES-GCM helpers shared by worker social integrations."""

from __future__ import annotations

import base64
import os
from datetime import datetime, timedelta, timezone

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from config import SOCIAL_ACCOUNT_ENCRYPTION_KEY

_TOKEN_PREFIX = "v1"


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _load_key() -> bytes:
    raw = (SOCIAL_ACCOUNT_ENCRYPTION_KEY or os.getenv("SOCIAL_ACCOUNT_ENCRYPTION_KEY") or "").strip()
    if not raw:
        raise RuntimeError("SOCIAL_ACCOUNT_ENCRYPTION_KEY is not configured.")

    try:
        decoded = base64.b64decode(raw, validate=True)
    except Exception as exc:  # pragma: no cover - invalid env
        raise RuntimeError("SOCIAL_ACCOUNT_ENCRYPTION_KEY must be base64 encoded.") from exc

    if len(decoded) != 32:
        raise RuntimeError("SOCIAL_ACCOUNT_ENCRYPTION_KEY must decode to exactly 32 bytes.")

    return decoded


def _b64url_encode(data: bytes) -> str:
    """Encode as base64url without padding, matching Node.js base64url output."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def encrypt_text(value: str) -> str:
    key = _load_key()
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)
    ciphertext = aesgcm.encrypt(nonce, value.encode("utf-8"), None)
    return ":".join(
        [
            _TOKEN_PREFIX,
            _b64url_encode(nonce),
            _b64url_encode(ciphertext),
        ]
    )


def _b64url_decode(s: str) -> bytes:
    """Decode base64url, adding back padding that Node.js base64url strips."""
    padded = s + "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(padded.encode("ascii"))


def decrypt_text(value: str) -> str:
    key = _load_key()
    parts = value.split(":")
    if len(parts) != 3 or parts[0] != _TOKEN_PREFIX:
        raise RuntimeError("Unsupported encrypted token format.")

    try:
        nonce = _b64url_decode(parts[1])
        ciphertext = _b64url_decode(parts[2])
    except Exception as exc:
        raise RuntimeError("Encrypted token is not valid base64 data.") from exc

    aesgcm = AESGCM(key)
    plaintext = aesgcm.decrypt(nonce, ciphertext, None)
    return plaintext.decode("utf-8")


def token_is_expired(token_expires_at: str | None, *, skew_seconds: int = 60) -> bool:
    parsed = _parse_iso_datetime(token_expires_at)
    if parsed is None:
        return False
    return parsed <= datetime.now(timezone.utc) + timedelta(seconds=skew_seconds)


def parse_token_expiry(token_expires_at: str | None) -> datetime | None:
    return _parse_iso_datetime(token_expires_at)

"""Secret-safe helpers for provider API-key pools."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable

MAX_PROVIDER_API_KEYS = 64
_KEY_SPLIT_RE = re.compile(r"[,\r\n]+")


def normalize_api_keys(
    raw: object,
    *,
    field_name: str = "api_keys",
) -> list[str]:
    """Return a trimmed, unique key list without echoing secrets in errors."""
    values: Iterable[object]
    if raw is None:
        values = ()
    elif isinstance(raw, str):
        text = raw.strip()
        if not text:
            values = ()
        elif text.startswith("["):
            try:
                decoded = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{field_name} must be an API key array") from exc
            if not isinstance(decoded, list):
                raise ValueError(f"{field_name} must be an API key array")
            values = decoded
        else:
            values = _KEY_SPLIT_RE.split(text)
    elif isinstance(raw, (list, tuple)):
        values = raw
    else:
        raise ValueError(f"{field_name} must be an API key array")

    normalized = [str(value).strip() for value in values if str(value).strip()]
    if len(normalized) > MAX_PROVIDER_API_KEYS:
        raise ValueError(
            f"{field_name} allows at most {MAX_PROVIDER_API_KEYS} API keys"
        )
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} cannot contain duplicate API keys")
    return normalized


def decode_api_keys(raw: object) -> list[str]:
    """Decode a persisted pool; malformed legacy content fails closed."""
    try:
        return normalize_api_keys(raw)
    except ValueError:
        return []


def encode_api_keys(api_keys: object) -> str:
    return json.dumps(normalize_api_keys(api_keys), ensure_ascii=False)


def key_identifier(api_key: str) -> str:
    """Return a non-reversible identifier safe for logs and diagnostics."""
    digest = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
    return f"key-{digest[:12]}"


def redact_api_keys(message: str, api_keys: Iterable[object]) -> str:
    """Remove exact configured credentials from a diagnostic string."""
    secrets = {str(value).strip() for value in api_keys if str(value).strip()}
    safe = message
    for secret in sorted(secrets, key=len, reverse=True):
        safe = safe.replace(secret, "<redacted>")
    return safe

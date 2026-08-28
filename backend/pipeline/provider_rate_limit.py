"""Process-wide request pacing for the configured primary model gateway."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from datetime import datetime, timezone
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from backend import config
from backend.provider_catalog import PROVIDER_PROFILE_MAP

logger = logging.getLogger(__name__)
LogCallback = Callable[[str], None]

_request_lock = asyncio.Lock()
_next_request_at: dict[str, float] = {}
_cooldown_until: dict[str, float] = {}


def _primary_gateway_key(endpoint: str) -> str | None:
    profile = PROVIDER_PROFILE_MAP.get(config.AI_PRIMARY_PROVIDER_TYPE)
    if profile is None:
        return None
    selected = urlsplit(endpoint.strip())
    primary = urlsplit(profile.default_endpoint)
    if not selected.hostname or selected.hostname.casefold() != (primary.hostname or "").casefold():
        return None
    port = selected.port or (443 if selected.scheme == "https" else 80)
    return f"{selected.scheme.casefold()}://{selected.hostname.casefold()}:{port}"


def _proactive_limit_active(now: datetime | None = None) -> bool:
    """Return whether weekday daytime request spacing is currently active."""
    try:
        zone = ZoneInfo(config.AI_PRIMARY_RATE_LIMIT_TIMEZONE)
    except ZoneInfoNotFoundError:
        zone = timezone.utc
    local_now = (now or datetime.now(timezone.utc)).astimezone(zone)
    start = max(0, min(23, int(config.AI_PRIMARY_RATE_LIMIT_START_HOUR)))
    end = max(1, min(24, int(config.AI_PRIMARY_RATE_LIMIT_END_HOUR)))
    return local_now.weekday() < 5 and start <= local_now.hour < end


async def wait_for_request_slot(
    endpoint: str,
    *,
    log: LogCallback | None = None,
    label: str = "AI call",
    now: datetime | None = None,
) -> float:
    """Reserve one globally paced OneAPI request start and return wait time."""
    key = _primary_gateway_key(endpoint)
    if key is None:
        return 0.0
    interval = (
        max(0.0, float(config.AI_PRIMARY_MIN_REQUEST_INTERVAL_SECONDS))
        if _proactive_limit_active(now)
        else 0.0
    )

    async with _request_lock:
        monotonic_now = time.monotonic()
        spacing_at = _next_request_at.get(key, monotonic_now) if interval > 0 else monotonic_now
        cooldown_at = _cooldown_until.get(key, monotonic_now)
        delay = max(0.0, max(spacing_at, cooldown_at) - monotonic_now)
        if delay > 0:
            policy = (
                "shared daytime limit: at most 5 requests/minute"
                if interval > 0
                else "explicit 429 cooldown"
            )
            message = (
                f"{label}: OneAPI rate gate waiting {delay:.1f}s before the next request "
                f"({policy})"
            )
            if log is not None:
                log(message)
            else:
                logger.info(message)
            await asyncio.sleep(delay)
        started_at = time.monotonic()
        if interval > 0:
            _next_request_at[key] = started_at + interval
        else:
            _next_request_at.pop(key, None)
        if _cooldown_until.get(key, 0.0) <= started_at:
            _cooldown_until.pop(key, None)
        return delay


async def record_rate_limit(
    endpoint: str,
    *,
    log: LogCallback | None = None,
    label: str = "AI call",
) -> float:
    """Open a process-wide cooldown after OneAPI reports HTTP 429."""
    key = _primary_gateway_key(endpoint)
    if key is None:
        return 0.0
    cooldown = max(
        60.0,
        float(config.AI_PRIMARY_RATE_LIMIT_COOLDOWN_SECONDS),
    )
    async with _request_lock:
        now = time.monotonic()
        _cooldown_until[key] = max(
            _cooldown_until.get(key, now),
            now + cooldown,
        )
    message = (
        f"{label}: OneAPI returned 429; pausing every local OneAPI request for "
        f"{cooldown:.0f}s so the rolling one-minute quota can clear"
    )
    if log is not None:
        log(message)
    else:
        logger.warning(message)
    return cooldown


def reset_for_tests() -> None:
    _next_request_at.clear()
    _cooldown_until.clear()

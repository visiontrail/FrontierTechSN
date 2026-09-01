"""Process-wide pacing for the persisted Galaxy OneAPI primary model."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from datetime import datetime, timezone
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from backend import config

logger = logging.getLogger(__name__)
LogCallback = Callable[[str], None]

_request_lock = asyncio.Lock()
_next_gateway_request_at: dict[str, float] = {}
_next_credential_request_at: dict[str, float] = {}
_gateway_cooldown_until: dict[str, float] = {}


def _yinhe_gateway_keys(
    endpoint: str,
    *,
    route_slot: str,
    provider_type: str,
    api_key_id: str,
) -> tuple[str, str] | None:
    if (
        route_slot != "primary"
        or provider_type.strip().casefold() != "yinhe"
        or not api_key_id
    ):
        return None
    selected = urlsplit(endpoint.strip())
    if not selected.hostname:
        return None
    port = selected.port or (443 if selected.scheme == "https" else 80)
    gateway_key = (
        f"{selected.scheme.casefold()}://{selected.hostname.casefold()}:{port}"
    )
    return gateway_key, f"{gateway_key}/{api_key_id}"


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
    route_slot: str = "standalone",
    provider_type: str = "",
    api_key_id: str = "",
    log: LogCallback | None = None,
    label: str = "AI call",
    now: datetime | None = None,
) -> float:
    """Reserve one globally paced Galaxy OneAPI request start."""
    keys = _yinhe_gateway_keys(
        endpoint,
        route_slot=route_slot,
        provider_type=provider_type,
        api_key_id=api_key_id,
    )
    if keys is None:
        return 0.0
    gateway_key, credential_key = keys
    gateway_interval = max(
        0.0,
        float(config.AI_YINHE_MIN_REQUEST_INTERVAL_SECONDS),
    )
    credential_interval = (
        max(0.0, float(config.AI_PRIMARY_MIN_REQUEST_INTERVAL_SECONDS))
        if _proactive_limit_active(now)
        else 0.0
    )

    async with _request_lock:
        monotonic_now = time.monotonic()
        gateway_spacing_at = (
            _next_gateway_request_at.get(gateway_key, monotonic_now)
            if gateway_interval > 0
            else monotonic_now
        )
        credential_spacing_at = (
            _next_credential_request_at.get(credential_key, monotonic_now)
            if credential_interval > 0
            else monotonic_now
        )
        cooldown_at = _gateway_cooldown_until.get(gateway_key, monotonic_now)
        delay = max(
            0.0,
            max(gateway_spacing_at, credential_spacing_at, cooldown_at) - monotonic_now,
        )
        if delay > 0:
            policies = []
            if gateway_spacing_at > monotonic_now:
                policies.append(
                    f"Galaxy pool spacing: one start/{gateway_interval:.1f}s"
                )
            if credential_spacing_at > monotonic_now:
                policies.append(
                    f"daytime per-key spacing: one start/{credential_interval:.1f}s"
                )
            if cooldown_at > monotonic_now:
                policies.append("shared 429 cooldown")
            message = (
                f"{label}: Galaxy OneAPI rate gate waiting {delay:.1f}s before the next "
                f"request for {api_key_id} ({'; '.join(policies)})"
            )
            if log is not None:
                log(message)
            else:
                logger.info(message)
            await asyncio.sleep(delay)
        started_at = time.monotonic()
        if gateway_interval > 0:
            _next_gateway_request_at[gateway_key] = started_at + gateway_interval
        else:
            _next_gateway_request_at.pop(gateway_key, None)
        if credential_interval > 0:
            _next_credential_request_at[credential_key] = (
                started_at + credential_interval
            )
        else:
            _next_credential_request_at.pop(credential_key, None)
        if _gateway_cooldown_until.get(gateway_key, 0.0) <= started_at:
            _gateway_cooldown_until.pop(gateway_key, None)
        return delay


async def record_rate_limit(
    endpoint: str,
    *,
    route_slot: str = "standalone",
    provider_type: str = "",
    api_key_id: str = "",
    log: LogCallback | None = None,
    label: str = "AI call",
) -> float:
    """Open a process-wide pool cooldown after Galaxy OneAPI reports 429."""
    keys = _yinhe_gateway_keys(
        endpoint,
        route_slot=route_slot,
        provider_type=provider_type,
        api_key_id=api_key_id,
    )
    if keys is None:
        return 0.0
    gateway_key, _ = keys
    cooldown = max(
        60.0,
        float(config.AI_PRIMARY_RATE_LIMIT_COOLDOWN_SECONDS),
    )
    async with _request_lock:
        now = time.monotonic()
        _gateway_cooldown_until[gateway_key] = max(
            _gateway_cooldown_until.get(gateway_key, now),
            now + cooldown,
        )
    message = (
        f"{label}: Galaxy OneAPI returned 429 for {api_key_id}; pausing the complete "
        f"primary key pool for {cooldown:.0f}s so its rolling one-minute quota can clear"
    )
    if log is not None:
        log(message)
    else:
        logger.warning(message)
    return cooldown


def reset_for_tests() -> None:
    _next_gateway_request_at.clear()
    _next_credential_request_at.clear()
    _gateway_cooldown_until.clear()

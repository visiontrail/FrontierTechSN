"""Ordered provider routing for long-running automation tasks.

This is the narrow part of RavenAIService's model router that fits this
project: explicit primary/backup slots and auditable route selection.  The
interactive circuit breaker and short first-token deadlines are intentionally
omitted.  A caller must exhaust its normal, long per-provider retry ladder
before asking for the next route.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal
from urllib.parse import urlsplit, urlunsplit

from backend import config
from backend.provider_catalog import PROVIDER_PROFILE_MAP

RouteSlot = Literal["primary", "backup"]


@dataclass(frozen=True)
class ModelRoute:
    slot: RouteSlot
    provider_id: int | None
    provider_type: str
    provider_name: str
    endpoint: str
    model: str
    api_key: str = field(repr=False)

    @property
    def audit_label(self) -> str:
        role = "primary" if self.slot == "primary" else "backup"
        return f"{self.provider_name} {role} (type={self.provider_type}, model={self.model})"


class ModelRouteExhausted(RuntimeError):
    """Every configured model route failed after its own retry ladder."""


def max_retries_for_route(route: ModelRoute) -> int:
    if route.provider_type == config.AI_PRIMARY_PROVIDER_TYPE:
        return max(0, int(config.AI_PRIMARY_MAX_RETRIES))
    if route.provider_type == config.AI_BACKUP_PROVIDER_TYPE:
        return max(0, int(config.AI_BACKUP_MAX_RETRIES))
    return max(0, int(config.AI_MAX_RETRIES))


def _endpoint_identity(endpoint: str) -> str:
    """Compare a host/root independently of OpenAI chat suffix spelling."""
    raw = endpoint.strip()
    parts = urlsplit(raw)
    if not parts.scheme or not parts.netloc:
        return raw.rstrip("/").casefold()
    path = parts.path.rstrip("/")
    for suffix in ("/v1/chat/completions", "/chat/completions"):
        if path.endswith(suffix):
            path = path.removesuffix(suffix)
            break
    return urlunsplit(
        (parts.scheme.casefold(), parts.netloc.casefold(), path.rstrip("/"), "", "")
    )


def _selected_route(
    *,
    endpoint: str,
    model: str,
    api_key: str,
    provider_id: int | None = None,
    provider_type: str = "selected",
    provider_name: str = "Selected provider",
) -> ModelRoute:
    return ModelRoute(
        slot="primary",
        provider_id=provider_id,
        provider_type=provider_type,
        provider_name=provider_name,
        endpoint=endpoint,
        model=model,
        api_key=api_key,
    )


async def resolve_model_routes(
    *,
    endpoint: str,
    model: str,
    api_key: str,
    allow_failover: bool = True,
) -> tuple[ModelRoute, ...]:
    """Resolve selected provider plus the configured backup, if applicable.

    Failover is enabled only when the selected endpoint matches a saved row of
    the configured primary provider type.  Selecting DeepSeek explicitly, a
    custom provider, or testing an unsaved form therefore never silently uses
    another provider. Global Anthropic endpoint/model overrides also disable
    routing because they would make both slots address the same destination.
    """
    selected = _selected_route(endpoint=endpoint, model=model, api_key=api_key)
    if (
        not allow_failover
        or not config.AI_PROVIDER_FAILOVER_ENABLED
        or not config.AI_PRIMARY_PROVIDER_TYPE
        or not config.AI_BACKUP_PROVIDER_TYPE
        or config.AI_PRIMARY_PROVIDER_TYPE == config.AI_BACKUP_PROVIDER_TYPE
        or config.ANTHROPIC_BASE_URL
        or config.ANTHROPIC_MODEL
    ):
        return (selected,)

    # Avoid a database lookup on arbitrary/custom endpoints and keep unit
    # callers hermetic. The provider row is still the final source of truth.
    primary_profile = PROVIDER_PROFILE_MAP.get(config.AI_PRIMARY_PROVIDER_TYPE)
    if primary_profile is None or _endpoint_identity(endpoint) != _endpoint_identity(
        primary_profile.default_endpoint
    ):
        return (selected,)

    from backend import database

    rows = await database.list_providers_raw()
    primary_row = next(
        (
            row
            for row in rows
            if row["provider_type"] == config.AI_PRIMARY_PROVIDER_TYPE
            and _endpoint_identity(row["endpoint"]) == _endpoint_identity(endpoint)
        ),
        None,
    )
    if primary_row is None:
        return (selected,)

    primary = _selected_route(
        endpoint=endpoint,
        model=model,
        api_key=api_key,
        provider_id=int(primary_row["id"]),
        provider_type=str(primary_row["provider_type"]),
        provider_name=str(primary_row["name"]),
    )
    backup_row = next(
        (
            row
            for row in rows
            if row["provider_type"] == config.AI_BACKUP_PROVIDER_TYPE
            and str(row["endpoint"] or "").strip()
            and str(row["model"] or "").strip()
            and str(row["api_key"] or "").strip()
        ),
        None,
    )
    if backup_row is None:
        return (primary,)

    backup_endpoint = str(backup_row["endpoint"])
    backup_model = str(backup_row["model"])
    if (
        _endpoint_identity(backup_endpoint) == _endpoint_identity(primary.endpoint)
        and backup_model == primary.model
    ):
        return (primary,)

    backup = ModelRoute(
        slot="backup",
        provider_id=int(backup_row["id"]),
        provider_type=str(backup_row["provider_type"]),
        provider_name=str(backup_row["name"]),
        endpoint=backup_endpoint,
        model=backup_model,
        api_key=str(backup_row["api_key"]),
    )
    return (primary, backup)

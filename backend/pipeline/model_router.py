"""Persisted primary/backup routing with a primary API-key pool.

Provider roles and credentials are owned by Admin -> Models and stored in the
providers table. Environment variables are not part of route selection. A
primary pool is round-robin reserved for every model call; a 429 may rotate to
another primary key before the provider route is considered exhausted.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal
from urllib.parse import urlsplit, urlunsplit

from backend import config
from backend.provider_credentials import key_identifier

RouteSlot = Literal["primary", "backup", "standalone"]


@dataclass(frozen=True)
class ModelRoute:
    slot: RouteSlot
    provider_id: int | None
    provider_type: str
    provider_name: str
    endpoint: str
    model: str
    api_key: str = field(repr=False)
    api_keys: tuple[str, ...] = field(default=(), repr=False)
    api_key_index: int = 0

    def __post_init__(self) -> None:
        pool = tuple(self.api_keys) or ((self.api_key,) if self.api_key else ())
        index = self.api_key_index % len(pool) if pool else 0
        object.__setattr__(self, "api_keys", pool)
        object.__setattr__(self, "api_key_index", index)
        object.__setattr__(self, "api_key", pool[index] if pool else "")

    @property
    def api_key_count(self) -> int:
        return len(self.api_keys)

    @property
    def api_key_id(self) -> str:
        return key_identifier(self.api_key) if self.api_key else "key-missing"

    @property
    def audit_label(self) -> str:
        role = {
            "primary": "primary",
            "backup": "backup",
            "standalone": "standalone",
        }[self.slot]
        pool = f", keys={self.api_key_count}" if self.slot == "primary" else ""
        return (
            f"{self.provider_name} {role} "
            f"(type={self.provider_type}, model={self.model}{pool})"
        )

    def next_untried_key(self, tried_key_ids: set[str]) -> "ModelRoute | None":
        """Return the next primary credential that was not tried this cycle."""
        if self.slot != "primary" or self.api_key_count <= 1:
            return None
        for offset in range(1, self.api_key_count + 1):
            candidate = replace(
                self,
                api_key_index=(self.api_key_index + offset) % self.api_key_count,
            )
            if candidate.api_key_id not in tried_key_ids:
                return candidate
        return None


class ModelRouteExhausted(RuntimeError):
    """Every configured provider route failed after its bounded retry ladder."""


class ModelOutputCommittedError(RuntimeError):
    """A provider failed after emitting model/tool content; replay is unsafe."""


def max_retries_for_route(route: ModelRoute) -> int:
    if route.slot == "primary":
        return max(0, int(config.AI_PRIMARY_MAX_RETRIES))
    if route.slot == "backup":
        return max(0, int(config.AI_BACKUP_MAX_RETRIES))
    return max(0, int(config.AI_MAX_RETRIES))


def endpoint_identity(endpoint: str) -> str:
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
        slot="standalone",
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
    """Resolve the selected call against persisted primary/backup roles.

    Only a call whose endpoint matches the configured primary receives the
    primary pool and backup route. Explicitly selecting a backup, standalone,
    or unsaved endpoint remains a single-route call.
    """
    selected = _selected_route(endpoint=endpoint, model=model, api_key=api_key)
    if not allow_failover or config.ANTHROPIC_BASE_URL or config.ANTHROPIC_MODEL:
        return (selected,)

    from backend import database

    rows = await database.list_providers_raw()
    primary_row = next(
        (row for row in rows if str(row["route_role"]) == "primary"),
        None,
    )
    if primary_row is None or endpoint_identity(endpoint) != endpoint_identity(
        str(primary_row["endpoint"])
    ):
        matching_row = next(
            (
                row
                for row in rows
                if endpoint_identity(endpoint) == endpoint_identity(str(row["endpoint"]))
                and str(row["model"]) == model
            ),
            None,
        )
        if matching_row is None:
            return (selected,)
        return (
            _selected_route(
                endpoint=endpoint,
                model=model,
                api_key=api_key,
                provider_id=int(matching_row["id"]),
                provider_type=str(matching_row["provider_type"]),
                provider_name=str(matching_row["name"]),
            ),
        )

    api_keys, api_key_index = await database.reserve_provider_api_key(
        int(primary_row["id"])
    )
    if not api_keys:
        return (selected,)
    primary = ModelRoute(
        slot="primary",
        provider_id=int(primary_row["id"]),
        provider_type=str(primary_row["provider_type"]),
        provider_name=str(primary_row["name"]),
        endpoint=endpoint,
        model=model,
        api_key=api_keys[api_key_index],
        api_keys=tuple(api_keys),
        api_key_index=api_key_index,
    )

    backup_row = next(
        (
            row
            for row in rows
            if str(row["route_role"]) == "backup"
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
        endpoint_identity(backup_endpoint) == endpoint_identity(primary.endpoint)
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

from fastapi import APIRouter, HTTPException

from backend import settings_store
from backend.models import (
    SettingsResetRequest,
    SettingsSchemaResponse,
    SettingsValuesUpdate,
)

router = APIRouter(prefix="/api/settings", tags=["settings"])


def _schema(restart_required: list[str] | None = None) -> SettingsSchemaResponse:
    return SettingsSchemaResponse(
        groups=settings_store.schema(),
        restart_required=restart_required or [],
    )


@router.get("/schema", response_model=SettingsSchemaResponse)
async def get_settings_schema():
    """Every runtime setting, grouped, with its current and default value.

    The Admin console renders this generically, so adding a setting is a
    backend-only change (see backend/settings_store.py).
    """
    return _schema()


@router.put("/values", response_model=SettingsSchemaResponse)
async def update_settings_values(body: SettingsValuesUpdate):
    """Persist and immediately apply the submitted settings."""
    try:
        restart_required = settings_store.update(body.values)
    except settings_store.SettingsError as exc:
        raise HTTPException(400, str(exc))
    return _schema(restart_required)


@router.post("/values/reset", response_model=SettingsSchemaResponse)
async def reset_settings_values(body: SettingsResetRequest):
    """Drop the saved overrides for these keys, restoring the .env defaults."""
    try:
        restart_required = settings_store.reset(body.keys)
    except settings_store.SettingsError as exc:
        raise HTTPException(400, str(exc))
    return _schema(restart_required)

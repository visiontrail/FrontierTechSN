from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from backend import config, database
from backend.account_ops.schedule import local_event_date
from backend.account_ops.worker import get_worker_status
from backend.models import (
    AccountAutomationCreate,
    AccountAutomationListResponse,
    AccountAutomationResponse,
    AccountAutomationUpdate,
    AccountOpsStatusResponse,
    AccountRunListResponse,
    AccountRunResponse,
)

router = APIRouter(prefix="/api/account-operations", tags=["account-operations"])


@router.get("/status", response_model=AccountOpsStatusResponse)
async def get_status() -> AccountOpsStatusResponse:
    status = get_worker_status()
    return AccountOpsStatusResponse(**status)


@router.get("/automations", response_model=AccountAutomationListResponse)
async def list_automations() -> AccountAutomationListResponse:
    return AccountAutomationListResponse(
        automations=await database.list_account_automations()
    )


@router.post("/automations", response_model=AccountAutomationResponse)
async def create_automation(
    body: AccountAutomationCreate,
) -> AccountAutomationResponse:
    return await database.create_account_automation(body)


@router.put("/automations/{automation_id}", response_model=AccountAutomationResponse)
async def update_automation(
    automation_id: str, body: AccountAutomationUpdate
) -> AccountAutomationResponse:
    updated = await database.update_account_automation(
        automation_id, body.model_dump(exclude_unset=True)
    )
    if updated is None:
        raise HTTPException(404, "Account automation not found")
    return updated


@router.post(
    "/automations/{automation_id}/pause",
    response_model=AccountAutomationResponse,
)
async def pause_automation(automation_id: str) -> AccountAutomationResponse:
    updated = await database.update_account_automation(
        automation_id, {"enabled": False}
    )
    if updated is None:
        raise HTTPException(404, "Account automation not found")
    return updated


@router.post(
    "/automations/{automation_id}/resume",
    response_model=AccountAutomationResponse,
)
async def resume_automation(automation_id: str) -> AccountAutomationResponse:
    updated = await database.update_account_automation(
        automation_id, {"enabled": True}
    )
    if updated is None:
        raise HTTPException(404, "Account automation not found")
    return updated


@router.post(
    "/automations/{automation_id}/run",
    response_model=AccountRunResponse,
    status_code=202,
)
async def run_automation_now(automation_id: str) -> AccountRunResponse:
    automation = await database.get_account_automation(automation_id)
    if automation is None:
        raise HTTPException(404, "Account automation not found")
    return await database.create_account_run(
        automation,
        trigger="manual",
        event_date=local_event_date(automation.timezone),
    )


@router.get("/runs", response_model=AccountRunListResponse)
async def list_runs(
    limit: int = Query(default=200, ge=1, le=500),
) -> AccountRunListResponse:
    return AccountRunListResponse(runs=await database.list_account_runs(limit))


@router.get("/runs/{run_id}", response_model=AccountRunResponse)
async def get_run(run_id: str) -> AccountRunResponse:
    run = await database.get_account_run(run_id)
    if run is None:
        raise HTTPException(404, "Account-operation run not found")
    return run


@router.get("/runs/{run_id}/image")
async def get_run_image(run_id: str) -> FileResponse:
    run = await database.get_account_run(run_id)
    if run is None or not run.image_path:
        raise HTTPException(404, "Run image not available")
    path = Path(run.image_path).resolve()
    root = (config.OUTPUTS_DIR / "account-operations" / run_id).resolve()
    if root not in path.parents or not path.is_file():
        raise HTTPException(404, "Run image file missing")
    media_type = {
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(path.suffix.lower(), "image/jpeg")
    return FileResponse(path, media_type=media_type, filename=path.name)


@router.get("/capabilities")
async def capabilities() -> dict[str, object]:
    return {
        "opencode": {
            "available": shutil.which(config.OPENCODE_BIN) is not None,
            "binary": config.OPENCODE_BIN,
        },
        "opencli": {
            "available": Path(config.OPENCLI_BIN).is_file(),
            "binary": str(config.OPENCLI_BIN),
        },
        "supported_features": ["today_in_history", "x_engagement"],
        "supported_executors": ["opencode", "claude_sdk", "pipeline"],
        "supported_platforms": ["x"],
    }

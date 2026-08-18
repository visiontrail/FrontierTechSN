from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException, Query

from backend.daily_news.scheduler import (
    create_daily_task,
    load_settings,
    response,
    save_settings,
)
from backend.daily_news.source_catalog import load_source_catalog
from backend.models import DailyAutomationResponse, DailyAutomationSettings, TaskResponse

router = APIRouter(prefix="/api/daily-news", tags=["daily-news"])


@router.get("", response_model=DailyAutomationResponse)
async def get_daily_automation():
    return response()


@router.put("", response_model=DailyAutomationResponse)
async def update_daily_automation(body: DailyAutomationSettings):
    save_settings(body)
    return response()


@router.get("/sources")
async def get_daily_sources():
    return {"sources": [source.as_dict() for source in load_source_catalog()]}


@router.post("/run-now", response_model=TaskResponse)
async def run_daily_now(
    test_mode: bool = Query(default=False),
    duration_minutes: int | None = Query(default=None, ge=1, le=30),
):
    settings = load_settings()
    try:
        edition_date = datetime.now(ZoneInfo(settings.timezone)).date()
        return await create_daily_task(
            settings,
            edition_date=edition_date,
            trigger="manual",
            test_mode=test_mode,
            duration_override=duration_minutes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

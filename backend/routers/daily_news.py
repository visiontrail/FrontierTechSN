from __future__ import annotations

import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from backend.daily_news.scheduler import (
    create_daily_task,
    load_settings,
    response,
    save_settings,
)
from backend.daily_news.source_catalog import load_source_catalog
from backend.daily_news.source_check import check_sources
from backend.models import DailyAutomationResponse, DailyAutomationSettings, TaskResponse
from backend.pipeline.music import list_program_music_tracks, resolve_program_music_track
from backend.pipeline.outros import list_outro_presets, resolve_outro_preset
from backend.publishing import automatic_publication_configuration_errors

router = APIRouter(prefix="/api/daily-news", tags=["daily-news"])
_source_check_lock = asyncio.Lock()


@router.get("", response_model=DailyAutomationResponse)
async def get_daily_automation():
    return response()


@router.put("", response_model=DailyAutomationResponse)
async def update_daily_automation(body: DailyAutomationSettings):
    try:
        resolve_outro_preset(body.outro_style)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if body.background_music_provider == "local_library":
        try:
            resolve_program_music_track(body.background_music_track_id)
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    if body.auto_publish:
        errors = automatic_publication_configuration_errors(body.publish_targets)
        if errors:
            raise HTTPException(
                status_code=409,
                detail="Automatic publication is not ready: " + "; ".join(errors),
            )
    save_settings(body)
    return response()


@router.get("/sources")
async def get_daily_sources():
    return {"sources": [source.as_dict() for source in load_source_catalog()]}


@router.post("/sources/check")
async def check_daily_sources():
    if _source_check_lock.locked():
        raise HTTPException(status_code=409, detail="A source check is already running")
    async with _source_check_lock:
        return await check_sources()


@router.get("/music-library")
async def get_program_music_library():
    try:
        return {"tracks": list_program_music_tracks()}
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/outro-library")
async def get_outro_library():
    return {"presets": list_outro_presets()}


@router.get("/outro-library/{style}/video")
async def get_outro_background_video(style: str):
    try:
        preset = resolve_outro_preset(style)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(
        preset["background_path"],
        media_type="video/mp4",
        filename=f"{preset['id']}-gemini-background.mp4",
        headers={"Cache-Control": "private, max-age=3600"},
    )


@router.get("/music-library/{track_id}/audio")
async def get_program_music_audio(track_id: str):
    try:
        track = resolve_program_music_track(track_id)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(
        track["path"],
        media_type="audio/mpeg",
        filename=f"{track['id']}.mp3",
        headers={"Cache-Control": "private, max-age=3600"},
    )


@router.post("/run-now", response_model=TaskResponse)
async def run_daily_now(
    test_mode: bool = Query(default=False),
):
    settings = load_settings()
    try:
        edition_date = datetime.now(ZoneInfo(settings.timezone)).date()
        return await create_daily_task(
            settings,
            edition_date=edition_date,
            trigger="manual",
            test_mode=test_mode,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

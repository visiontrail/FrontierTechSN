from fastapi import APIRouter, HTTPException

from backend import config
from backend import database as db
from backend.models import (
    ContentPlanItemCreate,
    ContentPlanItemListResponse,
    ContentPlanItemResponse,
    ContentPlanItemUpdate,
    ContentSeriesCreate,
    ContentSeriesListResponse,
    ContentSeriesResponse,
    ContentSeriesUpdate,
    ManualPublicationRecord,
)


router = APIRouter(prefix="/api/content-planning", tags=["content-planning"])


@router.get("/status")
async def get_planning_status():
    return {
        "auto_publish_enabled": bool(config.VIDEO_AUTO_PUBLISH_ENABLED),
        "publication_mode": "automatic" if config.VIDEO_AUTO_PUBLISH_ENABLED else "human_review",
        "scheduler": "video_task_queue",
    }


@router.get("/series", response_model=ContentSeriesListResponse)
async def list_series():
    return ContentSeriesListResponse(series=await db.list_content_series())


@router.post("/series", response_model=ContentSeriesResponse)
async def create_series(body: ContentSeriesCreate):
    return await db.create_content_series(body)


@router.patch("/series/{series_id}", response_model=ContentSeriesResponse)
async def update_series(series_id: str, body: ContentSeriesUpdate):
    result = await db.update_content_series(
        series_id, **body.model_dump(exclude_unset=True)
    )
    if result is None:
        raise HTTPException(404, "Series not found")
    return result


@router.get("/items", response_model=ContentPlanItemListResponse)
async def list_items():
    return ContentPlanItemListResponse(items=await db.list_content_plan_items())


@router.post("/items", response_model=ContentPlanItemResponse)
async def create_item(body: ContentPlanItemCreate):
    try:
        return await db.create_content_plan_item(body)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/items/{item_id}", response_model=ContentPlanItemResponse)
async def get_item(item_id: str):
    result = await db.get_content_plan_item(item_id)
    if result is None:
        raise HTTPException(404, "Plan item not found")
    return result


@router.patch("/items/{item_id}", response_model=ContentPlanItemResponse)
async def update_item(item_id: str, body: ContentPlanItemUpdate):
    current = await db.get_content_plan_item(item_id)
    if current is None:
        raise HTTPException(404, "Plan item not found")
    merged = {
        "series_id": current.series_id,
        "title": current.title,
        "brief": current.brief,
        "source_type": current.source_type,
        "source_url": current.source_url,
        "episode_number": current.episode_number,
        "generation_at": current.generation_at,
        "publish_at": current.publish_at,
        "platform": current.platform,
        "auto_publish_requested": current.auto_publish_requested,
        "task_config": current.task_config,
    }
    merged.update(body.model_dump(exclude_unset=True))
    try:
        normalized = ContentPlanItemCreate(**merged)
        result = await db.update_content_plan_item(item_id, normalized)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    if result is None:
        raise HTTPException(404, "Plan item not found")
    return result


@router.post("/items/{item_id}/approve", response_model=ContentPlanItemResponse)
async def approve_item(item_id: str):
    try:
        result = await db.approve_content_plan_item(item_id)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    if result is None:
        raise HTTPException(404, "Plan item not found")
    return result


@router.post("/items/{item_id}/publish", response_model=ContentPlanItemResponse)
async def record_publication(item_id: str, body: ManualPublicationRecord):
    try:
        result = await db.record_manual_publication(item_id, body.publication_url)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    if result is None:
        raise HTTPException(404, "Plan item not found")
    return result


@router.delete("/items/{item_id}")
async def delete_item(item_id: str):
    try:
        deleted = await db.delete_content_plan_item(item_id)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    if not deleted:
        raise HTTPException(404, "Plan item not found")
    return {"ok": True}

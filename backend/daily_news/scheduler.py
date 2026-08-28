from __future__ import annotations

import asyncio
import json
import logging
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from backend import config, database
from backend.models import (
    DailyAutomationResponse,
    DailyAutomationSettings,
    ScriptFormat,
    TaskConfig,
    TaskResponse,
)

logger = logging.getLogger(__name__)
_scheduler_task: asyncio.Task | None = None


def _settings_path() -> Path:
    return config.PROJECT_ROOT / "data" / "daily_automation.json"


def _state_path() -> Path:
    return config.PROJECT_ROOT / "data" / "daily_automation_state.json"


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def load_settings() -> DailyAutomationSettings:
    payload = _read_json(_settings_path())

    # Recipes written by older releases can reference a model or voice that is
    # no longer in the live catalog. Heal only that catalog drift on read so the
    # desk remains operable; all other fields still go through strict Pydantic
    # validation below. New PUT requests remain strict as well.
    default_model = (
        config.TTS_DEFAULT_MODEL
        if config.TTS_DEFAULT_MODEL in config.TTS_MODELS
        else next(iter(config.TTS_MODELS))
    )
    requested_model = payload.get("tts_model", default_model)
    migrated = False
    if requested_model not in config.TTS_MODELS:
        requested_model = default_model
        payload["tts_model"] = requested_model
        migrated = True

    voices = config.voices_for_model(requested_model)
    default_voice = (
        config.TTS_DEFAULT_VOICE_1
        if config.TTS_DEFAULT_VOICE_1 in voices
        else next(iter(voices))
    )
    if payload.get("voice", config.TTS_DEFAULT_VOICE_1) not in voices:
        payload["voice"] = default_voice
        migrated = True

    settings = DailyAutomationSettings(**payload)
    if migrated:
        _write_json(_settings_path(), settings.model_dump(mode="json"))
    return settings


def save_settings(settings: DailyAutomationSettings) -> DailyAutomationSettings:
    _write_json(_settings_path(), settings.model_dump(mode="json"))
    return settings


def next_run_at(settings: DailyAutomationSettings, now: datetime | None = None) -> datetime:
    zone = ZoneInfo(settings.timezone)
    local_now = (now or datetime.now(timezone.utc)).astimezone(zone)
    hour, minute = (int(part) for part in settings.generation_time.split(":"))
    candidate = datetime.combine(local_now.date(), time(hour, minute), zone)
    if candidate <= local_now:
        candidate += timedelta(days=1)
    return candidate.astimezone(timezone.utc)


def response() -> DailyAutomationResponse:
    settings = load_settings()
    state = _read_json(_state_path())
    return DailyAutomationResponse(
        settings=settings,
        next_run_at=next_run_at(settings).isoformat() if settings.enabled else None,
        last_run_at=state.get("last_run_at"),
        last_run_date=state.get("last_run_date"),
        last_task_id=state.get("last_task_id"),
    )


async def create_daily_task(
    settings: DailyAutomationSettings,
    *,
    edition_date: date,
    trigger: str,
    test_mode: bool = False,
    duration_override: int | None = None,
) -> TaskResponse:
    task_config = TaskConfig(
        target_duration_minutes=duration_override or settings.target_duration_minutes,
        script_format=ScriptFormat.MONOLOGUE,
        voice_1=settings.voice,
        tts_model=settings.tts_model,
        video_template="swiss",
        video_orientation="landscape",
        opening_style="paper_collage",
        footage_enabled=settings.public_footage_enabled,
        footage_clip_count=settings.footage_clip_count,
        collage_broll_enabled=True,
        collage_broll_count=settings.collage_broll_count,
        news_images_enabled=True,
        news_image_count=settings.news_image_count,
        thumbnail_enabled=True,
        auto_render=True,
        news_language=settings.language,
        news_max_stories=settings.max_stories,
        news_window_hours=settings.source_window_hours,
        news_timezone=settings.timezone,
        background_music_enabled=True,
        background_music_provider=settings.background_music_provider,
        background_music_track_id=settings.background_music_track_id,
        program_music_pacing_enabled=True,
        program_music_intro_seconds=2.0,
        program_music_opening_gap_seconds=3.0,
        program_music_story_gap_seconds=1.5,
        auto_publish=settings.auto_publish and not test_mode,
        publish_targets=settings.publish_targets,
        publish_visibility=settings.publish_visibility,
        publish_test_mode=test_mode,
    )
    payload = json.dumps(
        {
            "edition_date": edition_date.isoformat(),
            "trigger": trigger,
            "test_mode": test_mode,
        },
        ensure_ascii=False,
    )
    task = await database.create_task(
        "news_daily",
        payload,
        task_config,
        source_title=f"Frontier Tech Daily — {edition_date.isoformat()}",
        origin_type="daily_news",
        origin_id=edition_date.isoformat(),
        origin_label=f"Daily automation · {edition_date.isoformat()}",
    )
    now = datetime.now(timezone.utc).isoformat()
    state = _read_json(_state_path())
    if test_mode:
        # A one-minute validation run is not today's published edition. Keeping
        # its receipt separate prevents the test button from suppressing the
        # real scheduled run through `_is_due`'s once-per-edition guard.
        state.update(
            {
                "last_test_run_at": now,
                "last_test_task_id": task.id,
                "last_test_trigger": trigger,
            }
        )
    else:
        state.update(
            {
                "last_run_at": now,
                "last_run_date": edition_date.isoformat(),
                "last_task_id": task.id,
                "last_trigger": trigger,
            }
        )
    _write_json(_state_path(), state)
    return task


def _is_due(settings: DailyAutomationSettings, now: datetime, state: dict) -> tuple[bool, date]:
    zone = ZoneInfo(settings.timezone)
    local_now = now.astimezone(zone)
    hour, minute = (int(part) for part in settings.generation_time.split(":"))
    scheduled = datetime.combine(local_now.date(), time(hour, minute), zone)
    already_ran = state.get("last_run_date") == local_now.date().isoformat()
    if already_ran:
        return False, local_now.date()
    if local_now >= scheduled:
        return True, local_now.date()
    return False, local_now.date()


async def _scheduler_loop() -> None:
    logger.info("Daily frontier-tech scheduler started")
    while True:
        try:
            settings = load_settings()
            if settings.enabled:
                now = datetime.now(timezone.utc)
                state = _read_json(_state_path())
                due, edition_date = _is_due(settings, now, state)
                if due and (settings.catch_up_after_restart or now - datetime.combine(
                    edition_date,
                    time(*(int(part) for part in settings.generation_time.split(":"))),
                    ZoneInfo(settings.timezone),
                ).astimezone(timezone.utc) < timedelta(hours=2)):
                    task = await create_daily_task(
                        settings,
                        edition_date=edition_date,
                        trigger="schedule",
                    )
                    logger.info("Daily scheduler queued task %s", task.id)
            await asyncio.sleep(20)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Daily scheduler iteration failed")
            await asyncio.sleep(30)


def start_daily_scheduler() -> None:
    global _scheduler_task
    if _scheduler_task is None or _scheduler_task.done():
        _scheduler_task = asyncio.create_task(_scheduler_loop(), name="daily-frontier-tech-scheduler")


async def stop_daily_scheduler() -> None:
    global _scheduler_task
    if _scheduler_task is None:
        return
    _scheduler_task.cancel()
    try:
        await _scheduler_task
    except asyncio.CancelledError:
        pass
    finally:
        _scheduler_task = None

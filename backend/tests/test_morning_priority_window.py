"""Hourly admission and dispatch respect the Singapore production window."""

import asyncio
import json
from datetime import date, datetime, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend import config, database, explainers
from backend.daily_news import priority, scheduler
from backend.models import DailyAutomationSettings, TaskConfig, TaskStatus
from backend.routers import daily_news


class FrozenDateTime(datetime):
    instant = datetime(2026, 10, 9, 4, tzinfo=timezone.utc)

    @classmethod
    def now(cls, tz=None):
        return cls.instant.astimezone(tz) if tz else cls.instant.replace(tzinfo=None)


@pytest.fixture
def window_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "tasks.db")
    monkeypatch.setattr(config, "OUTPUTS_DIR", tmp_path / "outputs")
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(explainers, "_jobs", {})
    monkeypatch.setattr(scheduler, "_state_path", lambda: tmp_path / "state.json")
    monkeypatch.setattr(scheduler, "_admission_lock", asyncio.Lock())
    monkeypatch.setattr(priority, "datetime", FrozenDateTime)
    asyncio.run(database.init_db())

    def freeze(utc_instant):
        FrozenDateTime.instant = datetime.fromisoformat(utc_instant)
        return FrozenDateTime.instant

    return freeze


def reserve_explainer(state="queued"):
    project = explainers.create_project(explainers.ProjectInput(
        title="Retained Explainer",
        research="The production must preserve its original research and completed scenes.",
        tts_model="orpheus-en",
        voice="tara",
    ))
    explainers.start(project["id"], "full")
    if state != "queued":
        explainers.patch_project(project["id"], {
            "status": "writing_script" if state == "running" else "interrupted",
            "queue": None,
            "production": {"operation": "full", "started_at": "2026-10-09T00:00:00Z"},
            "cancellation_reason": None if state == "running" else "service_interruption",
        })
    return project["id"]


async def daily_task(trigger="hourly_check", *, status=TaskStatus.QUEUED, output_dir=None):
    task = await database.create_task(
        "news_daily",
        json.dumps({"edition_date": "2026-10-09", "trigger": trigger, "test_mode": False}),
        TaskConfig(),
        origin_type="daily_news",
        origin_id="2026-10-09",
    )
    await database.update_task(task.id, status=status.value, output_dir=output_dir)
    return await database.get_task(task.id)


@pytest.mark.parametrize("utc_instant,allowed,reason", [
    ("2026-10-08T16:00:00+00:00", False, "before_noon"),
    ("2026-10-09T03:59:59+00:00", False, "before_noon"),
    ("2026-10-09T04:00:00+00:00", True, "created"),
    ("2026-10-09T05:00:00+00:00", True, "created"),
    ("2026-10-09T06:00:00+00:00", True, "created"),
    ("2026-10-09T06:00:01+00:00", False, "insufficient_time_before_deadline"),
    ("2026-10-09T15:59:59+00:00", False, "insufficient_time_before_deadline"),
])
@pytest.mark.parametrize("explainer_state", ["queued", "running", "reserved"])
def test_hourly_admission_obeys_singapore_window(
    window_runtime, utc_instant, allowed, reason, explainer_state
):
    instant = window_runtime(utc_instant)
    project_id = reserve_explainer(explainer_state)
    result = asyncio.run(scheduler.ensure_daily_task(
        DailyAutomationSettings(enabled=False, auto_publish=False),
        edition_date=date(2026, 10, 9), trigger="hourly_check", now=instant,
    ))
    assert result.created is allowed
    assert result.reason == reason
    assert (result.task is not None) is allowed
    assert len(asyncio.run(database.list_tasks())) == int(allowed)
    assert explainers.read_project(project_id).get("priority_pause") is None


@pytest.mark.parametrize("recipe_timezone", ["America/Los_Angeles", "Asia/Tokyo"])
def test_recipe_timezone_does_not_move_singapore_preemption_window(
    window_runtime, recipe_timezone
):
    instant = window_runtime("2026-10-09T03:59:59+00:00")
    reserve_explainer()
    result = asyncio.run(scheduler.ensure_daily_task(
        DailyAutomationSettings(timezone=recipe_timezone, auto_publish=False),
        edition_date=date(2026, 10, 9), trigger="hourly_check", now=instant,
    ))
    assert result.task is None
    assert result.reason == "before_noon"


@pytest.mark.parametrize("utc_instant", [
    "2026-10-08T16:00:00+00:00", "2026-10-09T07:00:00+00:00",
])
def test_idle_hourly_can_start_without_interrupting_explainer(window_runtime, utc_instant):
    instant = window_runtime(utc_instant)

    async def scenario():
        result = await scheduler.ensure_daily_task(
            DailyAutomationSettings(auto_publish=False),
            edition_date=date(2026, 10, 9), trigger="hourly_check", now=instant,
        )
        assert result.created
        assert (await database.get_next_queued_task()).id == result.task.id
        assert await database.get_next_priority_morning_task() is None
        assert await database.get_unfinished_priority_morning_task() is None

    asyncio.run(scenario())


@pytest.mark.parametrize("utc_instant,allowed", [
    ("2026-10-09T03:59:59+00:00", False),
    ("2026-10-09T04:00:00+00:00", True),
    ("2026-10-09T06:00:00+00:00", True),
    ("2026-10-09T06:00:01+00:00", False),
    ("2026-10-09T16:00:00+00:00", False),
])
def test_waiting_hourly_priority_is_rechecked_at_dispatch(
    window_runtime, utc_instant, allowed
):
    window_runtime(utc_instant)

    async def scenario():
        task = await daily_task()
        for lookup in (
            database.get_next_priority_morning_task,
            database.get_unfinished_priority_morning_task,
        ):
            selected = await lookup()
            assert (selected is not None) is allowed
            if selected:
                assert selected.id == task.id

    asyncio.run(scenario())


@pytest.mark.parametrize("trigger", ["schedule", "manual"])
def test_blocked_oldest_hourly_does_not_hide_authorized_morning(window_runtime, trigger):
    window_runtime("2026-10-09T07:00:00+00:00")

    async def scenario():
        blocked = await daily_task()
        authorized = await daily_task(trigger)
        await database.update_task(blocked.id, created_at="2026-10-09T00:00:00Z")
        await database.update_task(authorized.id, created_at="2026-10-09T01:00:00Z")
        assert (await database.get_next_priority_morning_task()).id == authorized.id
        assert (await database.get_unfinished_priority_morning_task()).id == authorized.id

    asyncio.run(scenario())


@pytest.mark.parametrize("status", [TaskStatus.QUEUED, TaskStatus.AWAITING_REVIEW])
def test_started_hourly_retains_priority_for_retry_and_review_after_window(
    window_runtime, status
):
    window_runtime("2026-10-09T13:00:00+00:00")

    async def scenario():
        task = await daily_task(status=status, output_dir=str(config.OUTPUTS_DIR / "started"))
        assert (await database.get_unfinished_priority_morning_task()).id == task.id
        queued = await database.get_next_priority_morning_task()
        if status == TaskStatus.QUEUED:
            assert queued.id == task.id
        else:
            assert queued is None

    asyncio.run(scenario())


@pytest.mark.parametrize("utc_instant", [
    "2026-10-08T16:00:00+00:00", "2026-10-09T07:00:00+00:00",
])
def test_schedule_bypasses_hourly_preemption_window(window_runtime, utc_instant):
    instant = window_runtime(utc_instant)
    reserve_explainer()

    async def scenario():
        result = await scheduler.ensure_daily_task(
            DailyAutomationSettings(auto_publish=False),
            edition_date=date(2026, 10, 9), trigger="schedule", now=instant,
        )
        assert result.created
        assert (await database.get_next_priority_morning_task()).id == result.task.id

    asyncio.run(scenario())


def test_schedule_promotes_existing_waiting_hourly_after_window(window_runtime):
    instant = window_runtime("2026-10-09T07:00:00+00:00")
    reserve_explainer()

    async def scenario():
        existing = await daily_task()
        assert await database.get_next_priority_morning_task() is None
        result = await scheduler.ensure_daily_task(
            DailyAutomationSettings(auto_publish=False),
            edition_date=date(2026, 10, 9), trigger="schedule", now=instant,
        )
        assert result.created is False
        assert result.task.id == existing.id
        assert len(await database.list_tasks()) == 1
        assert json.loads((await database.get_task(existing.id)).source_url)["trigger"] == "schedule"
        assert (await database.get_next_priority_morning_task()).id == existing.id

    asyncio.run(scenario())


def test_existing_started_morning_is_reused_outside_window(window_runtime):
    instant = window_runtime("2026-10-09T07:00:00+00:00")
    reserve_explainer("reserved")

    async def scenario():
        existing = await daily_task(status=TaskStatus.RESEARCHING)
        result = await scheduler.ensure_daily_task(
            DailyAutomationSettings(auto_publish=False),
            edition_date=date(2026, 10, 9), trigger="hourly_check", now=instant,
        )
        assert result.created is False
        assert result.reason == "edition_exists"
        assert result.task.id == existing.id
        assert (await database.get_unfinished_priority_morning_task()).id == existing.id

    asyncio.run(scenario())


@pytest.mark.parametrize("utc_instant,reason", [
    ("2026-10-09T03:59:59+00:00", "before_noon"),
    ("2026-10-09T06:00:01+00:00", "insufficient_time_before_deadline"),
])
def test_hourly_endpoint_reports_deferred_admission_without_creating_task(
    window_runtime, monkeypatch, utc_instant, reason
):
    window_runtime(utc_instant)
    reserve_explainer()
    monkeypatch.setattr(daily_news, "datetime", FrozenDateTime)
    monkeypatch.setattr(daily_news, "load_settings", lambda: DailyAutomationSettings(
        enabled=False, timezone="Asia/Singapore", auto_publish=False,
    ))
    app = FastAPI()
    app.include_router(daily_news.router)
    response = TestClient(app).post("/api/daily-news/ensure-today")
    assert response.status_code == 200
    assert response.json() == {
        "task": None, "created": False, "reason": reason,
        "edition_date": "2026-10-09", "trigger": "hourly_check",
    }
    assert not asyncio.run(database.list_tasks())

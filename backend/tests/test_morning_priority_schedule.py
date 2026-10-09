import asyncio
import json
from datetime import date, datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from backend import config, database
from backend.daily_news import scheduler
from backend.models import DailyAutomationSettings, TaskConfig, TaskStatus
from backend.routers import daily_news


@pytest.fixture
def isolated_desk(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "tasks.db")
    monkeypatch.setattr(scheduler, "_state_path", lambda: tmp_path / "state.json")
    monkeypatch.setattr(scheduler, "_admission_lock", asyncio.Lock())
    asyncio.run(database.init_db())
    return tmp_path


async def _daily_task(edition, *, origin_id=None, test_mode=False, status=TaskStatus.QUEUED):
    task = await database.create_task(
        "news_daily",
        json.dumps({"edition_date": edition, "test_mode": test_mode}),
        TaskConfig(publish_test_mode=test_mode),
        origin_type="daily_news",
        origin_id=origin_id,
    )
    if status != TaskStatus.QUEUED:
        await database.update_task(task.id, status=status.value)
        task = await database.get_task(task.id)
    return task


def test_schedule_and_hourly_admission_share_one_daily_edition(isolated_desk):
    async def scenario():
        settings = DailyAutomationSettings(auto_publish=False)
        scheduled, hourly = await asyncio.gather(
            scheduler.ensure_daily_task(
                settings, edition_date=date(2026, 10, 9), trigger="schedule"
            ),
            scheduler.ensure_daily_task(
                settings, edition_date=date(2026, 10, 9), trigger="hourly_check"
            ),
        )
        assert scheduled.task.id == hourly.task.id
        assert sum(result.created for result in (scheduled, hourly)) == 1
        assert len(await database.list_tasks()) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("with_origin", [False, True])
@pytest.mark.parametrize("status", [TaskStatus.QUEUED, TaskStatus.COMPLETE, TaskStatus.FAILED])
def test_database_edition_prevents_duplicates_after_receipt_loss(
    isolated_desk, with_origin, status
):
    async def scenario():
        existing = await _daily_task(
            "2026-10-09", origin_id="2026-10-09" if with_origin else None, status=status
        )
        if with_origin:
            # The stored origin alone is sufficient for older/changed payloads.
            await database.update_task(existing.id, source_url="{}")
        result = await scheduler.ensure_daily_task(
            DailyAutomationSettings(auto_publish=False),
            edition_date=date(2026, 10, 9),
            trigger="hourly_check",
        )
        assert result.task.id == existing.id
        assert result.created is False
        assert result.reason == "edition_exists"
        assert len(await database.list_tasks()) == 1

    asyncio.run(scenario())


def test_stale_receipt_does_not_hide_missing_daily_task(isolated_desk):
    (isolated_desk / "state.json").write_text(json.dumps({
        "last_run_date": "2026-10-09", "last_task_id": "deleted-task"
    }))
    result = asyncio.run(scheduler.ensure_daily_task(
        DailyAutomationSettings(auto_publish=False),
        edition_date=date(2026, 10, 9),
        trigger="hourly_check",
    ))
    assert result.created is True
    assert result.task.origin_id == "2026-10-09"


def test_receipt_failure_after_database_commit_does_not_create_second_edition(
    isolated_desk, monkeypatch
):
    write_json = scheduler._write_json

    def fail_receipt(*_):
        raise OSError("Simulated receipt write failure")

    monkeypatch.setattr(scheduler, "_write_json", fail_receipt)
    with pytest.raises(OSError, match="receipt write failure"):
        asyncio.run(scheduler.ensure_daily_task(
            DailyAutomationSettings(auto_publish=False),
            edition_date=date(2026, 10, 9),
            trigger="schedule",
        ))
    monkeypatch.setattr(scheduler, "_write_json", write_json)
    result = asyncio.run(scheduler.ensure_daily_task(
        DailyAutomationSettings(auto_publish=False),
        edition_date=date(2026, 10, 9),
        trigger="hourly_check",
    ))
    assert result.created is False
    assert result.reason == "edition_exists"
    assert len(asyncio.run(database.list_tasks())) == 1


def test_validation_run_does_not_fulfill_today_or_block_real_morning(isolated_desk):
    async def scenario():
        test_task = await _daily_task("2026-10-09", test_mode=True)
        result = await scheduler.ensure_daily_task(
            DailyAutomationSettings(auto_publish=False),
            edition_date=date(2026, 10, 9),
            trigger="schedule",
        )
        assert result.created is True
        assert result.task.id != test_task.id
        assert result.task.config.publish_test_mode is False
        priority = await database.get_next_priority_morning_task()
        assert priority.id == result.task.id

    asyncio.run(scenario())


def test_legacy_payload_validation_run_does_not_fulfill_today(isolated_desk):
    async def scenario():
        test_task = await _daily_task("2026-10-09", test_mode=True)
        await database.update_task(test_task.id, config_json=TaskConfig().model_dump_json())
        result = await scheduler.ensure_daily_task(
            DailyAutomationSettings(auto_publish=False),
            edition_date=date(2026, 10, 9),
            trigger="hourly_check",
        )
        assert result.created is True
        assert result.task.id != test_task.id
        assert (await database.get_next_priority_morning_task()).id == result.task.id

    asyncio.run(scenario())


@pytest.mark.parametrize("status", [
    TaskStatus.QUEUED, TaskStatus.COMPOSING, TaskStatus.AWAITING_REVIEW,
])
def test_unfinished_morning_is_reused_before_admitting_another_day(isolated_desk, status):
    async def scenario():
        existing = await _daily_task("2026-10-08", status=status)
        result = await scheduler.ensure_daily_task(
            DailyAutomationSettings(auto_publish=False),
            edition_date=date(2026, 10, 9),
            trigger="hourly_check",
        )
        assert result.created is False
        assert result.reason == "morning_in_progress"
        assert result.task.id == existing.id
        assert len(await database.list_tasks()) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("status", list(TaskStatus))
def test_unfinished_priority_lookup_releases_only_terminal_morning(isolated_desk, status):
    async def scenario():
        existing = await _daily_task("2026-10-09", status=status)
        result = await database.get_unfinished_priority_morning_task()
        if status in {TaskStatus.COMPLETE, TaskStatus.FAILED}:
            assert result is None
        else:
            assert result.id == existing.id

    asyncio.run(scenario())


def test_unfinished_priority_lookup_keeps_same_due_and_test_filters(isolated_desk):
    async def scenario():
        await _daily_task("2026-10-09", test_mode=True, status=TaskStatus.AWAITING_REVIEW)
        await database.create_task("news_daily", "manual news", TaskConfig())
        future = await _daily_task("2026-10-10", status=TaskStatus.COMPOSING)
        await database.update_task(
            future.id,
            scheduled_at=(datetime.now(timezone.utc) + timedelta(days=1)).isoformat(),
        )
        assert await database.get_unfinished_priority_morning_task() is None
        oldest = await _daily_task("2026-10-08", status=TaskStatus.AWAITING_REVIEW)
        newer = await _daily_task("2026-10-09", status=TaskStatus.COMPOSING)
        await database.update_task(oldest.id, created_at="2026-10-08T00:00:00+00:00")
        await database.update_task(newer.id, created_at="2026-10-09T00:00:00+00:00")
        assert (await database.get_unfinished_priority_morning_task()).id == oldest.id

    asyncio.run(scenario())


def test_priority_lookup_skips_tests_future_and_other_task_origins(isolated_desk):
    async def scenario():
        await _daily_task("2026-10-09", test_mode=True)
        await database.create_task("topic", "explainer", TaskConfig())
        await database.create_task("news_daily", "manual news", TaskConfig())
        future = await _daily_task("2026-10-10")
        await database.update_task(
            future.id,
            scheduled_at=(datetime.now(timezone.utc) + timedelta(days=1)).isoformat(),
        )
        assert await database.get_next_priority_morning_task() is None
        oldest = await _daily_task("2026-10-08")
        newer = await _daily_task("2026-10-09")
        await database.update_task(oldest.id, created_at="2026-10-08T00:00:00+00:00")
        await database.update_task(newer.id, created_at="2026-10-09T00:00:00+00:00")
        assert (await database.get_next_priority_morning_task()).id == oldest.id

    asyncio.run(scenario())


class _FrozenDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        instant = datetime(2026, 10, 8, 23, 30, tzinfo=timezone.utc)
        return instant.astimezone(tz) if tz is not None else instant.replace(tzinfo=None)


def test_hourly_endpoint_uses_recipe_timezone_even_with_schedule_disabled(
    isolated_desk, monkeypatch
):
    from backend.main import app

    monkeypatch.setattr(daily_news, "datetime", _FrozenDateTime)
    monkeypatch.setattr(daily_news, "load_settings", lambda: DailyAutomationSettings(
        enabled=False, timezone="Asia/Singapore", auto_publish=False
    ))
    client = TestClient(app)
    first = client.post("/api/daily-news/ensure-today")
    second = client.post("/api/daily-news/ensure-today")
    assert first.status_code == second.status_code == 200
    first_body, second_body = first.json(), second.json()
    assert first_body["edition_date"] == "2026-10-09"
    assert first_body["created"] is True
    assert second_body["created"] is False
    assert first_body["task"]["id"] == second_body["task"]["id"]
    assert first_body["trigger"] == "hourly_check"


def test_enabled_schedule_admits_late_edition_despite_old_receipt(
    isolated_desk, monkeypatch
):
    monkeypatch.setattr(scheduler, "datetime", _FrozenDateTime)
    monkeypatch.setattr(scheduler, "load_settings", lambda: DailyAutomationSettings(
        enabled=True, generation_time="04:00", timezone="Asia/Singapore",
        catch_up_after_restart=False, auto_publish=False,
    ))
    (isolated_desk / "state.json").write_text(json.dumps({
        "last_run_date": "2026-10-09", "last_task_id": "missing-task"
    }))

    async def stop_after_iteration(_):
        raise asyncio.CancelledError

    monkeypatch.setattr(scheduler.asyncio, "sleep", stop_after_iteration)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(scheduler._scheduler_loop())
    tasks = asyncio.run(database.list_tasks())
    assert len(tasks) == 1
    assert tasks[0].origin_id == "2026-10-09"
    assert json.loads(tasks[0].source_url)["trigger"] == "schedule"

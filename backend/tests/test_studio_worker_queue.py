"""The actual worker serializes both production paths without provider calls."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from backend import config, explainers as studio, worker
from backend.models import TaskConfig, TaskResponse, TaskStatus


@pytest.fixture
def queue_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUTS_DIR", tmp_path)
    monkeypatch.setattr(studio, "_jobs", {})
    monkeypatch.setattr(worker, "_active_task_ids", set())
    monkeypatch.setattr(worker, "_active_log_queues", {})
    monkeypatch.setattr(worker, "compare_and_set_task_status", AsyncMock(return_value=True))
    monkeypatch.setattr(worker, "update_task", AsyncMock())
    task = TaskResponse(
        id="morning-desk-queue-test", created_at="2026-09-28T00:00:00Z",
        updated_at="2026-09-28T00:00:00Z", source_type="news_daily", source_url="{}",
        source_title="Morning Desk test", status=TaskStatus.QUEUED,
        output_dir=str(tmp_path / "morning"), config=TaskConfig(),
    )
    monkeypatch.setattr(worker, "get_task", AsyncMock(return_value=task.model_copy(
        update={"status": TaskStatus.AWAITING_REVIEW})))
    return task


def topic(title):
    return studio.create_project(studio.ProjectInput(
        title=title, research="Research describes how a learner recalls a concept before checking an answer.",
        tts_model="orpheus-en", voice="tara",
    ))


def test_morning_desk_finishes_before_explainer_writing_and_rendering(queue_runtime, monkeypatch):
    async def scenario():
        project = topic("First explainer")
        studio.start(project["id"], "full")
        news_entered, release_news, rendered = asyncio.Event(), asyncio.Event(), asyncio.Event()
        order = []
        supplied = False

        async def next_news():
            nonlocal supplied
            if not supplied:
                supplied = True
                return queue_runtime
            return None

        async def news(*args, **kwargs):
            order.append("news:start")
            news_entered.set()
            await release_news.wait()
            order.append("news:end")

        async def write(*args, **kwargs):
            order.append("explainer:write")
            return [{"id": "scene-001", "narration": "Recall first.",
                     "visual_prompt": "A learner closes a book."}]

        async def render(*args, **kwargs):
            order.append("explainer:render")
            rendered.set()
            return {"video_path": "final.mp4"}

        monkeypatch.setattr(worker, "get_next_queued_task", next_news)
        monkeypatch.setattr(worker, "run_pipeline", news)
        monkeypatch.setattr(studio, "write_script", write)
        monkeypatch.setattr("backend.explainer_pipeline.render_project", render)
        job = asyncio.create_task(worker._worker_loop())
        try:
            await asyncio.wait_for(news_entered.wait(), 2)
            assert studio.read_project(project["id"])["status"] == "queued"
            assert order == ["news:start"]
            release_news.set()
            await asyncio.wait_for(rendered.wait(), 2)
            assert order == ["news:start", "news:end", "explainer:write", "explainer:render"]
        finally:
            job.cancel()
            await asyncio.gather(job, return_exceptions=True)

    asyncio.run(scenario())


def test_new_morning_job_waits_for_whole_explainer_then_precedes_next_topic(queue_runtime, monkeypatch):
    async def scenario():
        first, second = topic("First"), topic("Second")
        studio.start(first["id"], "full")
        studio.start(second["id"], "full")
        writing, release_write, rendering, release_render, second_done = [asyncio.Event() for _ in range(5)]
        news_available = False
        supplied = False
        order = []

        async def next_news():
            nonlocal supplied
            if news_available and not supplied:
                supplied = True
                return queue_runtime
            return None

        async def write(project, *args, **kwargs):
            order.append(project["title"] + ":write")
            if project["id"] == first["id"]:
                writing.set()
                await release_write.wait()
            return [{"id": "scene-001", "narration": "Recall first.",
                     "visual_prompt": "A learner closes a book."}]

        async def render(project, *args, **kwargs):
            order.append(project["title"] + ":render")
            if project["id"] == first["id"]:
                rendering.set()
                await release_render.wait()
            else:
                second_done.set()
            return {"video_path": "final.mp4"}

        async def news(*args, **kwargs):
            order.append("news")

        monkeypatch.setattr(worker, "get_next_queued_task", next_news)
        monkeypatch.setattr(worker, "run_pipeline", news)
        monkeypatch.setattr(studio, "write_script", write)
        monkeypatch.setattr("backend.explainer_pipeline.render_project", render)
        job = asyncio.create_task(worker._worker_loop())
        try:
            await asyncio.wait_for(writing.wait(), 2)
            news_available = True
            release_write.set()
            await asyncio.wait_for(rendering.wait(), 2)
            assert order == ["First:write", "First:render"]
            assert not supplied
            release_render.set()
            await asyncio.wait_for(second_done.wait(), 2)
            assert order == ["First:write", "First:render", "news", "Second:write", "Second:render"]
        finally:
            job.cancel()
            await asyncio.gather(job, return_exceptions=True)

    asyncio.run(scenario())


def test_cancelling_explainer_does_not_stop_shared_worker(queue_runtime, monkeypatch):
    async def scenario():
        first, second = topic("Cancelled"), topic("Next")
        studio.start(first["id"], "script")
        studio.start(second["id"], "script")
        started, continued = asyncio.Event(), asyncio.Event()

        async def write(project, *args, **kwargs):
            if project["id"] == first["id"]:
                started.set()
                await asyncio.Event().wait()
            continued.set()
            return [{"id": "scene-001", "narration": "Recall first.",
                     "visual_prompt": "A learner closes a book."}]

        monkeypatch.setattr(worker, "get_next_queued_task", AsyncMock(return_value=None))
        monkeypatch.setattr(studio, "write_script", write)
        job = asyncio.create_task(worker._worker_loop())
        try:
            await asyncio.wait_for(started.wait(), 2)
            await studio.cancel(first["id"])
            await asyncio.wait_for(continued.wait(), 2)
            assert not job.done()
            assert studio.read_project(first["id"])["status"] == "cancelled"
        finally:
            job.cancel()
            await asyncio.gather(job, return_exceptions=True)
        assert not studio._jobs

    asyncio.run(scenario())

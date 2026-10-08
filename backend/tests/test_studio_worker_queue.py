"""The actual worker serializes both production paths without provider calls."""

import asyncio
from unittest.mock import AsyncMock, patch

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


def test_failed_explainer_reserves_worker_through_repair_and_resume(queue_runtime, monkeypatch):
    async def scenario():
        first, second = topic("Recover me"), topic("Later")
        studio.start(first['id'], 'full')
        studio.start(second['id'], 'full')
        failed, news_done = asyncio.Event(), asyncio.Event()
        order = []
        attempts = 0

        async def write(project, *args):
            nonlocal attempts
            if project['id'] == first['id']:
                attempts += 1
                if attempts == 1:
                    raise RuntimeError('Provider offline')
            return [{'id': 'scene-001', 'narration': 'Recall first.',
                     'visual_prompt': 'A learner closes a book.'}]

        async def render(project, *args):
            order.append(project['title'])
            return {'video_path': 'final.mp4'}

        async def idle(_):
            if studio.read_project(first['id'])['status'] == 'failed':
                failed.set()
            await asyncio.Event().wait()

        next_news = AsyncMock(side_effect=[queue_runtime, None])
        # Only offer Morning Desk once the Explainer has actually failed.
        async def next_task():
            if attempts:
                if next_news.await_count < 2:
                    return await next_news()
            return None

        async def news(*args, **kwargs):
            order.append('news')
            news_done.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(worker, 'get_next_queued_task', next_task)
        monkeypatch.setattr(worker, 'run_pipeline', news)
        monkeypatch.setattr(studio, 'write_script', write)
        monkeypatch.setattr('backend.explainer_pipeline.render_project', render)
        # Replace the idle sleep only; no provider is called by these tests.
        with patch.object(worker.asyncio, 'sleep', idle), patch('backend.database.list_tasks', AsyncMock(return_value=[])):
            job = asyncio.create_task(worker._worker_loop())
            try:
                await asyncio.wait_for(failed.wait(), 2)
                next_news.assert_not_awaited()
                owner = studio.production_owner()
                assert owner['id'] == first['id']
                assert not await studio.run_next_queued()
                state = await studio.queue_state()
                assert state['reservation']['id'] == first['id']
                # Repair edits and service recovery must retain the disk claim.
                studio.update_project(first['id'], studio.ProjectUpdate(
                    **{**{k: owner[k] for k in studio.ProjectInput.model_fields}, 'title': 'Repaired'},
                    revision=owner['revision']))
                studio.recover_interrupted()
                studio.start(first['id'], 'full')
                assert studio.queue_positions()[first['id']] == 1
            finally:
                job.cancel()
                await asyncio.gather(job, return_exceptions=True)
        # Fresh worker simulates a restart; only persisted project state remains.
        job = asyncio.create_task(worker._worker_loop())
        try:
            await asyncio.wait_for(news_done.wait(), 2)
            assert order == ['Repaired', 'news']
            assert studio.read_project(first['id'])['status'] == 'completed'
            assert studio.production_owner() is None
            assert studio.read_project(second['id'])['status'] == 'queued'
        finally:
            job.cancel()
            await asyncio.gather(job, return_exceptions=True)
    asyncio.run(scenario())


def test_running_morning_finishes_before_reserved_recovery(queue_runtime, monkeypatch):
    async def scenario():
        project = topic('Recover after Morning Desk')
        news_entered, release_news, resumed = [asyncio.Event() for _ in range(3)]
        order = []
        supplied = False

        async def next_news():
            nonlocal supplied
            if not supplied:
                supplied = True
                return queue_runtime
            return None

        async def news(*args, **kwargs):
            order.append('news:start')
            news_entered.set()
            await release_news.wait()
            order.append('news:end')

        async def write(*args, **kwargs):
            order.append('explainer:resume')
            resumed.set()
            return [{'id': 'scene-001', 'narration': 'Recall first.',
                     'visual_prompt': 'A learner closes a book.'}]

        monkeypatch.setattr(worker, 'get_next_queued_task', next_news)
        monkeypatch.setattr(worker, 'run_pipeline', news)
        monkeypatch.setattr(studio, 'write_script', write)
        job = asyncio.create_task(worker._worker_loop())
        try:
            await asyncio.wait_for(news_entered.wait(), 2)
            studio.patch_project(project['id'], {'status': 'interrupted'})
            studio.start(project['id'], 'script')
            assert studio.production_owner()['id'] == project['id']
            assert order == ['news:start']
            release_news.set()
            await asyncio.wait_for(resumed.wait(), 2)
            assert order == ['news:start', 'news:end', 'explainer:resume']
        finally:
            job.cancel()
            await asyncio.gather(job, return_exceptions=True)
    asyncio.run(scenario())

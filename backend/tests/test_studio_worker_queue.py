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
    monkeypatch.setattr(worker, "get_next_priority_morning_task", AsyncMock(return_value=None))
    monkeypatch.setattr(worker, "get_unfinished_priority_morning_task", AsyncMock(return_value=None))
    task = TaskResponse(
        id="morning-desk-queue-test", created_at="2026-09-28T00:00:00Z",
        updated_at="2026-09-28T00:00:00Z", source_type="news_daily", source_url="{}",
        source_title="Morning Desk test", status=TaskStatus.QUEUED,
        output_dir=str(tmp_path / "morning"), config=TaskConfig(),
    )
    monkeypatch.setattr(worker, "get_task", AsyncMock(return_value=task.model_copy(
        update={"status": TaskStatus.AWAITING_REVIEW})))
    return task


def test_priority_morning_preempts_and_resumes_original_before_next_topic(queue_runtime, monkeypatch):
    async def scenario():
        first, second = topic("Original"), topic("Next")
        studio.start(first["id"], "full")
        studio.start(second["id"], "full")
        entered, cleaned, news_entered, release_news, resumed = [asyncio.Event() for _ in range(5)]
        priority_available = False
        priority_claimed = False
        order = []
        attempts = 0

        async def priority():
            return queue_runtime if priority_available and not priority_claimed else None

        async def claim(*args, **kwargs):
            nonlocal priority_claimed
            priority_claimed = True
            return True

        async def write(project, *args, **kwargs):
            nonlocal attempts
            if project["id"] == first["id"]:
                attempts += 1
                if attempts == 1:
                    order.append("explainer:start")
                    entered.set()
                    try:
                        await asyncio.Event().wait()
                    finally:
                        await asyncio.sleep(0)
                        order.append("explainer:cleanup")
                        cleaned.set()
                else:
                    order.append("explainer:resume")
                    resumed.set()
            else:
                order.append("next:start")
            return [{"id": "scene-001", "narration": "Recall first.",
                     "visual_prompt": "A learner closes a book."}]

        async def news(*args, **kwargs):
            assert cleaned.is_set()
            assert not studio._jobs
            assert studio.read_project(first["id"])["cancellation_reason"] == "morning_desk"
            order.append("news:start")
            news_entered.set()
            await release_news.wait()
            order.append("news:end")

        monkeypatch.setattr(worker, "get_next_priority_morning_task", priority)
        monkeypatch.setattr(worker, "get_next_queued_task", AsyncMock(return_value=None))
        monkeypatch.setattr(worker, "compare_and_set_task_status", claim)
        monkeypatch.setattr(worker, "run_pipeline", news)
        monkeypatch.setattr(studio, "write_script", write)
        monkeypatch.setattr("backend.explainer_pipeline.render_project", AsyncMock(return_value={"video_path": "final.mp4"}))
        job = asyncio.create_task(worker._worker_loop())
        try:
            await asyncio.wait_for(entered.wait(), 2)
            original_production = dict(studio.read_project(first["id"])["production"])
            priority_available = True
            await asyncio.wait_for(news_entered.wait(), 3)
            paused = studio.read_project(first["id"])
            assert paused["production"] == original_production
            assert paused["status"] == "interrupted"
            assert order == ["explainer:start", "explainer:cleanup", "news:start"]
            release_news.set()
            await asyncio.wait_for(resumed.wait(), 3)
            assert order[:5] == ["explainer:start", "explainer:cleanup", "news:start", "news:end", "explainer:resume"]
        finally:
            job.cancel()
            await asyncio.gather(job, return_exceptions=True)

    asyncio.run(scenario())


def test_waiting_priority_morning_yields_reserved_explainer(queue_runtime, monkeypatch):
    async def scenario():
        project = topic("Reserved recovery")
        studio.start(project["id"], "full")
        studio.patch_project(project["id"], {
            "status": "interrupted", "queue": None,
            "production": {"operation": "full", "started_at": "2026-10-08T00:00:00Z"},
            "cancellation_reason": "service_interruption",
        })
        admitted = True
        order = []
        news_done = asyncio.Event()

        async def priority():
            return queue_runtime if admitted else None

        async def news(*args, **kwargs):
            nonlocal admitted
            order.append("news")
            assert studio.production_owner() is None
            assert studio.priority_suspensions()[0]["id"] == project["id"]
            admitted = False
            news_done.set()

        async def write(*args, **kwargs):
            order.append("recovery")
            return [{"id": "scene-001", "narration": "Recall first.",
                     "visual_prompt": "A learner closes a book."}]

        monkeypatch.setattr(worker, "get_next_priority_morning_task", priority)
        monkeypatch.setattr(worker, "get_next_queued_task", AsyncMock(return_value=None))
        monkeypatch.setattr(worker, "run_pipeline", news)
        monkeypatch.setattr(studio, "write_script", write)
        monkeypatch.setattr("backend.explainer_pipeline.render_project", AsyncMock(return_value={"video_path": "final.mp4"}))
        job = asyncio.create_task(worker._worker_loop())
        try:
            await asyncio.wait_for(news_done.wait(), 2)
            for _ in range(20):
                if "recovery" in order:
                    break
                await asyncio.sleep(.01)
            assert order[:2] == ["news", "recovery"]
        finally:
            job.cancel()
            await asyncio.gather(job, return_exceptions=True)

    asyncio.run(scenario())


def test_priority_poll_failure_does_not_cancel_running_explainer(queue_runtime, monkeypatch):
    async def scenario():
        release = asyncio.Event()
        cancelled = []

        async def run():
            try:
                await release.wait()
                return True
            except asyncio.CancelledError:
                cancelled.append(True)
                raise

        async def priority():
            release.set()
            raise RuntimeError("Temporary database lookup failure")

        monkeypatch.setattr(studio, "run_next_queued", run)
        monkeypatch.setattr(worker, "get_next_priority_morning_task", priority)
        assert await worker._run_explainer_with_priority_checks()
        assert not cancelled

    asyncio.run(scenario())


def test_paused_explainer_waits_for_morning_review_to_finish(queue_runtime, monkeypatch):
    async def scenario():
        project = topic("Paused production")
        studio.start(project["id"], "full")
        studio.patch_project(project["id"], {
            "status": "interrupted", "queue": None,
            "production": {"operation": "full", "started_at": "2026-10-08T00:00:00Z"},
            "cancellation_reason": "service_interruption",
        })
        await studio.pause_for_morning_desk(project["id"])
        reviewing = True
        checked, resumed = asyncio.Event(), asyncio.Event()

        async def pending():
            checked.set()
            return queue_runtime.model_copy(update={"status": TaskStatus.AWAITING_REVIEW}) if reviewing else None

        async def write(*args, **kwargs):
            resumed.set()
            return [{"id": "scene-001", "narration": "Recall first.",
                     "visual_prompt": "A learner closes a book."}]

        monkeypatch.setattr(worker, "get_unfinished_priority_morning_task", pending)
        monkeypatch.setattr(worker, "get_next_queued_task", AsyncMock(return_value=None))
        monkeypatch.setattr(studio, "write_script", write)
        monkeypatch.setattr("backend.explainer_pipeline.render_project", AsyncMock(return_value={"video_path": "final.mp4"}))
        job = asyncio.create_task(worker._worker_loop())
        try:
            await asyncio.wait_for(checked.wait(), 2)
            assert studio.read_project(project["id"])["cancellation_reason"] == "morning_desk"
            assert not resumed.is_set()
            reviewing = False
            await asyncio.wait_for(resumed.wait(), 2)
        finally:
            job.cancel()
            await asyncio.gather(job, return_exceptions=True)

    asyncio.run(scenario())


def topic(title):
    return studio.create_project(studio.ProjectInput(
        title=title, research="Research describes how a learner recalls a concept before checking an answer.",
        tts_model="orpheus-en", voice="tara",
    ))


def test_queued_explainer_precedes_waiting_morning_desk(queue_runtime, monkeypatch):
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
            assert studio.read_project(project["id"])["status"] == "completed"
            assert order == ["explainer:write", "explainer:render", "news:start"]
            release_news.set()
            await asyncio.wait_for(rendered.wait(), 2)
            assert order[:3] == ["explainer:write", "explainer:render", "news:start"]
        finally:
            job.cancel()
            await asyncio.gather(job, return_exceptions=True)

    asyncio.run(scenario())


def test_new_morning_job_waits_for_whole_explainer_and_next_queued_topic(queue_runtime, monkeypatch):
    async def scenario():
        first, second = topic("First"), topic("Second")
        studio.start(first["id"], "full")
        studio.start(second["id"], "full")
        writing, release_write, rendering, release_render, news_done = [asyncio.Event() for _ in range(5)]
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
            return {"video_path": "final.mp4"}

        async def news(*args, **kwargs):
            order.append("news")
            news_done.set()

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
            await asyncio.wait_for(news_done.wait(), 2)
            assert order == ["First:write", "First:render", "Second:write", "Second:render", "news"]
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


def test_explainer_enqueued_during_news_lookup_precedes_unclaimed_news(queue_runtime, monkeypatch):
    async def scenario():
        project = topic("Arrived during lookup")
        queued = False
        news_done = asyncio.Event()
        order = []

        async def next_news():
            nonlocal queued
            if not queued:
                queued = True
                studio.start(project["id"], "script")
            return None if news_done.is_set() else queue_runtime

        async def write(*args, **kwargs):
            order.append("explainer")
            return [{"id": "scene-001", "narration": "Recall first.",
                     "visual_prompt": "A learner closes a book."}]

        async def news(*args, **kwargs):
            order.append("news")
            news_done.set()

        monkeypatch.setattr(worker, "get_next_queued_task", next_news)
        monkeypatch.setattr(worker, "run_pipeline", news)
        monkeypatch.setattr(studio, "write_script", write)
        job = asyncio.create_task(worker._worker_loop())
        try:
            await asyncio.wait_for(news_done.wait(), 2)
            assert order == ["explainer", "news"]
        finally:
            job.cancel()
            await asyncio.gather(job, return_exceptions=True)

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
            assert order == ['Repaired', 'Later', 'news']
            assert studio.read_project(first['id'])['status'] == 'completed'
            assert studio.production_owner() is None
            assert studio.read_project(second['id'])['status'] == 'completed'
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

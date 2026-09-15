import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from backend.models import TaskConfig, TaskResponse
from backend.pipeline import orchestrator
from backend.pipeline.tts import TtsIntegrityError
from backend import worker


@pytest.mark.parametrize('failed_branch', ['audio', 'footage'])
def test_parallel_failure_preserves_completed_sibling_and_blocks_composition(tmp_path, failed_branch):
    async def run():
        entered = asyncio.Event()
        completed = []
        task = TaskResponse(id='parallel-failure', created_at='2026-09-14T00:00:00Z',
            updated_at='2026-09-14T00:00:00Z', source_type='youtube', source_url='https://example.com',
            status='queued', output_dir=str(tmp_path),
            config=TaskConfig(footage_enabled=True, thumbnail_enabled=False, auto_render=True))

        async def footage(*args, **kwargs):
            entered.set()
            if failed_branch == 'footage':
                raise RuntimeError('footage blocked')
            await asyncio.sleep(0)
            completed.append('footage')

        async def audio(*args, **kwargs):
            await entered.wait()
            if failed_branch == 'audio':
                raise RuntimeError('audio blocked')
            completed.append('audio')
            return str(tmp_path / 'audio.wav')

        after = AsyncMock()
        with (
            patch.object(orchestrator, 'update_task', AsyncMock()),
            patch.object(orchestrator, 'extract_youtube', AsyncMock(return_value=SimpleNamespace(title='Test', text='Source', metadata={}))),
            patch.object(orchestrator, 'summarize', AsyncMock(return_value={})),
            patch.object(orchestrator, 'generate_script', AsyncMock(return_value='Narration.')),
            patch.object(orchestrator, '_generate_task_title', AsyncMock(return_value=SimpleNamespace(title='Title'))),
            patch.object(orchestrator, '_acquire_task_footage', footage),
            patch.object(orchestrator, 'generate_tts', audio),
            patch.object(orchestrator, '_after_audio', after),
        ):
            with pytest.raises(RuntimeError, match=failed_branch + ' blocked'):
                await orchestrator.run_pipeline(task)
        assert completed == ['audio' if failed_branch == 'footage' else 'footage']
        after.assert_not_awaited()
        if failed_branch == 'footage':
            assert task.audio_path == str(tmp_path / 'audio.wav')
    asyncio.run(run())


@pytest.fixture
def pipeline(tmp_path):
    task = TaskResponse(
        id='parallel-worker', created_at='2026-09-15T00:00:00Z',
        updated_at='2026-09-15T00:00:00Z', source_type='youtube',
        source_url='https://example.com', status='queued', output_dir=str(tmp_path),
        config=TaskConfig(footage_enabled=True, thumbnail_enabled=False, auto_render=True),
    )
    with (
        patch.object(orchestrator, 'update_task', AsyncMock()) as update,
        patch.object(orchestrator, 'extract_youtube', AsyncMock(return_value=SimpleNamespace(title='Test', text='Source', metadata={}))),
        patch.object(orchestrator, 'summarize', AsyncMock(return_value={})),
        patch.object(orchestrator, 'generate_script', AsyncMock(return_value='Narration.')),
        patch.object(orchestrator, '_generate_task_title', AsyncMock(return_value=SimpleNamespace(title='Title'))),
        patch.object(orchestrator, '_after_audio', AsyncMock()) as after,
    ):
        yield task, update, after


@pytest.mark.parametrize('failed_branch', ['audio', 'footage'])
def test_worker_reports_parallel_failure_without_waiting_for_sibling(
    tmp_path, pipeline, caplog, failed_branch,
):
    async def run():
        task, update, after = pipeline
        entered = asyncio.Event()
        cleaned = asyncio.Event()
        artifact = tmp_path / 'verified-segment'
        error = (TtsIntegrityError('audio integrity rejected', part_key='part_005')
                 if failed_branch == 'audio' else RuntimeError('footage unavailable'))

        async def waiting(*args, **kwargs):
            artifact.write_bytes(b'verified content')
            entered.set()
            try:
                # Simulate a provider wait that never completes by itself.
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(0)
                cleaned.set()

        async def failing(*args, **kwargs):
            await entered.wait()
            raise error

        events = worker.subscribe_task_logs(task.id)
        try:
            with (
                patch.object(orchestrator, 'generate_tts', failing if failed_branch == 'audio' else waiting),
                patch.object(orchestrator, '_acquire_task_footage', failing if failed_branch == 'footage' else waiting),
                patch.object(worker, 'get_next_queued_task', AsyncMock(side_effect=[task, asyncio.CancelledError()])),
                patch.object(worker, 'compare_and_set_task_status', AsyncMock(return_value=True)),
                patch.object(worker, 'update_task', update),
                patch.object(worker, 'run_auto_publish_pipeline', AsyncMock()) as publish,
            ):
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(worker._worker_loop(), timeout=1)

            assert cleaned.is_set()
            assert artifact.read_bytes() == b'verified content'
            after.assert_not_awaited()
            publish.assert_not_awaited()
            assert update.await_args.kwargs['status'] == 'failed'
            assert update.await_args.kwargs['error_message'] == str(error)
            assert not worker.is_task_logging_active(task.id)
            delivered = []
            while not events.empty():
                delivered.append(events.get_nowait())
            assert {'event': 'status', 'data': 'failed'} in delivered
            assert 'Traceback (most recent call last)' in caplog.text
            assert f'{type(error).__name__}: {error}' in caplog.text
            assert f'Task failed: {error}' in (tmp_path / 'logs/pipeline.log').read_text()
        finally:
            worker.finish_task_logs(task.id, 'failed')

    asyncio.run(run())


def test_parallel_success_waits_for_both_branches(pipeline, tmp_path):
    async def run():
        task, _, after = pipeline
        audio_ready = asyncio.Event()
        release_media = asyncio.Event()
        audio_path = str(tmp_path / 'audio.wav')

        async def audio(*args, **kwargs):
            audio_ready.set()
            return audio_path

        async def media(*args, **kwargs):
            await audio_ready.wait()
            await release_media.wait()

        with (
            patch.object(orchestrator, 'generate_tts', audio),
            patch.object(orchestrator, '_acquire_task_footage', media),
        ):
            pending = asyncio.create_task(orchestrator.run_pipeline(task))
            try:
                await asyncio.wait_for(audio_ready.wait(), timeout=1)
                assert not pending.done()
                after.assert_not_awaited()
                release_media.set()
                await asyncio.wait_for(pending, timeout=1)
            finally:
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
        after.assert_awaited_once()
        assert after.await_args.kwargs['audio_path'] == audio_path
        assert task.audio_path == audio_path

    asyncio.run(run())


def test_pipeline_cancellation_cleans_up_both_branches(pipeline):
    async def run():
        task, _, after = pipeline
        entered = {name: asyncio.Event() for name in ('audio', 'media')}
        cleaned = set()

        async def wait(name):
            entered[name].set()
            try:
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(0)
                cleaned.add(name)

        with (
            patch.object(orchestrator, 'generate_tts', lambda *a, **kw: wait('audio')),
            patch.object(orchestrator, '_acquire_task_footage', lambda *a, **kw: wait('media')),
        ):
            pending = asyncio.create_task(orchestrator.run_pipeline(task))
            try:
                await asyncio.wait_for(asyncio.gather(*(event.wait() for event in entered.values())), timeout=1)
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(pending, timeout=1)
            finally:
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
        assert cleaned == {'audio', 'media'}
        after.assert_not_awaited()

    asyncio.run(run())

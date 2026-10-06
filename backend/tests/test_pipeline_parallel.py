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
        config=TaskConfig(footage_enabled=True, thumbnail_enabled=False, auto_render=True,
                          tts_model='chatterbox-multilingual'),
    )
    (tmp_path / 'script.txt').write_text('Approved narration.')
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


@pytest.mark.parametrize('entry', ['run_pipeline', 'run_regenerate', 'run_tts_resume'])
def test_parallel_success_waits_for_both_branches(pipeline, tmp_path, entry):
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
            pending = asyncio.create_task(getattr(orchestrator, entry)(task))
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


@pytest.mark.parametrize('entry', ['run_pipeline', 'run_regenerate', 'run_tts_resume'])
def test_pipeline_cancellation_cleans_up_both_branches(pipeline, entry):
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
            pending = asyncio.create_task(getattr(orchestrator, entry)(task))
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


@pytest.mark.parametrize('failed_branch', ['audio', 'footage'])
@pytest.mark.parametrize('entry', ['run_regenerate', 'run_tts_resume'])
def test_recovery_failure_cancels_sibling_without_discarding_artifacts(pipeline, tmp_path, failed_branch, entry):
    async def run():
        task, _, after = pipeline
        entered = asyncio.Event()
        cleaned = asyncio.Event()
        artifact = tmp_path / 'verified-part'

        async def waiting(*args, **kwargs):
            artifact.write_bytes(b'preserved')
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaned.set()

        async def failing(*args, **kwargs):
            await entered.wait()
            raise TtsIntegrityError('audio rejected') if failed_branch == 'audio' else RuntimeError('footage rejected')

        with (
            patch.object(orchestrator, 'generate_tts', failing if failed_branch == 'audio' else waiting),
            patch.object(orchestrator, '_acquire_task_footage', failing if failed_branch == 'footage' else waiting),
        ):
            with pytest.raises(RuntimeError, match=failed_branch + ' rejected'):
                await asyncio.wait_for(getattr(orchestrator, entry)(task), timeout=1)
        assert cleaned.is_set()
        assert artifact.read_bytes() == b'preserved'
        after.assert_not_awaited()

    asyncio.run(run())


@pytest.mark.parametrize('thumbnail_fails', [False, True])
def test_regenerate_overlaps_audio_with_serial_browser_work(pipeline, thumbnail_fails):
    async def run():
        task, _, after = pipeline
        task.config.thumbnail_enabled = True
        audio_entered = asyncio.Event()
        media_finished = asyncio.Event()
        events = []

        async def thumbnail(*args, **kwargs):
            assert kwargs['allow_legacy_regeneration'] is True
            await audio_entered.wait()
            events.append('thumbnail')
            if thumbnail_fails:
                raise RuntimeError('cover unavailable')

        async def media(*args, **kwargs):
            assert events == ['thumbnail']
            events.append('footage')
            media_finished.set()

        async def audio(*args, **kwargs):
            audio_entered.set()
            await media_finished.wait()
            return 'audio.wav'

        with (
            patch.object(orchestrator, '_generate_task_thumbnail', thumbnail),
            patch.object(orchestrator, '_acquire_task_footage', media),
            patch.object(orchestrator, 'generate_tts', audio),
        ):
            await asyncio.wait_for(orchestrator.run_regenerate(task), timeout=1)
        assert events == ['thumbnail', 'footage']
        after.assert_awaited_once()

    asyncio.run(run())


@pytest.mark.parametrize('model,auto_render', [('chatterbox-multilingual', False), ('pocket-tts', True)])
def test_regenerate_preserves_manual_review_and_local_model_order(pipeline, model, auto_render):
    async def run():
        task, _, after = pipeline
        task.config.tts_model = model
        task.config.auto_render = auto_render
        task.config.thumbnail_enabled = True
        events = []

        async def thumbnail(*args, **kwargs):
            await asyncio.sleep(0)
            events.append('thumbnail')

        async def audio(*args, **kwargs):
            assert events == ['thumbnail']
            events.append('audio')
            return 'audio.wav'

        with (
            patch.object(orchestrator, '_generate_task_thumbnail', thumbnail),
            patch.object(orchestrator, '_ensure_task_footage', AsyncMock()) as media,
            patch.object(orchestrator, 'generate_tts', audio),
        ):
            await orchestrator.run_regenerate(task)
        media.assert_not_awaited()
        after.assert_awaited_once()

    asyncio.run(run())


@pytest.mark.parametrize('model,auto_render', [
    ('chatterbox-multilingual', False), ('pocket-tts', True),
])
def test_tts_resume_preserves_manual_review_and_local_model_order(pipeline, model, auto_render):
    async def run():
        task, _, after = pipeline
        task.config.tts_model = model
        task.config.auto_render = auto_render
        with (
            patch.object(orchestrator, 'generate_tts', AsyncMock(return_value='audio.wav')),
            patch.object(orchestrator, '_ensure_task_footage', AsyncMock()) as media,
        ):
            await orchestrator.run_tts_resume(task)
        media.assert_not_awaited()
        after.assert_awaited_once()

    asyncio.run(run())


def test_tts_resume_preserves_editorial_artifacts_and_reuses_verified_footage(pipeline, tmp_path):
    async def run():
        task, _, after = pipeline
        task.config.thumbnail_enabled = True
        task.generated_title = 'Saved title'
        task.thumbnail_path = str(tmp_path / 'saved-cover.png')
        script = (tmp_path / 'script.txt').read_text()
        Path(task.thumbnail_path).write_bytes(b'saved cover')
        manifest = {
            'clips': [{'id': 'verified-clip'}],
            'task_request_sha256': orchestrator._footage_request_sha256(task, script),
        }
        with (
            patch.object(orchestrator, '_generate_task_title', AsyncMock()) as title,
            patch.object(orchestrator, '_generate_task_thumbnail', AsyncMock()) as thumbnail,
            patch.object(orchestrator.footage, 'read_manifest', return_value=manifest),
            patch.object(orchestrator.footage, 'acquisition_is_complete', return_value=True) as complete,
            patch.object(orchestrator, '_acquire_task_footage', AsyncMock()) as acquire,
            patch.object(orchestrator, 'generate_tts', AsyncMock(return_value='audio.wav')) as audio,
        ):
            await orchestrator.run_tts_resume(task)
        title.assert_not_awaited()
        thumbnail.assert_not_awaited()
        acquire.assert_not_awaited()
        complete.assert_called_once_with(tmp_path, manifest, script)
        audio.assert_awaited_once()
        assert audio.await_args.args[:2] == (str(tmp_path / 'script.txt'), str(tmp_path / 'audio'))
        assert (tmp_path / 'script.txt').read_text() == script
        assert task.generated_title == 'Saved title'
        assert Path(task.thumbnail_path).read_bytes() == b'saved cover'
        after.assert_awaited_once()

    asyncio.run(run())

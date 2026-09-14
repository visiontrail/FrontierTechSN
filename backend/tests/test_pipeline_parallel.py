import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from backend.models import TaskConfig, TaskResponse
from backend.pipeline import orchestrator


@pytest.mark.parametrize('failed_branch', ['audio', 'footage'])
def test_parallel_failure_drains_sibling_and_blocks_composition(tmp_path, failed_branch):
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

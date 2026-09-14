import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.pipeline import timing


def test_wall_partition_does_not_add_concurrent_work(tmp_path):
    recorder = timing.Recorder(tmp_path)
    for identifier, name, kind, start, stop, parent in [
        ('root', 'pipeline', 'processing', 0, 10, None),
        ('a', 'tts', 'processing', 1, 8, 'root'),
        ('b', 'web', 'provider_wait', 2, 6, 'root'),
        ('c', 'pipeline', 'processing', 15, 20, None),
    ]:
        recorder.write(dict(event='start', id=identifier, name=name, kind=kind, at=start, parent=parent))
        recorder.write(dict(event='end', id=identifier, at=stop, seconds=stop-start, status='ok'))
    report = recorder.report()
    assert report['wall_seconds'] == 20
    assert report['active_wall_seconds'] == 15
    assert report['pause_or_recovery_gap_seconds'] == 5
    assert sum(report['wall_partition_seconds'].values()) == 20
    assert report['wall_partition_seconds']['parallel_overlap'] == 4
    assert report['stages']['pipeline']['work_seconds'] == 15


def test_failed_and_cancelled_spans_are_durable_and_context_resets(tmp_path):
    @timing.timed('pipeline', task_entry=True)
    async def run(task):
        with timing.span('external', 'external_response'):
            raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run(SimpleNamespace(output_dir=tmp_path)))
    report = json.loads((tmp_path / 'timing_report.json').read_text())
    assert report['stages']['external']['failed_attempts'] == 1
    assert report['unfinished_spans'] == []
    assert timing._recorder.get() is None


def test_incomplete_span_remains_unknown(tmp_path):
    recorder = timing.Recorder(tmp_path)
    recorder.write(dict(event='start', id='crash', name='render', kind='processing', at=1, parent=None))
    report = recorder.report()
    assert report['unfinished_spans'][0]['id'] == 'crash'
    assert report['active_wall_seconds'] == 0

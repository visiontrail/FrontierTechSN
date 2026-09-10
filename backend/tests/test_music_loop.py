from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import pytest

from backend.pipeline.music_loop import RATE, render_music_bed


def _source(path: Path, seconds: float = 8) -> np.ndarray:
    t = np.arange(round(seconds * RATE)) / RATE
    # Recurring pulse/harmony with deliberately unsuitable silent file edges.
    envelope = 0.2 + 0.15 * np.cos(2 * np.pi * 2 * t)
    tone = envelope * (np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 330 * t))
    tone[:RATE // 2] = 0
    tone[-RATE:] = 0
    samples = np.column_stack((tone, tone)).astype('<f4')
    samples.tofile(path)
    return samples


def _read(path: Path) -> np.ndarray:
    with wave.open(str(path)) as stream:
        assert stream.getframerate() == RATE
        return np.frombuffer(stream.readframes(stream.getnframes()), dtype='<i2').reshape(-1, 2) / 32768


def test_many_loops_remove_silent_edges_without_drift_or_clicks(tmp_path):
    source, output = tmp_path / 'source.f32', tmp_path / 'bed.wav'
    _source(source)
    report = render_music_bed(source, output, 63.137)
    audio = _read(output)
    assert len(audio) == round(63.137 * RATE)
    assert len(report['seam_seconds']) > 10
    assert np.max(abs(audio)) < 0.96
    # Every 100 ms block after the opening/before the final fade stays audible.
    blocks = audio[RATE:-RATE][: (len(audio) - 2 * RATE) // 4800 * 4800].reshape(-1, 4800, 2)
    assert np.min(np.sqrt(np.mean(blocks ** 2, axis=(1, 2)))) > 0.04
    for seam in report['seam_seconds']:
        for time in (seam, seam + report['crossfade_seconds']):
            frame = round(time * RATE)
            if frame < len(audio):
                assert np.max(abs(audio[frame] - audio[frame - 1])) < 0.04
    assert json.loads(output.with_suffix('.json').read_text())['duration_seconds'] == pytest.approx(63.137)


def test_short_program_keeps_single_pass(tmp_path):
    source, output = tmp_path / 'source.f32', tmp_path / 'bed.wav'
    original = _source(source)
    report = render_music_bed(source, output, 3.125)
    audio = _read(output)
    assert report['mode'] == 'single_pass'
    np.testing.assert_allclose(audio[RATE:2 * RATE], original[RATE:2 * RATE], atol=5e-5)


@pytest.mark.parametrize('value', [0, float('nan')])
def test_invalid_audio_fails_closed(tmp_path, value):
    source = tmp_path / 'bad.f32'
    np.full((RATE, 2), value, dtype='<f4').tofile(source)
    with pytest.raises(ValueError):
        render_music_bed(source, tmp_path / 'bed.wav', 4)


def test_subsecond_source_can_cover_longer_narration(tmp_path):
    source = tmp_path / 'short.f32'
    t = np.arange(round(0.3 * RATE)) / RATE
    signal = 0.2 * np.sin(2 * np.pi * 220 * t)
    np.column_stack((signal, signal)).astype('<f4').tofile(source)
    output = tmp_path / 'short-bed.wav'
    report = render_music_bed(source, output, 2.731)
    assert report['period_seconds'] > 0
    assert len(_read(output)) == round(2.731 * RATE)

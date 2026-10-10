import wave

import numpy as np
import pytest

from backend.pipeline.tts_tail_guard import inspect_tail


def waveform(tmp_path, segments):
    path = tmp_path / 'provider.wav'
    samples = np.concatenate([np.full(round(seconds * 24000), amplitude, dtype='<i2')
                              for seconds, amplitude in segments])
    with wave.open(str(path), 'wb') as writer:
        writer.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
        writer.writeframes(samples.tobytes())
    return path


@pytest.mark.parametrize('end,segments', [
    (6.7, [(6.7, 3000), (.74, 0), (.4, 4000)]),
    (.78, [(.9, 3000), (.68, 0), (.5, 4000)]),
])
def test_asr_invisible_noise_and_short_phrase_restart_are_rejected(tmp_path, end, segments):
    path = waveform(tmp_path, segments)
    before = path.read_bytes()
    report = inspect_tail(path, [{'text': 'complete requested speech', 'end': end}])
    assert not report['passed']
    assert report['residual_start_seconds'] >= end + .12
    assert path.read_bytes() == before


@pytest.mark.parametrize('segments,end', [
    ([(1, 3000), (1, 0)], 1),  # A naturally quiet tail.
    ([(1.1, 3000), (.4, 0)], 1),  # The final phoneme extends past the ASR edge.
    ([(1, 3000), (.8, 0), (.6, 3000), (.4, 0)], 2.4),  # Authored post-pause speech.
    ([(1, 3000), (.5, 0), (.2, 80)], 1),  # Low-level room noise.
])
def test_natural_tails_and_source_speech_pass(tmp_path, segments, end):
    assert inspect_tail(waveform(tmp_path, segments), [{'end': end}])['passed']


@pytest.mark.parametrize('words', [[], [{'text': 'no timing'}], [{'end': float('nan')}], [{'end': 9}]])
def test_unavailable_evidence_never_silently_accepts(tmp_path, words):
    with pytest.raises(RuntimeError, match='preserving the completed job'):
        inspect_tail(waveform(tmp_path, [(1, 0)]), words)

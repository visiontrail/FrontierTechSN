import json
import subprocess
from types import SimpleNamespace

import pytest

from backend.pipeline import chatterbox_independent as independent, tts, tts_pronunciation


@pytest.fixture
def audit_case(tmp_path, monkeypatch):
    wav = tmp_path / 'original.wav'
    wav.write_bytes(b'original provider waveform')
    observed = {'text': 'The report mentions DIY today.'}
    calls = []

    def decode(command, **kwargs):
        calls.append((command, kwargs))
        with open(command[5], 'w') as stream:
            json.dump([observed], stream)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(independent.subprocess, 'run', decode)
    source = 'The report mentions DIY today.'
    _, edits = tts_pronunciation.validate_decisions(source, [{
        'id': 0, 'source': 'DIY', 'action': 'spell', 'spoken': 'dee eye Y',
        'reason': 'Source-bound fixture decision.',
    }])
    return wav, source, edits, tmp_path / 'audit', observed, calls


@pytest.mark.asyncio
async def test_independent_cache_binds_audio_source_decisions_and_transcript(audit_case):
    wav, source, edits, directory, observed, calls = audit_case
    first = await independent.audit_part(wav, source, edits, directory)
    assert first['passed'] and first['required']
    assert calls[0][0][7] == independent.MODEL
    assert calls[0][1]['env']['HF_HUB_OFFLINE'] == '1'
    await independent.audit_part(wav, source, edits, directory)
    assert len(calls) == 1
    transcript = next(directory.rglob('transcript.json'))
    transcript.write_text('[{"text":"forged transcript"}]')
    await independent.audit_part(wav, source, edits, directory)
    assert len(calls) == 2
    observed['text'] = source + ' Why today?'
    wav.write_bytes(b'different original provider waveform')
    with pytest.raises(tts.TtsIntegrityError, match='unrequested speech'):
        await independent.audit_part(wav, source, edits, directory)
    assert len(calls) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize('observation', [
    'The report mentions DIY today. Why today?',
    'The report mentions D I Y today. Why today?',
])
async def test_independent_tail_rejects_the_primary_decoders_missed_diy_echo(audit_case, observation):
    wav, source, edits, directory, observed, _calls = audit_case
    original = wav.read_bytes()
    observed['text'] = observation
    with pytest.raises(tts.TtsIntegrityError, match='unrequested speech'):
        await independent.audit_part(wav, source, edits, directory)
    assert wav.read_bytes() == original
    report = json.loads(next(directory.rglob('audit.json')).read_text())
    assert not report['passed']
    assert not report['tail']['passed'] or not report['normalized_tail']['passed']


@pytest.mark.asyncio
@pytest.mark.parametrize(('source', 'observation', 'accepted'), [
    ('AI plans to ship.', 'AI AI plans to ship.', False),
    ('AI reports on AI.', 'AI reports on AI.', True),
    ('GPT-6 is available.', 'GPT6 is available.', True),
    ('GPT-6 is available.', 'GPT7 is available.', False),
    ('GPT-6 is available.', 'GPT seven is available.', False),
    ('GPT-6 is available.', 'GPT six is available.', True),
    ('The GPUs work.', 'The GP use work.', True),
])
async def test_independent_check_distinguishes_repetition_and_model_change_from_formatting(
        audit_case, source, observation, accepted):
    wav, _source, _edits, directory, observed, _calls = audit_case
    edits = tts_pronunciation.candidates(source)
    observed['text'] = observation
    if accepted:
        assert (await independent.audit_part(wav, source, edits, directory))['passed']
    else:
        with pytest.raises(tts.TtsIntegrityError):
            await independent.audit_part(wav, source, edits, directory)


@pytest.mark.asyncio
async def test_decoder_unavailability_preserves_the_original_wave_and_never_passes(audit_case, monkeypatch):
    wav, source, edits, directory, _observed, _calls = audit_case
    original = wav.read_bytes()
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired('independent recognizer', 180)
    monkeypatch.setattr(independent.subprocess, 'run', timeout)
    with pytest.raises(RuntimeError, match='preserving the completed job'):
        await independent.audit_part(wav, source, edits, directory)
    assert wav.read_bytes() == original and not list(directory.rglob('audit.json'))


@pytest.mark.asyncio
async def test_no_pronunciation_candidates_do_not_load_another_model(audit_case):
    wav, _source, _edits, directory, _observed, calls = audit_case
    report = await independent.audit_part(wav, 'An ordinary sentence.', [], directory)
    assert report['passed'] and not report['required'] and not calls


@pytest.mark.asyncio
async def test_empty_observation_can_recover_without_changing_the_wave(audit_case, monkeypatch):
    wav, source, edits, directory, _observed, calls = audit_case
    def decode(command, **kwargs):
        calls.append(command)
        with open(command[5], 'w') as stream:
            json.dump([] if len(calls) == 1 else [{'text': source}], stream)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(independent.subprocess, 'run', decode)
    original = wav.read_bytes()
    with pytest.raises(RuntimeError, match='preserving the completed job'):
        await independent.audit_part(wav, source, edits, directory)
    assert (await independent.audit_part(wav, source, edits, directory))['passed']
    assert len(calls) == 2 and wav.read_bytes() == original

import json
import subprocess
import wave
from types import SimpleNamespace

import pytest

from backend.pipeline import chatterbox_independent as independent, tts, tts_pronunciation


@pytest.fixture
def audit_case(tmp_path, monkeypatch):
    wav = tmp_path / 'original.wav'
    with wave.open(str(wav), 'wb') as writer:
        writer.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
        writer.writeframes(b'\0\0' * 48000)
    observed = {'text': 'The report mentions DIY today.', 'start': 0.0, 'end': 1.0}
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
    wav.write_bytes(wav.read_bytes() + b'different waveform identity')
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


@pytest.mark.parametrize('text,overlap,rejected', [
    ('To deal news. Bloomberg reports.', False, True),
    ('To deal news. Bloomberg reports.', True, False),
    ('To deal news. To deal news. Bloomberg reports.', False, False),
])
def test_interior_ordinary_phrase_repeats_need_distinct_acoustic_spans(text, overlap, rejected):
    words = [{'text': word, 'start': i * .2, 'end': (i + 1) * .2}
             for i, word in enumerate('To deal news To deal news Bloomberg reports'.split())]
    if overlap:
        for i in range(3):
            words[i + 3].update(start=words[i]['start'], end=words[i]['end'])
    assert bool(independent._repeated_source_phrases(text, words)) is rejected


@pytest.mark.asyncio
@pytest.mark.parametrize(('source', 'observation', 'accepted'), [
    ('AI plans to ship.', 'AI AI plans to ship.', False),
    ('AI reports on AI.', 'AI reports on AI.', True),
    ('GPT-6 is available.', 'GPT6 is available.', True),
    ('GPT-6 is available.', 'GPT7 is available.', False),
    ('GPT-6 is available.', 'GPT seven is available.', False),
    ('GPT-6 is available.', 'GPT six is available.', True),
    ('GPT-4 is available.', 'GPT for is available.', True),
    ('GPT-4 is available.', 'GPT five is available.', False),
    ('GPT-4 is available.', 'GPT for for is available.', False),
    ('GPT-4 is available.', 'GPT for is available. GPT for.', False),
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
async def test_ordinary_sentences_require_independent_observations_too(audit_case):
    wav, _source, _edits, directory, observed, calls = audit_case
    observed['text'] = 'An ordinary sentence.'
    report = await independent.audit_part(wav, 'An ordinary sentence.', [], directory)
    assert report['passed'] and report['required'] and len(calls) == 1
    assert report['waveform_tail']['passed']


@pytest.mark.asyncio
async def test_empty_observation_can_recover_without_changing_the_wave(audit_case, monkeypatch):
    wav, source, edits, directory, _observed, calls = audit_case
    def decode(command, **kwargs):
        calls.append(command)
        with open(command[5], 'w') as stream:
            json.dump([] if len(calls) == 1 else [{'text': source, 'start': 0.0, 'end': 1.0}], stream)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(independent.subprocess, 'run', decode)
    original = wav.read_bytes()
    with pytest.raises(RuntimeError, match='preserving the completed job'):
        await independent.audit_part(wav, source, edits, directory)
    assert (await independent.audit_part(wav, source, edits, directory))['passed']
    assert len(calls) == 2 and wav.read_bytes() == original


@pytest.mark.asyncio
@pytest.mark.parametrize('source,transcript', [
    ('To deal news.', 'to DealNews.'),
    ('The release simply erased.', 'The release simply erased.'),
])
async def test_transcript_match_or_spelling_drift_cannot_hide_untranscribed_tail(
        audit_case, source, transcript):
    wav, _source, _edits, directory, observed, _calls = audit_case
    observed.update(text=transcript, end=.8)
    with wave.open(str(wav), 'wb') as writer:
        writer.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
        writer.writeframes((3000).to_bytes(2, 'little') * 21600
                           + b'\0\0' * 16800
                           + (4000).to_bytes(2, 'little') * 9600)
    original = wav.read_bytes()
    with pytest.raises(tts.TtsIntegrityError, match='waveform_tail'):
        await independent.audit_part(wav, source, [], directory)
    report = json.loads(next(directory.rglob('audit.json')).read_text())
    assert report['tail']['passed'] and not report['waveform_tail']['passed']
    assert wav.read_bytes() == original


@pytest.mark.asyncio
@pytest.mark.parametrize('defect', [None, 'extra', 'omitted', 'weak_primary', 'disabled', 'derivative_extra'])
async def test_delivery_tail_requires_two_complete_transcripts_and_derivative_audit(audit_case, defect):
    import hashlib
    from backend.pipeline import chatterbox_tail

    wav, _source, _edits, directory, observed, calls = audit_case
    source = 'The release simply erased.'
    observed.update(text=source, end=.8)
    if defect == 'extra':
        observed['text'] += ' And.'
    if defect == 'omitted':
        observed['text'] = 'The release erased.'
    with wave.open(str(wav), 'wb') as writer:
        writer.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
        writer.writeframes((3000).to_bytes(2, 'little') * 21600
                           + b'\0\0' * 16800 + (4000).to_bytes(2, 'little') * 9600)
    original = wav.read_bytes()
    integrity = {
        'accepted': True, 'expected_words': 4, 'transcript_words': 4,
        'matched_exact_words': 4, 'exact_asr_word_coverage': 1.,
        'transcript_word_ratio': 1., 'leading_anchor': True, 'trailing_anchor': True,
        'exact_leading_anchor': True, 'exact_trailing_anchor': True,
        'repeat_start_seconds': None, 'token_differences': [],
        'speech_end_seconds': .9, 'transcription': {'passed': True},
    }
    if defect == 'weak_primary':
        integrity['exact_asr_word_coverage'] = .7
    async def audit(path, text, edits):
        if path != wav and defect == 'derivative_extra':
            observed['text'] = source + ' And.'
        return await independent.audit_part(path, text, edits, directory)

    async def run():
        return await chatterbox_tail.audit_delivery_tail(
            wav, source, [], integrity, audit, directory, enabled=defect != 'disabled')

    if defect:
        with pytest.raises(tts.TtsIntegrityError):
            await run()
        assert 'independent_tail_recovery' not in integrity
    else:
        report = await run()
        assert report['passed'] and len(calls) == 2
        assert integrity['independent_tail_recovery']['speech_end_seconds'] == .9
        metadata = {'audio_sha256': hashlib.sha256(original).hexdigest(),
                    'integrity': integrity, 'identity': {'text': source}}
        output, rows = chatterbox_tail.prepare_delivery_parts([wav], [metadata])
        assert rows[0]['audio_sha256'] == report['identity']['audio_sha256']
        assert output[0] != wav and (await audit(output[0], source, []))['passed']
        integrity['independent_tail_recovery']['cleanup']['output_audio_sha256'] = 'changed'
        with pytest.raises(RuntimeError, match='independently audited tail'):
            chatterbox_tail.prepare_delivery_parts([wav], [metadata])
    assert wav.read_bytes() == original
    assert not list(directory.glob('independent-tail-verification.wav'))

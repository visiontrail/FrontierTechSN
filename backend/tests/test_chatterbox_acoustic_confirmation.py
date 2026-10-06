import hashlib
import json
import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from backend import config
from backend.pipeline import av_sync, chatterbox, chatterbox_phonemes, tts, tts_pronunciation
from scripts.chatterbox_phoneme_evidence import MODEL, REVISION


def proof(wav, source, passed=True):
    return {'passed': passed, 'supported': True, 'method': 'complete_source_phone_sequence',
            'audio_sha256': hashlib.sha256(wav.read_bytes()).hexdigest(),
            'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
            'conditioned_on_reference_text': False}


@pytest.mark.asyncio
async def test_complete_waveform_confirmation_preserves_original_failed_asr_report(tmp_path, monkeypatch):
    wav = tmp_path / 'audio.wav'
    wav.write_bytes(b'unchanged')
    source = 'The report mentions URI today.'
    raw = 'The report mentions your eye today.'
    monkeypatch.setattr(av_sync, 'ensure_word_transcript', AsyncMock(return_value=([{'text':raw}], {})))
    confirm = AsyncMock(return_value=proof(wav, source))
    monkeypatch.setattr(chatterbox_phonemes, 'confirm', confirm)
    report = await chatterbox._verify_pronunciation(wav, source, tmp_path / 'verification',
        emit=lambda _message:None, edits=tts_pronunciation.candidates(source))
    assert report['passed'] and report['asr_spelling_passed'] is False
    assert report['transcript'] == raw and report['differences']
    assert report['verification_mode'] == 'complete_source_phone_sequence'
    assert wav.read_bytes() == b'unchanged'


@pytest.mark.asyncio
@pytest.mark.parametrize('field', ['audio_sha256','source_sha256','conditioned_on_reference_text','method','supported'])
async def test_unbound_or_reference_conditioned_confirmation_cannot_accept_speech(tmp_path, monkeypatch, field):
    wav = tmp_path / 'audio.wav';wav.write_bytes(b'unchanged')
    source='The report mentions RPC today.'
    monkeypatch.setattr(av_sync,'ensure_word_transcript',AsyncMock(return_value=([{'text':'The report mentions our PC today.'}],{})))
    result=proof(wav,source);result[field]=True if field=='conditioned_on_reference_text' else 'different'
    monkeypatch.setattr(chatterbox_phonemes,'confirm',AsyncMock(return_value=result))
    with pytest.raises(RuntimeError,match='complete source waveform'):
        await chatterbox._verify_pronunciation(wav,source,tmp_path/'verification',emit=lambda _m:None,
            edits=tts_pronunciation.candidates(source))


@pytest.mark.asyncio
async def test_observer_cannot_clear_positive_extra_tail_detection(tmp_path, monkeypatch):
    source='The report mentions AI today.'
    monkeypatch.setattr(av_sync,'ensure_word_transcript',AsyncMock(return_value=([{'text':source+' AI.'}],{})))
    observer=AsyncMock(side_effect=AssertionError('Tail rejection must not be overridden'))
    monkeypatch.setattr(chatterbox_phonemes,'confirm',observer)
    with pytest.raises(tts.TtsIntegrityError,match='trailing speech'):
        await chatterbox._verify_pronunciation(tmp_path/'audio.wav',source,tmp_path/'verification',
            emit=lambda _m:None,edits=tts_pronunciation.candidates(source))
    observer.assert_not_awaited()


@pytest.mark.asyncio
async def test_missing_primary_observation_remains_unavailable(tmp_path, monkeypatch):
    source='The report mentions AI today.'
    monkeypatch.setattr(av_sync,'ensure_word_transcript',AsyncMock(return_value=([],{})))
    observer=AsyncMock(side_effect=AssertionError('Do not replace unavailable primary observation'))
    monkeypatch.setattr(chatterbox_phonemes,'confirm',observer)
    with pytest.raises(RuntimeError,match='unavailable'):
        await chatterbox._verify_pronunciation(tmp_path/'audio.wav',source,tmp_path/'verification',
            emit=lambda _m:None,edits=tts_pronunciation.candidates(source))
    observer.assert_not_awaited()


@pytest.mark.asyncio
async def test_missing_isolated_runtime_keeps_existing_strict_policy(tmp_path, monkeypatch):
    monkeypatch.setattr(config,'CHATTERBOX_PHONEME_RUNTIME_LIB',str(tmp_path/'absent'))
    assert await chatterbox_phonemes.confirm(tmp_path/'missing.wav','Today.',[],tmp_path) is None


@pytest.mark.asyncio
@pytest.mark.parametrize('failure',['exit','timeout'])
async def test_local_observation_failure_is_uncertainty_not_a_fresh_attempt(tmp_path, monkeypatch, failure):
    runtime=tmp_path/'runtime';runtime.mkdir()
    monkeypatch.setattr(config,'CHATTERBOX_PHONEME_RUNTIME_LIB',str(runtime))
    wav=tmp_path/'audio.wav';wav.write_bytes(b'preserved')
    def fail(command,**kwargs):
        assert 'Today.' not in command and kwargs['env']['HF_HUB_OFFLINE']=='1'
        if failure=='timeout':raise subprocess.TimeoutExpired(command,180)
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(chatterbox_phonemes.subprocess,'run',fail)
    with pytest.raises(RuntimeError,match='preserving the completed job'):
        await chatterbox_phonemes.confirm(wav,'Today.',[],tmp_path/'verification')
    assert wav.read_bytes()==b'preserved'


@pytest.mark.asyncio
async def test_hash_bound_phone_observation_reuses_same_audio_without_another_model_run(tmp_path, monkeypatch):
    runtime=tmp_path/'runtime';runtime.mkdir()
    monkeypatch.setattr(config,'CHATTERBOX_PHONEME_RUNTIME_LIB',str(runtime))
    wav=tmp_path/'audio.wav';wav.write_bytes(b'preserved')
    digest=hashlib.sha256(wav.read_bytes()).hexdigest()
    directory=tmp_path/'verification';target=directory/'acoustic-phones'/digest;target.mkdir(parents=True)
    (target/'observation.json').write_text(json.dumps({'audio_sha256':digest,'model':MODEL,'model_revision':REVISION,
        'conditioned_on_reference_text':False,'phones':'t ə d eɪ'}))
    monkeypatch.setattr(chatterbox_phonemes.subprocess,'run',lambda *_a,**_k:pytest.fail('Bound observation is reusable'))
    report=await chatterbox_phonemes.confirm(wav,'Today.',[],directory)
    assert report['passed'] and report['audio_sha256']==digest
    assert wav.read_bytes()==b'preserved'

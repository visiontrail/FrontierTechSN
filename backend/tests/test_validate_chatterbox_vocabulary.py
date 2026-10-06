import copy
import json

import pytest

from backend.pipeline import tts_pronunciation as pronunciation
from scripts import validate_chatterbox_vocabulary as validation


def words(text):
    return [{'text': word} for word in text.split()]


def fixture():
    return json.loads(validation.FIXTURE.read_text())


def test_inventory_covers_legacy_terms_every_letter_and_all_observed_forms():
    observed = {'terms': {'UnknownAI': [], 'AI-driven': [], 'NASA': []}}
    rows = validation.corpus(fixture(), observed)
    terms = {r['term'] for r in rows}
    legacy = set('AI AGI API BBC CBS CEO CFO CIA CLI CNBC CNN CPU CTO EU FBI FCC FDA FT GDP GPU GPT HBM IBM IPO LLM MIT NBC NPR PC PDF RBC RTX SCP SEC TSMC TV UK UN US USB VR WSJ'.split())
    assert legacy <= terms
    assert set(observed['terms']) <= terms
    assert set('ABCDEFGHIJKLMNOPQRSTUVWXYZ') <= {c for term in terms for c in term}
    assert len({r['id'] for r in rows}) == len(rows)


def test_context_contrasts_preserve_distinct_reading_requirements():
    rows = {r['id']: r for r in validation.corpus(fixture())}
    assert rows['emphasized-it']['expected_actions']['IT'] == 'keep'
    assert rows['emphasized-us']['expected_actions']['US'] == 'keep'
    assert rows['chemical-no']['expected_actions']['NO'] == 'spell'
    assert rows['postal-in']['expected_actions']['IN'] == 'spell'
    assert rows['emphasized-in']['expected_actions']['IN'] == 'keep'


def test_sata_corpus_does_not_force_a_common_word_reading_to_be_spelled():
    case = next(row for row in validation.corpus(fixture()) if row['term'] == 'SATA')
    assert case['expected_actions']['SATA'] == ['keep', 'spell']


def test_duplicate_inventory_entries_are_rejected():
    value = copy.deepcopy(fixture())
    value['contextual_readings'] += ' AI'
    with pytest.raises(ValueError, match='Duplicate'):
        validation.corpus(value)


@pytest.mark.parametrize(('source', 'observed'), [
    ('The API report is ready.', 'The A P I report is ready.'),
    ('The OpenAI report is ready.', 'The Open AI report is ready.'),
    ('The NASA report is ready.', 'The NASA report is ready.'),
    ('The WiFi report is ready.', 'The Wi Fi report is ready.'),
    ('The PhD report is ready.', 'The PhD report is ready.'),
    ('The B2B report is ready.', 'The B two B report is ready.'),
    ('The FP16 report is ready.', 'The FP sixteen report is ready.'),
    ('The GPT4o report is ready.', 'The GPT four o report is ready.'),
    ('The TLS1.3 report is ready.', 'The TLS 1.3 report is ready.'),
    ('The AI R&D team is ready.', 'The AIR &D team is ready.'),
    ('The SQL database is ready.', 'The sequel database is ready.'),
    ('The SaaS service is ready.', 'The sass service is ready.'),
    ('The study reaches Phase III.', 'The study reaches Phase three.'),
    ('The C++ example is ready.', 'The C plus plus example is ready.'),
    ('The C# example is ready.', 'The C sharp example is ready.'),
    ('The AI-driven report is ready.', 'The AI driven report is ready.'),
    ('The GPT-Image-2 report is ready.', 'The GPT Image 2 report is ready.'),
])
def test_audit_accepts_only_preserved_letter_and_value_representations(source, observed):
    assert validation.all_target_report(source, words(observed))['passed']


@pytest.mark.parametrize(('source', 'observed'), [
    ('The API report is ready.', 'The A I report is ready.'),
    ('The OpenAI report is ready.', 'The Open I I report is ready.'),
    ('The OpenAI report is ready.', 'The Open AI AI report is ready.'),
    ('The NASA report is ready.', 'The NSA report is ready.'),
    ('The NASA report is ready.', 'The NASA report is ready with NASA.'),
    ('The WiFi report is ready.', 'The Wi report is ready.'),
    ('The PhD report is ready.', 'The Ph report is ready.'),
    ('The B2B report is ready.', 'The B three B report is ready.'),
    ('The FP16 report is ready.', 'The FP sixty report is ready.'),
    ('The GPT4o report is ready.', 'The GPT four report is ready.'),
    ('The TLS1.3 report is ready.', 'The TLS 1.2 report is ready.'),
    ('The AI R&D team is ready.', 'The AI RAN team is ready.'),
    ('The SQL report is ready.', 'The sequel report is ready then sequel.'),
    ('The SaaS report is ready.', 'The sass report is ready then sass.'),
    ('The study reaches Phase III.', 'The study reaches Phase two.'),
    ('The C++ example is ready.', 'The C plus example is ready.'),
    ('The C# example is ready.', 'The C example is ready.'),
    ('The AI-driven report is ready.', 'The AI draft report is ready.'),
    ('The GPT-Image-2 report is ready.', 'The GPT two report is ready.'),
])
def test_audit_rejects_missing_repeated_or_changed_target_content(source, observed):
    assert not validation.all_target_report(source, words(observed))['passed']


def test_plan_mode_validation_checks_each_occurrence_in_its_own_context():
    rows = [
        {'id': 'department', 'text': 'The IT department is ready.', 'expected_actions': {'IT': 'spell'}},
        {'id': 'emphasis', 'text': 'The sign reads DO IT.', 'expected_actions': {'DO': 'keep', 'IT': 'keep'}},
    ]
    source = '\n'.join(r['text'] for r in rows)
    decisions = [{**c, 'action': 'keep' if c['id'] else 'spell'} for c in pronunciation.candidates(source)]
    assert validation.mode_report(source, rows, {'decisions': decisions})['passed']
    decisions[-1]['action'] = 'spell'
    report = validation.mode_report(source, rows, {'decisions': decisions})
    assert not report['passed']
    assert report['failures'][0]['case_id'] == 'emphasis'


def test_frozen_corpus_cannot_rebind_submitted_provider_jobs(tmp_path):
    value = fixture()
    saved = validation.prepare(tmp_path / 'run', value, None)
    assert validation.prepare(tmp_path / 'run', value, None) == saved
    value['contextual_readings'] += ' FOO'
    with pytest.raises(ValueError, match='Frozen corpus'):
        validation.prepare(tmp_path / 'run', value, None)


@pytest.mark.parametrize('field', ['batch_count','case_count','term_count','corpus_sha256','uppercase_letters'])
def test_resumed_manifest_cannot_claim_full_coverage_with_corrupt_counts(tmp_path, field):
    directory = tmp_path / 'run'
    saved = validation.prepare(directory, fixture(), None)
    saved[field] = 1 if isinstance(saved[field], int) else 'incorrect'
    validation.write(directory / 'corpus.json', saved)
    with pytest.raises(ValueError, match='coverage/hash metadata'):
        validation.prepare(directory, fixture(), None)


@pytest.mark.asyncio
async def test_mode_rejection_does_not_submit_audio_or_claim_acoustic_completion(tmp_path, monkeypatch):
    case = {'id': 'ai', 'term': 'AI', 'text': 'The AI report is ready.', 'expected_actions': {'AI': 'spell'}}
    async def idle(_url):
        return {}
    async def plan(text, _directory, **_kwargs):
        return pronunciation._plan(text, [{'id': 0, 'source': 'AI', 'action': 'keep',
                                          'spoken': 'AI', 'reason': 'Diagnostic mismatch'}], {})
    async def unexpected_audio(*_args, **_kwargs):
        pytest.fail('A wrong mode must stop before submitting audio')
    monkeypatch.setattr(validation, 'require_idle', idle)
    monkeypatch.setattr(pronunciation, 'prepare_pronunciation', plan)
    monkeypatch.setattr(validation.chatterbox, '_generate_chatterbox', unexpected_audio)
    marker = {'identity': {'cases': [case], 'cache_contract': {'pronunciation_policy': {}}},
              'batch_count': 1, 'case_count': 1, 'term_count': 1, 'corpus_sha256': 'test'}
    await validation.run(tmp_path, marker)
    summary = json.loads((tmp_path / 'summary.json').read_text())
    assert summary['all_batches_attempted']
    assert not summary['all_batches_audited']
    assert not summary['passed']


@pytest.mark.asyncio
async def test_observation_timeout_keeps_current_job_and_stops_before_next_batch(tmp_path, monkeypatch):
    case = {'id': 'ai', 'term': 'AI', 'text': 'The AI report is ready.', 'expected_actions': {'AI': 'spell'}}
    async def idle(_url):
        return {}
    async def plan(text, _directory, **_kwargs):
        return pronunciation._plan(text, [{'id': 0, 'source': 'AI', 'action': 'spell',
                                          'spoken': 'eigh eye', 'reason': 'Letter reading'}], {})
    async def uncertain_audio(_script, directory, *_args, **_kwargs):
        checkpoint = {'job_id': 'owned-live-job', 'idempotency_key': 'preserved-key'}
        validation.write(validation.Path(directory) / 'tts_input_part_001.request-0.json', checkpoint)
        raise TimeoutError('Observation expired; service job may still be running')
    monkeypatch.setattr(validation, 'require_idle', idle)
    monkeypatch.setattr(pronunciation, 'prepare_pronunciation', plan)
    monkeypatch.setattr(validation.chatterbox, '_generate_chatterbox', uncertain_audio)
    marker = {'identity': {'cases': [case], 'cache_contract': {'pronunciation_policy': {}}},
              'batch_count': 2, 'case_count': 1, 'term_count': 1, 'corpus_sha256': 'test'}
    with pytest.raises(TimeoutError):
        await validation.run(tmp_path, marker)
    checkpoint = json.loads((tmp_path / 'batch-001/audio/tts_input_part_001.request-0.json').read_text())
    assert checkpoint == {'job_id': 'owned-live-job', 'idempotency_key': 'preserved-key'}
    assert not (tmp_path / 'batch-002').exists()


@pytest.mark.asyncio
@pytest.mark.parametrize('observed', [
    'The report mentions OCR today. Art.',
    'The report mentions A are today, are today.',
])
async def test_independent_part_audit_rejects_raw_and_normalized_extra_speech(tmp_path, monkeypatch, observed):
    from types import SimpleNamespace
    from backend.pipeline import tts
    source = 'The report mentions AR today.' if 'A are' in observed else 'The report mentions OCR today.'
    wav = tmp_path / 'part.wav'
    wav.write_bytes(b'unchanged-waveform')
    def transcribe(command, **_kwargs):
        validation.write(validation.Path(command[5]), words(observed))
        return SimpleNamespace(returncode=0, stderr=b'')
    monkeypatch.setattr(validation.subprocess, 'run', transcribe)
    with pytest.raises(tts.TtsIntegrityError, match='Independent vocabulary audit'):
        await validation.independent_part_audit(wav, source, pronunciation.candidates(source),
            directory=tmp_path, plan={'plan_sha256': 'frozen-plan'})
    report = json.loads(next((tmp_path / 'independent-parts').glob('*/*/audit.json')).read_text())
    assert not report['passed']
    assert report['transcript_sha256']
    assert wav.read_bytes() == b'unchanged-waveform'


@pytest.mark.asyncio
async def test_independent_part_audit_retains_unavailable_observation(tmp_path, monkeypatch):
    from types import SimpleNamespace
    wav = tmp_path / 'part.wav'
    wav.write_bytes(b'preserved-waveform')
    monkeypatch.setattr(validation.subprocess, 'run', lambda *_args, **_kwargs:
        SimpleNamespace(returncode=1, stderr=b'offline model unavailable'))
    with pytest.raises(RuntimeError, match='recognizer unavailable'):
        await validation.independent_part_audit(wav, 'The AI report is ready.', [],
            directory=tmp_path, plan={'plan_sha256': 'plan'})
    assert wav.read_bytes() == b'preserved-waveform'
    assert not list((tmp_path / 'independent-parts').glob('*/*/identity.json'))


@pytest.mark.asyncio
@pytest.mark.parametrize(('key','wrong'), [
    ('text', 'A different AI sentence.'), ('synthesis_text', 'eye eye'),
    ('voice', 'different-voice'), ('pronunciation_plan_sha256', 'different-plan'),
])
async def test_final_vocabulary_audit_compares_identity_with_source_not_itself(tmp_path, key, wrong):
    audio = tmp_path / 'audio'
    audio.mkdir()
    source = 'The AI report is ready.'
    part = audio / 'tts_input_part_001.txt'
    part.write_text(source)
    wav = audio / 'tts_input_part_001_generated.wav'
    wav.write_bytes(b'preserved')
    plan = {'edits': [], 'plan_sha256': 'frozen-plan'}
    identity = {'text': source, 'synthesis_text': source, 'voice': 'space-en-f1',
                'language': 'en', 'pronunciation_plan_sha256': 'frozen-plan', 'synthesis_speed_ratio': 1.0}
    identity[key] = wrong
    validation.write(validation.tts._part_metadata_path(wav), {'identity':identity})
    with pytest.raises(ValueError, match='frozen source'):
        await validation.audit_audio(tmp_path, source, plan)

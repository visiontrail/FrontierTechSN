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

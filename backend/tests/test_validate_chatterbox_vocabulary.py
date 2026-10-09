import copy
import json

import pytest

from backend.pipeline import tts_pronunciation as pronunciation
from scripts import validate_chatterbox_vocabulary as validation


def words(text):
    return [{'text': word} for word in text.split()]


def test_cited_phonetic_word_cue_keeps_word_mode_and_cannot_satisfy_letter_mode():
    source = 'The GUI displays the window.'
    row = pronunciation.candidates(source, policy_version=16)[0]
    decisions = [{**row, 'action':'sound', 'spoken':'gooey', 'phones':['G','UW','IY'],
                  'reason':'Source-cited conventional word reading.'}]
    identity = {'source_sha256':validation.hashlib.sha256(source.encode()).hexdigest(),
                'language':'en', 'policy':pronunciation.policy_contract(policy_version=16)}
    plan = pronunciation._plan(source, decisions, identity)
    case = {'id':'case-gui','text':source,'expected_actions':{'GUI':'keep'}}
    assert validation.mode_report(source, [case], plan)['passed']
    assert not validation.mode_report(source, [{**case,'expected_actions':{'GUI':'spell'}}], plan)['passed']


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


def test_medical_pet_reference_uses_its_conventional_word_reading_without_changing_source():
    current_fixture = fixture()
    previous_fixture = copy.deepcopy(current_fixture)
    previous_fixture['mode_overrides'].pop('PET')
    previous_fixture['reference_readings'] = [row for row in previous_fixture['reference_readings'] if row['term'] != 'PET']
    current = validation.corpus(current_fixture)
    previous = validation.corpus(previous_fixture)
    assert [(row['id'], row['term'], row['text'], row['category']) for row in current] == [
        (row['id'], row['term'], row['text'], row['category']) for row in previous]
    changed = [(old, new) for old, new in zip(previous, current, strict=True) if old != new]
    assert len(changed) == 1 and changed[0][0]['term'] == 'PET'
    assert changed[0][0]['expected_actions'] == {'PET': 'spell'}
    assert changed[0][1]['expected_actions'] == {'PET': 'keep'}
    assert next(row for row in current_fixture['reference_readings'] if row['term'] == 'PET')['source'] == (
        'https://dictionary.cambridge.org/pronunciation/english/pet-scan')


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
    ('The TLS1.3 report is ready.', 'The TLS one point three report is ready.'),
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


@pytest.mark.parametrize('previous_version', [1, 2])
def test_tail_gate_upgrade_preserves_frozen_corpus_and_rejects_other_provider_changes(tmp_path, monkeypatch, previous_version):
    current = validation.chatterbox.cache_contract()
    previous = {**current, 'trailing_speech_verifier_version': previous_version}
    monkeypatch.setattr(validation.chatterbox, 'cache_contract', lambda: previous)
    directory = tmp_path / 'run'
    saved = validation.prepare(directory, fixture(), None)
    marker_bytes = (directory / 'corpus.json').read_bytes()
    monkeypatch.setattr(validation.chatterbox, 'cache_contract', lambda: current)
    assert validation.prepare(directory, fixture(), None) == saved
    assert (directory / 'corpus.json').read_bytes() == marker_bytes
    monkeypatch.setattr(validation.chatterbox, 'cache_contract', lambda: {**current, 'seed': current['seed'] + 1})
    with pytest.raises(ValueError, match='Frozen corpus'):
        validation.prepare(directory, fixture(), None)


@pytest.mark.asyncio
@pytest.mark.parametrize('selected', [[], [0], [3], [1, 1], [True], ['1'], (1,)])
async def test_batch_resume_selection_rejects_invalid_indices_before_any_remote_work(tmp_path, monkeypatch, selected):
    async def unexpected(*_args, **_kwargs):
        pytest.fail('Invalid batch selection must stop before remote work')
    monkeypatch.setattr(validation, 'require_audio_window', unexpected)
    marker = {'identity': {'cases': []}, 'batch_count': 2}
    with pytest.raises(ValueError, match='unique existing one-based'):
        await validation.run(tmp_path, marker, batch_indices=selected)


@pytest.mark.asyncio
async def test_resuming_one_batch_preserves_full_scope_and_never_claims_full_completion(tmp_path, monkeypatch):
    cases = [{'id': f'case-{index}', 'term': 'AI', 'text': 'The AI report is ready.',
              'expected_actions': {'AI': 'spell'}} for index in range(24)]
    marker = {'identity': {'cases': cases, 'cache_contract': {'pronunciation_policy': {}}},
              'batch_count': 2, 'case_count': 24, 'term_count': 1, 'corpus_sha256': 'full-frozen-scope'}
    original = copy.deepcopy(marker)
    planned = []

    async def window(*_args, **_kwargs):
        return {'available': True}

    async def plan(text, directory, **_kwargs):
        planned.append(directory)
        rows = [{**candidate, 'action': 'keep', 'spoken': candidate['source'],
                 'reason': 'A controlled mode mismatch must stop before synthesis.'}
                for candidate in pronunciation.candidates(text)]
        return pronunciation._plan(text, rows, {})

    async def unexpected(*_args, **_kwargs):
        pytest.fail('A wrong mode must not submit any audio')

    monkeypatch.setattr(validation, 'require_audio_window', window)
    monkeypatch.setattr(pronunciation, 'prepare_pronunciation', plan)
    monkeypatch.setattr(validation.chatterbox, '_generate_chatterbox', unexpected)
    await validation.run(tmp_path, marker, batch_indices=[2])
    assert marker == original
    assert planned == [tmp_path / 'batch-002/audio']
    assert not (tmp_path / 'batch-001').exists()
    summary = validation.read(tmp_path / 'summary.json')
    assert summary['case_count'] == 24 and summary['corpus_sha256'] == 'full-frozen-scope'
    assert summary['total_batches'] == 2 and summary['selected_batches'] == [2]
    assert not summary['passed'] and not summary['all_batches_attempted'] and not summary['all_batches_audited']
    assert summary['outcomes'][0]['case_ids'] == [case['id'] for case in cases[12:]]


def test_old_corpus_resumes_with_delivery_cleanup_disabled_and_still_rejects_generation_drift(tmp_path, monkeypatch):
    current = validation.chatterbox.cache_contract()
    old = {key: value for key, value in current.items() if key != 'tail_cleanup_version'}
    monkeypatch.setattr(validation.chatterbox, 'cache_contract', lambda: old)
    saved = validation.prepare(tmp_path / 'run', fixture(), None)
    monkeypatch.setattr(validation.chatterbox, 'cache_contract', lambda: current)
    assert validation.prepare(tmp_path / 'run', fixture(), None) == saved
    changed = {**current, 'seed': current['seed'] + 1}
    monkeypatch.setattr(validation.chatterbox, 'cache_contract', lambda: changed)
    with pytest.raises(ValueError, match='Frozen corpus'):
        validation.prepare(tmp_path / 'run', fixture(), None)


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
    monkeypatch.setattr(validation, 'require_validation_source', lambda: {'commit': 'isolated-test-source'})
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
    monkeypatch.setattr(validation, 'require_validation_source', lambda: {'commit': 'isolated-test-source'})
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


def completed_composition_task(tmp_path):
    import hashlib,struct,wave
    audio = tmp_path / 'audio'
    audio.mkdir()
    wav = audio / 'tts_input_generated.wav'
    with wave.open(str(wav), 'wb') as stream:
        stream.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
        stream.writeframes(struct.pack('<h',1000) * 24000)
    manifest = {'output_audio_sha256':hashlib.sha256(wav.read_bytes()).hexdigest(),
                'source_word_count':5, 'chunk_count':1, 'parts':[{}],
                'integrity':{'required':True,'passed':True,'accepted_source_coverage':1.0,'accepted_source_words':5}}
    validation.write(audio / 'tts_manifest.json',manifest)
    return {'id':'production','status':'composing','audio_path':str(wav)},manifest


def test_composition_window_requires_hash_bound_finished_voice(tmp_path):
    task,_manifest = completed_composition_task(tmp_path)
    result = validation.composition_audio_window([task], {'active':{'id':'production'},'items':[]})
    assert result['available']
    assert result['production_audio_anchors'][0]['audio_sha256'] == validation.sha(validation.Path(task['audio_path']))


@pytest.mark.parametrize('status',['tts','writing','review','queued','music','publishing'])
def test_composition_window_never_overlaps_unfinished_production_speech(tmp_path,status):
    task,_manifest = completed_composition_task(tmp_path)
    task['status'] = status
    with pytest.raises(RuntimeError):
        validation.composition_audio_window([task], {'active':{'id':'production'},'items':[]})


@pytest.mark.parametrize('defect',['failed','preview','missing_coverage','changed_wave','changed_word_count','missing_part'])
def test_composition_window_rejects_incomplete_or_changed_narration(tmp_path,defect):
    task,manifest = completed_composition_task(tmp_path)
    wav = validation.Path(task['audio_path'])
    if defect == 'failed':manifest['integrity']['passed'] = False
    elif defect == 'preview':manifest['integrity']['required'] = False
    elif defect == 'missing_coverage':manifest['integrity'].pop('accepted_source_coverage')
    elif defect == 'changed_wave':wav.write_bytes(b'changed')
    elif defect == 'changed_word_count':manifest['integrity']['accepted_source_words'] = 4
    elif defect == 'missing_part':manifest['parts'] = []
    validation.write(wav.parent / 'tts_manifest.json',manifest)
    with pytest.raises(RuntimeError):
        validation.composition_audio_window([task], {'active':{'id':'production'},'items':[]})


@pytest.mark.parametrize('queue',[
    {'active':{'id':'unidentified-explainer'},'items':[]},
    {'active':None,'items':[{'id':'next-production'}]},
])
def test_composition_window_rejects_other_queue_work(tmp_path,queue):
    task,_manifest = completed_composition_task(tmp_path)
    with pytest.raises(RuntimeError):
        validation.composition_audio_window([task],queue)


def test_legacy_corpus_preserves_case_order_and_original_candidate_contract(tmp_path,monkeypatch):
    value=fixture();current=validation.chatterbox.cache_contract()
    legacy=pronunciation.policy_contract(policy_version=3)
    frozen={**current,'pronunciation_policy':legacy}
    monkeypatch.setattr(validation.chatterbox,'cache_contract',lambda:frozen)
    directory=tmp_path/'run';original=validation.prepare(directory,value,None)
    raw=(directory/'corpus.json').read_bytes()
    monkeypatch.setattr(validation.chatterbox,'cache_contract',lambda:current)
    resumed=validation.prepare(directory,value,None)
    assert resumed==original and (directory/'corpus.json').read_bytes()==raw


def test_legacy_corpus_cannot_hide_a_changed_synthesis_contract(tmp_path,monkeypatch):
    value=fixture();current=validation.chatterbox.cache_contract()
    legacy=pronunciation.policy_contract(policy_version=3)
    frozen={**current,'pronunciation_policy':legacy}
    monkeypatch.setattr(validation.chatterbox,'cache_contract',lambda:frozen)
    directory=tmp_path/'run';validation.prepare(directory,value,None)
    monkeypatch.setattr(validation.chatterbox,'cache_contract',lambda:{**current,'seed':current['seed']+1})
    with pytest.raises(ValueError,match='Frozen corpus'):
        validation.prepare(directory,value,None)


@pytest.mark.parametrize('drift', ['commit', 'backend_sha256', 'reference', 'uncommitted'])
def test_validator_source_drift_stops_before_an_audio_window_or_new_request(tmp_path, monkeypatch, drift):
    loaded = {'commit': 'loaded-commit', 'backend_sha256': 'loaded-backend', 'loaded_at': 'loaded-time'}
    current = loaded.copy()
    path = tmp_path / 'reference.json'
    path.write_text('original')
    monkeypatch.setattr(validation, 'ROOT', tmp_path)
    monkeypatch.setattr(validation, 'RUNTIME_IDENTITY', loaded)
    monkeypatch.setattr(validation, 'VALIDATION_SOURCE_HASHES', {'reference.json': validation.sha(path)})
    monkeypatch.setattr(validation, '_runtime_identity', lambda: current)
    monkeypatch.setattr(validation.subprocess, 'check_output', lambda *_args, **_kwargs: (
        b' M backend/pipeline/tts_pronunciation.py\0' if drift == 'uncommitted' else b''))
    if drift in current:
        current[drift] = 'changed'
    elif drift == 'reference':
        path.write_text('changed')
    with pytest.raises(RuntimeError, match='Local validation'):
        validation.require_validation_source()


@pytest.mark.asyncio
async def test_finished_narration_window_keeps_web_and_private_validator_versions_distinct(tmp_path, monkeypatch):
    import httpx
    task, _manifest = completed_composition_task(tmp_path)
    application = {'commit': 'older-web-service', 'backend_sha256': 'web-bytes'}
    validator = {'commit': 'committed-validator', 'backend_sha256': 'validator-bytes'}
    monkeypatch.setattr(validation, 'require_validation_source', lambda: validator)
    payloads = {'/api/tasks': {'tasks': [task]}, '/api/explainers/queue': {
        'active': {'id': task['id']}, 'items': []}, '/api/runtime': application}
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=payloads[request.url.path]))
    client = httpx.AsyncClient
    monkeypatch.setattr(httpx, 'AsyncClient', lambda **kwargs: client(transport=transport, **kwargs))
    report = await validation.require_audio_window('http://example.invalid', allow_composing=True)
    assert report['available']
    assert report['application_runtime'] == application
    assert report['validation_runtime'] == validator
    assert report['production_audio_anchors'][0]['audio_sha256'] == validation.sha(validation.Path(task['audio_path']))


@pytest.mark.asyncio
async def test_final_audio_audit_ignores_delivery_sidecars_without_omitting_canonical_parts(tmp_path, monkeypatch):
    texts = ['The AI report is ready.', 'The GPU report is ready.']
    source = '\n'.join(texts)
    plan = {'edits': [], 'plan_sha256': 'frozen-plan'}
    audio = tmp_path / 'audio'
    audio.mkdir()
    monkeypatch.setattr(validation.chatterbox, '_cached_part', lambda *_args, **_kwargs: True)
    for index, text in enumerate(texts, 1):
        part = audio / f'tts_input_part_{index:03d}.txt'
        part.write_text(text)
        wav = part.with_name(part.stem + '_generated.wav')
        wav.write_bytes(text.encode())
        identity = {'text': text, 'synthesis_text': text, 'voice': 'space-en-f1', 'language': 'en',
                    'pronunciation_plan_sha256': plan['plan_sha256'], 'synthesis_speed_ratio': 1.0}
        validation.write(validation.tts._part_metadata_path(wav), {
            'identity': identity, 'synthesis_request': {'text': text}})
        validation.write(audio / 'verification' / part.stem / 'transcript.json', words(text))
        target = tmp_path / 'independent' / part.stem
        validation.write(target / 'transcript.json', words(text))
        validation.write(target / 'identity.json', {'audio_sha256': validation.sha(wav),
            'source_sha256': validation.sha(part), 'plan_sha256': plan['plan_sha256'],
            'model': validation.SECONDARY_MODEL})
        part.with_name(part.stem + '_tail_12345678.txt').write_text(text)
    (audio / 'tts_input_part_001_diagnostic.txt').write_text('An unrelated diagnostic.')
    result = await validation.audit_audio(tmp_path, source, plan)
    assert result['passed']
    assert [row['part'] for row in result['parts']] == ['tts_input_part_001.txt', 'tts_input_part_002.txt']
    (audio / 'tts_input_part_002.txt').write_text('Changed canonical source.')
    with pytest.raises(ValueError):
        await validation.audit_audio(tmp_path, source, plan)

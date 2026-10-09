import hashlib

import pytest

from backend.pipeline import tts_pronunciation as pronunciation
from backend.pipeline import tts_word_pronunciation as word
from scripts import chatterbox_phoneme_evidence as evidence


@pytest.mark.parametrize(('source', 'cue', 'phones'), [
    ('EBITDA', 'ebb it duh', ['EH', 'B', 'IH', 'T', 'D', 'AH']),
    ('CUDA', 'coo duh', ['K', 'UW', 'D', 'AH']),
    ('ADAS', 'aid ass', ['EY', 'D', 'AE', 'S']),
])
def test_shared_cited_word_reference_authorizes_only_matching_opt_in_cues(source, cue, phones):
    assert word.validate_homophone_hint(source, cue, phones, version=8) == phones
    with pytest.raises(ValueError):
        word.validate_homophone_hint(source, cue, phones, version=7)
    with pytest.raises(ValueError):
        word.validate_homophone_hint(source, cue + ' today', phones, version=8)
    assert tuple(phones) in evidence.independent_word_reference(source, word_hint_version=8)['variants']


def test_shared_source_reference_preserves_context_and_replaces_unrelated_name():
    with pytest.raises(ValueError):
        word.validate_homophone_hint('ADAS', 'adas,', ['EY', 'D', 'AH', 'Z'], version=8)
    with pytest.raises(ValueError):
        word.validate_homophone_hint('cuda', 'coo duh', ['K', 'UW', 'D', 'AH'], version=8)
    assert word.validate_word_hint('CUDA', 'cuda,', ['K', 'UW', 'D', 'AH'], version=8) == ['K', 'UW', 'D', 'AH']
    assert word.policy_contract(version=8)['cited_body_word_references'] == evidence.body_reference_contract()


def test_storage_nas_word_cue_uses_opt_in_cited_reference_without_spelling_fallback():
    phones = ['N', 'AE', 'S']
    assert word.validate_homophone_hint('NAS', 'nass', phones, version=7) == phones
    for version in [5, 6]:
        with pytest.raises(ValueError, match='independent complete-source'):
            word.validate_homophone_hint('NAS', 'nass', phones, version=version)
    for cue, wrong in [('nays', ['N', 'EY', 'Z']), ('nass today', phones),
                       ('nass', ['N', 'AH', 'S']), ('nass', ['N', 'AE'])]:
        with pytest.raises(ValueError):
            word.validate_homophone_hint('NAS', cue, wrong, version=7)
    assert word.policy_contract(version=6)['supplemental_word_references']['version'] == 3
    assert word.policy_contract(version=7)['supplemental_word_references']['version'] == 4


@pytest.mark.parametrize(('name', 'phones', 'wrong'), [
    ('PaaS', ['P', 'AE', 'S'], ['P', 'AA', 'Z']),
    ('IaaS', ['AY', 'AE', 'Z'], ['IY', 'AH', 'Z']),
])
def test_cited_cloud_reference_is_opt_in_and_rejects_native_wrong_vowels(name, phones, wrong):
    assert word.validate_word_hint(name, name+',', phones, version=5) == phones
    with pytest.raises(ValueError):
        word.validate_word_hint(name, name+',', phones, version=4)
    with pytest.raises(ValueError):
        word.validate_word_hint(name, name+',', wrong, version=5)
    assert word.policy_contract(version=4)['supplemental_word_references']['version'] == 1
    assert word.policy_contract(version=5)['supplemental_word_references']['version'] == 2


@pytest.mark.parametrize(('source', 'cue', 'phones'), [
    ('PaaS', 'pass', ['P', 'AE', 'S']),
    ('IaaS', 'eye as', ['AY', 'AE', 'Z']),
    ('GUI', 'gooey', ['G', 'UW', 'IY']),
])
def test_homophone_cue_requires_independently_known_source_and_exact_dictionary_cue(source, cue, phones):
    assert word.validate_homophone_hint(source, cue, phones) == phones
    with pytest.raises(ValueError):
        word.validate_homophone_hint(source, cue, phones, version=4)
    with pytest.raises(ValueError):
        word.validate_homophone_hint(source, cue+' today', phones)


@pytest.mark.parametrize('cue',['post gress cue ell','post-gress-queue-ell'])
def test_version6_postgresql_official_cue_matches_every_source_phone(cue):
    phones=['P','OW','S','T','G','R','EH','S','K','Y','UW','EH','L']
    assert word.validate_homophone_hint('PostgreSQL',cue,phones,version=6)==phones
    with pytest.raises(ValueError):word.validate_homophone_hint('PostgreSQL',cue,phones,version=5)
    with pytest.raises(ValueError):word.validate_homophone_hint('PostgreSQL',cue+' ess',phones,version=6)
    assert word.policy_contract(version=5)['supplemental_word_references']['version']==2
    assert word.policy_contract(version=6)['supplemental_word_references']['version']==3


@pytest.mark.parametrize(('source', 'cue', 'phones'), [
    ('PaaS', 'paz', ['P', 'AA', 'Z']),
    ('PaaS', 'zzunknown', ['P', 'AE', 'S']),
    ('IaaS', 'eye eye as', ['AY', 'AE', 'Z']),
    ('ZZUnknown', 'pass', ['P', 'AE', 'S']),
    ('GUI', 'goo', ['G', 'UW', 'IY']),
])
def test_homophone_cue_cannot_prove_unknown_source_or_change_its_complete_phones(source, cue, phones):
    with pytest.raises(ValueError):
        word.validate_homophone_hint(source, cue, phones)


def word_row(**changes):
    return {'id': 0, 'source': 'SATA', 'action': 'word', 'spoken': 'sat, a',
            'phones': ['S', 'AE', 'T', 'AH'], 'reason': 'Keep the conventional word reading.', **changes}


def test_word_hint_preserves_source_and_numeric_suffix_and_binds_phones():
    source = 'The SATA3 interface is ready.'
    rows, edits = pronunciation.validate_decisions(source, [word_row()])
    assert source == 'The SATA3 interface is ready.'
    assert pronunciation.render_span(source, 0, len(source), edits) == 'The sat, a 3 interface is ready.'
    assert rows[0]['phones'] == ['S', 'AE', 'T', 'AH']
    assert edits[0]['kind'] == 'word'
    assert pronunciation.validate_decisions(source, rows) == (rows, edits)


@pytest.mark.parametrize('changes', [
    {'spoken': 'sat uh'}, {'spoken': 'sat'}, {'spoken': 'sat a a'},
    {'spoken': 's a t a'}, {'spoken': 'sat/a'}, {'spoken': 'sat <break> a'},
    {'phones': ['S', 'AA', 'T', 'AH']}, {'phones': ['S', 'AE', 'T']},
    {'phones': ['S', 'AE', 'T', 'AH', 'AH']}, {'phones': ['s', 'ae', 't', 'ah']},
    {'phones': ['S', 'AE1', 'T', 'AH0']}, {'phones': None},
])
def test_word_hint_rejects_changed_spelling_unbound_phones_and_instructions(changes):
    with pytest.raises(ValueError):
        pronunciation.validate_decisions('SATA today.', [word_row(**changes)])


def test_known_word_cannot_turn_into_a_sequence_of_letter_names():
    with pytest.raises(ValueError, match='source word'):
        word.validate_word_hint('STOP', 's top', ['EH', 'S', 'T', 'AA', 'P'])


@pytest.mark.parametrize('version', [3, 4, 5, 6])
def test_old_policies_reject_new_word_action_and_keep_original_contract(version):
    with pytest.raises(ValueError, match='supported word'):
        pronunciation.validate_decisions('SATA today.', [word_row()], policy_version=version)
    assert pronunciation.compatible_policy(pronunciation.policy_contract(policy_version=version))


def test_word_policy_freezes_the_actual_dictionary_bytes():
    with word.cmudict.dict_stream() as stream:
        expected = hashlib.sha256(stream.read()).hexdigest()
    assert pronunciation.policy_contract()['word_hint_contract']['dictionary_sha256'] == expected


def test_mixed_case_word_hint_retains_its_prefix_letter_and_word_reading():
    source = 'The report mentions eSIM today.'
    row = word_row(source='eSIM', spoken='e, sim', phones=['IY', 'S', 'IH', 'M'])
    _, edits = pronunciation.validate_decisions(source, [row])
    assert pronunciation.render_span(source, 0, len(source), edits) == 'The report mentions e, sim today.'
    assert source == 'The report mentions eSIM today.'
    phones = 'ð ə ɹ ᵻ p ɔːɹ t m ɛ n ʃ ə n z iː s ɪ m t ə d eɪ'
    assert evidence.compare_source(source, edits, phones)['passed']
    assert not evidence.compare_source(source, edits, phones.replace('iː s ɪ m', 's ɪ m'))['passed']
    assert not evidence.compare_source(source, edits, phones.replace('iː s ɪ m', 'iː ɛ s aɪ ɛ m'))['passed']
    assert pronunciation.policy_contract()['word_hint_contract']['version'] == 4


def test_cited_complete_word_hint_retains_source_phones_and_pins_reference_bytes():
    from backend.pipeline import tts_phone_references as references
    source = 'The GUI displays the application window.'
    row = word_row(source='GUI', spoken='gui,', phones=['G', 'UW', 'IY'])
    _, edits = pronunciation.validate_decisions(source, [row])
    assert pronunciation.render_span(source, 0, len(source), edits) == 'The gui, displays the application window.'
    contract = pronunciation.policy_contract()['word_hint_contract']
    assert contract['supplemental_word_references'] == references.policy_contract()
    assert contract['supplemental_word_references']['reference_sha256'] == hashlib.sha256(references.WORD_REFERENCES.read_bytes()).hexdigest()
    correct = 'ð ə ɡ uː iː d ɪ s p l eɪ z ð ə æ p l ə k eɪ ʃ ə n w ɪ n d oʊ'
    assert evidence.compare_source(source, edits, correct)['passed']
    assert not evidence.compare_source(source, edits, correct.replace('uː iː', 'iː'))['passed']
    assert not evidence.compare_source(source, edits, correct + ' iː')['passed']


@pytest.mark.parametrize('version', [10, 11, 12])
def test_cited_single_word_hint_does_not_rebind_previous_word_policies(version):
    contract = pronunciation.policy_contract(policy_version=version)['word_hint_contract']
    assert contract['version'] == (3 if version == 12 else 2)
    assert 'supplemental_word_references' not in contract
    with pytest.raises(ValueError):
        pronunciation.validate_decisions('GUI today.', [word_row(
            source='GUI', spoken='gui,', phones=['G', 'UW', 'IY'])], policy_version=version)
    assert pronunciation.compatible_policy(pronunciation.policy_contract(policy_version=version))


@pytest.mark.parametrize(('spoken', 'phones'), [
    ('gui,', ['G', 'IY']), ('gui,', ['G', 'AY']), ('gui,', ['G', 'UW', 'IY', 'IY']),
    ('gooey,', ['G', 'UW', 'IY']), ('g u i', ['G', 'UW', 'IY']), ('gui', ['G', 'UW', 'IY']),
])
def test_cited_word_hint_rejects_missing_phones_rewritten_letters_and_fake_spelling(spoken, phones):
    with pytest.raises(ValueError):
        pronunciation.validate_decisions('GUI today.', [word_row(source='GUI', spoken=spoken, phones=phones)])


@pytest.mark.parametrize('version', [10, 11])
def test_submitted_word_policy_does_not_gain_mixed_case_hints(version):
    assert pronunciation.policy_contract(policy_version=version)['word_hint_contract']['version'] == 2
    with pytest.raises(ValueError):
        pronunciation.validate_decisions('eSIM today.', [word_row(
            source='eSIM', spoken='e, sim', phones=['IY', 'S', 'IH', 'M'])], policy_version=version)
    assert pronunciation.compatible_policy(pronunciation.policy_contract(policy_version=version))


@pytest.mark.parametrize(('spoken', 'phones'), [
    ('sim', ['S', 'IH', 'M']), ('e, sims', ['IY', 'S', 'IH', 'M', 'Z']),
    ('e s i m', ['IY', 'EH', 'S', 'AY', 'EH', 'M']), ('e, sim', ['IY', 'S', 'AY', 'M']),
])
def test_mixed_case_word_hint_rejects_lost_letters_and_changed_fragment_phones(spoken, phones):
    with pytest.raises(ValueError):
        pronunciation.validate_decisions('eSIM today.', [word_row(source='eSIM', spoken=spoken, phones=phones)])


def test_known_word_punctuation_hint_keeps_its_dictionary_phones_and_all_source_letters():
    assert word.validate_word_hint('RAM', 'ram,', ['R', 'AE', 'M']) == ['R', 'AE', 'M']
    for spoken, phones in [('rum,', ['R', 'AH', 'M']), ('ram,', ['R', 'AH', 'M']),
                           ('r a m,', ['R', 'AE', 'M']), ('ram', ['R', 'AE', 'M'])]:
        with pytest.raises(ValueError):
            word.validate_word_hint('RAM', spoken, phones)
    with pytest.raises(ValueError):
        word.validate_word_hint('SIGGRAPH', 'siggraph,', ['S', 'IH', 'G', 'R', 'AE', 'F'])


@pytest.mark.parametrize('version', [7, 8, 9])
def test_single_word_punctuation_hint_does_not_change_submitted_word_policies(version):
    assert pronunciation.policy_contract(policy_version=version)['word_hint_contract']['version'] == 1
    with pytest.raises(ValueError):
        pronunciation.validate_decisions('RAM today.', [word_row(
            source='RAM', spoken='ram,', phones=['R', 'AE', 'M'])], policy_version=version)
    assert pronunciation.compatible_policy(pronunciation.policy_contract(policy_version=version))


def test_word_hint_phone_evidence_checks_entire_source_and_conventional_word_reading():
    source = 'The report mentions SATA today.'
    _, edits = pronunciation.validate_decisions(source, [word_row()])
    phones = 'ð ə ɹ ᵻ p ɔːɹ t m ɛ n ʃ ə n d s æ t ɐ t ə d eɪ'
    report = evidence.compare_source(source, edits, phones)
    assert report['supported'] and report['passed']
    assert report['word_hint_references'][0]['phones'] == ['S', 'AE', 'T', 'AH']
    assert not evidence.compare_source(source, edits, phones + ' ɐ t')['passed']
    assert not evidence.compare_source(source, edits, phones.replace('s æ t ɐ', 'ð æ t ɐ'))['passed']
    assert not evidence.compare_source(source, edits, phones.replace('s æ t ɐ', 'ɛ s eɪ t iː eɪ'))['passed']


def test_word_coda_devoicing_is_limited_to_a_following_ordinary_s_onset():
    assert evidence.compare_source('His socks.', [], 'h ɪ s ɑː k s')['passed']
    assert not evidence.compare_source('His toes.', [], 'h ɪ s t oʊ z')['passed']
    _, edits = pronunciation.validate_decisions('ZS.', [
        {'id': 0, 'source': 'ZS', 'action': 'spell', 'spoken': 'zee ess', 'reason': 'Two letter names.'},
    ])
    assert not evidence.compare_source('ZS.', edits, 's iː ɛ s')['passed']


def test_storage_nas_complete_source_phones_reject_wrong_vowels_and_extra_tail():
    text = "The report mentions NAS today."
    _, edits = pronunciation.validate_decisions(text, [{"id": 0, "source": "NAS",
        "action": "sound", "spoken": "nass", "phones": ["N", "AE", "S"],
        "reason": "Contextual storage word reading."}], policy_version=19)
    correct = "ð ə ɹ ᵻ p ɔːɹ t m ɛ n ʃ ə n z n æ s t ə d eɪ"
    report = evidence.compare_source(text, edits, correct)
    assert report["passed"]
    assert report["independent_word_references"]["NAS"]["supplemental_word_reference_contract"]["version"] == 4
    assert not evidence.compare_source(text, edits, correct.replace("n æ s", "n ə s"))["passed"]
    assert not evidence.compare_source(text, edits, correct + " t ə d eɪ")["passed"]

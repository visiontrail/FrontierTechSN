import hashlib

import pytest

from backend.pipeline import tts_pronunciation
from scripts import chatterbox_phoneme_evidence as evidence

PREFIX = 'ð ə ɹ ᵻ p ɔːɹ t m ɛ n ʃ ə n z '
ENDING = ' t ə d eɪ'


def test_published_weak_verb_am_never_reduces_named_letters():
    source = 'The note reads I AM ready.'
    assert evidence.compare_source(source, [], 'ð ə n oʊ t ɹ iː d z aɪ ə m ɹ ɛ d i')['passed']
    named = 'The report mentions AM today.'
    row = tts_pronunciation.candidates(named)[0]
    _, prepared = tts_pronunciation.validate_decisions(named, [{**row,'action':'spell',
        'spoken':'eigh em','reason':'Two named letters in this context.'}])
    assert evidence.compare_source(named, prepared, PREFIX+'eɪ ɛ m'+ENDING)['passed']
    assert not evidence.compare_source(named, prepared, PREFIX+'ə m'+ENDING)['passed']


def test_cited_saas_word_requires_complete_source_and_does_not_authorize_new_vowels():
    source = 'The report mentions SaaS today.'
    assert evidence.compare_source(source, [], PREFIX+'s æ s'+ENDING)['passed']
    assert not evidence.compare_source(source, [], PREFIX+'s ɑː z'+ENDING)['passed']


@pytest.mark.parametrize(('name', 'phones', 'wrong'), [
    ('PaaS', 'p æ s', 'p ɑː z'), ('IaaS', 'aɪ æ z', 'iː ɐ z'),
])
def test_opt_in_cloud_word_reference_requires_exact_source_phones_and_bound_version(name, phones, wrong):
    source = f'The report mentions {name} today.'
    row = tts_pronunciation.candidates(source, policy_version=16)[0]
    declared = ['P', 'AE', 'S'] if name == 'PaaS' else ['AY', 'AE', 'Z']
    _, prepared = tts_pronunciation.validate_decisions(source, [{**row, 'action':'word',
        'spoken':name+',', 'phones':declared, 'reason':'Published cloud word pronunciation.'}], policy_version=16)
    correct = PREFIX + phones + ENDING
    report = evidence.compare_source(source, prepared, correct)
    assert report['supported'] and report['passed']
    assert report['word_hint_references'][0]['word_hint_version'] == 5
    assert report['independent_word_references'][name]['supplemental_word_reference_contract']['version'] == 2
    assert not evidence.compare_source(source, prepared, PREFIX + wrong + ENDING)['passed']
    assert not evidence.compare_source(source, prepared, correct + ' t ə d eɪ')['passed']
    assert not evidence.compare_source(source, [{k:v for k,v in prepared[0].items() if k!='word_hint_version'}], correct)['supported']


@pytest.mark.parametrize(('name', 'cue', 'declared', 'phones'), [
    ('PaaS', 'pass', ['P','AE','S'], 'p æ s'), ('IaaS', 'eye as', ['AY','AE','Z'], 'aɪ æ z'),
])
def test_source_word_phonetic_cue_is_checked_against_original_source_not_its_own_words(name, cue, declared, phones):
    source = f'The report mentions {name} today.'
    row = tts_pronunciation.candidates(source, policy_version=16)[0]
    _, prepared = tts_pronunciation.validate_decisions(source, [{**row,'action':'sound','spoken':cue,
        'phones':declared,'reason':'Published source reading.'}], policy_version=16)
    correct = PREFIX+phones+ENDING
    report = evidence.compare_source(source, prepared, correct)
    assert report['supported'] and report['passed']
    assert report['source_sha256'] == hashlib.sha256(source.encode()).hexdigest()
    assert report['word_hint_references'][0]['word_hint_type'] == 'homophone'
    assert not evidence.compare_source(source, prepared, correct+' p æ s')['passed']
    tampered = [{**prepared[0], 'source':'ZZZZ'}]
    assert not evidence.compare_source(source.replace(name,'ZZZZ'), tampered, correct)['supported']


@pytest.mark.parametrize(('source_name', 'cue', 'phones', 'wrong'), [
    ('C++', 'see plus plus', 's iː p l ʌ s p l ʌ s', 's iː p l ʌ s'),
    ('C#', 'see sharp', 's iː ʃ ɑːɹ p', 's iː ʃ ɑːɹ'),
])
def test_programming_source_phones_require_every_operator(source_name, cue, phones, wrong):
    source = f'The report mentions {source_name} today.'
    start = source.index(source_name)
    prepared = [{'source': source_name, 'start': start, 'end': start + len(source_name),
                 'spoken': cue, 'kind': 'programming'}]
    report = evidence.compare_source(source, prepared, PREFIX + phones + ENDING)
    assert report['supported'] and report['passed']
    assert report['programming_reading_references'][source_name] == evidence.PROGRAMMING_REFERENCES[source_name]
    assert not evidence.compare_source(source, prepared, PREFIX + wrong + ENDING)['passed']
    assert not evidence.compare_source(source, prepared, PREFIX + phones + ENDING + ' t ə d eɪ')['passed']
    tampered = [{**prepared[0], 'spoken': 'see plus'}]
    assert not evidence.compare_source(source, tampered, PREFIX + phones + ENDING)['supported']


@pytest.mark.parametrize(('term', 'phones'), [
    ('JSON', 'dʒ eɪ s ə n'), ('GIF', 'ɡ ɪ f'), ('GIF', 'dʒ ɪ f'),
    ('GUI', 'ɡ uː iː'),
])
def test_cited_missing_dictionary_acronym_requires_exact_word_phones(term, phones):
    source = f'The report mentions {term} today.'
    report = evidence.compare_source(source, [], PREFIX + phones + ENDING)
    assert report['supported'] and report['passed']
    assert report['cited_word_references'][term.casefold()]['source_url'].startswith('https://')
    assert report['word_reference_sha256'] == hashlib.sha256(evidence.WORD_REFERENCES.read_bytes()).hexdigest()
    assert not evidence.compare_source(source, [], PREFIX + phones + ENDING + ' t')['passed']


@pytest.mark.parametrize(('term', 'phones'), [
    ('JSON', 'dʒ eɪ s ə n t ə'), ('GIF', 'ɡ ɪ v'), ('GIF', 'ɡ ɪ f t ə'),
    ('GUI', 'ɡ iː'), ('GUI', 'ɡ aɪ'),
])
def test_cited_word_reference_never_accepts_missing_vowels_or_extra_consonants(term, phones):
    source = f'The report mentions {term} today.'
    report = evidence.compare_source(source, [], PREFIX + phones + ENDING)
    assert report['supported'] and not report['passed']


def test_spelled_candidate_does_not_gain_a_word_pronunciation_from_reference_data():
    source = 'The report mentions JSON today.'
    prepared = edits(source, 'JSON', 'jay ess oh N')
    report = evidence.compare_source(source, prepared, PREFIX + 'dʒ eɪ s ə n' + ENDING)
    assert report['supported'] and not report['passed']
    assert 'cited_word_references' not in report
    assert evidence.compare_source(source, prepared, PREFIX + 'dʒ eɪ ɛ s oʊ ɛ n' + ENDING)['passed']


def test_document_noun_reference_keeps_exact_schwa_and_does_not_reduce_other_words():
    source = 'The JSON parser reads the document.'
    phones = 'ð ə dʒ eɪ s ə n p ɑːɹ s ɚ ɹ iː d z ð ə d ɑː k j uː m ə n t'
    report = evidence.compare_source(source, [], phones)
    assert report['passed']
    assert set(report['cited_word_references']) == {'json', 'document'}
    assert not evidence.compare_source(source, [], phones + ' æ n f aɪ n d')['passed']
    assert not evidence.compare_source(source, [], phones.replace('dʒ eɪ s ə n', 'dʒ eɪ s ə n t'))['passed']
    assert not evidence.compare_source(source, [], phones.replace('p ɑːɹ s ɚ', 'p ə s ɚ'))['passed']


def test_cited_esim_word_does_not_omit_its_prefix_vowel():
    source = 'The report mentions eSIM today.'
    report = evidence.compare_source(source, [], PREFIX + 'iː s ɪ m' + ENDING)
    assert report['supported'] and report['passed']
    assert not evidence.compare_source(source, [], PREFIX + 's ɪ m' + ENDING)['passed']
    assert not evidence.compare_source(source, [], PREFIX + 'iː ɛ s aɪ ɛ m' + ENDING)['passed']


def test_cited_presents_verb_variant_preserves_every_degree_letter_and_source_word():
    source = 'The PhD student presents the research.'
    prepared = edits(source, 'PhD', 'pee aitch dee')
    phones = 'ð ə p iː eɪ tʃ d iː s t uː d ə n t p ɹ ᵻ z ɛ n t s ð ə ɹ iː s ɜː tʃ'
    report = evidence.compare_source(source, prepared, phones)
    assert report['passed'] and report['supported']
    assert report['cited_body_word_references']['presents']['source_url'] == (
        'https://www.oxfordlearnersdictionaries.com/definition/english/present_3')
    for changed in [phones.replace('p iː eɪ tʃ', 'eɪ tʃ'),
                    phones.replace('ð ə p iː', 'ɐ p iː'),
                    phones.replace('ɹ iː s ɜː tʃ', 'ɹ iː s ɜː'), phones + ' d iː']:
        assert not evidence.compare_source(source, prepared, changed)['passed']


@pytest.mark.parametrize(('term', 'phones', 'incorrect'), [
    ('COVID', 'k oʊ v ɪ d', 'k oʊ b ɪ d'),
    ('FOMO', 'f oʊ m oʊ', 'f ɪ l m oʊ'),
    ('CRISPR', 'k ɹ ɪ s p ɚ', 'k ɹ ɪ s p'),
    ('JPEG', 'dʒ eɪ p ɛ ɡ', 'dʒ eɪ p ɛ k'),
    ('LIDAR', 'l aɪ d ɑːɹ', 'l aɪ ɾ ɑːɹ'),
    ('EBITDA', 'ɛ b ɪ t d ə', 'ɛ b ɪ ɾ ə'),
    ('WASM', 'w ɑː z ə m', 'w ɑː z m'),
    ('README', 'ɹ iː d m iː', 'ɹ ɛ d m iː'),
    ('NAS', 'n æ s', 'n ɑː z'),
    ('ISO', 'aɪ s oʊ', 'aɪ z oʊ'),
])
def test_cited_native_word_reference_does_not_accept_caption_autocorrection(term, phones, incorrect):
    source = f'The report mentions {term} today.'
    report = evidence.compare_source(source, [], PREFIX + phones + ENDING)
    assert report['supported'] and report['passed']
    assert term.casefold() in report['cited_body_word_references']
    assert not evidence.compare_source(source, [], PREFIX + incorrect + ENDING)['passed']
    assert not evidence.compare_source(source, [], PREFIX + phones + ENDING + ' t')['passed']


def test_yaml_published_rhyme_preserves_its_word_reading_and_every_source_word():
    source = 'The report mentions YAML today.'
    phones = PREFIX + 'j æ m əl' + ENDING
    report = evidence.compare_source(source, [], phones)
    assert report['supported'] and report['passed']
    assert report['cited_body_word_references']['yaml']['source_url'] == 'https://yaml.org/spec/1.2.2/'
    assert not evidence.compare_source(source, [], PREFIX + 'j eɪ m əl' + ENDING)['passed']
    assert not evidence.compare_source(source, [], phones + ' ɛ l')['passed']


def test_cited_hyphenated_word_keeps_both_vowels_and_source_boundaries():
    source = 'The report mentions Wi-Fi today.'
    phones = PREFIX + 'w aɪ f aɪ' + ENDING
    report = evidence.compare_source(source, [], phones)
    assert report['supported'] and report['passed']
    assert 'wi-fi' in report['cited_body_word_references']
    for changed in ['w aɪ f', 'w ɪ f aɪ', 'w aɪ f aɪ aɪ']:
        assert not evidence.compare_source(source, [], PREFIX + changed + ENDING)['passed']
    assert evidence.ordinary_source_words('Wi-Fi high-speed Wi-Fi-extra') == [
        'Wi-Fi', 'high', 'speed', 'Wi', 'Fi', 'extra']
    with pytest.raises(ValueError, match='No source pronunciation reference'):
        evidence.reference_parts('The report mentions Wi-Fi-extra today.', [])
    assert evidence.reference_parts('A high-speed report.', []) == evidence.reference_parts(
        'A high speed report.', [])


def test_wasm_published_rhymes_preserve_the_complete_word_and_closing_source():
    source = 'The report mentions WASM today.'
    for vowel in ['ɑː', 'æ']:
        assert evidence.compare_source(source, [], PREFIX + f'w {vowel} z ə m' + ENDING)['passed']
    for phones in ['w ɑː z ɪ m', 'w ɑː z ə', 'w ɑː z ə m ə']:
        assert not evidence.compare_source(source, [], PREFIX + phones + ENDING)['passed']
    assert not evidence.compare_source(source, [], PREFIX + 'w ɑː z ə m')['passed']


def test_asic_technical_reference_replaces_unrelated_dictionary_reading_only_in_capitals():
    source = 'The report mentions ASIC today.'
    correct = PREFIX + 'eɪ s ɪ k' + ENDING
    unrelated = PREFIX + 'æ z ɪ k' + ENDING
    report = evidence.compare_source(source, [], correct)
    assert report['supported'] and report['passed']
    assert report['cited_body_word_references']['asic']['replace_dictionary'] is True
    assert not evidence.compare_source(source, [], unrelated)['passed']
    lower = source.replace('ASIC', 'asic')
    assert evidence.compare_source(lower, [], unrelated)['passed']
    assert 'asic' not in evidence.compare_source(lower, [], unrelated).get('cited_body_word_references', {})
    assert not evidence.compare_source(source, [], correct + ' ɛ s')['passed']


def test_nas_storage_word_reference_does_not_change_a_source_bound_spelled_reading():
    source = 'The report mentions NAS today.'
    spelled = edits(source, 'NAS', 'N eigh ess')
    assert evidence.compare_source(source, spelled, PREFIX + 'ɛ n eɪ ɛ s' + ENDING)['passed']
    assert not evidence.compare_source(source, spelled, PREFIX + 'n æ s' + ENDING)['passed']
    assert not evidence.compare_source(source, spelled, PREFIX + 'ɛ n eɪ ɛ s ɛ s' + ENDING)['passed']
    assert not evidence.compare_source(source, [], PREFIX + 'n æ s' + ENDING + 'n æ s')['passed']


def test_iso_word_reference_never_accepts_a_spelled_edit_or_an_extra_eye():
    source = 'The report mentions ISO today.'
    word = PREFIX + 'aɪ s oʊ' + ENDING
    assert evidence.compare_source(source, [], word)['passed']
    spelled = edits(source, 'ISO', 'eye ess oh')
    assert not evidence.compare_source(source, spelled, word)['passed']
    assert evidence.compare_source(source, spelled, PREFIX + 'aɪ ɛ s oʊ' + ENDING)['passed']
    assert not evidence.compare_source(source, [], word + ' aɪ')['passed']


def test_configuration_dictionary_variant_preserves_yaml_and_rhotic_vowel_identity():
    source = 'The YAML configuration file is ready.'
    phones = 'ð ə j æ m əl k ə n f ɪ ɡ ɚ ɹ eɪ ʃ ə n f aɪ l ɪ z ɹ ɛ d i'
    report = evidence.compare_source(source, [], phones)
    assert report['supported'] and report['passed']
    assert report['cited_body_word_references']['configuration']['source_url'] == (
        'https://www.oxfordlearnersdictionaries.com/definition/english/configuration')
    assert not evidence.compare_source(source, [], phones.replace('j æ m əl', 'j eɪ m əl'))['passed']
    assert not evidence.compare_source(source, [], phones + ' ɑːɹ')['passed']
    assert evidence.observed_phones('ɚ ɹ ɑːɹ') == ('ER', 'R', 'AA', 'R')
    assert not evidence.compare_source(source, [], phones.replace('ɚ ɹ eɪ', 'ɚ eɪ'))['passed']
    source_r = 'The report mentions RR today.'
    prepared = edits(source_r, 'RR', 'R R')
    assert not evidence.compare_source(source_r, prepared, PREFIX + 'ɚ ɹ' + ENDING)['passed']


def test_body_reference_does_not_change_frozen_word_hint_policy13_bytes():
    from backend.pipeline import tts_pronunciation
    contract = tts_pronunciation.policy_contract(policy_version=13)
    assert contract['word_hint_contract']['supplemental_word_references']['reference_sha256'] == (
        '64412398b4a22ab8273596108b55d40a50d5675e2537ddb6f78eb619153fdd8b')
    assert contract['prompt_sha256'] == '83ffb1466fdb9be77e69a2f89952ebf1fc7922deddc43e7caa938e83addf0e85'


def test_unknown_source_word_cannot_use_its_own_hint_as_pronunciation_evidence():
    source = 'The report mentions RAMRAM today.'
    prepared = [{'source': 'RAMRAM', 'start': source.index('RAMRAM'),
                 'end': source.index('RAMRAM') + 6, 'spoken': 'ram ram',
                 'kind': 'word', 'phones': ['R', 'AE', 'M', 'R', 'AE', 'M']}]
    from backend.pipeline.tts_word_pronunciation import validate_word_hint
    assert validate_word_hint('RAMRAM', 'ram ram', prepared[0]['phones'])
    report = evidence.compare_source(source, prepared, PREFIX + 'ɹ æ m ɹ æ m' + ENDING)
    assert not report['supported'] and not report['passed']
    assert 'independent source-word' in report['unsupported_reason']


def test_cited_oled_word_blocks_a_glyph_preserving_but_wrong_reading():
    source = 'The report mentions OLED today.'
    prepared = [{'source': 'OLED', 'start': source.index('OLED'),
                 'end': source.index('OLED') + 4, 'spoken': 'ole D',
                 'kind': 'word', 'phones': ['OW', 'L', 'EY', 'D', 'IY']}]
    from backend.pipeline.tts_word_pronunciation import validate_word_hint
    assert validate_word_hint('OLED', 'ole D', prepared[0]['phones'])
    report = evidence.compare_source(source, prepared, PREFIX + 'oʊ l eɪ d iː' + ENDING)
    assert not report['passed']
    assert 'independent source-word' in report['unsupported_reason']


@pytest.mark.parametrize(('term', 'spoken', 'phones', 'observed'), [
    ('OLED', 'O, led', ['OW', 'L', 'EH', 'D'], 'oʊ l ɛ d'),
    ('SATA', 'sat, a', ['S', 'AE', 'T', 'AH'], 's æ t ə'),
    ('SIGGRAPH', 'sig graph', ['S', 'IH', 'G', 'G', 'R', 'AE', 'F'], 's ɪ ɡ ɹ æ f'),
])
def test_legacy_word_cues_are_confirmed_against_independent_source_references(term, spoken, phones, observed):
    source = f'The report mentions {term} today.'
    prepared = [{'source': term, 'start': source.index(term), 'end': source.index(term) + len(term),
                 'spoken': spoken, 'kind': 'word', 'phones': phones}]
    report = evidence.compare_source(source, prepared, PREFIX + observed + ENDING)
    assert report['supported'] and report['passed']
    assert report['source_word_reference_policy_version'] == evidence.SOURCE_WORD_REFERENCE_POLICY_VERSION
    assert report['independent_word_references'][term]['source_url'].startswith('https://')


def test_published_the_weak_form_applies_only_before_an_actual_vowel_onset():
    source = "The OpenAI's result is available."
    prepared = edits(source, 'AI', 'eigh eye')
    phones = 'ð ɪ oʊ p ə n eɪ aɪ z ɹ ᵻ z ʌ l t ɪ z ɐ v eɪ l ə b əl'
    report = evidence.compare_source(source, prepared, phones)
    assert report['supported'] and report['passed']
    assert report['cited_body_word_references']['the']['source_url'] == (
        'https://www.collinsdictionary.com/dictionary/english/the')
    assert not evidence.compare_source(source, prepared, phones + ' ɪ z ɐ v eɪ l ə b əl')['passed']
    assert not evidence.compare_source(source, prepared, phones.replace('eɪ aɪ z', 'aɪ aɪ z'))['passed']
    consonant_source = 'The report is available.'
    consonant_phones = 'ð ɪ ɹ ᵻ p ɔːɹ t ɪ z ɐ v eɪ l ə b əl'
    assert not evidence.compare_source(consonant_source, [], consonant_phones)['passed']
    assert evidence.compare_source(consonant_source, [], consonant_phones.replace('ð ɪ', 'ð ə'))['passed']


def test_the_weak_form_uses_named_letter_sound_and_never_changes_the_letter():
    n_source = 'The NN is available.'
    n_edits = edits(n_source, 'NN', 'enn enn')
    n_phones = 'ð ɪ ɛ n ɛ n ɪ z ɐ v eɪ l ə b əl'
    assert evidence.compare_source(n_source, n_edits, n_phones)['passed']
    assert not evidence.compare_source(n_source, n_edits, n_phones.replace('ɛ n', 'ɪ n'))['passed']
    u_source = 'The UU is available.'
    u_edits = edits(u_source, 'UU', 'you you')
    u_phones = 'ð ɪ j uː j uː ɪ z ɐ v eɪ l ə b əl'
    assert not evidence.compare_source(u_source, u_edits, u_phones)['passed']
    assert evidence.compare_source(u_source, u_edits, u_phones.replace('ð ɪ', 'ð ə'))['passed']
    assert not evidence.compare_source('The.', [], 'ð ɪ')['passed']


def test_ordinary_connects_coda_elision_retains_every_pcie_letter_and_plural():
    source = 'The PCIe interface connects the device.'
    prepared = edits(source, 'PCIe', 'pee see eye ee')
    phones = 'ð ə p iː s iː aɪ i ɪ n t ɚ f eɪ s k ə n ɛ k s ð ə d ᵻ v aɪ s'
    report = evidence.compare_source(source, prepared, phones)
    assert report['passed']
    assert report['ordinary_coda_elision_reference'] == evidence.ORDINARY_CODA_ELISION_SOURCE
    for changed in [phones.replace('aɪ i ɪ', 'aɪ ɪ'),
                    phones.replace('k ə n ɛ k s', 'k ə n ɛ k'),
                    phones.replace('k ə n ɛ k s', 'k ə n æ k s'), phones + ' iː']:
        assert not evidence.compare_source(source, prepared, changed)['passed']


def test_ordinary_coda_elision_never_drops_a_spelled_t_letter():
    source = 'The report mentions KTS today.'
    prepared = edits(source, 'KTS', 'kay tee ess')
    assert evidence.compare_source(source, prepared, PREFIX + 'k eɪ t iː ɛ s' + ENDING)['passed']
    assert not evidence.compare_source(source, prepared, PREFIX + 'k eɪ ɛ s' + ENDING)['passed']


@pytest.mark.parametrize(('source', 'phones'), [
    ('Meeting.', 'm iː ɾ ɪ ŋ'), ('Data.', 'd eɪ ɾ ə'), ('Genetic.', 'dʒ ə n ɛ ɾ ɪ k'),
])
def test_dictionary_word_intervocalic_flap_retains_all_vowels_and_following_words(source, phones):
    report = evidence.compare_source(source, [], phones)
    assert report['supported'] and report['passed']
    assert report['ordinary_flap_reference'] == evidence.ORDINARY_FLAP_SOURCE
    assert not evidence.compare_source(source, [], phones + ' iː')['passed']


def test_flap_is_not_a_global_t_d_substitution_or_a_named_letter_pronunciation():
    source = 'The report mentions TD today.'
    prepared = edits(source, 'TD', 'tee dee')
    assert not evidence.compare_source(source, prepared, PREFIX + 'ɾ iː d iː' + ENDING)['passed']
    assert not evidence.compare_source(source, prepared, PREFIX + 't iː ɾ iː' + ENDING)['passed']
    assert not evidence.compare_source('Today.', [], 't ə ɾ eɪ')['passed']
    assert not evidence.compare_source('Meeting.', [], 'm iː ɾ ŋ')['passed']


def edits(source, term, spoken):
    return tts_pronunciation.validate_decisions(source, [
        {'id': 0, 'source': term, 'action': 'spell', 'spoken': spoken,
         'reason': 'Source-bound acoustic test.'},
    ])[1]


@pytest.mark.parametrize(('term', 'spoken', 'phones'), [
    ('RPC', 'ar pee see', 'ɑːɹ p iː s iː'),
    ('URI', 'you are eye', 'j uː ɑːɹ ɹ aɪ'),
    ('SSH', 'ess ess aitch', 'ɛ s ɛ s eɪ tʃ'),
    ('AI', 'eigh eye', 'eɪ aɪ'),
    ('SSH', 'double ess H', 'd ʌ b əl l ɛ s eɪ tʃ'),
])
def test_full_source_phone_evidence_retains_every_letter_and_source_word(term, spoken, phones):
    source = f'The report mentions {term} today.'
    report = evidence.compare_source(source, edits(source, term, spoken), PREFIX + phones + ENDING)
    assert report['supported'] and report['passed']


@pytest.mark.parametrize(('term', 'spoken', 'phones', 'ending'), [
    ('RPC', 'ar pee see', 'p iː s iː', ENDING),
    ('RPC', 'ar pee see', 'aʊ ɹ p iː s iː', ENDING),
    ('AI', 'eigh eye', 'aɪ aɪ', ENDING),
    ('AI', 'eigh eye', 'eɪ aɪ aɪ', ENDING),
    ('SSH', 'ess ess aitch', 'ɛ s ɛ s ɛ s eɪ tʃ', ENDING),
    ('SSH', 'ess ess aitch', 'ɛ s ɛ s ɛ dʒ', ENDING),
    ('SSH', 'ess ess aitch', 'ɛ s ɛ s eɪ tʃ', ' t eɪ'),
    ('URI', 'you are eye', 'j uː ɑːɹ aɪ', ENDING + ' j uː ɑːɹ'),
])
def test_phone_evidence_rejects_wrong_vowels_missing_letters_repetition_and_changed_closing(term, spoken, phones, ending):
    source = f'The report mentions {term} today.'
    report = evidence.compare_source(source, edits(source, term, spoken), PREFIX + phones + ending)
    assert report['supported'] and not report['passed']


def test_phone_token_boundaries_do_not_merge_the_schwa_with_the_next_word_r():
    assert evidence.observed_phones('ð ə ɹ') == ('DH', 'AH', 'R')
    assert evidence.observed_phones('ɚ') == ('ER',)


@pytest.mark.parametrize('prerhotic', ['oːɹ', 'oː ɹ', 'oɹ', 'o r'])
def test_documented_prerhotic_o_preserves_the_complete_source_and_named_r(prerhotic):
    source = 'The report mentions BRB today.'
    prepared = edits(source, 'BRB', 'bee R bee')
    phones = PREFIX.replace('ɔːɹ', prerhotic) + 'b iː ɑːɹ b iː' + ENDING
    report = evidence.compare_source(source, prepared, phones)
    assert report['supported'] and report['passed']
    assert report['prerhotic_vowel_reference'] == evidence.PRERHOTIC_VOWEL_SOURCE
    assert not evidence.compare_source(source, prepared, phones.replace('ɑːɹ', 'oːɹ'))['passed']
    assert not evidence.compare_source(source, prepared, phones + ' ɑːɹ')['passed']


def test_prerhotic_normalization_does_not_erase_vowel_only_letter_identity_or_another_r_name():
    source = 'The report mentions AI today.'
    prepared = edits(source, 'AI', 'eigh eye')
    assert not evidence.compare_source(source, prepared, PREFIX + 'aɪ aɪ' + ENDING)['passed']
    assert evidence.observed_phones('oʊ ɑːɹ') == ('OW', 'AA', 'R')
    with pytest.raises(ValueError, match='Unsupported observed phone'):
        evidence.observed_phones('o t')


def test_vowel_repetitions_survive_consonant_continuation_normalization():
    assert evidence.observed_phones('aɪ aɪ ɑːɹ ɹ') == ('AY', 'AY', 'AA', 'R')


@pytest.mark.parametrize(('source', 'phones'), [
    ('UnsupportedQbitThing today.', 't ə d eɪ'),
    ('The value is 1234567.', 'ð ə v æ l j uː ɪ z t uː'),
    ('Today.', 'ϕ'),
])
def test_unsupported_reference_or_phone_never_claims_acceptance(source, phones):
    report = evidence.compare_source(source, [], phones)
    assert not report['supported'] and not report['passed']


@pytest.mark.parametrize('source', ["Arizona's.", 'Arizona’s.'])
def test_dictionary_possessive_retains_its_complete_pronunciation(source):
    phones = 'ɛ ɹ ɪ z oʊ n ə z'
    assert evidence.compare_source(source, [], phones)['passed']
    assert not evidence.compare_source(source, [], phones.removesuffix(' z'))['passed']
    assert not evidence.compare_source(source, [], phones + ' eɪ')['passed']


def test_postal_abbreviation_with_dictionary_possessive_requires_each_letter():
    source = "AZ is Arizona's postal abbreviation."
    prepared = edits(source, 'AZ', 'eigh zee')
    ending = ' ɪ z ɛ ɹ ɪ z oʊ n ə z p oʊ s t ə l ə b ɹ iː v iː eɪ ʃ ə n'
    assert evidence.compare_source(source, prepared, 'eɪ z iː' + ending)['passed']
    assert not evidence.compare_source(source, prepared, 'aɪ z iː' + ending)['passed']
    assert not evidence.compare_source(source, prepared, 'eɪ z iː' + ending + ' z iː')['passed']


@pytest.mark.parametrize(('source', 'phones'), [
    ("It's.", 'ɪ t s'),
    ('Can’t.', 'k æ n t'),
    ("York's.", 'j ɔː ɹ k s'),
    ("Texas's.", 't ɛ k s ə s ɪ z'),
])
def test_known_dictionary_apostrophe_words_use_their_actual_word_phones(source, phones):
    assert evidence.compare_source(source, [], phones)['passed']


@pytest.mark.parametrize('source', [
    "UnsupportedQbitThing's.", "Arizona''s.", "Arizona'.", "'Arizona.",
])
def test_unknown_or_malformed_apostrophe_words_remain_unsupported(source):
    report = evidence.compare_source(source, [], 'ɛ ɹ ɪ z oʊ n ə z')
    assert not report['supported'] and not report['passed']


@pytest.mark.parametrize(('source', 'term', 'spoken', 'phones', 'suffix'), [
    ("AI's.", 'AI', 'eigh eye', 'eɪ aɪ', ' z'),
    ('AI’s.', 'AI', 'eigh eye', 'eɪ aɪ', ' z'),
    ('AIs.', 'AI', 'eigh eye', 'eɪ aɪ', ' z'),
    ('GPUs.', 'GPU', 'gee pee you', 'dʒ iː p iː j uː', ' z'),
    ("RF's.", 'RF', 'R F', 'ɑːɹ ɛ f', ' s'),
    ("HX's.", 'HX', 'H ex', 'eɪ tʃ ɛ k s', ' ɪ z'),
    ("HX's.", 'HX', 'H ex', 'eɪ tʃ ɛ k s', ' ə z'),
    ("U.S.'s.", 'U.S.', 'you ess', 'j uː ɛ s', ' ɪ z'),
])
def test_spelled_letter_inflection_requires_every_letter_and_the_regular_ending(
        source, term, spoken, phones, suffix):
    prepared = edits(source, term, spoken)
    assert evidence.compare_source(source, prepared, phones + suffix)['passed']
    assert not evidence.compare_source(source, prepared, phones)['passed']
    assert not evidence.compare_source(source, prepared, phones + ' ɛ s')['passed']
    assert not evidence.compare_source(source, prepared, phones + suffix + ' eɪ')['passed']


def test_possessive_openai_retains_the_prefix_following_words_and_letter_vowels():
    source = "OpenAI's own."
    prepared = edits(source, 'AI', 'eigh eye')
    phones = 'oʊ p ə n eɪ aɪ z oʊ n'
    assert evidence.compare_source(source, prepared, phones)['passed']
    assert not evidence.compare_source(source, prepared, phones.replace('eɪ aɪ', 'aɪ aɪ'))['passed']
    assert not evidence.compare_source(source, prepared, phones.replace('aɪ z', 'aɪ'))['passed']
    assert not evidence.compare_source(source, prepared, phones.replace('aɪ z', 'aɪ aɪ z'))['passed']


def test_letter_suffix_allomorph_does_not_accept_the_wrong_consonant():
    assert not evidence.compare_source("RF's.", edits("RF's.", 'RF', 'R F'), 'ɑːɹ ɛ f z')['passed']
    assert not evidence.compare_source("AI's.", edits("AI's.", 'AI', 'eigh eye'), 'eɪ aɪ s')['passed']


@pytest.mark.parametrize('source', ["AI'something.", "AI's's.", 'AIss.'])
def test_inflection_matching_does_not_discard_an_unknown_continuation(source):
    assert not evidence.compare_source(source, edits(source, 'AI', 'eigh eye'), 'eɪ aɪ z')['passed']


@pytest.mark.parametrize('defect', ['hash', 'model', 'revision', 'conditioned'])
def test_phone_provenance_requires_the_exact_unconditioned_waveform(tmp_path, defect):
    wav = tmp_path / 'original.wav'
    wav.write_bytes(b'unchanged-waveform')
    observation = {'audio_sha256': hashlib.sha256(wav.read_bytes()).hexdigest(),
                   'model': evidence.MODEL, 'model_revision': evidence.REVISION,
                   'conditioned_on_reference_text': False, 'phones': 't ə d eɪ'}
    key = {'hash': 'audio_sha256', 'model': 'model', 'revision': 'model_revision',
           'conditioned': 'conditioned_on_reference_text'}[defect]
    observation[key] = True if defect == 'conditioned' else 'different'
    with pytest.raises(ValueError, match='pinned unconditioned'):
        evidence.audit_observation(wav, 'Today.', [], observation)
    assert wav.read_bytes() == b'unchanged-waveform'


def test_positive_phone_evidence_does_not_claim_overall_audio_acceptance(tmp_path):
    wav = tmp_path / 'original.wav'
    wav.write_bytes(b'unchanged-waveform')
    observation = {'audio_sha256': hashlib.sha256(wav.read_bytes()).hexdigest(),
                   'model': evidence.MODEL, 'model_revision': evidence.REVISION,
                   'conditioned_on_reference_text': False, 'phones': 't ə d eɪ'}
    report = evidence.audit_observation(wav, 'Today.', [], observation)
    assert report['passed'] and not report['overall_audio_acceptance']


def test_numeric_model_keeps_each_letter_and_single_digit_in_original_order():
    source='The report mentions A2A today.'
    rows=[{'id':row['id'],'source':row['source'],'spoken':'eigh','action':'spell','reason':'Same A letter.'}
          for row in tts_pronunciation.candidates(source)]
    _,prepared=tts_pronunciation.validate_decisions(source,rows)
    assert evidence.compare_source(source,prepared,PREFIX+'eɪ t uː eɪ'+ENDING)['passed']
    assert not evidence.compare_source(source,prepared,PREFIX+'ɐ t uː eɪ'+ENDING)['passed']
    assert not evidence.compare_source(source,prepared,PREFIX+'eɪ θ ɹ iː eɪ'+ENDING)['passed']
    assert not evidence.compare_source(source,prepared,PREFIX+'eɪ t uː eɪ'+ENDING+' ɐ')['passed']


def test_legacy_numeric_source_cannot_treat_a_model_letter_as_an_indefinite_article():
    source='The report mentions A2A today.'
    assert evidence.compare_source(source,[],PREFIX+'eɪ t uː eɪ'+ENDING)['passed']
    assert not evidence.compare_source(source,[],PREFIX+'ɐ t uː eɪ'+ENDING)['passed']
    assert not evidence.compare_source(source,[],PREFIX+'eɪ t uː ɐ'+ENDING)['passed']


@pytest.mark.parametrize(('literal', 'words'), [
    ('16', ['sixteen']), ('58', ['fifty', 'eight']),
    ('200', ['two', 'hundred']), ('5090', ['five', 'thousand', 'ninety']),
    ('01', ['zero', 'one']), ('5.2', ['five', 'point', 'two']),
    ('1.03', ['one', 'point', 'zero', 'three']),
])
def test_numeric_source_reference_retains_value_leading_zeros_and_fraction_precision(literal, words):
    assert evidence.source_number_words(literal) == words


def test_multi_digit_product_retains_each_letter_and_exact_number():
    source = 'The report mentions FP16 today.'
    prepared = edits(source, 'FP', 'eff pee')
    correct = PREFIX + 'ɛ f p iː s ɪ k s t iː n' + ENDING
    assert evidence.compare_source(source, prepared, correct)['passed']
    for changed in [correct.replace('s ɪ k s t iː n', 's ɪ k s t i'),
                    correct.replace('ɛ f p iː', 'ɛ f'), correct + ' s ɪ k s t iː n']:
        assert not evidence.compare_source(source, prepared, changed)['passed']
    wrong = evidence.compare_source('The value is 21.', [], 'ð ə v æ l j uː ɪ z t uː')
    assert wrong['supported'] and not wrong['passed']


def test_decimal_model_requires_point_and_does_not_merge_or_reorder_digits():
    source = 'The report mentions GPT-5.2 today.'
    prepared = edits(source, 'GPT', 'gee pee tee')
    correct = PREFIX + 'dʒ iː p iː t iː f aɪ v p ɔɪ n t t uː' + ENDING
    assert evidence.compare_source(source, prepared, correct)['passed']
    for changed in [correct.replace('p ɔɪ n t ', ''),
                    correct.replace('f aɪ v p ɔɪ n t t uː', 'f aɪ v t w ɛ n t i t uː'),
                    correct.replace('f aɪ v p ɔɪ n t t uː', 't uː p ɔɪ n t f aɪ v')]:
        assert not evidence.compare_source(source, prepared, changed)['passed']


def test_versioned_numeric_cue_keeps_original_decimal_in_complete_source_evidence():
    source = 'The report mentions TLS1.3 today.'
    decisions = [{**row, 'action': 'number' if row.get('kind') == 'number' else 'spell',
                  'spoken': tts_pronunciation.numeric_cue(row['source']) if row.get('kind') == 'number' else 'tee ell ess',
                  'reason': 'Exact source representation.'}
                 for row in tts_pronunciation.candidates(source, policy_version=14)]
    _, prepared = tts_pronunciation.validate_decisions(source, decisions, policy_version=14)
    correct = PREFIX + 't iː ɛ l ɛ s w ʌ n p ɔɪ n t θ ɹ iː' + ENDING
    assert evidence.compare_source(source, prepared, correct)['passed']
    assert not evidence.compare_source(source, prepared, correct.replace('p ɔɪ n t ', ''))['passed']
    from backend.pipeline import chatterbox
    words = [{'text': 'The report mentions T L S one point three today.'}]
    assert chatterbox._pronunciation_report(source, words, prepared)['passed']


def test_fractional_leading_zero_and_trailing_zero_are_audible_source_precision():
    source = 'The value is 1.03.'
    correct = 'ð ə v æ l j uː ɪ z w ʌ n p ɔɪ n t z ɪ ɹ oʊ θ ɹ iː'
    assert evidence.compare_source(source, [], correct)['passed']
    assert not evidence.compare_source(source, [], correct.replace('z ɪ ɹ oʊ ', ''))['passed']
    assert evidence.source_number_words('1.30') == ['one', 'point', 'three', 'zero']
    unsupported = evidence.compare_source('The value is 1.2.3.', [], correct)
    assert not unsupported['supported'] and not unsupported['passed']


def test_numeric_reference_never_drops_a_leading_zero_or_unbounded_source():
    source = 'The report mentions GD01 today.'
    prepared = edits(source, 'GD', 'gee dee')
    correct = PREFIX + 'dʒ iː d iː z ɪ ɹ oʊ w ʌ n' + ENDING
    assert evidence.compare_source(source, prepared, correct)['passed']
    assert not evidence.compare_source(source, prepared, correct.replace('z ɪ ɹ oʊ ', ''))['passed']
    with pytest.raises(ValueError, match='bounds'):
        evidence.source_number_words('1234567')


def test_four_digit_identifier_groups_must_preserve_the_exact_written_value():
    assert ['fifty', 'ninety'] in evidence.source_number_readings('5090')
    assert ['nineteen', 'hundred'] in evidence.source_number_readings('1900')
    assert ['twenty', 'twenty', 'seven'] in evidence.source_number_readings('2027')
    source = 'The report mentions RTX5090 today.'
    prepared = edits(source, 'RTX', 'ar tee ex')
    correct = PREFIX + 'ɑːɹ t iː ɛ k s f ɪ f t i n aɪ n t i' + ENDING
    assert evidence.compare_source(source, prepared, correct)['passed']
    assert not evidence.compare_source(source, prepared, correct.replace('n aɪ n t i', 'eɪ t i'))['passed']
    assert evidence.source_number_readings('5.2') == [['five', 'point', 'two']]
    assert evidence.source_number_readings('01') == [['zero', 'one']]


def test_ordinary_dictionary_word_nasal_fricative_transition_preserves_the_complete_source():
    source='The report mentions AI today.'
    prepared=edits(source,'AI','eigh eye')
    phones=PREFIX.replace('n ʃ','n tʃ')+'eɪ aɪ'+ENDING
    assert evidence.compare_source(source,prepared,phones)['passed']
    assert not evidence.compare_source(source,prepared,phones+' t')['passed']
    assert not evidence.compare_source(source,prepared,phones.replace('eɪ aɪ','aɪ aɪ'))['passed']


def test_dictionary_transition_variant_never_waives_an_extra_named_letter_or_wrong_h():
    source='The report mentions NH today.'
    prepared=edits(source,'NH','N aitch')
    assert not evidence.compare_source(source,prepared,PREFIX+'ɛ n t iː eɪ tʃ'+ENDING)['passed']
    assert not evidence.compare_source(source,prepared,PREFIX+'ɛ n eɪ dʒ'+ENDING)['passed']


def test_voiced_stop_in_an_ordinary_word_final_nz_coda_preserves_the_full_source():
    source = 'The report mentions AI today.'
    prepared = edits(source, 'AI', 'eigh eye')
    phones = PREFIX.replace('ə n z', 'ə n d z') + 'eɪ aɪ' + ENDING
    assert evidence.compare_source(source, prepared, phones)['passed']
    assert not evidence.compare_source(source, prepared, phones + ' t')['passed']
    assert not evidence.compare_source(source, prepared, phones.replace('eɪ aɪ', 'aɪ aɪ'))['passed']
    assert not evidence.compare_source(source, prepared, phones.replace('ə n d z', 'ə m d z'))['passed']


def test_word_final_stop_variant_cannot_add_d_between_named_n_and_z():
    source = 'The report mentions NZ today.'
    prepared = edits(source, 'NZ', 'N zee')
    assert evidence.compare_source(source, prepared, PREFIX + 'ɛ n z iː' + ENDING)['passed']
    assert not evidence.compare_source(source, prepared, PREFIX + 'ɛ n d z iː' + ENDING)['passed']


def test_word_final_stop_variant_does_not_apply_to_an_internal_syllable_boundary():
    assert evidence.compare_source('Enzymes.', [], 'ɛ n z aɪ m z')['passed']
    assert not evidence.compare_source('Enzymes.', [], 'ɛ n d z aɪ m z')['passed']


def test_double_you_retains_w_with_syllabic_l_without_losing_any_target_letter():
    source='The report mentions WLAN today.'
    prepared=edits(source,'WLAN','double you el eigh N')
    phones=PREFIX+'d ʌ b l̩ j uː ɛ l eɪ ɛ n'+ENDING
    assert evidence.compare_source(source,prepared,phones)['passed']
    assert not evidence.compare_source(source,prepared,phones.replace('eɪ ɛ n','ɛ n'))['passed']
    assert not evidence.compare_source(source,prepared,phones.replace('ɛ n','ɛ m'))['passed']


def test_syllabic_l_reference_never_removes_the_stressed_named_letter_l_vowel():
    source='The report mentions LL today.'
    prepared=edits(source,'LL','el el')
    assert not evidence.compare_source(source,prepared,PREFIX+'l l'+ENDING)['passed']
    source='The report mentions SSH today.'
    prepared=edits(source,'SSH','double ess H')
    assert evidence.compare_source(source,prepared,PREFIX+'d ʌ b l ɛ s eɪ tʃ'+ENDING)['passed']
    assert not evidence.compare_source(source,prepared,PREFIX+'d ʌ b l s eɪ tʃ'+ENDING)['passed']

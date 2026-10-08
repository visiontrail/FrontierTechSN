import hashlib

import pytest

from backend.pipeline import tts_pronunciation
from scripts import chatterbox_phoneme_evidence as evidence

PREFIX = 'ð ə ɹ ᵻ p ɔːɹ t m ɛ n ʃ ə n z '
ENDING = ' t ə d eɪ'


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


def test_vowel_repetitions_survive_consonant_continuation_normalization():
    assert evidence.observed_phones('aɪ aɪ ɑːɹ ɹ') == ('AY', 'AY', 'AA', 'R')


@pytest.mark.parametrize(('source', 'phones'), [
    ('UnsupportedQbitThing today.', 't ə d eɪ'),
    ('The value is 21.', 'ð ə v æ l j uː ɪ z t uː'),
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

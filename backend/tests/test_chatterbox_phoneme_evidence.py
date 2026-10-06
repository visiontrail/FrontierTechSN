import hashlib

import pytest

from backend.pipeline import tts_pronunciation
from scripts import chatterbox_phoneme_evidence as evidence

PREFIX = 'ð ə ɹ ᵻ p ɔːɹ t m ɛ n ʃ ə n z '
ENDING = ' t ə d eɪ'


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

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
    ('The value is 2.', 'ð ə v æ l j uː ɪ z t uː'),
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

"""Compare unconditioned acoustic phone observations with the complete source.

This is additional evidence, not a pronunciation dictionary or an ASR override.
Unsupported vocabulary and acoustic differences remain unverified.
"""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from pathlib import Path

import cmudict

from backend.pipeline import tts_pronunciation as pronunciation

MODEL = 'facebook/wav2vec2-lv-60-espeak-cv-ft'
REVISION = 'ae45363bf3413b374fecd9dc8bc1df0e24c3b7f4'
LETTER_PHONES = {
    'A': [('EY',)], 'B': [('B', 'IY')], 'C': [('S', 'IY')],
    'D': [('D', 'IY')], 'E': [('IY',)], 'F': [('EH', 'F')],
    'G': [('JH', 'IY')], 'H': [('EY', 'CH'), ('HH', 'EY', 'CH')],
    'I': [('AY',)], 'J': [('JH', 'EY')], 'K': [('K', 'EY')],
    'L': [('EH', 'L')], 'M': [('EH', 'M')], 'N': [('EH', 'N')],
    'O': [('OW',)], 'P': [('P', 'IY')], 'Q': [('K', 'Y', 'UW')],
    'R': [('AA', 'R')], 'S': [('EH', 'S')], 'T': [('T', 'IY')],
    'U': [('Y', 'UW')], 'V': [('V', 'IY')],
    'W': [('D', 'AH', 'B', 'AH', 'L', 'Y', 'UW')],
    'X': [('EH', 'K', 'S')], 'Y': [('W', 'AY')],
    'Z': [('Z', 'IY'), ('Z', 'EH', 'D')],
}
IPA = {
    'eɪ': 'EY', 'aɪ': 'AY', 'oʊ': 'OW', 'aʊ': 'AW', 'ɔɪ': 'OY',
    'tʃ': 'CH', 'dʒ': 'JH', 'ɜɹ': 'ER', 'əɹ': 'ER', 'ɚ': 'ER', 'ɝ': 'ER',
    'i': 'IY', 'ɪ': 'IH', 'ᵻ': 'IH', 'ɛ': 'EH', 'æ': 'AE',
    'ɑ': 'AA', 'ɔ': 'AO', 'ʌ': 'AH', 'ə': 'AH', 'ɐ': 'AH',
    'u': 'UW', 'ʊ': 'UH', 'ɜ': 'ER', 'p': 'P', 'b': 'B', 't': 'T',
    'd': 'D', 'k': 'K', 'ɡ': 'G', 'g': 'G', 'f': 'F', 'v': 'V',
    'θ': 'TH', 'ð': 'DH', 's': 'S', 'z': 'Z', 'ʃ': 'SH', 'ʒ': 'ZH',
    'h': 'HH', 'm': 'M', 'n': 'N', 'ŋ': 'NG', 'l': 'L', 'ɹ': 'R',
    'r': 'R', 'j': 'Y', 'w': 'W',
}
VOWELS = {'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW'}


@lru_cache(maxsize=1)
def lexicon():
    return cmudict.dict()


def collapse_consonant_continuations(phones: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """A compound rhotic token followed by R is one continued consonant.

    Vowel repetitions are retained. An extra letter name necessarily retains
    its vowel, including additional ess/are/eye syllables.
    """
    result = []
    for phone in phones:
        if not result or phone != result[-1] or phone in VOWELS:
            result.append(phone)
    return tuple(result)


def observed_phones(phones: str) -> tuple[str, ...]:
    keys = sorted(IPA, key=len, reverse=True)
    values = []
    for token in phones.split():
        raw = re.sub(r'[ːˈˌ]', '', token)
        cursor = 0
        while cursor < len(raw):
            match = next((key for key in keys if raw.startswith(key, cursor)), None)
            if match is None:
                raise ValueError(f'Unsupported observed phone at {cursor}: {raw[cursor:cursor + 4]}')
            values.append(IPA[match])
            cursor += len(match)
    return collapse_consonant_continuations(values)


def reference_parts(text: str, edits: list[dict]) -> list[list[tuple[str, ...]]]:
    dictionary = lexicon()
    parts = []

    def ordinary(fragment):
        if re.search(r'[^A-Za-z0-9\s.,:;!?\-\u2014\u2013]', fragment):
            raise ValueError('Source interval contains unsupported symbols')
        for match in re.finditer(r'[A-Za-z]+|\d+', fragment):
            word = match[0]
            if (len(word) == 1 and word in LETTER_PHONES
                    and ((match.start() and fragment[match.start() - 1].isdigit())
                         or (match.end() < len(fragment) and fragment[match.end()].isdigit()))):
                # A in A2A names the letter; it is not the indefinite article.
                parts.append(LETTER_PHONES[word])
                continue
            if word.isdigit():
                if len(word) != 1:
                    raise ValueError('Multi-digit source number requires a supported numeric reference')
                word = ('zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine')[int(word)]
            variants = dictionary.get(word.casefold())
            if not variants:
                raise ValueError(f'No source pronunciation reference for {word}')
            parts.append([tuple(re.sub(r'\d', '', p) for p in variant) for variant in variants])

    aliases = {name: letter for letter, names in pronunciation.LETTER_NAMES.items()
               for name in (letter.casefold(), *names) if ' ' not in name}
    cursor = 0
    for edit in sorted(edits, key=lambda row: row['start']):
        start, end = edit['start'], edit['end']
        if not cursor <= start < end <= len(text) or text[start:end] != edit['source']:
            raise ValueError('Spelling edit differs from its source span')
        if pronunciation._spoken_letters(edit['spoken']) != edit['source'].replace('.', '').upper():
            raise ValueError('Spelling edit does not retain every source letter')
        ordinary(text[cursor:start])
        words = re.findall(r'[A-Za-z]+', edit['spoken'].casefold())
        index = 0
        while index < len(words):
            if words[index:index + 2] == ['double', 'you']:
                parts.append(LETTER_PHONES['W'])
                index += 2
            elif words[index] in {'double', 'triple'}:
                ordinary(words[index])
                index += 1
            else:
                parts.append(LETTER_PHONES[aliases[words[index]]])
                index += 1
        cursor = end
    ordinary(text[cursor:])
    return parts


def compare_source(text: str, edits: list[dict], phones: str) -> dict:
    report = {'source_sha256': hashlib.sha256(text.encode()).hexdigest(),
              'raw_phones': phones, 'supported': False, 'passed': False,
              'method': 'complete_source_phone_sequence'}
    try:
        observed = observed_phones(phones)
        parts = reference_parts(text, edits)
    except (ValueError, KeyError) as error:
        report['unsupported_reason'] = str(error)
        return report
    # Prune against the observation after every source unit, rather than
    # generating the Cartesian product of all dictionary variants.
    states = {()}
    for variants in parts:
        states = {joined for prefix in states for variant in variants
                  if (joined := collapse_consonant_continuations((*prefix, *variant)))
                  == observed[:len(joined)]}
        if not states:
            break
    report.update(supported=True, observed=list(observed),
                  passed=bool(observed) and observed in states,
                  source_units=len(parts), dictionary_version=cmudict.__version__)
    return report


def audit_observation(wav: Path, text: str, edits: list[dict], observation: dict) -> dict:
    """Bind additional acoustic evidence to the actual unchanged source WAV."""
    digest = hashlib.sha256(wav.read_bytes()).hexdigest()
    if (observation.get('audio_sha256') != digest
            or observation.get('model') != MODEL
            or observation.get('model_revision') != REVISION
            or observation.get('conditioned_on_reference_text') is not False):
        raise ValueError('Phone observation differs from the pinned unconditioned waveform evidence')
    report = compare_source(text, edits, observation['phones'])
    report.update(audio_sha256=digest, model=MODEL, model_revision=REVISION,
                  conditioned_on_reference_text=False, overall_audio_acceptance=False)
    if hashlib.sha256(wav.read_bytes()).hexdigest() != digest:
        raise ValueError('Waveform changed during its phone evidence audit')
    return report

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
from backend.pipeline.tts_phone_references import (
    WORD_REFERENCES, entries as supplemental_word_references,
    policy_contract as reference_contract,
)
from scripts.chatterbox_body_phone_references import (
    entries as body_word_references, policy_contract as body_reference_contract,
)

MODEL = 'facebook/wav2vec2-lv-60-espeak-cv-ft'
REVISION = 'ae45363bf3413b374fecd9dc8bc1df0e24c3b7f4'
EVIDENCE_VERSION = 14
ORDINARY_CODA_ELISION_SOURCE = 'https://downloads.bbc.co.uk/worldservice/learningenglish/pronunciation/pdf/scripts/talkaboutenglish_pron2.pdf'
PRERHOTIC_VOWEL_SOURCE = 'https://opentext.ku.edu/amenglishpronunciation/chapter/chapter-1/'
ORDINARY_FLAP_SOURCE = 'https://journals.sagepub.com/doi/10.1177/0023830908098542'
LETTER_PHONES = {
    'A': [('EY',)], 'B': [('B', 'IY')], 'C': [('S', 'IY')],
    'D': [('D', 'IY')], 'E': [('IY',)], 'F': [('EH', 'F')],
    'G': [('JH', 'IY')], 'H': [('EY', 'CH'), ('HH', 'EY', 'CH')],
    'I': [('AY',)], 'J': [('JH', 'EY')], 'K': [('K', 'EY')],
    'L': [('EH', 'L')], 'M': [('EH', 'M')], 'N': [('EH', 'N')],
    'O': [('OW',)], 'P': [('P', 'IY')], 'Q': [('K', 'Y', 'UW')],
    'R': [('AA', 'R')], 'S': [('EH', 'S')], 'T': [('T', 'IY')],
    'U': [('Y', 'UW')], 'V': [('V', 'IY')],
    'W': [('D', 'AH', 'B', 'AH', 'L', 'Y', 'UW'), ('D', 'AH', 'B', 'L', 'Y', 'UW')],
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
    'r': 'R', 'j': 'Y', 'w': 'W', 'l̩': 'L', 'ɾ': 'DX',
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
    tokens = [re.sub(r'[ːˈˌ]', '', token) for token in phones.split()]
    for index, raw in enumerate(tokens):
        cursor = 0
        while cursor < len(raw):
            # North American /ɔr/ can be realized as [or]. Restrict the
            # observed [o] to an immediately following rhotic; standalone
            # [o] and letter-name vowels keep their distinct requirements.
            following = raw[cursor + 1:]
            if (raw[cursor] == 'o' and
                    (following.startswith(('ɹ', 'r')) or
                     (not following and index + 1 < len(tokens)
                      and tokens[index + 1].startswith(('ɹ', 'r'))))):
                values.append('AO')
                cursor += 1
                continue
            match = next((key for key in keys if raw.startswith(key, cursor)), None)
            if match is None:
                raise ValueError(f'Unsupported observed phone at {cursor}: {raw[cursor:cursor + 4]}')
            values.append(IPA[match])
            cursor += len(match)
    return collapse_consonant_continuations(values)


def reference_parts(text: str, edits: list[dict]) -> list[list[tuple[str, ...]]]:
    dictionary = lexicon()
    parts = []
    ordinary_units = []

    def ordinary(fragment):
        if (re.search(r"[^A-Za-z0-9\s.,:;!?'’\-\u2014\u2013]", fragment)
                or re.search(r"(?<![A-Za-z])['’]|['’](?![A-Za-z])", fragment)):
            raise ValueError('Source interval contains unsupported symbols')
        for match in re.finditer(r"[A-Za-z]+(?:['’][A-Za-z]+)*|\d+", fragment):
            # Use the dictionary's complete possessive/contraction entry.
            # Never discard a suffix or infer phones for an unknown word.
            word = match[0].replace('’', "'")
            if (len(word) == 1 and word in LETTER_PHONES
                    and ((match.start() and fragment[match.start() - 1].isdigit())
                         or (match.end() < len(fragment) and fragment[match.end()].isdigit()))):
                # A in A2A names the letter; it is not the indefinite article.
                parts.append(LETTER_PHONES[word])
                ordinary_units.append(False)
                continue
            if word.isdigit():
                if len(word) != 1:
                    raise ValueError('Multi-digit source number requires a supported numeric reference')
                word = ('zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine')[int(word)]
            variants = dictionary.get(word.casefold()) or []
            reference = supplemental_word_references().get(word.casefold())
            if reference:
                variants = [*variants, *reference['variants']]
            body_reference = body_word_references().get(word.casefold())
            if body_reference:
                variants = [*variants, *body_reference['variants']]
            if not variants:
                raise ValueError(f'No source pronunciation reference for {word}')
            references = {tuple(re.sub(r'\d', '', p) for p in variant) for variant in variants}
            # Only an unstressed word-final schwa before L can merge into
            # syllabic L. Named letter vowels (e.g. L /ɛl/) never lose theirs.
            for variant in variants:
                if (len(variant) >= 3 and variant[-2:] == ['AH0', 'L']
                        and re.sub(r'\d', '', variant[-3]) not in VOWELS):
                    references.add(tuple(re.sub(r'\d', '', p) for p in (*variant[:-2], 'L')))
                # American English /t,d/ can flap between vowels before an
                # unstressed syllable. Keep the flap distinct from T/D and
                # offer it only inside an ordinary dictionary word; named
                # letters never gain a flap pronunciation or lose a vowel.
                plain = tuple(re.sub(r'\d', '', p) for p in variant)
                for index in range(1, len(variant) - 1):
                    if (plain[index] in {'T', 'D'} and plain[index - 1] in VOWELS
                            and plain[index + 1] in VOWELS and variant[index + 1].endswith('0')):
                        references.add((*plain[:index], 'DX', *plain[index + 1:]))
            # A word-internal nasal/fricative transition may include the
            # homorganic stop (n-sh -> n-t-sh, decoded as n-ch). Apply only
            # inside dictionary words, never to independently named letters.
            for variant in list(references):
                for index in range(len(variant) - 1):
                    if variant[index:index + 2] == ('N', 'SH'):
                        references.add((*variant[:index], 'N', 'CH', *variant[index + 2:]))
                # A voiced stop can occur in a word-final /nz/ coda, e.g.
                # mentions -> mention[d]s. Restrict this reference alternative
                # to the dictionary word's final cluster: neither a syllable
                # boundary inside a word nor a sequence of named letters
                # authorizes an extra D. All vowels and closing words remain.
                if variant[-2:] == ('N', 'Z'):
                    references.add((*variant[:-2], 'N', 'D', 'Z'))
                # Natural /t/ elision in an ordinary word-final /kts/
                # cluster (connects -> connec[s]). All source vowels and
                # the plural /s/ remain. Spelled letters use separate units,
                # so their T /ti:/ can never enter this rule.
                if variant[-3:] == ('K', 'T', 'S'):
                    references.add((*variant[:-3], 'K', 'S'))
            parts.append(sorted(references))
            ordinary_units.append(True)

    aliases = {name: letter for letter, names in pronunciation.LETTER_NAMES.items()
               for name in (letter.casefold(), *names) if ' ' not in name}
    cursor = 0
    for edit in sorted(edits, key=lambda row: row['start']):
        start, end = edit['start'], edit['end']
        if not cursor <= start < end <= len(text) or text[start:end] != edit['source']:
            raise ValueError('Spelling edit differs from its source span')
        if edit.get('kind') == 'word':
            from backend.pipeline.tts_word_pronunciation import validate_word_hint
            phones = validate_word_hint(edit['source'], edit['spoken'], edit.get('phones'))
            ordinary(text[cursor:start])
            parts.append([tuple(phones)])
            ordinary_units.append(True)
            cursor = end
            continue
        if pronunciation._spoken_letters(edit['spoken']) != edit['source'].replace('.', '').upper():
            raise ValueError('Spelling edit does not retain every source letter')
        ordinary(text[cursor:start])
        words = re.findall(r'[A-Za-z]+', edit['spoken'].casefold())
        index = 0
        while index < len(words):
            if words[index:index + 2] == ['double', 'you']:
                parts.append(LETTER_PHONES['W'])
                ordinary_units.append(False)
                index += 2
            elif words[index] in {'double', 'triple'}:
                ordinary(words[index])
                index += 1
            else:
                parts.append(LETTER_PHONES[aliases[words[index]]])
                ordinary_units.append(False)
                index += 1
        cursor = end
        suffix = re.match(r"(?:['’]s|s)(?![A-Za-z0-9'’])", text[end:])
        if suffix:
            # Regular plurals/genitives of named letters retain the complete
            # final letter, then /s/, /z/, or /iz/ according to its last phone.
            # These endings do not authorize another S letter name or permit
            # removal of the source suffix. Word readings use their dictionary.
            endings = set()
            for variant in parts[-1]:
                if variant[-1] in {'S', 'Z', 'SH', 'ZH', 'CH', 'JH'}:
                    endings.update({('IH', 'Z'), ('AH', 'Z')})
                elif variant[-1] in {'P', 'T', 'K', 'F', 'TH'}:
                    endings.add(('S',))
                else:
                    endings.add(('Z',))
            parts.append(sorted(endings))
            ordinary_units.append(False)
            cursor += suffix.end()
    ordinary(text[cursor:])
    # A dictionary word's coda /z/ can devoice before the next word's /s/.
    # Keep the source word and following onset; consonant-continuation
    # normalization merges the resulting s-s, never a named-letter vowel.
    for index in range(len(parts) - 1):
        if (ordinary_units[index] and ordinary_units[index + 1]
                and all(variant[0] == 'S' for variant in parts[index + 1])):
            variants = set(parts[index])
            variants.update((*variant[:-1], 'S') for variant in parts[index] if variant[-1] == 'Z')
            parts[index] = sorted(variants)
    return parts


def compare_source(text: str, edits: list[dict], phones: str) -> dict:
    report = {'source_sha256': hashlib.sha256(text.encode()).hexdigest(),
              'raw_phones': phones, 'supported': False, 'passed': False,
              'method': 'complete_source_phone_sequence', 'evidence_version': EVIDENCE_VERSION}
    if re.search(r'o[ːˈˌ]*(?:\s+)?[ɹr]', phones):
        report['prerhotic_vowel_reference'] = PRERHOTIC_VOWEL_SOURCE
    if 'ɾ' in phones:
        report['ordinary_flap_reference'] = ORDINARY_FLAP_SOURCE
    word_refs = [{key: edit[key] for key in ('source', 'start', 'end', 'phones')}
                 for edit in edits if edit.get('kind') == 'word']
    if word_refs:
        report['word_hint_references'] = word_refs
    unedited = text
    for edit in sorted(edits, key=lambda row: row['start'], reverse=True):
        unedited = unedited[:edit['start']] + ' ' + unedited[edit['end']:]
    cited = {word.casefold(): supplemental_word_references()[word.casefold()]
             for word in re.findall(r'[A-Za-z]+', unedited)
             if word.casefold() in supplemental_word_references()}
    if cited:
        report['cited_word_references'] = cited
        report['word_reference_sha256'] = reference_contract()['reference_sha256']
    body_cited = {word.casefold(): body_word_references()[word.casefold()]
                  for word in re.findall(r'[A-Za-z]+', unedited)
                  if word.casefold() in body_word_references()}
    if body_cited:
        report['cited_body_word_references'] = body_cited
        report['body_reference_sha256'] = body_reference_contract()['reference_sha256']
    if any(tuple(re.sub(r'\d', '', phone) for phone in variant)[-3:] == ('K', 'T', 'S')
           for word in re.findall(r'[A-Za-z]+', unedited)
           for variant in (lexicon().get(word.casefold()) or [])):
        report['ordinary_coda_elision_reference'] = ORDINARY_CODA_ELISION_SOURCE
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

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

from backend.pipeline import tts, tts_pronunciation as pronunciation
from backend.spoken_numbers import integer_words
from backend.pipeline.tts_phone_references import (
    WORD_REFERENCES, entries as supplemental_word_references,
    policy_contract as reference_contract,
)
from scripts.chatterbox_body_phone_references import (
    entries as body_word_references, policy_contract as body_reference_contract,
    reference_for as body_reference_for,
)

MODEL = 'facebook/wav2vec2-lv-60-espeak-cv-ft'
REVISION = 'ae45363bf3413b374fecd9dc8bc1df0e24c3b7f4'
EVIDENCE_VERSION = 29
SYMBOL_CONJUNCTION_REFERENCE_VERSION = 1
SYMBOL_INITIALISM_RE = re.compile(r'\b([A-Z])&([A-Z])\b')
CONTEXTUAL_CLUSTER_REFERENCE_VERSION = 1
CONTEXTUAL_CLUSTER_REFERENCE_SOURCE = 'https://downloads.bbc.co.uk/worldservice/learningenglish/pronunciation/pdf/scripts/talkaboutenglish_pron2.pdf'
PROGRAMMING_REFERENCES = {
    'C++': 'https://stroustrup.com/bs_faq.html',
    'C#': 'https://dotnet.microsoft.com/en-us/languages',
}
NUMERIC_REFERENCE_SOURCE = 'https://dictionaryblog.cambridge.org/tag/numbers/'
SOURCE_WORD_REFERENCE_POLICY_VERSION = 1
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


def ordinary_source_words(fragment: str) -> list[str]:
    """Keep a cited hyphenated word whole; retain prior splitting otherwise."""
    words = []
    for match in re.finditer(r"[A-Za-z]+(?:['’][A-Za-z]+)*(?:-[A-Za-z]+)*|\d+", fragment):
        word = match[0].replace('’', "'")
        words.extend([word] if word.casefold() in body_word_references() else word.split('-'))
    return words


def independent_word_reference(word: str, *, word_hint_version=4) -> dict:
    """The cue's own fragments cannot establish how its source word is read."""
    variants = lexicon().get(word.casefold()) or []
    if word_hint_version not in {1, 2, 3, 4, 5, 6}:
        raise ValueError('Unsupported independent word-hint reference version')
    reference_version = 3 if word_hint_version >= 6 else 2 if word_hint_version >= 5 else 1
    reference = supplemental_word_references(version=reference_version).get(word.casefold())
    if reference:
        variants = [*variants, *reference['variants']]
    body = body_reference_for(word)
    if body:
        variants = (body['variants'] if body.get('replace_dictionary') else [*variants, *body['variants']])
    if not variants:
        raise ValueError(f'No independent source-word pronunciation reference for {word}')
    result = {'source': word, 'variants': sorted({
        collapse_consonant_continuations(tuple(re.sub(r'\d', '', phone) for phone in variant))
        for variant in variants}), 'dictionary_version': cmudict.__version__}
    if body or reference:
        selected = body or reference
        result.update(source_url=selected['source_url'], source_location=selected['source_location'])
    if word_hint_version >= 5:
        result['supplemental_word_reference_contract'] = reference_contract(version=reference_version)
    return result


def source_number_words(literal: str) -> list[str]:
    """Keep a bounded written value and its decimal precision intact."""
    if not re.fullmatch(r'[0-9]{1,6}(?:\.[0-9]{1,6})?', literal):
        raise ValueError('Source number exceeds the supported numeric reference bounds')
    whole, dot, fraction = literal.partition('.')
    digits = ('zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine')
    words = ([digits[int(digit)] for digit in whole] if len(whole) > 1 and whole.startswith('0')
             else integer_words(int(whole)).replace('-', ' ').split())
    return words + (['point', *(digits[int(digit)] for digit in fraction)] if dot else [])


def source_number_readings(literal: str) -> list[list[str]]:
    """Additional identifier/year groups must round-trip to the same value."""
    forms = [source_number_words(literal)]
    if len(literal) == 4 and literal.isascii() and literal.isdigit() and not literal.startswith('0'):
        first, last = int(literal[:2]), int(literal[2:])
        if first >= 10 and last >= 10:
            paired = [*source_number_words(literal[:2]), *source_number_words(literal[2:])]
            if tts._lexical_tokens(' '.join(paired)) == [literal]:
                forms.append(paired)
        elif first >= 10 and last == 0:
            hundred = [*source_number_words(literal[:2]), 'hundred']
            if tts._lexical_tokens(' '.join(hundred)) == [literal]:
                forms.append(hundred)
    return forms


def reference_parts(text: str, edits: list[dict], *, contextual_word_cluster=False) -> list[list[tuple[str, ...]]]:
    dictionary = lexicon()
    parts = []
    ordinary_units = []
    before_vowel_references = {}

    def ordinary(fragment):
        symbolic = list(SYMBOL_INITIALISM_RE.finditer(fragment))
        if symbolic:
            protected = [(row.start(), row.end()) for row in pronunciation.PROTECTED_TEXT_RE.finditer(fragment)]
            cursor = 0
            for match in symbolic:
                if any(start < match.end() and end > match.start() for start, end in protected):
                    raise ValueError('Protected source text cannot become a symbol initialism')
                ordinary(fragment[cursor:match.start()])
                parts.append(LETTER_PHONES[match[1]])
                ordinary_units.append(False)
                ordinary('and')
                parts.append(LETTER_PHONES[match[2]])
                ordinary_units.append(False)
                cursor = match.end()
            ordinary(fragment[cursor:])
            return
        if re.search(r'\d\.\d+\.\d', fragment):
            raise ValueError('Multi-component source version requires an explicit reading reference')
        if (re.search(r"[^A-Za-z0-9\s.,:;!?'’\-\u2014\u2013]", fragment)
                or re.search(r"(?<![A-Za-z])['’]|['’](?![A-Za-z])", fragment)):
            raise ValueError('Source interval contains unsupported symbols')
        for match in re.finditer(r"[A-Za-z]+(?:['’][A-Za-z]+)*(?:-[A-Za-z]+)*|\d+(?:\.\d+)?", fragment):
            # Use the dictionary's complete possessive/contraction entry.
            # Never discard a suffix or infer phones for an unknown word.
            word = match[0].replace('’', "'")
            if '-' in word and word.casefold() not in body_word_references():
                ordinary(word.replace('-', ' '))
                continue
            if (len(word) == 1 and word in LETTER_PHONES
                    and ((match.start() and fragment[match.start() - 1].isdigit())
                         or (match.end() < len(fragment) and fragment[match.end()].isdigit()))):
                # A in A2A names the letter; it is not the indefinite article.
                parts.append(LETTER_PHONES[word])
                ordinary_units.append(False)
                continue
            if re.fullmatch(r'\d+(?:\.\d+)?', word):
                if len(source_number_readings(word)) == 1:
                    ordinary(' '.join(source_number_words(word)))
                else:
                    combined = set()
                    for form in source_number_readings(word):
                        states = {()}
                        for number_word in form:
                            choices = dictionary.get(number_word) or []
                            if not choices:
                                raise ValueError('Source numeric word lacks a dictionary reference')
                            states = {(*prefix, *(re.sub(r'\d', '', phone) for phone in variant))
                                      for prefix in states for variant in choices}
                        combined.update(states)
                    parts.append(sorted(combined))
                    ordinary_units.append(True)
                continue
            variants = dictionary.get(word.casefold()) or []
            reference = supplemental_word_references().get(word.casefold())
            if reference:
                variants = [*variants, *reference['variants']]
            body_reference = body_reference_for(word)
            if body_reference:
                variants = (body_reference['variants'] if body_reference.get('replace_dictionary') else
                            [*variants, *body_reference['variants']])
                if body_reference.get('before_vowel_variants'):
                    before_vowel_references[len(parts)] = body_reference['before_vowel_variants']
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
        if edit.get('kind') == 'programming':
            if (edit['source'] not in PROGRAMMING_REFERENCES
                    or edit['spoken'] != pronunciation.PROGRAMMING_CUES[edit['source']]):
                raise ValueError('Programming edit differs from the independently cited source reading')
            ordinary(text[cursor:start])
            parts.append(LETTER_PHONES['C'])
            ordinary_units.append(False)
            ordinary('plus plus' if edit['source'] == 'C++' else 'sharp')
            cursor = end
            continue
        if edit.get('kind') == 'number':
            if edit['spoken'] != pronunciation.numeric_cue(edit['source']):
                raise ValueError('Numeric edit differs from its exact source value')
            ordinary(text[cursor:start])
            ordinary(edit['source'])
            cursor = end
            continue
        if edit.get('kind') == 'word':
            from backend.pipeline.tts_word_pronunciation import validate_word_hint, validate_homophone_hint
            hint_version = edit.get('word_hint_version', 4)
            hint_type = edit.get('word_hint_type')
            if hint_type not in {None, 'homophone'}:
                raise ValueError('Unsupported word-hint type')
            validate_hint = validate_homophone_hint if hint_type == 'homophone' else validate_word_hint
            phones = validate_hint(edit['source'], edit['spoken'], edit.get('phones'), version=hint_version)
            canonical = independent_word_reference(edit['source'], word_hint_version=hint_version)
            if collapse_consonant_continuations(phones) not in canonical['variants']:
                raise ValueError('Word hint differs from the independent source-word pronunciation reference')
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
    # A cited weak article form depends on the following spoken onset, not
    # its written first letter. Named N /en/ starts with a vowel; U /you/
    # starts with a consonant. Never alter any named-letter reference.
    for index, variants in before_vowel_references.items():
        if index + 1 < len(parts) and all(variant[0] in VOWELS for variant in parts[index + 1]):
            parts[index] = sorted({*parts[index], *(tuple(variant) for variant in variants)})
    # A dictionary word's coda /z/ can devoice before the next word's /s/.
    # Keep the source word and following onset; consonant-continuation
    # normalization merges the resulting s-s, never a named-letter vowel.
    for index in range(len(parts) - 1):
        if (ordinary_units[index] and ordinary_units[index + 1]
                and all(variant[0] == 'S' for variant in parts[index + 1])):
            variants = set(parts[index])
            variants.update((*variant[:-1], 'S') for variant in parts[index] if variant[-1] == 'Z')
            parts[index] = sorted(variants)
    if contextual_word_cluster:
        # BBC's connected-speech reference permits /d/ elision between
        # consonants. This conservative subset requires an ordinary word's
        # final consonant-D cluster before an ordinary word starting in T.
        # Named letters are separate non-ordinary units and remain exact.
        for index in range(len(parts) - 1):
            if (ordinary_units[index] and ordinary_units[index + 1]
                    and all(variant[0] == 'T' for variant in parts[index + 1])):
                variants = set(parts[index])
                variants.update(variant[:-1] for variant in parts[index]
                                if len(variant) > 1 and variant[-1] == 'D'
                                and variant[-2] not in VOWELS)
                parts[index] = sorted(variants)
    return parts


def compare_contextual_word_cluster(text: str, edits: list[dict], phones: str,
                                    primary_words: list[dict], secondary_words: list[dict]) -> dict:
    """Supplement a phonetic difference only when both independent ASRs retain the exact source."""
    standard = compare_source(text, edits, phones)
    expected = tts._lexical_tokens(text)
    transcripts = [' '.join(str(row.get('text', '')) for row in words)
                   for words in (primary_words, secondary_words)]
    exact = [tts._lexical_tokens(transcript) == expected for transcript in transcripts]
    report = {'source_sha256': hashlib.sha256(text.encode()).hexdigest(),
              'method': 'complete_source_with_dual_exact_word_cluster_reference',
              'contextual_cluster_reference_version': CONTEXTUAL_CLUSTER_REFERENCE_VERSION,
              'source_url': CONTEXTUAL_CLUSTER_REFERENCE_SOURCE,
              'source_location': 'BBC Talk about English, 11 July 2005, page 6: word-final t and d may disappear between consonant sounds. This subset allows only ordinary consonant-D before ordinary T, with both recognizers retaining every source word.',
              'standard_phone_report': standard, 'exact_dual_source_transcripts': exact,
              'transcript_sha256': [hashlib.sha256(value.encode()).hexdigest() for value in transcripts],
              'supported': False, 'passed': False, 'overall_audio_acceptance': False}
    if standard['passed'] or not standard['supported'] or not all(exact):
        return report
    try:
        observed = observed_phones(phones)
        parts = reference_parts(text, edits, contextual_word_cluster=True)
    except (ValueError, KeyError) as error:
        report['unsupported_reason'] = str(error)
        return report
    states = {()}
    for variants in parts:
        states = {joined for prefix in states for variant in variants
                  if (joined := collapse_consonant_continuations((*prefix, *variant)))
                  == observed[:len(joined)]}
        if not states:
            break
    report.update(supported=True, passed=bool(observed) and observed in states,
                  observed=list(observed), source_units=len(parts))
    return report


def compare_source(text: str, edits: list[dict], phones: str) -> dict:
    report = {'source_sha256': hashlib.sha256(text.encode()).hexdigest(),
              'raw_phones': phones, 'supported': False, 'passed': False,
              'method': 'complete_source_phone_sequence', 'evidence_version': EVIDENCE_VERSION}
    if SYMBOL_INITIALISM_RE.search(text):
        report['source_symbol_conjunction_reference_version'] = SYMBOL_CONJUNCTION_REFERENCE_VERSION
    if re.search(r'o[ːˈˌ]*(?:\s+)?[ɹr]', phones):
        report['prerhotic_vowel_reference'] = PRERHOTIC_VOWEL_SOURCE
    if 'ɾ' in phones:
        report['ordinary_flap_reference'] = ORDINARY_FLAP_SOURCE
    word_refs = [pronunciation.word_hint_reference(edit)
                 for edit in edits if edit.get('kind') == 'word']
    programming_refs = {edit['source']: PROGRAMMING_REFERENCES[edit['source']]
                        for edit in edits if edit.get('kind') == 'programming'
                        and edit['source'] in PROGRAMMING_REFERENCES}
    if programming_refs:
        report['programming_reading_references'] = programming_refs
    if word_refs:
        report['word_hint_references'] = word_refs
        report['source_word_reference_policy_version'] = SOURCE_WORD_REFERENCE_POLICY_VERSION
    if re.search(r'\d{2}|\d\.\d', text):
        report['numeric_reference_source'] = NUMERIC_REFERENCE_SOURCE
    unedited = text
    for edit in sorted(edits, key=lambda row: row['start'], reverse=True):
        unedited = unedited[:edit['start']] + ' ' + unedited[edit['end']:]
    cited = {word.casefold(): supplemental_word_references()[word.casefold()]
             for word in re.findall(r'[A-Za-z]+', unedited)
             if word.casefold() in supplemental_word_references()}
    if cited:
        report['cited_word_references'] = cited
        report['word_reference_sha256'] = reference_contract()['reference_sha256']
    body_cited = {word.casefold(): body_reference_for(word)
                  for word in ordinary_source_words(unedited)
                  if body_reference_for(word)}
    if body_cited:
        report['cited_body_word_references'] = body_cited
        report['body_reference_sha256'] = body_reference_contract()['reference_sha256']
    if word_refs:
        try:
            report['independent_word_references'] = {
                edit['source']: independent_word_reference(edit['source'], word_hint_version=edit.get('word_hint_version', 4)) for edit in word_refs}
        except ValueError as error:
            report['unsupported_reason'] = str(error)
            return report
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


def audit_contextual_word_cluster_observation(wav: Path, text: str, edits: list[dict],
                                             observation: dict, primary_words: list[dict],
                                             secondary_words: list[dict], *, recognizer_bindings: list[dict]) -> dict:
    """Bind the supplemental phonetic reference to two unconditioned recognizers on this WAV."""
    standard = audit_observation(wav, text, edits, observation)
    digest = standard['audio_sha256']
    source_digest = standard['source_sha256']
    if (not isinstance(recognizer_bindings, list) or len(recognizer_bindings) != 2
            or any(not isinstance(row, dict) or not isinstance(row.get('model'), str)
                   or not row['model'] or row.get('audio_sha256') != digest
                   or row.get('source_sha256') != source_digest
                   or row.get('conditioned_on_reference_text') is not False
                   for row in recognizer_bindings)
            or recognizer_bindings[0]['model'] == recognizer_bindings[1]['model']):
        raise ValueError('Contextual source audit requires two independent unconditioned recognizers bound to this waveform')
    transcripts = [' '.join(str(row.get('text', '')) for row in words)
                   for words in (primary_words, secondary_words)]
    if any(binding.get('transcript_sha256') != hashlib.sha256(transcript.encode()).hexdigest()
           for binding, transcript in zip(recognizer_bindings, transcripts, strict=True)):
        raise ValueError('Contextual recognizer transcript differs from its waveform binding')
    report = compare_contextual_word_cluster(text, edits, observation['phones'], primary_words, secondary_words)
    report.update(audio_sha256=digest, model=MODEL, model_revision=REVISION,
                  conditioned_on_reference_text=False, recognizer_bindings=recognizer_bindings,
                  overall_audio_acceptance=False)
    if hashlib.sha256(wav.read_bytes()).hexdigest() != digest:
        raise ValueError('Waveform changed during its contextual source audit')
    return report

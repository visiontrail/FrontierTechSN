"""Validate spelling-preserving word hints without selecting entity readings."""

import hashlib
from functools import lru_cache
import re

import cmudict

from backend.pipeline.tts_phone_references import PHONES


WORD_HINT_VERSION = 4


@lru_cache(maxsize=1)
def lexicon():
    return cmudict.dict()


@lru_cache(maxsize=5)
def policy_contract(*, version=WORD_HINT_VERSION):
    if version not in {1, 2, 3, 4, 5}:
        raise ValueError('Unsupported word-hint policy')
    with cmudict.dict_stream() as stream:
        dictionary_sha256 = hashlib.sha256(stream.read()).hexdigest()
    result = {'version': version, 'dictionary_version': cmudict.__version__,
              'dictionary_sha256': dictionary_sha256}
    if version >= 4:
        from backend.pipeline.tts_phone_references import policy_contract as reference_contract
        result['supplemental_word_references'] = reference_contract(version=2 if version >= 5 else 1)
    return result


def normalized(variant):
    return tuple(re.sub(r'\d', '', phone) for phone in variant)


def validate_homophone_hint(source, spoken, phones, *, version=5):
    """Allow an ordinary English cue only with identical independently known source phones."""
    if version != 5 or not re.fullmatch(r'[A-Za-z]{3,32}', source):
        raise ValueError('A phonetic cue requires the explicit version5 source-word policy')
    if (not isinstance(spoken, str) or len(spoken) > 128
            or not re.fullmatch(r'[A-Za-z][A-Za-z ,.-]*', spoken)):
        raise ValueError('A phonetic cue must contain ordinary English words and separators')
    words = re.findall(r'[A-Za-z]+', spoken.casefold())
    if not 1 <= len(words) <= 8 or ''.join(words) == source.casefold():
        raise ValueError('A phonetic cue must differ from the source spelling')
    if (not isinstance(phones, list) or not 1 <= len(phones) <= 64
            or any(not isinstance(phone, str) or phone not in PHONES for phone in phones)):
        raise ValueError('A phonetic cue requires explicit valid ARPABET phones')
    from backend.pipeline.tts_phone_references import entries
    dictionary = lexicon()
    reference = entries(version=2).get(source.casefold())
    canonical = [*(dictionary.get(source.casefold()) or []), *(reference['variants'] if reference else [])]
    wanted = tuple(phones)
    if not canonical or wanted not in {normalized(choice) for choice in canonical}:
        raise ValueError('A phonetic cue requires an independent complete-source pronunciation reference')
    states = {()}
    for word in words:
        choices = dictionary.get(word) or []
        if not choices:
            raise ValueError('Every phonetic cue word needs an ordinary dictionary pronunciation')
        states = {joined for prefix in states for choice in choices
                  if (joined := (*prefix, *normalized(choice))) == wanted[:len(joined)]}
    if wanted not in states:
        raise ValueError('Phonetic cue words must match every source phone without additions or omissions')
    return list(wanted)


def validate_word_hint(source, spoken, phones, *, version=WORD_HINT_VERSION):
    if version not in {1, 2, 3, 4, 5}:
        raise ValueError('Unsupported word-hint policy')
    pattern = r'[A-Za-z]{3,32}' if version >= 3 else r'[A-Z]{3,32}'
    if not re.fullmatch(pattern, source):
        raise ValueError('A word hint requires an undotted alphabetic word of at least three letters')
    if (not isinstance(spoken, str) or len(spoken) > 256
            or not re.fullmatch(r'[A-Za-z][A-Za-z ,.-]*', spoken)
            or (version == 1 and not spoken[-1].isalpha())):
        raise ValueError('A word hint can contain only words and plain separators')
    words = re.findall(r'[A-Za-z]+', spoken.casefold())
    if ''.join(words) != source.casefold():
        raise ValueError('A word hint must preserve every source letter in its original order')
    dictionary = lexicon()
    def choices(word):
        variants = dictionary.get(word.casefold()) or []
        if version >= 4:
            from backend.pipeline.tts_phone_references import entries
            reference = entries(version=2 if version >= 5 else 1).get(word.casefold())
            if reference:
                variants = [*variants, *reference['variants']]
        return variants

    canonical = choices(source)
    single_word_hint = (version >= 2 and len(words) == 1
                        and bool(canonical)
                        and spoken.rstrip().endswith((',', '.')))
    if ((len(words) < 2 and not single_word_hint)
            or not any(len(word) > 1 for word in words)):
        raise ValueError('A word hint must separate word fragments, not spell every individual letter')
    if (not isinstance(phones, list) or not 1 <= len(phones) <= 64
            or any(not isinstance(phone, str) or phone not in PHONES for phone in phones)):
        raise ValueError('A word hint requires an explicit valid ARPABET phone sequence')
    variants = [choices(word) for word in words]
    if not all(variants):
        raise ValueError('Every word-hint fragment needs an English dictionary pronunciation')
    wanted = tuple(phones)
    # Prune the small reference lattice against the proposed sequence; never
    # materialize an unbounded Cartesian product from model-controlled text.
    states = {()}
    for choices in variants:
        states = {joined for prefix in states for choice in choices
                  if (joined := (*prefix, *normalized(choice))) == wanted[:len(joined)]}
        if not states:
            break
    if wanted not in states:
        raise ValueError('Declared word phones do not match the spelling-preserving hint fragments')
    if canonical and wanted not in {normalized(choice) for choice in canonical}:
        raise ValueError('A known source word must retain its dictionary pronunciation')
    return list(wanted)

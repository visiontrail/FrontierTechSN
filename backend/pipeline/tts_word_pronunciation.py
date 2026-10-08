"""Validate spelling-preserving word hints without selecting entity readings."""

import hashlib
from functools import lru_cache
import re

import cmudict


WORD_HINT_VERSION = 3
PHONES = frozenset('AA AE AH AO AW AY B CH D DH EH ER EY F G HH IH IY JH K L M N NG OW OY P R S SH T TH UH UW V W Y Z ZH'.split())


@lru_cache(maxsize=1)
def lexicon():
    return cmudict.dict()


@lru_cache(maxsize=2)
def policy_contract(*, version=WORD_HINT_VERSION):
    if version not in {1, 2, 3}:
        raise ValueError('Unsupported word-hint policy')
    with cmudict.dict_stream() as stream:
        dictionary_sha256 = hashlib.sha256(stream.read()).hexdigest()
    return {'version': version, 'dictionary_version': cmudict.__version__,
            'dictionary_sha256': dictionary_sha256}


def normalized(variant):
    return tuple(re.sub(r'\d', '', phone) for phone in variant)


def validate_word_hint(source, spoken, phones, *, version=WORD_HINT_VERSION):
    if version not in {1, 2, 3}:
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
    single_word_hint = (version >= 2 and len(words) == 1
                        and bool(dictionary.get(source.casefold()))
                        and spoken.rstrip().endswith((',', '.')))
    if ((len(words) < 2 and not single_word_hint)
            or not any(len(word) > 1 for word in words)):
        raise ValueError('A word hint must separate word fragments, not spell every individual letter')
    if (not isinstance(phones, list) or not 1 <= len(phones) <= 64
            or any(not isinstance(phone, str) or phone not in PHONES for phone in phones)):
        raise ValueError('A word hint requires an explicit valid ARPABET phone sequence')
    variants = [dictionary.get(word) for word in words]
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
    canonical = dictionary.get(source.casefold())
    if canonical and wanted not in {normalized(choice) for choice in canonical}:
        raise ValueError('A known source word must retain its dictionary pronunciation')
    return list(wanted)

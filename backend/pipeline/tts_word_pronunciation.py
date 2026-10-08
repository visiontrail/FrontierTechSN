"""Validate spelling-preserving word hints without selecting entity readings."""

import hashlib
from functools import lru_cache
import re

import cmudict


WORD_HINT_VERSION = 1
PHONES = frozenset('AA AE AH AO AW AY B CH D DH EH ER EY F G HH IH IY JH K L M N NG OW OY P R S SH T TH UH UW V W Y Z ZH'.split())


@lru_cache(maxsize=1)
def lexicon():
    return cmudict.dict()


@lru_cache(maxsize=1)
def policy_contract():
    with cmudict.dict_stream() as stream:
        dictionary_sha256 = hashlib.sha256(stream.read()).hexdigest()
    return {'version': WORD_HINT_VERSION, 'dictionary_version': cmudict.__version__,
            'dictionary_sha256': dictionary_sha256}


def normalized(variant):
    return tuple(re.sub(r'\d', '', phone) for phone in variant)


def validate_word_hint(source, spoken, phones):
    if not re.fullmatch(r'[A-Z]{3,32}', source):
        raise ValueError('A word hint requires an undotted uppercase word of at least three letters')
    if (not isinstance(spoken, str) or len(spoken) > 256
            or not re.fullmatch(r'[A-Za-z][A-Za-z ,.-]*[A-Za-z]', spoken)):
        raise ValueError('A word hint can contain only words and plain separators')
    words = re.findall(r'[A-Za-z]+', spoken.casefold())
    if ''.join(words) != source.casefold():
        raise ValueError('A word hint must preserve every source letter in its original order')
    if len(words) < 2 or not any(len(word) > 1 for word in words):
        raise ValueError('A word hint must separate word fragments, not spell every individual letter')
    if (not isinstance(phones, list) or not 1 <= len(phones) <= 64
            or any(not isinstance(phone, str) or phone not in PHONES for phone in phones)):
        raise ValueError('A word hint requires an explicit valid ARPABET phone sequence')
    dictionary = lexicon()
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

"""Load cited English word phones without rewriting or selecting source text."""

import hashlib
import json
from functools import lru_cache
from pathlib import Path
import re

PHONES = frozenset('AA AE AH AO AW AY B CH D DH EH ER EY F G HH IH IY JH K L M N NG OW OY P R S SH T TH UH UW V W Y Z ZH'.split())
WORD_REFERENCES = Path(__file__).with_name('english_word_phone_references.json')


@lru_cache(maxsize=1)
def _loaded_references():
    raw = WORD_REFERENCES.read_bytes()
    payload = json.loads(raw)
    if (not isinstance(payload, dict) or payload.get('version') != 1
            or not isinstance(payload.get('entries'), dict)):
        raise ValueError('Unsupported supplemental word reference data')
    for word, row in payload['entries'].items():
        if (not re.fullmatch(r'[a-z]{3,32}', word)
                or not isinstance(row, dict)
                or not isinstance(row.get('source_url'), str)
                or not row['source_url'].startswith('https://')
                or not isinstance(row.get('source_location'), str) or not row['source_location']
                or not isinstance(row.get('variants'), list) or not row['variants']
                or any(not isinstance(variant, list) or not variant
                       or any(not isinstance(phone, str) or phone not in PHONES for phone in variant)
                       for variant in row['variants'])):
            raise ValueError('Invalid cited word phone reference')
    return payload['entries'], hashlib.sha256(raw).hexdigest()


def entries():
    return _loaded_references()[0]


def policy_contract():
    return {'version': 1, 'reference_sha256': _loaded_references()[1]}

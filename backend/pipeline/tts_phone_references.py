"""Load cited English word phones without rewriting or selecting source text."""

import hashlib
import json
from functools import lru_cache
from pathlib import Path
import re

PHONES = frozenset('AA AE AH AO AW AY B CH D DH EH ER EY F G HH IH IY JH K L M N NG OW OY P R S SH T TH UH UW V W Y Z ZH'.split())
WORD_REFERENCES = Path(__file__).with_name('english_word_phone_references.json')
WORD_REFERENCES_V2 = Path(__file__).with_name('english_word_phone_references_v2.json')
WORD_REFERENCES_V3 = Path(__file__).with_name('english_word_phone_references_v3.json')


@lru_cache(maxsize=3)
def _loaded_references(version=1):
    if version not in {1, 2, 3}:
        raise ValueError('Unsupported supplemental word reference version')
    raw = {1:WORD_REFERENCES, 2:WORD_REFERENCES_V2, 3:WORD_REFERENCES_V3}[version].read_bytes()
    payload = json.loads(raw)
    if (not isinstance(payload, dict) or payload.get('version') != version
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


def entries(*, version=1):
    return _loaded_references(version)[0]


def policy_contract(*, version=1):
    return {'version': version, 'reference_sha256': _loaded_references(version)[1]}

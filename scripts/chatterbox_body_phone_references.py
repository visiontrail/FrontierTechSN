"""Supplement ordinary source words without changing frozen word-hint contracts."""

import hashlib
import json
from functools import lru_cache
from pathlib import Path
import re

from backend.pipeline.tts_phone_references import PHONES

BODY_REFERENCES = Path(__file__).resolve().parents[1] / 'backend/pipeline/english_body_phone_references.json'


@lru_cache(maxsize=1)
def _loaded():
    raw = BODY_REFERENCES.read_bytes()
    payload = json.loads(raw)
    if (not isinstance(payload, dict) or payload.get('version') != 1
            or not isinstance(payload.get('entries'), dict)):
        raise ValueError('Unsupported ordinary-word reference data')
    for word, row in payload['entries'].items():
        if (not re.fullmatch(r'(?:[a-z]{2,32}|[a-z]{2,32}-[a-z]{2,32})', word) or not isinstance(row, dict)
                or not isinstance(row.get('source_url'), str) or not row['source_url'].startswith('https://')
                or not isinstance(row.get('source_location'), str) or not row['source_location']
                or not isinstance(row.get('variants'), list) or not row['variants']
                or ('before_vowel_variants' in row and
                    (not isinstance(row['before_vowel_variants'], list) or not row['before_vowel_variants']))
                or any(key in row and type(row[key]) is not bool
                       for key in ('uppercase_only', 'replace_dictionary'))
                or any(not isinstance(variant, list) or not variant
                       or any(not isinstance(phone, str) or phone not in PHONES for phone in variant)
                       for variant in [*row['variants'], *row.get('before_vowel_variants', [])])):
            raise ValueError('Invalid cited ordinary-word reference')
    return payload['entries'], hashlib.sha256(raw).hexdigest()


def entries():
    return _loaded()[0]


def reference_for(word):
    row = entries().get(word.casefold())
    if row and row.get('uppercase_only') and not word.isupper():
        return None
    return row


def policy_contract():
    return {'version': 1, 'reference_sha256': _loaded()[1]}

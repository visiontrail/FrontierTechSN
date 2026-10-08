import hashlib
import json

import pytest

from backend.pipeline import tts_phone_references as references


@pytest.mark.parametrize('defect', ['schema', 'unknown_phone', 'missing_citation', 'invalid_phone_type'])
def test_invalid_cited_reference_data_fails_before_it_can_authorize_a_hint(tmp_path, monkeypatch, defect):
    payload = json.loads(references.WORD_REFERENCES.read_bytes())
    if defect == 'schema':
        payload = []
    elif defect == 'unknown_phone':
        payload['entries']['gui']['variants'] = [['INVENTED']]
    elif defect == 'invalid_phone_type':
        payload['entries']['gui']['variants'] = [[{}]]
    else:
        payload['entries']['gui'].pop('source_url')
    path = tmp_path / 'references.json'
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(references, 'WORD_REFERENCES', path)
    references._loaded_references.cache_clear()
    try:
        with pytest.raises(ValueError, match='reference'):
            references.entries()
    finally:
        references._loaded_references.cache_clear()


def test_reference_contract_hashes_the_same_bytes_as_the_loaded_word_phones(tmp_path, monkeypatch):
    raw = references.WORD_REFERENCES.read_bytes()
    path = tmp_path / 'references.json'
    path.write_bytes(raw)
    monkeypatch.setattr(references, 'WORD_REFERENCES', path)
    references._loaded_references.cache_clear()
    try:
        assert references.entries()['gui']['variants'] == [['G', 'UW', 'IY']]
        assert references.policy_contract()['reference_sha256'] == hashlib.sha256(raw).hexdigest()
        changed = json.loads(raw)
        changed['entries']['gui']['variants'] = [['G', 'IY']]
        path.write_text(json.dumps(changed))
        assert references.entries()['gui']['variants'] == [['G', 'UW', 'IY']]
        assert references.policy_contract()['reference_sha256'] == hashlib.sha256(raw).hexdigest()
    finally:
        references._loaded_references.cache_clear()

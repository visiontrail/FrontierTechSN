import hashlib
import json

import pytest

from backend.pipeline import tts_body_phone_references as references


def test_cited_technical_word_reading_is_bound_to_case_and_same_reference_bytes():
    assert references.reference_for('ADAS')['variants'] == [['EY', 'D', 'AE', 'S'], ['EY', 'D', 'AA', 'S']]
    assert references.reference_for('ADAS')['replace_dictionary'] is True
    assert references.reference_for('Adas') is None
    assert references.reference_for('CUDA')['variants'] == [['K', 'UW', 'D', 'AH']]
    assert references.reference_for('cuda') is None
    assert references.policy_contract()['reference_sha256'] == hashlib.sha256(references.BODY_REFERENCES.read_bytes()).hexdigest()


@pytest.mark.parametrize('defect', ['missing_citation', 'unknown_phone', 'invalid_context_flag'])
def test_invalid_cited_body_reference_cannot_authorize_a_word_cue(tmp_path, monkeypatch, defect):
    payload = json.loads(references.BODY_REFERENCES.read_bytes())
    row = payload['entries']['cuda']
    if defect == 'missing_citation':
        row.pop('source_url')
    elif defect == 'unknown_phone':
        row['variants'] = [['K', 'INVENTED', 'D', 'AH']]
    else:
        row['uppercase_only'] = 'yes'
    path = tmp_path / 'body-references.json'
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(references, 'BODY_REFERENCES', path)
    references._loaded.cache_clear()
    try:
        from backend.pipeline import tts_word_pronunciation as words
        with pytest.raises(ValueError, match='Invalid cited'):
            words.validate_homophone_hint('CUDA', 'coo duh', ['K', 'UW', 'D', 'AH'], version=8)
    finally:
        references._loaded.cache_clear()

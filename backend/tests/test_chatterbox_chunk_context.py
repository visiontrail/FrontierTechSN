"""Short-clause recovery preserves narration, accepted audio and owned jobs."""

import json

import pytest

from backend.pipeline import chatterbox as c


SOURCE = ('In plain terms, each memory cell becomes smaller while the transistor '
          'stands upright and leaves more room for storage. Good morning.')


def layouts():
    args = dict(max_words=12, max_chars=250)
    return (c._split_text(SOURCE, **args, legacy_short_clauses=True),
            c._split_text(SOURCE, **args))


def save_requests(directory, rejected=True, count=3):
    for i in range(count):
        value = {'identity': 'old', 'idempotency_key': f'key-{i}', 'job_id': f'job-{i}'}
        if rejected:
            value['rejected'] = 'Chatterbox unrequested trailing speech: ["extra"]'
        (directory / f'tts_input_part_001.request-{i}.json').write_text(json.dumps(value))


def test_intro_stays_with_context_and_complete_short_sentences_survive():
    old, new = layouts()
    assert old[0] == 'In plain terms,'
    assert new[0].startswith('In plain terms, each memory cell')
    assert ' '.join(new) == SOURCE
    assert all(len(part.split()) <= 12 and len(part) <= 250 for part in new)
    assert c._split_text('Good morning. ' + SOURCE, max_words=12, max_chars=250)[0] == 'Good morning.'


@pytest.mark.parametrize('kind', ['accepted', 'pending', 'partial', 'corrupt'])
def test_rechunk_cannot_displace_owned_or_accepted_work(tmp_path, kind):
    old, new = layouts()
    if kind == 'accepted':
        (tmp_path / 'tts_input_part_001_generated.wav').write_bytes(b'accepted')
    else:
        save_requests(tmp_path, rejected=kind != 'pending', count=1 if kind == 'partial' else 3)
    if kind == 'corrupt':
        (tmp_path / 'tts_input_part_001.request-0.json').write_text('{')
        with pytest.raises(RuntimeError, match='unreadable'):
            c._context_chunk_layout(tmp_path, old, new)
    else:
        chosen, names = c._context_chunk_layout(tmp_path, old, new)
        assert chosen == old
        assert names[0] == 'tts_input_part_001'
    assert not (tmp_path / 'tts_context_chunk_layout.json').exists()


def test_exhausted_tail_jobs_retained_with_stable_bounded_repair_namespace(tmp_path):
    old, new = layouts()
    save_requests(tmp_path)
    original = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    chosen, names = c._context_chunk_layout(tmp_path, old, new)
    assert chosen == new and names[0] == 'tts_input_part_001.context-v1'
    # A restart must follow the same corrected request, never allocate a new namespace.
    pending = tmp_path / f'{names[0]}.request-0.json'
    pending.write_text(json.dumps({'identity': 'new', 'idempotency_key': 'key-new', 'job_id': 'job-new'}))
    assert c._context_chunk_layout(tmp_path, old, new) == (chosen, names)
    assert all((tmp_path / name).read_bytes() == data for name, data in original.items())
    assert c.MAX_INTEGRITY_ATTEMPTS == 3
    assert not c._trailing_speech_report('In plain terms,', [{'text': 'In plain terms in plain terms'}])['passed']
    with pytest.raises(RuntimeError, match='layout changed'):
        c._context_chunk_layout(tmp_path, old, new + ['Added narration.'])
    (tmp_path / 'tts_input_part_001.request-0.json').write_text('{}')
    with pytest.raises(RuntimeError, match='original request changed'):
        c._context_chunk_layout(tmp_path, old, new)


def test_repair_retains_unchanged_accepted_prefix(tmp_path):
    old, new = layouts()
    # Shift the repair to the second part while retaining a completed first part.
    save_requests(tmp_path)
    for p in list(tmp_path.iterdir()):
        p.rename(p.with_name(p.name.replace('001', '002')))
    accepted = tmp_path / 'tts_input_part_001_generated.wav'
    accepted.write_bytes(b'original prefix audio')
    chosen, names = c._context_chunk_layout(tmp_path, ['Intro.'] + old, ['Intro.'] + new)
    assert chosen[0] == 'Intro.' and names[0] == 'tts_input_part_001'
    assert names[1] == 'tts_input_part_002.context-v1'
    assert accepted.read_bytes() == b'original prefix audio'

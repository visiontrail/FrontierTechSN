from __future__ import annotations

import copy
import hashlib
import json
import wave

import numpy as np
import pytest

from backend.pipeline import chatterbox_tail


@pytest.fixture
def part(tmp_path):
    source = tmp_path / "tts_input_part_001_generated.wav"
    samples = np.full(48000, 1200, dtype="<i2")
    samples[24000:38400] = 1
    with wave.open(str(source), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(24000)
        writer.writeframes(samples.tobytes())
    integrity = {
        "accepted": True, "expected_words": 10, "transcript_words": 10,
        "matched_exact_words": 10, "exact_asr_word_coverage": 1.0,
        "transcript_word_ratio": 1.0, "leading_anchor": True, "trailing_anchor": True,
        "exact_leading_anchor": True, "exact_trailing_anchor": True,
        "repeat_start_seconds": None, "token_differences": [],
        "speech_end_seconds": 1.0, "transcription": {"passed": True},
    }
    row = {"audio_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
           "integrity": integrity, "identity": {"text": "The original exact source text."},
           "job_id": "original-provider-job", "wav": {"frame_count": 48000}}
    source.with_suffix(".json").write_text(json.dumps(row))
    return source, row


def test_derivative_preserves_original_artifacts_and_separates_provenance(part):
    source, metadata = part
    original_bytes, original_metadata_bytes = source.read_bytes(), source.with_suffix(".json").read_bytes()
    before = copy.deepcopy(metadata)
    delivered, rows = chatterbox_tail.prepare_delivery_parts([source], [metadata])
    output, row = delivered[0], rows[0]
    assert output != source
    assert source.read_bytes() == original_bytes
    assert source.with_suffix(".json").read_bytes() == original_metadata_bytes
    assert metadata == before
    assert row["provider_audio_sha256"] == metadata["audio_sha256"]
    assert row["provider_audio_path"] == str(source.resolve())
    assert row["audio_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert row["audio_sha256"] != row["provider_audio_sha256"]
    assert row["job_id"] == metadata["job_id"]
    assert row["integrity"]["transcription"] == metadata["integrity"]["transcription"]
    assert json.loads(output.with_suffix(".json").read_text()) == row
    report = row["integrity"]["tail_cleanup"]
    assert report["proof"]["source_unchanged"] is True
    assert report["original_frame_count"] == report["output_frame_count"] == metadata["wav"]["frame_count"]
    assert json.loads((source.parent / "tts_tail_cleanup_report.json").read_text())["applied_count"] == 1


def test_cache_reuse_requires_matching_source_output_and_provenance(part, monkeypatch):
    source, metadata = part
    delivered, rows = chatterbox_tail.prepare_delivery_parts([source], [metadata])
    def unexpected(*args, **kwargs):
        raise AssertionError("A verified derivative should not be recreated")
    monkeypatch.setattr(chatterbox_tail, "clean_verified_tail", unexpected)
    again, new_rows = chatterbox_tail.prepare_delivery_parts([source], [metadata])
    assert again == delivered
    assert new_rows == rows


@pytest.mark.parametrize("changed", ["source", "derived", "report"])
def test_cache_tampering_fails_closed_without_overwriting_any_waveform(part, changed):
    source, metadata = part
    delivered, _ = chatterbox_tail.prepare_delivery_parts([source], [metadata])
    output = delivered[0]
    if changed == "source":
        source.write_bytes(source.read_bytes() + b"changed")
    elif changed == "derived":
        output.write_bytes(output.read_bytes() + b"changed")
    else:
        path = output.with_suffix(".cleanup.json")
        data = json.loads(path.read_text())
        data["original_audio_sha256"] = "0" * 64
        path.write_text(json.dumps(data))
    expected_source, expected_output = source.read_bytes(), output.read_bytes()
    with pytest.raises(RuntimeError, match="cache changed"):
        chatterbox_tail.prepare_delivery_parts([source], [metadata])
    assert source.read_bytes() == expected_source
    assert output.read_bytes() == expected_output


def test_orphan_output_cannot_be_treated_as_a_verified_cache(part):
    source, metadata = part
    delivered, _ = chatterbox_tail.prepare_delivery_parts([source], [metadata])
    output = delivered[0]
    output.with_suffix(".cleanup.json").unlink()
    before = output.read_bytes()
    with pytest.raises(RuntimeError, match="no committed provenance"):
        chatterbox_tail.prepare_delivery_parts([source], [metadata])
    assert output.read_bytes() == before


def test_orphan_report_is_replaced_only_by_a_fresh_verified_derivative(part):
    source, metadata = part
    delivered, _ = chatterbox_tail.prepare_delivery_parts([source], [metadata])
    output = delivered[0]
    output.unlink()
    report_path = output.with_suffix(".cleanup.json")
    report_path.write_text('{"applied":true,"output_audio_sha256":"stale"}')
    recreated, rows = chatterbox_tail.prepare_delivery_parts([source], [metadata])
    assert recreated == delivered
    assert json.loads(report_path.read_text())["output_audio_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert rows[0]["integrity"]["tail_cleanup"]["original_audio_sha256"] == metadata["audio_sha256"]


def test_changed_word_timing_gets_a_distinct_derivative_without_touching_prior_cut(part):
    source, metadata = part
    initial, _ = chatterbox_tail.prepare_delivery_parts([source], [metadata])
    old_bytes = initial[0].read_bytes()
    revised = copy.deepcopy(metadata)
    revised["integrity"]["speech_end_seconds"] = 1.2
    later, rows = chatterbox_tail.prepare_delivery_parts([source], [revised])
    assert later[0] != initial[0]
    assert initial[0].read_bytes() == old_bytes
    assert rows[0]["integrity"]["tail_cleanup"]["cleanup_start_seconds"] >= 1.32


def test_insufficient_evidence_preserves_original_delivery(part):
    source, metadata = part
    metadata["integrity"]["accepted"] = False
    delivered, rows = chatterbox_tail.prepare_delivery_parts([source], [metadata])
    assert delivered == [source]
    assert rows == [metadata]
    assert not list(source.parent.glob("*_tail_*.wav"))
    summary = json.loads((source.parent / "tts_tail_cleanup_report.json").read_text())
    assert summary["applied_count"] == 0


def test_each_part_requires_its_own_metadata(part):
    source, _ = part
    with pytest.raises(ValueError, match="metadata for every source part"):
        chatterbox_tail.prepare_delivery_parts([source], [])


def test_long_post_speech_gap_is_cleaned_before_continuity_acceptance(part):
    from backend.pipeline import tts

    source, metadata = part
    with wave.open(str(source), "rb") as reader:
        samples = np.frombuffer(reader.readframes(reader.getnframes()), dtype="<i2").copy()
    samples[24000:45600] = 1  # 0.9 s silence, then a non-speech tail.
    with wave.open(str(source), "wb") as writer:
        writer.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
        writer.writeframes(samples.tobytes())
    original = source.read_bytes()
    metadata['audio_sha256'] = hashlib.sha256(original).hexdigest()
    words = [{'text': 'ending.', 'start': .8, 'end': 1.0}]
    with pytest.raises(tts.TtsIntegrityError, match='continuity ceiling'):
        tts._validate_pocket_internal_silence(source, words)
    recovery = chatterbox_tail.verify_tail_continuity(
        source, metadata['integrity'], words, source.parent,
        expected_audio_sha256=metadata['audio_sha256'],
    )
    assert recovery['provider_internal_silence']['passed'] is False
    assert recovery['delivery_internal_silence']['passed'] is True
    metadata['integrity']['internal_silence'] = recovery['provider_internal_silence']
    metadata['integrity']['continuity_tail_recovery'] = recovery
    delivered, rows = chatterbox_tail.prepare_delivery_parts([source], [metadata])
    assert source.read_bytes() == original
    assert rows[0]['audio_sha256'] == recovery['cleanup']['output_audio_sha256']
    assert rows[0]['integrity']['internal_silence']['passed'] is True
    assert tts._validate_pocket_internal_silence(delivered[0], words)['passed'] is True


@pytest.mark.parametrize('defect', ['internal_pause', 'weak_transcript', 'extra_speech', 'wrong_hash'])
def test_tail_continuity_cannot_hide_real_gaps_or_missing_proof(part, defect):
    from backend.pipeline import tts

    source, metadata = part
    with wave.open(str(source), 'wb') as writer:
        writer.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
        writer.writeframes(np.concatenate((np.full(24000, 1200), np.zeros(24000),
                                          np.full(12000, 1200))).astype('<i2').tobytes())
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    evidence = metadata['integrity']
    if defect == 'internal_pause':
        evidence['speech_end_seconds'] = 2.4
    elif defect == 'weak_transcript':
        evidence['exact_asr_word_coverage'] = .7
    elif defect == 'extra_speech':
        evidence['pronunciation'] = {'passed': True, 'trailing_speech': {'passed': False}}
    else:
        digest = '0' * 64
    original = source.read_bytes()
    with pytest.raises(tts.TtsIntegrityError, match='continuity ceiling'):
        chatterbox_tail.verify_tail_continuity(source, evidence, [], source.parent,
                                              expected_audio_sha256=digest)
    assert source.read_bytes() == original
    assert not list(source.parent.glob('continuity-tail-*.wav'))


def test_weak_evidence_cannot_hide_a_changed_provider_waveform(part):
    source, metadata = part
    metadata["integrity"]["accepted"] = False
    source.write_bytes(source.read_bytes() + b"changed")
    with pytest.raises(RuntimeError, match="provider waveform"):
        chatterbox_tail.prepare_delivery_parts([source], [metadata])


def test_source_changed_during_noop_cleanup_is_not_joined(part, monkeypatch):
    source, metadata = part
    def changed(*args, **kwargs):
        source.write_bytes(source.read_bytes() + b"changed")
        return {"applied": False, "reason": "source_changed_during_cleanup"}
    monkeypatch.setattr(chatterbox_tail, "clean_verified_tail", changed)
    with pytest.raises(RuntimeError, match="changed during"):
        chatterbox_tail.prepare_delivery_parts([source], [metadata])

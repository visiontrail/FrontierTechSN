from __future__ import annotations

import copy
import hashlib
import wave

import numpy as np
import pytest

from backend.pipeline.tts_tail_cleanup import clean_verified_tail
from backend.pipeline import tts_tail_cleanup


RATE = 24000


def _evidence():
    return {
        "accepted": True,
        "expected_words": 24,
        "transcript_words": 24,
        "matched_exact_words": 24,
        "exact_asr_word_coverage": 1.0,
        "transcript_word_ratio": 1.0,
        "leading_anchor": True,
        "trailing_anchor": True,
        "exact_leading_anchor": True,
        "exact_trailing_anchor": True,
        "repeat_start_seconds": None,
        "token_differences": [],
        "speech_end_seconds": 1.0,
        "transcription": {"passed": True},
    }


def _write(path, samples, *, width=2):
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(samples.shape[1] if samples.ndim == 2 else 1)
        writer.setsampwidth(width)
        writer.setframerate(RATE)
        writer.writeframes(samples.astype("<i2").tobytes())
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _samples():
    # Speech ends at 1 s, quiet pause follows, then a provider-generated tail.
    t = np.arange(RATE * 2) / RATE
    samples = np.rint(12000 * np.sin(2 * np.pi * 220 * t)).astype("<i2")
    samples[RATE:int(1.6 * RATE)] = 10
    return samples


def _run(tmp_path, *, evidence=None, samples=None, digest=None):
    source, output = tmp_path / "original.wav", tmp_path / "clean.wav"
    digest = digest or _write(source, _samples() if samples is None else samples)
    report = clean_verified_tail(source, output, evidence or _evidence(), expected_audio_sha256=digest)
    return source, output, report


def test_removes_only_evidenced_tail_without_changing_source_or_timeline(tmp_path):
    samples = _samples()
    source, output, report = _run(tmp_path, samples=samples)
    with wave.open(str(output), "rb") as reader:
        cleaned = np.frombuffer(reader.readframes(reader.getnframes()), dtype="<i2")
        assert reader.getframerate() == RATE
    assert report["applied"] is True
    assert report["proof"]["source_unchanged"] is True
    assert hashlib.sha256(source.read_bytes()).hexdigest() == report["original_audio_sha256"]
    assert hashlib.sha256(output.read_bytes()).hexdigest() == report["output_audio_sha256"]
    assert report["original_audio_sha256"] != report["output_audio_sha256"]
    assert len(samples) == len(cleaned) == report["original_frame_count"] == report["output_frame_count"]
    start, end = report["fade_start_frame"], report["silence_start_frame"]
    np.testing.assert_array_equal(samples[:start], cleaned[:start])
    assert report["proof"]["preserved_prefix_sha256"] == hashlib.sha256(samples[:start].tobytes()).hexdigest()
    assert report["cleanup_start_seconds"] >= 1.12
    assert report["silence_start_seconds"] <= report["proof"]["quiet_end_seconds"]
    assert cleaned[start] == samples[start]
    assert np.all(np.diff(cleaned[start:end].astype(int)) <= 0)
    assert np.all(cleaned[end:] == 0)
    with wave.open(str(source), "rb") as reader:
        np.testing.assert_array_equal(np.frombuffer(reader.readframes(reader.getnframes()), dtype="<i2"), samples)


def test_allows_single_interior_spelling_difference_with_strong_exact_anchors(tmp_path):
    evidence = _evidence()
    evidence.update(verified=False, matched_exact_words=23, exact_asr_word_coverage=0.9583,
                    token_differences=[{"kind": "replace", "source": ["bytefront"], "asr": ["bitefront"]}])
    assert _run(tmp_path, evidence=evidence)[2]["applied"] is True


@pytest.mark.parametrize("changes", [
    {"accepted": False},
    {"transcription": {"passed": False}},
    {"expected_words": True},
    {"transcript_words": 25, "transcript_word_ratio": 25 / 24},
    {"transcript_words": 23},
    {"exact_asr_word_coverage": 0.91, "matched_exact_words": 22},
    {"exact_asr_word_coverage": float("nan")},
    {"matched_exact_words": 23},
    {"leading_anchor": False},
    {"exact_trailing_anchor": False},
    {"repeat_start_seconds": 0.0},
    {"token_differences": [{"kind": "insert", "source": [], "asr": ["again"]}]},
    {"token_differences": [{"kind": "delete", "source": ["word"], "asr": []}]},
    {"token_differences": [{"kind": "replace", "source": ["word"], "asr": ["two", "words"]}]},
    {"token_differences": None},
    {"pronunciation": {"passed": True, "trailing_speech": {"passed": False}}},
    {"speech_end_seconds": None},
    {"speech_end_seconds": float("inf")},
    {"speech_end_seconds": 0},
    {"speech_end_seconds": 2.1},
])
def test_incomplete_or_conflicting_evidence_does_not_create_audio(tmp_path, changes):
    evidence = {**_evidence(), **changes}
    source, output, report = _run(tmp_path, evidence=evidence)
    assert report["applied"] is False
    assert not output.exists()
    with wave.open(str(source), "rb") as reader:
        np.testing.assert_array_equal(np.frombuffer(reader.readframes(reader.getnframes()), dtype="<i2"), _samples())


def test_missing_repetition_check_is_not_treated_as_confirmed_absence(tmp_path):
    evidence = _evidence()
    del evidence["repeat_start_seconds"]
    assert _run(tmp_path, evidence=evidence)[2]["reason"] == "repetition_or_missing_repetition_check"


def test_report_for_different_audio_cannot_authorize_cleanup(tmp_path):
    source, output = tmp_path / "source.wav", tmp_path / "output.wav"
    _write(source, _samples())
    report = clean_verified_tail(source, output, _evidence(), expected_audio_sha256="0" * 64)
    assert report["reason"] == "observation_audio_hash_mismatch"
    assert not output.exists()


def test_never_cuts_through_continuous_audio_or_a_short_pause(tmp_path):
    samples = _samples()
    samples[RATE:int(1.6 * RATE)] = 10000
    samples[int(1.3 * RATE):int(1.35 * RATE)] = 0
    _, output, report = _run(tmp_path, samples=samples)
    assert report["reason"] == "no_verified_quiet_interval"
    assert not output.exists()


def test_quiet_gap_before_final_word_cannot_authorize_cutting_that_word(tmp_path):
    samples = _samples()
    evidence = {**_evidence(), "speech_end_seconds": 1.9}
    _, output, report = _run(tmp_path, samples=samples, evidence=evidence)
    assert report["applied"] is False
    assert not output.exists()


def test_naturally_quiet_tail_and_internal_noise_are_left_untouched(tmp_path):
    samples = _samples()
    samples[RATE:] = 0
    samples[RATE // 2:RATE // 2 + 20] = 32767
    _, output, report = _run(tmp_path, samples=samples)
    assert report["reason"] == "tail_already_quiet"
    assert not output.exists()


def test_stereo_requires_both_channels_to_have_a_quiet_interval(tmp_path):
    samples = np.column_stack((_samples(), np.full(RATE * 2, 10000, dtype="<i2")))
    _, output, report = _run(tmp_path, samples=samples)
    assert report["reason"] == "no_verified_quiet_interval"
    assert not output.exists()


def test_transient_peaks_inside_quiet_rms_windows_prevent_a_cut(tmp_path):
    samples = _samples()
    samples[RATE:] = 0
    # Isolated 2,000-count impulses every 10 ms are RMS-quiet but not peak-quiet.
    samples[RATE::240] = 2000
    _, output, report = _run(tmp_path, samples=samples)
    assert report["reason"] == "no_verified_quiet_interval"
    assert not output.exists()


def test_repeated_cleanup_of_cleaned_waveform_is_a_noop(tmp_path):
    _, cleaned, report = _run(tmp_path)
    second = tmp_path / "second.wav"
    repeated = clean_verified_tail(cleaned, second, _evidence(),
                                   expected_audio_sha256=report["output_audio_sha256"])
    assert repeated["reason"] == "tail_already_quiet"
    assert not second.exists()


def test_source_and_existing_outputs_cannot_be_overwritten(tmp_path):
    source, output, _ = _run(tmp_path)
    original, derived = source.read_bytes(), output.read_bytes()
    digest = hashlib.sha256(original).hexdigest()
    with pytest.raises(ValueError, match="distinct output"):
        clean_verified_tail(source, source, _evidence(), expected_audio_sha256=digest)
    with pytest.raises(FileExistsError):
        clean_verified_tail(source, output, _evidence(), expected_audio_sha256=digest)
    assert source.read_bytes() == original
    assert output.read_bytes() == derived
    assert not list(tmp_path.glob(".clean-*.wav"))


def test_invalid_pcm_payload_is_a_noop(tmp_path):
    source, output = tmp_path / "truncated.wav", tmp_path / "output.wav"
    _write(source, _samples())
    source.write_bytes(source.read_bytes()[:-100])
    report = clean_verified_tail(source, output, _evidence(),
                                 expected_audio_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    assert report["reason"] == "incomplete_wav_payload"
    assert not output.exists()


def test_does_not_mutate_integrity_report(tmp_path):
    evidence = _evidence()
    original = copy.deepcopy(evidence)
    _run(tmp_path, evidence=evidence)
    assert evidence == original


def test_concurrent_source_change_cannot_leave_an_applied_derivative(tmp_path, monkeypatch):
    source, output = tmp_path / "source.wav", tmp_path / "output.wav"
    digest = _write(source, _samples())
    changed = source.read_bytes() + b"concurrent-owner-change"
    original_link = tts_tail_cleanup.os.link
    def link_then_concurrent_change(staged, destination):
        original_link(staged, destination)
        source.write_bytes(changed)
    monkeypatch.setattr(tts_tail_cleanup.os, "link", link_then_concurrent_change)
    report = clean_verified_tail(source, output, _evidence(), expected_audio_sha256=digest)
    assert report["applied"] is False
    assert report["reason"] == "source_changed_during_cleanup"
    assert not output.exists()
    assert source.read_bytes() == changed
    assert not list(tmp_path.glob(".output-*.wav"))

"""The real signals/shaping failure must be resynthesized, never waved through."""

import json

import pytest

from backend.pipeline import tts


SOURCE = "A concentrated shot of the frontier-tech signals shaping what comes next."
OBSERVED = SOURCE.replace("signals", "signal")


def evidence(directory, source=SOURCE, normal=OBSERVED, slower=OBSERVED, status="rejected"):
    (directory / "llm_asr_adjudication.json").write_text(json.dumps({
        "status": status,
        "request": {"source_text": source, "normal_speed_transcript": normal,
                    "slower_speed_transcript": slower},
    }))


def test_failed_plural_gets_provider_pause_without_changing_source_tokens(tmp_path):
    evidence(tmp_path)
    repaired = tts._pocket_pronunciation_retry_text(SOURCE, SOURCE, tmp_path)
    assert repaired == SOURCE.replace("signals shaping", "signals, shaping")
    assert tts._lexical_tokens(repaired) == tts._lexical_tokens(SOURCE)
    words = [{"text": text, "start": i * .3, "end": i * .3 + .2}
             for i, text in enumerate(OBSERVED.split())]
    assert not tts._orpheus_transcript_report(SOURCE, words)["verified"]


@pytest.mark.parametrize("defect", [
    "no_evidence", "malformed", "wrong_source", "approved", "disagreeing_decodes",
    "missing_word", "two_differences", "changed_number", "not_followed_by_s",
])
def test_pronunciation_hint_requires_the_specific_corroborated_failure(tmp_path, defect):
    source = SOURCE
    normal = slower = OBSERVED
    status = "rejected"
    recorded_source = source
    if defect == "wrong_source":
        recorded_source = "An older paragraph."
    elif defect == "approved":
        status = "approved"
    elif defect == "disagreeing_decodes":
        slower = source
    elif defect == "missing_word":
        normal = slower = OBSERVED.replace("concentrated ", "")
    elif defect == "two_differences":
        normal = slower = OBSERVED.replace("next", "later")
    elif defect == "changed_number":
        source = recorded_source = "There are 15 signals shaping the next stage."
        normal = slower = source.replace("15", "50")
    elif defect == "not_followed_by_s":
        source = recorded_source = SOURCE.replace("shaping", "guiding")
        normal = slower = source.replace("signals", "signal")
    evidence(tmp_path, recorded_source, normal, slower, status)
    path = tmp_path / "llm_asr_adjudication.json"
    if defect == "no_evidence":
        path.unlink()
    elif defect == "malformed":
        path.write_text("[]")
    synthesis = tts._pocket_synthesis_text(source)
    assert tts._pocket_pronunciation_retry_text(synthesis, source, tmp_path) == synthesis


def test_retry_hint_preserves_existing_number_pronunciation(tmp_path):
    source = "The 44 signals shaping this report are clear."
    observed = source.replace("signals", "signal")
    evidence(tmp_path, source, observed, observed)
    synthesis = tts._pocket_synthesis_text(source)
    repaired = tts._pocket_pronunciation_retry_text(synthesis, source, tmp_path)
    assert "forty-four signals, shaping" in repaired
    assert tts._lexical_tokens(repaired) == tts._lexical_tokens(source)

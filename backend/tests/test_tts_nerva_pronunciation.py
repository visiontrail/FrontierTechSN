from backend.pipeline.tts import _orpheus_prompt_text, _orpheus_transcript_report


def _words(*tokens: str) -> list[dict]:
    return [
        {"text": token, "start": index * 0.25, "end": (index + 1) * 0.25}
        for index, token in enumerate(tokens)
    ]


def test_orpheus_prompt_exposes_nerva_pronunciation_only_before_rover():
    assert _orpheus_prompt_text("programs NERVA and Rover already") == (
        "programs Ner-vuh and Rover already."
    )
    assert _orpheus_prompt_text("the NERVA program") == "the NERVA program."


def test_orpheus_transcript_accepts_non_rhotic_rover_in_complete_name_pair():
    report = _orpheus_transcript_report(
        "programs NERVA and Rover already",
        _words("programs", "Nerva", "and", "ROVA", "already"),
    )

    assert report["verified"] is True
    assert report["exact_asr_word_coverage"] == 1.0


def test_orpheus_transcript_accepts_corroborated_nerv_spelling_in_name_pair():
    report = _orpheus_transcript_report(
        "programs NERVA and Rover already",
        _words("programs", "NERV", "and", "Rover", "already"),
    )

    assert report["verified"] is True
    assert report["exact_asr_word_coverage"] == 1.0


def test_orpheus_transcript_rejects_wrong_nerva_even_when_rover_is_non_rhotic():
    report = _orpheus_transcript_report(
        "programs NERVA and Rover already",
        _words("programs", "NAL", "and", "ROVA", "already"),
    )

    assert report["verified"] is False


def test_orpheus_transcript_does_not_globalize_rova_equivalence():
    report = _orpheus_transcript_report(
        "the Rover program continued",
        _words("the", "ROVA", "program", "continued"),
    )

    assert report["verified"] is False


def test_orpheus_transcript_does_not_globalize_nerv_equivalence():
    report = _orpheus_transcript_report(
        "the NERVA program continued",
        _words("the", "NERV", "program", "continued"),
    )

    assert report["verified"] is False

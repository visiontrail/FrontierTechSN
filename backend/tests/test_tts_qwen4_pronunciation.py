from backend.pipeline.tts import _orpheus_prompt_text, _orpheus_transcript_report


def _words(*tokens: str) -> list[dict]:
    return [
        {"text": token, "start": index * 0.25, "end": (index + 1) * 0.25}
        for index, token in enumerate(tokens)
    ]


def test_orpheus_prompt_binds_qwen_four_pronunciation():
    assert _orpheus_prompt_text("the Qwen 4 architecture") == (
        "the cue-when four architecture."
    )


def test_orpheus_transcript_accepts_homophones_only_as_complete_qwen4_name():
    report = _orpheus_transcript_report(
        "the Qwen 4 architecture",
        _words("the", "Q", "when", "for", "architecture"),
    )

    assert report["verified"] is True
    assert report["exact_asr_word_coverage"] == 1.0


def test_orpheus_transcript_rejects_qwen4_mispronunciations():
    for observed in (
        ("the", "Q", "Lune", "4", "architecture"),
        ("the", "queue", "went", "for", "architecture"),
    ):
        report = _orpheus_transcript_report(
            "the Qwen 4 architecture",
            _words(*observed),
        )
        assert report["verified"] is False


def test_orpheus_transcript_does_not_globalize_for_as_four():
    report = _orpheus_transcript_report(
        "version 4 is ready",
        _words("version", "for", "is", "ready"),
    )

    assert report["verified"] is False

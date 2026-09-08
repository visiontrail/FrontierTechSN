"""Regression evidence from the September 8 Pocket TTS narration failure."""

import pytest

from backend.pipeline import tts


SOURCE = (
    "Bloomberg reports that Shenzhen Longsys Electronics is set to begin trading "
    "in Hong Kong on Tuesday after raising 7.08 billion Hong Kong dollars, about "
    "903 million U.S. dollars, in an upsized share sale. Bloomberg frames the "
    "debut as a test of investor enthusiasm, coming after a flurry of "
    "artificial-intelligence supply-chain offerings in the city."
)
TRANSCRIPT = (
    "Bloomberg reports that Shenzhen Longsis Electronics is set to begin trading "
    "in Hong Kong on Tuesday after raising HK $7 .08 billion, about US $903 "
    "million, in an upsized share sale. Bloomberg frames the debut as a test "
    "of investor enthusiasm, coming after a flurry of artificial intelligence "
    "supply chain offerings in the city."
)


def words(text):
    return [
        {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
        for index, word in enumerate(text.split())
    ]


def test_live_currency_formatting_leaves_only_the_name_for_adjudication():
    report = tts._orpheus_transcript_report(SOURCE, words(TRANSCRIPT))
    assert report["expected_words"] == report["transcript_words"] == 56
    assert report["matched_exact_words"] == 55
    assert report["leading_anchor"] and report["trailing_anchor"]
    # The spelling drift still requires the existing acoustic/adjudication gate.
    assert not report["verified"]
    assert tts._orpheus_transcript_report(
        SOURCE, words(TRANSCRIPT.replace("Longsis", "Longsys"))
    )["verified"]


@pytest.mark.parametrize("amount", ["HK $7 .08", "HK $7.08", "HK$7.08", "HK$7 .08"])
def test_qualified_currency_split_forms_preserve_amount_and_onset(amount):
    transcript = words(f"Raised {amount} billion, about US $903 million.")
    expected = "Raised 7.08 billion Hong Kong dollars, about 903 million U.S. dollars."
    assert tts._orpheus_transcript_report(expected, transcript)["verified"]
    tokens, indexes = tts._transcript_tokens(transcript)
    assert len(tokens) == len(indexes)
    assert tokens[1] == "decimalnumber7point08"
    assert indexes[1] == 1
    assert all(0 <= index < len(transcript) for index in indexes)


@pytest.mark.parametrize("replacement", [
    "US $7 .08 billion", "HK $7 .09 billion", "HK $7 .08 million",
    "$7 .08 billion", "HK 7 .08 billion", "HK $7 .08 billion dollars",
])
def test_currency_normalization_rejects_changed_or_missing_content(replacement):
    observed = TRANSCRIPT.replace("Longsis", "Longsys").replace(
        "HK $7 .08 billion", replacement
    )
    assert not tts._orpheus_transcript_report(SOURCE, words(observed))["verified"]


def test_real_middle_omission_does_not_report_a_missing_ending():
    report = tts._orpheus_transcript_report(
        "The team built a complete working system today.",
        words("The team built a working system today."),
    )
    assert not report["verified"]
    assert report["trailing_anchor"]
    assert "closing words have no acoustic transcript anchor" not in report["failure_reasons"]
    assert not tts._orpheus_transcript_report(
        SOURCE, words(TRANSCRIPT.replace("in the city.", ""))
    )["trailing_anchor"]

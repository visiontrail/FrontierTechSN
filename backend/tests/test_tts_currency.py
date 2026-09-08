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


def test_pocket_request_spells_short_integers_without_changing_source_tokens():
    source = (
        "QbitAI reports that roughly 44 billion yuan went into China's "
        "embodied-intelligence sector in the first half of 2026. "
        "The D1 shipped 8.5 million units with 61.1 percent share, "
        "50 markets, 1 robot and 99 modules."
    )
    spoken = tts._pocket_synthesis_text(source)
    assert "forty-four billion yuan" in spoken
    assert "fifty markets, one robot and ninety-nine modules" in spoken
    assert "D1" in spoken and "2026" in spoken and "61.1" in spoken and "8.5" in spoken
    assert tts._lexical_tokens(spoken) == tts._lexical_tokens(source)


@pytest.mark.parametrize("source", [
    "D1 Qwen3 R2-D2 2026 44.5 8.5 61.1 7.08 1,000 $44 01",
    "It costs 1 dollar, 44 yuan, or 99 cents.",
    "From 20 to 50 percent in September 8, 2026.",
])
def test_pocket_number_pronunciation_preserves_numeric_identity(source):
    spoken = tts._pocket_synthesis_text(source)
    assert tts._lexical_tokens(spoken) == tts._lexical_tokens(source)
    if source.startswith("D1"):
        assert spoken == source

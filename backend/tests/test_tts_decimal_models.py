"""Regression evidence from task 20260916-061006-fd5fe9's DM0.5 sentence."""

import pytest

from backend.pipeline import tts


SOURCE = (
    "From robotics: the Chinese-language outlet Machine Heart reports that the "
    "embodied-AI model DM0.5, built by Yuanli Lingji, has swept all four "
    "leaderboards of RoboColiseum, an evaluation suite launched by the Chinese "
    "robotics firm Zhiyuan Robotics covering instruction following, spatial "
    "understanding, disturbance adaptation and general manipulation."
)


def words(text):
    return [{"text": token, "start": i * .3, "end": i * .3 + .2}
            for i, token in enumerate(text.split())]


@pytest.mark.parametrize("observed", ["DM0.5", "DM0 .5", "DM 0 .5", "DM .5", "DM 0.5"])
def test_decimal_model_formatting_preserves_every_letter_digit_and_word_index(observed):
    transcript = words(SOURCE.replace("DM0.5", observed))
    report = tts._orpheus_transcript_report(SOURCE, transcript)
    assert report["verified"]
    assert report["exact_asr_word_coverage"] == 1.0
    tokens, indexes = tts._transcript_tokens(transcript)
    assert tokens == tts._lexical_tokens(SOURCE)
    decimal_index = tokens.index("decimalnumber0point5")
    assert transcript[indexes[decimal_index]]["text"] in {"DM0.5,", "DM0", "0", ".5,", "0.5,"}
    assert transcript[indexes[tokens.index("built")]]["text"] == "built"


@pytest.mark.parametrize("observed", [
    "DM5", "DM 5", "DM0 5", "DM0 .6", "DM1 .5", "DM .05", "DN0 .5", "0 .5",
    "DM0 .5 .5", "DM0 extra .5",
])
def test_decimal_model_formatting_does_not_hide_changed_or_missing_content(observed):
    transcript = SOURCE.replace("DM0.5", observed)
    assert not tts._orpheus_transcript_report(SOURCE, words(transcript))["verified"]
    assert not tts._medium_asr_verdict_is_corroborated(
        tts._lexical_tokens(SOURCE), transcript, transcript,
    )


def test_live_name_drift_still_needs_adjudication_after_decimal_normalization():
    normal = SOURCE.replace("DM0.5", "DM .5").replace("Zhiyuan", "Jiuan")
    slower = SOURCE.replace("DM0.5", "DM0 .5").replace("Zhiyuan", "Jiwan")
    report = tts._orpheus_transcript_report(SOURCE, words(normal))
    assert not report["verified"]
    assert report["expected_words"] == report["transcript_words"] == 48
    assert report["matched_exact_words"] == 47
    assert report["token_differences"] == [
        {"kind": "replace", "source": ["zhiyuan"], "asr": ["jiuan"]},
    ]
    assert tts._medium_asr_verdict_is_corroborated(
        tts._lexical_tokens(SOURCE), normal, slower, words(normal), words(slower),
    )
    # A genuine omission must still block even these close name transcripts.
    assert not tts._medium_asr_verdict_is_corroborated(
        tts._lexical_tokens(SOURCE), normal.replace("spatial ", ""), slower.replace("spatial ", ""),
        words(normal.replace("spatial ", "")), words(slower.replace("spatial ", "")),
    )


@pytest.mark.parametrize(("source", "observed"), [
    ("The measured value is 0.05 today.", "The measured value is .05 today."),
    ("The robot AX12.75 is ready.", "The robot AX12 .75 is ready."),
])
def test_decimal_formatting_is_not_a_model_specific_alias(source, observed):
    assert tts._orpheus_transcript_report(source, words(observed))["verified"]
    if ".05" in observed:
        assert tts._lexical_tokens(source) == tts._lexical_tokens(observed)

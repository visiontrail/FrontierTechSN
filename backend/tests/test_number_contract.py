"""Generated invariants for the shared integer grammar, beyond incident cases."""

import random

import pytest

from backend.pipeline import tts
from backend.spoken_numbers import (
    MAX_SPOKEN_INTEGER,
    canonicalize_integer_tokens,
    integer_words,
    normalize_grouped_integer_literals,
    normalize_spoken_quantities,
)


def quantities():
    # Exhaust every sub-thousand coefficient at each supported magnitude,
    # then exercise mixed groups, missing groups and magnitude boundaries.
    values = {coefficient * scale for coefficient in range(1000)
              for scale in (1, 1000, 1_000_000, 1_000_000_000)}
    for power in range(1, 12):
        values.update((10**power - 1, 10**power, 10**power + 1))
    values.add(MAX_SPOKEN_INTEGER)
    rng = random.Random(20260922)
    values.update(rng.randrange(MAX_SPOKEN_INTEGER + 1) for _ in range(1000))
    return sorted(values)


def test_integer_round_trip_preserves_value_units_and_asr_indexes_across_magnitudes():
    for value in quantities():
        spoken = integer_words(value)
        expected = ["worth", str(value), "dollars", "today"]
        for amount in (str(value), f"{value:,}", spoken):
            text = f"Worth {amount} dollars today."
            assert tts._lexical_tokens(text) == expected, (value, text)
            assert tts._lexical_tokens(tts._pocket_synthesis_text(text)) == expected
            transcript = [{"text": word} for word in text.split()]
            tokens, indexes = tts._transcript_tokens(transcript)
            assert tokens == expected, (value, text, tokens)
            assert indexes == [0, 1, len(transcript) - 2, len(transcript) - 1]
        assert normalize_spoken_quantities(spoken) == spoken
        assert tts._lexical_tokens(f"Worth {value + 1} dollars today.") != expected
        assert tts._lexical_tokens(f"Worth {value} yuan today.") != expected
        assert tts._lexical_tokens(f"Worth {value} today.") != expected


@pytest.mark.parametrize(("spoken", "expected"), [
    ("one million two hundred and thirty-four thousand five hundred and sixty-seven", "1234567"),
    ("one thousand and one", "1001"),
    ("one million and ten", "1000010"),
    ("fifteen hundred", "1500"),
    ("a hundred thousand", "100000"),
    ("a billion", "1000000000"),
])
def test_grammatical_variants_have_the_same_exact_value(spoken, expected):
    assert tts._lexical_tokens(spoken) == [expected]


@pytest.mark.parametrize("extra", [
    "one billion billion", "one billion one billion", "one million and two billion",
    "one hundred hundred", "one hundred and", "one trillion", "one gazillion",
    "one billion zero",
])
def test_invalid_or_unsupported_phrases_do_not_erase_tokens(extra):
    tokens = tts._lexical_tokens(extra)
    assert len(tokens) >= 2
    assert all(tokens)


@pytest.mark.parametrize("text", [
    "001,234", "12,34,567", "1,000,000,000,000", "AX1,234", "ID-1,234",
    "1,234-R2", "1,234/AB", "1,234.50", "-1,234", "+1,234", "$1,234",
    "€44", "£44", "¥44", "$44", "+44", "-44", "44/AB", "44-R2", "V44", "04",
])
def test_narration_rewrite_preserves_protected_and_unsupported_forms(text):
    assert normalize_spoken_quantities(text) == text
    assert tts._pocket_synthesis_text(text) == text


def test_grouped_currency_keeps_its_unit_when_separators_are_normalized():
    assert normalize_grouped_integer_literals("$1,234,567 dollars") == "$1234567 dollars"
    assert tts._lexical_tokens("$1,234,567") == ["1234567", "dollars"]


@pytest.mark.parametrize("value", [-1, 10**12, 1.5, True])
def test_unsupported_integer_spelling_fails_explicitly(value):
    with pytest.raises(ValueError, match="English integer"):
        integer_words(value)


def test_number_index_contract_rejects_misaligned_input():
    with pytest.raises(ValueError, match="equal length"):
        canonicalize_integer_tokens(["1", "billion"], [0])

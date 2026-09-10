"""Script-level regression for the 600,000 homes pronunciation failure."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from backend.daily_news.scriptwriter import enforce_script_contract
from backend.pipeline import digester, tts
from backend.spoken_numbers import normalize_spoken_quantities


@pytest.mark.parametrize(("source", "expected"), [
    ("600,000 homes", "six hundred thousand homes"),
    ("1,234,567 homes", "one million two hundred thirty-four thousand five hundred sixty-seven homes"),
    ("3,000,000 units", "three million units"),
    ("12,001 items", "twelve thousand one items"),
    ("999,999,999,999", "nine hundred ninety-nine billion nine hundred ninety-nine million nine hundred ninety-nine thousand nine hundred ninety-nine"),
])
def test_grouped_quantity_spelling(source, expected):
    result = normalize_spoken_quantities(source)
    assert result == expected
    assert normalize_spoken_quantities(result) == result


@pytest.mark.parametrize("source", [
    "D1 Qwen3 2026 7.08 61.1% 01 000,123",
    "ID-600,000 AX600,000 600,000-R2 600,000/AB",
    "$600,000 €600,000 £600,000 ¥600,000 600,000.25",
    "12,34,567 1,000,000,000,000 -600,000 +600,000",
])
def test_protected_numeric_forms_are_unchanged(source):
    assert normalize_spoken_quantities(source) == source


def test_daily_script_contract_normalizes_body_before_narration():
    body = "Enough for the daily electricity needs of 600,000 homes."
    result = enforce_script_contract(body, opening="Opening 2026.", closing="Closing.", language="en")
    assert result.splitlines() == [
        "Opening 2026.",
        "Enough for the daily electricity needs of six hundred thousand homes.",
        "Closing.",
    ]
    expected = result.splitlines()[1]
    assert tts._pocket_synthesis_text(expected) == expected
    wrong = expected.replace("six hundred thousand", "six hundreds zero zero zero")
    words = [{"text": word, "start": i * .2, "end": i * .2 + .1} for i, word in enumerate(wrong.split())]
    assert not tts._orpheus_transcript_report(expected, words)["verified"]


def test_chinese_and_protected_passing_paragraphs_are_unchanged():
    for language, normalize in [("zh", True), ("en", False)]:
        body = "600,000 homes."
        result = enforce_script_contract(body, opening="Opening.", closing="Closing.", language=language, normalize_numbers=normalize)
        assert result.splitlines()[1] == body


def test_general_script_generation_spells_quantity_from_model_response():
    with (
        patch.object(digester, "_resolve_provider", AsyncMock(return_value=("http://x/v1", "model", "key"))),
        patch.object(digester, "_chat", AsyncMock(return_value="Enough for the daily electricity needs of 600,000 homes.")) as chat,
    ):
        result = asyncio.run(digester.generate_script({"title": "Test"}, closing_remarks="Closing."))
    assert "six hundred thousand homes." in result
    assert "600,000" not in result
    assert result.endswith("Closing.")
    assert "Never read a quantity digit by digit" in chat.await_args.args[0]

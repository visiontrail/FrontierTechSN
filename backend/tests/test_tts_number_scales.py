"""Regression coverage for the billion lookup failure in Pocket narration."""

import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from backend.pipeline import av_sync, tts
from backend.tests.test_tts import pocket_streaming_wav_bytes


def words(text):
    return [
        {"text": word, "start": index * .2, "end": index * .2 + .1}
        for index, word in enumerate(text.split())
    ]


@pytest.mark.parametrize(("spoken", "literal"), [
    ("one hundred and ten billion", "110000000000"),
    ("one hundred ten billion", "110 billion"),
    ("two hundred seventy-eight billion", "278 billion"),
    ("sixty-eight billion", "68000000000"),
    ("one hundred billion", "100000000000"),
    ("a billion", "1000000000"),
    ("one hundred and ten million", "110000000"),
    ("one hundred and ten thousand", "110000"),
    ("one point five billion", "1.5 billion"),
])
def test_spoken_scales_preserve_value_and_transcript_onset(spoken, literal):
    source = f"The deal is worth {spoken} dollars today."
    observed = f"The deal is worth {literal} dollars today."
    expected = tts._lexical_tokens(observed)
    assert tts._lexical_tokens(source) == expected
    assert tts._lexical_tokens(tts._pocket_synthesis_text(source)) == expected
    assert tts._orpheus_transcript_report(source, words(observed))["verified"]
    tokens, indexes = tts._transcript_tokens(words(source))
    assert tokens == expected
    assert indexes[4] == 4  # Amount starts at the first spoken number word.
    assert len(tokens) == len(indexes)
    assert indexes[-2:] == [len(source.split()) - 2, len(source.split()) - 1]


@pytest.mark.parametrize("observed", [
    "one hundred and ten million dollars",
    "one hundred and eleven billion dollars",
    "one hundred and ten dollars",
    "one hundred and ten billion billion dollars",
    "one hundred and ten billion yuan",
    "one hundred and ten billion",
])
def test_large_amount_normalization_does_not_hide_changed_content(observed):
    source = "The deal is worth one hundred and ten billion dollars today."
    assert not tts._orpheus_transcript_report(
        source, words(f"The deal is worth {observed} today."),
    )["verified"]


@pytest.mark.asyncio
async def test_normalization_failure_has_context_and_resume_reuses_completed_audio(tmp_path):
    paragraphs = [
        "The company confirmed its planned acquisition today, with the final "
        "agreement expected to be completed later this year after review.",
        "The paper describes the deal as a one hundred and ten billion dollar "
        "acquisition. The settlement terms were not disclosed.",
    ]
    script = tmp_path / "script.txt"
    source = "\n".join(paragraphs)
    script.write_text(source, encoding="utf-8")
    output = tmp_path / "audio"
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            200, headers={"Content-Type": "audio/wav"},
            content=pocket_streaming_wav_bytes(frames=288_000),
        )

    original_client = httpx.AsyncClient
    def client_factory(**kwargs):
        return original_client(transport=httpx.MockTransport(respond), **kwargs)

    original_synthesis = tts._pocket_synthesis_text
    def fail_second_paragraph(text):
        if "billion" in text:
            raise KeyError("billion")
        return original_synthesis(text)

    messages = []
    with (
        patch.object(tts.httpx, "AsyncClient", client_factory),
        patch.object(av_sync, "ensure_word_transcript", AsyncMock(
            side_effect=[(words(paragraph), {}) for paragraph in paragraphs],
        )) as transcribe,
    ):
        with patch.object(tts, "_pocket_synthesis_text", fail_second_paragraph):
            with pytest.raises(RuntimeError, match=(
                "TTS part 2/2: Pocket TTS pronunciation normalization failed "
                r"\(KeyError: 'billion'\)"
            )) as raised:
                await tts.generate_tts(
                    str(script), str(output), ["jane"], "pocket-tts-en", log=messages.append,
                )
        assert isinstance(raised.value.__cause__, KeyError)
        assert len(requests) == 1  # A code error must not retry the provider.
        first_audio = output / "tts_input_part_001_generated.wav"
        saved_audio = first_audio.read_bytes()
        saved_mtime = first_audio.stat().st_mtime_ns

        await tts.generate_tts(
            str(script), str(output), ["jane"], "pocket-tts-en", log=messages.append,
        )

    assert len(requests) == 2
    assert transcribe.await_count == 2
    assert first_audio.read_bytes() == saved_audio
    assert first_audio.stat().st_mtime_ns == saved_mtime
    assert script.read_text(encoding="utf-8") == source
    assert any("TTS part 1/2: reusing validated" in message for message in messages)
    manifest = json.loads((output / "tts_manifest.json").read_text())
    assert manifest["chunk_count"] == 2
    assert manifest["integrity"]["passed"]

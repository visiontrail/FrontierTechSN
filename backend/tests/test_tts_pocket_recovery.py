"""Regressions from the 2026-09-16 physicist paragraph integrity failure."""

import hashlib
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch
from urllib.parse import parse_qs

import httpx
import pytest

from backend import config
from backend.pipeline import tts
from backend.tests.test_tts import pocket_streaming_wav_bytes


def words(text):
    return [{"text": token, "start": i * .3, "end": i * .3 + .2}
            for i, token in enumerate(text.split())]


@pytest.mark.parametrize(("spoken", "digits"), [
    ("two hundred seventy-eight", "278"),
    ("two hundred and seventy-eight", "278"),
    ("one hundred one", "101"),
    ("nine hundred ninety-nine", "999"),
    ("two hundred fifty thousand", "250,000"),
    ("two hundred seventy-eight million", "278 million"),
])
def test_compound_hundreds_preserve_the_exact_value_and_word_indexes(spoken, digits):
    source = f"The program selected {spoken} projects this past July."
    transcript = words(source.replace(spoken, digits))
    assert tts._orpheus_transcript_report(source, transcript)["verified"]
    assert tts._orpheus_transcript_report(source.replace(spoken, digits), words(source))["verified"]
    normalized, indexes = tts._transcript_tokens(words(source))
    assert normalized == tts._lexical_tokens(source)
    assert indexes[normalized.index("projects")] == source.split().index("projects")


@pytest.mark.parametrize("wrong", ["78", "200", "277", "287", "278 million", "two seventy-eight"])
def test_compound_hundreds_do_not_accept_changed_values(wrong):
    source = "The program selected two hundred seventy-eight projects this past July."
    assert not tts._orpheus_transcript_report(
        source, words(f"The program selected {wrong} projects this past July."),
    )["verified"]


@pytest.mark.parametrize(("source_acronyms", "observed", "accepted"), [
    ("US AI", "USAI", True), ("EU AI", "EUAI", True),
    ("US AI", "USA", False), ("US AI", "USAGI", False),
    ("US AI", "AIUS", False), ("us AI", "USAI", False),
    ("US A1", "USA1", False), ("US AI", "USAI AI", False),
])
def test_acronym_boundary_normalization_requires_exact_source_letters(source_acronyms, observed, accepted):
    source = f"One physicist on a {source_acronyms}-for-science program offers a split verdict."
    report = tts._orpheus_transcript_report(source, words(source.replace(source_acronyms, observed)))
    assert report["verified"] is accepted


@pytest.mark.parametrize("marker", ["valid", "stale", "malformed", "absent"])
def test_sentence_recovery_is_source_bound_and_preserves_other_part_names(tmp_path, marker):
    paragraph = "Dr. Earley studies E. coli. The apparatus stays in place."
    source = f"Opening stays here.\n{paragraph}\nClosing stays here."
    directory = tmp_path / "verification" / "tts_input_part_002"
    directory.mkdir(parents=True)
    failure = directory / "integrity_failure.json"
    if marker != "absent":
        failure.write_text("[]" if marker == "malformed" else json.dumps({
            "text_sha256": hashlib.sha256((paragraph if marker == "valid" else "older text").encode()).hexdigest(),
        }))
    paths, chunks = tts._write_pocket_chunk_inputs(source, tmp_path, max_words=240, retry_failed_paragraphs=True)
    assert paths[0].name == "tts_input_part_001.txt"
    assert paths[-1].name == "tts_input_part_003.txt"
    assert " ".join(chunks) == " ".join(source.splitlines())
    assert len(chunks) == (4 if marker == "valid" else 3)
    if marker == "valid":
        assert chunks[1:3] == ["Dr. Earley studies E. coli.", "The apparatus stays in place."]
        assert paths[1].name == "tts_input_part_002_sentence_001.txt"
    # Voice previews do not inherit a production integrity retry plan.
    assert len(tts._write_pocket_chunk_inputs(source, tmp_path, max_words=240)[1]) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("sentence_fails", [False, True])
async def test_failed_paragraph_recovers_by_sentence_with_cache_and_bounded_retries(tmp_path, sentence_fails):
    opening = "Opening context stays here."
    first = "The apparatus runs near the magnet."
    second = "Its targets stay in the laboratory."
    closing = "Closing context stays here."
    paragraph = f"{first} {second}"
    script = tmp_path / "script.txt"
    script.write_text("\n".join([opening, paragraph, closing]))
    output_dir = tmp_path / "audio"
    requests = []

    def respond(request):
        requests.append(parse_qs(request.content.decode())["text"][0])
        return httpx.Response(200, headers={"Content-Type": "audio/wav"},
                              content=pocket_streaming_wav_bytes(frames=96_000))

    async def verify(_path, text, _directory, **_kwargs):
        if text == paragraph or (sentence_fails and text == second):
            raise tts.TtsIntegrityError("source 'magnet' -> ASR 'market'")
        return tts._orpheus_transcript_report(text, words(text))

    original_client = httpx.AsyncClient
    def client_factory(**kwargs):
        return original_client(transport=httpx.MockTransport(respond), **kwargs)

    with (patch.object(tts.httpx, "AsyncClient", client_factory),
          patch.object(tts, "_verify_orpheus_part", AsyncMock(side_effect=verify)),
          patch.object(config, "POCKET_TTS_CHUNK_WORDS", 240)):
        if sentence_fails:
            with pytest.raises(tts.TtsIntegrityError, match="sentence_002"):
                await tts.generate_tts(str(script), str(output_dir), ["alba"], "pocket-tts-en")
            assert requests == [opening, paragraph, first, second, second, second]
            assert not (output_dir / "tts_manifest.json").exists()
        else:
            result = await tts.generate_tts(str(script), str(output_dir), ["alba"], "pocket-tts-en")
            assert requests == [opening, paragraph, first, second, closing]
            manifest = json.loads((output_dir / "tts_manifest.json").read_text())
            assert manifest["integrity"]["verified_source_coverage"] == 1.0
            assert manifest["chunk_count"] == 4
            assert manifest["continuity"]["intra_line_application_join_count"] == 1
            assert tts._read_pcm_wav(Path(result)).frame_count == 4 * 96_000
            # A fresh call reconstructs the same plan and reuses every verified sentence.
            await tts.generate_tts(str(script), str(output_dir), ["alba"], "pocket-tts-en")
            assert len(requests) == 5
    failure_dir = output_dir / "verification" / "tts_input_part_002"
    assert (failure_dir / "rejected.wav").is_file()
    assert "magnet" in json.loads((failure_dir / "integrity_failure.json").read_text())["reason"]
    assert not (output_dir / "tts_input_part_002_generated.wav").exists()


@pytest.mark.asyncio
async def test_integrity_failure_reports_actual_token_differences(tmp_path):
    from backend.pipeline import av_sync

    source = "The program selected two hundred seventy-eight projects this past July."
    with patch.object(av_sync, "ensure_word_transcript", AsyncMock(return_value=(
        words(source.replace("two hundred seventy-eight", "277")), {},
    ))):
        with pytest.raises(tts.TtsIntegrityError, match="source '278' -> ASR '277'"):
            await tts._verify_orpheus_part(tmp_path / "sample.wav", source, tmp_path, emit=lambda _: None)
    report = json.loads((tmp_path / "integrity_report.json").read_text())
    assert report["token_differences"] == [{"kind": "replace", "source": ["278"], "asr": ["277"]}]


@pytest.mark.parametrize("defect", [None, "wrong_source", "approved", "existing_wav", "malformed"])
def test_recovers_legacy_rejected_paragraph_without_replacing_an_existing_wav(tmp_path, defect):
    source = "First complete sentence. Second complete sentence."
    directory = tmp_path / "verification" / "tts_input"
    directory.mkdir(parents=True)
    evidence = {"status": "approved" if defect == "approved" else "rejected", "request": {
        "source_text": "An older source." if defect == "wrong_source" else source,
    }}
    (directory / "llm_asr_adjudication.json").write_text(
        "[]" if defect == "malformed" else json.dumps(evidence),
    )
    if defect == "existing_wav":
        (tmp_path / "tts_input_generated.wav").write_bytes(b"existing audio")
    paths, chunks = tts._write_pocket_chunk_inputs(source, tmp_path, max_words=240, retry_failed_paragraphs=True)
    assert len(paths) == (2 if defect is None else 1)
    assert " ".join(chunks) == source


@pytest.mark.parametrize(("observed", "accepted"), [
    ("13–15", True), ("13 -15", True), ("13 through 15", True),
    ("13 to 15", False), ("13–16", False), ("12–15", False),
    ("15–13", False), ("13 and 15", False), ("13 15", False),
])
def test_through_range_accepts_only_exact_written_shorthand(observed, accepted):
    source = "The conference runs this October 13 through 15 in San Francisco."
    assert tts._orpheus_transcript_report(
        source, words(source.replace("13 through 15", observed)),
    )["verified"] is accepted


@pytest.mark.parametrize(("source_number", "transcript_number"), [
    ("two hundred seventy-eight", "278"),
    ("September 25", "September 25"),
    ("October 13 through 15", "October 13 -15"),
    ("twenty-three point five", "23.5"),
])
def test_medium_name_verdict_handles_numeric_formatting_without_treating_dates_as_brands(source_number, transcript_number):
    source = f"DeepTech China reports that venture firm Andreessen Horowitz announced {source_number} today."
    transcript = source.replace("DeepTech", "Deep Tech").replace("Andreessen", "Andreasen").replace(source_number, transcript_number)
    assert tts._medium_asr_verdict_is_corroborated(tts._lexical_tokens(source), transcript, transcript)
    # Keep the remainder of this otherwise close paragraph identical.
    for wrong in [transcript.replace(transcript_number, "999"), transcript + " 999"]:
        assert not tts._medium_asr_verdict_is_corroborated(tts._lexical_tokens(source), wrong, wrong)


@pytest.mark.parametrize(("spoken", "digits"), [
    ("twenty-three point five", "23.5"),
    ("two hundred seventy-eight point zero five", "278.05"),
    ("ninety-nine point nine", "99.9"),
])
def test_spoken_decimal_normalizes_its_entire_integer_prefix(spoken, digits):
    source = f"The reported amount is {spoken} million dollars today."
    observed = source.replace(spoken, digits)
    assert tts._orpheus_transcript_report(source, words(observed))["verified"]
    assert tts._orpheus_transcript_report(observed, words(source))["verified"]
    assert not tts._orpheus_transcript_report(source, words(observed.replace(digits, "33.5")))["verified"]
    tokens, indexes = tts._transcript_tokens(words(source))
    assert indexes[tokens.index("dollars")] == source.split().index("dollars")


@pytest.mark.parametrize("identity", ["a16z", "T6", "Mu2e"])
def test_medium_verdict_requires_exact_model_identity_in_both_transcripts(identity):
    source = f"DeepTech reports that Andreessen Horowitz discussed {identity} during the conference today."
    normal = source.replace("DeepTech", "Deep Tech").replace("Andreessen", "Andreasen")
    expected = tts._lexical_tokens(source)
    assert tts._medium_asr_verdict_is_corroborated(expected, normal, normal)
    changed = normal.replace(identity, identity.replace("6", "7").replace("2", "3"))
    assert not tts._medium_asr_verdict_is_corroborated(expected, normal, changed)
    assert not tts._medium_asr_verdict_is_corroborated(expected, changed, changed)


def test_sentence_recovery_keeps_uppercase_model_abbreviation_with_its_name():
    source = "The chip will go into the ID. AURA T6. The next shipment leaves tomorrow."
    assert tts._split_pocket_tts_text(source, max_words=1) == [
        "The chip will go into the ID. AURA T6.", "The next shipment leaves tomorrow.",
    ]


@pytest.mark.parametrize(("source", "observed", "accepted"), [
    ("Kernel", "Colonel", True), ("scent", "sent", True),
    ("read", "rid", False), ("read", "reed", False),
    ("Quanta", "Quantum", False), ("kernel", "kernels", False),
])
def test_dictionary_homophones_require_one_unambiguous_identical_pronunciation(source, observed, accepted):
    substitutions = tts._aligned_phonetic_substitutions([source.lower()], [observed.lower()])
    assert bool(substitutions) is accepted
    if accepted:
        assert substitutions[0]["phonetic_key"].startswith("evidenced:cmudict:")


@pytest.mark.asyncio
@pytest.mark.parametrize("corroborates", [True, False])
async def test_dictionary_homophone_requires_same_waveform_corroboration_without_model(tmp_path, corroborates):
    from backend.pipeline import av_sync

    source = "The chief executive of Kernel has announced his departure today."
    normal = words(source.replace("Kernel", "Colonel"))
    slower = normal if corroborates else words(source.replace("Kernel", "Kernels"))
    with (patch.object(av_sync, "ensure_word_transcript", AsyncMock(return_value=(normal, {}))),
          patch.object(tts, "_transcribe_orpheus_at_speed", AsyncMock(return_value=(slower, {}))),
          patch.object(tts, "_adjudicate_orpheus_asr_mismatch", AsyncMock()) as judge):
        if corroborates:
            report = await tts._verify_orpheus_part(tmp_path / "audio.wav", source, tmp_path, emit=lambda _: None)
            assert report["verified"]
            assert report["verification_mode"] == "corroborated_phonetic_substitution"
        else:
            with pytest.raises(tts.TtsIntegrityError, match="not corroborated"):
                await tts._verify_orpheus_part(tmp_path / "audio.wav", source, tmp_path, emit=lambda _: None)
    judge.assert_not_awaited()


@pytest.mark.parametrize("defect", [None, "both_decodes", "no_overlap", "different_word", "omission", "changed_number"])
def test_medium_verdict_requires_one_sided_timestamp_overlapped_identical_duplicate(defect):
    source = "DeepTech reports that the chief executive Andreessen announced the T6 model today."
    clean = words(source.replace("DeepTech", "Deep Tech").replace("Andreessen", "Andreasen"))
    duplicated = words(tts._raw_transcript(clean).replace("chief", "chief Chief"))
    index = next(i for i, word in enumerate(duplicated) if word["text"] == "Chief")
    duplicated[index]["start"] = duplicated[index - 1]["start"] + .04
    duplicated[index]["end"] = duplicated[index - 1]["end"] + .04
    if defect == "both_decodes":
        clean = duplicated
    elif defect == "no_overlap":
        duplicated = words(tts._raw_transcript(duplicated))
    elif defect == "different_word":
        duplicated[index]["text"] = "senior"
    elif defect == "omission":
        clean = [w for w in clean if w["text"] != "executive"]
        duplicated = [w for w in duplicated if w["text"] != "executive"]
    elif defect == "changed_number":
        duplicated = [dict(w, text="T7") if w["text"] == "T6" else w for w in duplicated]
    assert tts._medium_asr_verdict_is_corroborated(
        tts._lexical_tokens(source), tts._raw_transcript(clean), tts._raw_transcript(duplicated), clean, duplicated,
    ) is (defect is None)

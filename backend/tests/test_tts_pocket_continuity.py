"""Regressions from the September 27 quoted-sentence pause failure."""

import hashlib
import json
import struct
import wave
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from backend.pipeline import av_sync, tts
from backend.tests.test_tts import pocket_streaming_wav_bytes


def write_pause(path, seconds=1.1):
    with wave.open(str(path), "wb") as output:
        output.setparams((1, 2, 24_000, 0, "NONE", "not compressed"))
        output.writeframes(
            struct.pack("<h", 10_000) * 89_280
            + struct.pack("<h", 0) * round(seconds * 24_000)
            + struct.pack("<h", 10_000) * 96_000
        )


@pytest.mark.parametrize(("ending", "overlap", "pause", "passed"), [
    ("app.", .1, 1.1, True),
    ("app.'", .2, 1.1, True),
    ("app.\u201d", .1, 1.1, True),
    ("app.", .21, 1.1, False),
    ("app,", .1, 1.1, False),
    ("app", .1, .9, False),
    ("app.", .1, 1.21, False),
])
def test_pause_uses_adjacent_word_with_bounded_timestamp_drift(
    tmp_path, ending, overlap, pause, passed,
):
    path = tmp_path / "sample.wav"
    write_pause(path, pause)
    words = [
        {"text": "super", "start": 3.04, "end": 3.42},
        {"text": ending, "start": 3.42, "end": 3.72 + overlap},
        {"text": "The", "start": 4.22, "end": 3.72 + pause + .06},
    ]
    original = path.read_bytes()
    report = tts._pocket_internal_silence_report(path, words)
    assert report["passed"] is passed
    assert report["max_seconds"] == pause
    assert report["maximum_allowed_seconds"] == .8
    assert report["maximum_sentence_boundary_allowed_seconds"] == 1.2
    if passed:
        assert report["runs"][0]["previous_word"] == ending
        assert report["runs"][0]["sentence_boundary"] is True
    else:
        with pytest.raises(tts.TtsIntegrityError, match="continuity ceiling"):
            tts._validate_pocket_internal_silence(path, words)
    assert path.read_bytes() == original


@pytest.mark.asyncio
async def test_advisory_path_accepts_real_boundary_timestamps_without_editing_audio(tmp_path):
    path = tmp_path / "sample.wav"
    write_pause(path)
    original = path.read_bytes()
    words = [
        {"text": "super", "start": 3.04, "end": 3.42},
        {"text": "app.", "start": 3.42, "end": 3.82},
        {"text": "The", "start": 4.22, "end": 4.88},
    ]
    source = "Microsoft has bundled its AI tools into one 'super app.' The announcement follows."
    with patch.object(av_sync, "ensure_word_transcript", AsyncMock(return_value=(words, {}))):
        report = await tts._verify_orpheus_part(
            path, source, tmp_path / "verification", emit=lambda _: None,
            provider_label="Pocket TTS", asr_advisory=True, validate_pocket_continuity=True,
        )
    assert report["accepted"] is True
    assert report["asr_status"] == "mismatch"
    assert report["warnings"]
    assert report["internal_silence"]["runs"][0]["previous_word"] == "app."
    assert report["internal_silence"]["passed"] is True
    assert path.read_bytes() == original


@pytest.mark.parametrize(("opening", "closing"), [
    ("'", "'"), ('"', '"'), ("\u201c", "\u201d"), ("\u2018", "\u2019"), ("(", ")"), ("[", "]"),
])
def test_rejected_paragraph_splits_after_quoted_sentence(tmp_path, opening, closing):
    first = f"Microsoft calls it a {opening}super app.{closing}"
    second = "Dr. Earley spoke to U.S. manufacturers."
    source = f"{first} {second}"
    directory = tmp_path / "verification" / "tts_input"
    directory.mkdir(parents=True)
    (directory / "integrity_failure.json").write_text(json.dumps({
        "text_sha256": hashlib.sha256(source.encode()).hexdigest(),
    }))
    paths, chunks = tts._write_pocket_chunk_inputs(
        source, tmp_path, max_words=240, retry_failed_paragraphs=True,
    )
    assert chunks == [first, second]
    assert " ".join(chunks) == source
    assert [path.name for path in paths] == [
        "tts_input_sentence_001.txt", "tts_input_sentence_002.txt",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["source", "voice", "revision", "waveform", "malformed", "verifier"])
async def test_stale_cache_recovery_respects_identity_after_sentence_renumbering(tmp_path, change):
    source = "This complete sentence needs its own audio."
    script = tmp_path / "script.txt"
    script.write_text(source)
    output = tmp_path / "audio"
    output.mkdir()
    path = output / "tts_input_generated.wav"
    path.write_bytes(pocket_streaming_wav_bytes(frames=96_000))
    tts._normalize_pocket_streaming_wav(path)
    metadata = tts._write_pocket_part_metadata(
        path, source + " Another sentence." if change == "source" else source,
        "alba" if change == "voice" else "jane",
        integrity={"policy": tts.ASR_ADVISORY_POLICY, "accepted": True}, generation_seconds=1,
    )
    if change == "revision":
        metadata["model_revision"] = "older-deployment"
    elif change == "verifier":
        metadata["integrity_verifier_version"] -= 1
    elif change == "waveform":
        # Still a decodable WAV, but no longer the audio the sidecar describes.
        path.write_bytes(pocket_streaming_wav_bytes(frames=120_000))
        tts._normalize_pocket_streaming_wav(path)
    tts._part_metadata_path(path).write_text("[]" if change == "malformed" else json.dumps(metadata))
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, headers={"Content-Type": "audio/wav"},
                              content=pocket_streaming_wav_bytes(frames=144_000))

    original_client = httpx.AsyncClient

    def client_factory(**kwargs):
        return original_client(transport=httpx.MockTransport(respond), **kwargs)

    with (patch.object(tts.httpx, "AsyncClient", client_factory),
          patch.object(av_sync, "ensure_word_transcript", AsyncMock(return_value=([], {})))):
        await tts.generate_tts(str(script), str(output), ["jane"], "pocket-tts-en")
        await tts.generate_tts(str(script), str(output), ["jane"], "pocket-tts-en")
    assert len(requests) == (0 if change == "verifier" else 1)
    assert tts._read_pcm_wav(path).frame_count == (96_000 if change == "verifier" else 144_000)
    refreshed = tts._load_cached_pocket_part(path, source, "jane")
    assert refreshed is not None
    assert refreshed["integrity_verifier_version"] == tts.POCKET_TTS_INTEGRITY_VERIFIER_VERSION

"""ASR diagnostics must not stop synthesis, cache reuse, or video composition."""

import json
import wave
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from backend.pipeline import av_sync, composer, tts
from backend.tests.test_tts import pocket_streaming_wav_bytes


SOURCE = (
    "The outlet reports that RPent, an embodied-agent infrastructure co-initiated "
    "by Tsinghua University, Infinigence AI and Zhengxing Innovation, was open-sourced today."
)
OBSERVED = SOURCE.replace("Infinigence", "Infinegen's").replace("Zhengxing", "Jungxing").replace("sourced", "source")


def words(text):
    return [{"text": token, "start": i * .3, "end": i * .3 + .2}
            for i, token in enumerate(text.split())]


@pytest.mark.asyncio
@pytest.mark.parametrize("transcript", [OBSERVED, SOURCE + " " + SOURCE, "", None])
async def test_real_mismatch_and_asr_outage_continue_once_and_reuse_cache(tmp_path, transcript):
    script = tmp_path / "script.txt"
    script.write_text(SOURCE)
    output = tmp_path / "audio"
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, headers={"Content-Type": "audio/wav"},
                              content=pocket_streaming_wav_bytes(frames=288_000))

    original_client = httpx.AsyncClient
    def client_factory(**kwargs):
        return original_client(transport=httpx.MockTransport(respond), **kwargs)

    transcribe = (AsyncMock(side_effect=RuntimeError("ASR unavailable")) if transcript is None
                  else AsyncMock(return_value=(words(transcript), {})))
    messages = []
    with (patch.object(tts.httpx, "AsyncClient", client_factory),
          patch.object(av_sync, "ensure_word_transcript", transcribe),
          patch.object(tts, "_adjudicate_orpheus_asr_mismatch", AsyncMock()) as judge,
          patch.object(tts, "_transcribe_orpheus_at_speed", AsyncMock()) as slower,
          patch.object(tts, "_trim_pcm_wav") as trim):
        result = await tts.generate_tts(str(script), str(output), ["jane"], "pocket-tts-en", log=messages.append)
        await tts.generate_tts(str(script), str(output), ["jane"], "pocket-tts-en", log=messages.append)
    assert len(requests) == 1
    assert transcribe.await_count == 1
    judge.assert_not_awaited()
    slower.assert_not_awaited()
    trim.assert_not_called()
    manifest = json.loads((output / "tts_manifest.json").read_text())
    integrity = manifest["integrity"]
    assert integrity["passed"] is True
    assert integrity["policy"] == "asr_advisory"
    assert integrity["accepted_source_coverage"] == 1
    assert integrity["verified_source_coverage"] == 0
    assert integrity["asr_warning_parts"] == 1
    report = integrity["part_reports"][0]
    assert report["verified"] is False
    assert report["accepted"] is True
    assert report["warnings"]
    if transcript == OBSERVED:
        assert report["exact_asr_word_coverage"] == .8696
        assert report["token_differences"][-1]["asr"] == ["source"]
    assert any("ASR reference warning" in message for message in messages)
    assert not list(output.rglob("integrity_failure.json"))
    assert composer._narration_manifest_failures(script, result, "pocket-tts-en") == []
    segments = composer._manifest_program_segments(result, [{"text": SOURCE}])
    assert segments
    alignment = composer._apply_manifest_program_alignment(
        {"passed": False, "word_coverage": .8696}, {"passed": True}, segments,
    )
    assert alignment["source_contract"]["asr_policy"] == "asr_advisory"
    assert "per_utterance_acoustic_coverage" not in alignment["source_contract"]
    assert alignment["whole_file_asr_observation"]["word_coverage"] == .8696
    manifest["integrity"]["accepted_source_coverage"] = .5
    (output / "tts_manifest.json").write_text(json.dumps(manifest))
    assert composer._narration_manifest_failures(script, result, "pocket-tts-en")
    assert composer._manifest_program_segments(result, [{"text": SOURCE}]) is None
    script.write_text(SOURCE + " Changed script.")
    assert any("script changed" in failure for failure in composer._narration_manifest_failures(script, result, "pocket-tts-en"))


@pytest.mark.asyncio
@pytest.mark.parametrize("defect", ["corrupt", "empty", "truncated", "too_short"])
async def test_advisory_policy_still_rejects_invalid_audio(tmp_path, defect):
    path = tmp_path / "sample.wav"
    path.write_bytes(pocket_streaming_wav_bytes(frames=288_000))
    tts._normalize_pocket_streaming_wav(path)
    if defect == "corrupt":
        path.write_bytes(b"not a wav")
    elif defect == "empty":
        with wave.open(str(path), "wb") as handle:
            handle.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
            handle.writeframes(b"")
    elif defect == "truncated":
        path.write_bytes(path.read_bytes()[:100])
    else:
        path.write_bytes(pocket_streaming_wav_bytes(frames=2400))
        tts._normalize_pocket_streaming_wav(path)
    with patch.object(av_sync, "ensure_word_transcript", AsyncMock()) as transcribe:
        with pytest.raises(tts.TtsIntegrityError):
            await tts._verify_orpheus_part(path, SOURCE, tmp_path, emit=lambda _: None, asr_advisory=True)
    transcribe.assert_not_awaited()


def test_advisory_summary_cannot_hide_missing_parts_or_failed_validation():
    accepted = {"word_count": 5, "integrity": {"policy": "asr_advisory", "accepted": True, "verified": False}}
    failed = {"word_count": 5, "integrity": {"policy": "asr_advisory", "accepted": False, "verified": True}}
    for parts in ([accepted], [accepted, failed]):
        report = tts._narration_integrity_summary(parts, 10, required=True, method="test")
        assert report["passed"] is False
        assert report["accepted_source_coverage"] == .5


@pytest.mark.asyncio
async def test_orpheus_recovery_uses_advisory_policy_and_cache(tmp_path):
    path = tmp_path / "sample.wav"
    path.write_bytes(pocket_streaming_wav_bytes(frames=288_000))
    tts._normalize_pocket_streaming_wav(path)
    with patch.object(av_sync, "ensure_word_transcript", AsyncMock(return_value=(words(OBSERVED), {}))):
        metadata = await tts._recover_orpheus_part(
            path, SOURCE, tmp_path / "verification", request_token_budget=512, emit=lambda _: None,
        )
    assert metadata["integrity"]["accepted"] is True
    assert metadata["integrity"]["verified"] is False
    assert tts._load_cached_orpheus_part(path, SOURCE) == metadata

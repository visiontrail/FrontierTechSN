"""Narration boundary tests; no network, ASR model, or TTS model is loaded."""

import hashlib
import io
import json
import struct
import wave
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest

from backend import config
from backend.pipeline import chatterbox, composer, tts


def pcm_wav(*, seconds=3.0, sample_rate=24_000, channels=1):
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(struct.pack("<h", 1000) * int(seconds * sample_rate) * channels)
    return output.getvalue()


def advisory_report():
    return {
        "policy": tts.ASR_ADVISORY_POLICY,
        "accepted": True,
        "verified": False,
        "asr_status": "mismatch",
        "warnings": ["ASR used a different spelling"],
        "internal_silence": {"count": 0, "max_seconds": 0.0},
    }


class FakeService:
    """Emulate durable jobs and a lost POST response at the HTTP boundary."""

    def __init__(self):
        self.capabilities = {
            "source_revision": config.CHATTERBOX_TTS_SOURCE_REVISION,
            "model_revision": config.CHATTERBOX_TTS_MODEL_REVISION,
            "model_variant": "v2",
            "fingerprint": "deployment-a",
            "max_chars": 250,
            "sample_rate": 24_000,
            "channels": 1,
            "encoding": "PCM_16",
            "voices": [{"id": "jane", "fingerprint": "voice-a", "reference_sha256": "ref-a"}],
        }
        self.jobs = {}
        self.keys = {}
        self.requests = []
        self.post_keys = []
        self.lose_post_response = False
        self.fail_poll = False
        self.poll_errors = {}
        self.poll_transport_failure = False
        self.truncate_audio = False
        self.corrupt_download = False
        self.job_changes = {}
        self.job_changes_by_id = {}

    def handle(self, request):
        self.requests.append((request.method, request.url.path))
        assert request.headers.get("x-api-key") == "test-secret"
        if request.url.path == "/v1/capabilities":
            return httpx.Response(200, json=self.capabilities)
        if request.method == "POST" and request.url.path == "/v1/jobs":
            key = request.headers["idempotency-key"]
            self.post_keys.append(key)
            if key not in self.keys:
                payload = json.loads(request.content)
                job_id = f"job-{len(self.jobs) + 1}"
                audio = pcm_wav(seconds=max(3.0, len(payload["text"].split()) * 0.4))
                if self.truncate_audio:
                    audio = audio[:-100]
                self.keys[key] = job_id
                self.jobs[job_id] = {
                    "id": job_id,
                    "status": "completed",
                    "request": payload,
                    "text_sha256": hashlib.sha256(payload["text"].encode()).hexdigest(),
                    "fingerprint": self.capabilities["fingerprint"],
                    "voice_fingerprint": self.capabilities["voices"][0]["fingerprint"],
                    "source_revision": self.capabilities["source_revision"],
                    "model_revision": self.capabilities["model_revision"],
                    "model_variant": self.capabilities["model_variant"],
                    "audio_sha256": hashlib.sha256(audio).hexdigest(),
                    "generation_seconds": 1.0,
                    "audio": audio,
                }
            if self.lose_post_response:
                self.lose_post_response = False
                raise httpx.ReadError("accepted job response was lost", request=request)
            return httpx.Response(202, json={"id": self.keys[key]})
        job_id = request.url.path.split("/")[3]
        job = self.jobs[job_id]
        if request.url.path.endswith("/audio"):
            body = job["audio"] + (b"tampered" if self.corrupt_download else b"")
            return httpx.Response(200, content=body, headers={"content-type": "audio/wav"})
        if self.fail_poll:
            return httpx.Response(401, json={"detail": "temporary test stop"})
        if self.poll_transport_failure:
            raise httpx.ReadError("poll result is unknown", request=request)
        if job_id in self.poll_errors:
            return httpx.Response(self.poll_errors[job_id], json={"detail": "test stop"})
        return httpx.Response(200, json={
            **{key: value for key, value in job.items() if key != "audio"},
            **self.job_changes,
            **self.job_changes_by_id.get(job_id, {}),
        })


@pytest.fixture
def setup_client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CHATTERBOX_TTS_API_KEY", "test-secret")
    monkeypatch.setattr(config, "CHATTERBOX_TTS_URL", "http://chatterbox.test")
    monkeypatch.setattr(config, "CHATTERBOX_TTS_DEPLOYMENT_FINGERPRINT", "deployment-a", raising=False)
    monkeypatch.setattr(config, "CHATTERBOX_TTS_CHUNK_WORDS", 30)
    monkeypatch.setattr(config, "CHATTERBOX_TTS_MAX_CHARS", 250)
    server = FakeService()
    client_class = httpx.AsyncClient
    monkeypatch.setattr(chatterbox.httpx, "AsyncClient", lambda **kwargs: client_class(
        transport=httpx.MockTransport(server.handle), **kwargs,
    ))
    monkeypatch.setattr(chatterbox.asyncio, "sleep", AsyncMock())
    verifier = AsyncMock(side_effect=lambda *args, **kwargs: advisory_report())
    monkeypatch.setattr(tts, "_verify_orpheus_part", verifier)
    script = tmp_path / "script.txt"
    script.write_text("The complete narration remains available for the final video.", encoding="utf-8")
    directory = tmp_path / "audio"

    async def generate():
        return Path(await chatterbox._generate_chatterbox(
            str(script), str(directory), "jane", "en",
            log=None, emit=lambda _message: None, verify_text=True,
        ))

    return server, script, directory, verifier, generate


@pytest.mark.parametrize("max_words,max_chars", [(8, 60), (30, 250), (4, 22)])
def test_bounded_chunks_preserve_every_word_and_punctuation(max_words, max_chars):
    text = "  ".join(["The complete source includes 2.5 billion items, with no missing ending."] * 12)
    chunks = chatterbox._split_text(text, max_words=max_words, max_chars=max_chars)
    assert len(text) > 300
    assert " ".join(chunks) == " ".join(text.split())
    assert all(len(chunk) <= max_chars and len(chunk.split()) <= max_words for chunk in chunks)


def test_split_prefers_complete_sentences_without_separating_abbreviations():
    sentence = "Dr. Lane spoke to U.S. engineers."
    chunks = chatterbox._split_text(
        sentence + " Next comes the final point.", max_words=12, max_chars=44,
    )
    assert chunks == [sentence, "Next comes the final point."]


def test_split_preserves_physical_paragraph_boundaries():
    lines = ["Opening sentence. Another opening sentence.", "The second paragraph.", "Closing words."]
    assert chatterbox._split_text("\n\n".join(lines), max_words=30, max_chars=250) == lines


def test_single_oversize_token_is_rejected_without_truncation():
    with pytest.raises(ValueError, match="token exceeds"):
        chatterbox._split_text("a" * 251, max_words=30, max_chars=250)


@pytest.mark.asyncio
async def test_lost_submission_response_reuses_persisted_idempotency_key(setup_client):
    server, _script, directory, _verifier, generate = setup_client
    server.lose_post_response = True
    output = await generate()
    assert output.is_file()
    assert len(server.jobs) == 1
    assert len(server.post_keys) == 2
    assert server.post_keys[0] == server.post_keys[1]
    checkpoint = json.loads((directory / "tts_input_part_001.request-0.json").read_text())
    assert checkpoint["idempotency_key"] == server.post_keys[0]
    assert checkpoint["job_id"] == "job-1"


@pytest.mark.asyncio
async def test_resume_follows_persisted_job_without_another_post(setup_client):
    server, _script, directory, _verifier, generate = setup_client
    server.fail_poll = True
    with pytest.raises(httpx.HTTPStatusError):
        await generate()
    checkpoint = json.loads((directory / "tts_input_part_001.request-0.json").read_text())
    assert checkpoint["job_id"] == "job-1"
    server.fail_poll = False
    assert (await generate()).is_file()
    assert len(server.post_keys) == 1


@pytest.mark.parametrize("invalid", [
    b'{"identity":', b'\xff', b'[]', b'null', b'{}',
    b'{"identity":"known", "idempotency_key":false}',
    b'{"identity":"known", "idempotency_key":"key", "job_id":[]}',
    b'{"identity":"known", "idempotency_key":"key", "terminal_failure":true}',
    b'{"identity":"known", "idempotency_key":"key", "job_id":"job-1", '
    b'"terminal_failure":{"job_id":"job-1", "status":"failed", "error":[]}}',
])
@pytest.mark.asyncio
async def test_invalid_checkpoint_is_preserved_without_another_post(setup_client, invalid):
    server, _script, directory, verifier, generate = setup_client
    server.fail_poll = True
    with pytest.raises(httpx.HTTPStatusError):
        await generate()
    request_path = directory / "tts_input_part_001.request-0.json"
    original = request_path.read_bytes()
    request_path.write_bytes(invalid)

    with pytest.raises(RuntimeError, match="request checkpoint .* preserving"):
        await generate()
    assert request_path.read_bytes() == invalid
    assert len(server.jobs) == len(server.post_keys) == 1
    assert list(directory.glob("*.request-*.json")) == [request_path]
    verifier.assert_not_awaited()

    # Once the original identity is restored, ordinary recovery still follows
    # the accepted remote job instead of spending another generation attempt.
    request_path.write_bytes(original)
    server.fail_poll = False
    assert (await generate()).is_file()
    assert len(server.jobs) == len(server.post_keys) == 1


@pytest.mark.asyncio
async def test_unreadable_checkpoint_is_preserved_without_another_post(setup_client, monkeypatch):
    server, _script, directory, verifier, generate = setup_client
    server.fail_poll = True
    with pytest.raises(httpx.HTTPStatusError):
        await generate()
    request_path = directory / "tts_input_part_001.request-0.json"
    original = request_path.read_bytes()
    read_text = Path.read_text

    def denied(path, *args, **kwargs):
        if path == request_path:
            raise PermissionError("checkpoint cannot be read")
        return read_text(path, *args, **kwargs)

    with monkeypatch.context() as blocked:
        blocked.setattr(Path, "read_text", denied)
        with pytest.raises(RuntimeError, match="request checkpoint .* unreadable"):
            await generate()
    assert request_path.read_bytes() == original
    assert len(server.jobs) == len(server.post_keys) == 1
    assert list(directory.glob("*.request-*.json")) == [request_path]
    verifier.assert_not_awaited()


@pytest.mark.parametrize("error", ["service_restarted", "decoder_duration_limit"])
@pytest.mark.asyncio
async def test_confirmed_failed_job_retries_and_resume_skips_its_persisted_failure(setup_client, error):
    server, _script, directory, _verifier, generate = setup_client
    server.job_changes_by_id["job-1"] = {"status": "failed", "error": error}
    server.poll_errors["job-2"] = 401
    with pytest.raises(httpx.HTTPStatusError):
        await generate()
    failed = json.loads((directory / "tts_input_part_001.request-0.json").read_text())
    assert failed["terminal_failure"] == {
        "job_id": "job-1", "status": "failed", "error": error,
    }
    pending = json.loads((directory / "tts_input_part_001.request-1.json").read_text())
    assert pending["job_id"] == "job-2"
    server.poll_errors.clear()
    output = await generate()
    assert output.is_file()
    assert len(server.jobs) == len(server.post_keys) == 2
    assert len(set(server.post_keys)) == 2
    assert [job["request"]["seed"] for job in server.jobs.values()] == [
        (config.TTS_RANDOM_SEED + attempt) % (2 ** 32) for attempt in range(2)
    ]
    metadata = json.loads(tts._part_metadata_path(output).read_text())
    assert metadata["synthesis_request"] == server.jobs["job-2"]["request"]
    assert metadata["synthesis_request"]["seed"] == (config.TTS_RANDOM_SEED + 1) % (2 ** 32)


@pytest.mark.parametrize("error", ["generation_timeout", "decoder_duration_limit"])
@pytest.mark.asyncio
async def test_service_failure_attempt_budget_survives_repeated_resume(setup_client, error):
    server, _script, directory, _verifier, generate = setup_client
    server.job_changes = {"status": "failed", "error": error}
    for _ in range(2):
        with pytest.raises(RuntimeError, match="exhausted 3 persisted generation attempts"):
            await generate()
    assert len(server.jobs) == len(server.post_keys) == 3
    assert len(set(server.post_keys)) == 3
    assert [job["request"]["seed"] for job in server.jobs.values()] == [
        (config.TTS_RANDOM_SEED + attempt) % (2 ** 32) for attempt in range(3)
    ]
    assert len(list(directory.glob("*.request-*.json"))) == 3
    assert all(json.loads(path.read_text())["terminal_failure"]["error"] == error
               for path in directory.glob("*.request-*.json"))


@pytest.mark.asyncio
async def test_adapter_recovers_actual_service_decoder_rejection_without_model_loading(tmp_path, monkeypatch):
    from deploy.chatterbox import service

    class Runner:
        def __init__(self):
            self.requests = []

        def run(self, request, output):
            self.requests.append(request)
            output.write_bytes(pcm_wav(seconds=39.0 if len(self.requests) == 1 else 3.0))

        def close(self):
            pass

    runner = Runner()
    app = service.create_app(service.Settings(
        api_key="test-secret", data_dir=tmp_path / "jobs",
    ), runner=runner)
    monkeypatch.setattr(config, "CHATTERBOX_TTS_API_KEY", "test-secret")
    monkeypatch.setattr(config, "CHATTERBOX_TTS_URL", "http://chatterbox.test")
    monkeypatch.setattr(config, "CHATTERBOX_TTS_POLL_SECONDS", 0.1)
    monkeypatch.setattr(tts, "_verify_orpheus_part", AsyncMock(return_value=advisory_report()))
    script = tmp_path / "script.txt"
    script.write_text("The complete source reaches the video.", encoding="utf-8")
    directory = tmp_path / "audio"
    client_class = httpx.AsyncClient
    async with app.router.lifespan_context(app):
        monkeypatch.setattr(config, "CHATTERBOX_TTS_DEPLOYMENT_FINGERPRINT", app.state.jobs.fingerprint)
        monkeypatch.setattr(chatterbox.httpx, "AsyncClient", lambda **kwargs: client_class(
            transport=httpx.ASGITransport(app=app), **kwargs,
        ))
        output = Path(await chatterbox._generate_chatterbox(
            str(script), str(directory), "default", "en",
            log=None, emit=lambda _message: None, verify_text=True,
        ))
    assert output.is_file()
    assert [request["seed"] for request in runner.requests] == [
        (config.TTS_RANDOM_SEED + attempt) % (2 ** 32) for attempt in range(2)
    ]
    failed = json.loads((directory / "tts_input_part_001.request-0.json").read_text())
    accepted = json.loads((directory / "tts_input_part_001.request-1.json").read_text())
    assert failed["terminal_failure"]["error"] == "decoder_duration_limit"
    assert failed["idempotency_key"] != accepted["idempotency_key"]
    assert not (tmp_path / "jobs" / f"{failed['job_id']}.wav").exists()
    assert composer._narration_manifest_failures(script, output, chatterbox.MODEL_ID) == []


@pytest.mark.asyncio
async def test_uncertain_poll_transport_failure_never_advances_attempt(setup_client):
    server, _script, directory, _verifier, generate = setup_client
    server.poll_transport_failure = True
    with pytest.raises(httpx.ReadError):
        await generate()
    checkpoint = json.loads((directory / "tts_input_part_001.request-0.json").read_text())
    assert checkpoint["job_id"] == "job-1"
    assert "terminal_failure" not in checkpoint
    assert len(list(directory.glob("*.request-*.json"))) == 1
    server.poll_transport_failure = False
    assert (await generate()).is_file()
    assert len(server.post_keys) == 1


@pytest.mark.parametrize("changes", [
    {"status": "failed", "error": "voice_reference_changed"},
    {"status": "failed", "error": "service_restarted", "fingerprint": "unexpected-build"},
    {"status": "failed", "error": "decoder_duration_limit", "fingerprint": "unexpected-build"},
    {"status": "unknown"},
])
@pytest.mark.asyncio
async def test_unrecoverable_or_unidentified_job_cannot_authorize_fresh_attempt(setup_client, changes):
    server, _script, directory, _verifier, generate = setup_client
    server.job_changes = changes
    with pytest.raises(RuntimeError):
        await generate()
    checkpoint = json.loads((directory / "tts_input_part_001.request-0.json").read_text())
    assert "terminal_failure" not in checkpoint
    assert len(server.post_keys) == 1


@pytest.mark.asyncio
async def test_model_load_failure_stops_without_retrying_or_reposting_on_resume(setup_client):
    server, _script, directory, verifier, generate = setup_client
    server.job_changes = {"status": "failed", "error": "model_load_failed"}
    with pytest.raises(RuntimeError, match=r"job-1 failed: model_load_failed"):
        await generate()
    request_path = directory / "tts_input_part_001.request-0.json"
    checkpoint = json.loads(request_path.read_text())
    assert checkpoint["job_id"] == "job-1"
    assert "terminal_failure" not in checkpoint
    assert "rejected" not in checkpoint
    assert len(server.post_keys) == len(server.jobs) == 1
    assert list(directory.glob("*.request-*.json")) == [request_path]
    assert not any(path.endswith("/audio") for _method, path in server.requests)
    assert not (directory / "tts_manifest.json").exists()
    verifier.assert_not_awaited()

    # Retrying the application follows the same terminal job. Missing models
    # cannot be repaired by spending another seed/inference attempt.
    with pytest.raises(RuntimeError, match=r"job-1 failed: model_load_failed"):
        await generate()
    assert json.loads(request_path.read_text()) == checkpoint
    assert len(server.post_keys) == len(server.jobs) == 1
    assert list(directory.glob("*.request-*.json")) == [request_path]
    assert not any(path.endswith("/audio") for _method, path in server.requests)
    verifier.assert_not_awaited()


@pytest.mark.parametrize("field,value", [
    ("source_revision", "different-code"),
    ("model_revision", "different-weights"),
    ("model_variant", "v3"),
    ("fingerprint", "different-deployment"),
])
@pytest.mark.asyncio
async def test_changed_capabilities_refuse_before_submission(setup_client, field, value):
    server, _script, _directory, _verifier, generate = setup_client
    server.capabilities[field] = value
    with pytest.raises(RuntimeError):
        await generate()
    assert not server.post_keys


@pytest.mark.asyncio
async def test_unpinned_deployment_refuses_before_submission(setup_client, monkeypatch):
    server, _script, _directory, _verifier, generate = setup_client
    monkeypatch.setattr(config, "CHATTERBOX_TTS_DEPLOYMENT_FINGERPRINT", "", raising=False)
    with pytest.raises(RuntimeError):
        await generate()
    assert not server.post_keys


@pytest.mark.parametrize("field,value", [
    ("source_revision", "different-code"),
    ("model_revision", "different-weights"),
    ("model_variant", "v3"),
    ("fingerprint", "different-deployment"),
    ("voice_fingerprint", "different-voice"),
    ("text_sha256", "different-source"),
    ("id", "a-different-job"),
])
@pytest.mark.asyncio
async def test_changed_completed_job_identity_refuses_audio(setup_client, field, value):
    server, _script, directory, _verifier, generate = setup_client
    server.job_changes[field] = value
    with pytest.raises(RuntimeError):
        await generate()
    assert not any(path.endswith("/audio") for _method, path in server.requests)
    assert not (directory / "tts_manifest.json").exists()


@pytest.mark.asyncio
async def test_completed_job_with_changed_generation_request_is_refused(setup_client):
    server, script, directory, _verifier, generate = setup_client
    server.job_changes["request"] = {
        "text": script.read_text(), "language_id": "en", "voice": "jane",
        "seed": config.TTS_RANDOM_SEED,
        **chatterbox.SYNTHESIS_PARAMETERS, "temperature": 1.2,
    }
    with pytest.raises(RuntimeError, match="generation parameters"):
        await generate()
    assert not any(path.endswith("/audio") for _method, path in server.requests)
    assert not (directory / "tts_manifest.json").exists()


@pytest.mark.asyncio
async def test_download_hash_mismatch_is_never_promoted(setup_client):
    server, _script, directory, verifier, generate = setup_client
    server.corrupt_download = True
    with pytest.raises(RuntimeError, match="hash"):
        await generate()
    assert not (directory / "tts_input_part_001_generated.wav").exists()
    assert not (directory / "tts_manifest.json").exists()
    verifier.assert_not_awaited()


@pytest.mark.asyncio
async def test_truncated_wav_exhausts_physical_retries_without_becoming_final(setup_client):
    server, _script, directory, verifier, generate = setup_client
    server.truncate_audio = True
    with pytest.raises(tts.TtsIntegrityError, match="truncated"):
        await generate()
    assert len(server.jobs) == chatterbox.MAX_INTEGRITY_ATTEMPTS
    assert len(set(server.post_keys)) == chatterbox.MAX_INTEGRITY_ATTEMPTS
    assert not (directory / "tts_input_part_001_generated.wav").exists()
    assert not (directory / "tts_manifest.json").exists()
    verifier.assert_not_awaited()


@pytest.mark.parametrize("seconds,sample_rate,channels", [(39.0, 24_000, 1), (3.0, 22_050, 1), (3.0, 24_000, 2)])
def test_invalid_audio_format_or_decoder_ceiling_refuses(tmp_path, seconds, sample_rate, channels):
    path = tmp_path / "invalid.wav"
    path.write_bytes(pcm_wav(seconds=seconds, sample_rate=sample_rate, channels=channels))
    with pytest.raises(tts.TtsIntegrityError):
        chatterbox._validate_audio(path, "A complete short sentence.")


@pytest.mark.asyncio
async def test_asr_advisory_mismatch_is_accepted_without_resynthesis(setup_client):
    server, script, directory, verifier, generate = setup_client
    output = await generate()
    manifest = json.loads((directory / "tts_manifest.json").read_text())
    assert len(server.jobs) == 1
    assert manifest["integrity"]["passed"] is True
    assert manifest["integrity"]["policy"] == tts.ASR_ADVISORY_POLICY
    assert manifest["integrity"]["accepted_source_coverage"] == 1.0
    assert manifest["integrity"]["verified_source_coverage"] == 0.0
    assert manifest["integrity"]["asr_warning_parts"] == 1
    assert verifier.await_args.kwargs["asr_advisory"] is True
    assert composer._narration_manifest_failures(script, output, chatterbox.MODEL_ID) == []


@pytest.mark.asyncio
async def test_voice_identity_change_invalidates_audio_and_pending_request_cache(setup_client, monkeypatch):
    server, _script, _directory, verifier, generate = setup_client
    await generate()
    await generate()
    assert len(server.jobs) == 1
    assert verifier.await_count == 1
    server.capabilities["voices"][0]["fingerprint"] = "voice-b"
    server.capabilities["fingerprint"] = "deployment-b"
    monkeypatch.setattr(config, "CHATTERBOX_TTS_DEPLOYMENT_FINGERPRINT", "deployment-b", raising=False)
    await generate()
    assert len(server.jobs) == 2
    assert len(set(server.post_keys)) == 2
    assert verifier.await_count == 2


@pytest.mark.asyncio
async def test_generation_parameter_change_invalidates_audio_and_request_cache(setup_client, monkeypatch):
    server, _script, _directory, _verifier, generate = setup_client
    await generate()
    monkeypatch.setattr(chatterbox, "SYNTHESIS_PARAMETERS", {
        **chatterbox.SYNTHESIS_PARAMETERS, "temperature": 0.7,
    })
    output = await generate()
    assert len(server.jobs) == len(set(server.post_keys)) == 2
    metadata = json.loads(tts._part_metadata_path(output).read_text())
    assert metadata["synthesis_request"] == server.jobs["job-2"]["request"]
    assert metadata["synthesis_request"]["temperature"] == 0.7


@pytest.mark.asyncio
async def test_tampered_cached_wav_is_redownloaded_from_owned_job(setup_client):
    server, _script, _directory, verifier, generate = setup_client
    output = await generate()
    original = output.read_bytes()
    output.write_bytes(original[:-50])
    recovered = await generate()
    assert recovered.read_bytes() == original
    assert len(server.jobs) == 1
    assert len(server.post_keys) == 1
    assert verifier.await_count == 2


@pytest.mark.asyncio
async def test_manifest_drives_composer_line_timing_and_detects_source_or_audio_change(setup_client, monkeypatch):
    _server, script, directory, _verifier, generate = setup_client
    monkeypatch.setattr(config, "CHATTERBOX_TTS_CHUNK_WORDS", 6)
    lines = [
        "Opening sentence ends here. Another sentence follows here.",
        "The second paragraph stays separate.",
    ]
    script.write_text("\n".join(lines), encoding="utf-8")
    output = await generate()
    manifest = json.loads((directory / "tts_manifest.json").read_text())
    assert manifest["chunk_count"] == 3
    assert manifest["pacing_policy"] == tts.NARRATION_PACING_POLICY
    assert manifest["synthesis_speed_ratio"] == 1.0
    assert manifest["output_audio_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert composer._narration_manifest_failures(script, output, chatterbox.MODEL_ID) == []
    segments = composer._manifest_program_segments(output, [{"text": line} for line in lines])
    assert segments is not None
    assert [row["text"] for row in segments] == lines
    assert [row["start"] for row in segments] == [0.0, 6.0]
    assert [row["duration"] for row in segments] == [6.0, 3.0]
    assert all(row["timing_source"] == "validated_chunk_manifest" for row in segments)
    script.write_text("An altered source.", encoding="utf-8")
    assert any("script changed" in issue for issue in composer._narration_manifest_failures(script, output, chatterbox.MODEL_ID))
    script.write_text("\n".join(lines), encoding="utf-8")
    output.write_bytes(output.read_bytes() + b"changed")
    assert any("WAV changed" in issue for issue in composer._narration_manifest_failures(script, output, chatterbox.MODEL_ID))

"""Narration boundary tests; no network, ASR model, or TTS model is loaded."""

import hashlib
import io
import json
import struct
import wave
from dataclasses import asdict
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest

from backend import config
from backend.pipeline import av_sync, chatterbox, composer, digester, tts, tts_pronunciation


@pytest.mark.asyncio
@pytest.mark.parametrize("voices, expected", [(None, "space-en-f1"), (["jane"], "jane"), (["default"], "default")])
async def test_dispatcher_uses_official_default_but_preserves_explicit_chatterbox_voice(tmp_path, monkeypatch, voices, expected):
    generate = AsyncMock(return_value=str(tmp_path / "audio.wav"))
    monkeypatch.setattr(chatterbox, "_generate_chatterbox", generate)
    monkeypatch.setattr(config, "TTS_DEFAULT_VOICE_1", "jane")
    pocket = AsyncMock(side_effect=AssertionError("Chatterbox must not route through Pocket"))
    monkeypatch.setattr(tts, "_generate_pocket_tts", pocket)
    script = tmp_path / "script.txt"
    script.write_text("A short voice selection test.")
    result = await tts.generate_tts(script, tmp_path, voices=voices, tts_model="chatterbox-multilingual")
    assert result == str(tmp_path / "audio.wav")
    assert generate.await_args.args[2:] == (expected, "en")
    assert generate.await_args.kwargs["verify_text"] is True
    pocket.assert_not_awaited()


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


def spelling_decisions(source):
    """A deterministic test double for contextual model output, never production rules."""
    return [
        {
            "id": candidate["id"], "source": candidate["source"],
            "action": "keep" if candidate["source"] == "NASA" else "spell",
            "spoken": candidate["source"] if candidate["source"] == "NASA" else "-".join(
                candidate["source"].replace(".", "")
            ),
            "reason": "Fixture pronunciation decision in this sentence.",
        }
        for candidate in tts_pronunciation.candidates(source)
    ]


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
        self.audio_seconds = None
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
                audio = pcm_wav(seconds=(
                    self.audio_seconds if self.audio_seconds is not None
                    else max(3.0, len(payload["text"].replace("-", " ").split()) * 0.4)
                ))
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
    server.pronunciation_route = AsyncMock(return_value=("test-endpoint", "test-model", "test-key"))
    server.pronunciation_ai = AsyncMock(side_effect=lambda _prompt, payload, **_kwargs: json.dumps({
        "decisions": spelling_decisions(json.loads(payload)["source_text"]),
    }))
    monkeypatch.setattr(digester, "_resolve_provider", server.pronunciation_route)
    monkeypatch.setattr(digester, "_chat", server.pronunciation_ai)
    client_class = httpx.AsyncClient
    monkeypatch.setattr(chatterbox.httpx, "AsyncClient", lambda **kwargs: client_class(
        transport=httpx.MockTransport(server.handle), **kwargs,
    ))
    monkeypatch.setattr(chatterbox.asyncio, "sleep", AsyncMock())
    verifier = AsyncMock(side_effect=lambda *args, **kwargs: advisory_report())
    monkeypatch.setattr(tts, "_verify_orpheus_part", verifier)
    def transcript(_path, verification_dir, **_kwargs):
        source = (verification_dir.parents[1] / f"{verification_dir.name}.txt").read_text()
        return ([{"text": word} for word in source.split()], {"passed": True})
    monkeypatch.setattr(av_sync, "ensure_word_transcript", AsyncMock(side_effect=transcript))
    script = tmp_path / "script.txt"
    script.write_text("The complete narration remains available for the final video.", encoding="utf-8")
    directory = tmp_path / "audio"

    async def generate():
        return Path(await chatterbox._generate_chatterbox(
            str(script), str(directory), "jane", "en",
            log=None, emit=lambda _message: None, verify_text=True,
        ))

    return server, script, directory, verifier, generate


@pytest.mark.asyncio
async def test_generation_freezes_contract_before_first_network_await(setup_client, monkeypatch):
    server, _script, directory, _verifier, generate = setup_client
    initial = chatterbox.cache_contract()
    handle = server.handle
    def change_settings_after_request(request):
        response = handle(request)
        if request.url.path == "/v1/capabilities":
            monkeypatch.setattr(config, "TTS_RANDOM_SEED", initial["seed"] + 1)
            monkeypatch.setattr(config, "CHATTERBOX_TTS_CHUNK_WORDS", 1)
            monkeypatch.setattr(config, "CHATTERBOX_TTS_DEPLOYMENT_FINGERPRINT", "changed-deployment")
        return response
    server.handle = change_settings_after_request
    output = await generate()
    assert output.is_file()
    assert len(server.jobs) == 1
    assert server.jobs["job-1"]["request"]["seed"] == initial["seed"]
    manifest = json.loads((directory / "tts_manifest.json").read_text())
    assert manifest["cache_contract"] == initial
    assert manifest["deployment_fingerprint"] == initial["deployment_fingerprint"]
    metadata = json.loads((directory / "tts_input_part_001_generated.json").read_text())
    assert metadata["identity"]["seed"] == initial["seed"]
    assert metadata["identity"]["deployment_fingerprint"] == initial["deployment_fingerprint"]


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


@pytest.mark.parametrize("max_words,max_chars", [(8, 250), (30, 24), (8, 24)])
def test_initialism_expansion_is_budgeted_without_changing_or_splitting_source_names(max_words, max_chars):
    source = "CNBC TSMC report AI chip demand.\nCNBC explains the AI market."
    _decisions, edits = tts_pronunciation.validate_decisions(source, spelling_decisions(source))
    chunks = chatterbox._split_text(source, max_words=max_words, max_chars=max_chars, pronunciation_edits=edits)
    assert " ".join(chunks) == " ".join(source.split())
    assert len(chunks) > 2
    cursor = 0
    for chunk in chunks:
        start = source.index(chunk, cursor)
        cursor = start + len(chunk)
        spoken = tts_pronunciation.render_span(source, start, cursor, edits)
        assert len(spoken) <= max_chars
        assert len(spoken.replace("-", " ").split()) <= max_words
        assert all(token not in {"C", "N", "B", "T", "S", "M"} for token in chunk.split())


@pytest.mark.parametrize("source, max_words, max_chars", [
    ("CNBC", 3, 250),
    ("ABCDEFGHIJK", 30, 20),
])
def test_single_initialism_that_exceeds_expanded_budget_is_rejected(source, max_words, max_chars):
    _decisions, edits = tts_pronunciation.validate_decisions(source, spelling_decisions(source))
    with pytest.raises(ValueError, match="token exceeds"):
        chatterbox._split_text(source, max_words=max_words, max_chars=max_chars, pronunciation_edits=edits)


@pytest.mark.asyncio
async def test_spelling_hints_only_change_provider_request_not_canonical_narration(setup_client):
    server, script, directory, verifier, generate = setup_client
    source = "CNBC TSMC report on AI chips while NASA studies space."
    script.write_text(source, encoding="utf-8")
    output = await generate()
    expected_spoken = "C-N-B-C T-S-M-C report on A-I chips while NASA studies space."
    server.pronunciation_ai.assert_awaited_once()
    assert json.loads(server.pronunciation_ai.await_args.args[1])["source_text"] == source
    assert len(server.jobs) == 1
    assert server.jobs["job-1"]["request"]["text"] == expected_spoken
    assert server.jobs["job-1"]["text_sha256"] == hashlib.sha256(expected_spoken.encode()).hexdigest()
    assert script.read_text() == (directory / "tts_input.txt").read_text() == source
    assert (directory / "tts_input_part_001.txt").read_text() == source
    assert (directory / "synthesis" / "tts_input_part_001.txt").read_text() == expected_spoken
    assert list(directory.glob("tts_input_part_*.txt")) == [directory / "tts_input_part_001.txt"]
    assert verifier.await_args.args[1] == source
    metadata = json.loads((directory / "tts_input_part_001_generated.json").read_text())
    assert metadata["identity"]["text"] == source
    assert metadata["identity"]["synthesis_text"] == metadata["synthesis_request"]["text"] == expected_spoken
    plan = json.loads((directory / "tts_pronunciation.json").read_text())
    assert metadata["identity"]["pronunciation_policy"] == plan["identity"]["policy"]
    assert metadata["identity"]["pronunciation_plan_sha256"] == plan["plan_sha256"]
    manifest = json.loads((directory / "tts_manifest.json").read_text())
    assert manifest["source_text_sha256"] == manifest["parts"][0]["text_sha256"] == hashlib.sha256(source.encode()).hexdigest()
    assert manifest["source_word_count"] == manifest["parts"][0]["word_count"] == len(source.split())
    assert composer._narration_manifest_failures(script, output, chatterbox.MODEL_ID) == []
    assert composer._manifest_program_segments(output, [{"text": source}])[0]["text"] == source


@pytest.mark.asyncio
async def test_same_spelling_can_receive_different_contextual_readings_in_one_script(setup_client):
    server, script, directory, verifier, generate = setup_client
    source = "CAD software supports this design. CAD is the currency code here."
    script.write_text(source, encoding="utf-8")
    server.pronunciation_ai.side_effect = None
    server.pronunciation_ai.return_value = json.dumps({"decisions": [
        {"id": 0, "source": "CAD", "action": "keep", "spoken": "CAD",
         "reason": "Computer aided design is conventionally read as a word here."},
        {"id": 1, "source": "CAD", "action": "spell", "spoken": "C-A-D",
         "reason": "This occurrence is an explicitly spoken currency code."},
    ]})
    await generate()
    assert server.jobs["job-1"]["request"]["text"] == (
        "CAD software supports this design. C-A-D is the currency code here."
    )
    assert (directory / "tts_input_part_001.txt").read_text() == source
    assert verifier.await_args.args[1] == source
    server.pronunciation_ai.assert_awaited_once()


@pytest.mark.asyncio
async def test_service_must_hash_spoken_request_instead_of_canonical_script(setup_client):
    server, script, directory, verifier, generate = setup_client
    source = "CNBC reported the complete story today."
    script.write_text(source, encoding="utf-8")
    server.job_changes["text_sha256"] = hashlib.sha256(source.encode()).hexdigest()
    with pytest.raises(RuntimeError, match="complete requested text"):
        await generate()
    assert not any(path.endswith("/audio") for _method, path in server.requests)
    assert not (directory / "tts_manifest.json").exists()
    verifier.assert_not_awaited()


@pytest.mark.asyncio
async def test_spelled_initialism_resume_reuses_original_request_identity(setup_client):
    server, script, directory, _verifier, generate = setup_client
    script.write_text("CNBC reported the complete story today.", encoding="utf-8")
    server.lose_post_response = True
    server.fail_poll = True
    with pytest.raises(httpx.HTTPStatusError):
        await generate()
    checkpoint_path = directory / "tts_input_part_001.request-0.json"
    checkpoint = checkpoint_path.read_bytes()
    plan_path = directory / "tts_pronunciation.json"
    plan = plan_path.read_bytes()
    server.pronunciation_ai.assert_awaited_once()
    server.pronunciation_ai.side_effect = AssertionError("Resume must reuse its frozen pronunciation plan")
    server.pronunciation_route.side_effect = AssertionError("Resume must not resolve a new model")
    server.fail_poll = False
    assert (await generate()).is_file()
    assert checkpoint_path.read_bytes() == checkpoint
    assert plan_path.read_bytes() == plan
    server.pronunciation_ai.assert_awaited_once()
    server.pronunciation_route.assert_awaited_once()
    assert len(server.jobs) == 1
    assert len(server.post_keys) == 2
    assert len(set(server.post_keys)) == 1


@pytest.mark.asyncio
async def test_initialism_heavy_audio_must_cover_spoken_letters_before_asr(setup_client):
    server, script, directory, verifier, generate = setup_client
    # Six source words expand to twenty spoken letter names. Canonical word
    # counting alone would bypass the eight-word duration validation threshold.
    script.write_text("CNBC TSMC BBC API GPU GPT.", encoding="utf-8")
    server.audio_seconds = 1.0
    with pytest.raises(tts.TtsIntegrityError, match="too short.*20 words"):
        await generate()
    assert len(server.jobs) == len(set(server.post_keys)) == chatterbox.MAX_INTEGRITY_ATTEMPTS
    assert not (directory / "tts_input_part_001_generated.wav").exists()
    assert not (directory / "tts_manifest.json").exists()
    assert len(list((directory / "verification" / "tts_input_part_001").glob("rejected-*.wav"))) == chatterbox.MAX_INTEGRITY_ATTEMPTS
    verifier.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("source_name,spoken,letters", [
    ("CNBC", "c-n-b-c", 4),
    ("CNBC", "C--N--B--C", 4),
    ("CNBC", "C.-N.-B.-C", 4),
    ("API", "a.p.i", 3),
    ("API", "eigh-pee-eye", 3),
])
async def test_model_hint_separators_count_for_chunk_budget_and_audio_duration(setup_client, source_name, spoken, letters):
    server, script, directory, verifier, generate = setup_client
    source = " ".join([source_name] * 3)
    script.write_text(source, encoding="utf-8")
    decisions = spelling_decisions(source)
    for decision in decisions:
        decision["spoken"] = spoken
    _validated, edits = tts_pronunciation.validate_decisions(source, decisions)
    assert chatterbox._split_text(
        source, max_words=letters, max_chars=250, pronunciation_edits=edits,
    ) == [source_name] * 3

    server.pronunciation_ai.side_effect = None
    server.pronunciation_ai.return_value = json.dumps({"decisions": decisions})
    server.audio_seconds = 1.0
    with pytest.raises(tts.TtsIntegrityError, match=rf"too short.*{letters * 3} words"):
        await generate()
    assert len(server.jobs) == chatterbox.MAX_INTEGRITY_ATTEMPTS
    assert all(job["request"]["text"] == " ".join([spoken] * 3) for job in server.jobs.values())
    assert not (directory / "tts_input_part_001_generated.wav").exists()
    assert not (directory / "tts_manifest.json").exists()
    verifier.assert_not_awaited()


def test_decimal_periods_preserve_word_budget_and_audio_duration_count(tmp_path):
    source = "Revenue grew 2.5 percent in this quarter."
    assert chatterbox._split_text(source, max_words=7, max_chars=250) == [source]
    path = tmp_path / "decimal.wav"
    path.write_bytes(pcm_wav(seconds=1.0))
    # Seven words remain below the eight-word duration guard. Splitting the
    # decimal point would incorrectly turn this into eight spoken words.
    assert chatterbox._validate_audio(path, source).duration_seconds == 1.0


@pytest.mark.asyncio
async def test_cached_initialism_audio_rechecks_duration_even_when_hash_and_sidecar_match(setup_client):
    server, script, _directory, verifier, generate = setup_client
    script.write_text("CNBC TSMC BBC API GPU GPT.", encoding="utf-8")
    output = await generate()
    valid_audio = output.read_bytes()
    # Mimic a self-consistent old cache accepted with canonical word counting.
    # Matching metadata and WAV hashes must not bypass the expanded-word guard.
    output.write_bytes(pcm_wav(seconds=1.0))
    metadata_path = tts._part_metadata_path(output)
    metadata = json.loads(metadata_path.read_text())
    metadata["audio_sha256"] = tts._file_sha256(output)
    metadata["wav"] = asdict(tts._read_pcm_wav(output))
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    recovered = await generate()
    assert recovered.read_bytes() == valid_audio
    assert len(server.jobs) == len(server.post_keys) == 1
    assert verifier.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["version", "prompt"])
async def test_pronunciation_policy_invalidates_preview_cache_and_preserves_submitted_jobs(setup_client, monkeypatch, tmp_path, change):
    server, script, directory, verifier, generate = setup_client
    script.write_text("CNBC reported the complete story today.", encoding="utf-8")
    await generate()
    await generate()
    original_preview = config.voice_preview_cache_dir(chatterbox.MODEL_ID)
    original_contract = chatterbox.cache_contract()
    plan_path = directory / "tts_pronunciation.json"
    checkpoint_path = directory / "tts_input_part_001.request-0.json"
    plan, checkpoint = plan_path.read_bytes(), checkpoint_path.read_bytes()
    if change == "version":
        monkeypatch.setattr(tts_pronunciation, "POLICY_VERSION", tts_pronunciation.POLICY_VERSION + 1)
    else:
        prompts = tmp_path / "changed-prompts"
        prompts.mkdir()
        prompt = (config.PROMPTS_DIR / "tts_pronunciation.txt").read_text()
        (prompts / "tts_pronunciation.txt").write_text(prompt + "\nTest policy change.\n")
        monkeypatch.setattr(config, "PROMPTS_DIR", prompts)
    assert config.voice_preview_cache_dir(chatterbox.MODEL_ID) != original_preview
    assert chatterbox.cache_contract() != original_contract
    with pytest.raises(RuntimeError, match="submitted jobs"):
        await generate()
    assert len(server.jobs) == len(set(server.post_keys)) == verifier.await_count == 1
    assert plan_path.read_bytes() == plan
    assert checkpoint_path.read_bytes() == checkpoint
    server.pronunciation_ai.assert_awaited_once()
    manifest = json.loads((directory / "tts_manifest.json").read_text())
    assert manifest["cache_contract"] == original_contract
    metadata = json.loads((directory / "tts_input_part_001_generated.json").read_text())
    assert metadata["identity"]["pronunciation_policy"] == original_contract["pronunciation_policy"]


@pytest.mark.asyncio
async def test_generation_freezes_pronunciation_policy_before_network_await(setup_client, monkeypatch):
    server, script, directory, _verifier, generate = setup_client
    script.write_text("CNBC reported the complete story today.", encoding="utf-8")
    initial_contract = chatterbox.cache_contract()
    handle = server.handle

    def change_pronunciation_after_request(request):
        response = handle(request)
        if request.url.path == "/v1/capabilities":
            monkeypatch.setattr(tts_pronunciation, "POLICY_VERSION", initial_contract["pronunciation_policy"]["version"] + 1)
        return response

    server.handle = change_pronunciation_after_request
    await generate()
    assert server.jobs["job-1"]["request"]["text"] == "C-N-B-C reported the complete story today."
    manifest = json.loads((directory / "tts_manifest.json").read_text())
    assert manifest["cache_contract"] == initial_contract
    plan = json.loads((directory / "tts_pronunciation.json").read_text())
    assert plan["identity"]["policy"] == initial_contract["pronunciation_policy"]


@pytest.mark.asyncio
async def test_invalid_ai_pronunciation_fails_before_any_synthesis_submission(setup_client):
    server, script, directory, verifier, generate = setup_client
    source = "CNBC reported the complete story today."
    script.write_text(source, encoding="utf-8")
    server.pronunciation_ai.side_effect = None
    server.pronunciation_ai.return_value = json.dumps({"decisions": [{
        "id": 0, "source": "CNBC", "action": "spell", "spoken": "C-B-C",
        "reason": "Invalid fixture drops the source letter N.",
    }]})
    with pytest.raises(RuntimeError, match="AI plan failed validation"):
        await generate()
    assert server.pronunciation_ai.await_count == tts_pronunciation.MAX_PLAN_ATTEMPTS
    assert not server.post_keys
    assert not server.jobs
    assert not (directory / "tts_pronunciation.json").exists()
    assert not (directory / "tts_manifest.json").exists()
    assert script.read_text() == source
    verifier.assert_not_awaited()


@pytest.mark.asyncio
async def test_script_without_pronunciation_candidates_skips_model_request(setup_client):
    server, script, directory, _verifier, generate = setup_client
    assert (await generate()).is_file()
    server.pronunciation_route.assert_not_awaited()
    server.pronunciation_ai.assert_not_awaited()
    plan = json.loads((directory / "tts_pronunciation.json").read_text())
    assert plan["decisions"] == plan["edits"] == []
    assert plan["synthesis_text"] == script.read_text()


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
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "deploy" / "chatterbox"))
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
        api_key="test-secret", data_dir=tmp_path / "jobs", default_voice="default",
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


@pytest.mark.parametrize("observed", [
    "Open AI reports on A I chips.", "OpenAI reports on AI chips.",
])
def test_pronunciation_guard_accepts_only_identical_aligned_splits(observed):
    report = chatterbox._pronunciation_report(
        "OpenAI reports on AI chips.", [{"text": word} for word in observed.split()],
    )
    assert report["passed"] is True
    assert report["targets"] == ["ai", "ai"]


@pytest.mark.parametrize("observed", [
    "Open II reports on AI chips.", "Open eye reports on AI chips.",
    "Open AI AI reports on AI chips.", "OpenAI AI reports on AI chips.",
    "OpenAI reports on AI AI chips.", "OpenAI reports on eye chips.",
    "OpenAI reports on chips.", "reports on AI chips.", "",
])
def test_pronunciation_guard_rejects_wrong_missing_and_repeated_letters(observed):
    assert chatterbox._pronunciation_report(
        "OpenAI reports on AI chips.", [{"text": word} for word in observed.split()],
    )["passed"] is False


def test_pronunciation_guard_checks_every_repeated_source_occurrence():
    assert chatterbox._pronunciation_report(
        "AI and AI chips.", [{"text": word} for word in "AI and chips.".split()],
    )["passed"] is False


@pytest.mark.parametrize("extra", ["AI", "A I"])
@pytest.mark.parametrize("replacement", [False, True])
def test_pronunciation_guard_detects_a_repeated_initialism_away_from_its_source(extra, replacement):
    source = "OpenAI reports the latest news today."
    observed = f"OpenAI reports the latest {extra} {'news ' if not replacement else ''}today."
    assert chatterbox._pronunciation_report(source, [{"text": word} for word in observed.split()])["passed"] is False


@pytest.mark.parametrize("observed,passed", [
    ("From materials science.", True),
    ("From materials science. GPM POSPIR to and SES and.", False),
    ("From materials science. Thank you everyone.", False),
    ("From materials science. Thank you.", True),
    ("From material science.", True),
    ("From materials GPM POSPIR to science.", True),
])
def test_trailing_speech_guard_rejects_extra_phrases_with_complete_source(observed, passed):
    assert chatterbox._trailing_speech_report(
        "From materials science.", [{"text": word} for word in observed.split()],
    )["passed"] is passed


@pytest.mark.asyncio
@pytest.mark.parametrize("defect", ["missing", "null", "stale", "failed", "missing_passed"])
async def test_tail_gate_upgrade_reuses_completed_good_audio_without_resynthesis(setup_client, defect):
    server, _script, directory, verifier, generate = setup_client
    await generate()
    path = directory / "tts_input_part_001_generated.json"
    metadata = json.loads(path.read_text())
    guard = metadata["integrity"]["pronunciation"]
    if defect == "missing":
        guard.pop("trailing_speech")
    else:
        guard["trailing_speech"] = {
            "null": None, "stale": {"version": 0, "passed": True},
            "failed": {"version": 1, "passed": False}, "missing_passed": {"version": 1},
        }[defect]
    path.write_text(json.dumps(metadata))
    await generate()
    assert len(server.jobs) == 1
    assert verifier.await_count == (2 if defect in {"failed", "missing_passed"} else 1)
    assert json.loads(path.read_text())["integrity"]["pronunciation"]["trailing_speech"]["passed"]


@pytest.mark.asyncio
async def test_tail_gate_rejects_legacy_extra_speech_and_preserves_bounded_retry(setup_client, monkeypatch):
    server, script, directory, verifier, generate = setup_client
    script.write_text("From materials science.")
    await generate()
    path = directory / "tts_input_part_001_generated.json"
    metadata = json.loads(path.read_text())
    bad = tts._orpheus_transcript_report(script.read_text(), [{"text": word} for word in
        "From materials science. GPM POSPIR to and SES and.".split()])
    metadata["integrity"].update(bad)
    guard = metadata["integrity"]["pronunciation"]
    guard.pop("trailing_speech")
    guard["transcript"] = "From materials science. GPM POSPIR to and SES and."
    path.write_text(json.dumps(metadata))
    verifier.side_effect = [{**advisory_report(), **bad}, advisory_report()]
    monkeypatch.setattr(av_sync, "ensure_word_transcript", AsyncMock(side_effect=[
        ([{"text": word} for word in guard["transcript"].split()], {"passed": True}),
        ([{"text": word} for word in script.read_text().split()], {"passed": True}),
    ]))
    await generate()
    assert len(server.jobs) == 2
    assert [job["request"]["seed"] for job in server.jobs.values()] == [42, 43]
    rejected = directory / "verification/tts_input_part_001/rejected-0.wav"
    assert rejected.exists()
    assert "unrequested trailing speech" in json.loads(
        (directory / "tts_input_part_001.request-0.json").read_text()
    )["rejected"]
    await generate()
    assert len(server.jobs) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("defect", ["missing", "null", "failed"])
async def test_missing_cached_pronunciation_acceptance_rechecks_the_same_job(setup_client, defect):
    server, _script, directory, verifier, generate = setup_client
    await generate()
    path = directory / "tts_input_part_001_generated.json"
    metadata = json.loads(path.read_text())
    if defect == "missing":
        metadata["integrity"].pop("pronunciation")
    else:
        metadata["integrity"]["pronunciation"] = None if defect == "null" else {"passed": False}
    path.write_text(json.dumps(metadata))
    await generate()
    assert len(server.jobs) == 1
    assert verifier.await_count == 2
    assert json.loads(path.read_text())["integrity"]["pronunciation"]["passed"] is True


def test_pronunciation_guard_leaves_other_spelling_differences_advisory():
    report = chatterbox._pronunciation_report(
        "Techmeme reports that OpenAI made tools.",
        [{"text": word} for word in "Tech meme reports that Open AI made tools.".split()],
    )
    assert report["passed"] is True


def test_pronunciation_guard_preserves_suffixes_models_dotted_initialisms_and_urls():
    source = "OpenAI's GPUs use GPT4 in the U.S. See https://API.example/AI."
    observed = "Open AI's G P Us use G P T 4 in the U S. See https://API.example/AI."
    report = chatterbox._pronunciation_report(source, [{"text": word} for word in observed.split()])
    assert report["passed"] is True
    assert report["targets"] == ["ai's", "gpus", "gpt", "us"]


@pytest.mark.parametrize("observed", [
    "Qubit AI reports that Chat GPT is available.",
    "QubitAI reports that ChatGPT is available.",
    "Qubit A I reports that Chat G P T is available.",
])
def test_pronunciation_guard_checks_suffix_without_rejecting_advisory_brand_spelling(observed):
    source = "QbitAI reports that ChatGPT is available."
    assert chatterbox._pronunciation_report(source, [{"text": word} for word in observed.split()])["passed"]


@pytest.mark.parametrize("observed", [
    "QubitII reports on AI.", "QubitAIAI reports on AI.",
    "QubitaiAI reports on AI.", "AI A I reports on AI.",
    "Qubit A I I reports on AI.", "QubitAI AI reports on AI.",
])
def test_pronunciation_guard_brand_alias_never_hides_wrong_or_extra_letters(observed):
    assert chatterbox._pronunciation_report(
        "QbitAI reports on AI.", [{"text": word} for word in observed.split()],
    )["passed"] is False


@pytest.mark.parametrize("observed", [
    "By ObserveDU S consumer spending.", "By observedUS consumer spending.",
])
def test_pronunciation_guard_restores_exact_source_word_letter_boundary(observed):
    assert chatterbox._pronunciation_report(
        "By observed US consumer spending.", [{"text": word} for word in observed.split()],
    )["passed"] is True


@pytest.mark.parametrize("observed", [
    "By observedU S US consumer spending.", "By observedU S U S consumer spending.",
    "By observedI S consumer spending.", "By observedU consumer spending.",
    "By observedU S S consumer spending.",
])
def test_pronunciation_guard_preserves_missing_wrong_and_repeated_attached_letters(observed):
    assert chatterbox._pronunciation_report(
        "By observed US consumer spending.", [{"text": word} for word in observed.split()],
    )["passed"] is False


def test_pronunciation_guard_keeps_model_selected_word_acronyms_advisory():
    source = "CoRL reports on AI."
    _, edits = tts_pronunciation.validate_decisions(source, [
        {"id": 0, "source": "RL", "action": "keep", "spoken": "RL", "reason": "Word acronym."},
        {"id": 1, "source": "AI", "action": "spell", "spoken": "eigh eye", "reason": "Letter reading."},
    ])
    report = chatterbox._pronunciation_report(source, [{"text": word} for word in "Coral reports on AI.".split()], edits)
    assert report["passed"] is True
    assert report["targets"] == ["ai"]


@pytest.mark.asyncio
async def test_pronunciation_mismatch_retries_and_reuses_the_accepted_audio(setup_client, monkeypatch):
    server, script, directory, _verifier, generate = setup_client
    source = "OpenAI reports on AI chips."
    script.write_text(source)
    transcribe = AsyncMock(side_effect=[
        ([{"text": word} for word in "Open AI AI reports on AI chips.".split()], {}),
        ([{"text": word} for word in source.split()], {}),
    ])
    monkeypatch.setattr(av_sync, "ensure_word_transcript", transcribe)
    await generate()
    assert len(server.jobs) == 2
    assert [job["request"]["seed"] for job in server.jobs.values()] == [42, 43]
    checkpoint = json.loads((directory / "tts_input_part_001.request-0.json").read_text())
    assert "pronunciation mismatch" in checkpoint["rejected"]
    assert (directory / "verification" / "tts_input_part_001" / "rejected-0.wav").is_file()
    rejected = directory / "verification" / "tts_input_part_001" / "rejected-0" / "pronunciation_report.json"
    assert json.loads(rejected.read_text())["passed"] is False
    metadata = json.loads((directory / "tts_input_part_001_generated.json").read_text())
    assert metadata["integrity"]["pronunciation"]["passed"] is True
    transcribe.side_effect = AssertionError("Verified cache must not transcribe or submit again")
    await generate()
    assert len(server.jobs) == 2


@pytest.mark.asyncio
async def test_pronunciation_rejections_stop_at_persisted_attempt_limit(setup_client, monkeypatch):
    server, script, directory, _verifier, generate = setup_client
    script.write_text("OpenAI reports on AI chips.")
    monkeypatch.setattr(av_sync, "ensure_word_transcript", AsyncMock(return_value=(
        [{"text": word} for word in "Open II reports on eye chips.".split()], {},
    )))
    for _ in range(2):
        with pytest.raises(tts.TtsIntegrityError):
            await generate()
    assert len(server.jobs) == chatterbox.MAX_INTEGRITY_ATTEMPTS
    assert not (directory / "tts_manifest.json").exists()
    assert not (directory / "tts_input_part_001_generated.wav").exists()


@pytest.mark.asyncio
async def test_unavailable_pronunciation_verification_preserves_completed_job(setup_client, monkeypatch):
    server, script, directory, _verifier, generate = setup_client
    source = "OpenAI reports on AI chips."
    script.write_text(source)
    transcribe = AsyncMock(return_value=([], {"passed": False}))
    monkeypatch.setattr(av_sync, "ensure_word_transcript", transcribe)
    with pytest.raises(RuntimeError, match="verification unavailable"):
        await generate()
    original = (directory / "tts_input_part_001.request-0.json").read_bytes()
    assert len(server.jobs) == 1
    assert "rejected" not in json.loads(original)
    assert not (directory / "tts_manifest.json").exists()
    transcribe.return_value = ([{"text": word} for word in source.split()], {})
    await generate()
    assert len(server.jobs) == 1
    assert (directory / "tts_input_part_001.request-0.json").read_bytes() == original


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

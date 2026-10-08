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
from backend.pipeline import av_sync, chatterbox, chatterbox_independent, composer, digester, tts, tts_pronunciation


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
            "spoken": candidate["source"] if candidate["source"] == "NASA" else (
                ('eigh' if candidate['source'] == 'A' else tts_pronunciation.LETTER_NAMES[candidate['source']][0])
                if len(candidate['source']) == 1 else '-'.join(candidate['source'].replace('.', ''))
            ),
            "reason": "Fixture pronunciation decision in this sentence.",
        }
        for candidate in tts_pronunciation.candidates(source)
    ]


@pytest.mark.parametrize(('source', 'observation'), [
    ('GPT-6 is available.', 'GPT6 is available.'),
    ('GPT 6 is available.', 'GPT6 is available.'),
    ('ChatGPT-6 is available.', 'ChatGPT6 is available.'),
    ('RTX-4090 is available.', 'RTX4090 is available.'),
    ('GPT-6 is available.', 'GPT six is available.'),
    ('H100 is available.', 'H one hundred is available.'),
    ('GPT-4o is available.', 'GPT4o is available.'),
    ('GPT-4o is available.', 'GPT four oh is available.'),
    ('TLS1.3 is available.', 'TLS one point three is available.'),
])
def test_model_separator_observation_keeps_every_source_letter_and_digit(source, observation):
    report = chatterbox._pronunciation_report(source, [{'text': observation}])
    assert report['passed']


@pytest.mark.parametrize(('source', 'observation'), [
    ('GPT-6 is available.', 'GPT7 is available.'),
    ('ChatGPT-6 is available.', 'ChatGPT7 is available.'),
    ('RTX-4090 is available.', 'RTX409 is available.'),
    ('GPT-6 is available.', 'GPT6 GPT6 is available.'),
    ('GPT. 6 is available.', 'GPT6 is available.'),
    ('GPT6 7 is available.', 'GPT67 is available.'),
    ('GPT-6 is available.', 'GPT 7 is available.'),
    ('H100 is available.', 'H one hundred six is available.'),
    ('H001 is available.', 'H1 is available.'),
    ('GPT-4o is available.', 'GPT four zero is available.'),
    ('TLS1.3 is available.', 'TLS1.4 is available.'),
    ('TLS1.30 is available.', 'TLS1.3 is available.'),
])
def test_model_separator_observation_never_changes_digits_repeats_or_sentence_boundaries(source, observation):
    assert not chatterbox._pronunciation_report(source, [{'text': observation}])['passed']


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
    monkeypatch.setattr(config, 'CHATTERBOX_PHONEME_RUNTIME_LIB', str(tmp_path / 'not-installed'))
    monkeypatch.setattr(config, "CHATTERBOX_TTS_API_KEY", "test-secret")
    monkeypatch.setattr(config, "CHATTERBOX_TTS_URL", "http://chatterbox.test")
    monkeypatch.setattr(config, "CHATTERBOX_TTS_DEPLOYMENT_FINGERPRINT", "deployment-a", raising=False)
    monkeypatch.setattr(config, "CHATTERBOX_TTS_CHUNK_WORDS", 30)
    monkeypatch.setattr(config, "CHATTERBOX_TTS_MAX_CHARS", 250)
    server = FakeService()
    server.independent_audit = AsyncMock(return_value={'required': True, 'passed': True})
    monkeypatch.setattr(chatterbox_independent, 'audit_part', server.independent_audit)
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
async def test_default_production_independent_check_uses_the_original_remaining_attempts(setup_client):
    server, script, directory, _verifier, generate = setup_client
    script.write_text('The report mentions DIY today.')
    passed = {'required': True, 'passed': True, 'scope': 'independent_tail_and_extra_initialism_check'}
    server.independent_audit.side_effect = [tts.TtsIntegrityError('independent trailing why'), passed, passed]
    await generate()
    assert len(server.jobs) == 2
    rejected = json.loads((directory / 'tts_input_part_001.request-0.json').read_text())
    assert 'independent trailing why' in rejected['rejected']
    accepted = json.loads((directory / 'tts_input_part_001_generated.json').read_text())
    assert accepted['synthesis_request']['seed'] == chatterbox.cache_contract()['seed'] + 1
    assert accepted['integrity']['independent_audit'] == passed
    await generate()
    assert len(server.jobs) == 2 and server.independent_audit.await_count == 3


@pytest.mark.asyncio
async def test_default_production_independent_unavailability_reuses_the_completed_job(setup_client):
    server, script, directory, _verifier, generate = setup_client
    script.write_text('The report mentions DIY today.')
    server.independent_audit.side_effect = RuntimeError('independent decoder unavailable')
    with pytest.raises(RuntimeError, match='decoder unavailable'):
        await generate()
    first = json.loads((directory / 'tts_input_part_001.request-0.json').read_text())
    assert first['job_id'] and 'rejected' not in first
    server.independent_audit.side_effect = None
    await generate()
    assert len(server.jobs) == 1
    assert json.loads((directory / 'tts_input_part_001.request-0.json').read_text()) == first


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


@pytest.mark.asyncio
async def test_frozen_legacy_corpus_retains_provider_waveforms_without_new_delivery_cleanup(setup_client, monkeypatch):
    from backend.pipeline import chatterbox_tail
    _server, script, directory, _verifier, _generate = setup_client
    monkeypatch.setattr(chatterbox_tail, 'prepare_delivery_parts',
                        lambda *_args: pytest.fail('A frozen legacy corpus must retain its original delivery policy'))
    output = await chatterbox._generate_chatterbox(str(script), str(directory), 'jane', 'en',
        log=None, emit=lambda _message: None, verify_text=True, frozen_tail_cleanup_version=0)
    assert Path(output).is_file()
    assert 'tail_cleanup_version' not in json.loads((directory / 'tts_manifest.json').read_text())['cache_contract']


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


def test_symbol_initialisms_expand_only_provider_input_outside_protected_text():
    source = "R&D and M&A. N&D. See https://example.com/R&D and R&D@example.com."
    spoken = chatterbox._render_provider_span(source, 0, len(source), [])
    assert spoken == "ar and dee and em and eigh. N and dee. See https://example.com/R&D and R&D@example.com."
    assert source.startswith("R&D and M&A.")


def test_symbol_initialism_expansion_is_budgeted_without_changing_source():
    source = "R&D makes tools. M&A follows."
    chunks = chatterbox._split_text(source, max_words=4, max_chars=250)
    assert " ".join(chunks) == source
    assert all(chatterbox._synthesis_word_count(chatterbox._render_provider_span(chunk, 0, len(chunk), [])) <= 4
               for chunk in chunks)
    with pytest.raises(ValueError, match="token exceeds"):
        chatterbox._split_text("R&D", max_words=2, max_chars=250)


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
async def test_symbol_policy_change_reuses_other_completed_audio(setup_client, monkeypatch):
    server, script, directory, _verifier, generate = setup_client
    source = "R&D reports.\nAI news."
    script.write_text(source)
    render = chatterbox._render_provider_span
    monkeypatch.setattr(chatterbox, "_render_provider_span", tts_pronunciation.render_span)
    await generate()
    second = json.loads((directory / "tts_input_part_002_generated.json").read_text())
    monkeypatch.setattr(chatterbox, "_render_provider_span", render)
    await generate()
    assert len(server.jobs) == 3
    assert list(server.jobs.values())[-1]["request"]["text"] == "ar and dee reports."
    assert json.loads((directory / "tts_input_part_002_generated.json").read_text())["job_id"] == second["job_id"]
    assert script.read_text() == source


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
        prompt = tts_pronunciation.prompt_path().read_text()
        (prompts / tts_pronunciation.prompt_path().name).write_text(prompt + "\nTest policy change.\n")
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
    monkeypatch.setattr(av_sync, 'ensure_word_transcript', AsyncMock(return_value=(
        [{'text': word} for word in script.read_text().split()], {'passed': True},
    )))
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
    ("From materials science. Thank you.", False),
    ("From materials science. If", False),
    ("From material science.", True),
    ("From materials GPM POSPIR to science.", True),
])
def test_trailing_speech_guard_rejects_extra_phrases_with_complete_source(observed, passed):
    assert chatterbox._trailing_speech_report(
        "From materials science.", [{"text": word} for word in observed.split()],
    )["passed"] is passed


@pytest.mark.parametrize("observed,passed", [
    ("OpenAI's results is available. Is available.", False),
    ("The open AI's results is available. Is available.", False),
    ("OpenAI's results is available.", True),
    ("OpenAI's results is available. Available.", True),
    ("OpenAI's results is available. A different observation.", True),
])
def test_repeated_closing_phrase_is_rejected_despite_earlier_advisory_differences(observed, passed):
    report = chatterbox._trailing_speech_report(
        "The OpenAI's result is available.", [{"text": word} for word in observed.split()],
    )
    assert report['passed'] is passed
    if not passed:
        assert report['extra_words'] == ['is', 'available']


def test_source_authored_repeated_ending_is_not_extra_speech():
    source = "The result is available. Is available."
    assert chatterbox._trailing_speech_report(source, [{'text': source}])['passed']


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
            "null": None, "stale": {"version": 1, "passed": True},
            "failed": {"version": chatterbox.TRAILING_SPEECH_VERIFIER_VERSION, "passed": False},
            "missing_passed": {"version": chatterbox.TRAILING_SPEECH_VERIFIER_VERSION},
        }[defect]
    path.write_text(json.dumps(metadata))
    await generate()
    assert len(server.jobs) == 1
    assert verifier.await_count == (2 if defect in {"failed", "missing_passed"} else 1)
    assert json.loads(path.read_text())["integrity"]["pronunciation"]["trailing_speech"]["passed"]


@pytest.mark.asyncio
@pytest.mark.parametrize("extra", ["GPM POSPIR to and SES and.", "If"])
async def test_tail_gate_rejects_legacy_extra_speech_and_preserves_bounded_retry(setup_client, monkeypatch, extra):
    server, script, directory, verifier, generate = setup_client
    script.write_text("From materials science.")
    await generate()
    path = directory / "tts_input_part_001_generated.json"
    metadata = json.loads(path.read_text())
    bad = tts._orpheus_transcript_report(script.read_text(), [{"text": word} for word in
        f"From materials science. {extra}".split()])
    metadata["integrity"].update(bad)
    guard = metadata["integrity"]["pronunciation"]
    guard.pop("trailing_speech")
    guard["transcript"] = f"From materials science. {extra}"
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
async def test_tail_gate_rechecks_legacy_repeat_with_missing_opening_using_remaining_attempt(setup_client, monkeypatch):
    server, script, directory, verifier, generate = setup_client
    source = 'The result is available.'
    raw = 'Result is available. Is available.'
    script.write_text(source)
    await generate()
    path = directory / 'tts_input_part_001_generated.json'
    metadata = json.loads(path.read_text())
    metadata['integrity']['pronunciation'].update(
        transcript=raw, trailing_speech={'version': 2, 'passed': True})
    path.write_text(json.dumps(metadata))
    original_key = json.loads((directory / 'tts_input_part_001.request-0.json').read_text())['idempotency_key']
    verifier.side_effect = [advisory_report(), advisory_report()]
    monkeypatch.setattr(av_sync, 'ensure_word_transcript', AsyncMock(side_effect=[
        ([{'text': raw}], {'passed': True}), ([{'text': source}], {'passed': True}),
    ]))
    await generate()
    assert len(server.jobs) == 2
    assert [job['request']['seed'] for job in server.jobs.values()] == [42, 43]
    original = json.loads((directory / 'tts_input_part_001.request-0.json').read_text())
    assert original['idempotency_key'] == original_key
    assert 'unrequested trailing speech' in original['rejected']
    assert (directory / 'verification/tts_input_part_001/rejected-0.wav').exists()
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


@pytest.mark.parametrize("observed,passed", [
    ("AI R&D reports.", True), ("AI R and D reports.", True),
    ("AIR &D reports.", True), ("AIR&D reports.", True),
    ("AI RAN reports.", False), ("AI R and reports.", False),
    ("AI R and D D reports.", False), ("AI R and D reports R.", False),
    ("AI AIR &D reports.", False), ("AIR & reports.", False),
])
def test_pronunciation_guard_checks_both_symbol_initialism_letters(observed, passed):
    report = chatterbox._pronunciation_report(
        "AI R&D reports.", [{"text": word} for word in observed.split()],
        [{"source": "AI", "start": 0, "end": 2}],
    )
    assert report["targets"] == ["ai", "r", "d"]
    assert report["passed"] is passed


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
@pytest.mark.parametrize(('source', 'observed'), [
    ('The report mentions UI today.', 'The report mentions you I today.'),
    ('The report mentions UI today.', 'The report mentions you aye today.'),
    ('The report mentions UX today.', 'The report mentions you ex today.'),
    ('The report mentions AR today.', 'The report mentions A are today.'),
    ('The report mentions API today.', 'The report mentions eigh pee eye today.'),
    ('The report mentions WHO today.', 'The report mentions double U H O today.'),
    ('OpenAI published a report.', 'Open eigh eye published a report.'),
])
def test_source_bound_letter_names_preserve_every_observed_letter(source, observed):
    report = chatterbox._pronunciation_report(source, [{'text': observed}])
    assert report['passed']
    assert report['transcript'] == observed
    assert chatterbox._trailing_speech_report(source, [{'text': report['normalized_transcript']}])['passed']


@pytest.mark.parametrize(('source', 'observed'), [
    ('The report mentions UX today.', 'The report mentions you exited today.'),
    ('The report mentions UI today.', 'The report mentions you eye eye today.'),
    ('The report mentions AI today.', 'The report mentions eye eye today.'),
    ('The report mentions AI today.', 'The report mentions aye eye today.'),
    ('The report mentions UI today.', 'The report mentions you aye aye today.'),
    ('The report mentions API today.', 'The report mentions eigh eye today.'),
    ('The report mentions API today.', 'The report mentions eigh pee eye eye today.'),
    ('The report mentions WHO today.', 'The report mentions double U O today.'),
    ('The report mentions UX today.', 'The report mentions UX today and you ex.'),
])
def test_letter_name_normalization_cannot_hide_loss_repetition_or_extra_syllables(source, observed):
    assert not chatterbox._pronunciation_report(source, [{'text': observed}])['passed']


@pytest.mark.asyncio
async def test_repaired_spelling_observation_rechecks_original_jobs_without_new_synthesis(setup_client, monkeypatch):
    server, script, directory, _verifier, generate = setup_client
    source = 'The report mentions UX today.'
    observed = 'The report mentions you ex today.'
    script.write_text(source)
    real_report = chatterbox._pronunciation_report
    def legacy_report(text, words, edits):
        report = real_report(text, words, edits)
        report.update(passed=False, differences=[{'kind': 'replace', 'source': ['ux'], 'asr': ['you', 'ex']}])
        return report
    monkeypatch.setattr(chatterbox, '_pronunciation_report', legacy_report)
    monkeypatch.setattr(av_sync, 'ensure_word_transcript', AsyncMock(return_value=([{'text': observed}], {})))
    with pytest.raises(tts.TtsIntegrityError):
        await generate()
    assert len(server.jobs) == chatterbox.MAX_INTEGRITY_ATTEMPTS
    checkpoints = {}
    for path in directory.glob('tts_input_part_001.request-*.json'):
        record = json.loads(path.read_text())
        record.pop('rejection_observation_version')
        path.write_text(json.dumps(record))
        checkpoints[path.name] = (record['job_id'], record['idempotency_key'])
    archived = directory / 'verification/tts_input_part_001/rejected-0/pronunciation_report.json'
    original_report = archived.read_bytes()
    original_audio = (directory / 'verification/tts_input_part_001/rejected-0.wav').read_bytes()
    monkeypatch.setattr(chatterbox, '_pronunciation_report', real_report)
    await generate()
    assert len(server.jobs) == chatterbox.MAX_INTEGRITY_ATTEMPTS
    assert archived.read_bytes() == original_report
    assert (directory / 'tts_input_part_001_generated.wav').read_bytes() == original_audio
    for path in directory.glob('tts_input_part_001.request-*.json'):
        record = json.loads(path.read_text())
        assert (record['job_id'], record['idempotency_key']) == checkpoints[path.name]
    accepted = json.loads((directory / 'tts_input_part_001.request-0.json').read_text())
    assert accepted['rejected']
    assert accepted['revalidated_after_rejection']['observation_spelling_version'] == chatterbox.OBSERVATION_SPELLING_VERSION


@pytest.mark.asyncio
async def test_normalized_letter_names_still_reject_extra_trailing_speech(tmp_path, monkeypatch):
    source = 'The report mentions AR today.'
    observed = 'The report mentions A are today, are today.'
    monkeypatch.setattr(av_sync, 'ensure_word_transcript', AsyncMock(return_value=([{'text': observed}], {})))
    with pytest.raises(tts.TtsIntegrityError, match='trailing speech'):
        await chatterbox._verify_pronunciation(tmp_path / 'audio.wav', source, tmp_path,
            emit=lambda _message: None, edits=tts_pronunciation.candidates(source))


@pytest.mark.asyncio
async def test_independent_audit_rejects_cached_audio_and_retains_remaining_attempts(setup_client):
    server, script, directory, _verifier, generate = setup_client
    await generate()
    original = (directory / 'tts_input_part_001_generated.wav').read_bytes()
    original_metadata = (directory / 'tts_input_part_001_generated.json').read_bytes()
    keys = dict(server.keys)
    calls = 0
    async def audit(_wav, _text, _edits):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise tts.TtsIntegrityError('Observed extra trailing speech')
        return {'passed': True}
    await chatterbox._generate_chatterbox(str(script), str(directory), 'jane', 'en',
        log=None, emit=lambda _message: None, verify_text=True, part_auditor=audit)
    assert len(server.jobs) == 2
    assert [job['request']['seed'] for job in server.jobs.values()] == [42, 43]
    assert all(server.keys[key] == job for key, job in keys.items())
    archive = directory / 'verification/tts_input_part_001/independent-rejected-job-1'
    assert (archive / 'tts_input_part_001_generated.wav').read_bytes() == original
    assert (archive / 'tts_input_part_001_generated.json').read_bytes() == original_metadata
    assert json.loads((directory / 'tts_input_part_001.request-0.json').read_text())['rejected'].startswith('Chatterbox independent')


@pytest.mark.asyncio
async def test_independent_audit_rejection_never_replenishes_retry_budget(setup_client):
    server, script, directory, _verifier, generate = setup_client
    await generate()
    async def reject(*_args):
        raise tts.TtsIntegrityError('Extra speech remains present')
    for _ in range(2):
        with pytest.raises(tts.TtsIntegrityError):
            await chatterbox._generate_chatterbox(str(script), str(directory), 'jane', 'en',
                log=None, emit=lambda _message: None, verify_text=True, part_auditor=reject)
    assert len(server.jobs) == chatterbox.MAX_INTEGRITY_ATTEMPTS
    assert [job['request']['seed'] for job in server.jobs.values()] == [42, 43, 44]


@pytest.mark.asyncio
@pytest.mark.parametrize('outcome', ['passed', 'still_wrong', 'unavailable'])
@pytest.mark.parametrize('rejection', [
    'Independent vocabulary audit rejected speech: unsupported word',
    'Required complete-source phone sequence differs from the source',
    'Word-hint audio differs from its complete source phone sequence',
])
async def test_acoustic_reference_upgrade_reaudits_independent_rejection_without_new_jobs(
        setup_client, outcome, rejection):
    from scripts.chatterbox_phoneme_evidence import EVIDENCE_VERSION
    server, script, directory, _verifier, _generate = setup_client

    async def reject(*_args):
        raise tts.TtsIntegrityError(rejection)

    async def generate(auditor):
        return await chatterbox._generate_chatterbox(
            str(script), str(directory), 'jane', 'en', log=None, emit=lambda _message: None,
            verify_text=True, part_auditor=auditor,
        )

    with pytest.raises(tts.TtsIntegrityError):
        await generate(reject)
    before = {}
    for path in directory.glob('tts_input_part_001.request-*.json'):
        row = json.loads(path.read_text())
        row['rejection_acoustic_reference_version'] = EVIDENCE_VERSION - 1
        path.write_text(json.dumps(row))
        before[path.name] = (row['job_id'], row['idempotency_key'], row['identity'])
    archived = {str(path): path.read_bytes() for path in
                (directory / 'verification/tts_input_part_001').rglob('rejected-*') if path.is_file()}
    calls = 0

    async def revised(*_args):
        nonlocal calls
        calls += 1
        if outcome == 'unavailable':
            raise RuntimeError('Independent decoder unavailable')
        if outcome == 'still_wrong':
            raise tts.TtsIntegrityError('Independent vocabulary audit rejected speech: wrong phones')
        return {'passed': True}

    if outcome == 'passed':
        await generate(revised)
        assert calls == 1
        row = json.loads((directory / 'tts_input_part_001.request-0.json').read_text())
        assert row['revalidated_after_rejection']['acoustic_reference_version'] == EVIDENCE_VERSION
    elif outcome == 'unavailable':
        with pytest.raises(RuntimeError, match='decoder unavailable'):
            await generate(revised)
        assert calls == 1
    else:
        for _ in range(2):
            with pytest.raises(tts.TtsIntegrityError):
                await generate(revised)
        assert calls == chatterbox.MAX_INTEGRITY_ATTEMPTS
    assert len(server.jobs) == chatterbox.MAX_INTEGRITY_ATTEMPTS
    for path in directory.glob('tts_input_part_001.request-*.json'):
        row = json.loads(path.read_text())
        assert (row['job_id'], row['idempotency_key'], row['identity']) == before[path.name]
    for path, contents in archived.items():
        assert Path(path).read_bytes() == contents


@pytest.mark.asyncio
async def test_independent_audit_uncertainty_preserves_cached_audio_and_request(setup_client):
    server, script, directory, _verifier, generate = setup_client
    await generate()
    checkpoint = directory / 'tts_input_part_001.request-0.json'
    before = checkpoint.read_bytes()
    wav = directory / 'tts_input_part_001_generated.wav'
    audio = wav.read_bytes()
    async def unavailable(*_args):
        raise TimeoutError('Recognizer unavailable')
    with pytest.raises(TimeoutError):
        await chatterbox._generate_chatterbox(str(script), str(directory), 'jane', 'en',
            log=None, emit=lambda _message: None, verify_text=True, part_auditor=unavailable)
    assert len(server.jobs) == 1
    assert checkpoint.read_bytes() == before
    assert wav.read_bytes() == audio


@pytest.mark.asyncio
async def test_independent_audit_checks_new_audio_before_acceptance(setup_client):
    server, script, directory, _verifier, _generate = setup_client
    calls = 0
    async def audit(wav, _text, _edits):
        nonlocal calls
        calls += 1
        assert wav.name.endswith('.tmp.wav')
        assert not (directory / 'tts_input_part_001_generated.wav').exists()
        if calls == 1:
            raise tts.TtsIntegrityError('Extra speech')
        return {'passed': True}
    await chatterbox._generate_chatterbox(str(script), str(directory), 'jane', 'en',
        log=None, emit=lambda _message: None, verify_text=True, part_auditor=audit)
    assert len(server.jobs) == 2
    metadata = json.loads((directory / 'tts_input_part_001_generated.json').read_text())
    assert metadata['integrity']['independent_audit']['passed']


@pytest.mark.asyncio
@pytest.mark.parametrize('result', [None, {}, {'passed': False}])
async def test_missing_independent_acceptance_cannot_publish_or_spend_retry(setup_client, result):
    server, script, directory, _verifier, _generate = setup_client
    async def unknown(*_args):
        return result
    with pytest.raises(RuntimeError, match='confirmed acceptance'):
        await chatterbox._generate_chatterbox(str(script), str(directory), 'jane', 'en',
            log=None, emit=lambda _message: None, verify_text=True, part_auditor=unknown)
    assert len(server.jobs) == 1
    assert not (directory / 'tts_input_part_001_generated.wav').exists()
    assert not json.loads((directory / 'tts_input_part_001.request-0.json').read_text()).get('rejected')


@pytest.mark.asyncio
async def test_complete_audit_checks_later_parts_after_rejection_without_partial_concat(setup_client, monkeypatch):
    server, script, directory, _verifier, _generate = setup_client
    script.write_text('The API report is ready.\nThe GPU report is ready.\nThe BBC report is ready.')
    concat = AsyncMock(side_effect=AssertionError('Rejected coverage cannot concatenate'))
    monkeypatch.setattr(tts, '_concat_wav_parts', concat)
    async def audit(_wav, text, _edits):
        if 'API' in text:
            raise tts.TtsIntegrityError('Missing first API letter')
        return {'passed': True}
    async def run():
        return await chatterbox._generate_chatterbox(str(script), str(directory), 'jane', 'en',
            log=None, emit=lambda _message: None, verify_text=True, part_auditor=audit, audit_all_parts=True)
    with pytest.raises(tts.TtsIntegrityError, match='complete part audit'):
        await run()
    assert len(server.jobs) == 5
    report = json.loads((directory / 'tts_audit_coverage.json').read_text())
    assert report['all_parts_observed'] and not report['passed']
    assert report['expected_parts'] == report['observed_parts'] == 3
    assert [row['status'] for row in report['parts']] == ['rejected', 'accepted', 'accepted']
    good_waves = {p.name: p.read_bytes() for p in directory.glob('*_generated.wav')}
    assert len(good_waves) == 2
    assert not (directory / 'tts_manifest.json').exists()
    with pytest.raises(tts.TtsIntegrityError, match='complete part audit'):
        await run()
    assert len(server.jobs) == 5
    assert {p.name:p.read_bytes() for p in directory.glob('*_generated.wav')} == good_waves
    concat.assert_not_awaited()


@pytest.mark.asyncio
async def test_complete_audit_stops_on_uncertainty_and_resumes_identical_job(setup_client):
    server, script, directory, _verifier, _generate = setup_client
    script.write_text('The API report is ready.\nThe GPU report is ready.\nThe BBC report is ready.')
    unavailable = True
    async def audit(_wav, text, _edits):
        if 'GPU' in text and unavailable:
            raise TimeoutError('Observation unavailable')
        return {'passed': True}
    async def run():
        return await chatterbox._generate_chatterbox(str(script), str(directory), 'jane', 'en',
            log=None, emit=lambda _message: None, verify_text=True, part_auditor=audit, audit_all_parts=True)
    with pytest.raises(TimeoutError):
        await run()
    checkpoint = directory / 'tts_input_part_002.request-0.json'
    retained = checkpoint.read_bytes()
    assert len(server.jobs) == 2
    report = json.loads((directory / 'tts_audit_coverage.json').read_text())
    assert report['observed_parts'] == 1 and not report['all_parts_observed'] and not report['passed']
    assert not (directory / 'tts_input_part_003.request-0.json').exists()
    unavailable = False
    await run()
    assert len(server.jobs) == 3
    assert checkpoint.read_bytes() == retained
    report = json.loads((directory / 'tts_audit_coverage.json').read_text())
    assert report['passed'] and report['all_parts_observed']
    assert report['source_sha256'] == hashlib.sha256(script.read_text().encode()).hexdigest()


@pytest.mark.asyncio
@pytest.mark.parametrize(('verify','auditor'), [(False, None), (True, None), (False, AsyncMock())])
async def test_complete_audit_cannot_disable_required_guards(setup_client, verify, auditor):
    server, script, directory, _verifier, _generate = setup_client
    with pytest.raises(ValueError, match='requires text verification'):
        await chatterbox._generate_chatterbox(str(script), str(directory), 'jane', 'en',
            log=None, emit=lambda _message: None, verify_text=verify,
            part_auditor=auditor, audit_all_parts=True)
    assert not server.requests


@pytest.mark.asyncio
async def test_submission_window_stops_new_requests_but_keeps_durable_identity(setup_client):
    server, script, directory, _verifier, _generate = setup_client
    async def blocked(context):
        assert context['part'] == 1 and context['attempt'] == 0 and context['request_identity']
        raise RuntimeError('Production speech stage is active')
    with pytest.raises(RuntimeError, match='Production speech'):
        await chatterbox._generate_chatterbox(str(script), str(directory), 'jane', 'en',
            log=None, emit=lambda _message: None, verify_text=True, before_submit=blocked)
    checkpoint = directory / 'tts_input_part_001.request-0.json'
    before = checkpoint.read_bytes()
    assert not server.jobs
    async def available(_context):
        return {'available': True}
    await chatterbox._generate_chatterbox(str(script), str(directory), 'jane', 'en',
        log=None, emit=lambda _message: None, verify_text=True, before_submit=available)
    accepted = json.loads(checkpoint.read_text())
    retained = json.loads(before)
    assert accepted['identity'] == retained['identity']
    assert accepted['idempotency_key'] == retained['idempotency_key']
    assert len(server.jobs) == 1


@pytest.mark.asyncio
async def test_existing_completed_job_can_resume_without_new_submission_window(setup_client):
    server, _script, directory, _verifier, generate = setup_client
    await generate()
    (directory / 'tts_input_part_001_generated.json').unlink()
    blocked = AsyncMock(side_effect=AssertionError('Existing job must not be resubmitted'))
    script = directory.parent / 'script.txt'
    await chatterbox._generate_chatterbox(str(script), str(directory), 'jane', 'en',
        log=None, emit=lambda _message: None, verify_text=True, before_submit=blocked)
    assert len(server.jobs) == 1
    blocked.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('result', [None, {}, {'available': False}])
async def test_unconfirmed_submission_window_cannot_create_remote_job(setup_client, result):
    server, script, directory, _verifier, _generate = setup_client
    async def unknown(_context):
        return result
    with pytest.raises(RuntimeError, match='window is unverified'):
        await chatterbox._generate_chatterbox(str(script), str(directory), 'jane', 'en',
            log=None, emit=lambda _message: None, verify_text=True, before_submit=unknown)
    assert not server.jobs
    assert not json.loads((directory / 'tts_input_part_001.request-0.json').read_text()).get('job_id')


def test_verification_binding_removes_legacy_reports_without_deleting_evidence(tmp_path):
    original={'integrity_report.json':b'{"passed":true}', 'pronunciation_report.json':b'{"transcript":"old wave"}',
              'transcript.json':b'[{"text":"old wave"}]'}
    for name,data in original.items():(tmp_path/name).write_bytes(data)
    binding={'audio_sha256':'new-wave','job_id':'new-job','source_sha256':'source','request_identity':'request'}
    chatterbox._bind_verification(tmp_path,binding)
    assert json.loads((tmp_path/'observation_binding.json').read_text()) == binding
    assert not any((tmp_path/name).exists() for name in original)
    archive=next((tmp_path/'observation_history').iterdir())
    assert {name:(archive/name).read_bytes() for name in original} == original
    assert json.loads((archive/'provenance.json').read_text())['binding'] == {}


def test_same_wave_binding_reuses_transcript_but_refreshes_decision_reports(tmp_path):
    binding={'audio_sha256':'same-wave','job_id':'same-job','source_sha256':'source','request_identity':'request'}
    chatterbox._bind_verification(tmp_path,binding)
    transcript=b'[{"text":"source"}]'
    (tmp_path/'transcript.json').write_bytes(transcript)
    (tmp_path/'transcript.meta.json').write_text('{"model":"recognizer"}')
    (tmp_path/'integrity_report.json').write_text('{"passed":true}')
    (tmp_path/'pronunciation_report.json').write_text('{"passed":true}')
    chatterbox._bind_verification(tmp_path,binding)
    assert (tmp_path/'transcript.json').read_bytes() == transcript
    assert (tmp_path/'transcript.meta.json').exists()
    assert not (tmp_path/'integrity_report.json').exists()
    assert not (tmp_path/'pronunciation_report.json').exists()
    archive=next((tmp_path/'observation_history').iterdir())
    assert (archive/'integrity_report.json').exists()


def test_changed_wave_cannot_reuse_previous_transcription(tmp_path):
    first={'audio_sha256':'first','job_id':'first-job','source_sha256':'source','request_identity':'request'}
    chatterbox._bind_verification(tmp_path,first)
    (tmp_path/'transcript.json').write_text('[{"text":"old"}]')
    (tmp_path/'transcript.meta.json').write_text('{"audio_size":100}')
    second={**first,'audio_sha256':'second','job_id':'second-job'}
    chatterbox._bind_verification(tmp_path,second)
    assert not (tmp_path/'transcript.json').exists()
    assert not (tmp_path/'transcript.meta.json').exists()
    archive=next((tmp_path/'observation_history').iterdir())
    assert json.loads((archive/'provenance.json').read_text())['binding']==first


@pytest.mark.asyncio
async def test_physical_rejection_cannot_archive_a_prior_passing_verdict(setup_client,monkeypatch):
    server,script,directory,_verifier,generate=setup_client
    await generate()
    original=json.loads((directory/'tts_input_part_001_generated.json').read_text())
    root=directory/'verification/tts_input_part_001'
    root.mkdir(parents=True,exist_ok=True)
    (root/'integrity_report.json').write_text('{"passed":true,"audio":"previous"}')
    (root/'pronunciation_report.json').write_text('{"passed":true,"audio":"previous"}')
    (directory/'tts_input_part_001_generated.json').unlink()
    validate=chatterbox._validate_audio
    def reject_staged(path,text):
        if path.name.endswith('.tmp.wav'):raise tts.TtsIntegrityError('New physical rejection')
        return validate(path,text)
    monkeypatch.setattr(chatterbox,'_validate_audio',reject_staged)
    with pytest.raises(tts.TtsIntegrityError,match='physical rejection'):
        await generate()
    assert len(server.jobs)==3
    for attempt in range(3):
        folder=root/f'rejected-{attempt}'
        assert not (folder/'integrity_report.json').exists()
        assert not (folder/'pronunciation_report.json').exists()
        reason=json.loads((folder/'rejection.json').read_text())
        waveform=root/f'rejected-{attempt}.wav'
        assert reason['audio_sha256']==hashlib.sha256(waveform.read_bytes()).hexdigest()
        assert reason['job_id']==f'job-{attempt+1}'
    assert original['job_id']=='job-1'


@pytest.mark.parametrize(('source','observed'),[
    ('SSH','double S H'),('SSH','double SH'),('SSH','double ess haitch'),
    ('IEEE','I triple E'),('IEEE','I triple ee'),('WW','double double you'),
    ('WWW','triple double U'),('WWW','triple W'),('W','double U'),('W','double you'),
    ('AAI','double A I'),('AAI','double AI'),
])
def test_source_bound_spelling_counts_conventional_repeated_letter_names(source,observed):
    text=f'The report mentions {source} today.'
    report=chatterbox._pronunciation_report(text,[{'text':f'The report mentions {observed} today.'}])
    assert report['passed'],report
    assert report['observation_spelling_version']==chatterbox.OBSERVATION_SPELLING_VERSION


@pytest.mark.parametrize(('source','observed'),[
    ('SH','double S H'),('SSH','triple S H'),('SSH','double S H H'),
    ('SSH','double SSSH'),('SSH','double sausage H'),('SSH','double exited H'),
    ('AI','double A I'),('UU','double you'),('UU','double U'),
    ('WWW','triple U'),('IEEE','I double E'),('IEEE','I triple exited'),
])
def test_repeated_letter_observation_never_drops_glyphs_or_confuses_w_with_two_u(source,observed):
    text=f'The report mentions {source} today.'
    assert not chatterbox._pronunciation_report(text,[{'text':f'The report mentions {observed} today.'}])['passed']


def test_repeated_letter_observation_still_rejects_a_distant_extra_initialism():
    source='The report mentions SSH today.'
    report=chatterbox._pronunciation_report(source,[{'text':'The report mentions SSH and double S H today.'}])
    assert not report['passed']


@pytest.mark.asyncio
async def test_acoustic_observation_upgrade_rechecks_original_completed_jobs_without_resynthesis(setup_client,monkeypatch):
    from backend.pipeline import chatterbox_phonemes
    server,script,directory,_verifier,generate=setup_client
    script.write_text('The report mentions RPC today.')
    monkeypatch.setattr(av_sync,'ensure_word_transcript',AsyncMock(return_value=([{'text':'The report mentions our PC today.'}],{})))
    monkeypatch.setattr(chatterbox_phonemes,'confirm',AsyncMock(return_value=None))
    with pytest.raises(tts.TtsIntegrityError):await generate()
    original={}
    for p in directory.glob('tts_input_part_001.request-*.json'):
        d=json.loads(p.read_text());d['rejection_observation_version']=chatterbox.OBSERVATION_SPELLING_VERSION-1
        p.write_text(json.dumps(d));original[p.name]=(d['job_id'],d['idempotency_key'])
    async def confirm(wav,source,_edits,_directory):
        return {'supported':True,'passed':True,'method':'complete_source_phone_sequence',
                'conditioned_on_reference_text':False,'audio_sha256':hashlib.sha256(wav.read_bytes()).hexdigest(),
                'source_sha256':hashlib.sha256(source.encode()).hexdigest()}
    monkeypatch.setattr(chatterbox_phonemes,'confirm',confirm)
    await generate()
    assert len(server.jobs)==chatterbox.MAX_INTEGRITY_ATTEMPTS
    for p in directory.glob('tts_input_part_001.request-*.json'):
        d=json.loads(p.read_text());assert (d['job_id'],d['idempotency_key'])==original[p.name]
    m=json.loads((directory/'tts_input_part_001_generated.json').read_text())
    assert m['job_id']=='job-1' and m['synthesis_request']['seed']==42
    assert m['integrity']['pronunciation']['passed']
    assert m['integrity']['pronunciation']['asr_spelling_passed'] is False


@pytest.mark.asyncio
@pytest.mark.parametrize('outcome', ['passed', 'wrong_phones', 'unavailable', 'uncited'])
async def test_word_hint_requires_acoustic_proof_even_when_asr_returns_exact_source(setup_client, monkeypatch, outcome):
    from backend.pipeline import chatterbox_phonemes
    server, script, directory, _verifier, generate = setup_client
    source = 'The report mentions SATA today.'
    script.write_text(source)
    row = {'id': 0, 'source': 'SATA', 'action': 'word', 'spoken': 'sat, a',
           'phones': ['S', 'AE', 'T', 'AH'], 'reason': 'A conventional word reading.'}
    monkeypatch.setattr(digester, '_chat', AsyncMock(return_value=json.dumps({'decisions': [row]})))
    monkeypatch.setattr(av_sync, 'ensure_word_transcript', AsyncMock(return_value=([{'text': source}], {})))

    async def confirm(wav, text, edits, _directory):
        from scripts.chatterbox_phoneme_evidence import independent_word_reference
        if outcome == 'unavailable':
            return None
        return {'supported': True, 'passed': outcome in {'passed', 'uncited'}, 'method': 'complete_source_phone_sequence',
                'source_word_reference_policy_version': 0 if outcome == 'uncited' else 1,
                'independent_word_references': {} if outcome == 'uncited' else {'SATA': independent_word_reference('SATA')},
                'conditioned_on_reference_text': False, 'audio_sha256': hashlib.sha256(wav.read_bytes()).hexdigest(),
                'source_sha256': hashlib.sha256(text.encode()).hexdigest(),
                'word_hint_references': [{key: edit[key] for key in ('source', 'start', 'end', 'phones')}
                                         for edit in edits if edit.get('kind') == 'word']}

    monkeypatch.setattr(chatterbox_phonemes, 'confirm', confirm)
    if outcome == 'wrong_phones':
        with pytest.raises(tts.TtsIntegrityError, match='Word-hint audio differs'):
            await generate()
        assert len(server.jobs) == chatterbox.MAX_INTEGRITY_ATTEMPTS
        report = json.loads((directory / 'verification/tts_input_part_001/rejected-0/pronunciation_report.json').read_text())
        assert report['asr_spelling_passed'] is True and report['passed'] is False
    elif outcome == 'unavailable':
        with pytest.raises(RuntimeError, match='word-hint acoustic verification unavailable'):
            await generate()
        assert len(server.jobs) == 1
        checkpoint = json.loads((directory / 'tts_input_part_001.request-0.json').read_text())
        assert checkpoint['job_id'] == 'job-1' and not checkpoint.get('rejected')
    elif outcome == 'uncited':
        with pytest.raises(RuntimeError, match='Independent source-word pronunciation reference unavailable'):
            await generate()
        assert len(server.jobs) == 1
        checkpoint = json.loads((directory / 'tts_input_part_001.request-0.json').read_text())
        assert checkpoint['job_id'] == 'job-1' and not checkpoint.get('rejected')
    else:
        await generate()
        meta_path = directory / 'tts_input_part_001_generated.json'
        metadata = json.loads(meta_path.read_text())
        assert metadata['identity']['word_hint_references'][0]['phones'] == ['S', 'AE', 'T', 'AH']
        assert metadata['integrity']['pronunciation']['asr_spelling_passed'] is True
        metadata['integrity']['pronunciation']['required_word_phone_confirmation']['word_hint_references'][0]['phones'][-1] = 'AA'
        meta_path.write_text(json.dumps(metadata))
        await generate()
        assert len(server.jobs) == 1
        repaired = json.loads(meta_path.read_text())
        assert repaired['integrity']['pronunciation']['required_word_phone_confirmation']['word_hint_references'][0]['phones'][-1] == 'AH'
        repaired['integrity']['pronunciation']['required_word_phone_confirmation'].pop('source_word_reference_policy_version')
        meta_path.write_text(json.dumps(repaired))
        await generate()
        assert len(server.jobs) == 1
        upgraded = json.loads(meta_path.read_text())['integrity']['pronunciation']['required_word_phone_confirmation']
        assert upgraded['source_word_reference_policy_version'] == 1
        assert set(upgraded['independent_word_references']) == {'SATA'}


@pytest.mark.asyncio
async def test_existing_numeric_jobs_retain_legacy_policy_input_seed_and_identity(setup_client):
    server,script,directory,_verifier,_generate=setup_client
    script.write_text('The report mentions A2A today.')
    current=tts_pronunciation.policy_contract()
    legacy=tts_pronunciation.policy_contract(policy_version=3)
    await chatterbox._generate_chatterbox(str(script),str(directory),'jane','en',log=None,
        emit=lambda _m:None,verify_text=True,frozen_pronunciation_policy=legacy)
    checkpoint=directory/'tts_input_part_001.request-0.json';before=checkpoint.read_bytes()
    wav=directory/'tts_input_part_001_generated.wav';original=wav.read_bytes()
    await chatterbox._generate_chatterbox(str(script),str(directory),'jane','en',log=None,
        emit=lambda _m:None,verify_text=True)
    assert len(server.jobs)==1 and checkpoint.read_bytes()==before and wav.read_bytes()==original
    metadata=json.loads((directory/'tts_input_part_001_generated.json').read_text())
    assert metadata['identity']['pronunciation_policy']['version']==3
    assert metadata['synthesis_request']['text']==script.read_text()


@pytest.mark.asyncio
async def test_new_numeric_generation_spells_single_letters_but_retains_digit_and_canonical_text(setup_client):
    server,script,directory,_verifier,generate=setup_client
    script.write_text('The report mentions A2A today.')
    await generate()
    assert server.jobs['job-1']['request']['text']=='The report mentions eigh 2 eigh today.'
    assert script.read_text()=='The report mentions A2A today.'
    metadata=json.loads((directory/'tts_input_part_001_generated.json').read_text())
    assert metadata['identity']['pronunciation_policy']['version']==tts_pronunciation.IMPLEMENTED_POLICY_VERSION
    assert metadata['identity']['text']==script.read_text()


@pytest.mark.asyncio
async def test_opt_in_numeric_policy_sends_exact_decimal_words_and_reuses_the_same_job(setup_client, monkeypatch):
    server, script, directory, _verifier, _generate = setup_client
    source = 'The report mentions TLS1.3 today.'
    script.write_text(source)
    decisions = [{**row, 'action': 'number' if row.get('kind') == 'number' else 'spell',
                  'spoken': tts_pronunciation.numeric_cue(row['source']) if row.get('kind') == 'number' else 'tee ell ess',
                  'reason': 'Exact source-bound numeric and letter reading.'}
                 for row in tts_pronunciation.candidates(source, policy_version=14)]
    monkeypatch.setattr(digester, '_chat', AsyncMock(return_value=json.dumps({'decisions': decisions})))
    policy = tts_pronunciation.policy_contract(policy_version=14)
    async def run():
        return await chatterbox._generate_chatterbox(str(script), str(directory), 'jane', 'en',
            log=None, emit=lambda _message: None, verify_text=True, frozen_pronunciation_policy=policy)
    await run()
    assert len(server.jobs) == 1
    assert server.jobs['job-1']['request']['text'] == 'The report mentions tee ell ess one point three today.'
    metadata = json.loads((directory / 'tts_input_part_001_generated.json').read_text())
    assert metadata['identity']['text'] == source
    assert metadata['identity']['pronunciation_policy']['version'] == 14
    checkpoint = (directory / 'tts_input_part_001.request-0.json').read_bytes()
    await run()
    assert len(server.jobs) == 1
    assert (directory / 'tts_input_part_001.request-0.json').read_bytes() == checkpoint
    assert script.read_text() == source


@pytest.mark.asyncio
@pytest.mark.parametrize('name', ['C++', 'C#'])
async def test_opt_in_programming_policy_sends_all_named_symbols_and_reuses_same_job(setup_client, monkeypatch, name):
    server, script, directory, _verifier, _generate = setup_client
    source = f'The programmer reads a {name} example.'
    script.write_text(source)
    decisions = [{**row, 'action': 'programming',
                  'spoken': tts_pronunciation.PROGRAMMING_CUES[row['source']],
                  'reason': 'Programming language name in the source sentence.'}
                 for row in tts_pronunciation.candidates(source, policy_version=15)]
    monkeypatch.setattr(digester, '_chat', AsyncMock(return_value=json.dumps({'decisions': decisions})))
    policy = tts_pronunciation.policy_contract(policy_version=15)
    async def run():
        return await chatterbox._generate_chatterbox(str(script), str(directory), 'jane', 'en',
            log=None, emit=lambda _message: None, verify_text=True, frozen_pronunciation_policy=policy)
    await run()
    assert len(server.jobs) == 1
    assert server.jobs['job-1']['request']['text'] == source.replace(name, tts_pronunciation.PROGRAMMING_CUES[name])
    metadata = json.loads((directory / 'tts_input_part_001_generated.json').read_text())
    assert metadata['identity']['text'] == source
    assert metadata['identity']['pronunciation_policy']['version'] == 15
    checkpoint = (directory / 'tts_input_part_001.request-0.json').read_bytes()
    await run()
    assert len(server.jobs) == 1
    assert (directory / 'tts_input_part_001.request-0.json').read_bytes() == checkpoint
    assert script.read_text() == source

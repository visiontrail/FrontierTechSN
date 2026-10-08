from __future__ import annotations

import asyncio
import json
import wave

import pytest

from backend.pipeline import chatterbox, tts, tts_pronunciation
from scripts import repair_chatterbox_delivery as repair


def make_audio(directory, texts=("First source sentence.", "Second source sentence."), seed=42,
               policy_version=None, frame_count=12000):
    directory.mkdir(parents=True, exist_ok=True)
    source = "\n".join(texts)
    (directory / "tts_input.txt").write_text(source)
    policy = tts_pronunciation.policy_contract(policy_version=policy_version)
    plan = tts_pronunciation._plan(source, [], {
        "source_sha256": repair.hashlib.sha256(source.encode()).hexdigest(),
        "language": "en", "policy": policy,
    })
    chatterbox._write_json(directory / "tts_pronunciation.json", plan)
    provenance = {key: f"pinned-{key}" for key in repair.PROVENANCE_KEYS}
    provenance.update(voice="space-en-f1", language="en")
    paths, rows = [], []
    for number, text in enumerate(texts, 1):
        stem = f"tts_input_part_{number:03d}"
        path = directory / f"{stem}_generated.wav"
        (directory / f"{stem}.txt").write_text(text)
        with wave.open(str(path), "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(24000)
            writer.writeframes((number + seed).to_bytes(2, "little") * frame_count)
        identity = {**provenance, "provider": "chatterbox", "text": text,
                    "synthesis_text": text, "seed": seed, "synthesis_speed_ratio": 1.0,
                    "pronunciation_policy": policy, "pronunciation_plan_sha256": plan["plan_sha256"]}
        request = {"text": text, "language_id": "en", "voice": provenance["voice"],
                   "speed": 1.0, "seed": seed}
        metadata = {"identity": identity, "synthesis_request": request,
                    "word_count": tts._spoken_word_count(text), "job_id": f"job-{seed}-{number}",
                    "audio_sha256": tts._file_sha256(path),
                    "integrity": {"verified": True, "accepted": True,
                                  "exact_asr_word_coverage": 1.0,
                                  "exact_leading_anchor": True, "exact_trailing_anchor": True,
                                  "token_differences": [], "failure_reasons": []}}
        chatterbox._write_json(tts._part_metadata_path(path), metadata)
        chatterbox._write_json(directory / f"{stem}.request-0.json", {
            "identity": chatterbox._digest({"identity": identity, "payload": request}),
            "idempotency_key": f"key-{seed}-{number}", "job_id": metadata["job_id"],
        })
        paths.append(path)
        rows.append(metadata)
    joined = asyncio.run(tts._concat_wav_parts(paths, directory, log=None))
    integrity = tts._narration_integrity_summary(rows, tts._spoken_word_count(source),
                                               required=True, method="test-exact")
    tts._write_tts_manifest(directory, model=chatterbox.MODEL_ID, source_text=source,
                            chunks=list(texts), wav_parts=paths, output=joined,
                            deterministic=False, integrity=integrity,
                            extra={**provenance, "cache_contract": {"seed": seed, "pronunciation_policy": policy}})
    return directory


def test_repair_preserves_source_bytes_and_records_heterogeneous_jobs(tmp_path):
    original = make_audio(tmp_path / "original")
    before = repair._inventory(original)
    destination = tmp_path / "repair"
    state = repair.prepare_repair(original, destination, [2], 43)
    make_audio(destination / "candidates" / "part_002", ("Second source sentence.",), seed=43)
    joined = asyncio.run(repair.finalize_repair(state))
    assert joined == str(destination / "tts_input_generated.wav")
    assert repair._inventory(original) == before
    assert repair._inventory(destination / "frozen-source") == before
    assert (destination / "tts_input_part_001_generated.wav").read_bytes() == (
        original / "tts_input_part_001_generated.wav").read_bytes()
    report = repair._json(destination / "repair_report.json")
    assert [part["actual_seed"] for part in report["parts"]] == [42, 43]
    assert [part["job_id"] for part in report["parts"]] == ["job-42-1", "job-43-1"]
    assert report["original_manifest_sha256"] == before["tts_manifest.json"]
    manifest = repair._json(destination / "tts_manifest.json")
    assert manifest["source_text_sha256"] == repair._json(original / "tts_manifest.json")["source_text_sha256"]
    assert manifest["cache_contract"]["seed"] is None
    assert manifest["integrity"]["passed"] is True
    # A completed repair resumes without resubmitting or replacing originals.
    resumed = repair.prepare_repair(original, destination, [2], 43, resume=True)
    assert asyncio.run(repair.finalize_repair(resumed)) == joined
    assert repair._inventory(original) == before


@pytest.mark.parametrize("filename", ["tts_input.txt", "tts_input_part_001_generated.wav"])
def test_invalid_source_hash_rejected_before_output_creation(tmp_path, filename):
    source = make_audio(tmp_path / "original")
    path = source / filename
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        repair.prepare_repair(source, tmp_path / "repair", [1], 43)
    assert not (tmp_path / "repair").exists()


def test_candidate_wrong_source_fails_before_delivery_creation(tmp_path):
    source = make_audio(tmp_path / "original")
    destination = tmp_path / "repair"
    state = repair.prepare_repair(source, destination, [2], 43)
    make_audio(destination / "candidates" / "part_002", ("Different source sentence.",), seed=43)
    with pytest.raises(ValueError, match="Candidate source mismatch"):
        asyncio.run(repair.finalize_repair(state))
    assert not (destination / "tts_input_generated.wav").exists()


def test_advisory_accepted_candidate_does_not_satisfy_strict_verification(tmp_path):
    source = make_audio(tmp_path / "original")
    destination = tmp_path / "repair"
    state = repair.prepare_repair(source, destination, [2], 43)
    candidate = make_audio(destination / "candidates" / "part_002", ("Second source sentence.",), seed=43)
    metadata_path = candidate / "tts_input_part_001_generated.json"
    metadata = repair._json(metadata_path)
    metadata["integrity"].update(verified=False, accepted=True, policy=tts.ASR_ADVISORY_POLICY)
    chatterbox._write_json(metadata_path, metadata)
    with pytest.raises(ValueError, match="strict complete source verification"):
        asyncio.run(repair.finalize_repair(state))


def test_resume_rejects_changed_frozen_files_and_other_request(tmp_path):
    source = make_audio(tmp_path / "original")
    destination = tmp_path / "repair"
    state = repair.prepare_repair(source, destination, [], 43)
    with pytest.raises(ValueError, match="collision"):
        repair.prepare_repair(source, destination, [], 44, resume=True)
    frozen = destination / "frozen-source" / "tts_input.txt"
    frozen.write_text("Modified source")
    with pytest.raises(ValueError, match="Frozen source hash mismatch"):
        asyncio.run(repair.finalize_repair(state))


def test_unsafe_manifest_filename_rejected(tmp_path):
    source = make_audio(tmp_path / "original")
    path = source / "tts_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["parts"][0]["input"] = "../outside.txt"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Unsafe manifest filename"):
        repair.prepare_repair(source, tmp_path / "repair", [], 43)


def test_strict_candidate_auditor_rejects_extra_words_without_editing_audio(tmp_path, monkeypatch):
    from types import SimpleNamespace

    source = make_audio(tmp_path / "candidate", ("First source sentence.",), seed=43)
    wav = source / "tts_input_part_001_generated.wav"
    original_hash = tts._file_sha256(wav)

    def transcribe(command, **kwargs):
        assert command[command.index("--model") + 1] == repair.STRICT_AUDIT_MODEL
        words = [{"text": word, "start": index * 0.1, "end": index * 0.1 + 0.09}
                 for index, word in enumerate(("First", "source", "sentence.", "Uh"))]
        repair.Path(command[5]).write_text(json.dumps(words))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(repair.subprocess, "run", transcribe)
    with pytest.raises(tts.TtsIntegrityError, match="strict complete source verification"):
        asyncio.run(repair._audit_candidate_part(wav, "First source sentence.", [], source))
    assert tts._file_sha256(wav) == original_hash
    reports = list((source / "strict-audit").rglob("audit.json"))
    assert len(reports) == 1
    assert repair._json(reports[0])["passed"] is False


def test_regeneration_reaudits_completed_candidates_inside_bounded_adapter(tmp_path, monkeypatch):
    source = make_audio(tmp_path / "original", policy_version=7)
    destination = tmp_path / "repair"
    state = repair.prepare_repair(source, destination, [2], 43)
    candidate = make_audio(destination / "candidates" / "part_002", ("Second source sentence.",), seed=43)
    calls = []

    async def generate(script_path, output_dir, voice, language, **kwargs):
        assert kwargs["verify_text"] is True
        assert callable(kwargs["part_auditor"])
        assert kwargs["frozen_tail_cleanup_version"] == 0
        assert kwargs["frozen_pronunciation_policy"] == tts_pronunciation.policy_contract()
        assert kwargs["frozen_pronunciation_policy"]["version"] != 7
        calls.append(output_dir)

    monkeypatch.setattr(chatterbox, "_generate_chatterbox", generate)
    old_seed = repair.config.TTS_RANDOM_SEED
    asyncio.run(repair.regenerate_selected(state))
    assert calls == [str(candidate)]
    assert repair.config.TTS_RANDOM_SEED == old_seed
    asyncio.run(repair.finalize_repair(state))
    provenance = repair._json(destination / "repair_report.json")["parts"][1]
    assert provenance["original_pronunciation_policy"]["version"] == 7
    assert provenance["pronunciation_policy"] == tts_pronunciation.policy_contract()


def test_submitted_candidate_policy_rejects_changed_source_without_rebinding(tmp_path):
    directory = make_audio(tmp_path / "candidate", ("Original source.",), seed=43)
    before = repair._inventory(directory)
    with pytest.raises(ValueError, match="pronunciation policy is missing or changed"):
        repair._candidate_pronunciation_policy(directory, "Different source.", "en", None)
    assert repair._inventory(directory) == before


@pytest.mark.parametrize("tamper", [None, "source", "transcript"])
def test_independent_exact_observation_is_bound_and_preserves_primary_warning(tmp_path, monkeypatch, tamper):
    from types import SimpleNamespace

    source = make_audio(tmp_path / "original")
    destination = tmp_path / "repair"
    state = repair.prepare_repair(source, destination, [2], 43)
    candidate = make_audio(destination / "candidates" / "part_002", ("Second source sentence.",), seed=43)
    wav = candidate / "tts_input_part_001_generated.wav"

    def transcribe(command, **kwargs):
        assert command[command.index("--model") + 1] == repair.STRICT_AUDIT_MODEL
        words = [{"text": word, "start": index * 0.1, "end": index * 0.1 + 0.09}
                 for index, word in enumerate(("Second", "source", "sentence."))]
        repair.Path(command[5]).write_text(json.dumps(words))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(repair.subprocess, "run", transcribe)
    audit = asyncio.run(repair._audit_candidate_part(wav, "Second source sentence.", [], candidate))
    assert audit["passed"] is True
    if tamper == "source":
        audit["identity"]["source_sha256"] = "wrong-source"
    elif tamper == "transcript":
        (candidate / audit["transcript_path"]).write_text("[]")
    metadata_path = tts._part_metadata_path(wav)
    metadata = repair._json(metadata_path)
    metadata["integrity"].update({
        "verified": False, "accepted": True, "policy": tts.ASR_ADVISORY_POLICY,
        "warnings": ["Primary ASR inserted an"],
        "token_differences": [{"kind": "insert", "source": [], "asr": ["an"]}],
        "failure_reasons": ["Extra primary word"], "independent_audit": audit,
    })
    chatterbox._write_json(metadata_path, metadata)
    manifest = repair._json(candidate / "tts_manifest.json")
    manifest["integrity"] = tts._narration_integrity_summary(
        [metadata], 3, required=True, method="primary-with-independent-observation",
    )
    chatterbox._write_json(candidate / "tts_manifest.json", manifest)
    if tamper:
        with pytest.raises(ValueError, match="strict complete source verification"):
            asyncio.run(repair.finalize_repair(state))
        return
    asyncio.run(repair.finalize_repair(state))
    final = repair._json(destination / "tts_manifest.json")
    assert final["integrity"]["accepted_source_coverage"] == 1.0
    assert final["integrity"]["verified_source_coverage"] == 0.5
    original_observation = final["integrity"]["part_reports"][1]
    assert original_observation["verified"] is False
    assert original_observation["warnings"] == ["Primary ASR inserted an"]
    assert original_observation["independent_audit"]["passed"] is True


def test_shorter_retake_appends_only_silence_and_preserves_slot_and_provider(tmp_path):
    source = make_audio(tmp_path / "original", frame_count=24000)
    destination = tmp_path / "repair"
    state = repair.prepare_repair(source, destination, [2], 43)
    candidate = make_audio(destination / "candidates" / "part_002", ("Second source sentence.",),
                           seed=43, frame_count=12000)
    raw = candidate / "tts_input_part_001_generated.wav"
    raw_hash = tts._file_sha256(raw)
    asyncio.run(repair.finalize_repair(state))
    delivered = destination / "tts_input_part_002_generated_slot.wav"
    with wave.open(str(raw), "rb") as reader:
        provider_samples = reader.readframes(reader.getnframes())
    with wave.open(str(delivered), "rb") as reader:
        samples = reader.readframes(reader.getnframes())
        assert reader.getnframes() == 24000
    assert samples[:len(provider_samples)] == provider_samples
    assert samples[len(provider_samples):] == b"\0" * 24000
    assert tts._file_sha256(raw) == raw_hash
    assert tts._file_sha256(destination / "tts_input_part_002_generated.wav") == raw_hash
    metadata = repair._json(tts._part_metadata_path(delivered))
    assert metadata["provider_audio_sha256"] == raw_hash
    assert metadata["slot_padding"]["added_trailing_frames"] == 12000
    assert metadata["slot_padding"]["speech_retimed"] is False
    report = repair._json(destination / "repair_report.json")["parts"][1]
    assert report["original_frames"] == report["effective_frames"] == 24000
    assert report["slot_padding"]["provider_samples_preserved"] is True

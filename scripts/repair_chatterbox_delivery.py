#!/usr/bin/env python3
"""Repair selected Chatterbox parts in a separate, hash-bound delivery directory.

This script never edits a task, its original audio, or service settings. Prepare
first, optionally inspect/run candidates separately, then finalize and render the
returned narration with the normal composition pipeline. Speech is never retimed.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import wave
from dataclasses import asdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend import config  # noqa: E402
from backend.pipeline import chatterbox, chatterbox_tail, tts, tts_pronunciation  # noqa: E402
from backend.pipeline.tts_tail_cleanup import VERSION as TAIL_VERSION  # noqa: E402

PROVENANCE_KEYS = (
    "source_revision", "model_revision", "model_variant", "deployment_fingerprint",
    "voice", "voice_fingerprint", "language",
)
STRICT_AUDIT_MODEL = "mlx-community/whisper-medium-mlx"


def _json(path: Path) -> dict:
    _path(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {path}")
    return value


def _path(value: str | Path) -> Path:
    path = Path(value).expanduser().absolute()
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise ValueError("Repair paths must not contain symlinks")
    return path.resolve()


def _child(directory: Path, name: str) -> Path:
    if not isinstance(name, str) or not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("Unsafe manifest filename")
    path = directory / name
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Missing or unsafe artifact: {path}")
    return path


def _inventory(directory: Path) -> dict[str, str]:
    directory = _path(directory)
    result = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError(f"Unsafe source entry: {path}")
        if path.is_file():
            result[path.relative_to(directory).as_posix()] = tts._file_sha256(path)
    return result


def _same_bytes(source: Path, target: Path) -> None:
    """Copy once; a resume must never replace a different artifact."""
    _path(source)
    _path(target)
    if target.exists():
        if tts._file_sha256(source) != tts._file_sha256(target):
            raise ValueError(f"Output collision: {target}")
    else:
        shutil.copy2(source, target)


def _load_audio(directory: Path) -> tuple[dict, str, list[dict]]:
    manifest = _json(_child(directory, "tts_manifest.json"))
    source = _child(directory, "tts_input.txt").read_text(encoding="utf-8")
    if manifest.get("model") != chatterbox.MODEL_ID or manifest.get("synthesis_speed_ratio") != 1.0:
        raise ValueError("Only natural-speed Chatterbox manifests can be repaired")
    if manifest.get("source_text_sha256") != hashlib.sha256(source.encode()).hexdigest():
        raise ValueError("Source text hash mismatch")
    output = _child(directory, manifest["output"])
    if tts._file_sha256(output) != manifest.get("output_audio_sha256"):
        raise ValueError("Source joined audio hash mismatch")
    entries = manifest.get("parts")
    if not isinstance(entries, list) or not entries or len(entries) != manifest.get("chunk_count"):
        raise ValueError("Incomplete source parts")
    parts, seen = [], set()
    checkpoints = [chatterbox._read_request_checkpoint(path)
                   for path in directory.glob("tts_input_part_*.request-*.json")]
    for number, entry in enumerate(entries, 1):
        wav = _child(directory, entry["output"])
        text_path = _child(directory, entry["input"])
        text = text_path.read_text(encoding="utf-8")
        if wav.name in seen:
            raise ValueError("Duplicate source waveform")
        seen.add(wav.name)
        metadata_path = _child(directory, tts._part_metadata_path(wav).name)
        metadata = _json(metadata_path)
        identity = metadata.get("identity") or {}
        if tts._file_sha256(wav) != entry.get("audio_sha256") or metadata.get("audio_sha256") != entry.get("audio_sha256"):
            raise ValueError(f"Source audio hash mismatch at part {number}")
        if entry.get("text_sha256") != hashlib.sha256(text.encode()).hexdigest() or identity.get("text") != text:
            raise ValueError(f"Source part text hash mismatch at part {number}")
        if (identity.get("provider") != "chatterbox"
                or any(not manifest.get(key) or identity.get(key) != manifest[key] for key in PROVENANCE_KEYS)
                or identity.get("synthesis_speed_ratio") != 1.0
                or metadata.get("word_count") != tts._spoken_word_count(text)
                or not tts._part_integrity_accepted(metadata.get("integrity"))):
            raise ValueError(f"Unsafe source metadata at part {number}")
        request = metadata.get("synthesis_request") or {}
        if (request.get("text") != identity.get("synthesis_text")
                or request.get("voice") != manifest["voice"]
                or request.get("language_id") != manifest["language"]
                or request.get("speed") != 1.0):
            raise ValueError(f"Unsafe synthesis request at part {number}")
        expected_identity = chatterbox._digest({"identity": identity, "payload": request})
        if not any(item.get("job_id") == metadata.get("job_id") and item.get("identity") == expected_identity
                   for item in checkpoints):
            raise ValueError(f"Missing matching durable job checkpoint at part {number}")
        tts._read_pcm_wav(wav)
        parts.append({"number": number, "text": text, "wav": wav, "metadata": metadata,
                      "metadata_path": metadata_path, "text_path": text_path})
    if " ".join(" ".join(part["text"].split()) for part in parts) != " ".join(source.split()):
        raise ValueError("Parts do not cover the complete source in order")
    if sum(part["metadata"]["word_count"] for part in parts) != manifest.get("source_word_count"):
        raise ValueError("Source word count mismatch")
    return manifest, source, parts


def prepare_repair(source_audio_dir: str | Path, output_dir: str | Path,
                   regenerate_parts: list[int], seed: int, *, resume: bool = False) -> dict:
    source, output = _path(source_audio_dir), _path(output_dir)
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Repair output and source must be separate directories")
    if type(seed) is not int or seed < 0:
        raise ValueError("Seed must be a nonnegative integer")
    manifest, _, parts = _load_audio(source)
    selected = sorted(set(regenerate_parts))
    if any(type(number) is not int or not 1 <= number <= len(parts) for number in selected):
        raise ValueError("Regeneration part numbers must exist in the source")
    inventory = _inventory(source)
    state = {"version": 1, "source_audio_dir": str(source), "output_dir": str(output),
             "original_manifest_sha256": tts._file_sha256(source / "tts_manifest.json"),
             "source_text_sha256": manifest["source_text_sha256"], "source_files": inventory,
             "regenerate_parts": selected, "seed": seed, "tail_cleanup_version": TAIL_VERSION}
    marker = output / "repair_request.json"
    if output.exists():
        if not resume or not marker.is_file() or _json(marker) != state:
            raise ValueError("Output directory collision; use a new directory or matching --resume")
    else:
        output.mkdir(parents=True)
        chatterbox._write_json(marker, state)
    frozen = output / "frozen-source"
    frozen.mkdir(exist_ok=True)
    for name in inventory:
        target = frozen / name
        target.parent.mkdir(parents=True, exist_ok=True)
        _same_bytes(source / name, target)
    _verify_frozen(state)
    (output / "candidates").mkdir(exist_ok=True)
    for number in selected:
        directory = output / "candidates" / f"part_{number:03d}"
        directory.mkdir(exist_ok=True)
        _same_bytes(parts[number - 1]["text_path"], directory / "source.txt")
    return state


def _verify_frozen(state: dict) -> None:
    output = _path(state["output_dir"])
    if _json(output / "repair_request.json") != state:
        raise ValueError("Repair request changed")
    for directory in (_path(state["source_audio_dir"]), output / "frozen-source"):
        if _inventory(directory) != state["source_files"]:
            raise ValueError(f"Frozen source hash mismatch: {directory}")


def _validate_candidate(state: dict, number: int, original: dict) -> dict:
    directory = _path(Path(state["output_dir"]) / "candidates" / f"part_{number:03d}")
    manifest, source, parts = _load_audio(directory)
    if source != original["text"] or len(parts) != 1:
        raise ValueError(f"Candidate source mismatch at part {number}")
    candidate = parts[0]
    metadata = candidate["metadata"]
    identity, integrity = metadata["identity"], metadata["integrity"]
    if any(identity.get(key) != original["metadata"]["identity"].get(key) for key in PROVENANCE_KEYS):
        raise ValueError("Candidate voice or provider provenance changed")
    primary_exact = (_exact_source_report(integrity)
                     and manifest.get("integrity", {}).get("verified_source_coverage") == 1.0)
    independent_exact = _valid_strict_audit(directory, candidate["wav"], source,
                                           integrity.get("independent_audit"))
    if identity.get("seed") != state["seed"] or not (primary_exact or independent_exact):
        raise ValueError(f"Candidate lacks strict complete source verification at part {number}")
    return candidate


def _exact_source_report(report: dict) -> bool:
    return (report.get("verified") is True and report.get("exact_asr_word_coverage") == 1.0
            and report.get("token_differences") == [] and report.get("failure_reasons") == []
            and report.get("exact_leading_anchor") is True and report.get("exact_trailing_anchor") is True
            and report.get("repeat_start_seconds") is None)


def _valid_strict_audit(directory: Path, wav: Path, text: str, audit: object) -> bool:
    if not isinstance(audit, dict) or audit.get("passed") is not True:
        return False
    expected = {"audio_sha256": tts._file_sha256(wav), "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "model": STRICT_AUDIT_MODEL, "version": 1}
    if audit.get("identity") != expected or not _exact_source_report(audit.get("strict_source_report") or {}):
        return False
    relative = audit.get("transcript_path")
    if not isinstance(relative, str) or Path(relative).is_absolute():
        return False
    transcript = _path(directory / relative)
    if not transcript.is_relative_to(directory.resolve()) or not transcript.is_file():
        return False
    if tts._file_sha256(transcript) != audit.get("transcript_sha256"):
        return False
    words = json.loads(transcript.read_text())
    if not isinstance(words, list) or not words or any(not isinstance(word, dict) for word in words):
        return False
    checked = tts._orpheus_transcript_report(text, words)
    return checked == audit["strict_source_report"] and _exact_source_report(checked)


async def _audit_candidate_part(wav: Path, text: str, edits: list[dict], directory: Path) -> dict:
    """Reject text mismatches inside the adapter's durable bounded retry loop.

    The strict Orpheus verifier may trim repetitions in place. Use its pure
    transcript comparison instead so this auditor cannot modify provider audio.
    """
    from backend.pipeline import chatterbox_independent

    audio_hash = tts._file_sha256(wav)
    source_hash = hashlib.sha256(text.encode()).hexdigest()
    target = directory / "strict-audit" / STRICT_AUDIT_MODEL.rsplit("/", 1)[-1] / source_hash / audio_hash
    target.mkdir(parents=True, exist_ok=True)
    identity = {"audio_sha256": audio_hash, "source_sha256": source_hash,
                "model": STRICT_AUDIT_MODEL, "version": 1}
    transcript, marker = target / "transcript.json", target / "identity.json"
    saved = _json(marker) if marker.exists() else {}
    if (not transcript.is_file() or saved.get("identity") != identity
            or saved.get("transcript_sha256") != tts._file_sha256(transcript)):
        env = {**os.environ, "HF_HUB_OFFLINE": "1"}
        try:
            result = await asyncio.to_thread(subprocess.run, [
                sys.executable, "-m", "backend.pipeline.av_sync", "transcribe-mlx",
                str(wav), str(transcript), "--model", STRICT_AUDIT_MODEL, "--language", "en",
            ], cwd=config.PROJECT_ROOT, env=env, capture_output=True, timeout=240)
        except (OSError, subprocess.SubprocessError) as error:
            raise RuntimeError("Strict independent recognizer unavailable; preserving candidate job") from error
        if result.returncode or not transcript.is_file():
            raise RuntimeError("Strict independent recognizer unavailable; preserving candidate job")
        chatterbox._write_json(marker, {"identity": identity, "transcript_sha256": tts._file_sha256(transcript)})
    words = json.loads(transcript.read_text())
    if not isinstance(words, list) or not words or any(not isinstance(word, dict) for word in words):
        raise RuntimeError("Strict independent recognizer returned invalid observations")
    report = tts._orpheus_transcript_report(text, words)
    passed = _exact_source_report(report)
    audit = {"identity": identity, "passed": passed, "strict_source_report": report,
             "transcript_path": transcript.relative_to(directory).as_posix(),
             "transcript_sha256": tts._file_sha256(transcript)}
    chatterbox._write_json(target / "audit.json", audit)
    if tts._file_sha256(wav) != audio_hash:
        raise RuntimeError("Provider waveform changed during strict candidate verification")
    if not passed:
        raise tts.TtsIntegrityError("Repair candidate failed strict complete source verification")
    audit["independent"] = await chatterbox_independent.audit_part(
        wav, text, edits, directory / "independent-production",
    )
    if audit["independent"].get("passed") is not True:
        raise tts.TtsIntegrityError("Repair candidate failed independent verification")
    chatterbox._write_json(target / "audit.json", audit)
    return audit


def _candidate_pronunciation_policy(directory: Path, text: str, language: str,
                                    original_policy: dict | None) -> dict | None:
    """A submitted candidate owns its saved policy, even if the source is older."""
    if any(directory.glob("tts_input_part_*.request-*.json")):
        path = directory / "tts_pronunciation.json"
        if not path.is_file():
            raise ValueError("Submitted candidate has no frozen pronunciation plan")
        plan = _json(path)
        identity = plan.get("identity") or {}
        policy = identity.get("policy")
        normalized = tts_pronunciation.normalize_text(text)
        if (identity.get("source_sha256") != hashlib.sha256(normalized.encode()).hexdigest()
                or identity.get("language") != language or not isinstance(policy, dict)
                or not tts_pronunciation.compatible_policy(policy)):
            raise ValueError("Submitted candidate pronunciation policy is missing or changed")
        checked = tts_pronunciation._plan(normalized, plan.get("decisions"), identity)
        if any(plan.get(key) != value for key, value in checked.items()):
            raise ValueError("Submitted candidate pronunciation plan hash mismatch")
        return copy.deepcopy(policy)
    if original_policy is not None and not tts_pronunciation.compatible_policy(original_policy):
        raise ValueError("Original pronunciation policy is no longer supported")
    return copy.deepcopy(original_policy)


async def regenerate_selected(state: dict) -> None:
    _verify_frozen(state)
    output = Path(state["output_dir"])
    manifest, _, parts = _load_audio(output / "frozen-source")
    previous_seed = config.TTS_RANDOM_SEED
    try:
        config.TTS_RANDOM_SEED = state["seed"]
        for number in state["regenerate_parts"]:
            directory = output / "candidates" / f"part_{number:03d}"
            if (directory / "source.txt").read_text(encoding="utf-8") != parts[number - 1]["text"]:
                raise ValueError(f"Candidate input changed at part {number}")
            policy = _candidate_pronunciation_policy(
                directory, parts[number - 1]["text"], manifest["language"],
                manifest.get("cache_contract", {}).get("pronunciation_policy"),
            )
            async def auditor(wav, text, edits):
                return await _audit_candidate_part(wav, text, edits, directory)

            # Audit existing candidate caches too: a formerly advisory-accepted
            # mismatch must enter the same checkpointed remaining retry budget.
            await chatterbox._generate_chatterbox(
                str(directory / "source.txt"), str(directory), manifest["voice"], manifest["language"],
                log=None, emit=lambda text: print(text, flush=True), verify_text=True,
                frozen_tail_cleanup_version=0,
                frozen_pronunciation_policy=policy,
                part_auditor=auditor,
            )
            _validate_candidate(state, number, parts[number - 1])
    finally:
        config.TTS_RANDOM_SEED = previous_seed


def _pad_shorter_replacement(path: Path, metadata: dict, original: Path) -> tuple[Path, dict, dict]:
    """Preserve the old visual slot by appending silence to a shorter retake."""
    current, slot = tts._read_pcm_wav(path), tts._read_pcm_wav(original)
    if (current.channels, current.sample_width, current.sample_rate) != (slot.channels, slot.sample_width, slot.sample_rate):
        raise ValueError("Replacement audio format differs from its original visual slot")
    digest = tts._file_sha256(path)
    if digest != metadata["audio_sha256"]:
        raise ValueError("Replacement waveform hash changed before slot padding")
    added = max(0, slot.frame_count - current.frame_count)
    report = {"applied": added > 0, "added_trailing_frames": added,
              "original_slot_frames": slot.frame_count, "source_frames": current.frame_count,
              "effective_frames": current.frame_count + added,
              "source_audio_sha256": digest, "source_audio_path": str(path.resolve()),
              "speech_retimed": False, "input_samples_preserved": True,
              "provider_samples_preserved": digest == metadata.get("provider_audio_sha256", digest)}
    if not added:
        return path, metadata, report
    with wave.open(str(path), "rb") as reader:
        params, samples = reader.getparams(), reader.readframes(reader.getnframes())
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setparams(params)
        silence = b"\x80" if current.sample_width == 1 else b"\0" * current.sample_width
        writer.writeframes(samples + silence * current.channels * added)
    output = path.with_name(f"{path.stem}_slot.wav")
    _path(output)
    contents = buffer.getvalue()
    if output.exists():
        if output.read_bytes() != contents:
            raise ValueError("Slot derivative collision; preserving existing audio")
    else:
        with output.open("xb") as stream:
            stream.write(contents)
    report["output_audio_sha256"] = tts._file_sha256(output)
    row = copy.deepcopy(metadata)
    row.setdefault("provider_audio_sha256", digest)
    row.setdefault("provider_audio_path", str(path.resolve()))
    row.update(audio_sha256=report["output_audio_sha256"], wav=asdict(tts._read_pcm_wav(output)), slot_padding=report)
    metadata_path = tts._part_metadata_path(output)
    if metadata_path.exists() and _json(metadata_path) != row:
        raise ValueError("Slot derivative metadata collision")
    chatterbox._write_json(metadata_path, row)
    text_path = output.with_suffix(".txt").with_name(output.with_suffix(".txt").name.replace("_generated", ""))
    if text_path.exists() and text_path.read_text() != row["identity"]["text"]:
        raise ValueError("Slot derivative source collision")
    text_path.write_text(row["identity"]["text"], encoding="utf-8")
    if tts._file_sha256(path) != digest:
        raise ValueError("Provider audio changed during slot padding")
    return output, row, report


async def finalize_repair(state: dict) -> str:
    _verify_frozen(state)
    output = Path(state["output_dir"])
    original, source, originals = _load_audio(output / "frozen-source")
    # Validate every candidate before creating any delivery waveform.
    replacements = {number: _validate_candidate(state, number, originals[number - 1])
                    for number in state["regenerate_parts"]}
    parts, rows, provenance = [], [], []
    for entry in originals:
        number = entry["number"]
        chosen = replacements.get(number, entry)
        stem = f"tts_input_part_{number:03d}"
        wav = output / f"{stem}_generated.wav"
        _same_bytes(chosen["wav"], wav)
        _same_bytes(chosen["metadata_path"], tts._part_metadata_path(wav))
        _same_bytes(entry["text_path"], output / f"{stem}.txt")
        parts.append(wav)
        rows.append(copy.deepcopy(chosen["metadata"]))
        provenance.append({"part": number, "regenerated": number in replacements,
                           "source_text": entry["text"], "source_audio_sha256": entry["metadata"]["audio_sha256"],
                           "provider_audio_sha256": chosen["metadata"]["audio_sha256"],
                           "provider_metadata_sha256": tts._file_sha256(chosen["metadata_path"]),
                           "provider_audio_path": str(chosen["wav"]), "job_id": chosen["metadata"]["job_id"],
                           "requested_seed": chosen["metadata"]["identity"]["seed"],
                           "actual_seed": chosen["metadata"]["synthesis_request"]["seed"],
                           "original_pronunciation_policy": entry["metadata"]["identity"].get("pronunciation_policy"),
                           "pronunciation_policy": chosen["metadata"]["identity"].get("pronunciation_policy"),
                           "pronunciation_plan_sha256": chosen["metadata"]["identity"].get("pronunciation_plan_sha256")})
    _same_bytes(output / "frozen-source" / "tts_input.txt", output / "tts_input.txt")
    parts, rows = chatterbox_tail.prepare_delivery_parts(parts, rows)
    for index, original_part in enumerate(originals):
        if original_part["number"] in replacements:
            parts[index], rows[index], padding = _pad_shorter_replacement(parts[index], rows[index], original_part["wav"])
            provenance[index]["slot_padding"] = padding
        provenance[index].update(
            original_frames=tts._read_pcm_wav(original_part["wav"]).frame_count,
            effective_frames=tts._read_pcm_wav(parts[index]).frame_count,
            delivered_audio_sha256=tts._file_sha256(parts[index]),
        )
    joined = await tts._concat_wav_parts(parts, output, log=None)
    chunks = [entry["text"] for entry in originals]
    integrity = tts._narration_integrity_summary(rows, tts._spoken_word_count(source), required=True,
                                               method="preserved_parts_and_strict_verified_replacements")
    if not integrity["passed"]:
        raise ValueError("Repaired narration has incomplete source coverage")
    report = {"version": 1, "original_manifest_sha256": state["original_manifest_sha256"],
              "source_text_sha256": state["source_text_sha256"], "original_files_unchanged": True,
              "speech_retimed": False, "tail_cleanup_version": TAIL_VERSION, "parts": provenance,
              "output_audio_sha256": tts._file_sha256(joined),
              "original_cache_contract": original.get("cache_contract"),
              "candidate_files": {str(number): _inventory(output / "candidates" / f"part_{number:03d}")
                                  for number in replacements},
              "original_provider_provenance": {key: original[key] for key in PROVENANCE_KEYS}}
    contract = copy.deepcopy(original.get("cache_contract", {}))
    contract.update({"seed": None, "per_part_seed_provenance": "repair_report.json", "tail_cleanup_version": TAIL_VERSION})
    tts._write_tts_manifest(
        output, model=chatterbox.MODEL_ID, source_text=source, chunks=chunks, wav_parts=parts,
        output=joined, deterministic=False, integrity=integrity,
        extra={**{key: original[key] for key in PROVENANCE_KEYS}, "cache_contract": contract,
               "pronunciation_plan_sha256": None, "per_part_provenance": "repair_report.json",
               "repair_original_manifest_sha256": state["original_manifest_sha256"],
               "continuity": chatterbox._continuity_report(source, chunks, parts, rows)},
    )
    _verify_frozen(state)
    report["output_manifest_sha256"] = tts._file_sha256(output / "tts_manifest.json")
    chatterbox._write_json(output / "repair_report.json", report)
    return str(joined)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-audio-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--regenerate-parts", default="", help="One-based comma-separated part numbers")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--resume", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()
    state = prepare_repair(args.source_audio_dir, args.output_dir,
                           [int(value) for value in args.regenerate_parts.split(",") if value],
                           args.seed, resume=args.resume)
    if args.prepare_only:
        print(json.dumps({"prepared": state["output_dir"], "parts": state["regenerate_parts"]}))
        return
    if not args.finalize_only:
        asyncio.run(regenerate_selected(state))
    print(asyncio.run(finalize_repair(state)))


if __name__ == "__main__":
    main()

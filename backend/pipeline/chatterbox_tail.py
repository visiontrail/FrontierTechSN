"""Derive clean delivery parts while retaining provider WAVs and observations."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from backend.pipeline.tts_tail_cleanup import VERSION, clean_verified_tail


async def audit_delivery_tail(source, text, edits, integrity, auditor, verification, *, enabled):
    """Re-audit an evidenced delivery derivative without accepting its noisy source."""
    from backend.pipeline.chatterbox_independent import WaveformTailError

    try:
        return await auditor(source, text, edits)
    except WaveformTailError as error:
        if not enabled:
            raise
        original = error.report
        evidence = copy.deepcopy(integrity)
        evidence['speech_end_seconds'] = max(
            evidence.get('speech_end_seconds', 0), original['waveform_tail']['speech_end_seconds'])
        verification.mkdir(parents=True, exist_ok=True)
        output = verification / 'independent-tail-verification.wav'
        output.unlink(missing_ok=True)
        try:
            proof = clean_verified_tail(
                source, output, evidence,
                expected_audio_sha256=original['identity']['audio_sha256'])
            if not proof['applied']:
                raise error
            checked = await auditor(output, text, edits)
            if checked.get('passed') is not True:
                raise RuntimeError('Independent delivery tail audit did not pass')
            if (checked['identity']['audio_sha256'] != proof['output_audio_sha256']
                    or hashlib.sha256(source.read_bytes()).hexdigest() != proof['original_audio_sha256']):
                raise RuntimeError('Independent tail verification audio changed')
            integrity['independent_tail_recovery'] = {
                'provider_audit': original, 'cleanup': proof,
                'speech_end_seconds': evidence['speech_end_seconds'],
            }
            return checked
        finally:
            output.unlink(missing_ok=True)


def verify_tail_continuity(source: Path, integrity: dict, words: list[dict],
                           verification: Path, *, expected_audio_sha256: str) -> dict:
    """Prove a delivery derivative passes continuity before accepting its source.

    Only the existing complete-speech tail proof can authorize a derivative.
    Real internal pauses, weak transcripts and extra speech still fail. Keep
    the provider waveform and its failed measurement as provenance.
    """
    from backend.pipeline import tts

    original = tts._pocket_internal_silence_report(source, words)
    output = verification / f"continuity-tail-{expected_audio_sha256}.wav"
    # This is a disposable verification derivative; delivery creates its own
    # hash-bound artifact after the provider file receives its permanent name.
    output.unlink(missing_ok=True)
    try:
        proof = clean_verified_tail(source, output, integrity,
                                    expected_audio_sha256=expected_audio_sha256)
        if not proof["applied"]:
            tts._validate_pocket_internal_silence(source, words)
            raise RuntimeError("Continuity recovery requires an evidenced tail defect")
        checked = tts._validate_pocket_internal_silence(output, words)
        return {"provider_internal_silence": original, "delivery_internal_silence": checked,
                "cleanup": proof, "verification_derivative_retained": False}
    finally:
        output.unlink(missing_ok=True)


def prepare_delivery_parts(parts: list[Path], metadata_rows: list[dict]) -> tuple[list[Path], list[dict]]:
    if len(parts) != len(metadata_rows):
        raise ValueError("Tail cleanup requires metadata for every source part")
    delivered, rows, reports = [], [], []
    for source, metadata in zip(parts, metadata_rows):
        digest = metadata["audio_sha256"]
        if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise RuntimeError("Tail-cleaned cache changed: provider waveform no longer matches its observation")
        integrity = metadata.get("integrity") or {}
        independent_recovery = integrity.get('independent_tail_recovery')
        cleanup_integrity = copy.deepcopy(integrity)
        if independent_recovery:
            cleanup_integrity['speech_end_seconds'] = independent_recovery['speech_end_seconds']
        # Include the observation: a newer transcription can move the safe edge.
        binding = hashlib.sha256(json.dumps(
            {"audio": digest, "integrity": integrity, "version": VERSION},
            sort_keys=True, allow_nan=False,
        ).encode()).hexdigest()[:16]
        output = source.with_name(f"{source.stem}_tail_{binding}.wav")
        report_path = output.with_suffix(".cleanup.json")
        report = None
        if output.exists() and report_path.exists():
            candidate = json.loads(report_path.read_text())
            if (candidate.get("version") == VERSION and candidate.get("applied") is True
                    and (candidate.get("proof") or {}).get("source_unchanged") is True
                    and candidate.get("original_audio_sha256") == digest
                    and hashlib.sha256(source.read_bytes()).hexdigest() == digest
                    and candidate.get("output_audio_sha256") == hashlib.sha256(output.read_bytes()).hexdigest()):
                report = candidate
            else:
                raise RuntimeError("Tail-cleaned cache changed; preserving both waveforms")
        if report is None:
            # An interrupted output without its report must not be trusted.
            if output.exists():
                raise RuntimeError("Tail-cleaned audio has no committed provenance report")
            report = clean_verified_tail(source, output, cleanup_integrity, expected_audio_sha256=digest)
            if report["applied"]:
                report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        reports.append(report)
        if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise RuntimeError("Provider waveform changed during tail cleanup; refusing to join it")
        recovery = integrity.get("continuity_tail_recovery")
        if independent_recovery and (not report['applied'] or report['output_audio_sha256'] !=
                                     independent_recovery['cleanup']['output_audio_sha256']):
            raise RuntimeError('Delivery differs from the independently audited tail')
        if recovery and (not report["applied"] or report["output_audio_sha256"] !=
                         recovery["cleanup"]["output_audio_sha256"]):
            raise RuntimeError("Delivery differs from the tail verified by the continuity gate")
        if not report["applied"]:
            delivered.append(source)
            rows.append(metadata)
            continue
        row = copy.deepcopy(metadata)
        row["provider_audio_sha256"] = digest
        row["provider_audio_path"] = str(source.resolve())
        row["audio_sha256"] = report["output_audio_sha256"]
        row["integrity"]["tail_cleanup"] = report
        from backend.pipeline.tts import _pocket_internal_silence_report
        row["integrity"]["provider_internal_silence"] = row["integrity"].get("internal_silence")
        row["integrity"]["internal_silence"] = (
            recovery["delivery_internal_silence"] if recovery else _pocket_internal_silence_report(output)
        )
        # ASR/phone observations still name the immutable provider waveform;
        # the proof records that every speech sample survives in this derivative.
        output.with_suffix(".json").write_text(json.dumps(row, indent=2, allow_nan=False) + "\n")
        input_path = output.with_suffix(".txt").with_name(output.with_suffix(".txt").name.replace("_generated", ""))
        input_path.write_text(metadata["identity"]["text"], encoding="utf-8")
        delivered.append(output)
        rows.append(row)
    if parts:
        summary = {"version": VERSION, "parts": reports, "applied_count": sum(r["applied"] for r in reports)}
        (parts[0].parent / "tts_tail_cleanup_report.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return delivered, rows

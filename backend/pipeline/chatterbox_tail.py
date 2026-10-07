"""Derive clean delivery parts while retaining provider WAVs and observations."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from backend.pipeline.tts_tail_cleanup import VERSION, clean_verified_tail


def prepare_delivery_parts(parts: list[Path], metadata_rows: list[dict]) -> tuple[list[Path], list[dict]]:
    if len(parts) != len(metadata_rows):
        raise ValueError("Tail cleanup requires metadata for every source part")
    delivered, rows, reports = [], [], []
    for source, metadata in zip(parts, metadata_rows):
        digest = metadata["audio_sha256"]
        if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise RuntimeError("Tail-cleaned cache changed: provider waveform no longer matches its observation")
        integrity = metadata.get("integrity") or {}
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
            report = clean_verified_tail(source, output, integrity, expected_audio_sha256=digest)
            if report["applied"]:
                report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        reports.append(report)
        if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise RuntimeError("Provider waveform changed during tail cleanup; refusing to join it")
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
        row["integrity"]["internal_silence"] = _pocket_internal_silence_report(output)
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

"""Silence an evidenced post-speech TTS tail without moving the timeline.

This is not a general noise filter. The caller supplies an ASR integrity report
bound to the original WAV hash. The original file is never modified, and weak
or incomplete evidence leaves the audio alone.
"""
from __future__ import annotations

import hashlib
import io
import math
import os
import tempfile
import wave
from pathlib import Path

import numpy as np

VERSION = 1
MIN_EXACT_COVERAGE = 0.95
SPEECH_GUARD_SECONDS = 0.12
QUIET_WINDOW_SECONDS = 0.01
MIN_QUIET_SECONDS = 0.12
QUIET_RMS_DBFS = -42.0
QUIET_PEAK_DBFS = -30.0
FADE_SECONDS = 0.02


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _evidence_failure(integrity: object) -> str | None:
    if not isinstance(integrity, dict) or integrity.get("accepted") is not True:
        return "integrity_not_accepted"
    transcription = integrity.get("transcription")
    if not isinstance(transcription, dict) or transcription.get("passed") is not True:
        return "transcription_not_confirmed"
    expected = _number(integrity.get("expected_words"))
    observed = _number(integrity.get("transcript_words"))
    coverage = _number(integrity.get("exact_asr_word_coverage"))
    matched = _number(integrity.get("matched_exact_words"))
    ratio = _number(integrity.get("transcript_word_ratio"))
    if (expected is None or expected < 2 or not expected.is_integer()
            or observed != expected or ratio != 1.0):
        return "incomplete_or_extra_transcript"
    if (coverage is None or not MIN_EXACT_COVERAGE <= coverage <= 1.0
            or matched is None or not matched.is_integer()
            or not MIN_EXACT_COVERAGE <= matched / expected <= 1.0
            or abs(coverage - matched / expected) > 0.0001):
        return "insufficient_exact_coverage"
    if not all(integrity.get(key) is True for key in (
        "leading_anchor", "trailing_anchor", "exact_leading_anchor", "exact_trailing_anchor",
    )):
        return "missing_exact_speech_anchors"
    if "repeat_start_seconds" not in integrity or integrity["repeat_start_seconds"] is not None:
        return "repetition_or_missing_repetition_check"
    differences = integrity.get("token_differences")
    if not isinstance(differences, list):
        return "missing_token_comparison"
    replacements = 0
    for difference in differences:
        if not isinstance(difference, dict) or difference.get("kind") != "replace":
            return "unmatched_or_extra_transcript"
        source, observed = difference.get("source"), difference.get("asr")
        if (not isinstance(source, list) or not isinstance(observed, list)
                or not source or len(source) != len(observed)
                or any(not isinstance(word, str) or not word for word in source + observed)):
            return "unmatched_or_extra_transcript"
        replacements += len(source)
    if replacements != expected - matched:
        return "inconsistent_token_comparison"
    pronunciation = integrity.get("pronunciation")
    if pronunciation is not None:
        if not isinstance(pronunciation, dict) or pronunciation.get("passed") is not True:
            return "pronunciation_not_accepted"
        trailing = pronunciation.get("trailing_speech")
        if (not isinstance(trailing, dict) or trailing.get("passed") is not True
                or trailing.get("extra_words")):
            return "trailing_speech_not_accepted"
    return None


def clean_verified_tail(
    source: Path,
    output: Path,
    integrity: dict,
    *,
    expected_audio_sha256: str,
) -> dict:
    """Write a new PCM16 WAV only when the complete tail proof is available.

    A 120 ms guard follows the final recognized word. A further 120 ms quiet
    interval must have every 10 ms window below -42 dBFS RMS and -30 dBFS peak.
    The 20 ms fade starts inside that interval; all earlier frames and the
    complete frame count are retained. These bounds avoid editing speech on
    ASR timing alone. Equal-length interior spelling differences may remain at
    >=95% exact coverage, but omissions, extra words and repeats cannot pass.

    ``expected_audio_sha256`` must come from the caller's validated observation
    binding or original part metadata, not a newly computed substitute for it.
    No output is created for an ineligible/invalid source. An existing output
    is never overwritten. Retain ``source`` as the immutable audit original.
    """
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve():
        raise ValueError("Tail cleanup requires a distinct output; preserve the original WAV")
    report = {
        "version": VERSION,
        "method": "verified_post_speech_tail_silence",
        "applied": False,
        "source_path": str(source.resolve()),
        "output_path": None,
        "original_audio_sha256": None,
        "output_audio_sha256": None,
        "proof": {
            "minimum_exact_coverage": MIN_EXACT_COVERAGE,
            "speech_guard_seconds": SPEECH_GUARD_SECONDS,
            "quiet_window_seconds": QUIET_WINDOW_SECONDS,
            "minimum_quiet_seconds": MIN_QUIET_SECONDS,
            "quiet_rms_dbfs": QUIET_RMS_DBFS,
            "quiet_peak_dbfs": QUIET_PEAK_DBFS,
            "fade_seconds": FADE_SECONDS,
        },
    }

    def unchanged(reason: str) -> dict:
        report["reason"] = reason
        return report

    failure = _evidence_failure(integrity)
    if failure:
        return unchanged(failure)
    try:
        original = source.read_bytes()
        digest = hashlib.sha256(original).hexdigest()
        report["original_audio_sha256"] = digest
        if digest != expected_audio_sha256:
            return unchanged("observation_audio_hash_mismatch")
        with wave.open(io.BytesIO(original), "rb") as reader:
            params = reader.getparams()
            if (params.comptype != "NONE" or params.sampwidth != 2
                    or params.nchannels not in (1, 2) or not 8000 <= params.framerate <= 192000):
                return unchanged("unsupported_wav_format")
            pcm = reader.readframes(params.nframes)
        if not params.nframes or len(pcm) != params.nframes * params.nchannels * 2:
            return unchanged("incomplete_wav_payload")
    except (OSError, EOFError, wave.Error):
        return unchanged("unreadable_wav")

    rate, count = params.framerate, params.nframes
    speech_end = _number(integrity.get("speech_end_seconds"))
    if speech_end is None or not 0 < speech_end < count / rate:
        return unchanged("invalid_speech_end")
    samples = np.frombuffer(pcm, dtype="<i2").reshape(-1, params.nchannels)
    window = max(1, round(QUIET_WINDOW_SECONDS * rate))
    quiet_count = math.ceil(MIN_QUIET_SECONDS * rate / window)
    # Round forward so neither sample rounding nor window alignment enters the guard.
    search_start = math.ceil((speech_end + SPEECH_GUARD_SECONDS) * rate / window) * window
    end = search_start + (count - search_start) // window * window
    if search_start >= count or end - search_start < quiet_count * window:
        return unchanged("no_room_for_guard_and_quiet_interval")
    windows = samples[search_start:end].astype(np.float64).reshape(-1, window, params.nchannels)
    rms = np.sqrt(np.mean(windows ** 2, axis=1)).max(axis=1)
    peak = np.max(np.abs(windows), axis=(1, 2))
    quiet = ((rms <= 32768 * 10 ** (QUIET_RMS_DBFS / 20))
             & (peak <= 32768 * 10 ** (QUIET_PEAK_DBFS / 20)))
    run = 0
    first = None
    for index, is_quiet in enumerate(quiet):
        run = run + 1 if is_quiet else 0
        if run >= quiet_count:
            first = index - quiet_count + 1
            break
    if first is None:
        return unchanged("no_verified_quiet_interval")
    fade_start = search_start + first * window
    quiet_end = fade_start + quiet_count * window
    fade_end = fade_start + max(2, round(FADE_SECONDS * rate))
    # Do not rewrite a naturally silent tail or use the quiet interval itself as
    # evidence of a defect. Residual sound must occur after the complete interval.
    remainder = samples[quiet_end:].astype(np.float64)
    if not remainder.size or np.max(np.abs(remainder)) <= 32768 * 10 ** (QUIET_RMS_DBFS / 20):
        return unchanged("tail_already_quiet")
    cleaned = samples.copy()
    envelope = np.linspace(1.0, 0.0, fade_end - fade_start)[:, None]
    cleaned[fade_start:fade_end] = np.rint(cleaned[fade_start:fade_end] * envelope).astype("<i2")
    cleaned[fade_end:] = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, staged_name = tempfile.mkstemp(prefix=f".{output.stem}-", suffix=".wav", dir=output.parent)
    os.close(descriptor)
    staged = Path(staged_name)
    try:
        with wave.open(str(staged), "wb") as writer:
            writer.setparams(params)
            writer.writeframes(cleaned.tobytes())
        if source.read_bytes() != original:
            return unchanged("source_changed_during_cleanup")
        # A concurrent writer cannot replace an existing output or the source.
        os.link(staged, output)
        if source.read_bytes() != original:
            output.unlink()
            return unchanged("source_changed_during_cleanup")
    finally:
        staged.unlink(missing_ok=True)
    report.update({
        "applied": True,
        "reason": "evidenced_post_speech_residual",
        "output_path": str(output.resolve()),
        "output_audio_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "sample_rate": rate,
        "channels": params.nchannels,
        "original_frame_count": count,
        "output_frame_count": len(cleaned),
        "preserved_prefix_frames": fade_start,
        "fade_start_frame": fade_start,
        "silence_start_frame": fade_end,
        "cleanup_start_seconds": fade_start / rate,
        "silence_start_seconds": fade_end / rate,
        "cleanup_end_seconds": count / rate,
    })
    report["proof"].update({
        "expected_audio_sha256": expected_audio_sha256,
        "exact_asr_word_coverage": integrity["exact_asr_word_coverage"],
        "speech_end_seconds": speech_end,
        "quiet_start_seconds": fade_start / rate,
        "quiet_end_seconds": quiet_end / rate,
        "quiet_interval_max_rms_dbfs": 20 * math.log10(max(float(rms[first:first + quiet_count].max()), 1e-9) / 32768),
        "quiet_interval_max_peak_dbfs": 20 * math.log10(max(float(peak[first:first + quiet_count].max()), 1e-9) / 32768),
        "preserved_prefix_sha256": hashlib.sha256(pcm[:fade_start * params.nchannels * 2]).hexdigest(),
        "source_unchanged": source.read_bytes() == original,
    })
    return report

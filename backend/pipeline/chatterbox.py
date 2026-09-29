"""Pinned Chatterbox jobs with the same narration contract as Pocket TTS.

The public demonstration truncates long text. This adapter instead sends bounded
sentence/clause chunks to our serialized service and records each accepted job
before polling, so a resume never submits an uncertain request under a new key.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import httpx

from backend import config
from backend.pipeline import tts

MODEL_ID = "chatterbox-multilingual"
VERIFIER_VERSION = 1
MAX_INTEGRITY_ATTEMPTS = 3
RETRYABLE_SERVICE_ERRORS = frozenset({
    "service_restarted", "service_stopping", "generation_timeout",
    "model_process_exited", "generation_failed",
    # This is a confirmed physical rejection before the service publishes WAV.
    "decoder_duration_limit",
})
SYNTHESIS_PARAMETERS = {
    "exaggeration": 0.5, "cfg_weight": 0.5, "temperature": 0.8,
    "repetition_penalty": 2.0, "min_p": 0.05, "top_p": 1.0, "speed": 1.0,
}


def cache_contract() -> dict:
    """Bind outer audio caches to the same synthesis and verification policy.

    Read settings and verifier constants at call time: Admin changes and code
    upgrades must also invalidate callers that can skip this adapter entirely.
    """
    return {
        "url": config.CHATTERBOX_TTS_URL.rstrip("/"),
        "source_revision": config.CHATTERBOX_TTS_SOURCE_REVISION,
        "model_revision": config.CHATTERBOX_TTS_MODEL_REVISION,
        "model_variant": config.CHATTERBOX_TTS_MODEL_VARIANT,
        "deployment_fingerprint": config.CHATTERBOX_TTS_DEPLOYMENT_FINGERPRINT,
        "seed": config.TTS_RANDOM_SEED,
        "chunk_words": config.CHATTERBOX_TTS_CHUNK_WORDS,
        "max_chars": config.CHATTERBOX_TTS_MAX_CHARS,
        "verifier_version": VERIFIER_VERSION,
        "shared_verifier_version": tts.POCKET_TTS_INTEGRITY_VERIFIER_VERSION,
        "synthesis_parameters": dict(SYNTHESIS_PARAMETERS),
    }


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    staged = path.with_suffix(path.suffix + ".tmp")
    staged.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(staged, path)


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _read_request_checkpoint(path: Path) -> dict:
    """An unreadable submitted-job identity must never become a new request."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"Chatterbox request checkpoint {path.name} is unreadable; "
            "preserving it and refusing to submit another job"
        ) from exc
    valid = isinstance(value, dict) and all(
        isinstance(value.get(key), str) and bool(value[key].strip())
        for key in ("identity", "idempotency_key")
    )
    if valid and "job_id" in value:
        valid = isinstance(value["job_id"], str) and bool(value["job_id"].strip())
    if valid and "rejected" in value:
        valid = bool(value.get("job_id")) and isinstance(value["rejected"], str) and bool(value["rejected"])
    if valid and "terminal_failure" in value:
        failure = value["terminal_failure"]
        valid = (
            isinstance(failure, dict) and bool(value.get("job_id"))
            and failure.get("job_id") == value["job_id"]
            and failure.get("status") == "failed"
            and isinstance(failure.get("error"), str)
            and failure.get("error") in RETRYABLE_SERVICE_ERRORS
        )
    if not valid:
        raise RuntimeError(
            f"Chatterbox request checkpoint {path.name} is invalid; "
            "preserving it and refusing to submit another job"
        )
    return value


def _split_text(text: str, *, max_words: int, max_chars: int) -> list[str]:
    """Keep physical script lines, complete sentences and every source token."""
    if max_words < 1 or max_chars < 20:
        raise ValueError("Chatterbox chunk limits must be positive (at least 20 characters)")
    chunks = []
    for line in text.splitlines():
        remaining = " ".join(line.split())
        while remaining:
            words = remaining.split()
            limit = min(max_chars, len(" ".join(words[:max_words])))
            if len(remaining) <= limit:
                chunks.append(remaining)
                break
            # Prefer sentence punctuation, then clause punctuation. Do not cut
            # an abbreviation/decimal at an internal period.
            window = remaining[:limit + 1]
            boundaries = [m.end() for m in re.finditer(r"[.!?][\"'”’)]*\s+", window)
                          if not re.search(r"\b(?:[A-Z]|Dr|Mr|Mrs|Ms|Prof|St|Inc|Ltd|vs)\.$", window[:m.start() + 1])]
            if not boundaries:
                boundaries = [m.end() for m in re.finditer(r"[,;:]\s+", window)]
            cut = boundaries[-1] if boundaries else window.rfind(" ")
            if cut <= 0:
                raise ValueError("A Chatterbox input token exceeds the character limit; refusing to truncate it")
            chunks.append(remaining[:cut].strip())
            remaining = remaining[cut:].strip()
    if " ".join(chunks) != " ".join(text.split()):
        raise ValueError("Chatterbox splitting changed source text")
    return chunks


async def _request(client: httpx.AsyncClient, method: str, url: str, *, emit, **kwargs):
    """POST retries are safe only because the caller persists its idempotency key."""
    for attempt in range(4):
        try:
            response = await client.request(method, url, **kwargs)
            response.raise_for_status()
            return response
        except httpx.HTTPError as exc:
            if tts._is_permanent_orpheus_http_error(exc) or attempt == 3:
                raise
            delay = min(2 ** attempt, 8)
            emit(f"Chatterbox transient {method} error; retrying the same request in {delay}s")
            await asyncio.sleep(delay)
    raise AssertionError("unreachable")


def _validate_audio(path: Path, text: str) -> tts.WavInfo:
    info = tts._validate_downloaded_wav_container(path, provider="Chatterbox")
    if (info.sample_rate, info.channels, info.sample_width) != (24000, 1, 2):
        raise tts.TtsIntegrityError("Chatterbox must return 24 kHz mono PCM16 WAV")
    tts._validate_wav_part(path, text)
    # The pinned V2 decoder has a 1,000-token, 25-token/s ceiling. A response
    # at that ceiling can be a cut-off utterance even with a valid WAV header.
    if info.duration_seconds >= 38.8:
        raise tts.TtsIntegrityError("Chatterbox reached its decoder duration ceiling")
    return info


def _cached_part(path: Path, identity: dict, *, verify_text: bool) -> dict | None:
    metadata = _read_json(tts._part_metadata_path(path))
    if metadata.get("identity") != identity or not path.is_file():
        return None
    if verify_text and not tts._part_integrity_accepted(metadata.get("integrity")):
        return None
    try:
        info = _validate_audio(path, identity["text"])
        if metadata.get("audio_sha256") != tts._file_sha256(path) or metadata.get("wav") != asdict(info):
            return None
    except tts.TtsIntegrityError:
        return None
    return metadata


def _validate_job_identity(
    job: dict, *, job_id: str, text: str, payload: dict,
    fingerprint: str, revisions: dict, voice_fingerprint: str,
) -> None:
    """Only an identified terminal job can authorize audio or a fresh attempt."""
    if job.get("id") != job_id:
        raise RuntimeError("Chatterbox response does not match its checkpoint job ID")
    if job.get("text_sha256") != hashlib.sha256(text.encode()).hexdigest():
        raise RuntimeError("Chatterbox job does not match the complete requested text")
    if job.get("request") != payload:
        raise RuntimeError("Chatterbox job generation parameters do not match its checkpoint")
    if job.get("fingerprint") != fingerprint or any(job.get(key) != value for key, value in revisions.items()):
        raise RuntimeError("Chatterbox deployment changed during synthesis")
    if job.get("voice_fingerprint") != voice_fingerprint:
        raise RuntimeError("Chatterbox voice changed during synthesis")


def _service_attempts_exhausted(part_index: int, failure: dict) -> RuntimeError:
    return RuntimeError(
        f"Chatterbox part {part_index} exhausted {MAX_INTEGRITY_ATTEMPTS} persisted "
        f"generation attempts; last service job {failure.get('job_id')} failed: "
        f"{failure.get('error')}"
    )


def _continuity_report(text: str, chunks: list[str], parts: list[Path], rows: list[dict]) -> dict:
    # Reuse only waveform measurements, not Pocket's provider-specific claims.
    measurements = tts._pocket_continuity_report(text, chunks, parts, rows)
    return {
        "strategy": "physical_lines_then_bounded_sentences_clauses_and_words",
        "external_chunk_word_limit": config.CHATTERBOX_TTS_CHUNK_WORDS,
        "external_chunk_character_limit": config.CHATTERBOX_TTS_MAX_CHARS,
        "provider_speech_token_limit": 1000,
        "provider_speech_tokens_per_second": 25,
        **{key: measurements[key] for key in (
            "physical_script_line_count", "external_part_count", "application_join_count",
            "intra_line_application_join_count", "silence_measurement",
        )},
    }


async def _generate_chatterbox(
    script_path: str, output_dir: str, voice: str, language: str, *,
    log, emit, verify_text: bool = False,
) -> str:
    # The service can speak 23 languages. The project's existing source-word,
    # ASR and subtitle contract is currently certified for English only.
    if language != "en":
        raise ValueError("The Chatterbox video adapter currently validates English narration only")
    if not config.CHATTERBOX_TTS_API_KEY:
        raise RuntimeError("Set CHATTERBOX_TTS_API_KEY in Admin -> System -> Voice & TTS")
    _, directory, _, cleaned = tts._prepare_tts_input(script_path, output_dir, strip_speaker_labels=True)
    base_url = config.CHATTERBOX_TTS_URL.rstrip("/")
    timeout = httpx.Timeout(max(5, config.CHATTERBOX_TTS_REQUEST_TIMEOUT), connect=15)
    async with httpx.AsyncClient(timeout=timeout, headers={"X-API-Key": config.CHATTERBOX_TTS_API_KEY}, follow_redirects=False) as client:
        response = await _request(client, "GET", f"{base_url}/v1/capabilities", emit=emit)
        capabilities = response.json()
        revisions = {
            "source_revision": config.CHATTERBOX_TTS_SOURCE_REVISION,
            "model_revision": config.CHATTERBOX_TTS_MODEL_REVISION,
            "model_variant": config.CHATTERBOX_TTS_MODEL_VARIANT,
        }
        if any(capabilities.get(key) != value for key, value in revisions.items()):
            raise RuntimeError("Chatterbox deployed revisions do not match the configured pinned build")
        fingerprint = config.CHATTERBOX_TTS_DEPLOYMENT_FINGERPRINT
        if not fingerprint or capabilities.get("fingerprint") != fingerprint:
            raise RuntimeError("Chatterbox deployment fingerprint is missing or changed; verify the service and its voice references in Admin -> System")
        voices = {item["id"]: item for item in capabilities.get("voices", [])}
        voice_info = voices.get(voice)
        if not voice_info or not voice_info.get("fingerprint"):
            raise RuntimeError(f"Chatterbox voice '{voice}' has no installed reference fingerprint")
        chunks = _split_text(cleaned, max_words=config.CHATTERBOX_TTS_CHUNK_WORDS,
                             max_chars=min(config.CHATTERBOX_TTS_MAX_CHARS, int(capabilities["max_chars"])))
        emit(f"Chatterbox: {len(chunks)} bounded request(s), natural 1.0x speech, voice={voice}")
        parts, metadata_rows = [], []
        for index, text in enumerate(chunks, 1):
            stem = f"tts_input_part_{index:03d}"
            (directory / f"{stem}.txt").write_text(text, encoding="utf-8")
            path = directory / f"{stem}_generated.wav"
            identity = {
                **revisions, "provider": "chatterbox", "url": base_url,
                "deployment_fingerprint": fingerprint,
                "voice": voice, "voice_fingerprint": voice_info["fingerprint"],
                "language": language, "text": text,
                "seed": config.TTS_RANDOM_SEED, "verifier_version": VERIFIER_VERSION,
                "shared_verifier_version": tts.POCKET_TTS_INTEGRITY_VERIFIER_VERSION,
                "synthesis_speed_ratio": 1.0,
                "synthesis_parameters": SYNTHESIS_PARAMETERS,
            }
            cached = _cached_part(path, identity, verify_text=verify_text)
            if cached:
                emit(f"Chatterbox part {index}/{len(chunks)}: reusing hash-validated audio")
                parts.append(path)
                metadata_rows.append(cached)
                continue
            for attempt in range(MAX_INTEGRITY_ATTEMPTS):
                request_path = directory / f"{stem}.request-{attempt}.json"
                payload = {
                    "text": text, "language_id": language, "voice": voice,
                    "seed": (config.TTS_RANDOM_SEED + attempt) % (2 ** 32),
                    **SYNTHESIS_PARAMETERS,
                }
                request_identity = _digest({"identity": identity, "payload": payload})
                checkpoint = _read_request_checkpoint(request_path)
                if checkpoint.get("identity") != request_identity:
                    checkpoint = {"identity": request_identity, "idempotency_key": uuid.uuid4().hex}
                    _write_json(request_path, checkpoint)
                if checkpoint.get("rejected"):
                    if attempt + 1 == MAX_INTEGRITY_ATTEMPTS:
                        raise tts.TtsIntegrityError(f"Chatterbox part {index} exhausted physical audio validation attempts")
                    continue
                if checkpoint.get("terminal_failure"):
                    if attempt + 1 == MAX_INTEGRITY_ATTEMPTS:
                        raise _service_attempts_exhausted(index, checkpoint["terminal_failure"])
                    continue
                if not checkpoint.get("job_id"):
                    response = await _request(client, "POST", f"{base_url}/v1/jobs", emit=emit,
                                              json=payload, headers={"Idempotency-Key": checkpoint["idempotency_key"]})
                    checkpoint["job_id"] = response.json()["id"]
                    _write_json(request_path, checkpoint)
                job_id = checkpoint["job_id"]
                emit(f"Chatterbox part {index}/{len(chunks)}: following job {job_id}")
                deadline = time.monotonic() + config.TTS_TIMEOUT
                previous_status = None
                while True:
                    response = await _request(client, "GET", f"{base_url}/v1/jobs/{job_id}", emit=emit)
                    job = response.json()
                    status = job.get("status")
                    if status != previous_status:
                        emit(f"Chatterbox job {job_id}: {status}")
                        previous_status = status
                    if status in {"completed", "failed"}:
                        _validate_job_identity(
                            job, job_id=job_id, text=text, payload=payload,
                            fingerprint=fingerprint, revisions=revisions,
                            voice_fingerprint=voice_info["fingerprint"],
                        )
                        if status == "completed":
                            break
                    if status == "failed":
                        error = job.get("error", "unknown error")
                        if error not in RETRYABLE_SERVICE_ERRORS:
                            raise RuntimeError(f"Chatterbox job {job_id} failed: {error}")
                        # The service has positively ended this exact job. Save
                        # that evidence before advancing; transport uncertainty
                        # never reaches this branch or spends another attempt.
                        failure = {"job_id": job_id, "status": status, "error": error}
                        checkpoint["terminal_failure"] = failure
                        _write_json(request_path, checkpoint)
                        if attempt + 1 == MAX_INTEGRITY_ATTEMPTS:
                            raise _service_attempts_exhausted(index, failure)
                        emit(f"Chatterbox job {job_id} ended with {error}; advancing to the next bounded attempt")
                        break
                    if status not in {"queued", "running"}:
                        raise RuntimeError(f"Chatterbox job {job_id} returned an invalid status")
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"Chatterbox job {job_id} timed out; its checkpoint is preserved")
                    await asyncio.sleep(max(0.1, config.CHATTERBOX_TTS_POLL_SECONDS))
                if checkpoint.get("terminal_failure"):
                    continue
                response = await _request(client, "GET", f"{base_url}/v1/jobs/{job_id}/audio", emit=emit)
                staged = path.with_suffix(".tmp.wav")
                staged.write_bytes(response.content)
                try:
                    if tts._file_sha256(staged) != job.get("audio_sha256"):
                        raise RuntimeError("Chatterbox download does not match the completed job audio hash")
                    _validate_audio(staged, text)
                    if verify_text:
                        integrity = await tts._verify_orpheus_part(
                            staged, text, directory / "verification" / stem, emit=emit,
                            asr_advisory=True, provider_label="Chatterbox",
                            validate_pocket_continuity=True,
                        )
                    else:
                        integrity = {"method": "duration_only_preview", "verified": True,
                                     "internal_silence": tts._validate_pocket_internal_silence(staged)}
                except tts.TtsIntegrityError as exc:
                    rejected = directory / "verification" / stem
                    rejected.mkdir(parents=True, exist_ok=True)
                    os.replace(staged, rejected / f"rejected-{attempt}.wav")
                    checkpoint["rejected"] = str(exc)
                    _write_json(request_path, checkpoint)
                    if attempt + 1 == MAX_INTEGRITY_ATTEMPTS:
                        raise
                    emit(f"Chatterbox physical audio retry for part {index}: {exc}")
                    continue
                os.replace(staged, path)
                metadata = {
                    "identity": identity, "job_id": job_id, "integrity": integrity,
                    "synthesis_request": payload,
                    "word_count": tts._spoken_word_count(text),
                    "audio_sha256": tts._file_sha256(path),
                    "wav": asdict(tts._read_pcm_wav(path)),
                    "generation_seconds": float(job.get("generation_seconds") or max(
                        0, float(job.get("finished_at") or 0) - float(job.get("started_at") or 0),
                    )),
                }
                _write_json(tts._part_metadata_path(path), metadata)
                parts.append(path)
                metadata_rows.append(metadata)
                break
    output = await tts._concat_wav_parts(parts, directory, log=log)
    integrity = tts._narration_integrity_summary(metadata_rows, tts._spoken_word_count(cleaned),
                                                required=verify_text,
                                                method="per_chunk_mlx_whisper" if verify_text else "duration_only_preview")
    if not integrity["passed"]:
        raise tts.TtsIntegrityError("Chatterbox narration has incomplete source-part coverage")
    tts._write_tts_manifest(
        directory, model=MODEL_ID, source_text=cleaned, chunks=chunks, wav_parts=parts,
        output=output, deterministic=False, integrity=integrity,
        extra={**revisions, "deployment_fingerprint": fingerprint,
               "voice": voice, "voice_fingerprint": voice_info["fingerprint"],
               "language": language, "continuity": _continuity_report(cleaned, chunks, parts, metadata_rows)},
    )
    return str(output)

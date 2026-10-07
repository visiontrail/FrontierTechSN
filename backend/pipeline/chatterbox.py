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
from collections.abc import Awaitable, Callable
from dataclasses import asdict
from difflib import SequenceMatcher
from pathlib import Path

import httpx

from backend import config
from backend.pipeline import tts, tts_pronunciation as pronunciation

MODEL_ID = "chatterbox-multilingual"
VERIFIER_VERSION = 3
TRAILING_SPEECH_VERIFIER_VERSION = 2
SYMBOL_INITIALISM_POLICY_VERSION = 1
SYMBOL_INITIALISM_RE = re.compile(r"\b([A-Z])&([A-Z])\b")
SYNTHESIS_LETTER_SEPARATOR_RE = re.compile(r"(?<=[A-Za-z])[-.]+(?=[A-Za-z])")
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
    from backend.pipeline.tts_tail_cleanup import VERSION as tail_cleanup_version

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
        "trailing_speech_verifier_version": TRAILING_SPEECH_VERIFIER_VERSION,
        "symbol_initialism_policy_version": SYMBOL_INITIALISM_POLICY_VERSION,
        "tail_cleanup_version": tail_cleanup_version,
        "shared_verifier_version": tts.POCKET_TTS_INTEGRITY_VERIFIER_VERSION,
        "synthesis_parameters": dict(SYNTHESIS_PARAMETERS),
        "pronunciation_policy": pronunciation.policy_contract(),
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


def _bind_verification(directory: Path, binding: dict) -> None:
    """Keep reports from another waveform out of the current attempt's evidence."""
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / 'observation_binding.json'
    previous = _read_json(marker)
    same_wave = previous.get('audio_sha256') == binding['audio_sha256']
    files = [p for p in directory.glob('*.json') if p != marker and
             (not same_wave or p.name in {'integrity_report.json', 'pronunciation_report.json', 'rejection.json'})]
    if files:
        identity = {'binding': previous, 'reports': {p.name: tts._file_sha256(p) for p in files}}
        archive = directory / 'observation_history' / _digest(identity)
        archive.mkdir(parents=True, exist_ok=True)
        for path in files:
            retained = archive / path.name
            if retained.exists() and retained.read_bytes() != path.read_bytes():
                raise RuntimeError('Verification history changed; preserving the current reports')
            os.replace(path, retained)
        _write_json(archive / 'provenance.json', identity)
    _write_json(marker, binding)


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


def _synthesis_word_count(text: str) -> int:
    # Model-proposed separators may join letters or their phonetic names,
    # regardless of case. Do not count a.p.i or ay-pee-eye as just one word.
    return len(SYNTHESIS_LETTER_SEPARATOR_RE.sub(" ", text).split())


def _symbol_initialisms(text: str) -> list[re.Match]:
    protected = [(match.start(), match.end()) for match in pronunciation.PROTECTED_TEXT_RE.finditer(text)]
    return [match for match in SYMBOL_INITIALISM_RE.finditer(text)
            if not any(start < match.end() and end > match.start() for start, end in protected)]


def _render_provider_span(text: str, start: int, end: int, edits: list[dict]) -> str:
    rendered = pronunciation.render_span(text, start, end, edits)
    for match in reversed(_symbol_initialisms(rendered)):
        letters = [{"A": "eigh", "N": "N"}.get(letter, pronunciation.LETTER_NAMES[letter][0])
                   for letter in match.groups()]
        rendered = rendered[:match.start()] + " and ".join(letters) + rendered[match.end():]
    return rendered


def _split_text(
    text: str, *, max_words: int, max_chars: int,
    pronunciation_edits: list[dict] | None = None,
) -> list[str]:
    """Preserve source tokens while budgeting the expanded provider input."""
    if max_words < 1 or max_chars < 20:
        raise ValueError("Chatterbox chunk limits must be positive (at least 20 characters)")
    normalized = pronunciation.normalize_text(text)
    chunks = []
    offset = 0
    for line in normalized.split("\n"):
        remaining = line
        line_offset = offset
        offset += len(line) + 1
        while remaining:
            limit = 0
            for word in re.finditer(r"\S+", remaining):
                candidate = _render_provider_span(
                    normalized, line_offset, line_offset + word.end(), pronunciation_edits or [],
                )
                if len(candidate) > max_chars or _synthesis_word_count(candidate) > max_words:
                    break
                limit = word.end()
            if not limit:
                raise ValueError("A Chatterbox input token exceeds the expanded request limit; refusing to truncate it")
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
            tail = remaining[cut:]
            line_offset += cut + len(tail) - len(tail.lstrip())
            remaining = tail.strip()
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
    # Count every spoken letter for duration sanity, including provider hints.
    duration_text = SYNTHESIS_LETTER_SEPARATOR_RE.sub(" ", text)
    tts._validate_wav_part(path, duration_text)
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
    if verify_text:
        guard = metadata["integrity"].get("pronunciation")
        if not isinstance(guard, dict) or guard.get("passed") is not True:
            return None
    try:
        info = _validate_audio(path, identity["synthesis_text"])
        if metadata.get("audio_sha256") != tts._file_sha256(path) or metadata.get("wav") != asdict(info):
            return None
    except tts.TtsIntegrityError:
        return None
    if verify_text:
        # The existing ASR observation is bound to this verified waveform hash.
        # Recheck it when upgrading the tail gate, retaining the original job
        # identity and avoiding resynthesis of accepted speech.
        tail = guard.get("trailing_speech")
        if not isinstance(tail, dict) or tail.get("version") != TRAILING_SPEECH_VERIFIER_VERSION:
            transcript = guard.get("transcript")
            if transcript:
                words = [{"text": transcript}]
            else:
                observation = metadata["integrity"].get("transcription") or {}
                stat = path.stat()
                if (observation.get("audio_size"), observation.get("audio_mtime_ns")) != (stat.st_size, stat.st_mtime_ns):
                    return None
                try:
                    words = json.loads((path.parent / "verification" / path.stem.removesuffix("_generated") / "transcript.json").read_text())
                except (OSError, ValueError):
                    return None
                if not isinstance(words, list):
                    return None
            tail = _trailing_speech_report(identity["text"], words)
        if tail.get("passed") is not True:
            return None
        guard["trailing_speech"] = tail
    return metadata


def _trailing_speech_report(text: str, words: list[dict]) -> dict:
    """Reject a clearly observed extra phrase after an exact complete source.

    Ordinary spelling/recognition differences retain their advisory policy.
    A final insertion of any word, with every source word matched
    and the exact opening retained but closing displaced, is extra speech.
    """
    expected = tts._lexical_tokens(text)
    observed, _ = tts._transcript_tokens(words)
    opcodes = SequenceMatcher(a=expected, b=observed, autojunk=False).get_opcodes()
    exact_prefix = len(opcodes) == 2 and opcodes[0] == ("equal", 0, len(expected), 0, len(expected))
    extra = observed[len(expected):] if exact_prefix and opcodes[-1][0] == "insert" else []
    rejected = bool(extra)
    return {"version": TRAILING_SPEECH_VERIFIER_VERSION,
            "required": bool(words), "passed": not rejected,
            "extra_words": extra if rejected else [],
            "policy": "exact_complete_source_with_extra_tail"}


PRONUNCIATION_TOKEN_RE = re.compile(
    r"(?:[A-Za-z]\.)+[A-Za-z]\.?(?:['’][A-Za-z]+)?|[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?"
)

OBSERVATION_SPELLING_VERSION = 6


def _match_spoken_letters(tokens: list[str], start: int, end: int, target: str) -> int | None:
    """Consume an exact source-bound spelling, retaining every observed atom.

    ASR can write a letter's English name (you/ex/are) rather than its glyph.
    An ordinary word is never shortened to a name: exited cannot become X.
    """
    aliases = {name: letter.casefold() for letter, names in pronunciation.LETTER_NAMES.items()
               for name in names if ' ' not in name}
    # English "aye" is /aɪ/, like I, not the letter A /eɪ/. Recognition can
    # write this homophone for the unchanged provider hint "eye".
    aliases['aye'] = 'i'
    pending = [(start, '')]
    visited = set()
    while pending:
        cursor, joined = pending.pop()
        if (cursor, joined) in visited:
            continue
        visited.add((cursor, joined))
        if joined == target:
            return cursor
        if cursor >= end:
            continue
        token = tokens[cursor]
        options = [(1, token)]
        if token in aliases:
            options.append((1, aliases[token]))
        if tokens[cursor:cursor + 2] in [['double', 'you'], ['double', 'u']]:
            options.append((2, 'w'))
        if token in {'double', 'triple'} and cursor + 1 < end:
            repeats = 2 if token == 'double' else 3
            following = tokens[cursor + 1]
            if tokens[cursor + 1:cursor + 3] in [['double', 'you'], ['double', 'u']]:
                options.append((3, 'w' * repeats))
            elif not (token == 'double' and following in {'you', 'u'}):
                letter = aliases.get(following, following if len(following) == 1 else '')
                if len(letter) == 1:
                    options.append((2, letter * repeats))
                elif re.fullmatch(r'[a-z]{2,4}', following) and following not in aliases:
                    # ASR can fuse the named repeat and the following glyphs:
                    # "double S H" -> "double SH". Repeat only its first
                    # glyph and retain every remaining observed glyph.
                    options.append((2, following[0] * repeats + following[1:]))
        for consumed, letters in options:
            if cursor + consumed <= end and target.startswith(joined + letters):
                pending.append((cursor + consumed, joined + letters))
    return None


def _pronunciation_report(text: str, words: list[dict], edits: list[dict] | None = None) -> dict:
    """Require source initialisms, including brand suffixes, exactly once in place.

    Join only an aligned, letter-identical ASR split (Open AI / C N B C).
    Wrong letters and additional repetitions can never become a spelling alias.
    Other words retain the shared advisory policy.
    """
    found = list(pronunciation.candidates(text) if edits is None else edits)
    symbols = _symbol_initialisms(text)
    for match in symbols:
        for group in (1, 2):
            found.append({"source": match[group], "start": match.start(group), "end": match.end(group)})
    ampersands = {match.start() + 1 for match in symbols}
    source = list(PRONUNCIATION_TOKEN_RE.finditer(text))

    def normalize(value: str) -> str:
        return value.replace(".", "").replace("’", "'").casefold()

    expected, targets, embedded, compound_suffixes, symbol_conjunctions = [], set(), {}, {}, set()
    embedded_ranges = []
    for token in source:
        if token.start() - 1 in ampersands:
            symbol_conjunctions.add(len(expected))
            expected.append("and")
        local = [candidate for candidate in found if
                 candidate["start"] < token.end() and candidate["end"] > token.start()]
        token_start = len(expected)
        cursor, expanded = token.start(), []
        for candidate in local:
            if candidate["start"] > cursor:
                prefix = normalize(text[cursor:candidate["start"]])
                expected.append(prefix)
                expanded.append(prefix)
            end = min(candidate["end"], token.end())
            # Preserve the audible plural/possessive ending as part of the
            # initialism, while ordinary brand prefixes remain advisory.
            if text[end:token.end()] in {"s", "'s", "’s"}:
                end = token.end()
            target = normalize(text[candidate["start"]:end])
            targets.add(len(expected))
            expected.append(target)
            expanded.append(target)
            cursor = end
        if cursor < token.end():
            suffix = normalize(text[cursor:token.end()])
            expected.append(suffix)
            expanded.append(suffix)
        if local and len(expanded) > 1:
            embedded[normalize(token[0])] = expanded
            embedded_ranges.append((token_start, len(expected), normalize(token[0])))
            if len(local) == 1 and local[0]["start"] > token.start():
                suffix = normalize(text[local[0]["start"]:token.end()])
                compound_suffixes.setdefault(suffix, []).append(
                    (expanded[0], expanded[1:], normalize(local[0]["source"]))
                )
    observed = []
    for word in words:
        for token in PRONUNCIATION_TOKEN_RE.finditer(str(word.get("text") or "").replace('&', ' and ')):
            value = normalize(token[0])
            expanded = embedded.get(value)
            # ASR may change the ordinary brand prefix while retaining the
            # exact capitalized initialism suffix (QbitAI -> QubitAI).
            compound = re.fullmatch(r"([A-Za-z][a-z]+)([A-Z]{2,}[0-9]*(?:s|['’]s)?)", token[0])
            if expanded is None and compound:
                prefix = normalize(compound[1])
                for source_prefix, suffix_parts, initialism in compound_suffixes.get(normalize(compound[2]), []):
                    # A new initialism hidden in the prefix remains a repeat.
                    if prefix.endswith(initialism) and not source_prefix.endswith(initialism):
                        continue
                    expanded = [prefix, *suffix_parts]
                    break
            observed.extend(expanded if expanded is not None else [value])
    # ASR can attach the first spoken letter to the exact preceding source
    # word ("observed US" -> "ObserveDU S"). Restore that boundary while
    # retaining every observed letter, including any following repetition.
    boundaries = sorted(
        {(expected[index - 1], expected[index]) for index in targets
         if index > 0 and index - 1 not in targets},
        key=lambda pair: len(pair[0]), reverse=True,
    )
    separated = []
    for token in observed:
        for prefix, initialism in boundaries:
            fragment = token[len(prefix):] if token.startswith(prefix) else ""
            if fragment and initialism.startswith(fragment):
                separated.extend([prefix, fragment])
                break
        else:
            separated.append(token)
    observed = separated
    # Collapse exact split spellings without discarding any observed letters.
    replacements = {}
    for tag, start, end, left, right in SequenceMatcher(
        a=expected, b=observed, autojunk=False,
    ).get_opcodes():
        if tag != "replace":
            continue
        # "AI R&D" may be decoded as "AIR &D". Repartition only an
        # aligned block of initialisms and the source ampersand conjunction;
        # all observed letters must match, with none omitted or repeated.
        indices = range(start, end)
        if (all(index in targets or index in symbol_conjunctions for index in indices)
                and "".join(observed[left:right]) == "".join(expected[start:end])):
            replacements[left] = (right, expected[start:end])
            continue
        cursor, aligned = left, []
        for index in range(start, end):
            if index not in targets:
                if cursor >= right:
                    break
                aligned.append(observed[cursor])
                cursor += 1
                continue
            matched = _match_spoken_letters(observed, cursor, right, expected[index])
            if matched is None:
                break
            cursor = matched
            aligned.append(expected[index])
        else:
            if cursor == right:
                replacements[left] = (right, aligned)
    collapsed, cursor = [], 0
    while cursor < len(observed):
        if cursor in replacements:
            cursor, tokens = replacements[cursor]
            collapsed.extend(tokens)
        else:
            collapsed.append(observed[cursor])
            cursor += 1
    differences = []
    target_forms = {expected[index] for index in targets} | {normalize(row["source"]) for row in found}
    for tag, start, end, left, right in SequenceMatcher(
        a=expected, b=collapsed, autojunk=False,
    ).get_opcodes():
        if tag == "equal":
            continue
        affected = any(index in targets for index in range(start, end))
        # Extra initialisms can be inserted or replace distant ordinary words,
        # even when the intended occurrence itself matched exactly.
        if tag in {"insert", "replace"}:
            inserted = collapsed[left:right]
            repeated = any(_match_spoken_letters(inserted, i, len(inserted), form) is not None
                           for i in range(len(inserted)) for form in target_forms)
            affected = affected or repeated or (
                tag == "insert" and (start in targets or start - 1 in targets)
            )
        if affected:
            differences.append({"kind": tag, "source": expected[start:end],
                                "asr": collapsed[left:right], "source_start": start})
    aligned = {}
    for tag, start, end, left, right in SequenceMatcher(a=expected, b=collapsed, autojunk=False).get_opcodes():
        if tag == 'equal':
            aligned.update({start + offset: left + offset for offset in range(end - start)})
    joins = {}
    for start, end, spelling in embedded_ranges:
        positions = [aligned.get(index) for index in range(start, end)]
        if all(index is not None for index in positions) and positions == list(range(positions[0], positions[0] + end - start)):
            joins[positions[0]] = (positions[-1] + 1, spelling)
    normalized, cursor = [], 0
    while cursor < len(collapsed):
        if cursor in joins:
            cursor, spelling = joins[cursor]
            normalized.append(spelling)
        else:
            normalized.append(collapsed[cursor])
            cursor += 1
    return {"required": bool(targets), "passed": not targets or bool(words) and not differences,
            "targets": [expected[index] for index in sorted(targets)],
            "transcript": tts._raw_transcript(words), "normalized_transcript": ' '.join(normalized),
            "observation_spelling_version": OBSERVATION_SPELLING_VERSION,
            "differences": differences,
            "policy": "exact_initialism_occurrences"}


async def _verify_pronunciation(path: Path, text: str, directory: Path, *, emit, edits: list[dict]) -> dict:
    from backend.pipeline import av_sync

    directory.mkdir(parents=True, exist_ok=True)
    # The preceding advisory check already decoded this exact WAV; reuse its
    # transcript rather than running a second recognition pass per chunk.
    words, transcription = await av_sync.ensure_word_transcript(path, directory, minimum_words=1)
    report = _pronunciation_report(text, words, edits)
    report["trailing_speech"] = _trailing_speech_report(text, words)
    normalized_tail = _trailing_speech_report(text, [{"text": report['normalized_transcript']}])
    if not normalized_tail['passed']:
        report['trailing_speech'] = normalized_tail
    if words and not report['passed'] and report['trailing_speech']['passed']:
        from backend.pipeline import chatterbox_phonemes
        acoustic = await chatterbox_phonemes.confirm(path, text, edits, directory)
        if acoustic is not None:
            report['acoustic_source_confirmation'] = acoustic
            if acoustic.get('passed') is True:
                if (acoustic.get('audio_sha256') != tts._file_sha256(path)
                        or acoustic.get('source_sha256') != hashlib.sha256(text.encode()).hexdigest()
                        or acoustic.get('conditioned_on_reference_text') is not False
                        or acoustic.get('method') != 'complete_source_phone_sequence'
                        or acoustic.get('supported') is not True):
                    raise RuntimeError('Acoustic confirmation differs from the complete source waveform')
                report.update(asr_spelling_passed=False, passed=True,
                              verification_mode='complete_source_phone_sequence')
                emit('Chatterbox spelling drift: complete source and letter phones independently confirmed')
    report["transcription"] = transcription
    _write_json(directory / "pronunciation_report.json", report)
    if not words and edits:
        raise RuntimeError("Chatterbox pronunciation verification unavailable; preserving the completed job for resume")
    if not report["trailing_speech"]["passed"]:
        raise tts.TtsIntegrityError(
            "Chatterbox unrequested trailing speech: " + json.dumps(report["trailing_speech"]["extra_words"])
        )
    if not report["passed"]:
        raise tts.TtsIntegrityError("Chatterbox initialism pronunciation mismatch: " + json.dumps(report["differences"]))
    if edits:
        emit("Chatterbox pronunciation: initialism occurrences matched without repetition")
    return report


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
    part_auditor: Callable[[Path, str, list[dict]], Awaitable[dict]] | None = None,
    audit_all_parts: bool = False,
    before_submit: Callable[[dict], Awaitable[dict]] | None = None,
    frozen_pronunciation_policy: dict | None = None,
) -> str:
    # The service can speak 23 languages. The project's existing source-word,
    # ASR and subtitle contract is currently certified for English only.
    if language != "en":
        raise ValueError("The Chatterbox video adapter currently validates English narration only")
    if audit_all_parts and (not verify_text or part_auditor is None):
        raise ValueError("Complete part auditing requires text verification and an independent part auditor")
    if not config.CHATTERBOX_TTS_API_KEY:
        raise RuntimeError("Set CHATTERBOX_TTS_API_KEY in Admin -> System -> Voice & TTS")
    contract = cache_contract()
    if contract['pronunciation_policy']['version'] not in pronunciation.SUPPORTED_POLICY_VERSIONS:
        raise RuntimeError('Pronunciation policy is unsupported; preserving submitted jobs')
    _, directory, _, cleaned = tts._prepare_tts_input(script_path, output_dir, strip_speaker_labels=True)
    if frozen_pronunciation_policy is not None:
        if not pronunciation.compatible_policy(frozen_pronunciation_policy, contract['pronunciation_policy']):
            raise RuntimeError('Frozen pronunciation contract changed; preserving submitted jobs')
        contract['pronunciation_policy'] = dict(frozen_pronunciation_policy)
    elif any(directory.glob('tts_input_part_*.request-*.json')):
        saved = _read_json(directory / 'tts_pronunciation.json').get('identity') or {}
        policy = saved.get('policy')
        if (saved.get('source_sha256') == hashlib.sha256(pronunciation.normalize_text(cleaned).encode()).hexdigest()
                and saved.get('language') == language and isinstance(policy, dict)
                and pronunciation.compatible_policy(policy, contract['pronunciation_policy'])):
            contract['pronunciation_policy'] = dict(policy)
    base_url = contract["url"]
    timeout = httpx.Timeout(max(5, config.CHATTERBOX_TTS_REQUEST_TIMEOUT), connect=15)
    async with httpx.AsyncClient(timeout=timeout, headers={"X-API-Key": config.CHATTERBOX_TTS_API_KEY}, follow_redirects=False) as client:
        response = await _request(client, "GET", f"{base_url}/v1/capabilities", emit=emit)
        capabilities = response.json()
        revisions = {
            key: contract[key] for key in ("source_revision", "model_revision", "model_variant")
        }
        if any(capabilities.get(key) != value for key, value in revisions.items()):
            raise RuntimeError("Chatterbox deployed revisions do not match the configured pinned build")
        fingerprint = contract["deployment_fingerprint"]
        if not fingerprint or capabilities.get("fingerprint") != fingerprint:
            raise RuntimeError("Chatterbox deployment fingerprint is missing or changed; verify the service and its voice references in Admin -> System")
        voices = {item["id"]: item for item in capabilities.get("voices", [])}
        voice_info = voices.get(voice)
        if not voice_info or not voice_info.get("fingerprint"):
            raise RuntimeError(f"Chatterbox voice '{voice}' has no installed reference fingerprint")
        normalized = pronunciation.normalize_text(cleaned)
        pronunciation_plan = await pronunciation.prepare_pronunciation(
            normalized, directory, policy=contract["pronunciation_policy"], emit=emit,
        )
        chunks = _split_text(cleaned, max_words=contract["chunk_words"],
                             max_chars=min(contract["max_chars"], int(capabilities["max_chars"])),
                             pronunciation_edits=pronunciation_plan["edits"])
        emit(f"Chatterbox: {len(chunks)} bounded request(s), natural 1.0x speech, voice={voice}")
        synthesis_directory = directory / "synthesis"
        synthesis_directory.mkdir(exist_ok=True)
        parts, metadata_rows = [], []
        rejected_parts = []
        coverage = []
        def save_coverage():
            if audit_all_parts:
                _write_json(directory / 'tts_audit_coverage.json', {
                    'source_sha256': hashlib.sha256(normalized.encode()).hexdigest(),
                    'plan_sha256': pronunciation_plan['plan_sha256'],
                    'expected_parts': len(chunks), 'observed_parts': len(coverage),
                    'all_parts_observed': len(coverage) == len(chunks),
                    'passed': len(coverage) == len(chunks) and not rejected_parts,
                    'parts': coverage,
                })
        save_coverage()
        source_offset = 0
        for index, text in enumerate(chunks, 1):
            start = normalized.index(text, source_offset)
            source_offset = start + len(text)
            synthesis_text = _render_provider_span(
                normalized, start, source_offset, pronunciation_plan["edits"],
            )
            chunk_edits = [{**edit, "start": edit["start"] - start, "end": edit["end"] - start}
                           for edit in pronunciation_plan["edits"]
                           if start <= edit["start"] and edit["end"] <= source_offset]
            stem = f"tts_input_part_{index:03d}"
            (directory / f"{stem}.txt").write_text(text, encoding="utf-8")
            (synthesis_directory / f"{stem}.txt").write_text(synthesis_text, encoding="utf-8")
            path = directory / f"{stem}_generated.wav"
            identity = {
                **revisions, "provider": "chatterbox", "url": base_url,
                "deployment_fingerprint": fingerprint,
                "voice": voice, "voice_fingerprint": voice_info["fingerprint"],
                "language": language, "text": text,
                "synthesis_text": synthesis_text,
                "pronunciation_policy": contract["pronunciation_policy"],
                "pronunciation_plan_sha256": pronunciation_plan["plan_sha256"],
                "seed": contract["seed"], "verifier_version": contract["verifier_version"],
                "shared_verifier_version": contract["shared_verifier_version"],
                "synthesis_speed_ratio": 1.0,
                "synthesis_parameters": contract["synthesis_parameters"],
            }
            cached = _cached_part(path, identity, verify_text=verify_text)
            if cached and part_auditor is not None:
                try:
                    audit = await part_auditor(path, text, chunk_edits)
                    if not isinstance(audit, dict) or audit.get('passed') is not True:
                        raise RuntimeError('Independent part auditor did not return confirmed acceptance')
                except tts.TtsIntegrityError as exc:
                    # A new independent audit may disprove a formerly accepted
                    # part. Preserve its original WAV and metadata, then spend
                    # only the remaining attempts of this same durable request.
                    request_path = next((p for p in directory.glob(f"{stem}.request-*.json")
                                         if _read_request_checkpoint(p).get("job_id") == cached["job_id"]), None)
                    if request_path is None:
                        raise RuntimeError("Audited cached audio has no matching durable job checkpoint") from exc
                    checkpoint = _read_request_checkpoint(request_path)
                    if checkpoint.get("identity") != _digest({"identity": identity, "payload": cached["synthesis_request"]}):
                        raise RuntimeError("Audited cached audio does not match its durable request identity") from exc
                    archive = directory / "verification" / stem / f"independent-rejected-{cached['job_id']}"
                    archive.mkdir(parents=True, exist_ok=True)
                    for original in (path, tts._part_metadata_path(path)):
                        retained = archive / original.name
                        if retained.exists() and retained.read_bytes() != original.read_bytes():
                            raise RuntimeError("Independent rejection archive changed; preserving current inputs") from exc
                        if not retained.exists():
                            retained.write_bytes(original.read_bytes())
                    checkpoint["rejected"] = "Chatterbox independent part audit: " + str(exc)
                    _write_json(request_path, checkpoint)
                    emit(f"Chatterbox part {index}: independent audit rejected cached audio; remaining bounded attempts retained")
                    cached = None
            if cached:
                emit(f"Chatterbox part {index}/{len(chunks)}: reusing hash-validated audio")
                _write_json(tts._part_metadata_path(path), cached)
                parts.append(path)
                metadata_rows.append(cached)
                coverage.append({'part': index, 'source_sha256': hashlib.sha256(text.encode()).hexdigest(),
                                 'status': 'accepted', 'audio_sha256': cached['audio_sha256']})
                save_coverage()
                continue
            for attempt in range(MAX_INTEGRITY_ATTEMPTS):
                request_path = directory / f"{stem}.request-{attempt}.json"
                payload = {
                    "text": synthesis_text, "language_id": language, "voice": voice,
                    "seed": (contract["seed"] + attempt) % (2 ** 32),
                    **contract["synthesis_parameters"],
                }
                request_identity = _digest({"identity": identity, "payload": payload})
                checkpoint = _read_request_checkpoint(request_path)
                if checkpoint.get("identity") != request_identity:
                    checkpoint = {"identity": request_identity, "idempotency_key": uuid.uuid4().hex}
                    _write_json(request_path, checkpoint)
                rechecking = bool(
                    str(checkpoint.get('rejected') or '').startswith((
                        'Chatterbox initialism pronunciation mismatch:',
                        'Required complete-source phone sequence differs from the source',
                    ))
                    and checkpoint.get('job_id')
                    and checkpoint.get('rejection_observation_version', 0) < OBSERVATION_SPELLING_VERSION
                )
                if checkpoint.get("rejected") and not rechecking:
                    if attempt + 1 == MAX_INTEGRITY_ATTEMPTS:
                        if audit_all_parts:
                            rejected_parts.append({'part': index, 'source_sha256': hashlib.sha256(text.encode()).hexdigest(),
                                                   'status': 'rejected', 'error': checkpoint['rejected']})
                            coverage.append(rejected_parts[-1])
                            save_coverage()
                            break
                        raise tts.TtsIntegrityError(f"Chatterbox part {index} exhausted audio integrity attempts")
                    continue
                if checkpoint.get("terminal_failure"):
                    if attempt + 1 == MAX_INTEGRITY_ATTEMPTS:
                        raise _service_attempts_exhausted(index, checkpoint["terminal_failure"])
                    continue
                if not checkpoint.get("job_id"):
                    if before_submit is not None:
                        window = await before_submit({'part': index, 'attempt': attempt,
                                                      'request_identity': request_identity})
                        if not isinstance(window, dict) or window.get('available') is not True:
                            raise RuntimeError('Chatterbox synthesis window is unverified; retaining the request checkpoint')
                    response = await _request(client, "POST", f"{base_url}/v1/jobs", emit=emit,
                                              json=payload, headers={"Idempotency-Key": checkpoint["idempotency_key"]})
                    checkpoint["job_id"] = response.json()["id"]
                    _write_json(request_path, checkpoint)
                job_id = checkpoint["job_id"]
                if rechecking:
                    emit(f'Chatterbox part {index}: rechecking the same completed job after spelling-observation repair')
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
                            job, job_id=job_id, text=synthesis_text, payload=payload,
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
                    verification = directory / "verification" / stem
                    _bind_verification(verification, {
                        'audio_sha256': job['audio_sha256'], 'job_id': job_id,
                        'source_sha256': hashlib.sha256(text.encode()).hexdigest(),
                        'request_identity': request_identity,
                    })
                    _validate_audio(staged, synthesis_text)
                    if verify_text:
                        integrity = await tts._verify_orpheus_part(
                            staged, text, verification, emit=emit,
                            asr_advisory=True, provider_label="Chatterbox",
                            validate_pocket_continuity=True,
                        )
                        integrity["pronunciation"] = await _verify_pronunciation(
                            staged, text, verification, emit=emit,
                            edits=chunk_edits,
                        )
                    else:
                        integrity = {"method": "duration_only_preview", "verified": True,
                                     "internal_silence": tts._validate_pocket_internal_silence(staged)}
                    if part_auditor is not None:
                        audit = await part_auditor(staged, text, chunk_edits)
                        if not isinstance(audit, dict) or audit.get('passed') is not True:
                            raise RuntimeError('Independent part auditor did not return confirmed acceptance')
                        integrity["independent_audit"] = audit
                except tts.TtsIntegrityError as exc:
                    rejected = directory / "verification" / stem
                    rejected.mkdir(parents=True, exist_ok=True)
                    marker = (f'rechecked-{attempt}-spelling-{OBSERVATION_SPELLING_VERSION}'
                              if rechecking else f'rejected-{attempt}')
                    evidence = rejected / marker
                    evidence.mkdir(exist_ok=True)
                    _write_json(rejected / 'rejection.json', {
                        **_read_json(rejected / 'observation_binding.json'),
                        'error': str(exc), 'observation_spelling_version': OBSERVATION_SPELLING_VERSION,
                    })
                    for report_path in rejected.glob("*.json"):
                        (evidence / report_path.name).write_bytes(report_path.read_bytes())
                    os.replace(staged, rejected / f"{marker}.wav")
                    checkpoint["rejected"] = str(exc)
                    checkpoint['rejection_observation_version'] = OBSERVATION_SPELLING_VERSION
                    _write_json(request_path, checkpoint)
                    if attempt + 1 == MAX_INTEGRITY_ATTEMPTS:
                        if audit_all_parts:
                            rejected_parts.append({'part': index, 'source_sha256': hashlib.sha256(text.encode()).hexdigest(),
                                                   'status': 'rejected', 'error': str(exc)})
                            coverage.append(rejected_parts[-1])
                            save_coverage()
                            break
                        raise
                    emit(f"Chatterbox audio integrity retry for part {index}: {exc}")
                    continue
                os.replace(staged, path)
                if rechecking:
                    checkpoint['revalidated_after_rejection'] = {
                        'observation_spelling_version': OBSERVATION_SPELLING_VERSION,
                        'audio_sha256': tts._file_sha256(path),
                    }
                    _write_json(request_path, checkpoint)
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
                coverage.append({'part': index, 'source_sha256': hashlib.sha256(text.encode()).hexdigest(),
                                 'status': 'accepted', 'audio_sha256': metadata['audio_sha256']})
                save_coverage()
                break
        if rejected_parts:
            raise tts.TtsIntegrityError('Chatterbox complete part audit retained integrity rejections: ' + json.dumps(rejected_parts))
    from backend.pipeline.chatterbox_tail import prepare_delivery_parts
    parts, metadata_rows = prepare_delivery_parts(parts, metadata_rows)
    output = await tts._concat_wav_parts(parts, directory, log=log)
    integrity = tts._narration_integrity_summary(metadata_rows, tts._spoken_word_count(cleaned),
                                                required=verify_text,
                                                method="per_chunk_mlx_whisper" if verify_text else "duration_only_preview")
    if not integrity["passed"]:
        raise tts.TtsIntegrityError("Chatterbox narration has incomplete source-part coverage")
    tts._write_tts_manifest(
        directory, model=MODEL_ID, source_text=cleaned, chunks=chunks, wav_parts=parts,
        output=output, deterministic=False, integrity=integrity,
        extra={**revisions, "deployment_fingerprint": fingerprint, "cache_contract": contract,
               "voice": voice, "voice_fingerprint": voice_info["fingerprint"],
               "language": language, "pronunciation_plan_sha256": pronunciation_plan["plan_sha256"],
               "continuity": _continuity_report(cleaned, chunks, parts, metadata_rows)},
    )
    return str(output)

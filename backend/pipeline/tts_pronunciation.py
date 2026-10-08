"""Contextual, source-preserving AI pronunciation preparation for Chatterbox."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from backend import config

CANDIDATE_POLICY_VERSION = 4
IMPLEMENTED_POLICY_VERSION = 7
POLICY_VERSION = IMPLEMENTED_POLICY_VERSION
SUPPORTED_POLICY_VERSIONS = (3, 4, 5, 6, 7)
MAX_PLAN_ATTEMPTS = 2
PROTECTED_TEXT_RE = re.compile(
    r"(?:https?://|www\.)\S+|\b[\w.+-]+@[\w.-]+\.\w+|\b(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}(?:/\S*)?",
    re.IGNORECASE,
)
LEGACY_CANDIDATE_RE = re.compile(r"(?:[A-Z]\.)+[A-Z]\.?|(?<![A-Z])[A-Z]{2,}(?![A-Z])")
CANDIDATE_RE = re.compile(
    LEGACY_CANDIDATE_RE.pattern + r"|(?<![A-Z])[A-Z](?=\d)|(?<=\d)[A-Z](?![A-Z])"
)
# An alphabet validates the model's proposed sounds; it never decides which
# names to change. The model selects each occurrence and its spoken spelling.
LETTER_NAMES = {
    "A": ("ay", "eigh"), "B": ("bee",), "C": ("see", "cee", "sea"),
    "D": ("dee",), "E": ("ee",), "F": ("eff", "ef"), "G": ("gee",),
    "H": ("aitch", "haitch"), "I": ("eye",), "J": ("jay",),
    "K": ("kay",), "L": ("el", "ell"), "M": ("em",), "N": ("en", "enn"),
    "O": ("oh",), "P": ("pee",), "Q": ("cue", "queue"), "R": ("ar", "are"),
    "S": ("ess", "es"), "T": ("tee",), "U": ("you",), "V": ("vee",),
    "W": ("double you",), "X": ("ex",), "Y": ("why",), "Z": ("zee", "zed"),
}


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    staged = path.with_suffix(path.suffix + ".tmp")
    staged.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(staged, path)


def prompt_path(*, policy_version: int | None = None) -> Path:
    version = IMPLEMENTED_POLICY_VERSION if policy_version is None else policy_version
    if version not in SUPPORTED_POLICY_VERSIONS:
        raise ValueError('Unsupported pronunciation prompt policy')
    return config.PROMPTS_DIR / ('tts_pronunciation.txt' if version in {3, 4} else f'tts_pronunciation_v{version}.txt')


def policy_contract(*, policy_version: int | None = None) -> dict:
    version = POLICY_VERSION if policy_version is None else policy_version
    prompt = prompt_path(policy_version=policy_version).read_text(encoding="utf-8")
    result = {
        "version": version,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "validation_alphabet_sha256": _digest(LETTER_NAMES),
    }
    if version != 3:
        result['candidate_pattern_sha256'] = hashlib.sha256(CANDIDATE_RE.pattern.encode()).hexdigest()
    if version >= 7:
        from backend.pipeline.tts_word_pronunciation import policy_contract as word_contract
        result['word_hint_contract'] = word_contract().copy()
    return result


def compatible_policy(policy: dict, current: dict | None = None) -> bool:
    """Recognize the immutable v3 candidate contract without rebinding its jobs."""
    current = current or policy_contract()
    implemented = {**current, 'version': IMPLEMENTED_POLICY_VERSION}
    if policy == implemented:
        return True
    if policy.get('version') in {3, 4, 5, 6}:
        try:
            legacy = policy_contract(policy_version=policy['version'])
        except OSError:
            return False
        return policy == legacy and policy['validation_alphabet_sha256'] == current['validation_alphabet_sha256']
    return False


def normalize_text(text: str) -> str:
    """Match canonical TTS chunk whitespace while retaining physical lines."""
    return "\n".join(" ".join(line.split()) for line in text.splitlines())


def candidates(text: str, *, policy_version: int | None = None) -> list[dict]:
    version = CANDIDATE_POLICY_VERSION if policy_version is None else policy_version
    if version not in SUPPORTED_POLICY_VERSIONS:
        raise ValueError('Unsupported pronunciation candidate policy')
    pattern = LEGACY_CANDIDATE_RE if version == 3 else CANDIDATE_RE
    protected = [(match.start(), match.end()) for match in PROTECTED_TEXT_RE.finditer(text)]
    result = []
    for match in pattern.finditer(text):
        if any(start < match.end() and end > match.start() for start, end in protected):
            continue
        result.append({"id": len(result), "start": match.start(), "end": match.end(), "source": match[0]})
    return result


def _spoken_letters(spoken: str) -> str:
    if not spoken or len(spoken) > 256 or not re.fullmatch(r"[A-Za-z .-]+", spoken):
        raise ValueError("Spoken spelling must contain only English letter names and separators")
    words = re.findall(r"[A-Za-z]+", spoken.lower())
    aliases = {name: letter for letter, names in LETTER_NAMES.items() for name in (letter.lower(), *names)}
    result = []
    index = 0
    while index < len(words):
        if words[index:index + 2] == ["double", "you"]:
            result.append("W")
            index += 2
            continue
        repeats = 1
        if words[index] in {"double", "triple"}:
            repeats = 2 if words[index] == "double" else 3
            index += 1
        if index >= len(words) or words[index] not in aliases:
            raise ValueError("Spoken spelling contains a word that is not an English letter name")
        result.append(aliases[words[index]] * repeats)
        index += 1
    return "".join(result)


def validate_decisions(text: str, decisions: object, *, policy_version: int | None = None) -> tuple[list[dict], list[dict]]:
    expected = candidates(text, policy_version=policy_version)
    if not isinstance(decisions, list) or len(decisions) != len(expected):
        raise ValueError("Return exactly one decision for every candidate occurrence")
    indexed = {}
    for row in decisions:
        if not isinstance(row, dict) or type(row.get("id")) is not int or row["id"] in indexed:
            raise ValueError("Candidate IDs must be unique integers")
        indexed[row["id"]] = row
    if set(indexed) != set(range(len(expected))):
        raise ValueError("Candidate IDs do not match the complete source candidate set")
    validated, edits = [], []
    for candidate in expected:
        row = indexed[candidate["id"]]
        if row.get("source") != candidate["source"]:
            raise ValueError("Decision source does not match its exact source occurrence")
        spoken, action, reason = row.get("spoken"), row.get("action"), row.get("reason")
        if not isinstance(reason, str) or not 1 <= len(reason.strip()) <= 600:
            raise ValueError("Each pronunciation decision requires a short contextual reason")
        if not isinstance(spoken, str) or spoken != spoken.strip():
            raise ValueError("Spoken spelling must be a trimmed string")
        if action == "keep":
            if spoken != candidate["source"]:
                raise ValueError("A keep decision must preserve its exact source spelling")
        elif action == "spell":
            letters = candidate["source"].replace(".", "")
            try:
                decoded_letters = _spoken_letters(spoken)
            except ValueError as exc:
                raise ValueError(f"Candidate {candidate['id']} ({candidate['source']}), spoken {spoken!r}: {exc}") from exc
            if decoded_letters != letters:
                raise ValueError("Spoken spelling must retain every source letter in its original order")
            if spoken == candidate["source"]:
                raise ValueError(
                    f"Candidate {candidate['id']} ({candidate['source']}), spoken {spoken!r}: "
                    "A spell decision must provide an explicit pronunciation hint; "
                    "spoken must differ from source while retaining the same letters"
                )
            # The installed voice can read "ay" as "eye", losing the A in
            # AI/OpenAI. Use the letter-identical form accepted in live audio.
            spoken = re.sub(r"\bay\b", "eigh", spoken, flags=re.IGNORECASE)
            edits.append({**candidate, "spoken": spoken})
        elif action == 'word' and (policy_version or IMPLEMENTED_POLICY_VERSION) >= 7:
            from backend.pipeline.tts_word_pronunciation import validate_word_hint
            phones = validate_word_hint(candidate['source'], spoken, row.get('phones'))
            edits.append({**candidate, 'spoken': spoken, 'kind': 'word', 'phones': phones})
        else:
            raise ValueError("Pronunciation action must be keep, spell or a supported word hint")
        accepted = {"id": candidate["id"], "source": candidate["source"],
                    "action": action, "spoken": spoken, "reason": reason.strip()}
        if action == 'word':
            accepted['phones'] = phones
        validated.append(accepted)
    return validated, edits


def render_span(text: str, start: int, end: int, edits: list[dict]) -> str:
    pieces, cursor = [], start
    for edit in edits:
        if edit["end"] <= start or edit["start"] >= end:
            continue
        if edit["start"] < start or edit["end"] > end:
            raise ValueError("A pronunciation edit cannot be split between TTS requests")
        spoken = edit["spoken"]
        # Letter names inside a brand/model need a word boundary. Keep plural
        # and possessive suffixes attached (GPUs -> gee pee yous), but prevent
        # OpenAI -> Openay eye and GPT4 -> gee pee tee4.
        if edit["start"] > 0 and text[edit["start"] - 1].isalnum():
            spoken = " " + spoken
        if edit["end"] < len(text) and text[edit["end"]].isdigit():
            spoken += " "
        pieces.extend((text[cursor:edit["start"]], spoken))
        cursor = edit["end"]
    pieces.append(text[cursor:end])
    return "".join(pieces)


def _plan(text: str, decisions: object, identity: dict) -> dict:
    version = (identity.get('policy') or {}).get('version')
    validated, edits = validate_decisions(text, decisions, policy_version=version)
    core = {"identity": identity, "decisions": validated, "edits": edits}
    synthesis = render_span(text, 0, len(text), edits)
    return {**core, "plan_sha256": _digest(core), "synthesis_text": synthesis,
            "synthesis_text_sha256": hashlib.sha256(synthesis.encode()).hexdigest()}


async def prepare_pronunciation(text: str, directory: Path, *, policy: dict, emit) -> dict:
    """Freeze one validated model plan before submitting any synthesis jobs."""
    from backend.pipeline.digester import _chat, _resolve_provider

    try:
        selected = policy_contract(policy_version=policy['version'])
    except (OSError, KeyError, ValueError) as error:
        raise RuntimeError('Pronunciation policy is unsupported; preserving submitted jobs') from error
    if policy.get('prompt_sha256') != selected['prompt_sha256']:
        raise RuntimeError('Chatterbox pronunciation prompt changed during preparation')
    if policy != selected:
        raise RuntimeError('Pronunciation policy is unsupported or changed; preserving existing plans and jobs')

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "tts_pronunciation.json"
    identity = {"source_sha256": hashlib.sha256(text.encode()).hexdigest(), "language": "en", "policy": policy}
    checkpoints = list(directory.glob("tts_input_part_*.request-*.json"))
    if path.exists():
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(saved, dict):
                raise ValueError("plan is not an object")
            if saved.get("identity") == identity:
                checked = _plan(text, saved.get("decisions"), identity)
                if any(saved.get(key) != value for key, value in checked.items()):
                    raise ValueError("plan content/hash mismatch")
                emit("Chatterbox pronunciation: reusing the source-bound AI plan")
                return saved
        except (OSError, ValueError, TypeError) as exc:
            raise RuntimeError("Chatterbox pronunciation plan is unreadable or invalid; preserving it") from exc
    if checkpoints:
        raise RuntimeError("Chatterbox pronunciation plan is missing or changed beside submitted jobs; preserving their request identities")

    prompt = prompt_path(policy_version=policy['version']).read_text(encoding="utf-8")
    if hashlib.sha256(prompt.encode()).hexdigest() != policy["prompt_sha256"]:
        raise RuntimeError("Chatterbox pronunciation prompt changed during preparation")
    found = candidates(text, policy_version=policy['version'])
    route, requested_route = {}, {}
    if not found:
        result = _plan(text, [], identity)
    else:
        endpoint, model, api_key = await _resolve_provider(None, None, None)
        requested_route = {"endpoint": endpoint, "model": model}
        payload = {"language": "en", "source_text": text, "candidates": found,
                   "letter_name_forms": LETTER_NAMES}
        # A failed run may have corrected most decisions before exhausting its
        # bounded attempts. Treat that response only as repair context, never
        # as an accepted plan, and recheck its error under the current policy.
        for previous_path in sorted(
            directory.glob("tts_pronunciation_attempt_*.json"),
            key=lambda item: item.stat().st_mtime_ns, reverse=True,
        ):
            try:
                previous = json.loads(previous_path.read_text(encoding="utf-8"))
                request = previous.get("request", {})
                raw = previous.get("response")
                if (request.get("source_text") != text or request.get("candidates") != found
                        or not previous.get("validation_error") or not isinstance(raw, str)
                        or len(raw) > 131072):
                    continue
                body = raw.strip()
                fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", body, re.DOTALL)
                decoded = json.loads(fenced[1] if fenced else body)
                if not isinstance(decoded, dict):
                    continue
            except (OSError, ValueError, TypeError, AttributeError):
                continue
            try:
                _plan(text, decoded.get("decisions"), identity)
            except (ValueError, TypeError) as exc:
                payload.update(validation_error=str(exc), previous_response=raw)
                emit("Chatterbox pronunciation: using source-matched failed response as repair context")
                break
        emit(f"Chatterbox pronunciation: AI reviewing {len(found)} occurrences in full script context")

        def selected(selected_endpoint, selected_model, _selected_key):
            route.update(endpoint=selected_endpoint, model=selected_model)

        for attempt in range(MAX_PLAN_ATTEMPTS):
            raw = await _chat(
                prompt, json.dumps(payload, ensure_ascii=False),
                endpoint=endpoint, model=model, api_key=api_key, log=emit,
                label="Chatterbox pronunciation planning", max_tokens=min(32768, max(8192, 4096 + len(found) * 180)),
                enable_skills=False, disable_thinking=True, route_selected=selected,
            )
            evidence = {"request": payload.copy(), "response": raw, "requested_route": requested_route, "route": route.copy()}
            try:
                body = raw.strip()
                fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", body, re.DOTALL)
                if fenced:
                    body = fenced[1]
                decoded = json.loads(body)
                if not isinstance(decoded, dict):
                    raise ValueError("Return a JSON object with a decisions array")
                result = _plan(text, decoded.get("decisions"), identity)
            except (ValueError, TypeError) as exc:
                evidence["validation_error"] = str(exc)
                _write_json(directory / f"tts_pronunciation_attempt_{uuid.uuid4().hex}.json", evidence)
                if attempt + 1 == MAX_PLAN_ATTEMPTS:
                    raise RuntimeError(f"Chatterbox pronunciation AI plan failed validation: {exc}") from exc
                payload["validation_error"] = str(exc)
                payload["previous_response"] = raw
                emit("Chatterbox pronunciation: retrying invalid AI output once before synthesis")
                continue
            _write_json(directory / f"tts_pronunciation_attempt_{uuid.uuid4().hex}.json", evidence)
            break
    result.update(requested_route=requested_route, route=route,
                  created_at=datetime.now(timezone.utc).isoformat())
    _write_json(path, result)
    (directory / "tts_synthesis.txt").write_text(result["synthesis_text"], encoding="utf-8")
    emit(f"Chatterbox pronunciation: saved {len(result['edits'])} contextual pronunciation edits; canonical script retained")
    return result

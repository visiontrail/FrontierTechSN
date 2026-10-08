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

CANDIDATE_POLICY_VERSION = 13
IMPLEMENTED_POLICY_VERSION = 13
POLICY_VERSION = IMPLEMENTED_POLICY_VERSION
SUPPORTED_POLICY_VERSIONS = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)
MAX_PLAN_ATTEMPTS = 2
PROTECTED_TEXT_RE = re.compile(
    r"(?:https?://|www\.)\S+|\b[\w.+-]+@[\w.-]+\.\w+|\b(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}(?:/\S*)?",
    re.IGNORECASE,
)
LEGACY_CANDIDATE_RE = re.compile(r"(?:[A-Z]\.)+[A-Z]\.?|(?<![A-Z])[A-Z]{2,}(?![A-Z])")
V4_CANDIDATE_RE = re.compile(
    LEGACY_CANDIDATE_RE.pattern + r"|(?<![A-Z])[A-Z](?=\d)|(?<=\d)[A-Z](?![A-Z])"
)
CANDIDATE_RE = re.compile(
    r"(?<![A-Za-z])[A-Z][a-z][A-Z](?![A-Za-z])"
    r"|(?<![A-Za-z])[a-z][A-Z]{2,4}(?![A-Za-z])|" + V4_CANDIDATE_RE.pattern
)
NUMERIC_MODEL_RE = re.compile(r'[A-Za-z][A-Za-z0-9-]*(?:\.[0-9]+)*')
NUMERIC_LITERAL_RE = re.compile(r'[0-9]{1,6}(?:\.[0-9]{1,6})?')
NUMERIC_CUE_VERSION = 1

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

def numeric_cue(source: str) -> str:
    """Spell a finite model numeric literal without losing zeroes or point."""
    from backend.spoken_numbers import integer_words
    if not NUMERIC_LITERAL_RE.fullmatch(source):
        raise ValueError('Unsupported exact numeric cue literal')
    digits = ('zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine')
    whole, point, fraction = source.partition('.')
    words = ([digits[int(digit)] for digit in whole] if len(whole) > 1 and whole.startswith('0')
             else integer_words(int(whole)).replace('-', ' ').split())
    return ' '.join(words + (['point', *(digits[int(digit)] for digit in fraction)] if point else []))


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
        pattern = CANDIDATE_RE if version >= 11 else V4_CANDIDATE_RE
        result['candidate_pattern_sha256'] = hashlib.sha256(pattern.pattern.encode()).hexdigest()
        if version >= 11:
            result['candidate_boundary_version'] = 1
    if version >= 7:
        from backend.pipeline.tts_word_pronunciation import policy_contract as word_contract
        result['word_hint_contract'] = word_contract(version=4 if version >= 13 else 3 if version >= 12 else 2 if version >= 10 else 1).copy()
    if version >= 14:
        result['numeric_cue_version'] = NUMERIC_CUE_VERSION
        result['numeric_candidate_pattern_sha256'] = _digest((NUMERIC_MODEL_RE.pattern, NUMERIC_LITERAL_RE.pattern))
    return result


def compatible_policy(policy: dict, current: dict | None = None) -> bool:
    """Recognize the immutable v3 candidate contract without rebinding its jobs."""
    current = current or policy_contract()
    implemented = {**current, 'version': IMPLEMENTED_POLICY_VERSION}
    if policy == implemented:
        return True
    if policy.get('version') == 14:
        return policy == policy_contract(policy_version=14)
    if policy.get('version') in {3, 4, 5, 6, 7, 8, 9, 10, 11, 12}:
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
    pattern = (LEGACY_CANDIDATE_RE if version == 3 else
               V4_CANDIDATE_RE if version < 11 else CANDIDATE_RE)
    protected = [(match.start(), match.end()) for match in PROTECTED_TEXT_RE.finditer(text)]
    result = []
    for match in pattern.finditer(text):
        if any(start < match.end() and end > match.start() for start, end in protected):
            continue
        start, end = match.span()
        if version >= 11 and match[0].isupper() and '.' not in match[0]:
            suffix = re.match(r'[a-z]+', text[end:])
            if suffix:
                if (len(suffix[0]) == 1 and suffix[0] != 's'
                        and len(match[0]) + 1 <= 5
                        and (start == 0 or not text[start - 1].isalpha())):
                    # PCIe/MSc: the final lower-case letter is part of the
                    # short abbreviation; plural s remains an inflection.
                    end += 1
                elif len(suffix[0]) > 1 and len(match[0]) >= 3:
                    from backend.pipeline.tts_word_pronunciation import lexicon
                    # AIBase/APIClient: the last capital starts the following
                    # ordinary CamelCase word, rather than naming a letter.
                    if lexicon().get((match[0][-1] + suffix[0]).casefold()):
                        end -= 1
        result.append({"id": len(result), "start": start, "end": end, "source": text[start:end]})
    if version >= 14:
        for model in NUMERIC_MODEL_RE.finditer(text):
            if not any(char.isupper() for char in model[0]) or not any(char.isdigit() for char in model[0]):
                continue
            if any(start < model.end() and end > model.start() for start, end in protected):
                continue
            if re.search(r'\d\.\d+\.\d', model[0]):
                continue
            for number in NUMERIC_LITERAL_RE.finditer(model[0]):
                start, end = model.start() + number.start(), model.start() + number.end()
                if ((start and text[start - 1].isdigit()) or (end < len(text) and text[end].isdigit())):
                    continue
                result.append({'id': 0, 'start': start, 'end': end, 'source': text[start:end], 'kind': 'number'})
        result.sort(key=lambda row: row['start'])
        for identifier, row in enumerate(result):
            row['id'] = identifier
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
        if candidate.get('kind') == 'number' and action not in {'keep', 'number'}:
            raise ValueError('Numeric candidates require keep or the exact number action')
        if action == "keep":
            if spoken != candidate["source"]:
                raise ValueError("A keep decision must preserve its exact source spelling")
        elif action == 'number' and (policy_version or IMPLEMENTED_POLICY_VERSION) >= 14:
            if candidate.get('kind') != 'number' or spoken != numeric_cue(candidate['source']):
                raise ValueError('Numeric cue must retain the exact source value, point and every precision digit')
            edits.append({**candidate, 'spoken': spoken})
        elif action == "spell":
            letters = candidate["source"].replace(".", "").upper()
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
            edit = {**candidate, "spoken": spoken}
            if ((policy_version or IMPLEMENTED_POLICY_VERSION) >= 11
                    and candidate['end'] < len(text)
                    and text[candidate['end']].isupper()):
                edit['following_word_boundary'] = True
            edits.append(edit)
        elif action == 'word' and (policy_version or IMPLEMENTED_POLICY_VERSION) >= 7:
            from backend.pipeline.tts_word_pronunciation import validate_word_hint
            try:
                phones = validate_word_hint(candidate['source'], spoken, row.get('phones'),
                                            version=4 if (policy_version or IMPLEMENTED_POLICY_VERSION) >= 13 else
                                            3 if (policy_version or IMPLEMENTED_POLICY_VERSION) >= 12 else
                                            2 if (policy_version or IMPLEMENTED_POLICY_VERSION) >= 10 else 1)
            except ValueError as exc:
                raise ValueError(
                    f"Candidate {candidate['id']} ({candidate['source']}), word hint {spoken!r}: {exc}"
                ) from exc
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
        if (edit["start"] > 0 and text[edit["start"] - 1].isalnum()
                and not (edit.get('kind') == 'number' and cursor == edit['start']
                         and pieces and pieces[-1].endswith(' '))):
            spoken = " " + spoken
        if (edit.get('following_word_boundary')
                or (edit.get('kind') == 'number' and edit['end'] < len(text) and text[edit['end']].isalpha())
                or (edit["end"] < len(text) and text[edit["end"]].isdigit())):
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


async def prepare_pronunciation(text: str, directory: Path, *, policy: dict, emit,
                                audio_feedback: dict | None = None) -> dict:
    """Freeze one validated model plan before submitting any synthesis jobs."""
    from backend.pipeline.digester import _chat, _resolve_provider

    try:
        selected = policy_contract(policy_version=policy['version'])
    except (OSError, KeyError, ValueError) as error:
        raise RuntimeError('Pronunciation policy is unsupported; preserving submitted jobs') from error
    if policy.get('prompt_sha256') != selected['prompt_sha256']:
        raise RuntimeError('Chatterbox pronunciation prompt changed during preparation; preserving submitted jobs')
    if policy != selected:
        raise RuntimeError('Pronunciation policy is unsupported or changed; preserving existing plans and jobs')

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "tts_pronunciation.json"
    identity = {"source_sha256": hashlib.sha256(text.encode()).hexdigest(), "language": "en", "policy": policy}
    if audio_feedback is not None:
        if (not isinstance(audio_feedback, dict)
                or audio_feedback.get('source_sha256') != identity['source_sha256']
                or len(json.dumps(audio_feedback, ensure_ascii=False)) > 32768):
            raise ValueError('Pronunciation audio feedback must match the complete canonical source')
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
        if policy['version'] >= 14:
            payload['exact_numeric_cue_forms'] = [
                {'id': row['id'], 'source': row['source'], 'spoken': numeric_cue(row['source'])}
                for row in found if row.get('kind') == 'number']
        if policy['version'] >= 13:
            from backend.pipeline.tts_phone_references import entries
            references = entries()
            cited = [
                {'id': row['id'], 'source': row['source'], **references[row['source'].casefold()]}
                for row in found if row['source'].casefold() in references
            ]
            if cited:
                payload['cited_word_phone_references'] = cited
        if audio_feedback is not None:
            # Audio observations are advisory planning context, never an
            # accepted decision or permission to change the canonical script.
            # Cached/submitted plans above stay frozen even with new feedback.
            payload['audio_feedback'] = audio_feedback
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

"""OpenCLI-grounded, license-ledgered news imagery acquisition.

The spoken scene is the source of truth. OpenCLI/ChatGPT chooses the named brand,
person, place, product, or event worth illustrating; OpenCLI Google News/Search
records independent discovery evidence; Wikimedia Commons supplies
the actual image together with creator and license metadata. HyperFrames only
ever receives a local path, so capture remains deterministic and offline.
"""

from __future__ import annotations

import asyncio
import hashlib
import html
import json
import logging
import math
import re
from collections.abc import Callable
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from PIL import Image, UnidentifiedImageError

from backend import config
from backend.pipeline.opencli import (
    OpenCLIError,
    first_json,
    run_opencli,
    run_opencli_with_retries,
)

logger = logging.getLogger(__name__)
LogCallback = Callable[[str], None]

WIKIMEDIA_API = "https://commons.wikimedia.org/w/api.php"
MANIFEST_VERSION = 10
QUERY_SEMANTICS_VERSION = 3
WIKIMEDIA_SEARCH_ATTEMPTS = 4
NEWS_IMAGE_MAX_PIXELS = 40_000_000
SUPPORTED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}
OPEN_LICENSE_MARKERS = (
    "public domain",
    "cc0",
    "cc by",
    "cc-by",
    "cc by-sa",
    "cc-by-sa",
)
UNSAFE_LICENSE_MARKERS = (
    "all rights reserved",
    "copyright",
    "noncommercial",
    "no derivatives",
    "proprietary",
    "-nc",
    "-nd",
)
PLAN_KINDS = {"logo", "event", "person", "place", "product", "object"}
PLAN_MODES = {"inline", "fullscreen"}
TAG_RE = re.compile(r"<[^>]+>")
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)?")
CAPITAL_PHRASE_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9'’-]*|[0-9]+[A-Z][A-Za-z0-9'’-]*)"
    r"(?:\s+(?:[A-Z][A-Za-z0-9'’-]*|[0-9]+[A-Z][A-Za-z0-9'’-]*|of|the|and)){0,5}\b"
)
STOPWORDS = frozenset(
    "a an and are as at be been but by for from has have in into is it its of on or that the "
    "their this to was were with according reports report says said today morning thanks watching "
    "good tuesday monday wednesday thursday friday saturday sunday".split()
)
GENERIC_DIRECTION_TERMS = frozenset(
    "action biotech business engineering legislative morning news open science source technology".split()
)
GENERIC_ENTITY_TERMS = frozenset(
    "agent agents article august board boards browser business company content department dimension "
    "dimensions employee employees engineer engineers event germany global harness idea ideas innovation "
    "management model office photo product products project projects report research society system systems "
    "tested testing technology thursday time tools united world english language".split()
)
GENERIC_QUERY_TERMS = GENERIC_ENTITY_TERMS | frozenset(
    "corporate official image photograph portrait logo mark launch".split()
)
ENTITY_EDGE_TERMS = frozenset("and of the for in on at by from with".split())
MONTH_TERMS = frozenset("january february march april may june july august september october november december".split())
REPORTING_VERBS_RE = re.compile(
    r"^\s+(?:also\s+)?(?:reports|reported|notes?|noted|publishes|published|says|said|writes|wrote|"
    r"describes|described)\b",
    flags=re.I,
)
BAD_PHOTO_MARKERS = ("logo", "icon", "map", "diagram", "chart", "flag")
GENERIC_LOGO_TITLE_TERMS = frozenset(
    "black brand corporate english en icon logo mark png symbol transparent white wordmark svg".split()
)
IDENTITY_CONNECTORS = frozenset("and of the".split())
IDENTITY_DECORATORS = frozenset(
    "black brand chinese company corp corporate corporation english en event file financial group icon image "
    "jpeg jpg logo mark official photo photograph png product public screenshot symbol transparent white "
    "wordmark webp zh svg".split()
)
GENERATED_ASSET_RE = re.compile(
    r"^news_images/image-[0-9]+\.(?:jpe?g|png|webp|svg)$",
    flags=re.IGNORECASE,
)


def _emit(log: LogCallback | None, message: str) -> None:
    if log:
        log(message)
    else:
        logger.info(message)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_html(value: object) -> str:
    if not value:
        return ""
    return " ".join(html.unescape(TAG_RE.sub(" ", str(value))).split())


def _metadata_value(metadata: dict, key: str) -> str:
    raw = metadata.get(key) or {}
    if isinstance(raw, dict):
        return str(raw.get("value") or "")
    return str(raw or "")


def _is_open_license(short_name: str, license_code: str = "") -> bool:
    combined = f"{short_name} {license_code}".strip().lower()
    if not combined or any(marker in combined for marker in UNSAFE_LICENSE_MARKERS):
        return False
    return any(marker in combined for marker in OPEN_LICENSE_MARKERS)


def _terms(value: object) -> set[str]:
    output: set[str] = set()
    for token in WORD_RE.findall(str(value or "").replace("-", " ").replace("–", " ")):
        normalized = token.casefold().strip("'’-")
        if normalized.endswith(("'s", "’s")):
            normalized = normalized[:-2]
        if len(normalized) >= 3 and normalized not in STOPWORDS:
            output.add(normalized)
    return output


def _semantic_terms(value: object) -> set[str]:
    """Terms used by the strict image-grounding policy.

    The general lexical matcher intentionally drops two-character words. Image
    identity cannot do that: ``AI`` and names such as ``3M`` are meaningful.
    """
    output: set[str] = set()
    for raw in WORD_RE.findall(str(value or "").replace("-", " ").replace("–", " ")):
        token = raw.casefold().strip("'’-")
        if token.endswith(("'s", "’s")):
            token = token[:-2]
        if not token or token in STOPWORDS:
            continue
        if len(token) >= 3 or token == "ai" or any(char.isdigit() for char in token):
            output.add(token)
    text = str(value or "").casefold()
    if "artificial intelligence" in text:
        output.add("ai")
    return output


def _ordered_semantic_terms(value: object) -> list[str]:
    output: list[str] = []
    for raw in WORD_RE.findall(str(value or "").replace("-", " ").replace("–", " ")):
        token = raw.casefold().strip("'’-")
        if token.endswith(("'s", "’s")):
            token = token[:-2]
        if (
            token
            and token not in STOPWORDS
            and (len(token) >= 3 or token == "ai" or any(char.isdigit() for char in token))
            and token not in output
        ):
            output.append(token)
    if "artificial intelligence" in str(value or "").casefold() and "ai" not in output:
        output.insert(0, "ai")
    return output


def _identity_tokens(value: object) -> list[str]:
    """Return ordered identity tokens without grammatical connectors.

    Grounding must preserve otherwise-generic parts of proper names (``World``
    in ``Perfect World`` and ``Office`` in ``Qwen Office``).  It therefore
    deliberately does not use the broad query stopword lists.
    """
    output: list[str] = []
    for raw in WORD_RE.findall(str(value or "").replace("-", " ").replace("–", " ")):
        token = raw.casefold().strip("'’-")
        if token.endswith(("'s", "’s")):
            token = token[:-2]
        if not token or token in IDENTITY_CONNECTORS:
            continue
        if token == "uni":
            token = "university"
        output.append(token)
    return output


def _contains_token_phrase(field_tokens: list[str], subject_tokens: list[str]) -> bool:
    width = len(subject_tokens)
    return bool(width) and any(
        field_tokens[index : index + width] == subject_tokens
        for index in range(len(field_tokens) - width + 1)
    )


def _meaningful_identity_tokens(field_tokens: list[str]) -> list[str]:
    return [
        token
        for token in field_tokens
        if token not in IDENTITY_DECORATORS
        and not (len(token) == 4 and token.isdigit())
        and (len(token) > 1 or any(char.isdigit() for char in token))
    ]


def _candidate_identity_match(shot: dict, candidate: dict) -> tuple[str, str, list[str]] | None:
    """Find the complete subject identity in one metadata field.

    Combining title and description allowed a first name in one field and a
    surname in another to masquerade as a full person match.  Multi-token
    identities must now occur contiguously in one field.  Single-token brands
    are accepted only when no second meaningful identity is present, so
    ``Qwen`` cannot validate ``Qwen Audio`` and ``Google`` cannot validate
    ``Google Loon``.
    """
    subject_tokens = _identity_tokens(shot.get("expected_subject"))
    if not subject_tokens or subject_tokens == ["ai"]:
        return None
    kind = str(shot.get("kind") or "event")
    for field_name in ("title", "description"):
        value = str(candidate.get(field_name) or "")
        field_tokens = _identity_tokens(value)
        if not _contains_token_phrase(field_tokens, subject_tokens):
            continue
        if len(subject_tokens) == 1 and _meaningful_identity_tokens(field_tokens) != subject_tokens:
            continue
        if kind == "logo" and not ({"logo", "wordmark", "mark"} & set(field_tokens)):
            continue
        if kind != "logo" and ({"logo", "wordmark", "icon"} & set(field_tokens)) and not (
            {"event", "photo", "photograph", "screenshot", "launch", "conference", "expo"}
            & set(field_tokens)
        ):
            continue
        return field_name, " ".join(field_tokens), subject_tokens
    return None


def _distinctive_terms(value: object) -> set[str]:
    return _semantic_terms(value) - GENERIC_QUERY_TERMS


def _normalise_entity_phrase(value: object) -> str:
    tokens = WORD_RE.findall(str(value or ""))
    while tokens and tokens[0].casefold() in {"the"}:
        tokens.pop(0)
    while tokens and (
        tokens[-1].casefold().strip("'’") in ENTITY_EDGE_TERMS
        or tokens[-1].casefold() in MONTH_TERMS
        or tokens[-1].isdigit()
    ):
        tokens.pop()
    if tokens and tokens[-1].casefold().endswith(("'s", "’s")):
        tokens[-1] = tokens[-1][:-2]
    return " ".join(tokens).strip()


def _source_entity_keys(text: str) -> set[str]:
    """Return proper names used as attribution, not the reported subject."""
    output: set[str] = set()
    for match in CAPITAL_PHRASE_RE.finditer(text):
        phrase = _normalise_entity_phrase(match.group(0))
        if not phrase:
            continue
        before = text[max(0, match.start() - 18) : match.start()].casefold()
        after = text[match.end() : match.end() + 28]
        if re.search(r"(?:according to|described by|reported by|published by)\s*$", before) or REPORTING_VERBS_RE.match(
            after
        ):
            output.add(phrase.casefold())
    return output


def _entity_candidates(scene: dict, hint: dict | None = None) -> list[str]:
    """Rank concrete entities from narration, never free-floating caption text."""
    text = str(scene.get("text") or "")
    sources = _source_entity_keys(text)
    keyword_terms = _semantic_terms(" ".join(str(item) for item in scene.get("keywords") or []))
    hint_terms = _semantic_terms(
        " ".join(
            [
                str((hint or {}).get("kicker") or ""),
                str((hint or {}).get("headline") or ""),
                str((hint or {}).get("body") or ""),
                *[str(item) for item in (hint or {}).get("items") or []],
            ]
        )
    )
    ranked: dict[str, tuple[int, int, str]] = {}

    def add(subject: str, *, position: int, bonus: int = 0) -> None:
        subject = _normalise_entity_phrase(subject)
        key = subject.casefold()
        terms = _semantic_terms(subject)
        if (
            not subject
            or key in sources
            or not terms
            or terms.issubset(GENERIC_ENTITY_TERMS | MONTH_TERMS)
            or not terms.issubset(_semantic_terms(text))
        ):
            return
        score = (
            bonus
            + min(4, len(terms)) * 6
            + len(terms & keyword_terms) * 7
            + len(terms & hint_terms) * 2
            + max(0, 12 - position // 45)
        )
        current = ranked.get(key)
        candidate = (score, -position, subject)
        if current is None or candidate > current:
            ranked[key] = candidate

    for match in CAPITAL_PHRASE_RE.finditer(text):
        raw = _normalise_entity_phrase(match.group(0))
        if not raw or raw.casefold() in sources:
            continue
        before = text[max(0, match.start() - 28) : match.start()].casefold()
        after = text[match.end() : match.end() + 42].casefold()
        central_bonus = 0
        if re.search(
            r"(?:authored by|including|known in chinese as|results showing that)\s*$",
            before,
        ):
            central_bonus += 34
        if re.search(r"\b(?:ranked|score|introduced|demonstrated|recorded|allocated?)\b", after):
            central_bonus += 20

        raw_tokens = WORD_RE.findall(raw)
        possessive_index = next(
            (index for index, token in enumerate(raw_tokens) if token.casefold().endswith(("'s", "’s"))),
            -1,
        )
        if 0 <= possessive_index < len(raw_tokens) - 1:
            # "Alibaba's Qwen Office": the owned product is the visual subject;
            # the owner remains a useful reserve query.
            add(
                " ".join(raw_tokens[possessive_index + 1 :]),
                position=match.start(),
                bonus=central_bonus + 48,
            )
            owner = " ".join(raw_tokens[: possessive_index + 1])
            add(owner, position=match.start(), bonus=central_bonus + 50)
        else:
            add(raw, position=match.start(), bonus=central_bonus)

        if " and " in raw.casefold() and len(raw_tokens) <= 3:
            for part in re.split(r"\s+and\s+", raw, flags=re.I):
                add(part, position=match.start(), bonus=central_bonus + 28)

        first = raw_tokens[0] if raw_tokens else ""
        if len(raw_tokens) > 1 and (first.isupper() or any(char.isdigit() for char in first)):
            add(first, position=match.start(), bonus=central_bonus - 6)
        elif (
            len(raw_tokens) == 2
            and raw_tokens[-1].casefold() in {"office", "platform", "model", "system"}
            and _distinctive_terms(first)
        ):
            add(first, position=match.start(), bonus=central_bonus - 4)

    # If a visual hint contains a real narrated entity, it may break a tie but
    # can never introduce its own all-caps label (for example AGENTS TESTED).
    for phrase in CAPITAL_PHRASE_RE.findall(
        " ".join(
            [
                str((hint or {}).get("headline") or ""),
                str((hint or {}).get("body") or ""),
                *[str(item) for item in (hint or {}).get("items") or []],
            ]
        )
    ):
        add(phrase, position=len(text), bonus=2)

    return [item[2] for item in sorted(ranked.values(), reverse=True)]


def storyboard_fingerprint(storyboard: dict) -> str:
    payload = [
        {
            "id": str(scene.get("id") or ""),
            "text": str(scene.get("text") or ""),
            "duration": round(float(scene.get("duration") or 0), 3),
        }
        for scene in storyboard.get("scenes") or []
    ]
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def acquisition_contract_fingerprint(
    *,
    storyboard_sha256: str,
    requested_count: int,
    target_count: int,
    eligible_scene_ids: list[str],
    excluded_scene_ids: set[str],
    scene_hints: dict[str, dict] | None = None,
) -> str:
    """Bind cached assets to the exact selection and grounding contract."""
    payload = {
        "storyboard_sha256": storyboard_sha256,
        "requested_count": int(requested_count),
        "target_count": int(target_count),
        "eligible_scene_ids": list(eligible_scene_ids),
        "excluded_scene_ids": sorted(str(item) for item in excluded_scene_ids),
        "query_semantics_version": QUERY_SEMANTICS_VERSION,
        "grounding_policy_version": QUERY_SEMANTICS_VERSION,
        "license_policy": "open_only",
        "scene_hints": {
            scene_id: (scene_hints or {}).get(scene_id) or {}
            for scene_id in eligible_scene_ids
        },
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def manifest_path(task_dir: Path) -> Path:
    return task_dir / "news_images" / "manifest.json"


def read_manifest(task_dir: Path) -> dict | None:
    path = manifest_path(task_dir)
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _write_manifest(task_dir: Path, manifest: dict) -> None:
    path = manifest_path(task_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _path_has_symlink_component(path: Path) -> bool:
    absolute = path.absolute()
    return any(component.is_symlink() for component in (absolute, *absolute.parents))


def _generated_asset_path(task_dir: Path, local_path: object) -> Path | None:
    relative = str(local_path or "").replace("\\", "/")
    if not GENERATED_ASSET_RE.fullmatch(relative):
        return None
    absolute_task_dir = task_dir.absolute()
    root = absolute_task_dir / "news_images"
    if _path_has_symlink_component(absolute_task_dir) or _path_has_symlink_component(root):
        return None
    expected_resolved_root = absolute_task_dir.resolve() / "news_images"
    if root.resolve() != expected_resolved_root:
        return None
    candidate = absolute_task_dir / relative
    if candidate.parent != root or candidate.is_symlink():
        return None
    return candidate


def _owned_generated_asset_path(task_dir: Path, image: dict) -> Path | None:
    """Return a stale asset only when its manifest ownership proof still matches."""
    path = _generated_asset_path(task_dir, image.get("local_path"))
    if path is None or not path.is_file() or str(image.get("id") or "") != path.stem:
        return None
    try:
        expected_bytes = int(image.get("bytes") or 0)
    except (TypeError, ValueError, OverflowError):
        return None
    expected_sha256 = str(image.get("sha256") or "")
    if expected_bytes <= 0 or path.stat().st_size != expected_bytes or not expected_sha256:
        return None
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return path if digest.hexdigest() == expected_sha256 else None


def _cached_asset_is_intact(task_dir: Path, image: dict) -> bool:
    local = _generated_asset_path(task_dir, image.get("local_path"))
    if local is None:
        return False
    if not local.is_file() or not local.stat().st_size:
        return False
    if local.suffix.casefold() == ".svg":
        return False
    try:
        expected_bytes = int(image.get("bytes") or 0)
    except (TypeError, ValueError, OverflowError):
        return False
    expected_sha256 = str(image.get("sha256") or "")
    if (
        expected_bytes <= 0
        or expected_bytes > config.NEWS_IMAGE_MAX_BYTES
        or local.stat().st_size != expected_bytes
        or not expected_sha256
    ):
        return False
    digest = hashlib.sha256()
    try:
        with local.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected_sha256:
            return False
        with Image.open(local) as decoded:
            width, height = decoded.size
            if width * height > NEWS_IMAGE_MAX_PIXELS:
                return False
            if image.get("kind") == "logo":
                if max(width, height) < 240:
                    return False
            elif width < 640 or height < 360:
                return False
            decoded.verify()
    except (Image.DecompressionBombError, OSError, UnidentifiedImageError, ValueError):
        return False
    return True


def image_grounding_is_valid(image: dict, scene: dict) -> bool:
    """Recompute policy proof instead of trusting manifest booleans."""
    if (
        image.get("grounding_policy_version") != QUERY_SEMANTICS_VERSION
        or image.get("grounding_passed") is not True
        or not _is_open_license(str(image.get("license") or ""), str(image.get("license_code") or ""))
    ):
        return False
    evidence = _grounding_evidence(image, scene, image)
    return bool(
        evidence["grounding_passed"]
        and list(image.get("grounding_distinctive_anchors") or [])
        == evidence["grounding_distinctive_anchors"]
        and list(image.get("match_terms") or []) == evidence["grounding_distinctive_anchors"]
        and str(image.get("grounding_identity_field") or "") == evidence["grounding_identity_field"]
        and str(image.get("grounding_identity_phrase") or "") == evidence["grounding_identity_phrase"]
        and list(image.get("grounding_identity_field_terms") or [])
        == evidence["grounding_identity_field_terms"]
    )


def _cached_manifest(
    task_dir: Path,
    fingerprint: str,
    contract_sha256: str,
    storyboard: dict | None = None,
    *,
    expected_requested_count: int | None = None,
    expected_target: int | None = None,
    expected_eligible_scene_ids: list[str] | None = None,
    expected_excluded_scene_ids: set[str] | None = None,
) -> dict | None:
    manifest = read_manifest(task_dir)
    if (
        storyboard is None
        or expected_requested_count is None
        or expected_target is None
        or expected_eligible_scene_ids is None
        or expected_excluded_scene_ids is None
        or not manifest
        or manifest.get("manifest_version") != MANIFEST_VERSION
        or manifest.get("storyboard_sha256") != fingerprint
        or manifest.get("query_semantics_version") != QUERY_SEMANTICS_VERSION
        or manifest.get("grounding_policy_version") != QUERY_SEMANTICS_VERSION
        or manifest.get("cache_contract_sha256") != contract_sha256
        or manifest.get("status") != "ready"
        or manifest.get("license_policy") != "open_only"
    ):
        return None
    if storyboard_fingerprint(storyboard) != fingerprint:
        return None
    images = manifest.get("images") or []
    if not isinstance(images, list) or any(not isinstance(image, dict) for image in images):
        return None
    planned = manifest.get("planned_image_count")
    requested = manifest.get("requested_image_count")
    eligible_count = manifest.get("eligible_scene_count")
    if any(type(value) is not int for value in (planned, requested, eligible_count)):
        return None
    if not images or len(images) != planned:
        return None
    modes = _placement_mode_counts(images)
    if manifest.get("placement_modes") != modes:
        return None
    if planned >= 2 and (not modes["inline"] or not modes["fullscreen"]):
        return None
    raw_eligible = manifest.get("eligible_scene_ids") or []
    raw_excluded = manifest.get("excluded_scene_ids") or []
    if not isinstance(raw_eligible, list) or not isinstance(raw_excluded, list):
        return None
    eligible = [str(item) for item in raw_eligible]
    excluded = {str(item) for item in raw_excluded}
    expected_eligible = [
        str(scene.get("id") or "")
        for scene in storyboard.get("scenes") or []
        if str(scene.get("id") or "") and str(scene.get("id") or "") not in excluded
    ]
    if (
        not eligible
        or requested != expected_requested_count
        or planned != expected_target
        or eligible != expected_eligible_scene_ids
        or excluded != {str(item) for item in expected_excluded_scene_ids}
        or eligible != expected_eligible
        or eligible_count != len(eligible)
        or planned != min(max(0, requested), len(eligible))
        or len(set(eligible)) != len(eligible)
        or excluded & set(eligible)
    ):
        return None
    scenes_by_id = {
        str(scene.get("id") or ""): scene for scene in (storyboard or {}).get("scenes") or []
    }
    seen_scenes: set[str] = set()
    seen_sources: set[str] = set()
    seen_hashes: set[str] = set()
    seen_paths: set[str] = set()
    for image in images:
        scene_id = str(image.get("scene_id") or "")
        source = str(image.get("source_page_url") or "")
        digest = str(image.get("sha256") or "")
        local_path = str(image.get("local_path") or "")
        anchors = image.get("grounding_distinctive_anchors") or []
        if (
            not scene_id
            or scene_id not in eligible
            or scene_id in excluded
            or scene_id in seen_scenes
            or not source
            or source in seen_sources
            or not digest
            or digest in seen_hashes
            or not local_path
            or local_path in seen_paths
            or image.get("display_mode") not in PLAN_MODES
            or image.get("grounding_policy_version") != QUERY_SEMANTICS_VERSION
            or image.get("grounding_passed") is not True
            or not anchors
            or not _cached_asset_is_intact(task_dir, image)
        ):
            return None
        scene = scenes_by_id.get(scene_id)
        if scene is None or not image_grounding_is_valid(image, scene):
            return None
        seen_scenes.add(scene_id)
        seen_sources.add(source)
        seen_hashes.add(digest)
        seen_paths.add(local_path)
    return manifest


def _response_text(stdout: str) -> tuple[str, str]:
    payload = first_json(stdout)
    rows = payload if isinstance(payload, list) else [payload]
    for row in rows:
        if not isinstance(row, dict):
            continue
        folded = {str(key).casefold(): value for key, value in row.items()}
        response = str(folded.get("response") or "").strip()
        if response:
            return response, str(folded.get("conversationurl") or "")
    raise OpenCLIError("ChatGPT image plan returned no assistant response")


def _sanitize_query(value: object, limit: int = 10) -> str:
    return " ".join(WORD_RE.findall(str(value or ""))[:limit]).strip()


def _ordered_terms(value: object) -> list[str]:
    output: list[str] = []
    for token in WORD_RE.findall(str(value or "").replace("-", " ").replace("–", " ")):
        normalized = token.strip("'’-")
        if normalized.casefold().endswith(("'s", "’s")):
            normalized = normalized[:-2]
        if (
            len(normalized) >= 3
            and normalized.casefold() not in STOPWORDS
            and normalized.casefold() not in {item.casefold() for item in output}
        ):
            output.append(normalized)
    return output


def _wikimedia_query_variants(shot: dict, scene: dict) -> list[str]:
    """Broaden a precise news query without broadening its narrated subject."""
    subject_text = _sanitize_query(shot.get("expected_subject"), 8)
    subject_display_terms = WORD_RE.findall(subject_text)
    subject = _ordered_semantic_terms(shot.get("expected_subject"))
    scene_words = _ordered_semantic_terms(f"{scene.get('text', '')} {' '.join(scene.get('keywords') or [])}")
    raw = [shot.get("search_query") or ""]
    if shot.get("kind") == "logo":
        raw.append(f"{subject_text} logo")
        # Only an identity-bearing leading token may shorten a product name.
        # Never turn Perfect World into the dangerously broad "World logo".
        if len(subject) > 1 and subject[0] not in GENERIC_QUERY_TERMS:
            raw.append(f"{subject_display_terms[0]} logo")
    else:
        if any(term.casefold() in {"mouse", "mice"} for term in subject + scene_words):
            raw.extend(["laboratory mouse embryo", "CRISPR laboratory mouse"])
        if any(term.casefold() in {"rocket", "launch", "falcon"} for term in subject + scene_words):
            raw.extend(["Falcon 9 rocket launch", "SpaceX Falcon 9 launch"])
        raw.extend(
            [
                " ".join(subject + ["photo"]),
                " ".join(subject + ["event"]),
            ]
        )
    output: list[str] = []
    seen: set[str] = set()
    for value in raw:
        query = _sanitize_query(value)
        if query and query.casefold() not in seen:
            output.append(query)
            seen.add(query.casefold())
    return output


def _normalise_plan(
    value: object,
    *,
    eligible_scene_ids: list[str],
    count: int,
) -> list[dict]:
    if isinstance(value, dict):
        rows = value.get("images")
    else:
        rows = value
    if not isinstance(rows, list):
        raise ValueError("News image planner did not return an images array")

    allowed = set(eligible_scene_ids)
    output: list[dict] = []
    used: set[str] = set()
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        scene_id = str(raw.get("scene_id") or raw.get("id") or "").strip()
        if scene_id not in allowed or scene_id in used:
            continue
        query = _sanitize_query(raw.get("search_query") or raw.get("query"))
        subject = _normalise_entity_phrase(raw.get("expected_subject"))[:100]
        if not query or not subject:
            continue
        kind = str(raw.get("kind") or "event").strip().lower()
        mode = str(raw.get("display_mode") or raw.get("mode") or "inline").strip().lower()
        if kind not in PLAN_KINDS:
            kind = "event"
        if mode not in PLAN_MODES:
            mode = "inline"
        output.append(
            {
                "scene_id": scene_id,
                "search_query": query,
                "news_query": _sanitize_query(raw.get("news_query") or query, 14),
                "expected_subject": subject,
                "kind": kind,
                "display_mode": mode,
                "purpose": " ".join(str(raw.get("purpose") or "").split())[:180],
                "caption": " ".join(str(raw.get("caption") or "").split())[:120],
            }
        )
        used.add(scene_id)
        if len(output) >= count:
            break

    if not output:
        raise ValueError("News image planner returned no usable scene assignments")
    _ensure_placement_mode_mix(output)
    return output


def _ensure_placement_mode_mix(rows: list[dict]) -> None:
    """Keep every multi-image result usable by both supported compositions."""
    if len(rows) < 2:
        return
    modes = {str(item.get("display_mode") or "") for item in rows}
    if "inline" not in modes:
        rows[0]["display_mode"] = "inline"
    if "fullscreen" not in modes:
        rows[-1]["display_mode"] = "fullscreen"


def _placement_mode_counts(rows: list[dict]) -> dict[str, int]:
    return {mode: sum(1 for item in rows if item.get("display_mode") == mode) for mode in ("inline", "fullscreen")}


def _is_generic_uppercase_keyword(value: object, scene: dict) -> bool:
    """Detect a direction label such as ROBOT without discarding real acronyms."""
    cleaned = " ".join(str(value or "").split())
    tokens = WORD_RE.findall(cleaned)
    terms = _terms(cleaned)
    keyword_terms = {term for keyword in scene.get("keywords") or [] for term in _terms(keyword)}
    if len(tokens) != 1 or len(terms) != 1 or cleaned != cleaned.upper() or not terms.issubset(keyword_terms):
        return False
    # A real acronym/name retained verbatim in the narration (MIT, NVIDIA) is
    # useful. An upper-cased form of an otherwise lower/title-case keyword is
    # only a generic visual-direction label and must not dominate the subject.
    return tokens[0] not in WORD_RE.findall(str(scene.get("text") or ""))


def _fallback_subject(scene: dict, hint: dict | None = None) -> str:
    entities = _entity_candidates(scene, hint)
    if entities:
        return entities[0][:100]

    hint = hint or {}
    text = str(scene.get("text") or "")
    kicker = str(hint.get("kicker") or "")
    hint_fields = [
        (10, str(hint.get("headline") or "")),
        *[(max(4, 8 - index), str(item)) for index, item in enumerate(hint.get("items") or [])],
        (6, str(hint.get("body") or "")),
    ]
    if kicker and not _is_generic_uppercase_keyword(kicker, scene):
        hint_fields.insert(0, (12, kicker))
    hint_text = " ".join(value for _, value in hint_fields)
    phrases: list[tuple[int, int, int, str]] = []
    source_phrases = {
        match.group(1).casefold()
        for match in re.finditer(
            r"(?:according to|reports?|reported by)\s+([A-Z][A-Za-z0-9'’-]*(?:\s+[A-Z][A-Za-z0-9'’-]*){0,3})",
            text,
            flags=re.I,
        )
    }
    scene_terms = _terms(text)
    hint_terms = _terms(hint_text)
    fields = [*hint_fields, (1, text)]
    ordinal = 0
    for field_weight, field_text in fields:
        for phrase in CAPITAL_PHRASE_RE.findall(field_text):
            cleaned = phrase.strip(" ,.;:—-")
            words = _terms(cleaned)
            if (
                words
                and words & scene_terms
                and not words.issubset(GENERIC_DIRECTION_TERMS)
                and cleaned.casefold() not in {"good morning", "frontier tech daily"}
                and cleaned.casefold() not in source_phrases
            ):
                # Direction fields have already compressed the narration into
                # the actual visual subject, so their priority must dominate a
                # longer incidental proper name in the raw transcript.
                score = field_weight * 100 + len(words & hint_terms) * 8 + len(words & scene_terms) * 3
                phrases.append((score, len(words), -ordinal, cleaned))
            ordinal += 1
    if phrases:
        return max(phrases)[3][:100]
    keywords = [str(value) for value in scene.get("keywords") or [] if _distinctive_terms(value)]
    return " ".join(keywords[:3])[:100] or "technology news"


def _entity_kind(subject: str, scene: dict) -> str:
    escaped = re.escape(subject).replace(r"\ ", r"\s+")
    text = str(scene.get("text") or "")
    if re.search(
        rf"(?:authored by\s+{escaped}|{escaped}\s*,[^,.]{{0,48}}\b(?:professor|editor|analyst|researcher)\b)",
        text,
        flags=re.I,
    ):
        return "person"
    return "logo"


def _fallback_shots_for_scene(
    scene: dict,
    hint: dict | None = None,
    *,
    display_mode: str = "inline",
) -> list[dict]:
    subjects = _entity_candidates(scene, hint)
    if not subjects:
        subjects = [_fallback_subject(scene, hint)]
    output: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for subject in subjects[:5]:
        kind = _entity_kind(subject, scene)
        if kind == "logo":
            query = f"{subject} logo"
        elif kind == "person":
            query = f"{subject} portrait"
        else:
            query = f"{subject} photo"
        signature = (subject.casefold(), query.casefold())
        if signature in seen or not _distinctive_terms(subject):
            continue
        seen.add(signature)
        output.append(
            {
                "scene_id": str(scene.get("id") or ""),
                "search_query": _sanitize_query(query),
                "news_query": _sanitize_query(subject, 14),
                "expected_subject": subject[:100],
                "kind": kind,
                "display_mode": display_mode,
                "purpose": "Deterministic visual derived from a named narration entity",
                "caption": subject[:120],
            }
        )
    return output


def _subject_identity_key(shot: dict) -> str:
    ordered = [
        term
        for term in _ordered_semantic_terms(shot.get("expected_subject"))
        if term not in GENERIC_QUERY_TERMS and term != "ai"
    ]
    return ordered[0] if ordered else ""


def _allocate_unique_scene_shots(
    scenes: list[dict],
    preferred_by_scene: dict[str, dict] | None = None,
    scene_hints: dict[str, dict] | None = None,
) -> list[dict]:
    """Assign scarce entities first so related scenes do not all claim one logo."""
    preferred_by_scene = preferred_by_scene or {}
    options_by_scene: dict[str, list[dict]] = {}
    for index, scene in enumerate(scenes):
        scene_id = str(scene.get("id") or "")
        preferred = preferred_by_scene.get(scene_id)
        display_mode = str((preferred or {}).get("display_mode") or ("fullscreen" if index % 3 == 2 else "inline"))
        options: list[dict] = []
        if preferred and _shot_is_grounded_to_scene(preferred, scene):
            options.append(dict(preferred))
        options.extend(
            _fallback_shots_for_scene(
                scene,
                (scene_hints or {}).get(scene_id) or {},
                display_mode=display_mode,
            )
        )
        deduped: list[dict] = []
        signatures: set[tuple[str, str, str]] = set()
        for option in options:
            signature = _shot_signature(option)
            if signature not in signatures:
                signatures.add(signature)
                deduped.append(option)
        options_by_scene[scene_id] = deduped

    ordered_scenes = sorted(
        scenes,
        key=lambda scene: (
            len(
                {
                    _subject_identity_key(option)
                    for option in options_by_scene.get(str(scene.get("id") or ""), [])
                    if _subject_identity_key(option)
                }
            ),
            next(index for index, candidate in enumerate(scenes) if candidate is scene),
        ),
    )
    used_identities: set[str] = set()
    selected: dict[str, dict] = {}
    for scene in ordered_scenes:
        scene_id = str(scene.get("id") or "")
        options = options_by_scene.get(scene_id) or []
        chosen = next(
            (
                option
                for option in options
                if _subject_identity_key(option) and _subject_identity_key(option) not in used_identities
            ),
            options[0] if options else None,
        )
        if chosen is None:
            continue
        selected[scene_id] = chosen
        identity = _subject_identity_key(chosen)
        if identity:
            used_identities.add(identity)
    return [selected[str(scene.get("id") or "")] for scene in scenes if str(scene.get("id") or "") in selected]


def _fallback_plan(scenes: list[dict], count: int, scene_hints: dict[str, dict] | None = None) -> list[dict]:
    selected_scenes = scenes[:count]
    output = _allocate_unique_scene_shots(
        selected_scenes,
        scene_hints=scene_hints,
    )
    _ensure_placement_mode_mix(output)
    return output


def _shot_signature(shot: dict) -> tuple[str, str, str]:
    return (
        str(shot.get("expected_subject") or "").casefold(),
        str(shot.get("search_query") or "").casefold(),
        str(shot.get("kind") or "").casefold(),
    )


def _reserve_plan(
    primary: list[dict],
    scenes: list[dict],
    scene_hints: dict[str, dict] | None = None,
    *,
    per_scene: int = 2,
) -> list[dict]:
    """Create alternate entity queries even when every eligible scene is primary."""
    primary_by_scene = {str(shot.get("scene_id") or ""): shot for shot in primary}
    primary_identities = {identity for shot in primary if (identity := _subject_identity_key(shot))}
    output: list[dict] = []
    for scene in scenes:
        scene_id = str(scene.get("id") or "")
        primary_shot = primary_by_scene.get(scene_id)
        planned_mode = str((primary_shot or {}).get("display_mode") or "inline")
        variants = _fallback_shots_for_scene(
            scene,
            (scene_hints or {}).get(scene_id) or {},
            display_mode=planned_mode,
        )
        own_identity = _subject_identity_key(primary_shot or {})
        variants.sort(key=lambda variant: (_subject_identity_key(variant) in (primary_identities - {own_identity}),))
        used = {_shot_signature(primary_shot)} if primary_shot else set()
        added = 0
        for variant in variants:
            signature = _shot_signature(variant)
            if signature in used:
                continue
            used.add(signature)
            output.append(
                {
                    **variant,
                    "reserve_reason": (
                        "alternate narrated entity for failed primary" if primary_shot else "unused eligible scene"
                    ),
                }
            )
            added += 1
            if added >= per_scene:
                break
    return output


def _complete_plan(
    plan: list[dict],
    scenes: list[dict],
    target: int,
    scene_hints: dict[str, dict] | None = None,
) -> list[dict]:
    """Preserve usable planned rows and deterministically fill unused scenes."""
    scenes_by_id = {str(scene.get("id") or ""): scene for scene in scenes}
    output: list[dict] = []
    used: set[str] = set()
    for row in plan:
        scene_id = str(row.get("scene_id") or "")
        if scene_id not in scenes_by_id or scene_id in used:
            continue
        output.append(dict(row))
        used.add(scene_id)
        if len(output) >= target:
            break

    remaining = [scene for scene in scenes if str(scene.get("id") or "") not in used]
    output.extend(_fallback_plan(remaining, max(0, target - len(output)), scene_hints))
    _ensure_placement_mode_mix(output)
    return output[:target]


def _shot_is_grounded_to_scene(shot: dict, scene: dict) -> bool:
    subject = _normalise_entity_phrase(shot.get("expected_subject"))
    subject_terms = _semantic_terms(subject)
    identity_terms = _identity_tokens(subject)
    scene_text = str(scene.get("text") or "")
    if (
        not subject_terms
        or not _distinctive_terms(subject)
        or not _contains_token_phrase(_identity_tokens(scene_text), identity_terms)
        or subject.casefold() in _source_entity_keys(scene_text)
    ):
        return False
    return True


def _prepare_primary_plan(
    plan: list[dict],
    scenes: list[dict],
    target: int,
    scene_hints: dict[str, dict] | None = None,
) -> list[dict]:
    """Replace hallucinated or generic planner rows with narration entities."""
    scenes_by_id = {str(scene.get("id") or ""): scene for scene in scenes}
    completed = _complete_plan(plan, scenes, target, scene_hints)
    preferred: dict[str, dict] = {}
    selected_scenes: list[dict] = []
    for shot in completed:
        scene_id = str(shot.get("scene_id") or "")
        scene = scenes_by_id.get(scene_id)
        if scene is None:
            continue
        selected_scenes.append(scene)
        normalized = {
            **shot,
            "expected_subject": _normalise_entity_phrase(shot.get("expected_subject")),
        }
        if _shot_is_grounded_to_scene(normalized, scene):
            preferred[scene_id] = normalized
            continue
        replacements = _fallback_shots_for_scene(
            scene,
            (scene_hints or {}).get(scene_id) or {},
            display_mode=str(shot.get("display_mode") or "inline"),
        )
        if replacements:
            preferred[scene_id] = replacements[0]
    output = _allocate_unique_scene_shots(
        selected_scenes,
        preferred,
        scene_hints,
    )
    _ensure_placement_mode_mix(output)
    return output[:target]


def _planner_prompt(scenes: list[dict], count: int, scene_hints: dict[str, dict] | None = None) -> str:
    payload = [
        {
            "id": scene.get("id"),
            "seconds": round(float(scene.get("duration") or 0), 2),
            "narration": scene.get("text"),
            "keywords": scene.get("keywords") or [],
            "visual_direction": (scene_hints or {}).get(str(scene.get("id") or ""), {}),
        }
        for scene in scenes
    ]
    return (
        "You are the picture editor for a factual technology-news video. Choose exactly "
        f"{count} different eligible scenes that deserve a real still image. Prefer an exact "
        "brand logo when a named brand is central, and a concrete event/person/place/product photo "
        "when the narration describes one. Do not invent a subject. Return JSON only with shape "
        '{"images":[{"scene_id":"scene-02","search_query":"NVIDIA logo",'
        '"news_query":"NVIDIA Vera CPU","expected_subject":"NVIDIA","kind":"logo",'
        '"display_mode":"inline","purpose":"...","caption":"..."}]}. '
        "kind must be logo, event, person, place, product, or object. display_mode must be inline "
        "or fullscreen. When two or more scenes are selected, include at least one of each mode. "
        "Queries must be concrete English phrases suitable for Wikimedia Commons. A source name "
        "such as a news publication is not the visual subject unless the narration is actually "
        "about that publication. Prefer the product, company, event, person, or place being "
        "reported. Use only the "
        "scene ids supplied here:\n" + json.dumps(payload, ensure_ascii=False)
    )


async def plan_news_images(
    storyboard: dict,
    *,
    eligible_scene_ids: list[str],
    count: int,
    scene_hints: dict[str, dict] | None = None,
    log: LogCallback | None = None,
) -> tuple[list[dict], str, str]:
    scenes_by_id = {str(scene.get("id") or ""): scene for scene in storyboard.get("scenes") or []}
    scenes = [scenes_by_id[scene_id] for scene_id in eligible_scene_ids if scene_id in scenes_by_id]
    target = min(max(0, count), len(scenes))
    if not target:
        return [], "none", ""

    try:
        result = await run_opencli_with_retries(
            [
                "chatgpt",
                "ask",
                _planner_prompt(scenes, target, scene_hints),
                "--new",
                "true",
                "--wait",
                "true",
                "--timeout",
                str(config.NEWS_IMAGE_OPENCLI_TIMEOUT),
                "--window",
                "background",
                "--site-session",
                "ephemeral",
                "--keep-tab",
                "false",
                "-f",
                "json",
            ],
            timeout=config.NEWS_IMAGE_OPENCLI_TIMEOUT + 45,
            attempts=config.NEWS_IMAGE_OPENCLI_ATTEMPTS,
            log=log,
            label="OpenCLI news-image planning",
        )
        response, conversation_url = _response_text(result.stdout)
        planned = _normalise_plan(first_json(response), eligible_scene_ids=eligible_scene_ids, count=target)
        plan = _complete_plan(planned, scenes, target, scene_hints)
        filled = len(plan) - len(planned)
        if filled:
            _emit(
                log,
                f"News images: OpenCLI planned {len(planned)}/{target} usable assignment(s); "
                f"filled {filled} from unused eligible scenes",
            )
            planner = "opencli:chatgpt-picture-editor+deterministic-fill"
        else:
            _emit(
                log,
                f"News images: OpenCLI planned {len(plan)} exact scene assignment(s)",
            )
            planner = "opencli:chatgpt-picture-editor"
        return plan, planner, conversation_url
    except Exception as exc:  # noqa: BLE001 - deterministic planning remains available
        _emit(log, f"News images: OpenCLI planning fallback ({exc})")
        return _fallback_plan(scenes, target, scene_hints), "deterministic-fallback", ""


def _reference_rows(value: object) -> list[dict]:
    rows = value if isinstance(value, list) else [value]
    output = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        folded = {str(key).casefold(): value for key, value in row.items()}
        url = str(folded.get("url") or "").strip()
        title = str(folded.get("title") or "").strip()
        if url and title:
            output.append(
                {
                    "title": title[:240],
                    "url": url,
                    "source": str(folded.get("source") or "")[:100],
                    "date": str(folded.get("date") or "")[:100],
                }
            )
    return output[:4]


async def research_references(shot: dict) -> tuple[list[dict], str]:
    if shot.get("kind") == "logo":
        args = [
            "google",
            "search",
            f"{shot['expected_subject']} official logo Wikimedia Commons",
            "--limit",
            "6",
            "--lang",
            "en",
            "--window",
            "background",
            "--site-session",
            "ephemeral",
            "--keep-tab",
            "false",
            "-f",
            "json",
        ]
        provider = "Google Search via OpenCLI"
    else:
        args = [
            "google",
            "news",
            shot.get("news_query") or shot["search_query"],
            "--limit",
            "6",
            "--lang",
            "en",
            "--region",
            "US",
            "-f",
            "json",
        ]
        provider = "Google News via OpenCLI"
    result = await run_opencli(args, timeout=config.NEWS_IMAGE_SEARCH_TIMEOUT)
    return _reference_rows(first_json(result.stdout)), provider


def _grounding_evidence(shot: dict, scene: dict, candidate: dict) -> dict:
    """Require the complete subject identity in narration and one candidate field."""
    subject_terms = _identity_tokens(shot.get("expected_subject"))
    scene_text = str(scene.get("text") or "")
    scene_terms = _semantic_terms(
        f"{scene.get('text', '')} {' '.join(str(item) for item in scene.get('keywords') or [])}"
    )
    candidate_terms = _semantic_terms(
        f"{candidate.get('title', '')} {candidate.get('description', '')} {candidate.get('attribution', '')}"
    )
    scene_has_identity = bool(subject_terms) and _contains_token_phrase(
        _identity_tokens(scene_text), subject_terms
    )
    candidate_match = _candidate_identity_match(shot, candidate)
    strong_subject_terms = [
        term for term in subject_terms if term != "ai" and term not in GENERIC_QUERY_TERMS
    ]
    passed = True
    reason = "complete subject identity matched narration and one candidate metadata field"
    if not subject_terms or not strong_subject_terms:
        passed = False
        reason = "shot subject had no usable complete identity"
    elif not scene_has_identity:
        passed = False
        reason = "shot subject was not a contiguous identity in the narrated scene"
    elif candidate_match is None:
        passed = False
        reason = "candidate did not contain the complete identity in one metadata field"

    identity_field = candidate_match[0] if candidate_match else ""
    identity_phrase = " ".join(subject_terms) if candidate_match else ""
    identity_field_terms = candidate_match[1].split() if candidate_match else []
    anchors = list(subject_terms) if passed else []

    return {
        "grounding_policy_version": QUERY_SEMANTICS_VERSION,
        "grounding_passed": passed,
        "grounding_distinctive_anchors": anchors,
        "grounding_reason": reason,
        "grounding_identity_field": identity_field,
        "grounding_identity_phrase": identity_phrase,
        "grounding_identity_field_terms": identity_field_terms,
        "grounding_subject_terms": subject_terms,
        "grounding_scene_terms": sorted(scene_terms),
        "grounding_candidate_terms": sorted(candidate_terms),
    }


def _embedded_scene(shot: dict) -> dict:
    return {
        "text": str(shot.get("_scene_text") or ""),
        "keywords": list(shot.get("_scene_keywords") or []),
    }


def _candidate_from_page(page: dict, shot: dict) -> dict | None:
    info = (page.get("imageinfo") or [None])[0]
    if not isinstance(info, dict):
        return None
    mime = str(info.get("thumbmime") or info.get("mime") or "").lower()
    if mime not in SUPPORTED_MIME_TYPES:
        return None
    metadata = info.get("extmetadata") or {}
    license_name = _clean_html(_metadata_value(metadata, "LicenseShortName"))
    license_code = _clean_html(_metadata_value(metadata, "License"))
    if not _is_open_license(license_name, license_code):
        return None

    download_url = str(info.get("thumburl") or info.get("url") or "")
    source_page_url = str(info.get("descriptionurl") or "")
    if not download_url or not source_page_url:
        return None
    width = int(info.get("thumbwidth") or info.get("width") or 0)
    height = int(info.get("thumbheight") or info.get("height") or 0)
    if shot.get("kind") != "logo" and (width < 640 or height < 360):
        return None
    if shot.get("kind") == "logo" and max(width, height) < 240:
        return None

    candidate = {
        "provider": "Wikimedia Commons",
        "provider_id": "wikimedia",
        "title": str(page.get("title") or "").removeprefix("File:"),
        "source_page_url": source_page_url,
        "download_url": download_url,
        "creator": _clean_html(_metadata_value(metadata, "Artist")) or "Unknown creator",
        "license": license_name or license_code,
        "license_code": license_code,
        "license_url": _clean_html(_metadata_value(metadata, "LicenseUrl")),
        "attribution": _clean_html(_metadata_value(metadata, "Credit")),
        "description": _clean_html(_metadata_value(metadata, "ImageDescription")),
        "width": width,
        "height": height,
        "mime_type": mime,
        "kind": shot.get("kind"),
    }
    evidence = _grounding_evidence(shot, _embedded_scene(shot), candidate)
    if not evidence["grounding_passed"]:
        return None
    return {**candidate, **evidence}


def _candidate_rank(candidate: dict, shot: dict) -> tuple:
    title = candidate["title"].casefold()
    description = candidate.get("description", "").casefold()
    haystack = f"{title} {description}"
    subject_terms = _terms(shot.get("expected_subject"))
    query_terms = _terms(shot.get("search_query"))
    title_terms = _terms(title)
    description_terms = _terms(description)
    subject_title_overlap = len(subject_terms & title_terms)
    query_title_overlap = len(query_terms & title_terms)
    subject_description_overlap = len(subject_terms & description_terms)
    query_description_overlap = len(query_terms & description_terms)
    score = (
        subject_title_overlap * 16
        + query_title_overlap * 7
        + subject_description_overlap * 3
        + query_description_overlap
    )
    if shot.get("kind") == "logo":
        score += 12 if "logo" in haystack else -8
        score += 3 if candidate["mime_type"] in {"image/svg+xml", "image/png"} else 0
        # Prefer the corporate mark itself over a logo for a related product.
        # Commons search can otherwise rank a large "Nvidia Shield" image above
        # the exact NVIDIA wordmark simply because it has more pixels.
        extra_title_terms = title_terms - subject_terms - GENERIC_LOGO_TITLE_TERMS
        score -= len(extra_title_terms) * 12
        if not extra_title_terms:
            score += 18
        ordered_subject = " ".join(_ordered_terms(shot.get("expected_subject"))).casefold()
        normalized_title = " ".join(_ordered_terms(candidate.get("title"))).casefold()
        if ordered_subject and normalized_title.startswith(ordered_subject):
            score += 10
        if "en" in title_terms or "english" in title_terms:
            score += 2
        if "zh" in title_terms or "chinese" in title_terms:
            score -= 2
    elif any(marker in haystack for marker in BAD_PHOTO_MARKERS):
        score -= 6
    width, height = candidate["width"], candidate["height"]
    ratio = width / max(1, height)
    if shot.get("display_mode") == "fullscreen":
        score += max(0, 5 - abs(ratio - 16 / 9) * 3)
    pixels = width * height
    return (-score, -pixels, candidate["source_page_url"])


def _wikimedia_retry_delay(response: httpx.Response, attempt: int) -> float:
    fallback = min(10.0, 1.25 * 2**attempt)
    raw = str(response.headers.get("retry-after") or "").strip()
    if not raw:
        return fallback
    try:
        seconds = float(raw)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(raw)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            seconds = (retry_at - datetime.now(timezone.utc)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return fallback
    if not math.isfinite(seconds):
        return fallback
    return min(30.0, max(0.5, seconds))


async def search_wikimedia_images(
    client: httpx.AsyncClient,
    *,
    shot: dict,
    limit: int = 30,
) -> list[dict]:
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": shot["search_query"],
        "gsrnamespace": 6,
        "gsrlimit": limit,
        "prop": "imageinfo",
        "iiprop": "url|extmetadata|mime|size",
        "iiurlwidth": 1600,
        "format": "json",
        "formatversion": 2,
    }
    response = None
    for attempt in range(WIKIMEDIA_SEARCH_ATTEMPTS):
        response = await client.get(WIKIMEDIA_API, params=params)
        if response.status_code != 429:
            break
        if attempt + 1 < WIKIMEDIA_SEARCH_ATTEMPTS:
            await asyncio.sleep(_wikimedia_retry_delay(response, attempt))
    assert response is not None
    response.raise_for_status()
    pages = (response.json().get("query") or {}).get("pages") or []
    candidates = [candidate for page in pages if (candidate := _candidate_from_page(page, shot)) is not None]
    return sorted(candidates, key=lambda candidate: _candidate_rank(candidate, shot))


def _extension_for(candidate: dict) -> str:
    # Commons serves SVG thumbnails as raster PNGs at URLs ending in
    # ``.svg.png``. Trust the actual download resource before the original
    # asset MIME type; otherwise Chromium receives PNG bytes from a file named
    # ``.svg`` and renders a blank image.
    suffix = Path(urlsplit(candidate.get("download_url", "")).path).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".svg"}:
        return suffix
    mime = candidate.get("mime_type")
    if mime == "image/jpeg":
        return ".jpg"
    if mime == "image/png":
        return ".png"
    if mime == "image/webp":
        return ".webp"
    if mime == "image/svg+xml":
        return ".svg"
    return ".jpg"


async def _download_candidate(
    client: httpx.AsyncClient,
    *,
    candidate: dict,
    destination: Path,
) -> tuple[int, str]:
    digest = hashlib.sha256()
    written = 0
    async with client.stream("GET", candidate["download_url"]) as response:
        response.raise_for_status()
        advertised = int(response.headers.get("content-length") or 0)
        if advertised and advertised > config.NEWS_IMAGE_MAX_BYTES:
            raise ValueError(f"remote image is {advertised} bytes, above the download limit")
        with destination.open("wb") as handle:
            async for chunk in response.aiter_bytes():
                written += len(chunk)
                if written > config.NEWS_IMAGE_MAX_BYTES:
                    raise ValueError("download exceeded the autonomous image byte limit")
                digest.update(chunk)
                handle.write(chunk)
    if written < 1024:
        raise ValueError("downloaded image is unexpectedly small")
    return written, digest.hexdigest()


def _public_value(value: object) -> object:
    """Strip internal context and secret-shaped keys before persistence."""
    if isinstance(value, dict):
        output: dict = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            folded = key.casefold()
            if key.startswith("_") or any(
                marker in folded
                for marker in ("api_key", "apikey", "authorization", "cookie", "password", "secret", "token")
            ):
                continue
            output[key] = _public_value(item)
        return output
    if isinstance(value, list):
        return [_public_value(item) for item in value]
    return value


def _maximum_source_matching(
    candidate_pools: dict[str, list[dict]],
    scene_order: list[str],
    *,
    excluded_sources: set[str] | None = None,
    source_hashes: dict[str, str] | None = None,
) -> dict[str, dict]:
    """Return a deterministic maximum-cardinality scene/source assignment."""
    excluded = excluded_sources or set()
    priority = {scene_id: index for index, scene_id in enumerate(scene_order)}
    usable = {
        scene_id: [
            candidate
            for candidate in candidate_pools.get(scene_id, [])
            if str(candidate.get("source_page_url") or "") not in excluded
        ]
        for scene_id in scene_order
    }
    hashes = source_hashes or {}
    resource_to_scene: dict[str, str] = {}
    scene_to_candidate: dict[str, dict] = {}

    def augment(scene_id: str, visited_sources: set[str]) -> bool:
        for candidate in usable.get(scene_id, []):
            source = str(candidate.get("source_page_url") or "")
            resource = f"sha256:{hashes[source]}" if source in hashes else f"source:{source}"
            if not source or resource in visited_sources:
                continue
            visited_sources.add(resource)
            owner = resource_to_scene.get(resource)
            if owner is None or augment(owner, visited_sources):
                resource_to_scene[resource] = scene_id
                scene_to_candidate[scene_id] = candidate
                return True
        return False

    # Scarce scenes enter first. Within an equal-size pool, earlier requested
    # scenes keep the contested source while later scenes look for alternates.
    for scene_id in sorted(
        scene_order,
        key=lambda item: (len(usable.get(item, [])), priority[item]),
    ):
        if usable.get(scene_id):
            augment(scene_id, set())
    return scene_to_candidate


def _conflict_component_scenes(
    candidate_pools: dict[str, list[dict]],
    matching: dict[str, dict],
    desired_scenes: list[str],
    *,
    source_hashes: dict[str, str] | None = None,
) -> set[str]:
    """Find scenes connected to an unmatched desired scene by shared sources."""
    hashes = source_hashes or {}
    resource_scenes: dict[str, set[str]] = {}
    for scene_id, candidates in candidate_pools.items():
        for candidate in candidates:
            source = str(candidate.get("source_page_url") or "")
            if source:
                resource = f"sha256:{hashes[source]}" if source in hashes else f"source:{source}"
                resource_scenes.setdefault(resource, set()).add(scene_id)
    pending = [scene_id for scene_id in desired_scenes if scene_id not in matching]
    connected = set(pending)
    while pending:
        scene_id = pending.pop()
        for candidate in candidate_pools.get(scene_id, []):
            source = str(candidate.get("source_page_url") or "")
            resource = f"sha256:{hashes[source]}" if source in hashes else f"source:{source}"
            for neighbour in resource_scenes.get(resource, set()):
                if neighbour not in connected:
                    connected.add(neighbour)
                    pending.append(neighbour)
    return connected


async def _discover_shot_candidates(
    client: httpx.AsyncClient,
    *,
    shot: dict,
    scene: dict,
    manifest: dict,
    position: int,
    total: int,
    log: LogCallback | None,
) -> list[dict]:
    scene_id = str(shot.get("scene_id") or "")
    if not _shot_is_grounded_to_scene(shot, scene):
        manifest["errors"].append(
            {
                "scene_id": scene_id,
                "stage": "query_grounding",
                "message": "Image query subject was not a distinctive narrated entity",
            }
        )
        return []
    _emit(
        log,
        f"News image candidate {position}/{total}: researching "
        f"{shot['expected_subject']} for {scene_id} ({shot['display_mode']})",
    )
    references: list[dict] = []
    reference_provider = ""
    try:
        references, reference_provider = await research_references(shot)
    except Exception as exc:  # noqa: BLE001 - Commons remains usable
        manifest["errors"].append(
            {
                "scene_id": scene_id,
                "stage": "opencli_search",
                "message": str(exc),
            }
        )

    candidates_by_source: dict[str, dict] = {}
    try:
        for query in _wikimedia_query_variants(shot, scene):
            lookup = {
                **shot,
                "search_query": query,
                "_scene_text": scene.get("text") or "",
                "_scene_keywords": scene.get("keywords") or [],
            }
            for raw_candidate in await search_wikimedia_images(client, shot=lookup):
                evidence = _grounding_evidence(shot, scene, raw_candidate)
                if not evidence["grounding_passed"]:
                    continue
                source = str(raw_candidate.get("source_page_url") or "")
                if not source:
                    continue
                candidate = {
                    **dict(_public_value(raw_candidate)),
                    **dict(_public_value(shot)),
                    **evidence,
                    "reference_provider": reference_provider,
                    "references": _public_value(references),
                    "resolved_search_query": query,
                }
                candidates_by_source.setdefault(source, candidate)
            if len(candidates_by_source) >= 8:
                break
    except Exception as exc:  # noqa: BLE001 - record exact failed scene
        manifest["errors"].append(
            {
                "scene_id": scene_id,
                "stage": "wikimedia_search",
                "message": str(exc),
            }
        )
        return []
    return sorted(
        candidates_by_source.values(),
        key=lambda candidate: _candidate_rank(candidate, shot),
    )


def _next_asset_destination(image_dir: Path, start: int, extension: str) -> tuple[int, Path]:
    serial = max(1, start)
    while any((image_dir / f"image-{serial:02d}{suffix}").exists() for suffix in (".jpg", ".jpeg", ".png", ".webp", ".svg")):
        serial += 1
    return serial, image_dir / f"image-{serial:02d}{extension}"


async def acquire_news_images(
    storyboard: dict,
    task_dir: Path,
    *,
    count: int,
    excluded_scene_ids: set[str] | None = None,
    scene_hints: dict[str, dict] | None = None,
    log: LogCallback | None = None,
) -> dict:
    """Research, acquire, and ledger still images for exact narrated scenes."""
    if _path_has_symlink_component(task_dir) or _path_has_symlink_component(task_dir / "news_images"):
        raise ValueError("News image task and asset directories must not traverse symbolic links")
    task_dir.mkdir(parents=True, exist_ok=True)
    image_dir = task_dir / "news_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = storyboard_fingerprint(storyboard)
    excluded = excluded_scene_ids or set()
    eligible = [
        str(scene.get("id") or "")
        for scene in storyboard.get("scenes") or []
        if str(scene.get("id") or "") and str(scene.get("id") or "") not in excluded
    ]
    target = min(max(0, count), len(eligible))
    contract_sha256 = acquisition_contract_fingerprint(
        storyboard_sha256=fingerprint,
        requested_count=count,
        target_count=target,
        eligible_scene_ids=eligible,
        excluded_scene_ids=excluded,
        scene_hints=scene_hints,
    )
    previous_manifest = read_manifest(task_dir)
    cached = _cached_manifest(
        task_dir,
        fingerprint,
        contract_sha256,
        storyboard,
        expected_requested_count=count,
        expected_target=target,
        expected_eligible_scene_ids=eligible,
        expected_excluded_scene_ids=excluded,
    )
    if cached:
        _emit(
            log,
            f"News images: reusing {len(cached['images'])} cached licensed image(s)",
        )
        return cached

    # A retry removes only generated paths explicitly owned by the preceding
    # manifest.  A broad image-* glob could delete an operator's source image.
    previous_version = (previous_manifest or {}).get("manifest_version")
    previous_status = str((previous_manifest or {}).get("status") or "")
    previous_owns_assets = (
        (previous_manifest or {}).get("storyboard_sha256") == fingerprint
        and isinstance(previous_version, int)
        and 8 <= previous_version <= MANIFEST_VERSION
        and previous_status in {"searching", "partial", "no_results", "ready"}
    )
    if previous_owns_assets:
        for stale_image in (previous_manifest or {}).get("images") or []:
            if not isinstance(stale_image, dict):
                continue
            stale_path = _owned_generated_asset_path(task_dir, stale_image)
            if stale_path is not None:
                stale_path.unlink()

    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "query_semantics_version": QUERY_SEMANTICS_VERSION,
        "grounding_policy_version": QUERY_SEMANTICS_VERSION,
        "cache_contract_sha256": contract_sha256,
        "status": "planning",
        "created_at": _now(),
        "updated_at": _now(),
        "storyboard_sha256": fingerprint,
        "requested_image_count": count,
        "eligible_scene_count": len(eligible),
        "eligible_scene_ids": eligible,
        "excluded_scene_ids": sorted(str(item) for item in excluded),
        "planned_image_count": target,
        "planner": "",
        "planner_conversation_url": "",
        "search_providers": [
            "ChatGPT picture-editor planning via OpenCLI",
            "Google News/Search via OpenCLI",
            "Wikimedia Commons",
        ],
        "license_policy": "open_only",
        "license_allowlist": ["Public Domain", "CC0", "CC BY", "CC BY-SA"],
        "queries": [],
        "images": [],
        "errors": [],
        "placement_modes": {"inline": 0, "fullscreen": 0},
    }
    _write_manifest(task_dir, manifest)
    if not target:
        manifest["status"] = "not_needed"
        manifest["updated_at"] = _now()
        _write_manifest(task_dir, manifest)
        return manifest

    plan, planner, conversation_url = await plan_news_images(
        storyboard,
        eligible_scene_ids=eligible,
        count=target,
        scene_hints=scene_hints,
        log=log,
    )
    scenes_by_id = {str(scene.get("id") or ""): scene for scene in storyboard.get("scenes") or []}
    eligible_scenes = [scenes_by_id[scene_id] for scene_id in eligible if scene_id in scenes_by_id]
    plan = _prepare_primary_plan(plan, eligible_scenes, target, scene_hints)
    reserves = _reserve_plan(plan, eligible_scenes, scene_hints)
    candidate_plan = [
        dict(
            _public_value(
                {
                    **shot,
                    "planned_display_mode": shot["display_mode"],
                    "candidate_role": "primary" if index < len(plan) else "reserve",
                }
            )
        )
        for index, shot in enumerate([*plan, *reserves])
    ]
    manifest["planner"] = planner
    manifest["planner_conversation_url"] = conversation_url
    manifest["queries"] = candidate_plan
    manifest["primary_query_count"] = len(plan)
    manifest["reserve_query_count"] = len(reserves)
    manifest["status"] = "searching"
    manifest["updated_at"] = _now()
    _write_manifest(task_dir, manifest)
    _emit(
        log,
        f"News images: prepared {len(plan)} primary and {len(reserves)} reserve scene assignment(s)",
    )

    headers = {"User-Agent": config.FOOTAGE_USER_AGENT}
    candidate_pools: dict[str, list[dict]] = {scene_id: [] for scene_id in eligible}
    primary_rows = list(enumerate(candidate_plan[: len(plan)], start=1))
    reserve_rows = list(enumerate(candidate_plan[len(plan) :], start=len(plan) + 1))
    priority_scenes = list(
        dict.fromkeys(
            [str(shot.get("scene_id") or "") for shot in plan]
            + eligible
        )
    )
    failed_sources: set[str] = set()
    source_hashes: dict[str, str] = {}

    async with httpx.AsyncClient(
        timeout=config.FOOTAGE_TIMEOUT,
        follow_redirects=True,
        headers=headers,
    ) as client:

        async def discover(position: int, shot: dict) -> None:
            scene_id = str(shot.get("scene_id") or "")
            scene = scenes_by_id.get(scene_id)
            if scene is None:
                return
            discovered = await _discover_shot_candidates(
                client,
                shot=shot,
                scene=scene,
                manifest=manifest,
                position=position,
                total=len(candidate_plan),
                log=log,
            )
            existing = {
                str(candidate.get("source_page_url") or "")
                for candidate in candidate_pools[scene_id]
            }
            candidate_pools[scene_id].extend(
                candidate
                for candidate in discovered
                if str(candidate.get("source_page_url") or "") not in existing
            )

        # Candidate discovery is separate from selection.  All primaries are
        # visible to the matcher before any source URL can be claimed.
        for position, shot in primary_rows:
            await discover(position, shot)

        while not manifest["images"]:
            matching = _maximum_source_matching(
                candidate_pools,
                priority_scenes,
                excluded_sources=failed_sources,
                source_hashes=source_hashes,
            )

            while len(matching) < target and reserve_rows:
                desired = priority_scenes[:target]
                connected = _conflict_component_scenes(
                    candidate_pools,
                    matching,
                    desired,
                    source_hashes=source_hashes,
                )
                next_index = next(
                    (
                        index
                        for index, (_position, shot) in enumerate(reserve_rows)
                        if str(shot.get("scene_id") or "") in connected
                    ),
                    0,
                )
                position, reserve_shot = reserve_rows.pop(next_index)
                await discover(position, reserve_shot)
                matching = _maximum_source_matching(
                    candidate_pools,
                    priority_scenes,
                    excluded_sources=failed_sources,
                    source_hashes=source_hashes,
                )

            if not matching:
                break
            selected_scene_ids = [
                scene_id for scene_id in priority_scenes if scene_id in matching
            ][:target]
            staged: list[dict] = []
            batch_failed = False
            duplicate_hash = False

            for scene_id in selected_scene_ids:
                candidate = matching[scene_id]
                source = str(candidate.get("source_page_url") or "")
                success_number, destination = _next_asset_destination(
                    image_dir,
                    len(staged) + 1,
                    _extension_for(candidate),
                )
                try:
                    byte_size, sha256 = await _download_candidate(
                        client,
                        candidate=candidate,
                        destination=destination,
                    )
                except Exception as exc:  # noqa: BLE001 - discard batch and rematch globally
                    destination.unlink(missing_ok=True)
                    failed_sources.add(source)
                    manifest["errors"].append(
                        {
                            "scene_id": scene_id,
                            "stage": "download",
                            "source_page_url": source,
                            "message": str(exc),
                        }
                    )
                    batch_failed = True
                    break

                source_hashes[source] = sha256
                asset_record = {
                    "local_path": destination.relative_to(task_dir).as_posix(),
                    "bytes": byte_size,
                    "sha256": sha256,
                    "kind": candidate.get("kind"),
                }
                if not _cached_asset_is_intact(task_dir, asset_record):
                    destination.unlink(missing_ok=True)
                    failed_sources.add(source)
                    manifest["errors"].append(
                        {
                            "scene_id": scene_id,
                            "stage": "decode",
                            "source_page_url": source,
                            "message": "Downloaded image failed byte, hash, or decoder verification",
                        }
                    )
                    batch_failed = True
                    break
                if sha256 in {image["sha256"] for image in staged}:
                    destination.unlink(missing_ok=True)
                    manifest["errors"].append(
                        {
                            "scene_id": scene_id,
                            "stage": "duplicate_content",
                            "source_page_url": source,
                            "message": "Downloaded bytes duplicate another candidate in the assignment",
                        }
                    )
                    batch_failed = True
                    duplicate_hash = True
                    break

                match_terms = list(candidate["grounding_distinctive_anchors"])
                downloaded = dict(
                    _public_value(
                        {
                            "id": f"image-{success_number:02d}",
                            **candidate,
                            "resolved_search_query": candidate.get("resolved_search_query")
                            or candidate.get("search_query"),
                            "source_mime_type": candidate.get("mime_type"),
                            "mime_type": {
                                ".jpg": "image/jpeg",
                                ".jpeg": "image/jpeg",
                                ".png": "image/png",
                                ".webp": "image/webp",
                                ".svg": "image/svg+xml",
                            }.get(destination.suffix.lower(), "application/octet-stream"),
                            "match_terms": match_terms,
                            "bytes": byte_size,
                            "sha256": sha256,
                            "local_path": destination.relative_to(task_dir).as_posix(),
                            "fit": "contain" if candidate["kind"] == "logo" else "cover",
                            "status": "downloaded",
                        }
                    )
                )
                if not image_grounding_is_valid(downloaded, scenes_by_id[scene_id]):
                    destination.unlink(missing_ok=True)
                    failed_sources.add(source)
                    manifest["errors"].append(
                        {
                            "scene_id": scene_id,
                            "stage": "grounding",
                            "source_page_url": source,
                            "message": "Selected image failed the persisted grounding contract",
                        }
                    )
                    batch_failed = True
                    break
                staged.append(downloaded)

            if batch_failed:
                for image in staged:
                    stale = _generated_asset_path(task_dir, image.get("local_path"))
                    if stale is not None:
                        stale.unlink(missing_ok=True)
                # A duplicate teaches the matcher that two URLs are one
                # content resource. No source is blacklisted; the next
                # augmenting path may move the flexible scene instead.
                if duplicate_hash:
                    continue
                continue

            manifest["images"] = staged
            for downloaded in staged:
                _emit(
                    log,
                    f"News image acquired for {downloaded['scene_id']}: {downloaded['title']} "
                    f"({downloaded['license']}, {downloaded['display_mode']})",
                )
            break

    if len(manifest["images"]) < target:
        manifest["errors"].append(
            {
                "stage": "selection",
                "message": "No unique scene/source/content assignment satisfied the grounding contract",
            }
        )

    acquired = len(manifest["images"])
    _ensure_placement_mode_mix(manifest["images"])
    manifest["placement_modes"] = _placement_mode_counts(manifest["images"])
    final_modes = {image["scene_id"]: image["display_mode"] for image in manifest["images"]}
    for query in manifest["queries"]:
        scene_id = str(query.get("scene_id") or "")
        if scene_id in final_modes:
            query["display_mode"] = final_modes[scene_id]
    manifest["status"] = "ready" if acquired == target else ("partial" if acquired else "no_results")
    manifest["updated_at"] = _now()
    _write_manifest(task_dir, manifest)
    _emit(
        log,
        f"News image scout finished: {acquired}/{target} licensed image(s); "
        f"inline={manifest['placement_modes']['inline']}, "
        f"fullscreen={manifest['placement_modes']['fullscreen']}",
    )
    return manifest


def _credit(image: dict, limit: int = 96) -> str:
    creator = " ".join(str(image.get("creator") or "Unknown creator").split())
    license_name = " ".join(str(image.get("license") or "Open license").split())
    text = f"Image: {creator} · {license_name}"
    return text if len(text) <= limit else text[: limit - 1].rstrip(" .·:;-") + "…"


def attach_news_images(
    plans: list[dict],
    storyboard: dict,
    manifest: dict | None,
    task_dir: Path,
) -> dict:
    """Attach exact-scene images without displacing existing moving B-roll."""
    if (
        not manifest
        or manifest.get("status") != "ready"
        or manifest.get("manifest_version") != MANIFEST_VERSION
        or manifest.get("query_semantics_version") != QUERY_SEMANTICS_VERSION
        or manifest.get("grounding_policy_version") != QUERY_SEMANTICS_VERSION
        or manifest.get("storyboard_sha256") != storyboard_fingerprint(storyboard)
        or manifest.get("license_policy") != "open_only"
    ):
        return {"attached": 0, "placement_modes": {"inline": 0, "fullscreen": 0}}
    images = manifest.get("images") or []
    raw_eligible = manifest.get("eligible_scene_ids") or []
    raw_excluded = manifest.get("excluded_scene_ids") or []
    if (
        not isinstance(images, list)
        or any(not isinstance(image, dict) for image in images)
        or not isinstance(raw_eligible, list)
        or not isinstance(raw_excluded, list)
    ):
        return {"attached": 0, "placement_modes": {"inline": 0, "fullscreen": 0}}
    requested = manifest.get("requested_image_count")
    planned = manifest.get("planned_image_count")
    eligible_count = manifest.get("eligible_scene_count")
    if any(type(value) is not int for value in (planned, requested, eligible_count)):
        return {"attached": 0, "placement_modes": {"inline": 0, "fullscreen": 0}}
    eligible = [str(item) for item in raw_eligible]
    excluded = {str(item) for item in raw_excluded}
    expected_eligible = [
        str(scene.get("id") or "")
        for scene in storyboard.get("scenes") or []
        if str(scene.get("id") or "") and str(scene.get("id") or "") not in excluded
    ]
    inventory_modes = _placement_mode_counts(images)
    if (
        not images
        or eligible != expected_eligible
        or eligible_count != len(eligible)
        or planned != min(max(0, requested), len(eligible))
        or len(images) != planned
        or manifest.get("placement_modes") != inventory_modes
        or (planned >= 2 and (not inventory_modes["inline"] or not inventory_modes["fullscreen"]))
    ):
        return {"attached": 0, "placement_modes": {"inline": 0, "fullscreen": 0}}
    by_id = {str(plan.get("id") or ""): plan for plan in plans}
    scenes_by_id = {
        str(scene.get("id") or ""): scene for scene in storyboard.get("scenes") or []
    }
    attached = 0
    modes = {"inline": 0, "fullscreen": 0}
    attached_scenes: set[str] = set()
    attached_sources: set[str] = set()
    attached_hashes: set[str] = set()
    inventory_scenes: set[str] = set()
    inventory_sources: set[str] = set()
    inventory_hashes: set[str] = set()
    inventory_paths: set[str] = set()
    for image in images:
        scene_id = str(image.get("scene_id") or "")
        source_page_url = str(image.get("source_page_url") or "")
        digest = str(image.get("sha256") or "")
        local_path = str(image.get("local_path") or "")
        scene = scenes_by_id.get(scene_id)
        if (
            scene is None
            or scene_id not in eligible
            or scene_id in inventory_scenes
            or not source_page_url
            or source_page_url in inventory_sources
            or not digest
            or digest in inventory_hashes
            or not local_path
            or local_path in inventory_paths
            or image.get("display_mode") not in PLAN_MODES
            or not image_grounding_is_valid(image, scene)
            or not _cached_asset_is_intact(task_dir, image)
        ):
            return {"attached": 0, "placement_modes": {"inline": 0, "fullscreen": 0}}
        inventory_scenes.add(scene_id)
        inventory_sources.add(source_page_url)
        inventory_hashes.add(digest)
        inventory_paths.add(local_path)

    for image in images:
        scene_id = str(image.get("scene_id") or "")
        source_page_url = str(image.get("source_page_url") or "")
        digest = str(image.get("sha256") or "")
        plan = by_id.get(scene_id)
        scene = scenes_by_id.get(scene_id)
        if not plan or scene is None:
            continue
        if (
            scene_id in attached_scenes
            or not source_page_url
            or source_page_url in attached_sources
            or not digest
            or digest in attached_hashes
            or not image_grounding_is_valid(image, scene)
            or not _cached_asset_is_intact(task_dir, image)
        ):
            continue
        if plan.get("collage_broll") or plan.get("archetype") == "footage":
            continue
        mode = str(image.get("display_mode") or "inline")
        if mode not in PLAN_MODES:
            mode = "inline"
        plan.update(
            {
                "news_image": True,
                # Scene HTML lives under compositions/, one directory below the
                # manifest and downloaded assets.
                "news_image_src": f"../{image['local_path']}",
                "news_image_mode": mode,
                "news_image_kind": image.get("kind") or "event",
                "news_image_fit": image.get("fit") or "cover",
                "news_image_credit": _credit(image),
                "news_image_caption": image.get("caption") or image.get("expected_subject") or "",
                "news_image_query": image.get("search_query") or "",
                "news_image_expected_subject": image.get("expected_subject") or "",
                "news_image_source_page_url": source_page_url,
                "news_image_source_scene_id": scene_id,
                "news_image_match_terms": image.get("match_terms") or [],
                "news_image_grounding_policy_version": image.get("grounding_policy_version"),
                "news_image_grounding_passed": image.get("grounding_passed") is True,
                "news_image_grounding_distinctive_anchors": image.get("grounding_distinctive_anchors") or [],
                "news_image_grounding_reason": image.get("grounding_reason") or "",
                "news_image_grounding_identity_field": image.get("grounding_identity_field") or "",
                "news_image_grounding_identity_phrase": image.get("grounding_identity_phrase") or "",
                "news_image_grounding_identity_field_terms": image.get("grounding_identity_field_terms") or [],
                "news_image_grounding_subject_terms": image.get("grounding_subject_terms") or [],
                "news_image_title": image.get("title") or "",
                "news_image_description": image.get("description") or "",
                "news_image_license": image.get("license") or "",
                "news_image_license_code": image.get("license_code") or "",
                "news_image_sha256": digest,
                "news_image_reference_count": len(image.get("references") or []),
            }
        )
        if mode == "fullscreen":
            plan["news_image_original_archetype"] = plan.get("archetype") or "topic"
            plan["archetype"] = "news_image"
        attached += 1
        attached_scenes.add(scene_id)
        attached_sources.add(source_page_url)
        attached_hashes.add(digest)
        modes[mode] += 1
    return {"attached": attached, "placement_modes": modes}

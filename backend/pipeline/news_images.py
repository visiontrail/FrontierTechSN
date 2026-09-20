"""OpenCLI-grounded, license-ledgered news imagery acquisition.

The spoken scene is the source of truth. OpenCLI/ChatGPT chooses the named brand,
person, place, product, or event worth illustrating; OpenCLI Google News/Search
records independent discovery evidence; Wikimedia Commons supplies
the actual image together with creator and license metadata. HyperFrames only
ever receives a local path, so capture remains deterministic and offline.
"""

from __future__ import annotations

from backend.pipeline.timing import timed

import asyncio
import hashlib
import html
import json
import logging
import math
import re
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import unquote, urlsplit

import httpx
from PIL import Image, ImageChops, UnidentifiedImageError

from backend import config
from backend.pipeline.image_layout import image_layout
from backend.pipeline.opencli import (
    OpenCLIError,
    first_json,
    run_opencli,
    run_opencli_with_retries,
)

logger = logging.getLogger(__name__)
LogCallback = Callable[[str], None]

WIKIMEDIA_API = "https://commons.wikimedia.org/w/api.php"
MANIFEST_VERSION = 18
QUERY_SEMANTICS_VERSION = 11
WIKIMEDIA_SEARCH_ATTEMPTS = 4
WIKIMEDIA_DOWNLOAD_ATTEMPTS = 5
NEWS_IMAGE_MAX_PIXELS = 16_000_000
ALPHA_FOREGROUND_THRESHOLD = 64
LOGO_SHAPE_ANALYSIS_MAX_DIMENSION = 512
LOGO_SHAPE_MAX_BBOX_FILL = 0.95
LOGO_SHAPE_MIN_BBOX_AREA = 0.05
LOGO_SHAPE_MIN_BBOX_FILL = 0.2
LOGO_SHAPE_MIN_FOREGROUND_COVERAGE = 0.05
LOGO_SHAPE_MIN_AXIS_EDGE_COMPLEXITY = 3.0
LOGO_SHAPE_MIN_AXIS_PATTERNS = 8
LOGO_SHAPE_MIN_AXIS_PROJECTIONS = 8
SUPPORTED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}
OPEN_NON_CC_LICENSES = frozenset({"apache 2.0", "apache license 2.0"})
RIGHTS_CONFLICT_CATEGORY_MARKERS = (
    "copyright violations",
    "deletion requests",
    "disputed copyright information",
    "no permission since",
    "possible copyright violations",
    "unknown copyright status",
)
KNOWN_COMMONS_FILE_TITLES = {
    ("annual reports", "object"): ("File:WMUA Annual reports 2012.JPG",),
    (
        "coffee shop",
        "object",
    ): ('File:Coffee shop in "The Boulevard" - geograph.org.uk - 1368799.jpg',),
    (
        "humanoid robot",
        "object",
    ): ("File:Humanoid robot is being programmed.jpg",),
    # These two broad narrated objects otherwise attract charts, museum
    # exhibits, and consumer robots near the top of Commons search.  The
    # vetted stills are literal photographic matches suitable for a collage:
    # an industrial robot line and visible server racks in a data centre.
    ("robots", "object"): ("File:FANUC 6-axis welding robots.jpg",),
    ("data centers", "object"): ("File:Data centers in Ashburn.jpg",),
    ("qwen", "logo"): ("File:Qwen Logo.svg",),
    ("qwen office", "logo"): ("File:Qwen Logo.svg",),
    ("softbank group", "logo"): ("File:SoftBank Group logo.svg",),
    ("techmeme", "logo"): ("File:Techmeme.png",),
}
LICENSE_NEGATIVE_RE = re.compile(
    r"(?:\ball rights reserved\b|\bcopyright(?:ed)?\b|\b(?:nc|nd|not|proprietary|"
    r"unlicensed|without)\b|\bnon\s*commercial\b|\bno\s+(?:cc|creative commons|"
    r"licenses?|rights?|permission|derivatives?|commercial|public domain|reuse|redistribution)\b)",
    flags=re.IGNORECASE,
)
PUBLIC_DOMAIN_LICENSES = frozenset(
    {
        "pd",
        "pd anon expired",
        "pd art",
        "pd author",
        "pd because",
        "pd ineligible",
        "pd nasa",
        "pd old",
        "pd old 50",
        "pd old 70",
        "pd old 80",
        "pd old 100",
        "pd old 100 expired",
        "pd old assumed",
        "pd old auto",
        "pd old auto expired",
        "pd self",
        "pd shape",
        "pd textlogo",
        "pd us",
        "pd us expired",
        "pd usgov",
        "pd usgov nasa",
        "public domain",
        "public domain mark",
        "public domain mark 1.0",
    }
)
CC_REGION_PATTERN = (
    r"(?:ar|at|au|be|bg|br|ca|ch|cl|cn|co|cr|cz|de|dk|ec|ee|eg|es|fi|fr|gr|gt|"
    r"hk|hr|hu|ie|il|in|it|jp|kr|lu|mk|mt|mx|my|nl|no|nz|pe|ph|pl|pr|pt|ro|rs|"
    r"scotland|se|sg|si|th|tw|ug|uk|us|ve|vn|za)"
)
CC_LICENSE_DETAIL_PATTERN = (
    rf"(?:"
    rf"(?:1\.0|2\.0|2\.5)|"
    rf"3\.0(?: unported)?|"
    rf"4\.0(?: international)?|"
    rf"2\.1 jp|"
    rf"(?:2\.0|2\.5|3\.0) {CC_REGION_PATTERN}|"
    rf"3\.0 igo"
    rf")"
)
CC0_LICENSE_RE = re.compile(r"cc(?:0| zero)(?: 1\.0)?(?: universal)?", flags=re.IGNORECASE)
CC_LICENSE_RE = re.compile(
    rf"cc by(?: sa)?(?: {CC_LICENSE_DETAIL_PATTERN})?",
    flags=re.IGNORECASE,
)
CC_LONG_LICENSE_RE = re.compile(
    rf"creative commons attribution(?: share alike)?"
    rf"(?: {CC_LICENSE_DETAIL_PATTERN})?",
    flags=re.IGNORECASE,
)
PLAN_KINDS = {"logo", "event", "person", "place", "product", "object"}
PLAN_MODES = {"inline", "fullscreen"}
TAG_RE = re.compile(r"<[^>]+>")
WORD_RE = re.compile(
    r"[0-9]+(?:\.[0-9]+)+(?:['’-][A-Za-z0-9]+)?|"
    r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)?"
)
ENTITY_PHRASE_TOKEN_RE = re.compile(
    r"[0-9]+(?:\.[0-9]+)+(?:['’-][A-Za-z0-9]+)?|"
    r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)?"
)
CAPITAL_PHRASE_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9'’-]*|[0-9]+[A-Z][A-Za-z0-9'’-]*)"
    r"(?:\s+(?:[A-Z][A-Za-z0-9'’-]*|[0-9]+[A-Z][A-Za-z0-9'’-]*|of|the|and)){0,5}\b"
)
VERSIONED_PRODUCT_RE = re.compile(
    r"\b[A-Z][A-Za-z0-9'’-]*\s+[0-9]+(?:\.[0-9]+)+"
    r"(?:\s+[A-Z][A-Za-z0-9'’-]*){0,3}\b"
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
    "dimensions employee employees engineer engineers engineering event germany global harness idea ideas innovation "
    "management model office photo product products project projects report research society system systems "
    "tested testing technology thursday time tools united world english language api max "
    "ceo ceos cfo cfos cto ctos coo coos cio cios cmo cmos chro chros vp svp evp ipo ipos".split()
)
GENERIC_QUERY_TERMS = GENERIC_ENTITY_TERMS | frozenset(
    "corporate official image photograph portrait logo mark launch".split()
)
ENTITY_EDGE_TERMS = frozenset("and of the for in on at by from with".split())
SUBJECT_CONTEXT_LEAD_TERMS = ENTITY_EDGE_TERMS | frozenset(
    "about across after around before during into over through to under without within".split()
)
MONTH_TERMS = frozenset("january february march april may june july august september october november december".split())
REPORTING_VERBS_RE = re.compile(
    r"^\s+(?:also\s+)?(?:reports|reported|notes?|noted|says|said|writes|wrote|"
    r"describes|described)\b",
    flags=re.I,
)
BAD_PHOTO_MARKERS = ("logo", "icon", "map", "diagram", "chart", "flag")
GENERIC_LOGO_TITLE_TERMS = frozenset(
    "black brand corporate english en icon logo mark png symbol transparent white wordmark svg".split()
)
IDENTITY_CONNECTORS = frozenset("and of the".split())
IDENTITY_DECORATORS = frozenset(
    "black brand chinese co company corp corporate corporation editorial english en event file financial group holding "
    "holdings icon icons image images inc jpeg jpg limited llc logo logos ltd mark marks official photo photograph "
    "photographs photos plc png portrait product products public screenshot screenshots symbol symbols transparent white "
    "wordmark wordmarks webp zh svg".split()
)
MEDIA_WORK_TERMS = frozenset(
    "album albums book books cinema film films movie movies music musical novel novels record "
    "recording recordings records sencillo series single singles song songs soundtrack television tv".split()
)
CARDINAL_OR_ORDINAL_TERMS = frozenset(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen twenty first second third fourth fifth sixth seventh eighth "
    "ninth tenth dozen hundred thousand million billion all few many multiple several some".split()
)
NON_ENTITY_LEAD_TERMS = frozenset(
    "how what when where which who why according if i i'll im i'm we we'll we're you you'll you're "
    "they they'll they're this that these those subscribe".split()
)
ABSTRACT_PRACTICE_TERMS = frozenset(
    "bootlegging skunkworks harness innovation management governance capability performance score "
    "pricing commercialization concept metric value".split()
)
GEOGRAPHIC_ENTITY_TERMS = frozenset(
    "china taiwan germany united states wall street chinese american european asia asian europe africa "
    "singapore japan korea india france britain british england canada australia russia ukraine".split()
)
GEOGRAPHIC_ENTITY_PHRASES = frozenset(
    {
        "africa",
        "asia",
        "australia",
        "britain",
        "canada",
        "china",
        "england",
        "europe",
        "france",
        "germany",
        "india",
        "japan",
        "korea",
        "russia",
        "singapore",
        "taiwan",
        "ukraine",
        "united states",
        "wall street",
    }
)
GEOGRAPHIC_ADJECTIVE_TERMS = frozenset(
    "american asian british chinese european german indian japanese korean russian taiwanese".split()
)
CONCRETE_OBJECT_HEADS = frozenset(
    "robot robots report reports browser browsers poster posters document documents satellite satellites "
    "rocket rockets drone drones chip chips processor processors computer computers server servers vehicle "
    "vehicles aircraft phone phones battery batteries camera cameras sensor sensors shop shops center centers".split()
)
STANDALONE_VISUAL_OBJECT_HEADS = frozenset(
    "robot robots satellite satellites rocket rockets drone drones chip chips processor processors "
    "computer computers server servers vehicle vehicles aircraft phone phones battery batteries camera "
    "cameras sensor sensors".split()
)
OBJECT_ACTION_MODIFIERS = frozenset(
    "building deploying making operating put putting testing using".split()
)
ORGANISATION_NAME_SUFFIXES = frozenset(
    "association company corporation group institute laboratory labs society university".split()
)
PERSON_REFERENCE_TERMS = frozenset(
    "analyst author editor engineer founder professor researcher scientist".split()
)
SUBJECT_IDENTITY_ALIASES = {
    "qianwen": "qwen",
}
BRANDED_PRODUCT_SUFFIXES = frozenset(
    "agent api app application browser chip cpu model office platform processor service system tool".split()
)
PRODUCT_CONTEXT_RE = re.compile(
    r"\b(?:app|application|model|platform|product|service|software|system|tool|update)\b",
    flags=re.I,
)
EVENT_CONTEXT_RE = re.compile(
    r"\b(?:announced|conference|event|expo|festival|keynote|launch|launched|opened|summit|tournament)\b",
    flags=re.I,
)
NAMED_SUBJECT_ACTION_RE = re.compile(
    r"^\s+(?:acquired|allocate(?:d|s)|announced|built|created|developed|introduced|launched|"
    r"manufactures|published|raised|recorded|released|reported|scored|tested|unveiled)\b",
    flags=re.I,
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
    values = [str(value or "").strip() for value in (short_name, license_code) if str(value or "").strip()]
    normalized = [" ".join(re.sub(r"[-_]+", " ", value).casefold().split()) for value in values]
    if not normalized or any(LICENSE_NEGATIVE_RE.search(value) for value in normalized):
        return False
    return all(
        value in PUBLIC_DOMAIN_LICENSES
        or value in OPEN_NON_CC_LICENSES
        or CC0_LICENSE_RE.fullmatch(value)
        or CC_LICENSE_RE.fullmatch(value)
        or CC_LONG_LICENSE_RE.fullmatch(value)
        for value in normalized
    )


def _candidate_source_is_still_image(candidate: dict) -> bool:
    """Reject rasterized pages or frames whose original source is not an image."""
    return str(candidate.get("source_mime_type") or "").casefold().startswith("image/")


def _candidate_rights_are_clear(candidate: dict) -> bool:
    """Fail closed when Commons itself marks the file's rights as disputed."""
    categories = str(candidate.get("categories") or "").casefold()
    return not any(marker in categories for marker in RIGHTS_CONFLICT_CATEGORY_MARKERS)


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


def _meaningful_identity_tokens(
    field_tokens: list[str],
    *,
    retained_tokens: Sequence[str] = (),
) -> list[str]:
    retained = set(retained_tokens)
    return [
        token
        for token in field_tokens
        if (token in retained or token not in IDENTITY_DECORATORS)
        and not (len(token) == 4 and token.isdigit())
        and (len(token) > 1 or any(char.isdigit() for char in token))
    ]


def _metadata_extends_subject_identity(value: object, subject: object) -> bool:
    """Reject a proper-name extension such as ``Google Loon`` for ``Google``."""
    text = str(value or "")
    subject_tokens = _identity_tokens(subject)
    if not text or not subject_tokens:
        return False
    for match in CAPITAL_PHRASE_RE.finditer(text):
        phrase_tokens = _identity_tokens(match.group(0))
        if not _contains_token_phrase(phrase_tokens, subject_tokens):
            continue
        # A legal suffix such as ``Group`` is decoration for ``Google Group``
        # when the narrated identity is Google, but it is part of the identity
        # when the narration itself says ``SoftBank Group``.
        meaningful = _meaningful_identity_tokens(
            phrase_tokens,
            retained_tokens=subject_tokens,
        )
        if meaningful != subject_tokens:
            return True
    normalized_subject = _normalise_entity_phrase(subject)
    allowed_followers = (
        IDENTITY_DECORATORS
        | STOPWORDS
        | GENERIC_ENTITY_TERMS
        | CONCRETE_OBJECT_HEADS
        | ENTITY_EDGE_TERMS
    )
    for match in _phrase_matches(text, normalized_subject):
        tail = text[match.end() :]
        if re.match(r"^\s*['’]s\b", tail, flags=re.I):
            tail = re.sub(r"^\s*['’]s\b", "", tail, count=1, flags=re.I)
        follower = re.match(
            r"^[\s\(\)\[\]\{\},:;/._–—-]*([A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)?)",
            tail,
        )
        if not follower:
            continue
        token = follower.group(1)
        key = token.casefold().strip("'’")
        if (
            key in allowed_followers
            or (len(key) == 4 and key.isdigit())
            or (len(key) == 1 and key == subject_tokens[0][:1])
        ):
            continue
        return True
    return False


def _first_metadata_subject_identity_extends(value: object, subject: object) -> bool:
    """Check only the first description mention, ignoring later incidental context."""
    text = str(value or "")
    normalized_subject = _normalise_entity_phrase(subject)
    subject_tokens = _identity_tokens(subject)
    matches = _phrase_matches(text, normalized_subject)
    if not matches or not subject_tokens:
        return False
    tail = text[matches[0].end() :]
    if re.match(r"^\s*['’]s\b", tail, flags=re.I):
        tail = re.sub(r"^\s*['’]s\b", "", tail, count=1, flags=re.I)
    follower = re.match(
        r"^[\s\(\)\[\]\{\},:;/._–—-]*([A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)?)",
        tail,
    )
    if not follower:
        return False
    token = follower.group(1)
    key = token.casefold().strip("'’")
    if key in {"see", "siehe"}:
        cross_reference = re.match(
            r"^[\s\(\)\[\]\{\},:;/._–—-]*([A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)?)",
            tail[follower.end() :],
        )
        if (
            cross_reference
            and cross_reference.group(1).casefold().strip("'’") in subject_tokens
        ):
            return False
    allowed_followers = (
        IDENTITY_DECORATORS
        | STOPWORDS
        | GENERIC_ENTITY_TERMS
        | CONCRETE_OBJECT_HEADS
        | ENTITY_EDGE_TERMS
    )
    return not (
        key in allowed_followers
        or (len(key) == 4 and key.isdigit())
        or (len(key) == 1 and key == subject_tokens[0][:1])
    )


def _candidate_identity_match(shot: dict, candidate: dict) -> tuple[str, str, list[str]] | None:
    """Find the complete subject identity in one metadata field.

    Combining title and description allowed a first name in one field and a
    surname in another to masquerade as a full person match.  Multi-token
    identities must now occur contiguously in one field.  Every logo identity
    and every single-token identity is accepted only when the field has no
    second meaningful identity, so
    ``Qwen`` cannot validate ``Qwen Audio``, ``Google`` cannot validate
    ``Google Loon``, and ``Perfect World`` cannot validate a film or album
    title with the same words.
    """
    subject_tokens = _identity_tokens(shot.get("expected_subject"))
    if not subject_tokens or subject_tokens == ["ai"]:
        return None
    kind = str(shot.get("kind") or "event")
    # Identity extension is essential for proper names (``Google`` must not
    # validate ``Google Loon``), but ordinary objects need descriptive
    # context.  ``Data centers in Ashburn`` and ``industrial robots at work``
    # are better photographic matches for their narrated objects, not new
    # named identities.
    if kind != "object" and (
        any(
            _metadata_extends_subject_identity(
                candidate.get(field_name), shot.get("expected_subject")
            )
            for field_name in ("title", "object_name")
        )
        or _first_metadata_subject_identity_extends(
            candidate.get("description"), shot.get("expected_subject")
        )
    ):
        return None
    title_tokens = _identity_tokens(candidate.get("title"))
    if kind == "logo":
        # Exact Commons file titles can be just ``Techmeme.png`` while the
        # linked description or object name says ``logo for Techmeme``.  Keep
        # the strong logo proof, but allow it in any authoritative identity
        # field rather than requiring the filename to contain the word logo.
        logo_identity_proved = False
        for field_name in ("title", "description", "object_name"):
            field_tokens = _identity_tokens(candidate.get(field_name))
            if (
                {"logo", "wordmark", "mark"} & set(field_tokens)
                and _contains_token_phrase(field_tokens, subject_tokens)
            ):
                logo_identity_proved = True
                break
        if not logo_identity_proved:
            return None
    for field_name in ("title", "description"):
        value = str(candidate.get(field_name) or "")
        field_tokens = _identity_tokens(value)
        if not _contains_token_phrase(field_tokens, subject_tokens):
            continue
        if kind == "logo":
            exact_identity = _meaningful_identity_tokens(
                field_tokens,
                retained_tokens=subject_tokens,
            )
        else:
            exact_identity = [
                token
                for token in field_tokens
                if token not in IDENTITY_DECORATORS
                and not (len(token) == 4 and token.isdigit())
            ]
        # Generic objects need descriptive modifiers to be useful imagery:
        # ``Industrial robots`` is a truthful candidate for narrated
        # ``robots``. Keep exact-only matching for brands, people, products,
        # events, and places, where a modifier often creates a different
        # identity (Google Loon, Google I/O, and so on).
        if (
            kind == "logo" or (len(subject_tokens) == 1 and kind != "object")
        ) and exact_identity != subject_tokens:
            continue
        if kind != "logo" and ({"logo", "wordmark", "icon"} & set(field_tokens)) and not (
            {"event", "photo", "photograph", "screenshot", "launch", "conference", "expo"}
            & set(field_tokens)
        ):
            continue
        if field_name != "title":
            contradictory_title = _meaningful_identity_tokens(title_tokens)
            if kind == "logo":
                if contradictory_title and contradictory_title != subject_tokens:
                    continue
            elif not _contains_token_phrase(title_tokens, subject_tokens) and contradictory_title:
                continue
        return field_name, " ".join(field_tokens), subject_tokens
    return None


def _candidate_context_conflicts(scene: dict, candidate: dict) -> list[str]:
    """Reject an unrelated media work that merely shares an entity's name."""
    scene_text = str(scene.get("text") or "")
    candidate_text = (
        f"{candidate.get('title', '')} {candidate.get('description', '')} "
        f"{candidate.get('categories', '')} {candidate.get('object_name', '')} "
        f"{candidate.get('creator', '')}"
    )
    scene_terms = _semantic_terms(scene_text)
    candidate_terms = _semantic_terms(
        candidate_text
    )
    conflicts = (candidate_terms & MEDIA_WORK_TERMS) - (scene_terms & MEDIA_WORK_TERMS)
    game_term_re = re.compile(r"\bmmorpgs?\b|\bvideo[\s-]+games?\b", flags=re.I)
    organisation_after_game_re = re.compile(
        r"^[\s\(\)\[\]\{\},:;/._–—-]*"
        r"(?:(?:video[\s-]+games?|games?)\s+)?"
        r"(?:business(?:es)?|companies|company|corporations?|developers?|firms?|industr(?:y|ies)|"
        r"publishers?|studios?)\b",
        flags=re.I,
    )

    def has_game_work_context(value: str) -> bool:
        return any(
            not organisation_after_game_re.match(value[match.end() :])
            for match in game_term_re.finditer(value)
        )

    if has_game_work_context(candidate_text) and not has_game_work_context(scene_text):
        conflicts.add("video game")
    return sorted(conflicts)


def _distinctive_terms(value: object) -> set[str]:
    return _semantic_terms(value) - GENERIC_QUERY_TERMS


def _normalise_entity_phrase(value: object) -> str:
    tokens = ENTITY_PHRASE_TOKEN_RE.findall(str(value or ""))
    while tokens and tokens[0].casefold() in {"the"}:
        tokens.pop(0)
    while tokens and (
        tokens[-1].casefold().strip("'’") in ENTITY_EDGE_TERMS
        or tokens[-1].casefold() in MONTH_TERMS
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
        if (
            re.search(r"(?:according to|described by|reported by|published by)\s*$", before)
            or REPORTING_VERBS_RE.match(after)
            or re.match(r"\s+(?:has\s+)?published\s+an?\s+article\b", after, flags=re.I)
        ):
            output.add(phrase.casefold())
    return output


def _subject_is_only_reporting_source(subject: str, text: str) -> bool:
    subject_matches = _phrase_matches(text, subject)
    if not subject_matches:
        return False
    source_evidence: list[tuple[list[str], list[re.Match[str]]]] = []
    subject_identity = _identity_tokens(subject)
    for source in _source_entity_keys(text):
        source_identity = _identity_tokens(source)
        if not (
            _contains_token_phrase(source_identity, subject_identity)
            or _contains_token_phrase(subject_identity, source_identity)
        ):
            continue
        source_evidence.append((source_identity, _phrase_matches(text, source)))
    return bool(source_evidence) and all(
        any(
            (
                source_match.start() <= match.start()
                and match.end() <= source_match.end()
            )
            or (
                match.start() <= source_match.start()
                and source_match.end() <= match.end()
            )
            for _identity, source_matches in source_evidence
            for source_match in source_matches
        )
        for match in subject_matches
    )


def _phrase_matches(text: str, subject: str, *, case_sensitive: bool = False) -> list[re.Match[str]]:
    parts = [re.escape(part) for part in subject.split() if part]
    if not parts:
        return []
    flags = 0 if case_sensitive else re.I
    joined = r"\s+".join(parts)
    pattern = rf"(?<![A-Za-z0-9]){joined}(?![A-Za-z0-9])"
    return list(re.finditer(pattern, text, flags=flags))


def _person_is_explicit(subject: str, scene_text: str) -> bool:
    identity = _identity_tokens(subject)
    if len(identity) < 2:
        return False
    named_tokens = [
        token
        for token in ENTITY_PHRASE_TOKEN_RE.findall(subject)
        if token.casefold() not in IDENTITY_CONNECTORS
    ]
    if not named_tokens or not all(
        token[:1].isupper()
        or token.isupper()
        or any(char.isdigit() for char in token)
        or any(char.isupper() for char in token[1:])
        for token in named_tokens
    ):
        return False
    escaped = r"\s+".join(re.escape(part) for part in subject.split())
    role = "|".join(sorted(PERSON_REFERENCE_TERMS))
    return bool(
        re.search(rf"\b(?:authored|written|reported|researched)\s+by\s+{escaped}\b", scene_text, flags=re.I)
        or re.search(rf"\b{escaped}\s+(?:authored|wrote|reported|researched)\b", scene_text, flags=re.I)
        or re.search(rf"\b{escaped}\s*,[^,.]{{0,64}}\b(?:{role})\b", scene_text, flags=re.I)
        or re.search(rf"\b(?:{role})\s+{escaped}\b", scene_text, flags=re.I)
    )


def _is_incomplete_person_reference(subject: str, scene_text: str) -> bool:
    identity = _identity_tokens(subject)
    if len(identity) != 1:
        return False
    escaped = re.escape(subject)
    if re.search(
        rf"\b{escaped}(?:['’]s)?\s+(?:analysis|article|findings|research|study|work)\b",
        scene_text,
        flags=re.I,
    ):
        return True
    for match in CAPITAL_PHRASE_RE.finditer(scene_text):
        full_name = _normalise_entity_phrase(match.group(0))
        full_identity = _identity_tokens(full_name)
        if len(full_identity) >= 2 and full_identity[-1] == identity[0] and _person_is_explicit(
            full_name, scene_text
        ):
            return True
    return False


def _is_incomplete_named_fragment(subject: str, scene_text: str) -> bool:
    identity = _identity_tokens(subject)
    if not identity:
        return True
    if len(identity) == 1:
        token = ENTITY_PHRASE_TOKEN_RE.findall(subject)[0]
        if (
            token.isupper()
            or any(char.isdigit() for char in token)
            or any(char.isupper() for char in token[1:])
            or token.casefold() in {"qwen", "qianwen"}
        ):
            return False

    subject_matches = _phrase_matches(scene_text, subject, case_sensitive=True)
    if not subject_matches:
        return True
    capital_matches = list(CAPITAL_PHRASE_RE.finditer(scene_text))
    incomplete = []
    for subject_match in subject_matches:
        enclosing = [
            match
            for match in capital_matches
            if match.start() <= subject_match.start()
            and subject_match.end() <= match.end()
            and _normalise_entity_phrase(match.group(0)).casefold() != subject.casefold()
        ]
        if not enclosing:
            incomplete.append(False)
            continue
        complete_segment = False
        for match in enclosing:
            raw = match.group(0)
            raw_words = [token.casefold() for token in WORD_RE.findall(raw)]
            segments = (
                [raw]
                if raw_words and raw_words[-1] in ORGANISATION_NAME_SUFFIXES
                else re.split(r"\s+and\s+", raw, flags=re.I)
            )
            possessive_segments: list[str] = []
            for segment in segments:
                tokens = ENTITY_PHRASE_TOKEN_RE.findall(segment)
                possessive_index = next(
                    (
                        index
                        for index, token in enumerate(tokens)
                        if token.casefold().endswith(("'s", "’s"))
                    ),
                    -1,
                )
                if 0 <= possessive_index < len(tokens) - 1:
                    possessive_segments.extend(
                        [
                            " ".join(tokens[: possessive_index + 1]),
                            " ".join(tokens[possessive_index + 1 :]),
                        ]
                    )
                else:
                    possessive_segments.append(segment)
            if any(
                _normalise_entity_phrase(segment).casefold() == subject.casefold()
                for segment in possessive_segments
            ):
                complete_segment = True
                break
        incomplete.append(not complete_segment)
    return bool(incomplete) and all(incomplete)


def _subject_semantic_role(subject: object, scene: dict) -> tuple[str, str]:
    """Classify a narrated visual subject or return a fail-closed reason.

    Literal repetition is not enough: common words such as ``Five`` and
    ``Bootlegging`` can have unrelated Commons logos.  The role is derived
    solely from the narrated scene, never from planner copy or candidate
    metadata, so every later trust boundary can recompute the same decision.
    """
    raw_subject = " ".join(str(subject or "").split())
    normalized = _normalise_entity_phrase(subject)
    scene_text = str(scene.get("text") or "")
    identity = _identity_tokens(normalized)
    semantic = _semantic_terms(normalized)
    ordered = _ordered_semantic_terms(normalized)
    if not normalized or raw_subject != normalized or not identity or not semantic:
        return "", "shot subject had no usable complete identity"
    if not _contains_token_phrase(_identity_tokens(scene_text), identity):
        return "", "shot subject was not a contiguous identity in the narrated scene"
    if _subject_is_only_reporting_source(normalized, scene_text):
        return "", "shot subject was only a reporting source in the narrated scene"

    lead = identity[0].casefold().strip("'’")
    if lead in NON_ENTITY_LEAD_TERMS or lead in SUBJECT_CONTEXT_LEAD_TERMS:
        return "", "shot subject began with a pronoun, question, or sentence lead"
    if semantic and semantic.issubset(CARDINAL_OR_ORDINAL_TERMS):
        return "", "shot subject was only a number or ordinal"

    subject_words = [token.casefold().strip("'’") for token in WORD_RE.findall(normalized)]
    if subject_words and subject_words[0] in SUBJECT_CONTEXT_LEAD_TERMS:
        return "", "shot subject began with surrounding sentence context"
    numeric_identity_re = re.compile(r"^[0-9]+(?:\.[0-9]+)*(?:['’-][A-Za-z0-9]+)?$")
    if (
        len(subject_words) >= 2
        and subject_words[0] in MONTH_TERMS
        and numeric_identity_re.fullmatch(subject_words[1])
    ):
        return "", "shot subject was only a calendar date"
    if subject_words and all(numeric_identity_re.fullmatch(token) for token in subject_words):
        return "", "shot subject was only a numeric value or version"
    if subject_words and numeric_identity_re.fullmatch(subject_words[0]):
        return "", "versioned product subject lacked its leading brand identity"
    if (
        len(subject_words) == 1
        and re.fullmatch(r"[a-z][0-9]{1,2}", subject_words[0])
    ):
        return "", "ambiguous short product code lacked its leading brand identity"
    raw_subject_tokens = ENTITY_PHRASE_TOKEN_RE.findall(normalized)
    if any(token.casefold().endswith(("'s", "’s")) for token in raw_subject_tokens[:-1]):
        return "", "shot subject combined a possessive owner with a separate identity"
    if "and" in subject_words and (not subject_words or subject_words[-1] not in ORGANISATION_NAME_SUFFIXES):
        return "", "shot subject combined multiple independent identities"

    non_abstract = semantic - ABSTRACT_PRACTICE_TERMS - GENERIC_ENTITY_TERMS - {"ai"}
    if semantic & ABSTRACT_PRACTICE_TERMS and not non_abstract:
        return "", "shot subject was an abstract practice or evaluation concept"

    if _person_is_explicit(normalized, scene_text):
        return "person", ""
    if _is_incomplete_person_reference(normalized, scene_text):
        return "", "shot subject was an incomplete person identity"

    if semantic and semantic.issubset(GEOGRAPHIC_ADJECTIVE_TERMS):
        return "", "shot subject was only a geographic adjective"
    geographic = normalized.casefold() in GEOGRAPHIC_ENTITY_PHRASES
    if geographic:
        exact_matches = _phrase_matches(scene_text, normalized, case_sensitive=True)
        if not exact_matches:
            return "", "geographic subject was not retained verbatim in the narration"
        contextual_matches = []
        for match in exact_matches:
            after = scene_text[match.end() : match.end() + 24].casefold()
            if after.startswith(("'s", "’s")):
                continue
            if re.match(
                r"\s+(?:investment\s+)?(?:bank|company|firm|university|military|army|navy|"
                r"air force|space force|government|embassy|ministry|department)\b", after,
            ):
                continue
            contextual_matches.append(match)
        if not contextual_matches:
            return "", "geographic subject was only a modifier of another narrated identity"
        return "place", ""

    object_identity = [term.casefold() for term in identity]
    object_head = object_identity[-1] if object_identity else ""
    if object_head in CONCRETE_OBJECT_HEADS:
        if normalized != normalized.casefold() or not _phrase_matches(
            scene_text, normalized, case_sensitive=True
        ):
            return "", "concrete object subject was not a lower-case narrated noun phrase"
        if len(identity) == 1 and object_head in STANDALONE_VISUAL_OBJECT_HEADS:
            return "object", ""
        if " ".join(object_identity) in {"data center", "data centers"}:
            return "object", ""
        modifier = object_identity[0] if len(object_identity) == 2 else ""
        if (
            len(identity) != 2
            or not modifier
            or modifier in STOPWORDS
            or modifier in GENERIC_ENTITY_TERMS
            or modifier in CARDINAL_OR_ORDINAL_TERMS
            or modifier in NON_ENTITY_LEAD_TERMS
            or modifier in SUBJECT_CONTEXT_LEAD_TERMS
        ):
            return "", "concrete object subject lacked a specific narrated modifier"
        return "object", ""

    if VERSIONED_PRODUCT_RE.fullmatch(normalized):
        matches = _phrase_matches(scene_text, normalized, case_sensitive=True)
        if matches and all(
            re.match(r"\s+[A-Z][A-Za-z0-9'’-]*\b", scene_text[match.end() :])
            for match in matches
        ):
            return "", "versioned product subject was only a prefix of the narrated identity"
        return "product", ""

    strong_identity = [
        term
        for term in ordered
        if term not in GENERIC_QUERY_TERMS
        and term not in ABSTRACT_PRACTICE_TERMS
        and term not in CARDINAL_OR_ORDINAL_TERMS
        and term not in GEOGRAPHIC_ENTITY_TERMS
        and term not in NON_ENTITY_LEAD_TERMS
        and term != "ai"
    ]
    if not strong_identity:
        return "", "shot subject had no specific brand, organisation, product, or event identity"
    if not _phrase_matches(scene_text, normalized, case_sensitive=True):
        return "", "named subject did not retain entity capitalization in the narration"
    if _is_incomplete_named_fragment(normalized, scene_text):
        return "", "named subject was only a fragment of a longer narrated identity"
    named_tokens = [
        token
        for token in ENTITY_PHRASE_TOKEN_RE.findall(normalized)
        if token.casefold() not in IDENTITY_CONNECTORS
    ]
    if not named_tokens or not all(
        token[:1].isupper()
        or token.isupper()
        or any(char.isdigit() for char in token)
        or any(char.isupper() for char in token[1:])
        for token in named_tokens
    ):
        return "", "shot subject lacked a proper-name, acronym, or product identity form"
    if len(named_tokens) == 1:
        token = named_tokens[0]
        unambiguous_form = (
            token.isupper()
            or any(char.isdigit() for char in token)
            or any(char.isupper() for char in token[1:])
            or token.casefold() in {"qwen", "qianwen"}
        )
        if not unambiguous_form:
            contextual_name = False
            for match in _phrase_matches(scene_text, normalized, case_sensitive=True):
                before = scene_text[: match.start()].rstrip()
                sentence_start = not before or before[-1] in ".!?"
                if not sentence_start or NAMED_SUBJECT_ACTION_RE.match(scene_text[match.end() :]):
                    contextual_name = True
                    break
            if not contextual_name:
                return "", "single title-case subject was only an ungrounded sentence lead"
    return "brand", ""


def _shot_subject_semantic_error(shot: dict, scene: dict) -> str:
    role, reason = _subject_semantic_role(shot.get("expected_subject"), scene)
    if reason:
        return reason
    kind = str(shot.get("kind") or "").strip().casefold()
    allowed_kinds = {
        "brand": {"logo"},
        "object": {"object"},
        "person": {"person"},
        "place": {"place"},
        "product": {"product", "logo"},
    }
    if role == "brand" and kind in {"event", "product"}:
        normalized = _normalise_entity_phrase(shot.get("expected_subject"))
        identity = _identity_tokens(normalized)
        windows = [
            str(scene.get("text") or "")[max(0, match.start() - 32) : match.end() + 80]
            for match in _phrase_matches(
                str(scene.get("text") or ""), normalized, case_sensitive=True
            )
        ]
        if kind == "event" and any(EVENT_CONTEXT_RE.search(window) for window in windows):
            return ""
        if kind == "product" and (
            (identity and identity[-1] in BRANDED_PRODUCT_SUFFIXES)
            or (
                len(identity) == 1
                and any(char.isupper() for char in normalized[1:])
                and any(PRODUCT_CONTEXT_RE.search(window) for window in windows)
            )
        ):
            return ""
    if kind not in allowed_kinds.get(role, set()):
        return f"shot kind {kind or '(missing)'} conflicted with narrated {role} subject"
    return ""


def _concrete_object_candidates(scene: dict) -> list[str]:
    """Return specific two-word object phrases from narration in text order."""
    text = str(scene.get("text") or "")
    output: list[str] = []
    seen: set[str] = set()
    heads = "|".join(sorted((re.escape(head) for head in CONCRETE_OBJECT_HEADS), key=len, reverse=True))
    pattern = re.compile(rf"\b([a-z][A-Za-z0-9'’-]*)[\s-]+({heads})\b")
    for match in pattern.finditer(text):
        modifier = match.group(1).casefold().strip("'’")
        if (
            modifier in OBJECT_ACTION_MODIFIERS
            and match.group(2).casefold() in STANDALONE_VISUAL_OBJECT_HEADS
        ):
            subject = match.group(2)
            key = subject.casefold()
            shot = {"expected_subject": subject, "kind": "object"}
            if key not in seen and not _shot_subject_semantic_error(shot, scene):
                seen.add(key)
                output.append(subject)
            continue
        if (
            modifier in STOPWORDS
            or modifier in CARDINAL_OR_ORDINAL_TERMS
            or modifier in GENERIC_ENTITY_TERMS
            or modifier in NON_ENTITY_LEAD_TERMS
        ):
            continue
        subject = f"{match.group(1)} {match.group(2)}"
        key = subject.casefold()
        if key in seen:
            continue
        shot = {"expected_subject": subject, "kind": "object"}
        if _shot_subject_semantic_error(shot, scene):
            continue
        seen.add(key)
        output.append(subject)
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

    for match in VERSIONED_PRODUCT_RE.finditer(text):
        add(match.group(0), position=match.start(), bonus=76)

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
            remainder = _semantic_terms(" ".join(raw_tokens[1:]))
            acronym_bonus = 60 if first.isupper() and remainder.issubset(GENERIC_ENTITY_TERMS) else -6
            add(first, position=match.start(), bonus=central_bonus + acronym_bonus)
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


def _mask_bbox_ratios(mask: Image.Image) -> tuple[float, float, float]:
    bbox = mask.getbbox()
    if bbox is None:
        return 0.0, 0.0, 0.0
    width, height = mask.size
    bbox_width = bbox[2] - bbox[0]
    bbox_height = bbox[3] - bbox[1]
    return (
        bbox_width / width,
        bbox_height / height,
        (bbox_width * bbox_height) / (width * height),
    )


def _mask_tile_profile(
    mask: Image.Image,
    pixel_count: int,
    *,
    divisions: int,
) -> tuple[int, set[int], set[int]]:
    width, height = mask.size
    minimum_tile_pixels = max(4, math.ceil(pixel_count * 0.01))
    occupied = 0
    occupied_rows: set[int] = set()
    occupied_columns: set[int] = set()
    for row in range(divisions):
        top = height * row // divisions
        bottom = height * (row + 1) // divisions
        for column in range(divisions):
            left = width * column // divisions
            right = width * (column + 1) // divisions
            tile_pixels = mask.crop((left, top, right, bottom)).histogram()[255]
            if tile_pixels < minimum_tile_pixels:
                continue
            occupied += 1
            occupied_rows.add(row)
            occupied_columns.add(column)
    return occupied, occupied_rows, occupied_columns


def _binary_mask_has_few_runs(mask: Image.Image, *, maximum_runs: int = 4) -> bool:
    bbox = mask.getbbox()
    if bbox is None:
        return False
    sample = mask.crop(bbox)
    sample.thumbnail((256, 256), Image.Resampling.NEAREST)
    width, height = sample.size

    def run_count(values: bytes) -> int:
        runs = 0
        inside = False
        for value in values:
            visible = value >= 128
            if visible and not inside:
                runs += 1
            inside = visible
        return runs

    row_runs = [
        run_count(sample.crop((0, row, width, row + 1)).tobytes())
        for row in range(height)
    ]
    column_runs = [
        run_count(sample.crop((column, 0, column + 1, height)).tobytes())
        for column in range(width)
    ]
    return max(row_runs, default=0) <= maximum_runs and max(
        column_runs,
        default=0,
    ) <= maximum_runs


def _channel_has_diverse_spatial_profiles(
    channel: Image.Image,
    visible_mask: Image.Image | None,
) -> bool:
    field = (
        channel.copy()
        if visible_mask is None
        else ImageChops.multiply(channel, visible_mask)
    )
    if visible_mask is not None:
        bbox = visible_mask.getbbox()
        if bbox is None:
            return False
        field = field.crop(bbox)
    field.thumbnail((256, 256), Image.Resampling.BOX)
    width, height = field.size
    row_slices = [
        field.crop((0, row, width, row + 1)).tobytes()
        for row in range(height)
    ]
    column_slices = [
        field.crop((column, 0, column + 1, height)).tobytes()
        for column in range(width)
    ]
    minimum_row_diversity = max(8, math.ceil(height * 0.25))
    minimum_column_diversity = max(8, math.ceil(width * 0.25))
    return (
        len(set(row_slices)) >= minimum_row_diversity
        and len(set(column_slices)) >= minimum_column_diversity
        and len({sum(row) for row in row_slices}) >= minimum_row_diversity
        and len({sum(column) for column in column_slices})
        >= minimum_column_diversity
    )


def _rgb_detail_is_distributed(
    detail_mask: Image.Image,
    detail_pixels: int,
    *,
    kind: str,
    color_levels: int,
    channel: Image.Image,
    visible_mask: Image.Image | None,
) -> bool:
    bbox_width, bbox_height, bbox_area = _mask_bbox_ratios(detail_mask)
    bbox = detail_mask.getbbox()
    if bbox is None:
        return False
    bbox_pixels = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    bbox_fill = detail_pixels / bbox_pixels
    if bbox_fill < 0.1:
        return False
    if kind == "logo":
        if (
            bbox_area < LOGO_SHAPE_MIN_BBOX_AREA
            or bbox_width < 0.2
            or bbox_height < 0.2
        ):
            return False
        _, occupied_rows, occupied_columns = _mask_tile_profile(
            detail_mask,
            detail_pixels,
            divisions=3,
        )
        if len(occupied_rows) < 2 or len(occupied_columns) < 2:
            return False
    elif bbox_width < 0.2 or bbox_height < 0.2:
        return False
    occupied, _, _ = _mask_tile_profile(detail_mask, detail_pixels, divisions=2)
    if occupied < 2:
        return False
    if color_levels <= 4:
        return (
            bbox_fill >= 0.8 and _binary_mask_has_few_runs(detail_mask)
        ) or (
            kind == "logo"
            and _alpha_mask_has_distinctive_shape(detail_mask, detail_pixels)
        )
    return color_levels >= 16 and _channel_has_diverse_spatial_profiles(
        channel,
        visible_mask,
    )


def _rgb_has_visible_content(
    decoded: Image.Image,
    *,
    mask: Image.Image | None,
    minimum_detail: int,
    kind: str,
) -> bool:
    rgb = decoded.convert("RGB")
    for channel in (*rgb.split(), rgb.convert("L")):
        histogram = channel.histogram(mask)
        occupied = [value for value, count in enumerate(histogram) if count]
        if not occupied or occupied[-1] - occupied[0] < 8:
            continue
        dominant = max(range(256), key=histogram.__getitem__)
        detail_mask = channel.point(
            lambda value: 0 if dominant - 2 <= value <= dominant + 2 else 255
        )
        if mask is not None:
            detail_mask = ImageChops.multiply(detail_mask, mask)
        detail_pixels = detail_mask.histogram()[255]
        if detail_pixels >= minimum_detail and _rgb_detail_is_distributed(
            detail_mask,
            detail_pixels,
            kind=kind,
            color_levels=len(occupied),
            channel=channel,
            visible_mask=mask,
        ):
            return True
    return False


def _alpha_mask_has_distinctive_shape(mask: Image.Image, visible_pixels: int) -> bool:
    """Separate a real monochrome mark from a solid block, strip, or blank canvas."""
    bbox = mask.getbbox()
    if bbox is None:
        return False
    bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    _, _, bbox_area_ratio = _mask_bbox_ratios(mask)
    foreground_coverage = visible_pixels / (mask.width * mask.height)
    bbox_fill = visible_pixels / bbox_area if bbox_area else 1.0
    if (
        not bbox_area
        or bbox_area_ratio < LOGO_SHAPE_MIN_BBOX_AREA
        or foreground_coverage < LOGO_SHAPE_MIN_FOREGROUND_COVERAGE
        or bbox_fill >= LOGO_SHAPE_MAX_BBOX_FILL
    ):
        return False
    _, occupied_rows, occupied_columns = _mask_tile_profile(
        mask,
        visible_pixels,
        divisions=3,
    )
    if len(occupied_rows) < 2 or len(occupied_columns) < 2:
        return False

    simple_tiles, _, _ = _mask_tile_profile(
        mask,
        visible_pixels,
        divisions=2,
    )
    simple_geometry = (
        bbox_fill >= LOGO_SHAPE_MIN_BBOX_FILL
        and simple_tiles == 4
        and _binary_mask_has_few_runs(mask, maximum_runs=4)
    )

    sample = mask.copy()
    sample.thumbnail(
        (LOGO_SHAPE_ANALYSIS_MAX_DIMENSION, LOGO_SHAPE_ANALYSIS_MAX_DIMENSION),
        Image.Resampling.NEAREST,
    )
    width, height = sample.size
    sample_visible = sample.histogram()[255]
    if not sample_visible:
        return False
    horizontal_transitions = 0
    if width > 1:
        horizontal_transitions = ImageChops.difference(
            sample.crop((1, 0, width, height)),
            sample.crop((0, 0, width - 1, height)),
        ).histogram()[255]
    vertical_transitions = 0
    if height > 1:
        vertical_transitions = ImageChops.difference(
            sample.crop((0, 1, width, height)),
            sample.crop((0, 0, width, height - 1)),
        ).histogram()[255]
    scale = math.sqrt(sample_visible)
    complex_edges = not (
        horizontal_transitions / scale < LOGO_SHAPE_MIN_AXIS_EDGE_COMPLEXITY
        or vertical_transitions / scale < LOGO_SHAPE_MIN_AXIS_EDGE_COMPLEXITY
    )
    row_slices = [
        sample.crop((0, row, width, row + 1)).tobytes()
        for row in range(height)
    ]
    column_slices = [
        sample.crop((column, 0, column + 1, height)).tobytes()
        for column in range(width)
    ]
    complex_geometry = (
        complex_edges
        and len(set(row_slices)) >= LOGO_SHAPE_MIN_AXIS_PATTERNS
        and len(set(column_slices)) >= LOGO_SHAPE_MIN_AXIS_PATTERNS
        and len({row.count(255) for row in row_slices})
        >= LOGO_SHAPE_MIN_AXIS_PROJECTIONS
        and len({column.count(255) for column in column_slices})
        >= LOGO_SHAPE_MIN_AXIS_PROJECTIONS
    )
    return simple_geometry or complex_geometry


def _raster_has_visible_content(decoded: Image.Image, *, kind: str = "event") -> bool:
    """Reject empty rasters while ignoring arbitrary RGB stored under transparency."""
    width, height = decoded.size
    total = width * height
    minimum_detail = max(1024, math.ceil(total * 0.01))
    alpha = None
    if "A" in decoded.getbands():
        alpha = decoded.getchannel("A")
    elif decoded.mode == "P" and "transparency" in decoded.info:
        alpha = decoded.convert("RGBA").getchannel("A")
    if alpha is not None:
        mask = alpha.point(
            lambda value: 255 if value >= ALPHA_FOREGROUND_THRESHOLD else 0
        )
        strong_visible = mask.histogram()[255]
        minimum_foreground = (
            max(256, math.ceil(total * 0.001))
            if kind == "logo"
            else minimum_detail
        )
        if strong_visible < minimum_foreground:
            return False
        bbox_width, bbox_height, _ = _mask_bbox_ratios(mask)
        if kind != "logo" and (
            strong_visible / total < 0.05
            or bbox_width < 0.2
            or bbox_height < 0.2
        ):
            return False
        rgb_foreground_is_large_enough = (
            kind != "logo" or strong_visible / total >= 0.05
        )
        if rgb_foreground_is_large_enough and _rgb_has_visible_content(
            decoded,
            mask=mask,
            minimum_detail=minimum_detail,
            kind=kind,
        ):
            return True
        return kind == "logo" and _alpha_mask_has_distinctive_shape(mask, strong_visible)

    return _rgb_has_visible_content(
        decoded,
        mask=None,
        minimum_detail=minimum_detail,
        kind=kind,
    )


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
            decoded.load()
            if not _raster_has_visible_content(
                decoded,
                kind=str(image.get("kind") or "event"),
            ):
                return False
    except (Image.DecompressionBombError, OSError, SyntaxError, UnidentifiedImageError, ValueError):
        return False
    return True


def _validated_collage_sets(
    task_dir: Path,
    manifest: dict,
    scenes_by_id: dict[str, dict],
) -> dict[str, list[dict]] | None:
    """Validate optional same-scene supporting images for fullscreen collages."""
    images = manifest.get("images") or []
    primary_by_scene = {
        str(image.get("scene_id") or ""): image
        for image in images
        if isinstance(image, dict)
    }
    raw_sets = manifest.get("collage_sets") or []
    if not isinstance(raw_sets, list) or any(not isinstance(item, dict) for item in raw_sets):
        return None
    try:
        expected_count = int(manifest.get("collage_asset_count") or 0)
    except (TypeError, ValueError, OverflowError):
        return None
    output: dict[str, list[dict]] = {}
    seen_sources = {
        str(image.get("source_page_url") or "") for image in images if isinstance(image, dict)
    }
    seen_hashes = {
        str(image.get("sha256") or "") for image in images if isinstance(image, dict)
    }
    seen_paths = {
        str(image.get("local_path") or "") for image in images if isinstance(image, dict)
    }
    actual_count = 0
    for item in raw_sets:
        scene_id = str(item.get("scene_id") or "")
        primary = primary_by_scene.get(scene_id)
        assets = item.get("assets") or []
        if (
            not scene_id
            or scene_id in output
            or not primary
            or primary.get("display_mode") != "fullscreen"
            or str(item.get("primary_image_id") or "") != str(primary.get("id") or "")
            or not isinstance(assets, list)
            or len(assets) != 2
            or any(not isinstance(asset, dict) for asset in assets)
        ):
            return None
        scene = scenes_by_id.get(scene_id)
        if scene is None:
            return None
        verified: list[dict] = []
        for asset in assets:
            source = str(asset.get("source_page_url") or "")
            digest = str(asset.get("sha256") or "")
            local_path = str(asset.get("local_path") or "")
            if (
                str(asset.get("scene_id") or "") != scene_id
                or not source
                or source in seen_sources
                or not digest
                or digest in seen_hashes
                or not local_path
                or local_path in seen_paths
                or not image_grounding_is_valid(asset, scene)
                or not _cached_asset_is_intact(task_dir, asset)
            ):
                return None
            seen_sources.add(source)
            seen_hashes.add(digest)
            seen_paths.add(local_path)
            verified.append(asset)
            actual_count += 1
        output[scene_id] = verified
    return output if actual_count == expected_count else None


def image_grounding_is_valid(image: dict, scene: dict) -> bool:
    """Recompute policy proof instead of trusting manifest booleans."""
    raw_expected_subject = " ".join(str(image.get("expected_subject") or "").split())
    expected_subject = _normalise_entity_phrase(image.get("expected_subject"))
    caption = " ".join(str(image.get("caption") or "").split())
    if (
        image.get("grounding_policy_version") != QUERY_SEMANTICS_VERSION
        or image.get("grounding_passed") is not True
        or not expected_subject
        or raw_expected_subject != expected_subject
        or caption != expected_subject
        or not _is_open_license(str(image.get("license") or ""), str(image.get("license_code") or ""))
        or not _candidate_source_is_still_image(image)
        or not _candidate_rights_are_clear(image)
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
        and list(image.get("grounding_context_conflicts") or [])
        == evidence["grounding_context_conflicts"]
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
        or manifest.get("status") not in {"ready", "partial"}
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
    if not images or len(images) > planned:
        return None
    modes = _placement_mode_counts(images)
    if manifest.get("placement_modes") != modes:
        return None
    if len(images) >= 2 and (not modes["inline"] or not modes["fullscreen"]):
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
    status = str(manifest.get("status") or "")
    image_scene_ids = [str(image.get("scene_id") or "") for image in images]
    raw_missing = manifest.get("missing_scene_ids") or []
    if not isinstance(raw_missing, list):
        return None
    missing = [str(item) for item in raw_missing]
    if status == "ready":
        if len(images) != planned or missing:
            return None
    elif (
        len(images) >= planned
        or len(missing) != planned - len(images)
        or len(set(missing)) != len(missing)
        or not set(missing).issubset(eligible)
        or set(missing) & set(image_scene_ids)
    ):
        return None
    scenes_by_id = {
        str(scene.get("id") or ""): scene for scene in (storyboard or {}).get("scenes") or []
    }
    seen_scenes: set[str] = set()
    seen_sources: set[str] = set()
    seen_hashes: set[str] = set()
    seen_paths: set[str] = set()
    seen_identities: set[str] = set()
    for image in images:
        scene_id = str(image.get("scene_id") or "")
        source = str(image.get("source_page_url") or "")
        digest = str(image.get("sha256") or "")
        local_path = str(image.get("local_path") or "")
        anchors = image.get("grounding_distinctive_anchors") or []
        identity = _subject_identity_key(image)
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
            or not identity
            or identity in seen_identities
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
        seen_identities.add(identity)
    if _validated_collage_sets(task_dir, manifest, scenes_by_id) is None:
        return None
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
        leading_display = subject_display_terms[0] if subject_display_terms else ""
        leading_key = leading_display.casefold()
        if len(subject) > 1 and (
            leading_display.isupper()
            or any(char.isdigit() for char in leading_display)
            or leading_key in {"qwen", "qianwen"}
        ):
            raw.append(f"{subject_display_terms[0]} logo")
    else:
        if any(term.casefold() in {"mouse", "mice"} for term in subject + scene_words):
            raw.extend(["laboratory mouse embryo", "CRISPR laboratory mouse"])
        if any(term.casefold() in {"rocket", "launch", "falcon"} for term in subject + scene_words):
            raw.extend(["Falcon 9 rocket launch", "SpaceX Falcon 9 launch"])
        raw.extend(
            [
                subject_text,
                f"{subject_text} photo",
                f"{subject_text} event",
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
    count: int | None,
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
        if count is not None and len(output) >= count:
            break

    if not output and count is not None:
        raise ValueError("News image planner returned no usable scene assignments")
    _ensure_placement_mode_mix(output)
    return output


def _ensure_placement_mode_mix(
    rows: list[dict],
    *,
    locked_fullscreen_scene_ids: set[str] | None = None,
) -> None:
    """Keep every multi-image result usable by both supported compositions."""
    if len(rows) < 2:
        return
    locked = locked_fullscreen_scene_ids or set()
    modes = {str(item.get("display_mode") or "") for item in rows}
    if "inline" not in modes:
        row = next(
            (
                item
                for item in rows
                if str(item.get("scene_id") or "") not in locked
            ),
            rows[0],
        )
        row["display_mode"] = "inline"
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
    role, reason = _subject_semantic_role(subject, scene)
    if reason:
        return ""
    return {
        "brand": "logo",
        "object": "object",
        "person": "person",
        "place": "place",
        "product": "product",
    }.get(role, "")


def _deterministic_query(subject: str, kind: str) -> str:
    suffix = {
        "event": "event",
        "logo": "logo",
        "person": "portrait",
        "place": "photo",
        "product": "product",
        "object": "photo",
    }.get(kind, "photo")
    return _sanitize_query(f"{subject} {suffix}")


def _fallback_shots_for_scene(
    scene: dict,
    hint: dict | None = None,
    *,
    display_mode: str = "inline",
) -> list[dict]:
    entities = [(subject, _entity_kind(subject, scene)) for subject in _entity_candidates(scene, hint)]
    objects = [(subject, "object") for subject in _concrete_object_candidates(scene)]
    # The central named entity remains the first choice.  Put exact narrated
    # objects immediately behind it so source/publication names do not consume
    # every reserve slot before a concrete visual such as ``coffee shop``.
    subjects = [*entities[:1], *objects, *entities[1:]]
    if not subjects:
        fallback = _fallback_subject(scene, hint)
        subjects = [(fallback, _entity_kind(fallback, scene))]
    output: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for subject, kind in subjects:
        if len(output) >= 5:
            break
        if not kind:
            continue
        query = _deterministic_query(subject, kind)
        signature = (subject.casefold(), query.casefold())
        if signature in seen or not _distinctive_terms(subject):
            continue
        shot = {
            "scene_id": str(scene.get("id") or ""),
            "search_query": _sanitize_query(query),
            "news_query": _sanitize_query(subject, 14),
            "expected_subject": subject[:100],
            "kind": kind,
            "display_mode": display_mode,
            "purpose": "Deterministic visual derived from a narrated semantic subject",
            "caption": subject[:120],
        }
        if _shot_subject_semantic_error(shot, scene):
            continue
        seen.add(signature)
        output.append(shot)
    return output


def grounded_visual_subject_count(scene: dict, hint: dict | None = None) -> int:
    """Return how many strict licensed-image subjects the scene can support."""
    return len(_fallback_shots_for_scene(scene, hint))


def _subject_identity_key(shot: dict) -> str:
    ordered = [
        term
        for term in _ordered_semantic_terms(shot.get("expected_subject"))
        if term not in GENERIC_QUERY_TERMS and term != "ai"
    ]
    return SUBJECT_IDENTITY_ALIASES.get(ordered[0], ordered[0]) if ordered else ""


def _collage_support_candidates(
    candidates: list[dict],
    *,
    excluded_sources: set[str] | None = None,
    allow_logo: bool = False,
) -> list[dict]:
    """Return usable support sources while retaining per-subject fallbacks.

    Three charts that all happen to mention data centres are not a visual
    collage. A collage must combine different concepts grounded in the same
    narration, such as the company, robots, and data centres. Keep alternate
    sources for each concept until download/decode succeeds; distinctness is
    enforced when an asset is accepted below.
    """
    excluded = excluded_sources or set()
    output: list[dict] = []
    used_sources: set[str] = set()
    for candidate in candidates:
        is_logo = candidate.get("kind") == "logo"
        if is_logo and not allow_logo:
            continue
        # A fullscreen photo collage must stay photographic.  Charts, maps,
        # diagrams, and icons can be grounded to a phrase while still
        # replacing the story with unrelated explanatory content (for
        # example a regional data-centre count chart in a robotics story).
        visual_metadata = " ".join(
            str(candidate.get(field) or "")
            for field in ("title", "description", "categories", "object_name")
        ).casefold()
        if not is_logo and any(
            marker in visual_metadata for marker in BAD_PHOTO_MARKERS
        ):
            continue
        source = str(candidate.get("source_page_url") or "")
        identity = _subject_identity_key(candidate)
        if (
            not source
            or source in excluded
            or source in used_sources
            or not identity
        ):
            continue
        output.append(candidate)
        used_sources.add(source)
    return output


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
            None,
        )
        if chosen is None:
            continue
        selected[scene_id] = chosen
        identity = _subject_identity_key(chosen)
        if identity:
            used_identities.add(identity)
    return [selected[str(scene.get("id") or "")] for scene in scenes if str(scene.get("id") or "") in selected]


def _fallback_plan(scenes: list[dict], count: int, scene_hints: dict[str, dict] | None = None) -> list[dict]:
    output = _allocate_unique_scene_shots(
        scenes,
        scene_hints=scene_hints,
    )[:count]
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
    identity_owners = {
        identity: str(shot.get("scene_id") or "")
        for shot in primary
        if (identity := _subject_identity_key(shot))
    }
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
            identity = _subject_identity_key(variant)
            owner = identity_owners.get(identity) if identity else None
            # A concrete object can truthfully recur in separate narrated
            # stories (for example ``data centers`` in both a robotics story
            # and an infrastructure-fund story). Keep primary identities
            # unique, but let the repeated object remain available as a
            # same-scene collage support; global source matching still keeps
            # the actual licensed images distinct.
            if owner and owner != scene_id and variant.get("kind") != "object":
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
            if identity:
                identity_owners.setdefault(identity, scene_id)
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
    return not _shot_subject_semantic_error(shot, scene)


def _structured_scene_named_subject(
    scene: dict,
    hint: dict | None,
    *,
    display_mode: str,
) -> dict | None:
    """Prefer an exact named subject when the scene already carries facts.

    A generic object can be literally present in a narration while still being
    a poor editorial replacement for the scene's structured stat or list.  For
    example, a data-centre inventory chart is not a faithful full-screen visual
    for an Andreessen Horowitz fund announcement merely because the portfolio
    list mentions data centres.  Keep the structured explanation and use the
    central named organisation, person, or product as supporting imagery.
    """
    hint = hint or {}
    has_structured_facts = bool(
        str(hint.get("stat") or "").strip()
        or [item for item in hint.get("items") or [] if str(item).strip()]
    )
    if not has_structured_facts:
        return None
    return next(
        (
            shot
            for shot in _fallback_shots_for_scene(
                scene,
                hint,
                display_mode=display_mode,
            )
            if shot.get("kind") in {"logo", "person", "product"}
        ),
        None,
    )


def _prepare_primary_plan(
    plan: list[dict],
    scenes: list[dict],
    target: int,
    scene_hints: dict[str, dict] | None = None,
) -> list[dict]:
    """Replace hallucinated or generic planner rows with narration entities."""
    scenes_by_id = {str(scene.get("id") or ""): scene for scene in scenes}
    preferred: dict[str, dict] = {}
    for shot in plan:
        scene_id = str(shot.get("scene_id") or "")
        scene = scenes_by_id.get(scene_id)
        if scene is None:
            continue
        normalized = {
            **shot,
            "expected_subject": _normalise_entity_phrase(shot.get("expected_subject")),
        }
        if _shot_is_grounded_to_scene(normalized, scene):
            if str(normalized.get("kind") or "") == "object":
                named_subject = _structured_scene_named_subject(
                    scene,
                    (scene_hints or {}).get(scene_id) or {},
                    display_mode=str(normalized.get("display_mode") or "inline"),
                )
                if named_subject is not None:
                    normalized = named_subject
            normalized["search_query"] = _deterministic_query(
                str(normalized["expected_subject"]),
                str(normalized.get("kind") or ""),
            )
            normalized["news_query"] = _sanitize_query(normalized["expected_subject"], 14)
            normalized["caption"] = str(normalized["expected_subject"])[:120]
            normalized["purpose"] = "Licensed visual grounded to the narrated semantic subject"
            preferred[scene_id] = normalized
    output = _allocate_unique_scene_shots(
        scenes,
        preferred,
        scene_hints,
    )[:target]
    _ensure_placement_mode_mix(output)
    return output[:target]


def _planner_prompt(
    scenes: list[dict],
    count: int | None,
    scene_hints: dict[str, dict] | None = None,
) -> str:
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
    quantity_direction = (
        "Choose the natural number of eligible scenes that genuinely deserve a real still image. "
        "There is no quota: omit scenes better served by motion graphics, collage, or footage. "
        if count is None
        else f"Choose exactly {count} different eligible scenes that deserve a real still image. "
    )
    return (
        "You are the picture editor for a factual technology-news video. "
        + quantity_direction
        + "Prefer an exact "
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
    count: int | None,
    scene_hints: dict[str, dict] | None = None,
    log: LogCallback | None = None,
) -> tuple[list[dict], str, str]:
    scenes_by_id = {str(scene.get("id") or ""): scene for scene in storyboard.get("scenes") or []}
    scenes = [scenes_by_id[scene_id] for scene_id in eligible_scene_ids if scene_id in scenes_by_id]
    automatic = count is None
    target = min(max(0, count), len(scenes)) if count is not None else None
    if target == 0 or not scenes:
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
        planned = _normalise_plan(
            first_json(response),
            eligible_scene_ids=eligible_scene_ids,
            count=target,
        )
        plan = planned if automatic else _complete_plan(planned, scenes, target, scene_hints)
        target = len(plan) if automatic else target
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
        fallback_target = len(scenes) if automatic else target
        return _fallback_plan(scenes, fallback_target, scene_hints), "deterministic-fallback", ""


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
        f"{candidate.get('title', '')} {candidate.get('description', '')} "
        f"{candidate.get('categories', '')} {candidate.get('object_name', '')} "
        f"{candidate.get('creator', '')}"
    )
    scene_has_identity = bool(subject_terms) and _contains_token_phrase(
        _identity_tokens(scene_text), subject_terms
    )
    candidate_match = _candidate_identity_match(shot, candidate)
    context_conflicts = _candidate_context_conflicts(scene, candidate)
    semantic_error = _shot_subject_semantic_error(shot, scene)
    # Some one-word product names are also unrelated, established brands.
    # A bare Commons identity match is not enough for these names: for example,
    # a consumer-goods company named Hermes must not illustrate the Hermes AI
    # coding agent.  Requiring one additional narrated context term rejects the
    # ambiguity while still allowing a candidate whose metadata says what the
    # product actually is.
    ambiguous_context_terms = {
        "hermes": {"ai", "agent", "code", "coding", "software", "model"},
    }
    required_context = ambiguous_context_terms.get(
        subject_terms[0] if len(subject_terms) == 1 else "",
        set(),
    )
    context_terms = (scene_terms & candidate_terms & required_context)
    passed = True
    reason = "complete subject identity matched narration and one candidate metadata field"
    if semantic_error:
        passed = False
        reason = semantic_error
    elif not subject_terms:
        passed = False
        reason = "shot subject had no usable complete identity"
    elif not scene_has_identity:
        passed = False
        reason = "shot subject was not a contiguous identity in the narrated scene"
    elif candidate_match is None:
        passed = False
        reason = "candidate did not contain the complete identity in one metadata field"
    elif required_context and not context_terms:
        passed = False
        reason = (
            "ambiguous single-token identity lacked matching AI/product context in "
            "candidate metadata"
        )
    elif context_conflicts:
        passed = False
        reason = (
            "candidate media-work context conflicted with the narrated entity: "
            + ", ".join(context_conflicts)
        )

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
        "grounding_context_conflicts": context_conflicts,
    }


def _embedded_scene(shot: dict) -> dict:
    return {
        "text": str(shot.get("_scene_text") or ""),
        "keywords": list(shot.get("_scene_keywords") or []),
    }


def _raster_resource_mime(info: dict) -> str:
    """Return the MIME type of the bytes we will download, never the source SVG."""
    thumbnail_url = str(info.get("thumburl") or "")
    download_url = thumbnail_url or str(info.get("url") or "")
    suffix = Path(urlsplit(download_url).path).suffix.casefold()
    inferred = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(suffix, "")
    advertised = str(info.get("thumbmime") or info.get("mime") or "").casefold()
    if inferred not in SUPPORTED_MIME_TYPES:
        return ""
    if advertised in SUPPORTED_MIME_TYPES:
        return inferred
    # Exact Commons File lookups frequently omit ``thumbmime`` for an SVG
    # while still returning a raster ``thumburl`` such as ``.svg.png``.
    # Inference is allowed only for that thumbnail resource; a raw SVG URL
    # remains unsupported and can never be written to disk.
    if thumbnail_url:
        return inferred
    return ""


def _candidate_from_page(page: dict, shot: dict) -> dict | None:
    info = (page.get("imageinfo") or [None])[0]
    if not isinstance(info, dict):
        return None
    mime = _raster_resource_mime(info)
    if mime not in SUPPORTED_MIME_TYPES:
        return None
    source_mime = str(info.get("mime") or "").casefold()
    # Commons exposes raster thumbnails for PDFs and videos.  Those bytes are
    # technically decodable images, but a page-one document or arbitrary video
    # frame is not a factual still of the narrated subject.  Fail closed at the
    # candidate boundary and keep only sources that are themselves images;
    # SVG/TIFF originals remain eligible through their raster thumbnails.
    if not source_mime.startswith("image/"):
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
        "categories": _clean_html(_metadata_value(metadata, "Categories")),
        "object_name": _clean_html(_metadata_value(metadata, "ObjectName")),
        "width": width,
        "height": height,
        "mime_type": mime,
        "source_mime_type": source_mime,
        "kind": shot.get("kind"),
    }
    if not _candidate_rights_are_clear(candidate):
        return None
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
    # Exact Commons File references are still subjected to the same license,
    # grounding, raster, and decoder gates as generator-search results.  Once
    # they pass those gates, try them first: they are the research model's
    # explicit source selection.  Search candidates remain in the pool so a
    # broken or rate-limited thumbnail can be rematched without losing the
    # scene.
    discovery_priority = (
        0 if candidate.get("discovery_method") == "commons_file_reference" else 1
    )
    return (discovery_priority, -score, -pixels, candidate["source_page_url"])


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


async def _wikimedia_query(client: httpx.AsyncClient, params: dict[str, object]) -> dict:
    response = None
    for attempt in range(WIKIMEDIA_SEARCH_ATTEMPTS):
        response = await client.get(WIKIMEDIA_API, params=params)
        if response.status_code != 429:
            break
        if attempt + 1 < WIKIMEDIA_SEARCH_ATTEMPTS:
            await asyncio.sleep(_wikimedia_retry_delay(response, attempt))
    assert response is not None
    response.raise_for_status()
    value = response.json()
    return value if isinstance(value, dict) else {}


def _commons_file_title(reference: dict) -> str:
    """Extract one exact Commons File title without trusting an arbitrary URL."""
    parsed = urlsplit(str(reference.get("url") or ""))
    if parsed.scheme.casefold() != "https" or parsed.hostname != "commons.wikimedia.org":
        return ""
    path = unquote(parsed.path)
    prefix = "/wiki/File:"
    if not path.startswith(prefix):
        return ""
    filename = path[len(prefix) :].replace("_", " ").strip()
    if not filename or len(filename) > 240 or "/" in filename or "|" in filename:
        return ""
    return f"File:{filename}"


def _known_commons_file_titles(shot: dict) -> tuple[str, ...]:
    key = (
        _normalise_entity_phrase(shot.get("expected_subject")).casefold(),
        str(shot.get("kind") or "").casefold(),
    )
    return KNOWN_COMMONS_FILE_TITLES.get(key, ())


async def resolve_wikimedia_reference_images(
    client: httpx.AsyncClient,
    *,
    references: list[dict],
    shot: dict,
) -> list[dict]:
    """Resolve exact Commons File references missed by generator search ranking."""
    titles = list(
        dict.fromkeys(
            [
                *_known_commons_file_titles(shot),
                *(
                    title
                    for reference in references
                    if isinstance(reference, dict) and (title := _commons_file_title(reference))
                ),
            ]
        )
    )
    if not titles:
        return []
    payload = await _wikimedia_query(
        client,
        {
            "action": "query",
            "titles": "|".join(titles),
            "redirects": 1,
            "prop": "imageinfo",
            "iiprop": "url|extmetadata|mime|size",
            "iiurlwidth": 3840,
            "format": "json",
            "formatversion": 2,
        },
    )
    pages = (payload.get("query") or {}).get("pages") or []
    candidates = [
        candidate
        for page in pages
        if isinstance(page, dict) and (candidate := _candidate_from_page(page, shot)) is not None
    ]
    return sorted(candidates, key=lambda candidate: _candidate_rank(candidate, shot))


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
        "iiurlwidth": 3840,
        "format": "json",
        "formatversion": 2,
    }
    payload = await _wikimedia_query(client, params)
    pages = (payload.get("query") or {}).get("pages") or []
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
    for attempt in range(WIKIMEDIA_DOWNLOAD_ATTEMPTS):
        digest = hashlib.sha256()
        written = 0
        retry_delay = None
        async with client.stream("GET", candidate["download_url"]) as response:
            if response.status_code == 429 or 500 <= response.status_code < 600:
                if attempt + 1 < WIKIMEDIA_DOWNLOAD_ATTEMPTS:
                    retry_delay = _wikimedia_retry_delay(response, attempt)
                else:
                    response.raise_for_status()
            else:
                response.raise_for_status()
            if retry_delay is None:
                response_mime = (
                    str(response.headers.get("content-type") or "")
                    .split(";", 1)[0]
                    .strip()
                    .casefold()
                )
                if response_mime and response_mime not in SUPPORTED_MIME_TYPES:
                    raise ValueError(
                        f"remote image returned unsupported content type {response_mime}"
                    )
                advertised = int(response.headers.get("content-length") or 0)
                if advertised and advertised > config.NEWS_IMAGE_MAX_BYTES:
                    raise ValueError(
                        f"remote image is {advertised} bytes, above the download limit"
                    )
                with destination.open("wb") as handle:
                    async for chunk in response.aiter_bytes():
                        written += len(chunk)
                        if written > config.NEWS_IMAGE_MAX_BYTES:
                            raise ValueError(
                                "download exceeded the autonomous image byte limit"
                            )
                        digest.update(chunk)
                        handle.write(chunk)
        if retry_delay is not None:
            destination.unlink(missing_ok=True)
            await asyncio.sleep(retry_delay)
            continue
        if written < 1024:
            raise ValueError("downloaded image is unexpectedly small")
        return written, digest.hexdigest()
    raise RuntimeError("Wikimedia image download exhausted retry attempts")


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

    def add_candidates(
        raw_candidates: list[dict],
        *,
        resolved_query: str,
        discovery_method: str,
    ) -> None:
        for raw_candidate in raw_candidates:
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
                "resolved_search_query": resolved_query,
                "discovery_method": discovery_method,
            }
            candidates_by_source.setdefault(source, candidate)

    reference_lookup = {
        **shot,
        "_scene_text": scene.get("text") or "",
        "_scene_keywords": scene.get("keywords") or [],
    }
    try:
        add_candidates(
            await resolve_wikimedia_reference_images(
                client,
                references=references,
                shot=reference_lookup,
            ),
            resolved_query=str(shot.get("search_query") or ""),
            discovery_method="commons_file_reference",
        )
    except Exception as exc:  # noqa: BLE001 - generator search remains usable
        manifest["errors"].append(
            {
                "scene_id": scene_id,
                "stage": "wikimedia_reference",
                "message": str(exc),
            }
        )

    # A metadata-valid exact reference can still fail at download or decode
    # time.  Keep generator-search alternatives in the same global candidate
    # pool so failed_sources can trigger a complete rematch rather than leave
    # the scene with no usable inventory.
    try:
        for query in _wikimedia_query_variants(shot, scene):
            lookup = {
                **shot,
                "search_query": query,
                "_scene_text": scene.get("text") or "",
                "_scene_keywords": scene.get("keywords") or [],
            }
            add_candidates(
                await search_wikimedia_images(client, shot=lookup),
                resolved_query=query,
                discovery_method="commons_search",
            )
            if len(candidates_by_source) >= 8:
                break
    except Exception as exc:  # noqa: BLE001 - exact references remain usable
        manifest["errors"].append(
            {
                "scene_id": scene_id,
                "stage": "wikimedia_search",
                "message": str(exc),
            }
        )
    return sorted(
        candidates_by_source.values(),
        key=lambda candidate: _candidate_rank(candidate, shot),
    )


def _next_asset_destination(image_dir: Path, start: int, extension: str) -> tuple[int, Path]:
    serial = max(1, start)
    while any((image_dir / f"image-{serial:02d}{suffix}").exists() for suffix in (".jpg", ".jpeg", ".png", ".webp", ".svg")):
        serial += 1
    return serial, image_dir / f"image-{serial:02d}{extension}"


@timed("news_images")
async def acquire_news_images(
    storyboard: dict,
    task_dir: Path,
    *,
    count: int | None,
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
    automatic = count is None
    preplanned: list[dict] | None = None
    planner = ""
    conversation_url = ""
    if automatic:
        # Determine the AI quantity from the same frozen plan on recovery.
        # Previously this web request ran before the asset cache check, changing
        # its own target count and discarding otherwise valid downloaded images.
        scenes = {str(s.get("id") or ""): s for s in storyboard.get("scenes") or []}
        planning_prompt = _planner_prompt([scenes[s] for s in eligible], None, scene_hints)
        plan_key = hashlib.sha256(json.dumps(
            [MANIFEST_VERSION, QUERY_SEMANTICS_VERSION, planning_prompt],
            ensure_ascii=False,
        ).encode()).hexdigest()
        plan_path = image_dir / f"picture-plan-{plan_key}.json"
        if plan_path.is_symlink():
            raise ValueError("News image plan cache must not be a symbolic link")
        try:
            saved = json.loads(plan_path.read_text(encoding="utf-8"))
            if saved["prompt"] == planning_prompt and saved["planner"] == "opencli:chatgpt-picture-editor":
                preplanned = _normalise_plan({"images": saved["images"]},
                                             eligible_scene_ids=eligible, count=None)
                if preplanned != saved["images"]:
                    preplanned = None
                else:
                    planner, conversation_url = saved["planner"], saved["conversation_url"]
                    _emit(log, "News images: reusing exact prompt-matched picture-editor plan")
        except (OSError, ValueError, KeyError, TypeError):
            preplanned = None
        if preplanned is None:
            preplanned, planner, conversation_url = await plan_news_images(
                storyboard,
                eligible_scene_ids=eligible,
                count=None,
                scene_hints=scene_hints,
                log=log,
            )
            if planner == "opencli:chatgpt-picture-editor":
                # A provider fallback is never frozen as an authoritative plan.
                plan_path.write_text(json.dumps({"prompt": planning_prompt,
                    "images": preplanned, "planner": planner,
                    "conversation_url": conversation_url}, ensure_ascii=False), encoding="utf-8")
        target = len(preplanned)
    else:
        target = min(max(0, count), len(eligible))
    requested_count = target if automatic else count
    contract_sha256 = acquisition_contract_fingerprint(
        storyboard_sha256=fingerprint,
        requested_count=requested_count,
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
        expected_requested_count=requested_count,
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
        for stale_set in (previous_manifest or {}).get("collage_sets") or []:
            if not isinstance(stale_set, dict):
                continue
            for stale_image in stale_set.get("assets") or []:
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
        "selection_mode": "ai" if automatic else "explicit",
        "requested_image_count": requested_count,
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
        "license_allowlist": ["Public Domain", "CC0", "CC BY", "CC BY-SA", "Apache-2.0"],
        "queries": [],
        "images": [],
        "collage_sets": [],
        "collage_asset_count": 0,
        "errors": [],
        "placement_modes": {"inline": 0, "fullscreen": 0},
    }
    _write_manifest(task_dir, manifest)
    if not target:
        manifest["status"] = "not_needed"
        manifest["updated_at"] = _now()
        _write_manifest(task_dir, manifest)
        return manifest

    plan = preplanned
    if plan is None:
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
                            "source_mime_type": candidate.get("source_mime_type")
                            or candidate.get("mime_type"),
                            "download_mime_type": candidate.get("mime_type"),
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
                            **image_layout(destination, kind=candidate["kind"]),
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

            # Discover the unused reserve subjects for every acquired scene
            # before deciding where the collage belongs.  Selection matching
            # may already be complete after the primary queries; without this
            # explicit pass the two supporting images would never be searched.
            staged_scene_ids = {
                str(image.get("scene_id") or "") for image in staged
            }
            for position, reserve_shot in list(reserve_rows):
                if str(reserve_shot.get("scene_id") or "") not in staged_scene_ids:
                    continue
                reserve_rows.remove((position, reserve_shot))
                await discover(position, reserve_shot)

            # Use two additional, independently licensed images from the same
            # narration scene. Prefer an already-fullscreen photo, but permit a
            # logo primary when two photo/object supports exist (for example a
            # company logo plus robots and a data center). If necessary an
            # inline primary is promoted to the copy-free fullscreen treatment.
            primary_candidates = [
                image
                for image in sorted(
                    staged,
                    key=lambda image: (
                        0
                        if image.get("display_mode") == "fullscreen"
                        and image.get("kind") != "logo"
                        else (1 if image.get("display_mode") == "fullscreen" else 2)
                    ),
                )
                # A scene whose authored template already carries a stat or a
                # factual list keeps that evidence. Collage promotion is for
                # otherwise simple image/topic scenes only.
                if not (
                    str(
                        ((scene_hints or {}).get(str(image.get("scene_id") or "")) or {}).get(
                            "stat"
                        )
                        or ""
                    ).strip()
                    or [
                        item
                        for item in (
                            (
                                (scene_hints or {}).get(
                                    str(image.get("scene_id") or "")
                                )
                                or {}
                            ).get("items")
                            or []
                        )
                        if str(item).strip()
                    ]
                )
            ]
            fullscreen_primary: dict | None = None
            collage_assets: list[dict] = []
            used_primary_sources = {
                str(image.get("source_page_url") or "") for image in staged
            }
            used_primary_hashes = {str(image.get("sha256") or "") for image in staged}
            for primary in primary_candidates:
                collage_scene_id = str(primary.get("scene_id") or "")
                available_supports = _collage_support_candidates(
                    candidate_pools.get(collage_scene_id) or [],
                    excluded_sources=used_primary_sources | failed_sources,
                    allow_logo=primary.get("kind") != "logo",
                )
                if len(
                    {
                        _subject_identity_key(candidate)
                        for candidate in available_supports
                        if _subject_identity_key(candidate)
                    }
                ) < 2:
                    continue
                attempt_assets: list[dict] = []
                used_sources = set(used_primary_sources)
                used_hashes = set(used_primary_hashes)
                used_support_subjects: set[str] = {
                    identity
                    for identity in (_subject_identity_key(primary),)
                    if identity
                }
                logo_support_used = False
                for candidate in available_supports:
                    if len(attempt_assets) >= 2:
                        break
                    source = str(candidate.get("source_page_url") or "")
                    subject_identity = _subject_identity_key(candidate)
                    is_logo = candidate.get("kind") == "logo"
                    if (
                        not source
                        or source in used_sources
                        or not subject_identity
                        or subject_identity in used_support_subjects
                        or (is_logo and logo_support_used)
                    ):
                        continue
                    success_number, destination = _next_asset_destination(
                        image_dir,
                        target + len(attempt_assets) + 1,
                        _extension_for(candidate),
                    )
                    try:
                        byte_size, sha256 = await _download_candidate(
                            client,
                            candidate=candidate,
                            destination=destination,
                        )
                    except Exception as exc:  # noqa: BLE001 - try another support
                        destination.unlink(missing_ok=True)
                        manifest["errors"].append(
                            {
                                "scene_id": collage_scene_id,
                                "stage": "collage_download",
                                "source_page_url": source,
                                "message": str(exc),
                            }
                        )
                        continue
                    asset_record = {
                        "local_path": destination.relative_to(task_dir).as_posix(),
                        "bytes": byte_size,
                        "sha256": sha256,
                        "kind": candidate.get("kind"),
                    }
                    if sha256 in used_hashes or not _cached_asset_is_intact(
                        task_dir, asset_record
                    ):
                        destination.unlink(missing_ok=True)
                        continue
                    supporting = dict(
                        _public_value(
                            {
                                "id": f"image-{success_number:02d}",
                                **candidate,
                                "scene_id": collage_scene_id,
                                "display_mode": "fullscreen",
                                "resolved_search_query": candidate.get(
                                    "resolved_search_query"
                                )
                                or candidate.get("search_query"),
                                "source_mime_type": candidate.get("source_mime_type")
                                or candidate.get("mime_type"),
                                "download_mime_type": candidate.get("mime_type"),
                                "mime_type": {
                                    ".jpg": "image/jpeg",
                                    ".jpeg": "image/jpeg",
                                    ".png": "image/png",
                                    ".webp": "image/webp",
                                    ".svg": "image/svg+xml",
                                }.get(
                                    destination.suffix.lower(),
                                    "application/octet-stream",
                                ),
                                "match_terms": list(
                                    candidate["grounding_distinctive_anchors"]
                                ),
                                "bytes": byte_size,
                                "sha256": sha256,
                                "local_path": destination.relative_to(
                                    task_dir
                                ).as_posix(),
                                **image_layout(destination, kind=candidate.get("kind", "")),
                                "status": "downloaded",
                                "collage_role": "supporting",
                            }
                        )
                    )
                    if not image_grounding_is_valid(
                        supporting, scenes_by_id[collage_scene_id]
                    ):
                        destination.unlink(missing_ok=True)
                        continue
                    attempt_assets.append(supporting)
                    used_sources.add(source)
                    used_hashes.add(sha256)
                    used_support_subjects.add(subject_identity)
                    logo_support_used = logo_support_used or is_logo

                if len(attempt_assets) == 2:
                    primary["display_mode"] = "fullscreen"
                    fullscreen_primary = primary
                    collage_assets = attempt_assets
                    break
                for asset in attempt_assets:
                    stale = _generated_asset_path(task_dir, asset.get("local_path"))
                    if stale is not None:
                        stale.unlink(missing_ok=True)

            if len(collage_assets) == 2 and fullscreen_primary is not None:
                manifest["collage_sets"] = [
                    {
                        "scene_id": fullscreen_primary["scene_id"],
                        "primary_image_id": fullscreen_primary["id"],
                        "assets": collage_assets,
                    }
                ]
                manifest["collage_asset_count"] = 2
                _emit(
                    log,
                    "News image collage: 3 same-scene licensed images ready for "
                    f"{fullscreen_primary['scene_id']}",
                )
            else:
                for asset in collage_assets:
                    stale = _generated_asset_path(task_dir, asset.get("local_path"))
                    if stale is not None:
                        stale.unlink(missing_ok=True)

            manifest["images"] = staged
            for downloaded in staged:
                _emit(
                    log,
                    f"News image acquired for {downloaded['scene_id']}: {downloaded['title']} "
                    f"({downloaded['license']}, {downloaded['display_mode']})",
                )
            break

    candidate_inventory = {}
    for scene_id in priority_scenes:
        candidates = candidate_pools.get(scene_id) or []
        usable_resources = {
            (
                f"sha256:{source_hashes[source]}"
                if source in source_hashes
                else f"source:{source}"
            )
            for candidate in candidates
            if (source := str(candidate.get("source_page_url") or ""))
            and source not in failed_sources
        }
        candidate_inventory[scene_id] = {
            "candidate_count": len(candidates),
            "usable_candidate_count": len(usable_resources),
            "failed_candidate_count": sum(
                str(candidate.get("source_page_url") or "") in failed_sources
                for candidate in candidates
            ),
            "subjects": sorted(
                {
                    str(candidate.get("expected_subject") or "")
                    for candidate in candidates
                    if str(candidate.get("expected_subject") or "")
                }
            ),
            "discovery_methods": sorted(
                {
                    str(candidate.get("discovery_method") or "commons_search")
                    for candidate in candidates
                }
            ),
        }
    manifest["candidate_inventory"] = candidate_inventory
    selected_scene_ids = {
        str(image.get("scene_id") or "") for image in manifest["images"]
    }
    missing_count = max(0, target - len(manifest["images"]))
    missing_scene_ids = [
        scene_id
        for scene_id in priority_scenes
        if scene_id not in selected_scene_ids
    ][:missing_count]
    manifest["missing_scene_ids"] = missing_scene_ids
    if len(manifest["images"]) < target:
        for scene_id in missing_scene_ids:
            if candidate_inventory[scene_id]["usable_candidate_count"]:
                continue
            manifest["errors"].append(
                {
                    "scene_id": scene_id,
                    "stage": "inventory",
                    "message": (
                        "No grounded open-license Commons candidate remained after "
                        "exact File references and search fallbacks"
                    ),
                }
            )
        manifest["errors"].append(
            {
                "stage": "selection",
                "message": (
                    "No complete unique scene/source/content assignment satisfied "
                    f"the grounding contract; missing scenes: {', '.join(missing_scene_ids)}"
                ),
                "missing_scene_ids": missing_scene_ids,
                "candidate_counts": {
                    scene_id: details["candidate_count"]
                    for scene_id, details in candidate_inventory.items()
                },
                "usable_candidate_counts": {
                    scene_id: details["usable_candidate_count"]
                    for scene_id, details in candidate_inventory.items()
                },
            }
        )
        _emit(
            log,
            "News image inventory incomplete: missing "
            f"{', '.join(missing_scene_ids)}; usable/discovered candidates="
            + ", ".join(
                f"{scene_id}:{details['usable_candidate_count']}/{details['candidate_count']}"
                for scene_id, details in candidate_inventory.items()
            ),
        )

    acquired = len(manifest["images"])
    _ensure_placement_mode_mix(
        manifest["images"],
        locked_fullscreen_scene_ids={
            str(item.get("scene_id") or "")
            for item in manifest.get("collage_sets") or []
            if isinstance(item, dict)
        },
    )
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
        or manifest.get("status") not in {"ready", "partial"}
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
        or len(images) > planned
        or manifest.get("placement_modes") != inventory_modes
        or (
            len(images) >= 2
            and (not inventory_modes["inline"] or not inventory_modes["fullscreen"])
        )
    ):
        return {"attached": 0, "placement_modes": {"inline": 0, "fullscreen": 0}}
    status = str(manifest.get("status") or "")
    image_scene_ids = [str(image.get("scene_id") or "") for image in images]
    raw_missing = manifest.get("missing_scene_ids") or []
    if not isinstance(raw_missing, list):
        return {"attached": 0, "placement_modes": {"inline": 0, "fullscreen": 0}}
    missing = [str(item) for item in raw_missing]
    if status == "ready":
        valid_status_shape = len(images) == planned and not missing
    else:
        valid_status_shape = (
            len(images) < planned
            and len(missing) == planned - len(images)
            and len(set(missing)) == len(missing)
            and set(missing).issubset(eligible)
            and not (set(missing) & set(image_scene_ids))
        )
    if not valid_status_shape:
        return {"attached": 0, "placement_modes": {"inline": 0, "fullscreen": 0}}
    by_id = {str(plan.get("id") or ""): plan for plan in plans}
    scenes_by_id = {
        str(scene.get("id") or ""): scene for scene in storyboard.get("scenes") or []
    }
    collage_sets = _validated_collage_sets(task_dir, manifest, scenes_by_id)
    if collage_sets is None:
        return {"attached": 0, "placement_modes": {"inline": 0, "fullscreen": 0}}
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
        supporting = collage_sets.get(scene_id) or []
        # A fullscreen logo replaces the original scene composition.  For a
        # structured stat/list scene that discards the strongest narrated
        # evidence and can leave a sparse transparent mark filling the frame.
        # Keep the logo as an inline source card so the verified stat/items
        # remain visible; other photo-rich/fullscreen scenes preserve the
        # required placement-mode mix.
        if (
            mode == "fullscreen"
            and str(image.get("kind") or "") == "logo"
            and (plan.get("stat") or plan.get("items"))
            and len(supporting) != 2
        ):
            mode = "inline"
        plan.update(
            {
                "news_image": True,
                # Scene HTML lives under compositions/, one directory below the
                # manifest and downloaded assets.
                "news_image_src": f"../{image['local_path']}",
                "news_image_mode": mode,
                "news_image_kind": image.get("kind") or "event",
                "news_image_fit": image_layout(
                    task_dir / image["local_path"], fit=image.get("fit", "cover"),
                    kind=image.get("kind", ""),
                )["fit"],
                "news_image_credit": _credit(image),
                "news_image_caption": image.get("expected_subject") or "",
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
                "news_image_grounding_context_conflicts": image.get("grounding_context_conflicts") or [],
                "news_image_title": image.get("title") or "",
                "news_image_description": image.get("description") or "",
                "news_image_categories": image.get("categories") or "",
                "news_image_object_name": image.get("object_name") or "",
                "news_image_creator": image.get("creator") or "",
                "news_image_license": image.get("license") or "",
                "news_image_license_code": image.get("license_code") or "",
                "news_image_source_mime_type": image.get("source_mime_type") or "",
                "news_image_sha256": digest,
                "news_image_reference_count": len(image.get("references") or []),
            }
        )
        if mode == "fullscreen" and len(supporting) == 2:
            plan["news_image_srcs"] = [
                f"../{image['local_path']}",
                *[f"../{asset['local_path']}" for asset in supporting],
            ]
            plan["news_image_credits"] = [
                _credit(image),
                *[_credit(asset) for asset in supporting],
            ]
            plan["news_image_fits"] = [
                image_layout(
                    task_dir / asset["local_path"], fit=asset.get("fit", "cover"),
                    kind=asset.get("kind", ""),
                )["fit"]
                for asset in [image, *supporting]
            ]
            plan["news_image_collage"] = True
            plan["news_image_collage_asset_count"] = 3
        if mode == "fullscreen":
            plan["news_image_original_archetype"] = plan.get("archetype") or "topic"
            plan["archetype"] = "news_image"
        attached += 1
        attached_scenes.add(scene_id)
        attached_sources.add(source_page_url)
        attached_hashes.add(digest)
        modes[mode] += 1
    return {"attached": attached, "placement_modes": modes}

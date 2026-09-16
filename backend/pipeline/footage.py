"""AI-planned, license-gated public B-roll acquisition.

The media scout intentionally starts with Wikimedia Commons: it works without
an API key and exposes creator/license/source-page metadata per file. Search
results are accepted only when their license is explicitly on the open-license
allowlist. Every downloaded clip is recorded in ``footage/manifest.json`` so a
human can review provenance before the video is published.
"""

from __future__ import annotations

import asyncio
import hashlib
import html
import json
import logging
import re
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from backend import config
from backend.pipeline.digester import _chat, _resolve_provider

logger = logging.getLogger(__name__)
LogCallback = Callable[[str], None]

WIKIMEDIA_API = "https://commons.wikimedia.org/w/api.php"
OPEN_LICENSE_MARKERS = (
    "public domain",
    "cc0",
    "cc by",
    "cc-by",
    "cc by-sa",
    "cc-by-sa",
)
UNSAFE_LICENSE_MARKERS = ("noncommercial", "no derivatives", "-nc", "-nd")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'-]{1,}")
TAG_RE = re.compile(r"<[^>]+>")
SAFE_FILENAME_RE = re.compile(r"[^a-z0-9]+")
USER_QUERY_PURPOSE = "User-supplied search direction"
PURPOSE_MATCH_STOPWORDS = frozenset(
    "a an and are as at be by for from in into is it of on or that the this to with".split()
)
FALLBACK_QUERY_STOPWORDS = PURPOSE_MATCH_STOPWORDS | frozenset(
    """
    about after again also because been before being between can could did does
    doing during each final finally first from gave get gets got had has have
    having here how if its itself just many may might more most next not now off
    once only other our out over own really report reported reports reporting
    said says should since some still story than their them then there these they
    think through today tomorrow too under unlike very was watching were what when
    where which while who will would your
    good morning concise briefing frontier tech daily journal magazine outlet
    according thanks subscribe
    documentary footage video scene
    january february march april may june july august september october november december
    those chinese-language
    """.split()
)


def _emit(log: LogCallback | None, message: str) -> None:
    if log:
        log(message)
    else:
        logger.info(message)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_html(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(html.unescape(TAG_RE.sub(" ", value)).split())


def _metadata_value(metadata: dict, key: str) -> str:
    raw = metadata.get(key) or {}
    if isinstance(raw, dict):
        return str(raw.get("value") or "")
    return str(raw or "")


def _is_open_license(short_name: str, license_code: str = "") -> bool:
    combined = f"{short_name} {license_code}".strip().lower()
    if not combined:
        return False
    if any(marker in combined for marker in UNSAFE_LICENSE_MARKERS):
        return False
    return any(marker in combined for marker in OPEN_LICENSE_MARKERS)


def _strip_json_fence(value: str) -> str:
    clean = value.strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[-1]
        clean = clean.rsplit("```", 1)[0]
    return clean.strip()


def _sanitize_query(value: str) -> str:
    words = WORD_RE.findall(value)
    return " ".join(words[:6]).strip()


def _script_purpose_for_query(query: str, script: str) -> str:
    """Ground a manual search direction in its best matching script sentence."""
    def match_term(word: str) -> str:
        term = word.casefold()
        if term.endswith("ese") and len(term) > 7:
            term = term[:-3]
        elif term.endswith("ied") and len(term) > 5:
            term = term[:-3] + "y"
        elif term.endswith("ed") and len(term) > 5:
            term = term[:-2]
        elif term.endswith("s") and not term.endswith("ss") and len(term) > 4:
            term = term[:-1]
        return term

    def terms(value: str) -> set[str]:
        # Hyphenated modifiers still contain the entity being searched:
        # catgirl-themed must anchor "catgirl", not lose to a generic "AI".
        return {
            match_term(part)
            for word in WORD_RE.findall(value)
            for part in (word, *re.split(r"[-']", word))
            if len(part) >= 2 and part.casefold() not in PURPOSE_MATCH_STOPWORDS
        }

    query_terms = terms(query)
    ordered = [match_term(word) for word in WORD_RE.findall(query)
               if word.casefold() not in PURPOSE_MATCH_STOPWORDS]
    weights = {word: len(ordered) - index for index, word in reversed(list(enumerate(ordered)))}
    if not query_terms:
        return ""
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<!\b[A-Z]\.)(?<=[.!?。！？])\s+", script)
        if sentence.strip()
    ]
    scored = []
    for index, sentence in enumerate(sentences):
        sentence_terms = terms(sentence)
        overlap = query_terms & sentence_terms
        if overlap:
            scored.append((len(overlap), sum(weights.get(word, 1) for word in overlap), -index, sentence))
    return max(scored, default=(0, 0, 0, ""))[3]


def _storyboard_scene_purposes(task_dir: Path) -> dict[str, str]:
    path = task_dir / "storyboard.json"
    if not path.is_file():
        return {}
    try:
        storyboard = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        str(scene.get("id") or ""): str(scene.get("text") or "").strip()
        for scene in storyboard.get("scenes") or []
        if isinstance(scene, dict) and scene.get("id") and scene.get("text")
    }


def _closest_storyboard_purpose(purpose: str, scenes: dict[str, str]) -> str:
    """Expand a sentence-level asset purpose to its complete storyboard scene."""
    terms = {word.casefold() for word in WORD_RE.findall(purpose)}
    best_overlap = 0
    best = ""
    for scene_text in scenes.values():
        scene_terms = {word.casefold() for word in WORD_RE.findall(scene_text)}
        overlap = len(terms & scene_terms)
        if overlap > best_overlap:
            best_overlap = overlap
            best = scene_text
    return best if best_overlap >= 2 else ""


def _reserved_collage_purposes(task_dir: Path) -> list[str]:
    """Return full narration scenes already bound to ready collage clips."""
    path = task_dir / "collage_broll" / "manifest.json"
    if not path.is_file():
        return []
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    scenes = _storyboard_scene_purposes(task_dir)
    purposes: list[str] = []
    for item in manifest.get("items") or []:
        if not isinstance(item, dict) or item.get("status") != "ready":
            continue
        scene_id = str(item.get("scene_id") or "")
        purpose = scenes.get(scene_id) or str(
            (item.get("spec") or {}).get("script_meaning") or ""
        ).strip()
        if purpose:
            purposes.append(purpose)
    return purposes


def _purpose_conflicts_with_reserved(purpose: str, reserved: list[str]) -> bool:
    """Detect when two assets are semantically locked to the same narration."""
    terms = {word.casefold() for word in WORD_RE.findall(purpose)}
    if not terms:
        return False
    for value in reserved:
        other = {word.casefold() for word in WORD_RE.findall(value)}
        if not other:
            continue
        overlap = len(terms & other)
        if overlap >= 2 and overlap / min(len(terms), len(other)) >= 0.65:
            return True
    return False


def _unoccupied_web_plan(query_plan: list[dict[str, str]], clips: list[dict]) -> list[dict[str, str]]:
    """Do not spend the web quota twice on the scene Commons already filled."""
    occupied = [
        str(clip.get("script_excerpt") or clip.get("purpose") or "").strip()
        for clip in clips
        if isinstance(clip, dict)
        and str(clip.get("script_excerpt") or clip.get("purpose") or "").strip()
    ]
    return [
        shot
        for shot in query_plan
        if not _purpose_conflicts_with_reserved(
            str(shot.get("script_excerpt") or shot.get("purpose") or "").strip(),
            occupied,
        )
    ]


def _parse_plan(value: str, count: int | None) -> list[dict[str, str]]:
    parsed = json.loads(_strip_json_fence(value))
    raw_queries = parsed.get("queries") if isinstance(parsed, dict) else None
    if not isinstance(raw_queries, list):
        raise ValueError("Footage planner did not return a queries array")

    output: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in raw_queries:
        if isinstance(item, str):
            query = _sanitize_query(item)
            purpose = ""
        elif isinstance(item, dict):
            query = _sanitize_query(str(item.get("query") or ""))
            purpose = str(item.get("purpose") or "").strip()
        else:
            continue
        key = query.lower()
        if not query or key in seen:
            continue
        seen.add(key)
        output.append({"query": query, "purpose": purpose})
        if isinstance(item, dict) and item.get("script_excerpt"):
            output[-1]["script_excerpt"] = str(item["script_excerpt"]).strip()
        if count is not None and len(output) >= count:
            break
    if not output and count is not None:
        raise ValueError("Footage planner returned no usable queries")
    return output


def _is_program_bookend(text: str) -> bool:
    value = " ".join(text.casefold().replace("’", "'").split())
    return (
        (value.startswith("it's ") and "bytefront espresso" in value)
        or value.startswith("that's today's bytefront espresso")
    )


def _search_subject_text(text: str) -> str:
    """Remove attribution without turning the publisher/date into the subject."""
    text = re.sub(
        r"^.{0,120}?\b(?:reports?|reported|says?|writes?|profiles?|revisits?)\b"
        r"(?:\s+on\s+[A-Za-z]+\s+\d{1,2})?(?:\s+(?:that|how))?[,;:\s-]*",
        "", text, count=1, flags=re.IGNORECASE,
    )
    return re.sub(
        r"^.{0,140}?\b(?:interview|conversation)\s+with\s+",
        "", text, count=1, flags=re.IGNORECASE,
    )


def _fallback_search_queries(shot: dict, *, excluded: set[str]) -> list[str]:
    """Propose intact narrated entity phrases, never a bag of paragraph leads.

    The sentence binding takes priority over the longer editorial purpose.
    This only proposes searches; metadata and real-frame review still decide
    whether a candidate is usable.
    """
    boundaries = FALLBACK_QUERY_STOPWORDS | frozenset(
        "one two three four five six seven eight nine ten hundred thousand million billion "
        "run runs cost costs roughly each like built backed participant relays "
        "filed confidentially offers example firm citing alleges described build builds".split()
    )
    phrases: list[tuple[int, str]] = []
    seen = {query.casefold() for query in excluded}
    for text in dict.fromkeys([shot.get("script_excerpt") or "", shot.get("purpose") or ""]):
        words: list[str] = []

        def flush() -> None:
            if len(words) >= 2:
                query = _sanitize_query(" ".join(words))
                key = query.casefold()
                if key not in seen:
                    seen.add(key)
                    # Product identifiers are stronger than introductory names.
                    priority = 2 * int(any(any(c.isdigit() for c in word) for word in words))
                    priority += int(any(word[:1].isupper() for word in words))
                    phrases.append((priority, query))
            words.clear()

        for token in re.findall(r"[A-Za-z][A-Za-z0-9'-]*|[.!?;—]", _search_subject_text(str(text))):
            word = token.removesuffix("'s").strip("'-")
            if word.casefold() in {"a", "an", "the"}:
                continue
            if not word or word.casefold() in boundaries or token in ".!?;—":
                flush()
            else:
                words.append(word)
        flush()
        # Prefer the exact bound sentence before reaching into story context.
        if len(phrases) >= 2:
            break
    return [query for _, query in sorted(phrases, key=lambda item: -item[0])][:2]


def _fallback_plan(title: str, script: str, count: int | None) -> list[dict[str, str]]:
    """Build story-specific queries when the model plan cannot be decoded.

    Daily-news scripts use one nonblank paragraph per editorial story.  The old
    fallback ranked words across the whole script, which elevated connective
    language such as ``and reports can`` and downloaded unrelated documentary
    footage.  Plan per story instead, prefer rare/proper-name anchors, and keep
    the exact narration paragraph as the placement purpose.
    """

    paragraphs = [
        " ".join(part.split())
        for part in re.split(r"\n\s*\n|\n+", script)
        if part.strip()
    ]
    if len(paragraphs) <= 1:
        paragraphs = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?。！？])\s+", script)
            if sentence.strip()
        ]

    def useful_words(value: str) -> list[str]:
        output: list[str] = []
        seen: set[str] = set()
        for word in WORD_RE.findall(value):
            key = word.casefold().strip("'-")
            if (
                key in FALLBACK_QUERY_STOPWORDS
                or key in seen
                or (len(key) < 3 and word not in {"AI", "Ig"})
                or len(key) > 32
            ):
                continue
            seen.add(key)
            output.append(word.strip("'-"))
        return output

    candidates: list[tuple[str, list[str]]] = []
    for paragraph in paragraphs:
        lowered = paragraph.casefold()
        if (
            _is_program_bookend(paragraph)
            or ("good morning" in lowered and "briefing" in lowered)
            or "thanks for watching" in lowered
            or "subscribe for more" in lowered
        ):
            continue
        # Strip a source-attribution lead ("QbitAI reports that ...") so the
        # fallback searches for the depicted subject instead of the publisher.
        subject_text = _search_subject_text(paragraph)
        words = useful_words(subject_text)
        if len(words) >= 2:
            candidates.append((paragraph, words))

    # If a non-news script was filtered too aggressively, retain any concrete
    # paragraph rather than returning no search plan at all.
    if not candidates:
        candidates = [
            (paragraph, words)
            for paragraph in paragraphs
            if len(words := useful_words(paragraph)) >= 2
        ]

    document_frequency: dict[str, int] = {}
    for _, words in candidates:
        for key in {word.casefold() for word in words}:
            document_frequency[key] = document_frequency.get(key, 0) + 1

    output: list[dict[str, str]] = []
    seen_queries: set[str] = set()
    for paragraph, words in candidates:
        indexed = list(enumerate(words))
        term_frequency: dict[str, int] = {}
        for word in words:
            key = word.casefold()
            term_frequency[key] = len(
                re.findall(rf"(?<![a-z0-9]){re.escape(key)}(?![a-z0-9])", paragraph.casefold())
            )

        def salience(item: tuple[int, str]) -> tuple[int, int, int, int, int]:
            index, word = item
            key = word.casefold()
            proper_name = int(word[:1].isupper())
            technical = int("-" in word or any(char.isdigit() for char in word))
            rarity = -document_frequency.get(key, 1)
            return proper_name, technical, rarity, -index, term_frequency[key]

        selected = sorted(indexed, key=salience, reverse=True)[:6]
        selected.sort(key=lambda item: item[0])
        query = _sanitize_query(" ".join(word for _, word in selected))
        if len(query.split()) < 2 or query.casefold() in seen_queries:
            continue
        seen_queries.add(query.casefold())
        output.append({"query": query, "purpose": paragraph})
        if count is not None and len(output) >= count:
            break
    return output


def _distinct_grounded_plan(
    plan: list[dict[str, str]],
    script: str,
    count: int | None,
    *,
    excluded_purposes: list[str] | None = None,
    keep_ungrounded: bool = False,
    allow_same_scene: bool = False,
) -> list[dict[str, str]]:
    """Bind each search query to a different narration segment.

    A visually plausible query is not sufficient evidence for placement.  The
    exact script sentence is persisted with the query, and duplicate queries
    for the same story are discarded before acquisition.  This prevents two
    clips planned for one story from being forced onto an unrelated second
    scene merely to satisfy the requested clip count.
    """
    bookends = [
        part.strip() for part in re.split(r"\n\s*\n|\n+", script)
        if _is_program_bookend(part)
    ]
    output: list[dict[str, str]] = []
    occupied = list(excluded_purposes or [])
    for raw in plan:
        query = _sanitize_query(str(raw.get("query") or ""))
        purpose = str(raw.get("purpose") or "").strip()
        # An exact full-paragraph purpose can contain several unrelated subjects.
        # Select its query-matching sentence, rather than rewarding the longest
        # sentence for overlapping the entire purpose.
        scope = purpose if purpose and purpose in script else script
        if purpose and purpose not in script:
            # The visual query may deliberately use generic objects. First
            # locate the story using the planner's entity-rich purpose, then
            # match the visual within that story rather than the whole edition.
            anchor = _script_purpose_for_query(purpose, script)
            paragraphs = re.split(r"\n\s*\n|\n+", script)
            scope = next((part for part in paragraphs if anchor and anchor in part), script)
        supplied_excerpt = str(raw.get("script_excerpt") or "").strip()
        excerpt = (
            supplied_excerpt if supplied_excerpt and supplied_excerpt in scope
            else _script_purpose_for_query(query, scope)
        )
        if not excerpt:
            excerpt = _script_purpose_for_query(purpose, scope)
        if (
            _is_program_bookend(purpose) or _is_program_bookend(excerpt)
            or (excerpt and any(excerpt in paragraph for paragraph in bookends))
        ):
            continue
        if not query:
            continue
        if not excerpt and keep_ungrounded:
            output.append(
                {
                    "query": query,
                    "purpose": purpose or USER_QUERY_PURPOSE,
                    "script_excerpt": "",
                }
            )
            if count is not None and len(output) >= count:
                break
            continue
        if not excerpt:
            continue
        conflict_pool = (excluded_purposes or []) if allow_same_scene else occupied
        if _purpose_conflicts_with_reserved(excerpt, conflict_pool):
            continue
        output.append(
            {
                "query": query,
                "purpose": purpose or excerpt,
                "script_excerpt": excerpt,
            }
        )
        if not allow_same_scene:
            occupied.append(excerpt)
        if count is not None and len(output) >= count:
            break
    return output


async def plan_footage_queries(
    *,
    title: str,
    script: str,
    count: int | None,
    provider_id: int | None,
    ai_endpoint: str | None,
    ai_model: str | None,
    supplied_queries: list[str] | None = None,
    excluded_purposes: list[str] | None = None,
    log: LogCallback | None = None,
) -> tuple[list[dict[str, str]], str]:
    automatic = count is None
    candidate_count = None if automatic else max(count * 3, count + 2)
    supplied = [
        {
            "query": query,
            "purpose": _script_purpose_for_query(query, script)
            or USER_QUERY_PURPOSE,
        }
        for query in (_sanitize_query(value) for value in supplied_queries or [])
        if query
    ]
    supplied = [
        item
        for item in supplied
        if not _purpose_conflicts_with_reserved(
            str(item.get("purpose") or ""), excluded_purposes or []
        )
    ]
    if supplied:
        grounded = _distinct_grounded_plan(
            supplied,
            script,
            candidate_count,
            excluded_purposes=excluded_purposes,
            keep_ungrounded=True,
            allow_same_scene=automatic,
        )
        return grounded, "user"

    endpoint, model, api_key = await _resolve_provider(provider_id, ai_endpoint, ai_model)
    system_prompt = (config.PROMPTS_DIR / "footage_plan.txt").read_text(encoding="utf-8")
    quantity_direction = (
        "Choose the number of final shots yourself from the narration. Use no quota or fixed "
        "per-edition count. A longer narration beat may receive multiple sequential shots when "
        "one source clip would be too short; otherwise use one shot or none.\n"
        if automatic
        else (
            f"Requested candidate queries: {candidate_count}\n"
            f"Final clips: {count}; every candidate must target a different narration story.\n"
        )
    )
    user_content = (
        quantity_direction
        + f"Title: {title}\n"
        f"Narration meanings already reserved for collage (do not target these): "
        f"{json.dumps(excluded_purposes or [], ensure_ascii=False)}\n"
        f"Narration:\n{script[:12000]}"
    )
    try:
        result = await _chat(
            system_prompt,
            user_content,
            endpoint,
            model,
            api_key,
            log,
            "Footage plan",
            max_tokens=2400,
            enable_skills=False,
            disable_thinking=True,
        )
        plan = _parse_plan(result, candidate_count)
        plan = _distinct_grounded_plan(
            plan,
            script,
            candidate_count,
            excluded_purposes=excluded_purposes,
            allow_same_scene=automatic,
        )
        if not plan and not automatic:
            raise ValueError("Footage planner targeted only collage-reserved narration")
        _emit(log, f"Footage agent planned {len(plan)} visual search queries")
        return plan, f"ai:{model}"
    except Exception as exc:
        _emit(log, f"Footage planning fallback: {exc}")
        fallback = _distinct_grounded_plan(
            _fallback_plan(title, script, candidate_count),
            script,
            candidate_count,
            excluded_purposes=excluded_purposes,
            allow_same_scene=automatic,
        )
        return fallback, "deterministic-fallback"


def _candidate_from_page(page: dict, orientation: str) -> dict | None:
    image_info = (page.get("imageinfo") or [None])[0]
    if not isinstance(image_info, dict):
        return None
    mime = str(image_info.get("mime") or "")
    if not mime.startswith("video/"):
        return None

    metadata = image_info.get("extmetadata") or {}
    license_name = _clean_html(_metadata_value(metadata, "LicenseShortName"))
    license_code = _clean_html(_metadata_value(metadata, "License"))
    if not _is_open_license(license_name, license_code):
        return None

    width = int(image_info.get("width") or 0)
    height = int(image_info.get("height") or 0)
    duration = float(image_info.get("duration") or 0)
    byte_size = int(image_info.get("size") or 0)
    download_url = str(image_info.get("url") or "")
    source_page_url = str(image_info.get("descriptionurl") or "")
    if not download_url or not source_page_url or duration < 2 or byte_size <= 0:
        return None
    if byte_size > config.FOOTAGE_MAX_BYTES:
        return None
    if orientation == "landscape" and width and height and width < height:
        return None
    if orientation == "portrait" and width and height and height < width:
        return None

    attribution_raw = _metadata_value(metadata, "AttributionRequired").lower()
    attribution_required = attribution_raw == "true" or "cc by" in license_name.lower()
    return {
        "provider": "Wikimedia Commons",
        "provider_id": "wikimedia",
        "title": str(page.get("title") or "").removeprefix("File:"),
        "source_page_url": source_page_url,
        "download_url": download_url,
        "creator": _clean_html(_metadata_value(metadata, "Artist")) or "Unknown",
        "license": license_name or license_code,
        "license_code": license_code,
        "license_url": _clean_html(_metadata_value(metadata, "LicenseUrl")),
        "attribution_required": attribution_required,
        "duration_seconds": round(duration, 3),
        "width": width,
        "height": height,
        "bytes": byte_size,
        "mime_type": mime,
        "description": _clean_html(_metadata_value(metadata, "ImageDescription")),
    }


def _rank_candidate(candidate: dict, query: str) -> tuple:
    title = candidate["title"].lower()
    overlap = sum(1 for word in query.lower().split() if word in title)
    pixels = candidate["width"] * candidate["height"]
    # Prefer textual relevance first, then usable resolution, then a smaller
    # download. All terms are deterministic for repeatable agent runs.
    return (-overlap, -pixels, candidate["bytes"], candidate["source_page_url"])


def _candidate_query_is_specific(candidate: dict, query: str) -> bool:
    """Require candidate metadata to prove at least two concrete query anchors."""
    query_terms = {
        word.casefold()
        for word in WORD_RE.findall(query)
        if word.casefold() not in FALLBACK_QUERY_STOPWORDS
    }
    if len(query_terms) < 2:
        return True
    candidate_terms = {
        word.casefold()
        for word in WORD_RE.findall(
            f"{candidate.get('title', '')} {candidate.get('description', '')}"
        )
    }
    return len(query_terms & candidate_terms) >= min(2, len(query_terms))


async def search_wikimedia(
    client: httpx.AsyncClient,
    *,
    query: str,
    orientation: str,
    limit: int = 16,
) -> list[dict]:
    response = await client.get(
        WIKIMEDIA_API,
        params={
            "action": "query",
            "generator": "search",
            "gsrsearch": f"{query} filetype:video",
            "gsrnamespace": 6,
            "gsrlimit": limit,
            "prop": "imageinfo",
            "iiprop": "url|extmetadata|mime|size",
            "format": "json",
            "formatversion": 2,
        },
    )
    response.raise_for_status()
    pages = (response.json().get("query") or {}).get("pages") or []
    candidates = []
    for page in pages:
        candidate = _candidate_from_page(page, orientation)
        if candidate:
            candidates.append(candidate)
    return sorted(candidates, key=lambda item: _rank_candidate(item, query))


def _extension_for(candidate: dict) -> str:
    suffix = Path(urlsplit(candidate["download_url"]).path).suffix.lower()
    if suffix in {".webm", ".ogv", ".ogg", ".mp4", ".mov"}:
        return suffix
    return ".webm" if candidate["mime_type"] == "video/webm" else ".mp4"


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
        if advertised and advertised > config.FOOTAGE_MAX_BYTES:
            raise ValueError(f"remote file is {advertised} bytes, above download limit")
        with destination.open("wb") as handle:
            async for chunk in response.aiter_bytes():
                written += len(chunk)
                if written > config.FOOTAGE_MAX_BYTES:
                    raise ValueError("download exceeded autonomous footage byte limit")
                digest.update(chunk)
                handle.write(chunk)
    return written, digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


async def _normalize_render_clip(source: Path, destination: Path) -> dict:
    """Create the bounded, seek-safe video copy consumed by HyperFrames."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.mp4")
    temporary.unlink(missing_ok=True)
    scale = (
        f"scale=w='min({config.FOOTAGE_RENDER_MAX_WIDTH},iw)':"
        f"h='min({config.FOOTAGE_RENDER_MAX_HEIGHT},ih)':"
        "force_original_aspect_ratio=decrease:flags=lanczos,format=yuv420p"
    )
    process = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-t",
        str(config.FOOTAGE_RENDER_MAX_SECONDS),
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        scale,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        str(config.FOOTAGE_RENDER_CRF),
        "-movflags",
        "+faststart",
        str(temporary),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=300)
    except TimeoutError:
        process.kill()
        await process.communicate()
        temporary.unlink(missing_ok=True)
        raise RuntimeError("footage render normalization timed out after 300s")
    if process.returncode or not temporary.is_file() or temporary.stat().st_size == 0:
        temporary.unlink(missing_ok=True)
        detail = (stderr or stdout).decode(errors="replace")[-1200:]
        raise RuntimeError(
            f"footage render normalization failed ({process.returncode}): {detail}"
        )
    temporary.replace(destination)
    from backend.pipeline.web_footage import _probe

    metadata = await _probe(destination)
    return {
        **metadata,
        "bytes": destination.stat().st_size,
        "sha256": _file_sha256(destination),
        "local_path": destination,
        "render_safe": True,
        "render_metadata_version": 1,
        "render_profile": {
            "container": "mp4",
            "video_codec": "h264",
            "pixel_format": "yuv420p",
            "audio": False,
            "max_width": config.FOOTAGE_RENDER_MAX_WIDTH,
            "max_height": config.FOOTAGE_RENDER_MAX_HEIGHT,
            "max_duration_seconds": config.FOOTAGE_RENDER_MAX_SECONDS,
            "faststart": True,
        },
    }


async def normalize_manifest_clips(
    task_dir: Path,
    manifest: dict | None = None,
    *,
    log: LogCallback | None = None,
) -> dict | None:
    """Migrate legacy Commons downloads to seek-safe render copies in place."""
    current = manifest if isinstance(manifest, dict) else read_manifest(task_dir)
    if not isinstance(current, dict):
        return current
    changed = False
    for clip in current.get("clips") or []:
        if not isinstance(clip, dict):
            continue
        relative = str(clip.get("local_path") or "").strip()
        if not relative:
            continue
        source = (task_dir / relative).resolve()
        if not source.is_file() or task_dir.resolve() not in source.parents:
            continue
        if clip.get("render_safe") is True:
            if clip.get("render_metadata_version") != 1:
                from backend.pipeline.web_footage import _probe

                metadata = await _probe(source)
                clip.setdefault("source_duration_seconds", clip.get("duration_seconds"))
                clip.update(metadata, render_metadata_version=1)
                changed = True
            continue
        clip_id = str(clip.get("id") or source.stem)
        destination = task_dir / "footage" / f"{clip_id}-render.mp4"
        try:
            normalized = await _normalize_render_clip(source, destination)
        except Exception as exc:
            _emit(log, f"Render-safe footage migration failed for {clip_id}: {exc}")
            continue
        clip["source_local_path"] = relative
        clip["source_bytes"] = int(clip.get("bytes") or source.stat().st_size)
        clip["source_sha256"] = str(clip.get("sha256") or _file_sha256(source))
        clip.setdefault("source_duration_seconds", clip.get("duration_seconds"))
        for key in ("duration_seconds", "width", "height", "render_metadata_version"):
            clip[key] = normalized[key]
        clip["bytes"] = normalized["bytes"]
        clip["sha256"] = normalized["sha256"]
        clip["local_path"] = normalized["local_path"].relative_to(task_dir).as_posix()
        clip["render_safe"] = True
        clip["render_profile"] = normalized["render_profile"]
        changed = True
        _emit(log, f"Normalized {clip_id} to seek-safe H.264/1080p footage")
    if changed:
        current["updated_at"] = _now()
        _write_manifest(task_dir, current)
    return current


def manifest_path(task_dir: Path) -> Path:
    return task_dir / "footage" / "manifest.json"


def read_manifest(task_dir: Path) -> dict | None:
    path = manifest_path(task_dir)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def acquisition_is_complete(task_dir: Path, manifest: dict | None, script: str) -> bool:
    """An absent ledger is not an AI decision to request zero clips."""
    if not isinstance(manifest, dict):
        return False
    requested = manifest.get("requested_clip_count")
    clips = manifest.get("clips")
    if (
        type(requested) is not int or requested < 0
        or not isinstance(clips, list) or len(clips) != requested
        or manifest.get("script_sha256") != hashlib.sha256(script.encode()).hexdigest()
        or manifest.get("status") not in {"ready", "no_results"}
        or (requested > 0 and manifest.get("status") != "ready")
        or (requested == 0 and manifest.get("selection_mode") != "ai")
    ):
        return False
    root = task_dir.resolve()
    for clip in clips:
        if not isinstance(clip, dict) or not clip.get("local_path") or not clip.get("sha256"):
            return False
        path = (root / clip["local_path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            return False
        if _file_sha256(path) != clip["sha256"]:
            return False
    return True


def _write_manifest(task_dir: Path, manifest: dict) -> None:
    path = manifest_path(task_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def _next_clip_id(footage_dir: Path, clips: list[dict]) -> str:
    """Allocate an audit-stable ID without overwriting an earlier download."""
    used = {
        str(clip.get("id") or "")
        for clip in clips
        if isinstance(clip, dict) and clip.get("id")
    }
    used.update(path.stem for path in footage_dir.glob("clip-*.*") if path.is_file())
    index = 1
    while f"clip-{index:02d}" in used:
        index += 1
    return f"clip-{index:02d}"


def _reusable_manifest_clips(
    task_dir: Path,
    previous: dict | None,
    *,
    orientation: str,
    script: str,
    reserved_purposes: list[str] | None = None,
    replacement_purposes: list[str] | None = None,
) -> list[dict]:
    """Return only previously downloaded Commons clips whose ledger still holds."""
    if not isinstance(previous, dict) or previous.get("orientation") != orientation:
        return []
    if previous.get("provider_id") != "wikimedia":
        return []
    root = task_dir.resolve()
    scenes = _storyboard_scene_purposes(task_dir)
    reusable: list[dict] = []
    occupied_purposes = list(reserved_purposes or [])
    used_sources: set[str] = set()
    for raw in previous.get("clips") or []:
        if not isinstance(raw, dict) or not _is_open_license(str(raw.get("license") or "")):
            continue
        source = str(raw.get("source_page_url") or "").strip()
        relative = str(raw.get("local_path") or "").strip()
        expected_sha256 = str(raw.get("sha256") or "").strip()
        if not source or source in used_sources or not relative or not expected_sha256:
            continue
        if (
            (raw.get("title") or raw.get("description"))
            and not _candidate_query_is_specific(
                raw,
                str(raw.get("query") or ""),
            )
        ):
            # Keep the original file as audit evidence, but do not reuse a
            # two-word discovery result that matched only the generic half of
            # the query (for example "research vessel" for "student research").
            continue
        path = (task_dir / relative).resolve()
        if path == root or root not in path.parents or not path.is_file():
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected_sha256:
            continue
        clip = dict(raw)
        if str(clip.get("purpose") or "") in {"", USER_QUERY_PURPOSE}:
            grounded_purpose = _script_purpose_for_query(
                str(clip.get("query") or ""),
                script,
            )
            if not grounded_purpose:
                # Retain the file on disk for audit, but do not keep an
                # unplaceable clip in the active manifest. The next re-scout
                # can fill this slot with a script-grounded result.
                continue
            clip["purpose"] = grounded_purpose
        scene_purpose = _closest_storyboard_purpose(
            str(clip.get("purpose") or ""), scenes
        ) or str(clip.get("purpose") or "")
        if _purpose_conflicts_with_reserved(
            scene_purpose,
            replacement_purposes or [],
        ):
            # An explicit re-scout query mapped to this same narration scene.
            # Preserve the file for audit, but free its active slot so the new
            # operator direction can actually replace it.
            continue
        if _purpose_conflicts_with_reserved(scene_purpose, occupied_purposes):
            # A ready collage is already semantically locked to this scene.
            # Keep the downloaded file as audit evidence, but free the active
            # slot so re-scout can acquire footage for a different scene.
            continue
        reusable.append(clip)
        occupied_purposes.append(scene_purpose)
        used_sources.add(source)
    return reusable


async def acquire_public_footage(
    *,
    task_id: str,
    task_dir: Path,
    title: str,
    script_path: Path,
    clip_count: int,
    orientation: str,
    license_policy: str,
    provider_id: int | None,
    ai_endpoint: str | None,
    ai_model: str | None,
    supplied_queries: list[str] | None = None,
    planned_queries: list[dict[str, str]] | None = None,
    log: LogCallback | None = None,
) -> dict:
    """Plan, search, license-check, and download public footage for one task."""
    task_dir.mkdir(parents=True, exist_ok=True)
    footage_dir = task_dir / "footage"
    footage_dir.mkdir(parents=True, exist_ok=True)
    script = script_path.read_text(encoding="utf-8")
    reserved_purposes = _reserved_collage_purposes(task_dir)
    replacement_purposes = [
        purpose
        for value in supplied_queries or []
        if (purpose := _script_purpose_for_query(_sanitize_query(value), script))
    ]

    previous = await normalize_manifest_clips(task_dir, read_manifest(task_dir), log=log)
    preserved_clips = _reusable_manifest_clips(
        task_dir,
        previous,
        orientation=orientation,
        script=script,
        reserved_purposes=reserved_purposes,
        replacement_purposes=replacement_purposes,
    )[:clip_count]
    remaining_count = max(0, clip_count - len(preserved_clips))
    storyboard_scenes = _storyboard_scene_purposes(task_dir)
    occupied_purposes = list(reserved_purposes)
    occupied_purposes.extend(
        _closest_storyboard_purpose(str(clip.get("purpose") or ""), storyboard_scenes)
        or str(clip.get("purpose") or "")
        for clip in preserved_clips
    )

    manifest = {
        "task_id": task_id,
        "status": "planning",
        "created_at": (
            str(previous.get("created_at"))
            if isinstance(previous, dict) and previous.get("created_at")
            else _now()
        ),
        "updated_at": _now(),
        "provider": "Wikimedia Commons",
        "provider_id": "wikimedia",
        "license_policy": "open_only",
        "requested_license_policy": license_policy,
        "license_allowlist": ["Public Domain", "CC0", "CC BY", "CC BY-SA"],
        "orientation": orientation,
        "requested_clip_count": clip_count,
        "planner": "",
        "queries": [],
        "clips": preserved_clips,
        "errors": [],
        "reserved_collage_purposes": reserved_purposes,
        "occupied_scene_purposes": occupied_purposes,
    }
    _write_manifest(task_dir, manifest)

    if remaining_count == 0:
        manifest["status"] = "ready"
        manifest["updated_at"] = _now()
        _write_manifest(task_dir, manifest)
        _emit(log, f"Public footage scout reused {len(preserved_clips)}/{clip_count} verified clips")
        return manifest

    if planned_queries is not None:
        plan = list(planned_queries)
        planner = "ai:auto-quantity"
    else:
        plan, planner = await plan_footage_queries(
            title=title,
            script=script,
            count=remaining_count,
            provider_id=provider_id,
            ai_endpoint=ai_endpoint,
            ai_model=ai_model,
            supplied_queries=supplied_queries,
            excluded_purposes=occupied_purposes,
            log=log,
        )
    manifest["planner"] = planner
    manifest["queries"] = plan
    manifest["status"] = "searching"
    manifest["updated_at"] = _now()
    _write_manifest(task_dir, manifest)

    headers = {"User-Agent": config.FOOTAGE_USER_AGENT}
    used_sources: set[str] = {
        str(clip.get("source_page_url") or "")
        for clip in preserved_clips
        if clip.get("source_page_url")
    }
    async with httpx.AsyncClient(
        timeout=config.FOOTAGE_TIMEOUT,
        follow_redirects=True,
        headers=headers,
    ) as client:
        for index, shot in enumerate(plan, start=1):
            if len(manifest["clips"]) >= clip_count:
                break
            query = shot["query"]
            _emit(log, f"Public footage {index}/{len(plan)}: searching '{query}'")
            try:
                candidates = await search_wikimedia(
                    client,
                    query=query,
                    orientation=orientation,
                )
            except Exception as exc:
                manifest["errors"].append({"query": query, "stage": "search", "message": str(exc)})
                _emit(log, f"Public footage search failed for '{query}': {exc}")
                continue

            available = [
                item for item in candidates
                if item["source_page_url"] not in used_sources
                and _candidate_query_is_specific(item, query)
            ]
            if not available:
                manifest["errors"].append(
                    {
                        "query": query,
                        "stage": "selection",
                        "message": "No unique result passed the open-license, size, and orientation gates",
                    }
                )
                _emit(log, f"No eligible open-license footage found for '{query}'")
                continue

            clip_id = _next_clip_id(footage_dir, manifest["clips"])
            downloaded = None
            for candidate in available[:4]:
                extension = _extension_for(candidate)
                source_destination = footage_dir / f"source-{clip_id}{extension}"
                destination = footage_dir / f"{clip_id}-render.mp4"
                try:
                    source_bytes, source_sha256 = await _download_candidate(
                        client,
                        candidate=candidate,
                        destination=source_destination,
                    )
                    normalized = await _normalize_render_clip(
                        source_destination,
                        destination,
                    )
                except Exception as exc:
                    source_destination.unlink(missing_ok=True)
                    destination.unlink(missing_ok=True)
                    manifest["errors"].append(
                        {
                            "query": query,
                            "stage": "download",
                            "source_page_url": candidate["source_page_url"],
                            "message": str(exc),
                        }
                    )
                    continue

                downloaded = {
                    "id": clip_id,
                    "query": query,
                    "purpose": shot.get("purpose") or "",
                    "script_excerpt": shot.get("script_excerpt") or "",
                    **candidate,
                    "source_duration_seconds": candidate.get("duration_seconds"),
                    "duration_seconds": normalized["duration_seconds"],
                    "width": normalized["width"],
                    "height": normalized["height"],
                    "render_metadata_version": normalized["render_metadata_version"],
                    "source_bytes": source_bytes,
                    "source_sha256": source_sha256,
                    "source_local_path": source_destination.relative_to(task_dir).as_posix(),
                    "bytes": normalized["bytes"],
                    "sha256": normalized["sha256"],
                    "local_path": destination.relative_to(task_dir).as_posix(),
                    "render_safe": True,
                    "render_profile": normalized["render_profile"],
                    "status": "downloaded",
                }
                break

            if downloaded is None:
                _emit(log, f"All eligible downloads failed for '{query}'")
                continue

            used_sources.add(downloaded["source_page_url"])
            manifest["clips"].append(downloaded)
            manifest["updated_at"] = _now()
            _write_manifest(task_dir, manifest)
            _emit(
                log,
                f"Downloaded {downloaded['title']} "
                f"({downloaded['license']}, {downloaded['bytes'] / 1024 / 1024:.1f} MB)",
            )

    manifest["status"] = "ready" if manifest["clips"] else "no_results"
    if manifest["clips"] and len(manifest["clips"]) < clip_count:
        manifest["status"] = "partial"
    manifest["updated_at"] = _now()
    _write_manifest(task_dir, manifest)
    _emit(
        log,
        f"Public footage scout finished: {len(manifest['clips'])}/{clip_count} clips acquired; "
        f"manifest saved to {manifest_path(task_dir)}",
    )
    return manifest


def _resume_web_manifest(
    task_dir: Path, *, provider: str, script: str, orientation: str,
    clip_count: int | None,
) -> dict | None:
    """Resume the same plan only with intact, still-bound reviewed artifacts."""
    previous = read_manifest(task_dir)
    expected_provider = {"hybrid": "hybrid-youtube", "opencli_web": "youtube-web"}.get(provider)
    if not previous or previous.get("provider_id") != expected_provider:
        return None
    if previous.get("orientation") != orientation or not previous.get("queries"):
        return None
    digest = hashlib.sha256(script.encode()).hexdigest()
    old_digest = previous.get("script_sha256")
    if old_digest and old_digest != digest:
        return None
    if clip_count is not None and previous.get("requested_clip_count") != clip_count:
        return None
    if (clip_count is None) != (previous.get("selection_mode") == "ai"):
        return None
    if not old_digest and any(
        not shot.get("script_excerpt") or shot["script_excerpt"] not in script
        for shot in previous["queries"]
    ):
        return None
    # Keep the original ledger before migrating legacy narration bindings.
    snapshot = json.dumps(previous, sort_keys=True, ensure_ascii=False)
    history = task_dir / "footage" / "history"
    history.mkdir(exist_ok=True)
    (history / f"manifest-{hashlib.sha256(snapshot.encode()).hexdigest()[:16]}.json").write_text(
        snapshot, encoding="utf-8",
    )
    for error in previous.get("errors", []):
        message = str(error.get("message") or "")
        if error.get("stage") == "web-download-edit" and message.startswith("OpenCLI "):
            # Older scouts charged an unavailable browser review against the
            # candidate budget. Preserve that audit record without blacklisting
            # a source whose pixels were never adjudicated.
            error["original_stage"] = error["stage"]
            error["stage"] = "web-review"
            if error.get("source_page_url"):
                error["pending_source_page_url"] = error.pop("source_page_url")
    plan = previous["queries"]
    if previous.get("binding_version") != 2:
        old_bindings = {shot["query"].casefold(): shot.get("script_excerpt", "") for shot in plan}
        for error in previous.get("errors", []):
            query = str(error.get("plan_query") or error.get("query") or "").removesuffix(" stock footage").casefold()
            if error.get("source_page_url") and query in old_bindings:
                error.setdefault("script_excerpt", old_bindings[query])
        plan = _distinct_grounded_plan(
            [{k: v for k, v in shot.items() if k != "script_excerpt"} for shot in plan],
            script, None, allow_same_scene=clip_count is None,
        )
        if len(plan) != len(previous["queries"]):
            return None
    by_query = {shot["query"].casefold(): shot for shot in plan}
    root = task_dir.resolve()
    reusable = []
    invalidated = []
    used_sources: set[str] = set()
    used_queries: set[str] = set()
    from backend.pipeline.web_footage import analysis_rejection

    for clip in previous.get("clips") or []:
        if not isinstance(clip, dict):
            continue
        query = str(clip.get("plan_query") or clip.get("query") or "")
        shot = by_query.get(query.casefold()) or by_query.get(query.removesuffix(" stock footage").casefold())
        path = (task_dir / str(clip.get("local_path") or "")).resolve()
        reason = ""
        source = str(clip.get("source_page_url") or "")
        if not shot or clip.get("script_excerpt") != shot.get("script_excerpt"):
            reason = "Narration binding changed; a fresh visual review is required"
        elif not source or source in used_sources or shot["query"] in used_queries:
            reason = "Missing or duplicate source/shot identity"
        elif root not in path.parents or not path.is_file():
            reason = "Local artifact missing or outside task directory"
        elif not clip.get("sha256") or _file_sha256(path) != clip["sha256"]:
            reason = "Artifact checksum mismatch"
        elif (clip.get("platform") == "youtube" or clip.get("provider_id") == "youtube-ytdlp"
              or (clip.get("platform") == "publisher"
                  and clip.get("provider_id") == "publisher-direct"
                  and clip.get("review_required") is True
                  and clip.get("rights_status") == "review_required")):
            analysis = clip.get("analysis") or {}
            reason = analysis_rejection(analysis)
            if (clip.get("platform") == "publisher"
                    and analysis.get("analyzer") != "gemini-web-contact-sheet"):
                reason = "Publisher footage requires actual downloaded-frame review"
            if analysis.get("analyzer") == "gemini-web-contact-sheet" and (
                analysis.get("image_received") is not True
                or analysis.get("suitable") is not True
                or not analysis.get("visible_content")
            ):
                reason = "Missing explicit preview review evidence"
        elif not _is_open_license(str(clip.get("license") or "")):
            reason = "Missing open-license evidence"
        if reason:
            invalidated.append({**clip, "invalidation_reason": reason})
        else:
            reusable.append({**clip, "plan_query": shot["query"]})
            used_sources.add(source)
            used_queries.add(shot["query"])
    previous.setdefault("invalidated_clips", []).extend(invalidated)
    previous.update(
        clips=reusable, queries=plan, script_sha256=digest, binding_version=2,
        status="searching", updated_at=_now(),
    )
    _write_manifest(task_dir, previous)
    return previous


async def acquire_footage(
    *,
    media_provider: str,
    task_id: str,
    task_dir: Path,
    title: str,
    script_path: Path,
    clip_count: int | None,
    orientation: str,
    license_policy: str,
    provider_id: int | None,
    ai_endpoint: str | None,
    ai_model: str | None,
    supplied_queries: list[str] | None = None,
    log: LogCallback | None = None,
) -> dict:
    """Route one task through Wikimedia-only or the hybrid web scout.

    Hybrid mode intentionally keeps at least one Commons clip when possible,
    then fills the remaining slots from YouTube.
    """
    provider = (media_provider or "wikimedia").strip().lower()
    script = script_path.read_text(encoding="utf-8")
    automatic = clip_count is None
    if provider in {"hybrid", "opencli_web"} and config.WEB_FOOTAGE_ENABLED and not supplied_queries:
        resumed = _resume_web_manifest(
            task_dir, provider=provider, script=script, orientation=orientation,
            clip_count=clip_count,
        )
        if resumed is not None:
            fulfilled = {clip["plan_query"].casefold() for clip in resumed["clips"]}
            pending = [shot for shot in resumed["queries"] if shot["query"].casefold() not in fulfilled]
            _emit(log, f"Public footage resume: reused {len(resumed['clips'])} verified clips; {len(pending)} planned shots remain")
            from backend.pipeline.web_footage import supplement_web_footage

            return await supplement_web_footage(
                task_dir=task_dir, manifest=resumed, query_plan=pending,
                target_total=resumed["requested_clip_count"], orientation=orientation,
                script=script, log=log, provider_id=provider_id,
                ai_endpoint=ai_endpoint, ai_model=ai_model,
            )
    automatic_plan: list[dict[str, str]] | None = None
    automatic_planner = ""
    if automatic:
        automatic_plan, automatic_planner = await plan_footage_queries(
            title=title,
            script=script,
            count=None,
            provider_id=provider_id,
            ai_endpoint=ai_endpoint,
            ai_model=ai_model,
            supplied_queries=supplied_queries,
            log=log,
        )
        clip_count = len(automatic_plan)
        _emit(log, f"Footage agent selected {clip_count} content-driven final shot(s)")
        if clip_count == 0:
            manifest = {
                "task_id": task_id,
                "status": "no_results",
                "created_at": _now(),
                "updated_at": _now(),
                "provider": "AI-directed public footage",
                "provider_id": provider,
                "license_policy": license_policy,
                "license_allowlist": ["Public Domain", "CC0", "CC BY", "CC BY-SA"],
                "orientation": orientation,
                "requested_clip_count": 0,
                "planned_clip_count": 0,
                "selection_mode": "ai",
                "planner": automatic_planner,
                "queries": [],
                "clips": [],
                "errors": [],
            }
            _write_manifest(task_dir, manifest)
            return manifest
    common = {
        "task_id": task_id,
        "task_dir": task_dir,
        "title": title,
        "script_path": script_path,
        "clip_count": clip_count,
        "orientation": orientation,
        "license_policy": license_policy,
        "provider_id": provider_id,
        "ai_endpoint": ai_endpoint,
        "ai_model": ai_model,
        "supplied_queries": supplied_queries,
        "planned_queries": automatic_plan,
        "log": log,
    }
    if provider == "wikimedia" or not config.WEB_FOOTAGE_ENABLED:
        manifest = await acquire_public_footage(**common)
        manifest["selection_mode"] = "ai" if automatic else "explicit"
        manifest["planned_clip_count"] = clip_count
        _write_manifest(task_dir, manifest)
        return manifest

    if automatic_plan is not None:
        query_plan, planner = automatic_plan, automatic_planner
    else:
        query_plan, planner = await plan_footage_queries(
            title=title,
            script=script,
            count=clip_count,
            provider_id=provider_id,
            ai_endpoint=ai_endpoint,
            ai_model=ai_model,
            supplied_queries=supplied_queries,
            log=log,
        )

    if provider == "hybrid":
        commons_quota = max(1, clip_count // 2)
        manifest = await acquire_public_footage(
            **{
                **common,
                "clip_count": commons_quota,
                "supplied_queries": [item["query"] for item in query_plan[:commons_quota]],
                "planned_queries": query_plan[:commons_quota],
            }
        )
    elif provider == "opencli_web":
        manifest = {
            "task_id": task_id,
            "status": "planning",
            "created_at": _now(),
            "updated_at": _now(),
            "provider": "YouTube",
            "provider_id": "opencli-web",
            "license_policy": "review_required",
            "requested_license_policy": license_policy,
            "license_allowlist": [],
            "orientation": orientation,
            "requested_clip_count": clip_count,
            "planner": planner,
            "queries": query_plan,
            "clips": [],
            "errors": [],
        }
        _write_manifest(task_dir, manifest)
    else:
        raise ValueError(f"Unsupported footage provider: {media_provider}")

    manifest["planner"] = planner
    manifest["queries"] = query_plan
    manifest["selection_mode"] = "ai" if automatic else "explicit"
    manifest["planned_clip_count"] = clip_count
    manifest["script_sha256"] = hashlib.sha256(script.encode()).hexdigest()
    manifest["binding_version"] = 2
    _write_manifest(task_dir, manifest)

    # Lazy import avoids a module cycle: web_footage reuses the query/search
    # helpers above, but footage remains the manifest-facing public API.
    from backend.pipeline.web_footage import supplement_web_footage

    active_clips = [
        clip for clip in manifest.get("clips") or [] if isinstance(clip, dict)
    ]
    if automatic:
        # In AI mode, two different queries may intentionally illustrate the
        # same longer beat.  Remove only queries Commons already fulfilled;
        # scene-level filtering would incorrectly discard the continuation.
        fulfilled_queries = {
            str(clip.get("query") or "").strip().casefold() for clip in active_clips
        }
        web_query_plan = [
            shot
            for shot in query_plan
            if str(shot.get("query") or "").strip().casefold() not in fulfilled_queries
        ]
    else:
        web_query_plan = _unoccupied_web_plan(query_plan, active_clips)

    return await supplement_web_footage(
        task_dir=task_dir,
        manifest=manifest,
        query_plan=web_query_plan,
        target_total=clip_count,
        orientation=orientation,
        script=script,
        log=log,
        provider_id=provider_id,
        ai_endpoint=ai_endpoint,
        ai_model=ai_model,
    )

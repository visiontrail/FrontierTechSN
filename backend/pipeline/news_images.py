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
import re
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import httpx

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
MANIFEST_VERSION = 7
SUPPORTED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/svg+xml",
}
OPEN_LICENSE_MARKERS = (
    "public domain",
    "cc0",
    "cc by",
    "cc-by",
    "cc by-sa",
    "cc-by-sa",
)
UNSAFE_LICENSE_MARKERS = ("noncommercial", "no derivatives", "-nc", "-nd")
PLAN_KINDS = {"logo", "event", "person", "place", "product", "object"}
PLAN_MODES = {"inline", "fullscreen"}
TAG_RE = re.compile(r"<[^>]+>")
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)?")
CAPITAL_PHRASE_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9'’-]*)(?:\s+(?:[A-Z][A-Za-z0-9'’-]*|of|the|and)){0,4}\b"
)
STOPWORDS = frozenset(
    "a an and are as at be been but by for from has have in into is it its of on or that the "
    "their this to was were with according reports report says said today morning thanks watching "
    "good tuesday monday wednesday thursday friday saturday sunday".split()
)
GENERIC_DIRECTION_TERMS = frozenset(
    "action biotech business engineering legislative morning news open science source technology".split()
)
BAD_PHOTO_MARKERS = ("logo", "icon", "map", "diagram", "chart", "flag")
GENERIC_LOGO_TITLE_TERMS = frozenset(
    "black brand corporate english en icon logo mark png symbol transparent white wordmark svg".split()
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
    temporary.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def _cached_manifest(task_dir: Path, fingerprint: str) -> dict | None:
    manifest = read_manifest(task_dir)
    if (
        not manifest
        or manifest.get("manifest_version") != MANIFEST_VERSION
        or manifest.get("storyboard_sha256") != fingerprint
        or manifest.get("status") != "ready"
    ):
        return None
    images = manifest.get("images") or []
    planned = int(manifest.get("planned_image_count") or 0)
    if not images or len(images) != planned:
        return None
    modes = manifest.get("placement_modes") or {}
    if planned >= 2 and (not modes.get("inline") or not modes.get("fullscreen")):
        return None
    for image in images:
        local = task_dir / str(image.get("local_path") or "")
        if not local.is_file() or not local.stat().st_size:
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
    subject = _ordered_terms(shot.get("expected_subject"))
    scene_words = _ordered_terms(
        f"{scene.get('text', '')} {' '.join(scene.get('keywords') or [])}"
    )
    raw = [shot.get("search_query") or ""]
    if shot.get("kind") == "logo":
        raw.extend(f"{term} logo" for term in subject[:4])
    else:
        if any(term.casefold() in {"mouse", "mice"} for term in subject + scene_words):
            raw.extend(["laboratory mouse", "mouse embryo", "CRISPR mouse"])
        if any(term.casefold() in {"rocket", "launch", "falcon"} for term in subject + scene_words):
            raw.extend(["rocket launch", "Falcon 9 launch"])
        raw.extend(
            [
                " ".join(subject[:3] + ["photo"]),
                " ".join(subject[:2] + ["event"]),
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
        subject = " ".join(str(raw.get("expected_subject") or "").split())[:100]
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
    if len(output) >= 2:
        modes = {item["display_mode"] for item in output}
        if "inline" not in modes:
            output[0]["display_mode"] = "inline"
        if "fullscreen" not in modes:
            output[-1]["display_mode"] = "fullscreen"
    return output


def _fallback_subject(scene: dict, hint: dict | None = None) -> str:
    hint = hint or {}
    text = str(scene.get("text") or "")
    hint_fields = [
        (12, str(hint.get("kicker") or "")),
        (10, str(hint.get("headline") or "")),
        *[
            (max(4, 8 - index), str(item))
            for index, item in enumerate(hint.get("items") or [])
        ],
        (6, str(hint.get("body") or "")),
    ]
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
    keywords = [str(value) for value in scene.get("keywords") or [] if _terms(value)]
    return " ".join(keywords[:3])[:100] or "technology news"


def _fallback_plan(
    scenes: list[dict], count: int, scene_hints: dict[str, dict] | None = None
) -> list[dict]:
    output = []
    for index, scene in enumerate(scenes[:count]):
        scene_id = str(scene.get("id") or "")
        hint = (scene_hints or {}).get(scene_id) or {}
        subject = _fallback_subject(scene, hint)
        text = f"{scene.get('text', '')} {hint.get('headline', '')} {' '.join(hint.get('items') or [])}".casefold()
        brand_like = any(
            marker in text
            for marker in (
                "announced",
                "company",
                "team",
                "open-sourced",
                "platform",
                "cpu",
                "server",
            )
        ) and len(subject.split()) <= 5
        kind = "logo" if brand_like else "event"
        if kind == "logo":
            owner = re.search(r"\b([A-Z][A-Za-z0-9-]+)['’]s\b", str(scene.get("text") or ""))
            if owner:
                subject = owner.group(1)
            query = f"{subject} logo"
        elif "mouse" in text or "mice" in text:
            subject = f"{subject} mouse"
            query = f"{subject} laboratory photo"
        else:
            query = f"{subject} event photo"
        output.append(
            {
                "scene_id": scene_id,
                "search_query": _sanitize_query(query),
                "news_query": _sanitize_query(subject, 14),
                "expected_subject": subject,
                "kind": kind,
                "display_mode": "fullscreen" if index % 3 == 2 else "inline",
                "purpose": "Deterministic visual derived from the narrated scene",
                "caption": subject,
            }
        )
    if len(output) >= 2 and not any(item["display_mode"] == "fullscreen" for item in output):
        output[-1]["display_mode"] = "fullscreen"
    return output


def _planner_prompt(
    scenes: list[dict], count: int, scene_hints: dict[str, dict] | None = None
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
        "scene ids supplied here:\n"
        + json.dumps(payload, ensure_ascii=False)
    )


async def plan_news_images(
    storyboard: dict,
    *,
    eligible_scene_ids: list[str],
    count: int,
    scene_hints: dict[str, dict] | None = None,
    log: LogCallback | None = None,
) -> tuple[list[dict], str, str]:
    scenes_by_id = {
        str(scene.get("id") or ""): scene for scene in storyboard.get("scenes") or []
    }
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
        plan = _normalise_plan(
            first_json(response), eligible_scene_ids=eligible_scene_ids, count=target
        )
        _emit(log, f"News images: OpenCLI planned {len(plan)} exact scene assignment(s)")
        return plan, "opencli:chatgpt-picture-editor", conversation_url
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
    title_terms = _terms(candidate["title"])
    candidate_terms = title_terms | _terms(candidate["description"])
    subject_terms = _terms(shot.get("expected_subject"))
    query_terms = _terms(shot.get("search_query"))
    # Search-result rank is not semantic proof. Commons can return a high-res
    # but unrelated file for a sparse query; require the file's own metadata to
    # name at least one expected subject/query anchor before it is downloadable.
    if subject_terms and not (subject_terms & candidate_terms):
        return None
    if not (query_terms & candidate_terms):
        return None
    if shot.get("kind") == "logo" and not (subject_terms & title_terms):
        return None
    return candidate


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
    for attempt in range(3):
        response = await client.get(WIKIMEDIA_API, params=params)
        if response.status_code != 429:
            break
        retry_after = float(response.headers.get("retry-after") or 0)
        await asyncio.sleep(max(retry_after, 1.25 * 2**attempt))
    assert response is not None
    response.raise_for_status()
    pages = (response.json().get("query") or {}).get("pages") or []
    candidates = [
        candidate
        for page in pages
        if (candidate := _candidate_from_page(page, shot)) is not None
    ]
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
    task_dir.mkdir(parents=True, exist_ok=True)
    image_dir = task_dir / "news_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = storyboard_fingerprint(storyboard)
    cached = _cached_manifest(task_dir, fingerprint)
    if cached:
        _emit(log, f"News images: reusing {len(cached['images'])} cached licensed image(s)")
        return cached

    excluded = excluded_scene_ids or set()
    eligible = [
        str(scene.get("id") or "")
        for scene in storyboard.get("scenes") or []
        if str(scene.get("id") or "") and str(scene.get("id") or "") not in excluded
    ]
    target = min(max(0, count), len(eligible))
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "status": "planning",
        "created_at": _now(),
        "updated_at": _now(),
        "storyboard_sha256": fingerprint,
        "requested_image_count": count,
        "eligible_scene_count": len(eligible),
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
    manifest["planner"] = planner
    manifest["planner_conversation_url"] = conversation_url
    manifest["queries"] = plan
    manifest["status"] = "searching"
    manifest["updated_at"] = _now()
    _write_manifest(task_dir, manifest)

    headers = {"User-Agent": config.FOOTAGE_USER_AGENT}
    used_sources: set[str] = set()
    async with httpx.AsyncClient(
        timeout=config.FOOTAGE_TIMEOUT,
        follow_redirects=True,
        headers=headers,
    ) as client:
        for index, shot in enumerate(plan, start=1):
            scene_id = shot["scene_id"]
            scene = next(
                scene
                for scene in storyboard.get("scenes") or []
                if str(scene.get("id") or "") == scene_id
            )
            _emit(
                log,
                f"News image {index}/{len(plan)}: researching {shot['expected_subject']} "
                f"for {scene_id} ({shot['display_mode']})",
            )
            references: list[dict] = []
            reference_provider = ""
            try:
                references, reference_provider = await research_references(shot)
            except Exception as exc:  # noqa: BLE001 - Commons remains usable
                manifest["errors"].append(
                    {"scene_id": scene_id, "stage": "opencli_search", "message": str(exc)}
                )

            try:
                candidates_by_source: dict[str, dict] = {}
                for query in _wikimedia_query_variants(shot, scene):
                    lookup = {**shot, "search_query": query}
                    for candidate in await search_wikimedia_images(client, shot=lookup):
                        candidate = {**candidate, "resolved_search_query": query}
                        candidates_by_source.setdefault(candidate["source_page_url"], candidate)
                    if len(candidates_by_source) >= 8:
                        break
                candidates = sorted(
                    candidates_by_source.values(),
                    key=lambda candidate: _candidate_rank(candidate, shot),
                )
            except Exception as exc:  # noqa: BLE001 - record exact failed scene
                manifest["errors"].append(
                    {"scene_id": scene_id, "stage": "wikimedia_search", "message": str(exc)}
                )
                continue

            candidates = [
                candidate
                for candidate in candidates
                if candidate["source_page_url"] not in used_sources
            ]
            if not candidates:
                manifest["errors"].append(
                    {
                        "scene_id": scene_id,
                        "stage": "selection",
                        "message": "No unique image passed subject, license, and resolution gates",
                    }
                )
                continue

            downloaded = None
            for candidate in candidates[:5]:
                destination = image_dir / f"image-{index:02d}{_extension_for(candidate)}"
                try:
                    byte_size, sha256 = await _download_candidate(
                        client, candidate=candidate, destination=destination
                    )
                except Exception as exc:  # noqa: BLE001 - try the next ranked source
                    if destination.exists():
                        destination.unlink()
                    manifest["errors"].append(
                        {
                            "scene_id": scene_id,
                            "stage": "download",
                            "source_page_url": candidate["source_page_url"],
                            "message": str(exc),
                        }
                    )
                    continue
                match_terms = sorted(
                    _terms(f"{shot['expected_subject']} {shot['search_query']}")
                    & _terms(f"{scene.get('text', '')} {' '.join(scene.get('keywords') or [])}")
                )
                if not match_terms:
                    destination.unlink(missing_ok=True)
                    manifest["errors"].append(
                        {
                            "scene_id": scene_id,
                            "stage": "grounding",
                            "message": "Expected image subject had no lexical anchor in the narrated scene",
                        }
                    )
                    break
                downloaded = {
                    "id": f"image-{index:02d}",
                    **shot,
                    **candidate,
                    "reference_provider": reference_provider,
                    "references": references,
                    "resolved_search_query": candidate.get("resolved_search_query")
                    or shot["search_query"],
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
                    "fit": "contain" if shot["kind"] == "logo" else "cover",
                    "status": "downloaded",
                }
                break

            if downloaded is None:
                continue
            used_sources.add(downloaded["source_page_url"])
            manifest["images"].append(downloaded)
            manifest["placement_modes"][downloaded["display_mode"]] += 1
            manifest["updated_at"] = _now()
            _write_manifest(task_dir, manifest)
            _emit(
                log,
                f"News image acquired for {scene_id}: {downloaded['title']} "
                f"({downloaded['license']}, {downloaded['display_mode']})",
            )

    acquired = len(manifest["images"])
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
    by_id = {str(plan.get("id") or ""): plan for plan in plans}
    scene_ids = {str(scene.get("id") or "") for scene in storyboard.get("scenes") or []}
    attached = 0
    modes = {"inline": 0, "fullscreen": 0}
    for image in (manifest or {}).get("images") or []:
        scene_id = str(image.get("scene_id") or "")
        plan = by_id.get(scene_id)
        if not plan or scene_id not in scene_ids:
            continue
        if plan.get("collage_broll") or plan.get("archetype") == "footage":
            continue
        local = task_dir / str(image.get("local_path") or "")
        if not local.is_file():
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
                "news_image_source_page_url": image.get("source_page_url") or "",
                "news_image_source_scene_id": scene_id,
                "news_image_match_terms": image.get("match_terms") or [],
                "news_image_license": image.get("license") or "",
                "news_image_reference_count": len(image.get("references") or []),
            }
        )
        if mode == "fullscreen":
            plan["news_image_original_archetype"] = plan.get("archetype") or "topic"
            plan["archetype"] = "news_image"
        attached += 1
        modes[mode] += 1
    return {"attached": attached, "placement_modes": modes}

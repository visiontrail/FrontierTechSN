"""AI-planned, license-gated public B-roll acquisition.

The media scout intentionally starts with Wikimedia Commons: it works without
an API key and exposes creator/license/source-page metadata per file. Search
results are accepted only when their license is explicitly on the open-license
allowlist. Every downloaded clip is recorded in ``footage/manifest.json`` so a
human can review provenance before the video is published.
"""

from __future__ import annotations

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
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'-]{2,}")
TAG_RE = re.compile(r"<[^>]+>")
SAFE_FILENAME_RE = re.compile(r"[^a-z0-9]+")


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


def _parse_plan(value: str, count: int) -> list[dict[str, str]]:
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
        if len(output) >= count:
            break
    if not output:
        raise ValueError("Footage planner returned no usable queries")
    return output


def _fallback_plan(title: str, script: str, count: int) -> list[dict[str, str]]:
    title_words = WORD_RE.findall(title)
    script_words = WORD_RE.findall(script)
    stop = {
        "about", "after", "again", "also", "because", "been", "before",
        "being", "between", "could", "from", "have", "into", "just",
        "more", "most", "other", "over", "really", "some", "than",
        "that", "their", "there", "these", "they", "this", "through",
        "very", "what", "when", "where", "which", "while", "with",
        "would", "your",
    }
    frequencies: dict[str, int] = {}
    for word in title_words + script_words[:400]:
        key = word.lower()
        if key not in stop:
            frequencies[key] = frequencies.get(key, 0) + 1
    ranked = sorted(frequencies, key=lambda word: (-frequencies[word], word))
    anchors = ranked[: max(2, count + 1)] or ["documentary", "people"]

    output = []
    for index in range(count):
        first = anchors[index % len(anchors)]
        second = anchors[(index + 1) % len(anchors)]
        query = _sanitize_query(f"{first} {second} documentary")
        output.append(
            {
                "query": query or "people documentary scene",
                "purpose": "Fallback visual derived from the narration",
            }
        )
    return output


async def plan_footage_queries(
    *,
    title: str,
    script: str,
    count: int,
    provider_id: int | None,
    ai_endpoint: str | None,
    ai_model: str | None,
    supplied_queries: list[str] | None = None,
    log: LogCallback | None = None,
) -> tuple[list[dict[str, str]], str]:
    supplied = [
        {"query": query, "purpose": "User-supplied search direction"}
        for query in (_sanitize_query(value) for value in supplied_queries or [])
        if query
    ]
    if supplied:
        return supplied[:count], "user"

    endpoint, model, api_key = await _resolve_provider(provider_id, ai_endpoint, ai_model)
    system_prompt = (config.PROMPTS_DIR / "footage_plan.txt").read_text(encoding="utf-8")
    user_content = (
        f"Requested queries: {count}\n"
        f"Title: {title}\n"
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
            max_tokens=1200,
        )
        plan = _parse_plan(result, count)
        _emit(log, f"Footage agent planned {len(plan)} visual search queries")
        return plan, f"ai:{model}"
    except Exception as exc:
        _emit(log, f"Footage planning fallback: {exc}")
        return _fallback_plan(title, script, count), "deterministic-fallback"


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


def manifest_path(task_dir: Path) -> Path:
    return task_dir / "footage" / "manifest.json"


def read_manifest(task_dir: Path) -> dict | None:
    path = manifest_path(task_dir)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_manifest(task_dir: Path, manifest: dict) -> None:
    path = manifest_path(task_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


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
    log: LogCallback | None = None,
) -> dict:
    """Plan, search, license-check, and download public footage for one task."""
    task_dir.mkdir(parents=True, exist_ok=True)
    footage_dir = task_dir / "footage"
    footage_dir.mkdir(parents=True, exist_ok=True)
    script = script_path.read_text(encoding="utf-8")

    manifest = {
        "task_id": task_id,
        "status": "planning",
        "created_at": _now(),
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
        "clips": [],
        "errors": [],
    }
    _write_manifest(task_dir, manifest)

    plan, planner = await plan_footage_queries(
        title=title,
        script=script,
        count=clip_count,
        provider_id=provider_id,
        ai_endpoint=ai_endpoint,
        ai_model=ai_model,
        supplied_queries=supplied_queries,
        log=log,
    )
    manifest["planner"] = planner
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

            clip_id = f"clip-{index:02d}"
            downloaded = None
            for candidate in available[:4]:
                extension = _extension_for(candidate)
                destination = footage_dir / f"{clip_id}{extension}"
                try:
                    byte_size, sha256 = await _download_candidate(
                        client,
                        candidate=candidate,
                        destination=destination,
                    )
                except Exception as exc:
                    if destination.exists():
                        destination.unlink()
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
                    **candidate,
                    "bytes": byte_size,
                    "sha256": sha256,
                    "local_path": destination.relative_to(task_dir).as_posix(),
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


async def acquire_footage(
    *,
    media_provider: str,
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
    log: LogCallback | None = None,
) -> dict:
    """Route one task through Wikimedia-only or the hybrid web scout.

    Hybrid mode intentionally keeps at least one Commons clip when possible,
    then fills the remaining slots from YouTube.
    """
    provider = (media_provider or "wikimedia").strip().lower()
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
        "log": log,
    }
    if provider == "wikimedia" or not config.WEB_FOOTAGE_ENABLED:
        return await acquire_public_footage(**common)

    script = script_path.read_text(encoding="utf-8")
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
    _write_manifest(task_dir, manifest)

    # Lazy import avoids a module cycle: web_footage reuses the query/search
    # helpers above, but footage remains the manifest-facing public API.
    from backend.pipeline.web_footage import supplement_web_footage

    return await supplement_web_footage(
        task_dir=task_dir,
        manifest=manifest,
        query_plan=query_plan,
        target_total=clip_count,
        orientation=orientation,
        script=script,
        log=log,
    )

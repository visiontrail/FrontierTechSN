"""Capture English news articles for deterministic HyperFrames overlays.

The daily-news dossier is the authority for language and source identity.  A
page is captured only when the selected dossier record is English *and* the
rendered DOM contains an English headline plus article prose.  HyperFrames
receives only the resulting local PNG, never a live iframe or network URL.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import logging
import os
import re
import shutil
import tempfile
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import websockets
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)
LogCallback = Callable[[str], None]

MANIFEST_VERSION = 2
MAX_PAGE_OVERLAYS = 2
VIEWPORT_WIDTH = 1440
VIEWPORT_HEIGHT = 900
PAGE_LOAD_TIMEOUT_SECONDS = 24.0
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'-]{2,}")
STOP_WORDS = {
    "about",
    "after",
    "again",
    "also",
    "been",
    "being",
    "from",
    "have",
    "into",
    "more",
    "most",
    "that",
    "their",
    "there",
    "these",
    "they",
    "this",
    "with",
    "would",
}


def _emit(log: LogCallback | None, message: str) -> None:
    if log:
        log(message)
    else:
        logger.info(message)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_manifest(task_dir: Path, manifest: dict) -> None:
    path = task_dir / "news_webpages" / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def read_manifest(task_dir: Path) -> dict | None:
    path = task_dir / "news_webpages" / "manifest.json"
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _tokens(value: object) -> set[str]:
    return {
        word.casefold()
        for word in WORD_RE.findall(str(value or ""))
        if word.casefold() not in STOP_WORDS
    }


def _story_text(article: dict) -> str:
    return " ".join(
        str(article.get(key) or "")
        for key in ("title", "summary", "evidence_text", "source_name")
    )


def _valid_english_source(article: dict) -> bool:
    if str(article.get("language") or "").casefold() != "en":
        return False
    parsed = urlsplit(str(article.get("url") or ""))
    return parsed.scheme in {"http", "https"} and bool(parsed.hostname)


def _visual_scene_ids(plans: list[dict]) -> set[str]:
    return {
        str(plan.get("id") or "")
        for plan in plans
        if not plan.get("collage_broll")
        and not plan.get("news_image_collage")
        and (plan.get("footage_src") or plan.get("news_image_src"))
    }


def select_overlay_assignments(
    dossier: dict,
    storyboard: dict,
    plans: list[dict],
    *,
    limit: int = MAX_PAGE_OVERLAYS,
) -> list[dict]:
    """Map selected English stories to exact visual narration scenes."""
    visual_ids = _visual_scene_ids(plans)
    scenes = [
        scene
        for scene in storyboard.get("scenes") or []
        if str(scene.get("id") or "") in visual_ids
    ]
    scored: list[tuple[int, int, int, dict, dict]] = []
    for article_index, article in enumerate(dossier.get("selected") or []):
        if not isinstance(article, dict) or not _valid_english_source(article):
            continue
        article_terms = _tokens(_story_text(article))
        if not article_terms:
            continue
        for scene_index, scene in enumerate(scenes):
            scene_terms = _tokens(scene.get("text"))
            overlap = article_terms & scene_terms
            if len(overlap) < 2:
                continue
            # Prefer the exact story scene, then preserve edition order.
            score = len(overlap) * 10 + min(9, len(overlap) * 2)
            scored.append((score, -article_index, -scene_index, article, scene))

    assignments: list[dict] = []
    used_articles: set[str] = set()
    used_scenes: set[str] = set()
    for _score, _article_order, _scene_order, article, scene in sorted(
        scored, reverse=True, key=lambda row: row[:3]
    ):
        url = str(article.get("url") or "")
        scene_id = str(scene.get("id") or "")
        if url in used_articles or scene_id in used_scenes:
            continue
        assignments.append(
            {
                "scene_id": scene_id,
                "source_url": url,
                "source_name": str(article.get("source_name") or "English news source"),
                "expected_headline": str(article.get("title") or ""),
                "dossier_language": "en",
            }
        )
        used_articles.add(url)
        used_scenes.add(scene_id)
        if len(assignments) >= max(0, limit):
            break
    return assignments


def _fingerprint(assignments: list[dict], storyboard: dict) -> str:
    payload = {
        "assignments": assignments,
        "scenes": [
            {
                "id": str(scene.get("id") or ""),
                "text": str(scene.get("text") or ""),
            }
            for scene in storyboard.get("scenes") or []
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _asset_is_intact(task_dir: Path, item: dict) -> bool:
    relative = str(item.get("local_path") or "")
    if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        return False
    path = task_dir / relative
    if not path.is_file() or path.stat().st_size != int(item.get("bytes") or 0):
        return False
    if _sha256(path) != str(item.get("sha256") or ""):
        return False
    try:
        with Image.open(path) as image:
            image.load()
            return image.size == (VIEWPORT_WIDTH, VIEWPORT_HEIGHT)
    except (OSError, UnidentifiedImageError):
        return False


def _cached_manifest(task_dir: Path, fingerprint: str) -> dict | None:
    manifest = read_manifest(task_dir)
    if (
        not manifest
        or manifest.get("manifest_version") != MANIFEST_VERSION
        or manifest.get("fingerprint") != fingerprint
        or manifest.get("status") not in {"ready", "partial"}
    ):
        return None
    pages = manifest.get("pages") or []
    if not isinstance(pages, list) or not pages:
        return None
    if any(not isinstance(item, dict) or not _asset_is_intact(task_dir, item) for item in pages):
        return None
    return manifest


def _chrome_binary() -> Path:
    configured = os.getenv("NEWS_WEBPAGE_CHROME_BIN", "").strip()
    candidates = [
        Path(configured) if configured else None,
        Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
        Path("/Applications/Chromium.app/Contents/MacOS/Chromium"),
    ]
    for candidate in candidates:
        if candidate and candidate.is_file():
            return candidate
    raise RuntimeError("No Chrome/Chromium binary is available for news-page capture")


async def _cdp_command(socket, counter: list[int], method: str, params: dict | None = None) -> dict:
    counter[0] += 1
    command_id = counter[0]
    await socket.send(
        json.dumps({"id": command_id, "method": method, "params": params or {}})
    )
    while True:
        payload = json.loads(await socket.recv())
        if payload.get("id") != command_id:
            continue
        if payload.get("error"):
            raise RuntimeError(
                f"Chrome DevTools {method} failed: {payload['error'].get('message') or payload['error']}"
            )
        return payload.get("result") or {}


async def _runtime_value(socket, counter: list[int], expression: str) -> object:
    result = await _cdp_command(
        socket,
        counter,
        "Runtime.evaluate",
        {
            "expression": expression,
            "returnByValue": True,
            "awaitPromise": True,
        },
    )
    remote = result.get("result") or {}
    if remote.get("subtype") == "error":
        raise RuntimeError(str(remote.get("description") or "page evaluation failed"))
    return remote.get("value")


async def _start_chrome() -> tuple[asyncio.subprocess.Process, str, Path]:
    profile = Path(tempfile.mkdtemp(prefix="frontier-news-pages-"))
    process = await asyncio.create_subprocess_exec(
        str(_chrome_binary()),
        "--headless=new",
        "--disable-background-networking",
        "--disable-component-update",
        "--disable-default-apps",
        "--disable-features=Translate,MediaRouter",
        "--hide-scrollbars",
        "--mute-audio",
        "--no-first-run",
        "--no-default-browser-check",
        "--remote-allow-origins=*",
        "--remote-debugging-port=0",
        f"--user-data-dir={profile}",
        "about:blank",
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        assert process.stderr is not None
        while True:
            line = await asyncio.wait_for(process.stderr.readline(), timeout=20)
            if not line:
                raise RuntimeError("Chrome exited before exposing DevTools")
            match = re.search(rb"DevTools listening on (ws://\S+)", line)
            if match:
                return process, match.group(1).decode("utf-8"), profile
    except Exception:
        process.terminate()
        await process.wait()
        shutil.rmtree(profile, ignore_errors=True)
        raise


async def _stop_chrome(process: asyncio.subprocess.Process, profile: Path) -> None:
    if process.returncode is None:
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=8)
        except TimeoutError:
            process.kill()
            await process.wait()
    shutil.rmtree(profile, ignore_errors=True)


_PAGE_INFO_SCRIPT = r"""
(() => {
  const expectedHeadline = __EXPECTED_HEADLINE__;
  const expectedTerms = new Set((expectedHeadline.match(/[A-Za-z][A-Za-z0-9'-]{2,}/g) || [])
    .map((word) => word.toLowerCase()));
  const visible = (node) => {
    if (!node) return false;
    const style = getComputedStyle(node);
    const rect = node.getBoundingClientRect();
    return style.display !== 'none' && style.visibility !== 'hidden' && rect.width > 40 && rect.height > 20;
  };
  const headlineCandidates = [...document.querySelectorAll(
    'h1, h1 span, [itemprop="headline"], article h2, main h2, .headline, .story-title, .custom-post-headline, .widget__headline-text, [class*="sticky-headline"], h2, h3, strong'
  )].filter((node) => {
    const text = (node.innerText || '').replace(/\s+/g, ' ').trim();
    return visible(node) && text.length >= 12 && text.length <= 420;
  });
  const headline = headlineCandidates.sort((left, right) => {
    const overlap = (node) => {
      const terms = new Set(((node.innerText || '').match(/[A-Za-z][A-Za-z0-9'-]{2,}/g) || [])
        .map((word) => word.toLowerCase()));
      return [...terms].filter((word) => expectedTerms.has(word)).length;
    };
    const scoreDifference = overlap(right) - overlap(left);
    if (scoreDifference) return scoreDifference;
    const leftSize = parseFloat(getComputedStyle(left).fontSize) || 0;
    const rightSize = parseFloat(getComputedStyle(right).fontSize) || 0;
    return rightSize - leftSize;
  })[0] || null;
  const root = (headline && headline.closest('article')) || document.querySelector('article') ||
    document.querySelector('main') || document.body || document.documentElement;
  const paragraphs = root ? [...root.querySelectorAll('p')]
    .filter(visible)
    .map((node) => (node.innerText || '').replace(/\s+/g, ' ').trim())
    .filter((text) => text.length >= 45) : [];
  const sample = `${headline ? headline.innerText : ''} ${paragraphs.slice(0, 6).join(' ')}`;
  const headlineTerms = new Set(((headline ? headline.innerText : '').match(/[A-Za-z][A-Za-z0-9'-]{2,}/g) || [])
    .map((word) => word.toLowerCase()));
  const metadataHeadlines = [
    document.querySelector('meta[property="og:title"]')?.content || '',
    document.querySelector('meta[name="twitter:title"]')?.content || '',
    document.title || '',
    ...[...document.querySelectorAll('script[type="application/ld+json"]')].flatMap((node) => {
      try {
        const value = JSON.parse(node.textContent || '{}');
        const items = Array.isArray(value) ? value : [value];
        return items.flatMap((item) => [item?.headline || '', item?.name || '']);
      } catch (_) {
        return [];
      }
    }),
  ];
  const metadataMatchWords = Math.max(0, ...metadataHeadlines.map((text) => {
    const terms = new Set(((text || '').match(/[A-Za-z][A-Za-z0-9'-]{2,}/g) || [])
      .map((word) => word.toLowerCase()));
    return [...terms].filter((word) => expectedTerms.has(word)).length;
  }));
  return {
    ready_state: document.readyState,
    document_language: (document.documentElement.lang || '').trim().toLowerCase(),
    page_title: document.title || '',
    headline: headline ? (headline.innerText || '').replace(/\s+/g, ' ').trim() : '',
    headline_match_words: [...headlineTerms].filter((word) => expectedTerms.has(word)).length,
    metadata_match_words: metadataMatchWords,
    paragraph_characters: paragraphs.join(' ').length,
    english_word_count: (sample.match(/[A-Za-z][A-Za-z0-9'-]{2,}/g) || []).length,
    cjk_character_count: (sample.match(/[\u3400-\u9fff]/g) || []).length,
  };
})()
"""


_PREPARE_CAPTURE_SCRIPT = r"""
(() => {
  const expectedHeadline = __EXPECTED_HEADLINE__;
  const expectedTerms = new Set((expectedHeadline.match(/[A-Za-z][A-Za-z0-9'-]{2,}/g) || [])
    .map((word) => word.toLowerCase()));
  const visible = (node) => {
    const style = getComputedStyle(node);
    const rect = node.getBoundingClientRect();
    return style.display !== 'none' && style.visibility !== 'hidden' && rect.width > 40 && rect.height > 20;
  };
  const headlineCandidates = [...document.querySelectorAll(
    'h1, h1 span, [itemprop="headline"], article h2, main h2, .headline, .story-title, .custom-post-headline, .widget__headline-text, [class*="sticky-headline"], h2, h3, strong'
  )].filter((node) => {
    const text = (node.innerText || '').replace(/\s+/g, ' ').trim();
    return visible(node) && text.length >= 12 && text.length <= 420;
  });
  const headline = headlineCandidates.sort((left, right) => {
    const overlap = (node) => {
      const terms = new Set(((node.innerText || '').match(/[A-Za-z][A-Za-z0-9'-]{2,}/g) || [])
        .map((word) => word.toLowerCase()));
      return [...terms].filter((word) => expectedTerms.has(word)).length;
    };
    const scoreDifference = overlap(right) - overlap(left);
    if (scoreDifference) return scoreDifference;
    const leftSize = parseFloat(getComputedStyle(left).fontSize) || 0;
    const rightSize = parseFloat(getComputedStyle(right).fontSize) || 0;
    return rightSize - leftSize;
  })[0] || null;
  const nuisance = /cookie|consent|subscribe|sign in|newsletter|notification|privacy choices/i;
  [...document.querySelectorAll('body *')].forEach((node) => {
    const style = getComputedStyle(node);
    const text = (node.innerText || '').trim();
    const rect = node.getBoundingClientRect();
    if ((style.position === 'fixed' || style.position === 'sticky') &&
        Number(style.zIndex || 0) >= 10 && rect.width > innerWidth * 0.45 &&
        rect.height > 70 && nuisance.test(text.slice(0, 600))) {
      node.style.setProperty('display', 'none', 'important');
    }
  });
  const articleRoot = (headline && headline.closest('article')) ||
    document.querySelector('article') || document.querySelector('main') || document.body;
  const storyParagraph = articleRoot ? [...articleRoot.querySelectorAll('p')]
    .find((node) => visible(node) && (node.innerText || '').trim().length >= 80) : null;
  if (headline && storyParagraph && storyParagraph.getBoundingClientRect().top > innerHeight * 0.78) {
    const headlineBottom = headline.getBoundingClientRect().bottom;
    const paragraphTop = storyParagraph.getBoundingClientRect().top;
    [...articleRoot.querySelectorAll('figure, picture, video, iframe, [class*="hero"], [class*="lead-image"], [class*="featured-image"], [class*="article-media"]')]
      .forEach((node) => {
        const rect = node.getBoundingClientRect();
        if (rect.top >= headlineBottom - 8 && rect.bottom <= paragraphTop + 8 &&
            rect.width > innerWidth * 0.4 && rect.height > 160 &&
            !node.contains(storyParagraph)) {
          node.style.setProperty('display', 'none', 'important');
        }
      });
  }
  if (headline) {
    headline.scrollIntoView({block: 'start', inline: 'nearest'});
    window.scrollBy(0, -72);
  }
  if (!headline) return {headline_found: false, focus_rect: null};

  // Capture the exact story block, not an unreadable overview of the entire
  // publication.  Prefer the smallest ancestor that contains the matched
  // headline plus visible prose; fall back to the union of the headline and
  // the first article paragraphs.  Coordinates stay viewport-relative so the
  // PNG can be cropped from the same screenshot without trusting page markup
  // to provide a stable class name.
  const headlineText = (headline.innerText || '').replace(/\s+/g, ' ').trim();
  let focusRoot = headline.parentElement;
  while (focusRoot && focusRoot !== document.body) {
    const rect = focusRoot.getBoundingClientRect();
    const text = (focusRoot.innerText || '').replace(/\s+/g, ' ').trim();
    if (visible(focusRoot) && rect.width >= 320 && rect.height >= 110 &&
        rect.width <= innerWidth * 0.94 && rect.height <= innerHeight * 0.88 &&
        text.length >= headlineText.length + 55) {
      break;
    }
    focusRoot = focusRoot.parentElement;
  }
  if (focusRoot === document.body) focusRoot = null;

  const focusNodes = [headline];
  if (focusRoot) {
    focusNodes.push(focusRoot);
  } else if (articleRoot) {
    focusNodes.push(...[...articleRoot.querySelectorAll('p')]
      .filter((node) => visible(node) && (node.innerText || '').trim().length >= 45)
      .slice(0, 3));
  }
  const rects = focusNodes.map((node) => node.getBoundingClientRect())
    .filter((rect) => rect.width > 0 && rect.height > 0);
  const padding = 28;
  const left = Math.max(0, Math.min(...rects.map((rect) => rect.left)) - padding);
  const top = Math.max(0, Math.min(...rects.map((rect) => rect.top)) - padding);
  const right = Math.min(innerWidth, Math.max(...rects.map((rect) => rect.right)) + padding);
  const bottom = Math.min(innerHeight, Math.max(...rects.map((rect) => rect.bottom)) + padding);
  const width = Math.max(0, right - left);
  const height = Math.max(0, bottom - top);
  return {
    headline_found: true,
    focus_rect: width >= 280 && height >= 100
      ? {x: left, y: top, width, height}
      : null,
  };
})()
"""


def _focused_screenshot(raw: bytes, focus_rect: object, destination: Path) -> dict:
    """Crop the exact story block and place it on a render-stable canvas."""
    with Image.open(io.BytesIO(raw)) as source:
        source.load()
        source = source.convert("RGB")
        if source.size != (VIEWPORT_WIDTH, VIEWPORT_HEIGHT):
            raise RuntimeError(f"Unexpected news screenshot size: {source.size}")

        rect = focus_rect if isinstance(focus_rect, dict) else {}
        try:
            left = max(0, min(VIEWPORT_WIDTH - 1, int(float(rect.get("x", 0)))))
            top = max(0, min(VIEWPORT_HEIGHT - 1, int(float(rect.get("y", 0)))))
            right = max(
                left + 1,
                min(
                    VIEWPORT_WIDTH,
                    int(float(rect.get("x", 0)) + float(rect.get("width", 0))),
                ),
            )
            bottom = max(
                top + 1,
                min(
                    VIEWPORT_HEIGHT,
                    int(float(rect.get("y", 0)) + float(rect.get("height", 0))),
                ),
            )
        except (TypeError, ValueError):
            left, top, right, bottom = 0, 0, VIEWPORT_WIDTH, VIEWPORT_HEIGHT

        if right - left < 280 or bottom - top < 100:
            left, top, right, bottom = 0, 0, VIEWPORT_WIDTH, VIEWPORT_HEIGHT
        focused = source.crop((left, top, right, bottom))
        scale = min(1320 / focused.width, 780 / focused.height)
        rendered_size = (
            max(1, round(focused.width * scale)),
            max(1, round(focused.height * scale)),
        )
        focused = focused.resize(rendered_size, Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (VIEWPORT_WIDTH, VIEWPORT_HEIGHT), "white")
        canvas.paste(
            focused,
            (
                (VIEWPORT_WIDTH - rendered_size[0]) // 2,
                (VIEWPORT_HEIGHT - rendered_size[1]) // 2,
            ),
        )
        canvas.save(destination, format="PNG", optimize=True)
    return {
        "focus_x": left,
        "focus_y": top,
        "focus_width": right - left,
        "focus_height": bottom - top,
        "focused_render_width": rendered_size[0],
        "focused_render_height": rendered_size[1],
    }


def _page_is_english(info: dict) -> bool:
    language = str(info.get("document_language") or "")
    words = int(info.get("english_word_count") or 0)
    cjk = int(info.get("cjk_character_count") or 0)
    return bool(
        info.get("headline")
        and max(
            int(info.get("headline_match_words") or 0),
            int(info.get("metadata_match_words") or 0),
        )
        >= 2
        and int(info.get("paragraph_characters") or 0) >= 80
        and words >= 30
        and (language.startswith("en") or (words >= 50 and cjk <= max(3, words // 12)))
    )


async def _capture_page(
    browser_ws_url: str,
    assignment: dict,
    destination: Path,
) -> dict:
    parsed = urlsplit(browser_ws_url)
    origin = f"http://{parsed.hostname}:{parsed.port}"
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.put(
            f"{origin}/json/new",
            params={"url": str(assignment["source_url"])},
        )
        response.raise_for_status()
        target = response.json()
    socket_url = str(target.get("webSocketDebuggerUrl") or "")
    if not socket_url:
        raise RuntimeError("Chrome did not expose the article page target")

    async with websockets.connect(socket_url, max_size=16 * 1024 * 1024) as socket:
        counter = [0]
        await _cdp_command(socket, counter, "Page.enable")
        await _cdp_command(socket, counter, "Runtime.enable")
        await _cdp_command(
            socket,
            counter,
            "Emulation.setDeviceMetricsOverride",
            {
                "width": VIEWPORT_WIDTH,
                "height": VIEWPORT_HEIGHT,
                "deviceScaleFactor": 1,
                "mobile": False,
            },
        )
        await _cdp_command(
            socket,
            counter,
            "Page.navigate",
            {"url": str(assignment["source_url"])},
        )

        loop = asyncio.get_running_loop()
        deadline = loop.time() + PAGE_LOAD_TIMEOUT_SECONDS
        info: dict = {}
        expected_headline_json = json.dumps(
            str(assignment.get("expected_headline") or ""), ensure_ascii=False
        )
        page_info_script = _PAGE_INFO_SCRIPT.replace(
            "__EXPECTED_HEADLINE__", expected_headline_json
        )
        prepare_capture_script = _PREPARE_CAPTURE_SCRIPT.replace(
            "__EXPECTED_HEADLINE__", expected_headline_json
        )
        while loop.time() < deadline:
            try:
                value = await _runtime_value(socket, counter, page_info_script)
            except RuntimeError:
                await asyncio.sleep(0.35)
                continue
            info = value if isinstance(value, dict) else {}
            if info.get("ready_state") in {"interactive", "complete"} and _page_is_english(info):
                break
            await asyncio.sleep(0.6)
        if not _page_is_english(info):
            raise RuntimeError(
                "Rendered source did not expose an English headline and article body "
                f"(lang={info.get('document_language') or 'unknown'}, "
                f"words={info.get('english_word_count') or 0}, "
                f"headline_match={info.get('headline_match_words') or 0}, "
                f"metadata_match={info.get('metadata_match_words') or 0}, "
                f"paragraph_chars={info.get('paragraph_characters') or 0})"
            )

        prepared = await _runtime_value(socket, counter, prepare_capture_script)
        await asyncio.sleep(0.45)
        refreshed = await _runtime_value(socket, counter, page_info_script)
        if isinstance(refreshed, dict) and _page_is_english(refreshed):
            info = refreshed
        screenshot = await _cdp_command(
            socket,
            counter,
            "Page.captureScreenshot",
            {
                "format": "png",
                "fromSurface": True,
                "captureBeyondViewport": False,
            },
        )
        raw = base64.b64decode(str(screenshot.get("data") or ""), validate=True)
        focus = prepared.get("focus_rect") if isinstance(prepared, dict) else None
        info.update(_focused_screenshot(raw, focus, destination))
        await _cdp_command(socket, counter, "Page.close")

    with Image.open(destination) as image:
        image.load()
        if image.size != (VIEWPORT_WIDTH, VIEWPORT_HEIGHT):
            raise RuntimeError(f"Unexpected news screenshot size: {image.size}")
    return info


async def acquire_news_webpages(
    storyboard: dict,
    plans: list[dict],
    task_dir: Path,
    *,
    log: LogCallback | None = None,
    limit: int = MAX_PAGE_OVERLAYS,
) -> dict:
    """Capture exact selected English story pages for compatible visual scenes."""
    dossier_path = task_dir / "research" / "dossier.json"
    try:
        dossier = json.loads(dossier_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        dossier = {}
    assignments = select_overlay_assignments(dossier, storyboard, plans, limit=limit)
    fingerprint = _fingerprint(assignments, storyboard)
    cached = _cached_manifest(task_dir, fingerprint)
    if cached:
        _emit(log, f"News webpages: reusing {len(cached['pages'])} verified capture(s)")
        return cached

    root = task_dir / "news_webpages"
    root.mkdir(parents=True, exist_ok=True)
    previous = read_manifest(task_dir) or {}
    for item in previous.get("pages") or []:
        if not isinstance(item, dict):
            continue
        relative = Path(str(item.get("local_path") or ""))
        if not relative.is_absolute() and ".." not in relative.parts:
            (task_dir / relative).unlink(missing_ok=True)

    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "fingerprint": fingerprint,
        "status": "capturing" if assignments else "not_needed",
        "created_at": _now(),
        "updated_at": _now(),
        "requested_count": len(assignments),
        "english_only": True,
        "assignments": assignments,
        "pages": [],
        "errors": [],
    }
    _write_manifest(task_dir, manifest)
    if not assignments:
        _emit(log, "News webpages: no English selected story matched an image/footage scene")
        return manifest

    process: asyncio.subprocess.Process | None = None
    profile: Path | None = None
    try:
        process, browser_ws_url, profile = await _start_chrome()
        for index, assignment in enumerate(assignments, start=1):
            destination = root / f"page-{index:02d}.png"
            _emit(
                log,
                f"News webpage {index}/{len(assignments)}: capturing English source "
                f"{urlsplit(assignment['source_url']).hostname}",
            )
            try:
                info = await _capture_page(browser_ws_url, assignment, destination)
            except Exception as exc:  # noqa: BLE001 - preserve other eligible captures
                destination.unlink(missing_ok=True)
                manifest["errors"].append(
                    {
                        "scene_id": assignment["scene_id"],
                        "source_url": assignment["source_url"],
                        "message": str(exc),
                    }
                )
                _emit(log, f"News webpage capture rejected: {exc}")
                continue
            item = {
                **assignment,
                "document_language": str(info.get("document_language") or ""),
                "page_title": str(info.get("page_title") or ""),
                "captured_headline": str(info.get("headline") or ""),
                "headline_match_words": int(info.get("headline_match_words") or 0),
                "metadata_match_words": int(info.get("metadata_match_words") or 0),
                "paragraph_characters": int(info.get("paragraph_characters") or 0),
                "english_word_count": int(info.get("english_word_count") or 0),
                "focus_x": int(info.get("focus_x") or 0),
                "focus_y": int(info.get("focus_y") or 0),
                "focus_width": int(info.get("focus_width") or VIEWPORT_WIDTH),
                "focus_height": int(info.get("focus_height") or VIEWPORT_HEIGHT),
                "focused_render_width": int(
                    info.get("focused_render_width") or VIEWPORT_WIDTH
                ),
                "focused_render_height": int(
                    info.get("focused_render_height") or VIEWPORT_HEIGHT
                ),
                "width": VIEWPORT_WIDTH,
                "height": VIEWPORT_HEIGHT,
                "bytes": destination.stat().st_size,
                "sha256": _sha256(destination),
                "local_path": destination.relative_to(task_dir).as_posix(),
                "status": "captured",
            }
            manifest["pages"].append(item)
            manifest["updated_at"] = _now()
            _write_manifest(task_dir, manifest)
    finally:
        if process is not None and profile is not None:
            await _stop_chrome(process, profile)

    captured = len(manifest["pages"])
    manifest["status"] = (
        "ready"
        if captured == len(assignments)
        else ("partial" if captured else "no_results")
    )
    manifest["updated_at"] = _now()
    _write_manifest(task_dir, manifest)
    _emit(
        log,
        f"News webpages: {captured}/{len(assignments)} English article capture(s) ready",
    )
    return manifest


def attach_news_webpages(plans: list[dict], manifest: dict | None, task_dir: Path) -> int:
    """Attach verified captures without changing the underlying media plate."""
    if not manifest or manifest.get("status") not in {"ready", "partial"}:
        return 0
    by_id = {str(plan.get("id") or ""): plan for plan in plans}
    attached = 0
    used_scenes: set[str] = set()
    used_urls: set[str] = set()
    for item in manifest.get("pages") or []:
        if not isinstance(item, dict) or not _asset_is_intact(task_dir, item):
            continue
        scene_id = str(item.get("scene_id") or "")
        source_url = str(item.get("source_url") or "")
        plan = by_id.get(scene_id)
        if (
            not plan
            or scene_id in used_scenes
            or source_url in used_urls
            or plan.get("collage_broll")
            or plan.get("news_image_collage")
            or not (plan.get("footage_src") or plan.get("news_image_src"))
            or str(item.get("dossier_language") or "") != "en"
        ):
            continue
        plan.update(
            {
                "news_webpage": True,
                "news_webpage_src": f"../{item['local_path']}",
                "news_webpage_url": source_url,
                "news_webpage_source": item.get("source_name") or "English news source",
                "news_webpage_headline": item.get("captured_headline") or "",
                "news_webpage_sha256": item.get("sha256") or "",
                "news_webpage_language": "en",
            }
        )
        used_scenes.add(scene_id)
        used_urls.add(source_url)
        attached += 1
    return attached

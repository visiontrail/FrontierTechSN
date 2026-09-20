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

MANIFEST_VERSION = 6
MAX_PAGE_OVERLAYS = 2
VIEWPORT_WIDTH = 1440
VIEWPORT_HEIGHT = 900
# DOM geometry stays in CSS pixels; screenshots and crop rectangles use 2x
# physical pixels so article text remains sharp in a UHD composition.
CAPTURE_SCALE = 2
CAPTURE_WIDTH = VIEWPORT_WIDTH * CAPTURE_SCALE
CAPTURE_HEIGHT = VIEWPORT_HEIGHT * CAPTURE_SCALE
PAGE_LOAD_TIMEOUT_SECONDS = 24.0
PAGE_CAPTURE_TIMEOUT_SECONDS = 75.0
CDP_COMMAND_TIMEOUT_SECONDS = 12.0
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'-]{2,}")
STOP_WORDS = {
    "about",
    "according",
    "after",
    "again",
    "all",
    "also",
    "and",
    "been",
    "being",
    "could",
    "from",
    "have",
    "into",
    "more",
    "most",
    "report",
    "reported",
    "reporting",
    "reports",
    "said",
    "says",
    "source",
    "sources",
    "that",
    "the",
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
    # Evidence excerpts can contain archive boilerplate, publication dates,
    # and unrelated navigation copy.  Assignment identity must come from the
    # selected headline and feed summary; evidence is still retained for the
    # script/fact-check pipeline, but must not map an article to a visual scene.
    return " ".join(
        str(article.get(key) or "")
        for key in ("title", "summary", "source_name")
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


def _fingerprint(
    assignments: list[dict], storyboard: dict, requested_count: int
) -> str:
    payload = {
        "assignments": assignments,
        "requested_count": requested_count,
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
            return image.size == (CAPTURE_WIDTH, CAPTURE_HEIGHT)
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


class CaptureCommandTimeout(RuntimeError):
    """A browser command stalled before page-level polling could proceed."""


async def _cdp_command(socket, counter: list[int], method: str, params: dict | None = None) -> dict:
    counter[0] += 1
    command_id = counter[0]
    try:
        async with asyncio.timeout(CDP_COMMAND_TIMEOUT_SECONDS):
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
    except TimeoutError as exc:
        raise CaptureCommandTimeout(
            f"Chrome DevTools {method} exceeded {CDP_COMMAND_TIMEOUT_SECONDS:g}s deadline"
        ) from exc


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
  // Techmeme's prose is a text node following the headline in the same .ii
  // block. Scope evidence to that matched story, excluding its headline and
  // all neighbouring stories, related links and archive/sidebar boilerplate.
  const techmemeStory = /^(www\.)?techmeme\.com$/i.test(location.hostname)
    ? headline?.closest('.ii') : null;
  const root = techmemeStory || (headline && headline.closest('article')) || document.querySelector('article') ||
    document.querySelector('main') || document.body || document.documentElement;
  const paragraphs = root ? (techmemeStory ? [techmemeStory] : [...root.querySelectorAll('p')])
    .filter(visible)
    .map((node) => (node.innerText || '').replace(/\s+/g, ' ').trim())
    .map((text) => techmemeStory
      ? text.replace((headline.innerText || '').replace(/\s+/g, ' ').trim(), '').replace(/^[\s—–-]+/, '')
      : text)
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
  const bodyPreview = (document.body?.innerText || '').replace(/\s+/g, ' ').trim().slice(0, 1200);
  const blockerText = `${document.title || ''} ${bodyPreview}`;
  const blockingReason = /human verification|confirm you are human|security check before continuing|performing security verification|verif(?:y|ies) you are not a bot|captcha|just a moment/i.test(blockerText)
    ? 'human verification challenge'
    : /access denied|request blocked|temporarily unavailable/i.test(blockerText)
      ? 'access denied page'
      : '';
  return {
    ready_state: document.readyState,
    page_url: location.href,
    document_language: (document.documentElement.lang || '').trim().toLowerCase(),
    page_title: document.title || '',
    publisher_name: document.querySelector('meta[property="og:site_name"]')?.content || '',
    content_kind: techmemeStory ? 'aggregator_excerpt' : 'article',
    headline: headline ? (headline.innerText || '').replace(/\s+/g, ' ').trim() : '',
    headline_href: headline
      ? (headline.closest('a[href]')?.href || headline.querySelector('a[href]')?.href || '')
      : '',
    headline_match_words: [...headlineTerms].filter((word) => expectedTerms.has(word)).length,
    metadata_match_words: metadataMatchWords,
    paragraph_characters: paragraphs.join(' ').length,
    english_word_count: (sample.match(/[A-Za-z][A-Za-z0-9'-]{2,}/g) || []).length,
    cjk_character_count: (sample.match(/[\u3400-\u9fff]/g) || []).length,
    blocking_reason: blockingReason,
  };
})()
"""


_PREPARE_CAPTURE_SCRIPT = r"""
(async () => {
  await Promise.race([
    document.fonts?.ready || Promise.resolve(),
    new Promise((resolve) => setTimeout(resolve, 2000)),
  ]);
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
  const techmemeStory = /^(www\.)?techmeme\.com$/i.test(location.hostname)
    ? headline?.closest('.ii') : null;
  const articleRoot = techmemeStory || (headline && headline.closest('article')) ||
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
    headline.scrollIntoView({block: 'start', inline: 'nearest', behavior: 'instant'});
    window.scrollBy({top: -72, behavior: 'instant'});
  }
  if (!headline) return {headline_found: false, focus_rect: null};
  await new Promise((resolve) => setTimeout(resolve, 250));
  window.__frontierNewsCaptureHeadline = headline;

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
    headline_rect: headline.getBoundingClientRect().toJSON(),
    focus_rect: width >= 280 && height >= 100
      ? {x: left, y: top, width, height}
      : null,
  };
})()
"""


_CAPTURE_HEADLINE_RECT_SCRIPT = r"""
(() => {
  const node = window.__frontierNewsCaptureHeadline;
  return node?.isConnected ? node.getBoundingClientRect().toJSON() : null;
})()
"""


def _headline_capture_is_stable(prepared: object, after: object) -> bool:
    """Require the actual headline to stay inside this screenshot and its crop."""
    if not isinstance(prepared, dict) or not isinstance(after, dict):
        return False
    before = prepared.get("headline_rect")
    focus = prepared.get("focus_rect")
    if not isinstance(before, dict) or not isinstance(focus, dict):
        return False
    try:
        if any(abs(float(before[key]) - float(after[key])) > 2 for key in ("x", "y", "width", "height")):
            return False
        x, y, width, height = (float(after[key]) for key in ("x", "y", "width", "height"))
        fx, fy, fw, fh = (float(focus[key]) for key in ("x", "y", "width", "height"))
        return (
            width > 40 and height > 20
            and 0 <= x and 0 <= y
            and x + width <= VIEWPORT_WIDTH and y + height <= VIEWPORT_HEIGHT
            and fx <= x and fy <= y
            and x + width <= fx + fw and y + height <= fy + fh
        )
    except (KeyError, TypeError, ValueError):
        return False


async def _capture_stable_article(socket, counter, prepare_script: str) -> tuple[bytes, dict]:
    # Late fonts, ads and responsive layout can move the headline after the DOM
    # passes its language gate. Never crop a later screenshot with stale bounds.
    for _ in range(3):
        prepared = await _runtime_value(socket, counter, prepare_script)
        screenshot = await _cdp_command(
            socket, counter, "Page.captureScreenshot",
            {"format": "png", "fromSurface": True, "captureBeyondViewport": False},
        )
        after = await _runtime_value(socket, counter, _CAPTURE_HEADLINE_RECT_SCRIPT)
        if _headline_capture_is_stable(prepared, after):
            return base64.b64decode(str(screenshot.get("data") or ""), validate=True), prepared
    raise RuntimeError("Article headline moved or was outside the captured viewport/crop")


def _focused_screenshot(raw: bytes, focus_rect: object, destination: Path) -> dict:
    """Crop the exact story block and place it on a render-stable canvas."""
    with Image.open(io.BytesIO(raw)) as source:
        source.load()
        source = source.convert("RGB")
        if source.size != (CAPTURE_WIDTH, CAPTURE_HEIGHT):
            raise RuntimeError(f"Unexpected news screenshot size: {source.size}")

        rect = focus_rect if isinstance(focus_rect, dict) else {}
        try:
            left = max(0, min(CAPTURE_WIDTH - 1, int(float(rect.get("x", 0)) * CAPTURE_SCALE)))
            top = max(0, min(CAPTURE_HEIGHT - 1, int(float(rect.get("y", 0)) * CAPTURE_SCALE)))
            right = max(
                left + 1,
                min(
                    CAPTURE_WIDTH,
                    int((float(rect.get("x", 0)) + float(rect.get("width", 0))) * CAPTURE_SCALE),
                ),
            )
            bottom = max(
                top + 1,
                min(
                    CAPTURE_HEIGHT,
                    int((float(rect.get("y", 0)) + float(rect.get("height", 0))) * CAPTURE_SCALE),
                ),
            )
        except (TypeError, ValueError):
            left, top, right, bottom = 0, 0, CAPTURE_WIDTH, CAPTURE_HEIGHT

        if right - left < 280 * CAPTURE_SCALE or bottom - top < 100 * CAPTURE_SCALE:
            left, top, right, bottom = 0, 0, CAPTURE_WIDTH, CAPTURE_HEIGHT
        focused = source.crop((left, top, right, bottom))
        scale = min(1320 * CAPTURE_SCALE / focused.width, 780 * CAPTURE_SCALE / focused.height)
        rendered_size = (
            max(1, round(focused.width * scale)),
            max(1, round(focused.height * scale)),
        )
        focused = focused.resize(rendered_size, Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (CAPTURE_WIDTH, CAPTURE_HEIGHT), "white")
        canvas.paste(
            focused,
            (
                (CAPTURE_WIDTH - rendered_size[0]) // 2,
                (CAPTURE_HEIGHT - rendered_size[1]) // 2,
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
    headline_evidence = max(
        int(info.get("headline_match_words") or 0),
        int(info.get("metadata_match_words") or 0),
    )
    return bool(
        not info.get("blocking_reason")
        and info.get("headline")
        and headline_evidence >= 2
        and int(info.get("paragraph_characters") or 0) >= 80
        and words >= 30
        and (
            language.startswith("en")
            or (
                # Many legitimate publishers, including Techmeme, omit the
                # root ``lang`` attribute.  In that case require strong title
                # grounding as well as English prose instead of treating an
                # arbitrary 50-word boundary as language evidence.
                not language
                and headline_evidence >= 3
                and cjk <= max(3, words // 12)
            )
        )
    )


def _article_relay_url(info: dict, assignment: dict) -> str:
    """Return Techmeme's matched original-publisher link when it is safe."""
    source = urlsplit(str(assignment.get("source_url") or ""))
    current = urlsplit(str(info.get("page_url") or ""))
    headline_link = urlsplit(str(info.get("headline_href") or ""))
    source_host = (source.hostname or "").casefold().removeprefix("www.")
    current_host = (current.hostname or "").casefold().removeprefix("www.")
    target_host = (headline_link.hostname or "").casefold().removeprefix("www.")
    headline_evidence = max(
        int(info.get("headline_match_words") or 0),
        int(info.get("metadata_match_words") or 0),
    )
    if (
        source_host != "techmeme.com"
        or current_host != "techmeme.com"
        or headline_link.scheme not in {"http", "https"}
        or not target_host
        or target_host == "techmeme.com"
        or headline_evidence < 3
    ):
        return ""
    return headline_link.geturl()


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
            params={"url": "about:blank"},
        )
        response.raise_for_status()
        target = response.json()
    # Close failed and cancelled targets too; a stalled publisher must not keep
    # running while the next candidate is captured in the same browser.
    try:
        return await _capture_target(target, assignment, destination)
    finally:
        if target.get("id"):
            try:
                async with httpx.AsyncClient(timeout=3) as client:
                    await client.get(f"{origin}/json/close/{target['id']}")
            except (httpx.HTTPError, OSError):
                logger.debug("Could not close news capture target", exc_info=True)


async def _capture_target(target: dict, assignment: dict, destination: Path) -> dict:
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
                "deviceScaleFactor": CAPTURE_SCALE,
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
        relay_url = ""
        while loop.time() < deadline:
            try:
                value = await _runtime_value(socket, counter, page_info_script)
            except CaptureCommandTimeout:
                raise
            except RuntimeError:
                await asyncio.sleep(0.35)
                continue
            info = value if isinstance(value, dict) else {}
            relay_url = _article_relay_url(info, assignment)
            if info.get("ready_state") in {"interactive", "complete"} and (
                _page_is_english(info) or info.get("blocking_reason") or relay_url
            ):
                break
            await asyncio.sleep(0.6)
        # A verified selected source is already usable. Follow its publisher
        # only when it does not itself expose enough grounded English prose.
        if relay_url and not _page_is_english(info):
            relay_source_host = (
                urlsplit(str(info.get("page_url") or "")).hostname or ""
            ).casefold().removeprefix("www.")
            await _cdp_command(
                socket,
                counter,
                "Page.navigate",
                {"url": relay_url},
            )
            deadline = loop.time() + PAGE_LOAD_TIMEOUT_SECONDS
            info = {}
            while loop.time() < deadline:
                try:
                    value = await _runtime_value(socket, counter, page_info_script)
                except CaptureCommandTimeout:
                    raise
                except RuntimeError:
                    await asyncio.sleep(0.35)
                    continue
                info = value if isinstance(value, dict) else {}
                current_host = (
                    urlsplit(str(info.get("page_url") or "")).hostname or ""
                ).casefold().removeprefix("www.")
                if (
                    current_host != relay_source_host
                    and info.get("ready_state") in {"interactive", "complete"}
                    and (
                        _page_is_english(info) or info.get("blocking_reason")
                    )
                ):
                    break
                await asyncio.sleep(0.6)
        if not _page_is_english(info):
            if info.get("blocking_reason"):
                raise RuntimeError(
                    "Rendered source blocked article capture: "
                    f"{info['blocking_reason']}"
                )
            raise RuntimeError(
                "Rendered source did not expose an English headline and article body "
                f"(lang={info.get('document_language') or 'unknown'}, "
                f"words={info.get('english_word_count') or 0}, "
                f"headline_match={info.get('headline_match_words') or 0}, "
                f"metadata_match={info.get('metadata_match_words') or 0}, "
                f"paragraph_chars={info.get('paragraph_characters') or 0})"
            )

        raw, prepared = await _capture_stable_article(socket, counter, prepare_capture_script)
        refreshed = await _runtime_value(socket, counter, page_info_script)
        if not isinstance(refreshed, dict) or not _page_is_english(refreshed):
            raise RuntimeError("Article content failed validation after screenshot capture")
        info = refreshed
        focus = prepared.get("focus_rect") if isinstance(prepared, dict) else None
        info.update(_focused_screenshot(raw, focus, destination))

    with Image.open(destination) as image:
        image.load()
        if image.size != (CAPTURE_WIDTH, CAPTURE_HEIGHT):
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
    # Keep later eligible stories as capture fallbacks.  Publisher bot gates
    # and transient delivery failures are page-specific; they must not turn a
    # healthy edition into a 0/N result when another grounded English article
    # scene is available.
    assignments = select_overlay_assignments(
        dossier,
        storyboard,
        plans,
        limit=max(max(0, limit), len(plans)),
    )
    requested_count = min(max(0, limit), len(assignments))
    fingerprint = _fingerprint(assignments, storyboard, requested_count)
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
        "status": "capturing" if requested_count else "not_needed",
        "created_at": _now(),
        "updated_at": _now(),
        "requested_count": requested_count,
        "candidate_count": len(assignments),
        "english_only": True,
        "assignments": assignments,
        "pages": [],
        "errors": [],
    }
    _write_manifest(task_dir, manifest)
    if not requested_count:
        _emit(log, "News webpages: no English selected story matched an image/footage scene")
        return manifest

    process: asyncio.subprocess.Process | None = None
    profile: Path | None = None
    try:
        process, browser_ws_url, profile = await _start_chrome()
        for index, assignment in enumerate(assignments, start=1):
            if len(manifest["pages"]) >= requested_count:
                break
            destination = root / f"page-{index:02d}.png"
            _emit(
                log,
                f"News webpage candidate {index}/{len(assignments)}: capturing English source "
                f"{urlsplit(assignment['source_url']).hostname}",
            )
            try:
                # Navigation and Runtime.evaluate can hang before the page's
                # load polling loop gets a chance to check its own deadline.
                # Bound the entire candidate, including publisher redirects.
                async with asyncio.timeout(PAGE_CAPTURE_TIMEOUT_SECONDS):
                    info = await _capture_page(browser_ws_url, assignment, destination)
            except Exception as exc:  # noqa: BLE001 - preserve other eligible captures
                destination.unlink(missing_ok=True)
                message = (
                    f"Article capture exceeded {PAGE_CAPTURE_TIMEOUT_SECONDS:g}s deadline"
                    if isinstance(exc, TimeoutError) else str(exc)
                )
                manifest["errors"].append(
                    {
                        "scene_id": assignment["scene_id"],
                        "source_url": assignment["source_url"],
                        "message": message,
                    }
                )
                _write_manifest(task_dir, manifest)
                _emit(log, f"News webpage capture rejected: {message}")
                continue
            item = {
                **assignment,
                "captured_url": str(info.get("page_url") or assignment["source_url"]),
                "captured_source_name": str(
                    info.get("publisher_name") or assignment["source_name"]
                ),
                "content_kind": str(info.get("content_kind") or "article"),
                "document_language": str(info.get("document_language") or ""),
                "page_title": str(info.get("page_title") or ""),
                "captured_headline": str(info.get("headline") or ""),
                "headline_match_words": int(info.get("headline_match_words") or 0),
                "metadata_match_words": int(info.get("metadata_match_words") or 0),
                "paragraph_characters": int(info.get("paragraph_characters") or 0),
                "english_word_count": int(info.get("english_word_count") or 0),
                "focus_x": int(info.get("focus_x") or 0),
                "focus_y": int(info.get("focus_y") or 0),
                "focus_width": int(info.get("focus_width") or CAPTURE_WIDTH),
                "focus_height": int(info.get("focus_height") or CAPTURE_HEIGHT),
                "focused_render_width": int(
                    info.get("focused_render_width") or CAPTURE_WIDTH
                ),
                "focused_render_height": int(
                    info.get("focused_render_height") or CAPTURE_HEIGHT
                ),
                "width": CAPTURE_WIDTH,
                "height": CAPTURE_HEIGHT,
                "capture_scale": CAPTURE_SCALE,
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
        if captured == requested_count
        else ("partial" if captured else "no_results")
    )
    manifest["updated_at"] = _now()
    _write_manifest(task_dir, manifest)
    _emit(
        log,
        f"News webpages: {captured}/{requested_count} English article capture(s) ready "
        f"from {len(assignments)} eligible candidate(s)",
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
        source_url = str(item.get("captured_url") or item.get("source_url") or "")
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
                "news_webpage_source": item.get("captured_source_name")
                or item.get("source_name")
                or "English news source",
                "news_webpage_headline": item.get("captured_headline") or "",
                "news_webpage_sha256": item.get("sha256") or "",
                "news_webpage_language": "en",
            }
        )
        used_scenes.add(scene_id)
        used_urls.add(source_url)
        attached += 1
    return attached

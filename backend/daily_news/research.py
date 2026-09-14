from __future__ import annotations

import asyncio
import hashlib
import html
import json
import logging
import math
import re
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit
from zoneinfo import ZoneInfo

import httpx
from bs4 import BeautifulSoup

from backend import config
from backend.pipeline.timing import timed
from backend.daily_news.source_catalog import NewsSource, enabled_sources
from backend.daily_news.http import get_public_page

logger = logging.getLogger(__name__)
LogCallback = Callable[[str], None]

TITLE_TOKEN_RE = re.compile(r"[\w\u3400-\u9fff]+", re.UNICODE)
SPACE_RE = re.compile(r"\s+")
HTML_TAG_RE = re.compile(r"<[/!?]?[a-zA-Z][^>]*>")
TRACKING_QUERY_PREFIXES = ("utm_", "spm", "from", "source", "ref")
MIN_HEADLINE_CHARS = 14
MAX_ITEMS_PER_SOURCE = 40
MAX_ARTICLE_TEXT_CHARS = 9000
DEFAULT_HEADERS = {
    "User-Agent": "FrontierTechSN/1.0 (+local autonomous news briefing; contact operator)",
    "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml, text/html;q=0.9,*/*;q=0.7",
    "Accept-Language": "en-US,en;q=0.8,zh-CN;q=0.7,zh;q=0.6",
}

CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "ai_models": (
        "artificial intelligence", " ai ", "model", "llm", "foundation model",
        "大模型", "人工智能", "多模态", "推理模型", "机器学习",
    ),
    "agents": ("agent", "agentic", "copilot", "智能体", "代理系统"),
    "robotics": ("robot", "robotics", "humanoid", "drone", "机器人", "具身", "无人机"),
    "chips": ("chip", "semiconductor", "gpu", "accelerator", "芯片", "半导体", "算力"),
    "science": (
        "research", "paper", "quantum", "biotech", "biology", "materials", "energy",
        "研究", "论文", "量子", "生物", "材料", "能源", "科研",
    ),
    "security_policy": ("security", "cyber", "regulation", "policy", "安全", "监管", "政策"),
    "business": ("startup", "funding", "acquisition", "revenue", "创业", "融资", "收购", "公司"),
}

# General-audience morning briefings must not select explicit adult/NSFW
# product stories merely because they rank highly on a real-time aggregator.
EDITORIAL_BLOCKLIST = (
    "adult content",
    "adult entertainment",
    "erotic",
    "nsfw",
    "porn",
    "pornographic",
    "sex video",
    "spicy content",
)


@dataclass
class NewsArticle:
    id: str
    source_id: str
    source_name: str
    language: str
    title: str
    url: str
    published_at: str | None
    summary: str = ""
    category: str = "general"
    score: float = 0.0
    corroboration_count: int = 1
    corroborating_sources: list[str] = field(default_factory=list)
    evidence_text: str = ""
    fetched_at: str = ""
    content_kind: str = "news"
    lookback_hours: int | None = None
    evidence_status: str = "not_fetched"
    evidence_url: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class SourceFetch:
    source_id: str
    source_name: str
    url: str
    ok: bool
    status_code: int | None
    item_count: int
    error: str = ""


@dataclass
class ResearchDossier:
    edition_date: str
    generated_at: str
    window_hours: int
    candidates: list[NewsArticle]
    selected: list[NewsArticle]
    fetches: list[SourceFetch]

    @property
    def successful_source_count(self) -> int:
        return sum(1 for fetch in self.fetches if fetch.ok and fetch.item_count)

    def as_dict(self) -> dict:
        return {
            "edition_date": self.edition_date,
            "generated_at": self.generated_at,
            "window_hours": self.window_hours,
            "successful_source_count": self.successful_source_count,
            "candidate_count": len(self.candidates),
            "selected_count": len(self.selected),
            "fetches": [asdict(fetch) for fetch in self.fetches],
            "candidates": [article.as_dict() for article in self.candidates],
            "selected": [article.as_dict() for article in self.selected],
        }


def _log(log: LogCallback | None, message: str) -> None:
    if log:
        log(message)
    else:
        logger.info(message)


def _clean_text(value: str | None) -> str:
    if not value:
        return ""
    decoded = html.unescape(value)
    if not HTML_TAG_RE.search(decoded):
        return SPACE_RE.sub(" ", decoded).strip()
    soup = BeautifulSoup(decoded, "html.parser")
    return SPACE_RE.sub(" ", soup.get_text(" ", strip=True)).strip()


def _canonical_url(value: str, base: str) -> str:
    absolute = urljoin(base, value.strip())
    parts = urlsplit(absolute)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        return ""
    query = [
        (key, val)
        for key, val in parse_qsl(parts.query, keep_blank_values=True)
        if not key.lower().startswith(TRACKING_QUERY_PREFIXES)
    ]
    return urlunsplit((parts.scheme, parts.netloc.lower(), parts.path or "/", urlencode(query), ""))


def _parse_datetime(value: str | None) -> datetime | None:
    text = (value or "").strip()
    if not text:
        return None
    try:
        parsed = parsedate_to_datetime(text)
    except (TypeError, ValueError, OverflowError):
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _article_id(source_id: str, url: str, title: str) -> str:
    digest = hashlib.sha256(f"{source_id}\0{url}\0{title}".encode("utf-8")).hexdigest()
    return digest[:16]


def _category(title: str, summary: str) -> str:
    haystack = f" {title.lower()} {summary.lower()} "
    scores = {
        category: sum(1 for keyword in keywords if keyword in haystack)
        for category, keywords in CATEGORY_KEYWORDS.items()
    }
    category, count = max(scores.items(), key=lambda item: item[1])
    return category if count else "general"


def _title_tokens(title: str) -> set[str]:
    stop = {"the", "and", "for", "with", "that", "this", "from", "into", "about", "will", "its", "a", "an", "of", "to", "in"}
    return {token.lower() for token in TITLE_TOKEN_RE.findall(title) if len(token) > 1 and token.lower() not in stop}


def _title_similarity(left: str, right: str) -> float:
    left_tokens = _title_tokens(left)
    right_tokens = _title_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _entry_text(entry: ET.Element, *names: str) -> str:
    wanted = {name.lower() for name in names}
    for child in entry.iter():
        local = child.tag.rsplit("}", 1)[-1].lower()
        if local in wanted and (child.text or "").strip():
            return (child.text or "").strip()
    return ""


def parse_rss(payload: str, source: NewsSource, fetched_at: datetime) -> list[NewsArticle]:
    root = ET.fromstring(payload)
    entries = [node for node in root.iter() if node.tag.rsplit("}", 1)[-1].lower() in {"item", "entry"}]
    articles: list[NewsArticle] = []
    for entry in entries[:MAX_ITEMS_PER_SOURCE]:
        title = _clean_text(_entry_text(entry, "title"))
        url = _entry_text(entry, "link")
        if not url:
            for child in entry.iter():
                if child.tag.rsplit("}", 1)[-1].lower() == "link" and child.attrib.get("href"):
                    url = child.attrib["href"]
                    break
        url = _canonical_url(url, source.homepage)
        if len(title) < MIN_HEADLINE_CHARS or not url:
            continue
        published = _parse_datetime(_entry_text(entry, "pubdate", "published", "updated", "date"))
        summary = _clean_text(_entry_text(entry, "description", "summary", "content"))[:1200]
        articles.append(
            NewsArticle(
                id=_article_id(source.id, url, title),
                source_id=source.id,
                source_name=source.name,
                language=source.language,
                title=title,
                url=url,
                published_at=published.isoformat() if published else None,
                summary=summary,
                category=_category(title, summary),
                fetched_at=fetched_at.isoformat(),
            )
        )
    return articles


def parse_html(payload: str, source: NewsSource, fetched_at: datetime) -> list[NewsArticle]:
    if source.id == "aibase_daily":
        return _parse_aibase_daily(payload, source, fetched_at)
    soup = BeautifulSoup(payload, "html.parser")
    for node in soup(["script", "style", "noscript", "svg", "nav", "footer"]):
        node.decompose()
    home_parts = urlsplit(source.homepage)
    seen: set[tuple[str, str]] = set()
    articles: list[NewsArticle] = []
    for anchor in soup.select(source.article_selector or "article a[href], main a[href], h1 a[href], h2 a[href], h3 a[href], a[href]"):
        title = _clean_text(anchor.get_text(" ", strip=True) or anchor.get("title"))
        if source.id == "the_rundown_ai":
            title = re.split(r"\s+PLUS:\s+", title, maxsplit=1, flags=re.I)[0].strip()
        if len(title) < MIN_HEADLINE_CHARS or len(title) > 240:
            continue
        url = _canonical_url(str(anchor.get("href") or ""), source.homepage)
        if not url:
            continue
        parts = urlsplit(url)
        same_site = (
            parts.netloc == home_parts.netloc
            or parts.netloc.removeprefix("www.") == home_parts.netloc.removeprefix("www.")
            or parts.netloc.endswith("." + home_parts.netloc.removeprefix("www."))
            or home_parts.netloc.endswith("." + parts.netloc.removeprefix("www."))
        )
        if not same_site and source.id not in {"techmeme"}:
            continue
        if parts.path.rstrip("/") in {"", home_parts.path.rstrip("/")}:
            continue
        key = (title.casefold(), url)
        if key in seen:
            continue
        seen.add(key)
        container = anchor.find_parent(["article", "li", "section", "div"])
        summary = ""
        if container is not None:
            summary = _clean_text(container.get_text(" ", strip=True))
            if summary.casefold().startswith(title.casefold()):
                summary = summary[len(title):].strip(" —:-")
        time_node = container.find("time") if container is not None else None
        published = _parse_datetime(
            str(time_node.get("datetime") or time_node.get_text(" ", strip=True)) if time_node else None
        )
        if source.id == "tldr_ai":
            # The publisher's archive URL carries the edition date, not an
            # exact publication time. Normalize date-only evidence to UTC.
            match = re.fullmatch(r"/ai/(\d{4}-\d{2}-\d{2})", parts.path)
            if not match:
                continue
            published = _parse_datetime(match.group(1))
        articles.append(
            NewsArticle(
                id=_article_id(source.id, url, title),
                source_id=source.id,
                source_name=source.name,
                language=source.language,
                title=title,
                url=url,
                published_at=published.isoformat() if published else None,
                summary=summary[:1200],
                category=_category(title, summary),
                fetched_at=fetched_at.isoformat(),
            )
        )
        if len(articles) >= MAX_ITEMS_PER_SOURCE:
            break
    return articles


def _parse_aibase_daily(payload: str, source: NewsSource, fetched_at: datetime) -> list[NewsArticle]:
    soup = BeautifulSoup(payload, "html.parser")
    node = soup.select_one('script#__NUXT_DATA__[type="application/json"]')
    if node is None:
        raise ValueError("AIBase daily page omitted its published article data")
    table = json.loads(node.get_text())
    if not isinstance(table, list):
        raise ValueError("AIBase article data is not a reference table")
    linked_urls = {_canonical_url(a["href"], source.homepage) for a in soup.select('a[href]')}

    def scalar(row: dict, key: str):
        index = row.get(key)
        if isinstance(index, int) and 0 <= index < len(table):
            value = table[index]
            return value if isinstance(value, (str, int, float)) else None
        return None

    articles = []
    seen = set()
    for row in table:
        if not isinstance(row, dict) or not {"title", "oid", "createTime"} <= row.keys():
            continue
        oid = scalar(row, "oid")
        if not isinstance(oid, int) or oid <= 0:
            continue
        url = _canonical_url(f"/zh/daily/{oid}", source.homepage)
        title = _clean_text(str(scalar(row, "title") or ""))
        if url not in linked_urls or url in seen or len(title) < MIN_HEADLINE_CHARS:
            continue
        try:
            published = datetime.strptime(str(scalar(row, "createTime")), "%Y-%m-%d %H:%M:%S").replace(
                tzinfo=ZoneInfo("Asia/Shanghai"),
            ).astimezone(timezone.utc)
        except ValueError:
            continue
        summary = _clean_text(str(scalar(row, "description") or ""))[:1200]
        articles.append(NewsArticle(
            id=_article_id(source.id, url, title), source_id=source.id,
            source_name=source.name, language=source.language, title=title, url=url,
            published_at=published.isoformat(), summary=summary,
            category=_category(title, summary), fetched_at=fetched_at.isoformat(),
        ))
        seen.add(url)
        if len(articles) >= MAX_ITEMS_PER_SOURCE:
            break
    return articles


def parse_json_api(payload: str, source: NewsSource, fetched_at: datetime) -> list[NewsArticle]:
    value = json.loads(payload)
    if source.id == "jiqizhixin_daily":
        return _parse_jiqizhixin_articles(value, source, fetched_at)
    data = value.get("data") if isinstance(value, dict) else None
    rows = data.get("items") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        return []
    articles: list[NewsArticle] = []
    for row in rows[:MAX_ITEMS_PER_SOURCE]:
        if not isinstance(row, dict):
            continue
        title = _clean_text(str(row.get("name") or row.get("title") or ""))
        identifier = str(row.get("id") or "").strip()
        raw_url = str(row.get("article_url") or "").strip()
        url = _canonical_url(raw_url, source.homepage) if raw_url else ""
        if not url and identifier and source.id == "deeptech_china":
            url = f"https://www.mittrchina.com/news/detail/{identifier}"
        if len(title) < MIN_HEADLINE_CHARS or not url:
            continue
        timestamp = row.get("start_time")
        published = (
            datetime.fromtimestamp(float(timestamp), timezone.utc)
            if isinstance(timestamp, (int, float)) and timestamp > 0
            else _parse_datetime(str(row.get("published_at") or ""))
        )
        summary = _clean_text(str(row.get("summary") or row.get("description") or ""))[:1200]
        articles.append(NewsArticle(
            id=_article_id(source.id, url, title),
            source_id=source.id,
            source_name=source.name,
            language=source.language,
            title=title,
            url=url,
            published_at=published.isoformat() if published else None,
            summary=summary,
            category=_category(title, summary),
            fetched_at=fetched_at.isoformat(),
        ))
    return articles


def _parse_jiqizhixin_articles(value: dict, source: NewsSource, fetched_at: datetime) -> list[NewsArticle]:
    if not isinstance(value, dict) or value.get("success") is not True:
        raise ValueError("Machine Heart article API did not report success")
    rows = value.get("articles")
    if not isinstance(rows, list):
        raise ValueError("Machine Heart article API omitted articles")
    articles = []
    for row in rows[:MAX_ITEMS_PER_SOURCE]:
        if not isinstance(row, dict):
            continue
        title = _clean_text(row.get("title"))
        slug = str(row.get("slug") or "")
        if len(title) < MIN_HEADLINE_CHARS or not re.fullmatch(r"[a-zA-Z0-9_-]+", slug):
            continue
        try:
            published = datetime.strptime(row["publishedAt"], "%Y/%m/%d %H:%M").replace(
                tzinfo=ZoneInfo("Asia/Shanghai"),
            ).astimezone(timezone.utc)
        except (KeyError, TypeError, ValueError):
            continue  # Never assign the fetch time to an undated article.
        url = _canonical_url(f"/articles/{slug}", source.homepage)
        summary = _clean_text(row.get("content"))[:1200]
        articles.append(NewsArticle(
            id=_article_id(source.id, url, title), source_id=source.id,
            source_name=source.name, language=source.language, title=title,
            url=url, published_at=published.isoformat(), summary=summary,
            category=_category(title, summary), fetched_at=fetched_at.isoformat(),
        ))
    return articles


async def _request_text(client: httpx.AsyncClient, url: str, attempts: int = 3) -> tuple[str, int]:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = await get_public_page(client, url)
            response.raise_for_status()
            return response.text, response.status_code
        except (httpx.HTTPError, UnicodeError) as exc:
            last_error = exc
            if isinstance(exc, httpx.HTTPStatusError) and (
                exc.response.status_code in {401, 403, 404}
                or exc.response.headers.get("x-vercel-mitigated") == "challenge"
            ):
                break
            if attempt + 1 < attempts:
                await asyncio.sleep(min(4.0, 0.5 * 2**attempt))
    assert last_error is not None
    raise last_error


async def fetch_source(client: httpx.AsyncClient, source: NewsSource) -> tuple[list[NewsArticle], SourceFetch]:
    fetched_at = datetime.now(timezone.utc)
    url = source.feed_url or source.homepage
    try:
        payload, status = await _request_text(client, url)
        if source.fetch_mode == "rss":
            articles = parse_rss(payload, source, fetched_at)
        elif source.fetch_mode == "json_api":
            articles = parse_json_api(payload, source, fetched_at)
        elif source.fetch_mode == "rss_or_html":
            try:
                articles = parse_rss(payload, source, fetched_at)
            except (ET.ParseError, ValueError):
                payload, status = await _request_text(client, source.homepage)
                articles = parse_html(payload, source, fetched_at)
        else:
            articles = parse_html(payload, source, fetched_at)
        if not articles:
            raise RuntimeError("No candidate headlines survived source parsing")
        for article in articles:
            article.content_kind = source.content_kind
            article.lookback_hours = source.lookback_hours
        return articles, SourceFetch(source.id, source.name, url, True, status, len(articles))
    except Exception as exc:  # noqa: BLE001 - one source cannot abort the edition
        status_code = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
        return [], SourceFetch(
            source.id,
            source.name,
            url,
            False,
            status_code,
            0,
            f"{exc.__class__.__name__}: {exc}",
        )


def _within_window(article: NewsArticle, now: datetime, window_hours: int) -> bool:
    window_hours = article.lookback_hours or window_hours
    if not article.published_at:
        return True
    try:
        published = datetime.fromisoformat(article.published_at)
    except ValueError:
        return True
    return now - timedelta(hours=window_hours + 6) <= published <= now + timedelta(hours=6)


def _cluster_articles(articles: list[NewsArticle]) -> list[list[NewsArticle]]:
    clusters: list[list[NewsArticle]] = []
    for article in articles:
        match = next(
            (
                cluster
                for cluster in clusters
                if any(
                    existing.url == article.url
                    or (
                        (existing.content_kind == "analysis") == (article.content_kind == "analysis")
                        and _title_similarity(existing.title, article.title) >= 0.68
                    )
                    for existing in cluster
                )
            ),
            None,
        )
        if match is None:
            clusters.append([article])
        else:
            match.append(article)
    return clusters


def score_and_deduplicate(
    articles: Iterable[NewsArticle], sources: Iterable[NewsSource], now: datetime
) -> list[NewsArticle]:
    source_map = {source.id: source for source in sources}
    clusters = _cluster_articles(list(articles))
    ranked: list[NewsArticle] = []
    for cluster in clusters:
        representative = min(cluster, key=lambda article: (
            source_map[article.source_id].content_kind == "aggregator",
            source_map[article.source_id].priority,
        ))
        lowered = representative.title.casefold().strip(" .!—-")
        if lowered in {"one daily email", "subscribe to our newsletter", "sign up for free"}:
            continue
        editorial_text = f"{representative.title} {representative.summary}".casefold()
        if any(blocked in editorial_text for blocked in EDITORIAL_BLOCKLIST):
            continue
        distinct_sources = sorted({article.source_name for article in cluster})
        source = source_map[representative.source_id]
        published = _parse_datetime(representative.published_at)
        age_hours = max(0.0, (now - published).total_seconds() / 3600) if published else 30.0
        recency = max(0.0, 24.0 - min(24.0, age_hours) * 0.75)
        source_quality = source.rating * 6.0 + max(0.0, 15.0 - source.priority)
        corroboration = min(18.0, (len(distinct_sources) - 1) * 7.0)
        hard_tech = 5.0 if representative.category in {"robotics", "chips", "science"} else 0.0
        representative.score = round(source_quality + recency + corroboration + hard_tech, 3)
        representative.corroboration_count = len(distinct_sources)
        representative.corroborating_sources = distinct_sources
        ranked.append(representative)
    return sorted(ranked, key=lambda article: (-article.score, article.source_name, article.title))


def select_balanced(articles: list[NewsArticle], max_stories: int) -> list[NewsArticle]:
    # One recent institutional reading can close an edition, after the news.
    # It needs article evidence, not just a headline or a subscription screen.
    analyses = [
        article for article in articles
        if article.content_kind == "analysis" and article.evidence_status == "article_excerpt"
    ]
    analysis = max(analyses, key=lambda article: article.score) if analyses and max_stories >= 3 else None
    news_slots = max_stories - int(analysis is not None)
    selected: list[NewsArticle] = []
    source_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()

    def adjusted(article: NewsArticle) -> float:
        diversity = 6.0 if not category_counts[article.category] else -4.0 * category_counts[article.category]
        language = 5.0 if language_counts[article.language] == 0 else 0.0
        # Prefer six independent desks before taking a second story from any
        # one aggregator. A second item remains possible when the source quorum
        # is unusually small, but it must clear a steep diversity penalty.
        source_penalty = 32.0 * source_counts[article.source_id]
        return article.score + diversity + language - source_penalty

    remaining = [article for article in articles if article.content_kind != "analysis"]
    while remaining and len(selected) < news_slots:
        unused_sources = [article for article in remaining if source_counts[article.source_id] == 0]
        pool = unused_sources or remaining
        pool.sort(key=lambda article: (-adjusted(article), article.title))
        candidate = pool[0]
        remaining.remove(candidate)
        if source_counts[candidate.source_id] >= 2:
            continue
        selected.append(candidate)
        source_counts[candidate.source_id] += 1
        category_counts[candidate.category] += 1
        language_counts[candidate.language] += 1
    if analysis is not None:
        selected.append(analysis)
    return selected


def _extract_article_text(payload: str) -> str:
    soup = BeautifulSoup(payload, "html.parser")
    # A successful HTTP response can still be a subscription shell. Treat a
    # declared paywall conservatively and keep the public feed evidence.
    for node in soup.select('script[type="application/ld+json"]'):
        if re.search(r'"isAccessibleForFree"\s*:\s*(?:false|"false")', node.get_text(), re.I):
            return ""
    for node in soup(["script", "style", "nav", "footer", "aside", "form", "noscript", "svg"]):
        node.decompose()
    container = soup.find("article") or soup.find("main") or soup.body
    if container is None:
        return ""
    paragraphs = []
    for node in container.find_all(["p", "h2", "h3", "li"]):
        text = _clean_text(node.get_text(" ", strip=True))
        if len(text) >= 35 and text not in paragraphs:
            paragraphs.append(text)
        if sum(len(item) for item in paragraphs) >= MAX_ARTICLE_TEXT_CHARS:
            break
    return "\n".join(paragraphs)[:MAX_ARTICLE_TEXT_CHARS]


def _extract_tldr_newsletter(payload: str) -> str:
    soup = BeautifulSoup(payload, "html.parser")
    excerpts = []
    for section in soup.select("article"):
        heading = section.select_one("h3")
        body = section.select_one(".newsletter-html")
        link = section.select_one("a[href]")
        if heading is None or body is None or link is None:
            continue
        title = _clean_text(heading.get_text(" ", strip=True))
        if "sponsor" in title.casefold() or "jobs." in str(link.get("href")):
            continue
        summary = _clean_text(body.get_text(" ", strip=True))
        if len(summary) >= 35:
            excerpts.append(f"{title}\n{summary}")
    return "\n\n".join(excerpts)[:MAX_ARTICLE_TEXT_CHARS]


def _extract_published_at(payload: str) -> datetime | None:
    """Extract a machine-verifiable publication timestamp from an article page."""
    soup = BeautifulSoup(payload, "html.parser")
    candidates: list[str] = []

    for selector, attribute in (
        ('meta[property="article:published_time"]', "content"),
        ('meta[name="article:published_time"]', "content"),
        ('meta[name="date"]', "content"),
        ('meta[name="publishdate"]', "content"),
        ('meta[itemprop="datePublished"]', "content"),
        ('time[datetime]', "datetime"),
    ):
        node = soup.select_one(selector)
        if node and node.get(attribute):
            candidates.append(str(node.get(attribute)))

    def collect_json_dates(value: object) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                if key in {"datePublished", "dateCreated", "uploadDate"} and isinstance(nested, str):
                    candidates.append(nested)
                else:
                    collect_json_dates(nested)
        elif isinstance(value, list):
            for nested in value:
                collect_json_dates(nested)

    for node in soup.select('script[type="application/ld+json"]'):
        try:
            collect_json_dates(json.loads(node.string or node.get_text()))
        except (json.JSONDecodeError, TypeError):
            continue

    return next(
        (parsed for candidate in candidates if (parsed := _parse_datetime(candidate)) is not None),
        None,
    )


async def hydrate_evidence(
    client: httpx.AsyncClient,
    articles: list[NewsArticle],
    log: LogCallback | None = None,
) -> None:
    semaphore = asyncio.Semaphore(4)

    async def hydrate(article: NewsArticle) -> None:
        async with semaphore:
            try:
                article.evidence_url = article.url
                if article.source_id == "jiqizhixin_daily":
                    slug = urlsplit(article.url).path.removeprefix("/articles/")
                    if not re.fullmatch(r"[a-zA-Z0-9_-]+", slug):
                        raise ValueError("Invalid Machine Heart article slug")
                    article.evidence_url = f"https://www.jiqizhixin.com/api/article_library/articles/{slug}"
                    raw, _ = await _request_text(client, article.evidence_url, attempts=2)
                    detail = json.loads(raw)
                    if _clean_text(detail.get("title")) != article.title:
                        raise ValueError("Machine Heart detail did not match the selected title")
                    payload = f"<article>{detail.get('content') or ''}</article>"
                else:
                    payload, _ = await _request_text(client, article.url, attempts=2)
                if not article.published_at:
                    published = _extract_published_at(payload)
                    if published is None and article.source_id == "ithome_ai":
                        node = BeautifulSoup(payload, "html.parser").select_one("#pubtime_baidu")
                        if node is not None:
                            published = datetime.strptime(node.get_text(strip=True), "%Y/%m/%d %H:%M:%S").replace(
                                tzinfo=ZoneInfo("Asia/Shanghai"),
                            ).astimezone(timezone.utc)
                    article.published_at = published.isoformat() if published else None
                article.evidence_text = _extract_article_text(payload)
                if article.source_id == "tldr_ai":
                    article.evidence_text = _extract_tldr_newsletter(payload)
                elif article.source_id == "techmeme" and "This is a Techmeme archive page" in payload:
                    article.evidence_text = ""  # The archive notice is not article evidence.
                article.evidence_status = "article_excerpt" if article.evidence_text else "feed_summary"
                if not article.evidence_text:
                    article.evidence_text = article.summary
                    if not article.summary:
                        article.evidence_status = "headline_only"
            except Exception as exc:  # noqa: BLE001 - keep feed evidence if detail is blocked
                article.evidence_text = article.summary
                article.evidence_status = "feed_summary" if article.summary else "headline_only"
                _log(log, f"Evidence detail unavailable for {article.source_name}: {exc}")

    await asyncio.gather(*(hydrate(article) for article in articles))


def dossier_markdown(dossier: ResearchDossier) -> str:
    from backend.daily_news.editorial import source_language_guidance

    lines = [
        f"# ByteFront Espresso evidence dossier — {dossier.edition_date}",
        "",
        f"Generated at: {dossier.generated_at}",
        f"Source window: {dossier.window_hours} hours",
        f"Successful sources: {dossier.successful_source_count}/{len(dossier.fetches)}",
        "",
        "## Selected stories",
        "",
    ]
    for index, article in enumerate(dossier.selected, 1):
        lines.extend(
            [
                f"### {index}. {article.title}",
                f"- Source: {article.source_name} ({article.language})",
                f"- Spoken provenance: {source_language_guidance(article)}",
                f"- Content kind: {article.content_kind}",
                f"- Evidence access: {article.evidence_status} (never assume the complete article was read)",
                f"- Evidence URL: {article.evidence_url or article.url}",
                f"- Freshness window: {article.lookback_hours or dossier.window_hours} hours",
                ("- Editorial treatment: institutional viewpoint, with potential investment interests. Use 3–4 sentences, at most 120 English words or 220 Chinese characters, to explain the author's thesis, one supporting example and stated limitations; attribute opinions to the institution, never present them as independent reporting or established fact."
                 if article.content_kind == "analysis" else "- Editorial treatment: report only supported claims with source attribution."),
                f"- URL: {article.url}",
                f"- Published: {article.published_at or 'not exposed by source'}",
                f"- Category: {article.category}",
                f"- Selection score: {article.score:.1f}",
                f"- Corroborating source count: {article.corroboration_count}",
                f"- Feed summary: {article.summary or 'none'}",
                "- Article evidence excerpt:",
                article.evidence_text[:MAX_ARTICLE_TEXT_CHARS] or "No detail text was available; rely only on the headline and feed summary.",
                "",
            ]
        )
    lines.extend(["## Source fetch audit", ""])
    for fetch in dossier.fetches:
        state = f"ok, {fetch.item_count} items" if fetch.ok else f"failed: {fetch.error}"
        lines.append(f"- {fetch.source_name}: {state}")
    return "\n".join(lines).strip() + "\n"


@timed("research")
async def run_research(
    edition_date: date,
    output_dir: Path,
    *,
    max_stories: int = 6,
    window_hours: int = 36,
    log: LogCallback | None = None,
) -> ResearchDossier:
    sources = enabled_sources()
    now = datetime.now(timezone.utc)
    _log(log, f"Research: fetching {len(sources)} configured bilingual sources")
    timeout = httpx.Timeout(35.0, connect=15.0)
    limits = httpx.Limits(max_connections=8, max_keepalive_connections=4)
    async with httpx.AsyncClient(headers=DEFAULT_HEADERS, timeout=timeout, limits=limits) as client:
        results = await asyncio.gather(*(fetch_source(client, source) for source in sources))
        fetches = [result[1] for result in results]
        all_articles = [
            article
            for articles, _ in results
            for article in articles
            if _within_window(article, now, window_hours)
        ]
        if sum(1 for fetch in fetches if fetch.ok and fetch.item_count) < 3:
            failures = "; ".join(f"{fetch.source_name}: {fetch.error}" for fetch in fetches if not fetch.ok)
            raise RuntimeError(f"Research source quorum failed; fewer than 3 sources returned stories. {failures}")
        ranked = score_and_deduplicate(all_articles, sources, now)
        # Undated institutional cards need a fair chance at date verification
        # before the news batch fills the edition. Bound this extra work.
        ranked_ids = {article.id for article in ranked}
        analysis_candidates: list[NewsArticle] = []
        analysis_counts: Counter[str] = Counter()
        # Preserve publishers' latest-first listing order for undated cards;
        # sorting their tied scores alphabetically would favor old essays.
        for article in all_articles:
            if (article.content_kind == "analysis" and article.id in ranked_ids
                    and analysis_counts[article.source_id] < 4):
                analysis_candidates.append(article)
                analysis_counts[article.source_id] += 1
            if len(analysis_candidates) >= 8:
                break
        # Verify one candidate per desk before spending the batch on repeated
        # high-scoring aggregator stories. Otherwise an expanded roster can
        # still yield an edition dominated by the first few sources.
        hydration_order = list(analysis_candidates)
        seen_sources = {article.source_id for article in hydration_order}
        for article in ranked:
            if article.source_id not in seen_sources:
                hydration_order.append(article)
                seen_sources.add(article.source_id)
        queued_ids = {article.id for article in hydration_order}
        hydration_order.extend(article for article in ranked if article.id not in queued_ids)
        # Headlines without feed dates are provisionally ranked, then hydrated
        # in bounded batches. Final selection is freshness-closed: every story
        # must have an article/feed timestamp inside this edition's window.
        freshness_verified: list[NewsArticle] = []
        batch_size = max(24, max_stories * 4)
        selected: list[NewsArticle] = []
        hydrated_count = 0
        for offset in range(0, len(hydration_order), batch_size):
            batch = hydration_order[offset : offset + batch_size]
            await hydrate_evidence(client, batch, log=log)
            hydrated_count += len(batch)
            freshness_verified.extend(
                article
                for article in batch
                if article.published_at and _within_window(article, now, window_hours)
            )
            selected = select_balanced(freshness_verified, max_stories)
            if len(selected) >= max_stories:
                break
        rejected_unknown_or_stale = hydrated_count - len(freshness_verified)
        if rejected_unknown_or_stale:
            _log(
                log,
                f"Research: rejected {rejected_unknown_or_stale} hydrated headline(s) "
                "with missing or stale publication timestamps",
            )
        if len(selected) < min(3, max_stories):
            raise RuntimeError(
                f"Research selection produced only {len(selected)} usable stories from {len(all_articles)} candidates"
            )

    dossier = ResearchDossier(
        edition_date=edition_date.isoformat(),
        generated_at=now.isoformat(),
        window_hours=window_hours,
        candidates=ranked,
        selected=selected,
        fetches=fetches,
    )
    research_dir = output_dir / "research"
    research_dir.mkdir(parents=True, exist_ok=True)
    (research_dir / "dossier.json").write_text(
        json.dumps(dossier.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (research_dir / "dossier.md").write_text(dossier_markdown(dossier), encoding="utf-8")
    _log(
        log,
        f"Research: selected {len(selected)} stories from {len(ranked)} deduplicated candidates "
        f"across {dossier.successful_source_count} working sources",
    )
    return dossier

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

from backend import config


@dataclass(frozen=True)
class NewsSource:
    id: str
    language: str
    name: str
    type: str
    coverage: str
    frequency: str
    rating: int
    priority: int
    homepage: str
    feed_url: str | None
    fetch_mode: str
    enabled: bool = True

    def as_dict(self) -> dict:
        return asdict(self)


@lru_cache(maxsize=1)
def load_source_catalog() -> tuple[NewsSource, ...]:
    path = Path(config.PROJECT_ROOT) / "config" / "news_sources.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    sources = tuple(NewsSource(**row) for row in payload.get("sources", []))
    if not sources:
        raise RuntimeError(f"Daily-news source catalog is empty: {path}")
    ids = [source.id for source in sources]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Daily-news source IDs must be unique")
    return tuple(sorted(sources, key=lambda source: source.priority))


def enabled_sources() -> tuple[NewsSource, ...]:
    return tuple(source for source in load_source_catalog() if source.enabled)

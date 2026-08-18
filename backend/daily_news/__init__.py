"""Autonomous daily frontier-technology briefing pipeline."""

from backend.daily_news.source_catalog import NewsSource, load_source_catalog

__all__ = ["NewsSource", "load_source_catalog"]

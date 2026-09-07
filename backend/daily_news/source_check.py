"""Run the production source adapters and save a reviewable coverage audit."""
from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from datetime import datetime, timezone
from uuid import uuid4

import httpx

from backend import config
from backend.daily_news.research import DEFAULT_HEADERS, fetch_source, hydrate_evidence, _within_window
from backend.daily_news.source_catalog import enabled_sources


async def check_sources() -> dict:
    sources = enabled_sources()
    now = datetime.now(timezone.utc)
    limits = httpx.Limits(max_connections=8, max_keepalive_connections=4)
    async with httpx.AsyncClient(
        headers=DEFAULT_HEADERS, timeout=httpx.Timeout(35.0, connect=15.0), limits=limits,
    ) as client:
        fetched = await asyncio.gather(*(fetch_source(client, source) for source in sources))
        # Check a bounded sample for real dated evidence, not navigation links.
        samples = [articles[:3] for articles, _ in fetched]
        await hydrate_evidence(client, [article for sample in samples for article in sample])
    results = []
    for source, (articles, audit), sample in zip(sources, fetched, samples, strict=True):
        usable = [a for a in sample if a.published_at and a.evidence_text.strip()
                  and a.evidence_status in {'article_excerpt', 'feed_summary'}]
        results.append({
            **asdict(audit), 'usable': bool(audit.ok and usable),
            'dated_evidence_count': len(usable),
            'fresh_sample_count': sum(_within_window(a, now, 36) for a in usable),
            'samples': [a.as_dict() for a in sample],
        })
    check_id = f"{now.strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"
    path = config.OUTPUTS_DIR / 'source-checks' / f'{check_id}.json'
    report = {
        'check_id': check_id, 'started_at': now.isoformat(),
        'completed_at': datetime.now(timezone.utc).isoformat(),
        'source_count': len(results),
        'fetched_source_count': sum(row['ok'] for row in results),
        'usable_source_count': sum(row['usable'] for row in results),
        'all_sources_usable': all(row['usable'] for row in results),
        'sources': results, 'report_path': str(path),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    return report

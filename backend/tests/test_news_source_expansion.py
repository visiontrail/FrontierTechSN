"""Source expansion: parsing, access boundaries and actual research selection."""
import asyncio
from datetime import date, datetime, timedelta, timezone
from dataclasses import replace
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from backend.daily_news import research
from backend.daily_news import freshness
from backend.daily_news.source_catalog import load_source_catalog


def source(name):
    return next(row for row in load_source_catalog() if row.id == name)


@pytest.fixture(autouse=True)
def isolate_editorial_assessment(monkeypatch):
    # These are adapter/selection tests; semantic eligibility has separate tests.
    async def assess(rows, *args, **kwargs):
        for row in rows:
            row.freshness = {"decision": "include"}
    monkeypatch.setattr(freshness, 'load_history', AsyncMock(return_value=[]))
    monkeypatch.setattr(freshness, 'assess_candidates', assess)


def article(sid, *, days=0, score=50, kind='news', evidence='article_excerpt'):
    now = datetime.now(timezone.utc)
    return research.NewsArticle(
        id=sid, source_id=sid, source_name=sid, language='en',
        title=f'{sid} publishes a detailed technology assessment',
        url=f'https://example.com/{sid}', published_at=(now - timedelta(days=days)).isoformat(),
        score=score, content_kind=kind, evidence_status=evidence,
        lookback_hours=168 if kind == 'analysis' else None,
        evidence_text='The institution argues that compute costs shape adoption.',
    )


def test_a16z_parses_article_cards_without_navigation_or_hero_promos():
    payload = '''<main><a href="/about/">About our investment firm</a>
      <h1>A timeless promotional headline</h1><a href="/old/">Learn More</a>
      <h4><span>new</span><a href="/ai-infrastructure/">The economics of AI infrastructure</a></h4>
      <h4><a href="https://other.test/foreign">Do not import another publisher</a></h4></main>'''
    rows = research.parse_html(payload, source('a16z'), datetime.now(timezone.utc))
    assert len(rows) == 1
    assert rows[0].title == 'The economics of AI infrastructure'
    assert rows[0].url == 'https://a16z.com/ai-infrastructure/'
    assert rows[0].published_at is None


def test_atom_and_rdf_feeds_retain_article_urls_and_dates():
    now = datetime.now(timezone.utc)
    atom = '''<feed xmlns="http://www.w3.org/2005/Atom"><entry>
    <title>A new technology breakthrough</title><link href="https://www.theverge.com/story"/>
    <published>2026-09-07T01:00:00Z</published><summary>Public excerpt only.</summary>
    </entry></feed>'''
    rdf = '''<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
    xmlns="http://purl.org/rss/1.0/" xmlns:dc="http://purl.org/dc/elements/1.1/">
    <item><title>A new scientific breakthrough</title><link>https://www.nature.com/articles/test</link>
    <dc:date>2026-09-07T01:00:00Z</dc:date></item></rdf:RDF>'''
    for payload, sid in [(atom, 'the_verge'), (rdf, 'nature')]:
        rows = research.parse_rss(payload, source(sid), now)
        assert len(rows) == 1
        assert rows[0].published_at == '2026-09-07T01:00:00+00:00'
        assert rows[0].url.startswith('https://www.')


def test_analysis_legacy_lookback_cannot_widen_news_window():
    now = datetime.now(timezone.utc)
    assert not research._within_window(article('a16z', days=5, kind='analysis'), now, 36)
    assert not research._within_window(article('a16z', days=8, kind='analysis'), now, 36)
    assert not research._within_window(article('bloomberg', days=5), now, 36)


def test_institutional_readings_compete_without_a_reserved_slot():
    news = [article(sid) for sid in ['bloomberg', 'bbc_technology', 'nature', 'wired']]
    analyses = [article(sid, kind='analysis', score=10) for sid in ['a16z', 'sequoia']]
    selected = research.select_balanced(news + analyses, 4)
    assert len(selected) == 4
    assert all(a.content_kind == 'news' for a in selected)
    assert research.select_balanced(news + [replace(analyses[0], score=100)], 4)[0].source_id == 'a16z'
    blocked = [replace(a, evidence_status='feed_summary') for a in analyses]
    assert research.select_balanced(news + blocked, 4) == research.select_balanced(news, 4)
    assert all(a.content_kind == 'news' for a in research.select_balanced(news + analyses, 2))


def test_direct_publication_wins_over_aggregator_for_same_url():
    bloomberg = article('bloomberg')
    aggregator = replace(bloomberg, id='techmeme', source_id='techmeme', content_kind='aggregator')
    ranked = research.score_and_deduplicate([aggregator, bloomberg], [source('techmeme'), source('bloomberg')], datetime.now(timezone.utc))
    assert len(ranked) == 1
    assert ranked[0].source_id == 'bloomberg'


def test_authorization_failure_is_isolated_and_audited():
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(401, request=req))) as client:
            with patch.object(research.asyncio, 'sleep', new=AsyncMock()):
                rows, audit = await research.fetch_source(client, replace(source('bloomberg'), id='reuters', name='Reuters', homepage='https://www.reuters.com/technology/', feed_url=None, fetch_mode='html'))
        assert rows == []
        assert not audit.ok and audit.status_code == 401
        assert audit.source_id == 'reuters'
    asyncio.run(run())


def test_paywall_shell_never_becomes_article_evidence():
    row = article('bloomberg')
    row.summary = 'Public feed summary.'
    async def run():
        payload = '''<script type="application/ld+json">{"isAccessibleForFree":false}</script>
        <main><p>Subscribe now to gain access to exclusive expert analysis and all our articles.</p></main>'''
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(200, text=payload))) as client:
            await research.hydrate_evidence(client, [row])
        assert row.evidence_text == row.summary
        assert row.evidence_status == 'feed_summary'
    asyncio.run(run())


def test_research_verifies_undated_analysis_and_rejects_old_essay(tmp_path):
    now = datetime.now(timezone.utc)
    sources = [source(s) for s in ['bloomberg', 'bbc_technology', 'nature', 'a16z']]
    rows = [article(s.id, kind=s.content_kind) for s in sources]
    rows[-1].published_at = None
    async def fetch(client, src):
        row = next(a for a in rows if a.source_id == src.id)
        return [row], research.SourceFetch(src.id, src.name, src.homepage, True, 200, 1)
    async def hydrate(client, batch, log=None):
        for row in batch:
            if row.source_id == 'a16z':
                row.published_at = (now - timedelta(days=3)).isoformat()
    with patch.object(research, 'enabled_sources', return_value=sources), patch.object(research, 'fetch_source', side_effect=fetch), patch.object(research, 'hydrate_evidence', side_effect=hydrate):
        dossier = asyncio.run(research.run_research(date.today(), tmp_path, max_stories=3))
    assert len(dossier.selected) == 3
    assert all(a.source_id != 'a16z' for a in dossier.selected)
    text = (tmp_path / 'research/dossier.md').read_text()
    assert 'Freshness window: 36 hours' in text
    assert '168 hours' not in text
    assert 'Evidence access: article_excerpt' in text
    assert (tmp_path / 'research/dossier.json').exists()


def test_hydration_batch_gives_expanded_roster_a_chance_before_repeat_aggregator(tmp_path):
    sources = [source(s) for s in ['techmeme', 'bloomberg', 'bbc_technology', 'nature', 'wired', 'a16z']]
    rows = [replace(article('techmeme', kind='aggregator'), id=f'tech-{i}', title=f'Signal{i} device{i} launch{i}', url=f'https://example.com/tech-{i}') for i in range(30)]
    rows += [article(s.id, kind=s.content_kind) for s in sources[1:]]
    batches = []
    async def fetch(client, src):
        matches = [a for a in rows if a.source_id == src.id]
        return matches, research.SourceFetch(src.id, src.name, src.homepage, True, 200, len(matches))
    async def hydrate(client, batch, log=None):
        batches.append({a.source_id for a in batch})
    with patch.object(research, 'enabled_sources', return_value=sources), patch.object(research, 'fetch_source', side_effect=fetch), patch.object(research, 'hydrate_evidence', side_effect=hydrate):
        dossier = asyncio.run(research.run_research(date.today(), tmp_path, max_stories=6))
    assert batches[0] == {s.id for s in sources}
    assert len({a.source_id for a in dossier.selected}) == 6


def test_institutional_interpretation_length_is_bounded_in_both_languages():
    from backend.daily_news.scriptwriter import _analysis_length_failures
    row = article('a16z', kind='analysis')
    dossier = research.ResearchDossier('2026-09-07', 'now', 36, [row], [row], [])
    assert not _analysis_length_failures('Opening\n' + 'word ' * 120 + '\nClosing', dossier, 'en')
    assert _analysis_length_failures('Opening\n' + 'word ' * 121 + '\nClosing', dossier, 'en')
    assert not _analysis_length_failures('开场\n' + '字' * 220 + '\n结语', dossier, 'zh')
    assert _analysis_length_failures('开场\n' + '字' * 221 + '\n结语', dossier, 'zh')


def test_independent_review_receives_institutional_and_access_context():
    from backend.daily_news.review import _full_article_evidence
    text = _full_article_evidence(article('a16z', kind='analysis'), 1)
    assert 'Institutional analysis' in text
    assert 'evidence access: article_excerpt' in text
    assert 'not the complete article' in text

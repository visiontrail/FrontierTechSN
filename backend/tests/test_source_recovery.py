import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi import FastAPI

from backend.daily_news import http as publisher_http, research, source_check
from backend.daily_news.source_catalog import enabled_sources
from backend.routers.daily_news import router


def source(sid):
    return next(s for s in enabled_sources() if s.id == sid)


def test_machine_heart_official_api_parses_china_time_and_ignores_invalid_rows():
    rows = [dict(title='机器人协作基础模型正式发布并开放研究报告', slug='2026-09-07-5', publishedAt='2026/09/07 12:58', content='<p>公开文章摘要。</p>'),
            dict(title='这是一篇缺少发布日期的文章不能进入候选列表', slug='undated', content='摘要')]
    parsed = research.parse_json_api(json.dumps({'success': True, 'articles': rows}), source('jiqizhixin_daily'), datetime.now(timezone.utc))
    assert len(parsed) == 1
    assert parsed[0].published_at == '2026-09-07T04:58:00+00:00'
    assert parsed[0].url == 'https://www.jiqizhixin.com/articles/2026-09-07-5'
    assert parsed[0].summary == '公开文章摘要。'


def test_machine_heart_detail_uses_official_api_and_checks_article_identity():
    row = research.NewsArticle('a', 'jiqizhixin_daily', '机器之心', 'zh', '机器人协作基础模型正式发布并开放研究报告', 'https://www.jiqizhixin.com/articles/2026-09-07-5', '2026-09-07T04:58:00+00:00', summary='公开摘要')
    async def run():
        def handler(request):
            assert request.url.path == '/api/article_library/articles/2026-09-07-5'
            return httpx.Response(200, json={'title': row.title, 'content': '<p>' + '真实文章正文' * 12 + '</p>'})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await research.hydrate_evidence(client, [row])
        assert row.evidence_status == 'article_excerpt'
        assert '真实文章正文' in row.evidence_text
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(200, json={'title': 'Other article', 'content': '<p>Wrong text</p>'}))) as client:
            await research.hydrate_evidence(client, [row])
        assert row.evidence_status == 'feed_summary'
        assert row.evidence_text == '公开摘要'
    asyncio.run(run())


def test_ft_dns_fallback_preserves_host_sni_and_ttl_cache():
    publisher_http._dns_cache.clear()
    calls = []
    def handler(req):
        calls.append(req)
        if req.url.host == 'www.ft.com':
            raise httpx.ConnectTimeout('Local DNS destination did not respond', request=req)
        if req.url.host == 'cloudflare-dns.com':
            assert req.url.params['name'] == 'www.ft.com'
            return httpx.Response(200, json={'Status': 0, 'Answer': [{'type': 1, 'data': '104.18.5.165', 'TTL': 120}]})
        assert req.url.host == '104.18.5.165'
        assert req.headers['host'] == 'www.ft.com'
        assert req.extensions['sni_hostname'] == 'www.ft.com'
        assert req.url.path == '/technology'
        assert req.url.params['format'] == 'rss'
        return httpx.Response(200, text='<rss/>')
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            for _ in range(2):
                response = await publisher_http.get_public_page(c, 'https://www.ft.com/technology?format=rss')
                assert response.status_code == 200
    try:
        asyncio.run(run())
        assert [r.url.host for r in calls] == ['www.ft.com', 'cloudflare-dns.com', '104.18.5.165', '104.18.5.165']
    finally:
        publisher_http._dns_cache.clear()


def test_dns_fallback_rejects_private_addresses_and_does_not_rewrite_other_publishers():
    publisher_http._dns_cache.clear()
    calls = []
    def handler(req):
        calls.append(req.url.host)
        if req.url.host == 'cloudflare-dns.com':
            return httpx.Response(200, json={'Status': 0, 'Answer': [{'type': 1, 'data': '127.0.0.1', 'TTL': 60}]})
        raise httpx.ConnectError('unreachable', request=req)
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            with pytest.raises(RuntimeError, match='non-public'):
                await publisher_http.get_public_page(c, 'https://www.ft.com/technology')
            with pytest.raises(httpx.ConnectError):
                await publisher_http.get_public_page(c, 'https://example.org/news')
    asyncio.run(run())
    assert calls == ['www.ft.com', 'cloudflare-dns.com', 'example.org']


def test_authentication_and_vercel_challenges_are_not_retried():
    async def run():
        for status, headers in [(401, {}), (429, {'x-vercel-mitigated': 'challenge'})]:
            calls = []
            def handler(req):
                calls.append(req)
                return httpx.Response(status, headers=headers)
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
                with pytest.raises(httpx.HTTPStatusError):
                    await research._request_text(c, 'https://example.org/feed')
            assert len(calls) == 1
    asyncio.run(run())


def test_source_check_endpoint_runs_adapters_and_rejects_navigation_only(tmp_path):
    sources = [source('bloomberg'), source('jiqizhixin_daily')]
    good = research.NewsArticle('good', 'bloomberg', 'Bloomberg', 'en', 'New chips enter production today', 'https://example.org/good', '2026-09-07T04:00:00+00:00', summary='An evidence-bearing public summary.')
    bad = research.NewsArticle('bad', 'jiqizhixin_daily', '机器之心', 'zh', 'Navigation-only promotional page', 'https://example.org/bad', None)
    async def fetch(client, src):
        row = good if src.id == 'bloomberg' else bad
        return [row], research.SourceFetch(src.id, src.name, src.homepage, True, 200, 1)
    async def hydrate(client, rows):
        good.evidence_text = good.summary
        good.evidence_status = 'feed_summary'
        bad.evidence_text = 'Subscribe to our newsletter'
        bad.evidence_status = 'article_excerpt'
    async def run():
        app = FastAPI()
        app.include_router(router)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as c:
            response = await c.post('/api/daily-news/sources/check')
        assert response.status_code == 200
        result = response.json()
        assert result['fetched_source_count'] == 2
        assert result['usable_source_count'] == 1
        assert not result['all_sources_usable']
        assert len(list((tmp_path / 'source-checks').glob('*.json'))) == 1
    with patch.object(source_check, 'enabled_sources', return_value=sources), patch.object(source_check, 'fetch_source', side_effect=fetch), patch.object(source_check, 'hydrate_evidence', side_effect=hydrate), patch.object(source_check.config, 'OUTPUTS_DIR', tmp_path):
        asyncio.run(run())


def test_aibase_reads_article_records_not_navigation_or_relative_dates():
    source_row = source('aibase_daily')
    table = [
        {'title': 1, 'oid': 2, 'createTime': 3, 'description': 4},
        'AI日报：最新模型发布以及行业进展报道', 30848, '2026-09-04 16:26:39', '公开日报内容摘要。',
        {'title': 1, 'oid': 6, 'createTime': 3, 'description': 4}, 99999,
    ]
    payload = '<a href="/daily">Browse the daily archive</a><a href="/zh/daily/30848">2 天前</a><script id="__NUXT_DATA__" type="application/json">' + json.dumps(table) + '</script>'
    rows = research.parse_html(payload, source_row, datetime.now(timezone.utc))
    assert len(rows) == 1
    assert rows[0].title == table[1]
    assert rows[0].published_at == '2026-09-04T08:26:39+00:00'
    assert rows[0].url == 'https://news.aibase.com/zh/daily/30848'


def test_tldr_archive_dates_and_newsletter_body_exclude_recruiting_card():
    payload = '<a href="/ai/2026-09-04">Latest models, robotics and scientific breakthroughs</a><a href="/ai">one daily email</a>'
    rows = research.parse_html(payload, source('tldr_ai'), datetime.now(timezone.utc))
    assert len(rows) == 1
    assert rows[0].published_at == '2026-09-04T00:00:00+00:00'
    body = '''<article><a href="https://jobs.example.com"><h3>Recruiting engineers</h3></a><div class="newsletter-html">This recruiting advertisement should not become news evidence.</div></article>
    <article><a href="https://publisher.example.org/new-model"><h3>A new AI model (5 minute read)</h3></a><div class="newsletter-html">The publisher describes a new model with supporting evaluation results.</div></article>'''
    text = research._extract_tldr_newsletter(body)
    assert 'supporting evaluation results' in text
    assert 'recruiting' not in text


def test_ithome_visible_publication_time_is_parsed_in_china_timezone():
    row = research.NewsArticle('a', 'ithome_ai', 'ITHome', 'zh', '科大讯飞发布最新人工智能模型并公布参数', 'https://www.ithome.com/0/999/204.htm', None)
    async def run():
        payload = '<span id="pubtime_baidu">2026/9/7 11:56:28</span><article><p>' + '真实报道内容' * 10 + '</p></article>'
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(200, text=payload))) as c:
            await research.hydrate_evidence(c, [row])
        assert row.published_at == '2026-09-07T03:56:28+00:00'
        assert row.evidence_status == 'article_excerpt'
    asyncio.run(run())


def test_techmeme_archive_notice_is_not_full_article_evidence():
    row = research.NewsArticle('a', 'techmeme', 'Techmeme', 'en', 'A newly published report about chip production', 'https://www.techmeme.com/260907/p3', '2026-09-07T00:00:00+00:00', summary='The original publisher reports new production capacity.')
    async def run():
        payload = '<p>This is a Techmeme archive page. It shows how the site appeared earlier today.</p>'
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(200, text=payload))) as c:
            await research.hydrate_evidence(c, [row])
        assert row.evidence_status == 'feed_summary'
        assert row.evidence_text == row.summary
    asyncio.run(run())

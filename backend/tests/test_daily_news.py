import asyncio
import json
import math
import struct
import wave
import warnings
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from backend import config
from backend.daily_news import scheduler, scriptwriter
from backend.daily_news.scheduler import _is_due, create_daily_task, next_run_at
from backend.daily_news import research, review
from backend.daily_news.scriptwriter import (
    SOURCE_SPOKEN_ALIASES,
    _minimalize_unsupported_paragraphs,
    daily_script_duration_report,
    enforce_script_contract,
    morning_opening,
    narration_duration_report,
    script_contract_report,
)
from backend.daily_news.source_catalog import load_source_catalog
from backend.models import DailyAutomationSettings, SourceType, TaskConfig, TaskResponse, TaskStatus
from backend.pipeline import orchestrator
from backend.pipeline.music import build_music_prompt, mix_narration_and_music
from backend.pipeline.opencli import OpenCLIError, OpenCLIResult
from backend.publishing import prepare_apple_podcast, read_publication_manifest


def _tone(path: Path, seconds: float, frequency: float) -> None:
    rate = 48_000
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        samples = bytearray()
        for index in range(int(rate * seconds)):
            value = int(5000 * math.sin(2 * math.pi * frequency * index / rate))
            samples.extend(struct.pack("<h", value))
        stream.writeframes(samples)


def _task(root: Path, audio: Path) -> TaskResponse:
    return TaskResponse(
        id="daily-test-001",
        created_at="2026-08-18T00:00:00+00:00",
        updated_at="2026-08-18T00:00:00+00:00",
        source_type=SourceType.NEWS_DAILY,
        source_title="Frontier Tech Daily — 2026-08-18",
        status=TaskStatus.COMPLETE,
        config=TaskConfig(),
        output_dir=str(root / "daily-test-001"),
        audio_path=str(audio),
        duration_seconds=2.0,
        origin_type="daily_news",
        origin_id="2026-08-18",
    )


def test_source_catalog_contains_global_reporting_and_institutional_analysis():
    load_source_catalog.cache_clear()
    sources = load_source_catalog()

    assert len(sources) == 23
    by_id = {source.id: source for source in sources}
    assert {"bloomberg", "financial_times", "wall_street_journal", "nature"} <= by_id.keys()
    assert by_id["a16z"].content_kind == "analysis"
    assert by_id["a16z"].lookback_hours == 168
    assert by_id["a16z"].feed_url is None
    assert by_id["sequoia"].content_kind == "analysis"
    assert [source.priority for source in sources[:6]] == [1, 2, 3, 4, 5, 6]
    assert {source.language for source in sources} == {"en", "zh"}


def test_article_page_json_ld_supplies_machine_verifiable_publication_date():
    payload = """<html><head><script type="application/ld+json">
    {"@type":"NewsArticle","datePublished":"2026-08-18T06:30:00+08:00"}
    </script></head><body><article><p>This paragraph contains enough text to be evidence.</p></article></body></html>"""

    published = research._extract_published_at(payload)

    assert published == datetime(2026, 8, 17, 22, 30, tzinfo=timezone.utc)


def test_clean_text_does_not_parse_plain_url_as_html():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cleaned = research._clean_text("https://www.example.com/story?a=1&amp;b=2")

    assert cleaned == "https://www.example.com/story?a=1&b=2"


def test_clean_text_still_removes_html_markup():
    assert research._clean_text("<p>Hello&nbsp;<b>world</b></p>") == "Hello world"


def test_balanced_selection_uses_distinct_sources_before_second_story():
    articles = [
        research.NewsArticle(
            id=str(index), source_id=source, source_name=source, language="en",
            title=f"A sufficiently descriptive headline number {index}", url=f"https://example.com/{index}",
            published_at="2026-08-18T00:00:00+00:00", score=100 - index,
        )
        for index, source in enumerate(["a", "a", "b", "c"])
    ]

    selected = research.select_balanced(articles, 3)

    assert {article.source_id for article in selected} == {"a", "b", "c"}


def test_general_audience_research_excludes_explicit_adult_product_stories():
    source = load_source_catalog()[0]
    now = datetime(2026, 8, 18, 6, 30, tzinfo=timezone.utc)
    articles = [
        research.NewsArticle(
            id="adult", source_id=source.id, source_name=source.name, language="en",
            title="Studio launches an AI video tool for adult content", url="https://example.com/adult",
            published_at=now.isoformat(), summary="An adult entertainment generator.",
        ),
        research.NewsArticle(
            id="robot", source_id=source.id, source_name=source.name, language="en",
            title="Engineers demonstrate a new assistive robotics controller", url="https://example.com/robot",
            published_at=now.isoformat(), summary="A mobility research prototype.",
        ),
    ]

    ranked = research.score_and_deduplicate(articles, [source], now)

    assert [article.id for article in ranked] == ["robot"]


def test_batch_review_token_preserves_per_story_error_codes():
    payload = review._batch_payload("1P2D3BC4P", 4)

    assert payload["approved"] is False
    assert [issue["evidence_story_numbers"] for issue in payload["issues"]] == [[2], [3]]
    assert "extrapolative sentence" in payload["issues"][0]["correction"]


def test_web_claim_protocol_requires_search_and_preserves_exact_failed_claims():
    claims = {
        1: {"1.1": "Story one is supported."},
        2: {
            "2.1": "The robot played table tennis.",
            "2.2": "It was the first autonomous complete match.",
        },
    }

    payload = review._batch_payload(
        "W1P2D@2.2",
        [1, 2],
        claim_catalog=claims,
        require_web=True,
    )

    assert payload["approved"] is False
    assert payload["web_searched"] is True
    assert payload["issues"][0]["claim_ids"] == ["2.2"]
    assert payload["issues"][0]["claim_texts"] == [
        "It was the first autonomous complete match."
    ]
    with pytest.raises(ValueError, match="mandatory live web search"):
        review._batch_payload(
            "1P2D@2.2",
            [1, 2],
            claim_catalog=claims,
            require_web=True,
        )
    with pytest.raises(ValueError, match="unavailable"):
        review._batch_payload(
            "N",
            [1, 2],
            claim_catalog=claims,
            require_web=True,
        )
    with pytest.raises(ValueError, match="unknown or cross-story"):
        review._batch_payload(
            "W1P2D@1.1",
            [1, 2],
            claim_catalog=claims,
            require_web=True,
        )


def test_web_claim_protocol_disambiguates_multiple_failed_story_segments():
    claims = {
        1: {
            "1.1": "The acquisition was reported.",
            "1.2": "The startup raised a seed round.",
        },
        2: {
            "2.1": "The robot watched one demonstration.",
            "2.2": "The task lasted ten minutes.",
            "2.4": "The success rate was 66 percent.",
        },
    }

    separated = review._batch_payload(
        "W1E@1.1;2E@2.2,2.4",
        [1, 2],
        claim_catalog=claims,
        require_web=True,
    )
    legacy_live_response = review._batch_payload(
        "W1E@1.12E\n@2.2,2.4",
        [1, 2],
        claim_catalog=claims,
        require_web=True,
    )

    for payload in (separated, legacy_live_response):
        assert payload["approved"] is False
        assert [issue["claim_ids"] for issue in payload["issues"]] == [
            ["1.1"],
            ["2.2", "2.4"],
        ]


def test_saved_web_review_is_reused_only_for_the_exact_prompt(tmp_path: Path):
    prompt_path = tmp_path / "prompt.txt"
    response_path = tmp_path / "response.txt"
    prompt = "exact dated prompt with numbered claims"
    claims = {
        1: {"1.1": "Story one is supported."},
        2: {"2.1": "Story two needs attribution."},
    }
    prompt_path.write_text(prompt, encoding="utf-8")
    response_path.write_text(
        "[GEMINI 1]\nN\n\n--- PROVIDER ATTEMPT ---\n\n"
        "[CHATGPT FALLBACK]\nW1P;2E@2.1",
        encoding="utf-8",
    )

    cached = review._cached_batch_review(
        prompt_path,
        response_path,
        prompt,
        [1, 2],
        claims,
    )

    assert cached is not None
    payload, raw, conversation_url, provider = cached
    assert payload["issues"][0]["claim_ids"] == ["2.1"]
    assert "CHATGPT FALLBACK" in raw
    assert conversation_url == ""
    assert provider == "chatgpt"
    assert review._cached_batch_review(
        prompt_path,
        response_path,
        prompt + " changed",
        [1, 2],
        claims,
    ) is None

    response_path.write_text("[GEMINI 1]\nW1P;2P", encoding="utf-8")
    assert review._cached_batch_review(
        prompt_path,
        response_path,
        prompt,
        [1, 2],
        claims,
    ) is None


def test_saved_duration_candidate_requires_a_complete_prompt_matched_audit(tmp_path: Path):
    articles = [
        research.NewsArticle(
            id=f"story-{number}",
            source_id=f"source-{number}",
            source_name=f"Source {number}",
            language="en",
            title=f"Evidence title {number}",
            url=f"https://example.com/{number}",
            published_at="2026-08-26T00:00:00+00:00",
            summary=f"Supported summary {number}.",
            evidence_text=f"Supported evidence {number}.",
        )
        for number in (1, 2)
    ]
    dossier = research.ResearchDossier(
        "2026-08-26", "now", 36, articles, articles, [],
    )
    candidate = (
        "Opening.\n"
        "Source 1 reports supported evidence 1.\n"
        "Source 2 reports supported evidence 2.\n"
        "Closing."
    )
    review_dir = tmp_path / "review"
    review_dir.mkdir()
    (review_dir / "candidate-duration-cycle-1.txt").write_text(
        candidate,
        encoding="utf-8",
    )
    prompt = review._batch_review_prompt(
        candidate,
        dossier,
        date(2026, 8, 26),
        [1, 2],
    )
    prompt_path = review_dir / "story-review-prompt-1-group-1.txt"
    prompt_path.write_text(prompt, encoding="utf-8")
    (review_dir / "story-review-response-1-group-1.txt").write_text(
        "[CHATGPT FALLBACK]\nW1P;2P",
        encoding="utf-8",
    )

    assert review._saved_full_review_candidate(
        tmp_path,
        dossier,
        date(2026, 8, 26),
    ) == candidate

    prompt_path.write_text(prompt + " changed", encoding="utf-8")
    assert review._saved_full_review_candidate(
        tmp_path,
        dossier,
        date(2026, 8, 26),
    ) is None


def test_review_prompt_requires_live_search_and_carries_full_evidence():
    article = research.NewsArticle(
        id="robot",
        source_id="qbitai",
        source_name="量子位 QbitAI",
        language="zh",
        title="人形机器人自主乒乓球完整对局",
        url="https://example.com/robot",
        published_at="2026-08-26T03:02:34+00:00",
        summary="现场爆满",
        evidence_text=(
            "机器人完成了公开演示。\n"
            "真·人类史上首场人形机器人自主乒乓球完整对局！"
        ),
    )
    dossier = research.ResearchDossier(
        "2026-08-26", "now", 36, [article], [article], [],
    )
    script = "Opening.\nQbitAI reports the first autonomous complete humanoid table-tennis match.\nClosing."

    prompt = review._batch_review_prompt(
        script,
        dossier,
        date(2026, 8, 26),
        [1],
    )

    assert "MUST use live internet search" in prompt
    assert "If live web search is unavailable" in prompt
    assert "Original URL: https://example.com/robot" in prompt
    assert "CLAIM 1.1:" in prompt
    assert "真·人类史上首场人形机器人自主乒乓球完整对局" in prompt
    assert "Do not browse" not in prompt
    assert "W1P2D" not in prompt


def test_bilingual_claim_matcher_preserves_dossier_order_for_translated_evidence():
    articles = [
        research.NewsArticle(
            id="clone", source_id="deeptech", source_name="DeepTech 深科技", language="zh",
            title="克隆研究", url="https://example.com/clone", published_at="2026-08-18T00:00:00+00:00",
            summary="CRISPR 技术去除 Y 染色体。", evidence_text="CRISPR 技术去除 Y 染色体。",
        ),
        research.NewsArticle(
            id="material", source_id="deeptech", source_name="DeepTech 深科技", language="zh",
            title="人工智能与材料验证", url="https://example.com/material", published_at="2026-08-18T00:00:00+00:00",
            summary="候选材料需要实验验证。", evidence_text="候选材料需要实验验证。",
        ),
    ]
    dossier = research.ResearchDossier("2026-08-18", "now", 36, articles, articles, [])
    script = "\n".join([
        "Opening.",
        "DeepTech China reports a CRISPR-based cloning technique.",
        "DeepTech China reports that AI-generated materials require laboratory validation.",
        "Closing.",
    ])

    matched = review._matched_script_claims(script, dossier)

    assert "CRISPR-based" in matched[1]
    assert "materials" in matched[2]


def test_repeated_d_audit_keeps_only_the_direct_lead_sentence():
    script = "\n".join([
        "Opening.",
        "QbitAI reports that MyContext is open source. It also crossed a promotional ranking.",
        "Closing.",
    ])
    issues = [{
        "claim": "Web audit codes D for story 1",
        "evidence_story_numbers": [1],
    }]

    revised = _minimalize_unsupported_paragraphs(script, issues)

    assert revised.splitlines()[1] == "QbitAI reports that MyContext is open source."


def test_d_only_audit_rewrites_from_evidence_before_trimming():
    edition = date(2026, 8, 26)
    opening = morning_opening(edition, "en")
    closing = "Thanks for listening."
    article = research.NewsArticle(
        id="robot",
        source_id="deeptech_china",
        source_name="DeepTech 深科技",
        language="zh",
        title="机器人尚未进入可预测阶段",
        url="https://example.com/robot",
        published_at="2026-08-25T12:57:50+00:00",
        summary="2024 年末，实验室里一台机器人正在接受叠衣服测试。",
        evidence_text="2024 年末，实验室里一台机器人正在接受叠衣服测试。",
    )
    dossier = research.ResearchDossier(
        "2026-08-26", "now", 36, [article], [article], [],
    )
    original = "\n".join([
        opening,
        "DeepTech China reports that robotics has entered a scaling era.",
        closing,
    ])
    corrected = json.dumps({
        "1": (
            "DeepTech China reports that a robot was tested on folding clothes in "
            "late 2024. This proves robotics can scale predictably."
        ),
    })
    issues = [{
        "claim": "Web audit codes D for story 1",
        "evidence_story_numbers": [1],
    }]

    with (
        patch.object(
            scriptwriter,
            "_resolve_provider",
            AsyncMock(return_value=("https://example.com", "yinhe-thinking", "key")),
        ),
        patch.object(scriptwriter, "_chat", AsyncMock(return_value=corrected)) as chat,
    ):
        revised = asyncio.run(
            scriptwriter.revise_daily_script(
                original,
                dossier,
                issues,
                edition,
                language="en",
                closing_remarks=closing,
                ai_endpoint=None,
                ai_model=None,
                provider_id=None,
            )
        )

    assert revised.splitlines()[1] == (
        "The Chinese-language outlet DeepTech China reports that a robot was tested on folding clothes in late 2024."
    )
    assert "Do not rely on the title alone" in chat.await_args.args[0]
    assert "MUST NOT copy a cited blocked sentence back unchanged" in chat.await_args.args[0]
    assert "distinguish source attribution from claim attribution" in chat.await_args.args[0]
    assert "CURRENT FAILED STORY PARAGRAPHS" in chat.await_args.args[1]
    assert "CURRENT SCRIPT" not in chat.await_args.args[1]
    assert chat.await_args.args[3] == "yinhe-thinking"
    assert chat.await_args.kwargs["disable_thinking"] is True


def test_c_audit_removes_a_rewritten_sentence_that_keeps_disputed_numbers():
    corrections = {
        2: (
            "Skild says S1 learns from one demonstration. "
            "QbitAI reports tasks over ten minutes and 66 percent success. "
            "The report describes traditional post-training as more expensive."
        )
    }
    issues = [{
        "claim": "Web audit codes C for story 2 at claims 2.2",
        "claim_ids": ["2.2"],
        "claim_texts": [
            "QbitAI reports tasks over ten minutes, with 66 percent success versus 9 percent."
        ],
        "evidence_story_numbers": [2],
    }]

    cleaned = scriptwriter._remove_persisting_cited_numeric_claims(
        corrections,
        issues,
    )

    assert cleaned[2] == (
        "Skild says S1 learns from one demonstration. "
        "The report describes traditional post-training as more expensive."
    )


def test_e_audit_attributes_the_company_claim_not_only_the_publication():
    corrections = {
        3: (
            "Luz Ding of Bloomberg reports Alibaba has released Qwen3.8-Flash. "
            "The company says the model is lower-priced."
        )
    }
    issues = [{
        "claim": "Web audit codes E for story 3 at claims 3.1",
        "claim_ids": ["3.1"],
        "claim_texts": [
            "Bloomberg reports Alibaba has released Qwen3.8-Flash."
        ],
        "evidence_story_numbers": [3],
    }]

    cleaned = scriptwriter._ensure_persisting_company_claim_attribution(
        corrections,
        issues,
    )

    assert cleaned[3].startswith(
        "Luz Ding of Bloomberg reports Alibaba says it has released Qwen3.8-Flash."
    )


def test_audit_correction_cannot_duplicate_passing_story_paragraphs():
    edition = date(2026, 8, 26)
    opening = morning_opening(edition, "en")
    closing = "Thanks for listening."
    articles = [
        research.NewsArticle(
            id=str(index),
            source_id=f"source-{index}",
            source_name=f"Source {index}",
            language="en",
            title=f"Story {index}",
            url=f"https://example.com/{index}",
            published_at="2026-08-25T00:00:00+00:00",
            summary=f"Evidence for story {index}.",
            evidence_text=f"Evidence for story {index}.",
        )
        for index in range(1, 7)
    ]
    dossier = research.ResearchDossier(
        "2026-08-26", "now", 36, articles, articles, [],
    )
    original_story_lines = [
        f"Source {index} reports the original paragraph for story {index}."
        for index in range(1, 7)
    ]
    original = "\n".join([opening, *original_story_lines, closing])
    issues = [{
        "claim": "Web audit codes B for story 3 at claim 3.1",
        "evidence_story_numbers": [3],
        "claim_ids": ["3.1"],
        "claim_texts": [original_story_lines[2]],
    }]
    response = json.dumps({
        "3": "Source 3 reports the corrected paragraph for story 3.",
    })

    with (
        patch.object(
            scriptwriter,
            "_resolve_provider",
            AsyncMock(return_value=("https://example.com", "yinhe-chat", "key")),
        ),
        patch.object(scriptwriter, "_chat", AsyncMock(return_value=response)),
    ):
        revised = asyncio.run(
            scriptwriter.revise_daily_script(
                original,
                dossier,
                issues,
                edition,
                language="en",
                closing_remarks=closing,
                ai_endpoint=None,
                ai_model=None,
                provider_id=None,
            )
        )

    revised_lines = revised.splitlines()
    assert len(revised_lines) == 8
    assert revised_lines[1:3] == original_story_lines[:2]
    assert revised_lines[3] == "Source 3 reports the corrected paragraph for story 3."
    assert revised_lines[4:7] == original_story_lines[3:]
    assert revised.count(original_story_lines[0]) == 1
    assert revised.count(original_story_lines[1]) == 1


def test_contract_rejects_observed_duplicate_story_paragraphs():
    dossier = _attribution_dossier()
    opening = morning_opening(date(2026, 8, 24), "en")
    closing = "Thanks for listening."
    stories = [
        "Axios reports growing data-center opposition.",
        "QbitAI reports a robot coffee shop.",
        "DeepTech China reports progress in coding agents.",
        "Bloomberg reports that a foldable phone prototype is being tested.",
    ]
    duplicated = "\n".join([
        opening,
        stories[0],
        stories[1],
        stories[0],
        stories[1],
        stories[2],
        stories[3],
        closing,
    ])

    report = script_contract_report(
        duplicated,
        dossier,
        date(2026, 8, 24),
        language="en",
        closing_remarks=closing,
    )

    assert report["passed"] is False
    assert "script paragraph contract changed: expected 6, found 8" in report["failures"]


def test_duration_repair_restores_evidence_only_story_paragraphs():
    edition = date(2026, 8, 26)
    opening = morning_opening(edition, "en")
    closing = "Thanks for listening."
    articles = [
        research.NewsArticle(
            id=str(index),
            source_id=f"source-{index}",
            source_name=f"Source {index}",
            language="en",
            title=f"Story {index}",
            url=f"https://example.com/{index}",
            published_at="2026-08-25T00:00:00+00:00",
            summary=f"Evidence for story {index}.",
            evidence_text=f"Evidence for story {index}.",
        )
        for index in (1, 2)
    ]
    dossier = research.ResearchDossier(
        "2026-08-26", "now", 36, articles, articles, [],
    )
    protected = "Source 2 reports direct evidence for story two."
    original = "\n".join([opening, "Too short.", protected, closing])
    expanded = "\n".join([
        opening,
        " ".join(["evidence"] * 390) + ".",
        "Source 2 reports an unsupported replacement.",
        closing,
    ])

    with (
        patch.object(
            scriptwriter,
            "_resolve_provider",
            AsyncMock(return_value=("https://example.com", "yinhe-thinking", "key")),
        ),
        patch.object(scriptwriter, "_chat", AsyncMock(return_value=expanded)) as chat,
    ):
        repaired, report = asyncio.run(
            scriptwriter.fit_daily_script_duration(
                original,
                dossier,
                edition,
                target_duration_minutes=3,
                language="en",
                closing_remarks=closing,
                ai_endpoint=None,
                ai_model=None,
                provider_id=None,
                protected_story_numbers={2},
            )
        )

    assert repaired.splitlines()[2] == protected
    assert report["passed"] is True
    assert "Story paragraphs [2]" in chat.await_args.args[0]
    assert chat.await_args.args[3] == "yinhe-thinking"
    assert chat.await_args.kwargs["max_tokens"] == scriptwriter.DAILY_NEWS_EDIT_MAX_TOKENS
    assert chat.await_args.kwargs["disable_thinking"] is True


def test_duration_repair_closes_remaining_gap_after_protected_restore():
    edition = date(2026, 8, 26)
    opening = morning_opening(edition, "en")
    closing = "Thanks for listening."
    articles = [
        research.NewsArticle(
            id=str(index),
            source_id=f"source-{index}",
            source_name=f"Source {index}",
            language="en",
            title=f"Story {index}",
            url=f"https://example.com/{index}",
            published_at="2026-08-25T00:00:00+00:00",
            summary=f"Evidence for story {index}.",
            evidence_text=f"Evidence for story {index}.",
        )
        for index in (1, 2)
    ]
    dossier = research.ResearchDossier(
        "2026-08-26", "now", 36, articles, articles, [],
    )
    protected = "Source 2 reports direct evidence for story two."
    original = "\n".join([opening, "Too short.", protected, closing])

    def candidate(words: int) -> str:
        return "\n".join([
            opening,
            " ".join(["evidence"] * words) + ".",
            "Source 2 reports an unsupported replacement.",
            closing,
        ])

    with (
        patch.object(
            scriptwriter,
            "_resolve_provider",
            AsyncMock(return_value=("https://example.com", "yinhe-thinking", "key")),
        ),
        patch.object(
            scriptwriter,
            "_chat",
                AsyncMock(side_effect=[candidate(250), candidate(350)]),
        ) as chat,
    ):
        repaired, report = asyncio.run(
            scriptwriter.fit_daily_script_duration(
                original,
                dossier,
                edition,
                target_duration_minutes=3,
                language="en",
                closing_remarks=closing,
                ai_endpoint=None,
                ai_model=None,
                provider_id=None,
                protected_story_numbers={2},
            )
        )

    assert chat.await_count == 2
    assert repaired.splitlines()[2] == protected
    assert report["passed"] is True
    assert "A prior edit still measured" in chat.await_args_list[1].args[0]


def test_duration_repair_pins_semantic_retry_to_successful_backup_route():
    edition = date(2026, 8, 28)
    opening = morning_opening(edition, "en")
    closing = "Thanks for listening."
    articles = [
        research.NewsArticle(
            id=str(index),
            source_id=f"source-{index}",
            source_name=f"Source {index}",
            language="en",
            title=f"Story {index}",
            url=f"https://example.com/{index}",
            published_at="2026-08-28T00:00:00+00:00",
            summary=f"Evidence for story {index}.",
            evidence_text=f"Evidence for story {index}.",
        )
        for index in (1, 2)
    ]
    dossier = research.ResearchDossier(
        "2026-08-28", "now", 36, articles, articles, [],
    )

    def candidate(words: int) -> str:
        return "\n".join([
            opening,
            "Source 1 reports " + " ".join(["evidence"] * words) + ".",
            "Source 2 reports " + " ".join(["detail"] * words) + ".",
            closing,
        ])

    calls: list[tuple[str, str]] = []
    prompts: list[str] = []

    async def routed_chat(*args, **kwargs):
        calls.append((args[2], args[3]))
        prompts.append(args[0])
        if len(calls) == 1:
            kwargs["route_selected"](
                "https://api.deepseek.com/anthropic",
                "deepseek-v4-flash",
                "backup-key",
            )
            return candidate(65)
        return candidate(40)

    with (
        patch.object(
            scriptwriter,
            "_resolve_provider",
            AsyncMock(return_value=("http://oneapi.example", "yinhe-thinking", "primary-key")),
        ),
        patch.object(scriptwriter, "_chat", side_effect=routed_chat),
    ):
        repaired, report = asyncio.run(
            scriptwriter.fit_daily_script_duration(
                candidate(70),
                dossier,
                edition,
                target_duration_minutes=1,
                language="en",
                closing_remarks=closing,
                ai_endpoint=None,
                ai_model=None,
                provider_id=None,
            )
        )

    assert report["passed"] is True
    assert calls == [
        ("http://oneapi.example", "yinhe-thinking"),
        ("https://api.deepseek.com/anthropic", "deepseek-v4-flash"),
    ]
    assert all("no more than" in prompt for prompt in prompts)
    assert len(repaired.splitlines()) == 4


def test_duration_repair_retries_a_semantically_invalid_oneapi_response():
    edition = date(2026, 8, 26)
    opening = morning_opening(edition, "en")
    closing = "Thanks for listening."
    articles = [
        research.NewsArticle(
            id=str(index),
            source_id=f"source-{index}",
            source_name=f"Source {index}",
            language="en",
            title=f"Story {index}",
            url=f"https://example.com/{index}",
            published_at="2026-08-25T00:00:00+00:00",
            summary=f"Evidence for story {index}.",
            evidence_text=f"Evidence for story {index}.",
        )
        for index in (1, 2)
    ]
    dossier = research.ResearchDossier(
        "2026-08-26", "now", 36, articles, articles, [],
    )
    original = "\n".join([opening, "Too short.", "Still too short.", closing])
    invalid = "\n".join([opening, "中文响应。", "Source 2 reports evidence.", closing])
    valid = "internal reasoning that must not become narration</think>" + "\n".join(
        [
            opening,
            " ".join(["evidence"] * 155) + ".",
            " ".join(["detail"] * 155) + ".",
            closing,
        ]
    )
    messages: list[str] = []

    with (
        patch.object(
            scriptwriter,
            "_resolve_provider",
            AsyncMock(return_value=("https://example.com", "yinhe-chat", "key")),
        ),
        patch.object(
            scriptwriter,
            "_chat",
            AsyncMock(side_effect=[invalid, valid]),
        ) as chat,
    ):
        repaired, report = asyncio.run(
            scriptwriter.fit_daily_script_duration(
                original,
                dossier,
                edition,
                target_duration_minutes=3,
                language="en",
                closing_remarks=closing,
                ai_endpoint=None,
                ai_model=None,
                provider_id=None,
                log=messages.append,
            )
        )

    assert chat.await_count == 2
    assert report["passed"] is True
    assert "中文" not in repaired
    assert "previous response was rejected by software" in chat.await_args_list[1].args[0]
    assert any("rejected semantic response 1/3" in message for message in messages)


def test_fixed_morning_opening_and_contract_are_software_owned():
    edition = date(2026, 8, 18)
    opening = morning_opening(edition, "en")
    closing = "Thanks for listening."
    final = enforce_script_contract(
        "Techmeme reported the launch.\nIEEE Spectrum explained the engineering tradeoff.",
        opening=opening,
        closing=closing,
        language="en",
    )

    assert final.startswith("Good morning. It's Tuesday, August 18, 2026")
    assert final.endswith(closing)
    assert final.count("Good morning") == 1
    assert "a concentrated shot of the frontier-tech signals" in opening
    assert SOURCE_SPOKEN_ALIASES["量子位 QbitAI"] == ("QbitAI",)


def test_script_generation_retries_missing_paragraph_on_successful_backup_route():
    edition = date(2026, 8, 28)
    closing = "Thanks for listening."
    articles = [
        research.NewsArticle(
            id=str(index),
            source_id=f"source-{index}",
            source_name=f"Source {index}",
            language="en",
            title=f"Story {index}",
            url=f"https://example.com/{index}",
            published_at="2026-08-28T00:00:00+00:00",
            summary=f"Evidence for story {index}.",
            evidence_text=f"Evidence for story {index}.",
        )
        for index in (1, 2)
    ]
    dossier = research.ResearchDossier(
        "2026-08-28", "now", 36, articles, articles, [],
    )
    invalid = "Source 1 reports evidence.Source 2 reports separate evidence."
    valid = "Source 1 reports evidence.\nSource 2 reports separate evidence."
    calls: list[tuple[str, str, bool]] = []
    prompts: list[str] = []

    async def routed_chat(*args, **kwargs):
        calls.append((args[2], args[3], kwargs["disable_thinking"]))
        prompts.append(args[0])
        if len(calls) == 1:
            kwargs["route_selected"](
                "https://api.deepseek.com/anthropic",
                "deepseek-v4-flash",
                "backup-key",
            )
            return invalid
        return valid

    with (
        patch.object(
            scriptwriter,
            "_resolve_provider",
            AsyncMock(return_value=("http://oneapi.example", "yinhe-thinking", "primary-key")),
        ),
        patch.object(scriptwriter, "_chat", side_effect=routed_chat),
    ):
        generated = asyncio.run(
            scriptwriter.generate_daily_script(
                dossier,
                edition,
                target_duration_minutes=1,
                language="en",
                closing_remarks=closing,
                ai_endpoint=None,
                ai_model=None,
                provider_id=None,
            )
        )

    assert len(generated.splitlines()) == 4
    assert calls == [
        ("http://oneapi.example", "yinhe-thinking", False),
        ("https://api.deepseek.com/anthropic", "deepseek-v4-flash", True),
    ]
    assert "exactly 2 nonblank story paragraphs" in prompts[0]
    assert "previous response was rejected by software" in prompts[1]


def test_daily_duration_contract_rejects_the_observed_two_minute_eight_minute_mismatch():
    short_script = " ".join(["word"] * 390)
    on_target_script = " ".join(["word"] * 360)

    short = daily_script_duration_report(short_script, 8, "en")
    on_target = daily_script_duration_report(on_target_script, 3, "en")
    narration_short = narration_duration_report(125.760792, 8)

    assert short["passed"] is False
    assert short["estimated_duration_minutes"] == pytest.approx(3.25, abs=0.001)
    assert on_target["passed"] is True
    assert on_target["target_units"] == 360
    assert on_target["maximum_units"] == 432
    assert narration_short["passed"] is False
    assert narration_duration_report(180.0, 3)["passed"] is True


def test_daily_audio_gate_blocks_render_when_real_narration_misses_target():
    task = TaskResponse(
        id="daily-duration-mismatch",
        created_at="2026-08-24T00:00:00+00:00",
        updated_at="2026-08-24T00:00:00+00:00",
        source_type=SourceType.NEWS_DAILY,
        status=TaskStatus.TTS,
        config=TaskConfig(target_duration_minutes=8, auto_render=True),
    )
    compose = AsyncMock()

    with (
        patch.object(orchestrator, "_probe_duration", AsyncMock(return_value=125.760792)),
        patch.object(orchestrator, "run_compose", compose),
    ):
        with pytest.raises(RuntimeError, match="target 8 min, actual 2.10 min"):
            asyncio.run(
                orchestrator._after_audio(
                    task,
                    script_path="script.txt",
                    audio_path="audio.wav",
                    task_log=lambda _message: None,
                    log=None,
                )
            )

    compose.assert_not_awaited()


def _attribution_dossier() -> research.ResearchDossier:
    rows = [
        (
            "techmeme-axios",
            "techmeme",
            "Techmeme",
            'Data-center backlash grows in Texas (Axios)',
            'Axios : Texas officials criticized data-center expansion — details.',
        ),
        (
            "qbitai",
            "qbitai",
            "量子位 QbitAI",
            "A robot coffee shop operates at WRC",
            "The robots served customers in an open environment.",
        ),
        (
            "deeptech",
            "deeptech_china",
            "DeepTech 深科技",
            "Multimodal intelligence may accelerate",
            "A coding agent can run experiments and return results.",
        ),
        (
            "techmeme-bloomberg",
            "techmeme",
            "Techmeme",
            "A foldable iPhone is being tested (Mark Gurman/Bloomberg)",
            "Mark Gurman / Bloomberg : Sources describe the prototype — details.",
        ),
    ]
    articles = [
        research.NewsArticle(
            id=article_id,
            source_id=source_id,
            source_name=source_name,
            language="en",
            title=title,
            url=f"https://example.com/{article_id}",
            published_at="2026-08-24T00:00:00+00:00",
            summary=summary,
            evidence_text=summary,
        )
        for article_id, source_id, source_name, title, summary in rows
    ]
    return research.ResearchDossier(
        "2026-08-24",
        "now",
        36,
        articles,
        articles,
        [],
    )


def _attribution_script(*paragraphs: str) -> str:
    opening = morning_opening(date(2026, 8, 24), "en")
    closing = "Thanks for listening."
    filler = " ".join(["evidence"] * 125)
    stories = list(paragraphs)
    if stories:
        stories[-1] = f"{stories[-1]} {filler}"
    return "\n".join([opening, *stories, closing])


def test_contract_accepts_original_publications_carried_by_techmeme_credit():
    dossier = _attribution_dossier()
    script = _attribution_script(
        "Axios reports growing data-center opposition.",
        "The Chinese-language outlet QbitAI reports a robot coffee shop.",
        "The Chinese-language outlet DeepTech China reports progress in coding agents.",
        "According to Bloomberg, a foldable iPhone prototype is being tested.",
    )

    report = script_contract_report(
        script,
        dossier,
        date(2026, 8, 24),
        language="en",
        closing_remarks="Thanks for listening.",
    )

    assert report["passed"] is True
    assert report["matched_publication_count"] == 4
    assert report["required_publication_count"] == 3
    assert report["source_mentions"] == {
        "Axios": True,
        "量子位 QbitAI": True,
        "DeepTech 深科技": True,
        "Bloomberg": True,
    }


def test_contract_counts_one_aggregator_mention_only_once():
    dossier = _attribution_dossier()
    script = _attribution_script(
        "Techmeme reports both aggregated stories.",
        "QbitAI reports the robot coffee shop.",
        "The coding-agent story follows without a publication attribution.",
    )

    report = script_contract_report(
        script,
        dossier,
        date(2026, 8, 24),
        language="en",
        closing_remarks="Thanks for listening.",
    )

    assert report["passed"] is False
    assert report["matched_publication_count"] == 2
    assert "fewer than three selected publications are attributed aloud" in report["failures"]


@pytest.mark.parametrize("paragraph,accepted", [
    ("The Chinese-language science outlet DeepTech reports on microbial pesticides.", True),
    ("DeepTech reports on microbial pesticides.", False),
    ("The Chinese-language science outlet reports on microbial pesticides.", False),
    ("The Chinese-language science outlet DeepTechnology reports on pesticides.", False),
])
def test_contract_recognizes_deeptech_short_name_with_explicit_media_provenance(paragraph, accepted):
    script = _attribution_script(
        "Axios reports growing data-center opposition.",
        "The Chinese-language outlet QbitAI reports a robot coffee shop.",
        paragraph,
        "According to Bloomberg, a foldable iPhone prototype is being tested.",
    )
    report = script_contract_report(
        script, _attribution_dossier(), date(2026, 8, 24),
        language="en", closing_remarks="Thanks for listening.",
    )
    assert report["passed"] is accepted


def test_d_only_trim_preserves_every_selected_publication_attribution():
    dossier = _attribution_dossier()
    script = _attribution_script(
        "Communities are resisting data-center expansion. Axios reports the dispute.",
        "A robot coffee shop operated in an open crowd. QbitAI documented the trial.",
        "Coding agents can run experiments. DeepTech China reports the workflow.",
        "A foldable phone prototype is being tested. Bloomberg reports the details.",
    )
    issues = [
        {
            "claim": f"Web audit codes D for story {story_number}",
            "evidence_story_numbers": [story_number],
        }
        for story_number in range(1, 5)
    ]

    revised = _minimalize_unsupported_paragraphs(script, issues, dossier)
    revised_lines = revised.splitlines()
    revised_lines[-2] += " " + " ".join(["evidence"] * 125)
    revised = "\n".join(revised_lines)
    report = script_contract_report(
        revised,
        dossier,
        date(2026, 8, 24),
        language="en",
        closing_remarks="Thanks for listening.",
    )

    assert "According to Axios, Communities are resisting data-center expansion." in revised
    assert "According to the Chinese-language outlet QbitAI, A robot coffee shop operated in an open crowd." in revised
    assert "According to the Chinese-language outlet DeepTech China, Coding agents can run experiments." in revised
    assert "According to Bloomberg, A foldable phone prototype is being tested." in revised
    assert report["passed"] is True
    assert report["matched_publication_count"] == 4


def test_daily_scheduler_uses_timezone_and_runs_once_after_desk_time():
    settings = DailyAutomationSettings(enabled=True, generation_time="05:30", timezone="Asia/Singapore")
    before = datetime(2026, 8, 17, 20, 0, tzinfo=timezone.utc)  # 04:00 SGT
    after = datetime(2026, 8, 17, 22, 0, tzinfo=timezone.utc)  # 06:00 SGT

    assert next_run_at(settings, before) == datetime(2026, 8, 17, 21, 30, tzinfo=timezone.utc)
    assert _is_due(settings, after, {}) == (True, date(2026, 8, 18))
    assert _is_due(settings, after, {"last_run_date": "2026-08-18"})[0] is False


def test_daily_desk_defaults_to_unattended_next_run():
    settings = DailyAutomationSettings()

    assert settings.enabled is True
    assert settings.catch_up_after_restart is False
    assert "target_duration_minutes" not in settings.model_dump()
    assert settings.tts_model == "orpheus-en"
    assert settings.voice == "leah"
    assert settings.opening_template.count("{date}") == 1
    assert "ByteFront Espresso" in settings.opening_template
    assert settings.closing_remarks.startswith("That's today's ByteFront Espresso")
    assert settings.auto_publish is True
    assert settings.publish_targets == ["youtube", "x", "apple_podcast"]
    assert settings.publish_visibility == "public"
    assert "news_image_count" not in settings.model_dump()
    assert "collage_broll_count" not in settings.model_dump()
    assert "footage_clip_count" not in settings.model_dump()
    assert settings.background_music_provider == "local_library"
    assert settings.background_music_track_id == "morning-blueprint"
    assert settings.outro_style == "morning-brief"


def test_daily_desk_recipe_validates_tts_model_and_voice_before_saving():
    with pytest.raises(ValueError, match="Unknown TTS model"):
        DailyAutomationSettings(tts_model="missing-model")

    with pytest.raises(ValueError, match="unavailable"):
        DailyAutomationSettings(tts_model="orpheus-en", voice="Carter")

    settings = DailyAutomationSettings(tts_model="orpheus-en", voice="tara")
    assert settings.voice == "tara"

    with pytest.raises(ValueError, match=r"contain \{date\} exactly once"):
        DailyAutomationSettings(opening_template="Good morning from ByteFront Espresso.")


def test_daily_desk_recipe_discards_retired_visual_count_settings():
    settings = DailyAutomationSettings.model_validate(
        {
            "tts_model": "vibevoice-0.5b",
            "voice": "Carter",
            "collage_broll_count": 5,
        }
    )

    payload = settings.model_dump()
    assert "footage_clip_count" not in payload
    assert "collage_broll_count" not in payload
    assert "news_image_count" not in payload


def test_load_settings_migrates_a_retired_tts_recipe(tmp_path: Path):
    settings_path = tmp_path / "daily_automation.json"
    settings_path.write_text(
        json.dumps(
            {
                "target_duration_minutes": 12,
                "tts_model": "retired-model",
                "voice": "retired-voice",
            }
        ),
        encoding="utf-8",
    )

    with (
        patch.object(scheduler, "_settings_path", return_value=settings_path),
        patch.object(scheduler.config, "TTS_DEFAULT_MODEL", "vibevoice-0.5b"),
        patch.object(scheduler.config, "TTS_DEFAULT_VOICE_1", "Carter"),
    ):
        settings = scheduler.load_settings()

    assert "target_duration_minutes" not in settings.model_dump()
    assert settings.tts_model == "vibevoice-0.5b"
    assert settings.voice == "Carter"
    persisted = json.loads(settings_path.read_text(encoding="utf-8"))
    assert persisted["tts_model"] == settings.tts_model
    assert persisted["voice"] == settings.voice


def test_retired_manual_and_content_planning_apis_are_not_exposed():
    from backend.main import app

    paths = app.openapi()["paths"]
    assert not any(path.startswith("/api/content-planning") for path in paths)
    assert "/api/daily-news" in paths
    assert "get" in paths["/api/tasks"]
    assert "post" not in paths["/api/tasks"]


def test_daily_task_snapshots_the_visible_automation_recipe(tmp_path: Path):
    queued = TaskResponse(
        id="daily-queued-001",
        created_at="2026-08-19T00:00:00+00:00",
        updated_at="2026-08-19T00:00:00+00:00",
        source_type=SourceType.NEWS_DAILY,
        source_title="Frontier Tech Daily — 2026-08-19",
        status=TaskStatus.QUEUED,
        config=TaskConfig(),
        origin_type="daily_news",
        origin_id="2026-08-19",
    )
    create = AsyncMock(return_value=queued)
    settings = DailyAutomationSettings(
        target_duration_minutes=12,
        tts_model="orpheus-en",
        voice="tara",
        opening_template="It is {date}. Your ByteFront Espresso is ready.",
        closing_remarks="That is the signal. Meet me here tomorrow.",
        collage_broll_count=7,
        news_image_count=6,
        public_footage_enabled=True,
        footage_clip_count=11,
        background_music_provider="local",
        background_music_track_id="strategic-outlook",
        outro_style="data-extraction",
        auto_publish=False,
    )

    with (
        patch.object(scheduler.database, "create_task", create),
        patch.object(scheduler, "_state_path", return_value=tmp_path / "state.json"),
    ):
        asyncio.run(
            create_daily_task(
                settings,
                edition_date=date(2026, 8, 19),
                trigger="schedule",
            )
        )

    task_config = create.await_args.args[2]
    assert task_config.target_duration_minutes is None
    assert task_config.tts_model == "orpheus-en"
    assert task_config.voice_1 == "tara"
    assert task_config.opening_remarks == "It is Wednesday, August 19, 2026. Your ByteFront Espresso is ready."
    assert task_config.closing_remarks == "That is the signal. Meet me here tomorrow."
    assert task_config.collage_broll_count is None
    assert task_config.news_images_enabled is True
    assert task_config.news_image_count is None
    assert task_config.footage_enabled is True
    assert task_config.footage_provider == "hybrid"
    assert task_config.footage_clip_count is None
    assert task_config.opening_style == "editorial_motion"
    assert task_config.background_music_provider == "local"
    assert task_config.background_music_track_id == "strategic-outlook"
    assert task_config.outro_style == "data-extraction"
    assert task_config.program_music_pacing_enabled is True
    assert task_config.program_music_intro_seconds == 2.0
    assert task_config.program_music_opening_gap_seconds == 3.0
    assert task_config.program_music_story_gap_seconds == 1.5
    assert task_config.auto_publish is False


def test_test_run_does_not_consume_the_scheduled_daily_edition(tmp_path: Path):
    queued = TaskResponse(
        id="daily-test-queued",
        created_at="2026-08-19T00:00:00+00:00",
        updated_at="2026-08-19T00:00:00+00:00",
        source_type=SourceType.NEWS_DAILY,
        source_title="Frontier Tech Daily — 2026-08-19",
        status=TaskStatus.QUEUED,
        config=TaskConfig(),
        origin_type="daily_news",
        origin_id="2026-08-19",
    )
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "last_run_at": "2026-08-18T00:00:00+00:00",
                "last_run_date": "2026-08-18",
                "last_task_id": "yesterday",
            }
        ),
        encoding="utf-8",
    )

    create = AsyncMock(return_value=queued)
    with (
        patch.object(scheduler.database, "create_task", create),
        patch.object(scheduler, "_state_path", return_value=state_path),
    ):
        asyncio.run(
            create_daily_task(
                DailyAutomationSettings(auto_publish=False),
                edition_date=date(2026, 8, 19),
                trigger="manual",
                test_mode=True,
            )
        )

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["last_run_date"] == "2026-08-18"
    assert state["last_task_id"] == "yesterday"
    assert state["last_test_task_id"] == queued.id
    test_config = create.await_args.args[2]
    assert test_config.target_duration_minutes is None
    assert test_config.auto_publish is False
    assert test_config.publish_test_mode is True


def test_music_prompt_is_instrumental_and_mix_is_duration_locked(tmp_path: Path):
    prompt = build_music_prompt("Morning edition", {"talking_points": [{"category": "robotics", "headline": "A new actuator"}]})
    assert "No vocals" in prompt
    assert "narration" in prompt

    narration = tmp_path / "narration.wav"
    music = tmp_path / "music.wav"
    _tone(narration, 2.0, 220)
    _tone(music, 1.0, 440)
    output = asyncio.run(
        mix_narration_and_music(narration, music, tmp_path, bed_db=-25, duck_db=-11)
    )
    report = json.loads((tmp_path / "audio" / "music_mix_report.json").read_text())
    assert output.is_file()
    assert abs(report["program_mix_duration_seconds"] - report["narration_duration_seconds"]) < 0.25
    assert report["content_music_db"] == -25


def test_apple_podcast_feed_is_prepared_without_external_submission(tmp_path: Path):
    task_dir = tmp_path / "daily-test-001"
    task_dir.mkdir()
    audio = task_dir / "audio.wav"
    _tone(audio, 2.0, 220)
    task = _task(tmp_path, audio)

    with (
        patch.object(config, "OUTPUTS_DIR", tmp_path),
        patch(
            "backend.settings_store.restart_required_change_pending",
            return_value=False,
        ),
    ):
        result = prepare_apple_podcast(task)
        manifest = read_publication_manifest(task)

    feed = (tmp_path / "podcast" / "feed.xml").read_text(encoding="utf-8")
    assert result["status"] == "feed_ready"
    assert "podcasts_connect_account" in result["external_submission"]
    assert "<enclosure" in feed
    assert manifest["platforms"]["apple_podcast"]["identity"] == {
        "feed_title": "ByteFront Espresso",
        "author": "FrontierTechSN",
    }


def test_review_token_normalization_accepts_only_known_ui_wrappers():
    payload = review._batch_payload(
        "💬️ \ufeff\u200b```text\n1P 2D\n```\u200b",
        [1, 2],
    )

    assert payload["approved"] is False
    assert payload["issues"][0]["evidence_story_numbers"] == [2]
    with pytest.raises(ValueError, match="incomplete or malformed"):
        review._batch_payload("Result: 1P2P", [1, 2])
    with pytest.raises(ValueError, match="incomplete or malformed"):
        review._batch_payload("1P2P because both stories pass", [1, 2])


@pytest.mark.parametrize("partial", ["W", "W1P;2BF@2.3,2", "W1P;2", "W5B@5.1;6", "W5B@5."])
def test_partial_live_chatgpt_verdicts_never_become_fact_check_approval(partial):
    claims = {1: {"1.1": "First story"}, 2: {"2.3": "Name", "2.6": "Result"}}
    with pytest.raises(ValueError, match="incomplete or malformed"):
        review._batch_payload(partial, [1, 2], claim_catalog=claims, require_web=True)
    complete = review._batch_payload(
        "W1P;2BDF@2.3,2.6", [1, 2], claim_catalog=claims, require_web=True,
    )
    assert complete["approved"] is False
    assert complete["issues"][0]["claim_ids"] == ["2.3", "2.6"]


def test_review_recovery_requires_the_current_request_anchor():
    current = "a" * 32
    other = "b" * 32
    rows = [
        {"Role": "Assistant", "Text": "1D2D"},
        {"Role": "User", "Text": f"REVIEW_REQUEST_ID:{current}. prompt"},
        {"Role": "User", "Text": "fragment of the same Gemini prompt"},
        {"Role": "Assistant", "Text": "1P2P"},
    ]

    assert review._assistant_for_review_request(rows, current) == "1P2P"
    assert review._assistant_for_review_request(rows, other) == ""
    assert review._assistant_for_review_request(
        [
            {"Role": "User", "Text": f"REVIEW_REQUEST_ID:{current}. prompt"},
            {"Role": "User", "Text": f"REVIEW_REQUEST_ID:{other}. another prompt"},
            {"Role": "Assistant", "Text": "1P2P"},
        ],
        current,
    ) == ""


def test_story_review_uses_chatgpt_only_without_touching_gemini():
    async def command(args, **kwargs):
        if args[:2] == ["chatgpt", "model"]:
            return OpenCLIResult(
                tuple(args), 0, '[{"Status":"Success","Model":"Medium"}]', ""
            )
        if args[:2] == ["chatgpt", "ask"]:
            return OpenCLIResult(
                tuple(args),
                0,
                '[{"response":"💬 W1P","conversationUrl":"https://chatgpt.com/c/fact"}]',
                "",
            )
        raise AssertionError(args)

    command = AsyncMock(side_effect=command)
    with patch.object(review, "run_opencli", command):
        payload, _, url, provider = asyncio.run(
            review._web_story_review("audit\nmore", story_number=1, log=None)
        )

    assert payload["approved"] is True
    assert url.endswith("/fact")
    assert provider == "chatgpt"
    assert command.await_count == 2
    calls = [call.args[0] for call in command.await_args_list]
    assert not any(args[0] == "gemini" for args in calls)
    assert calls[0][:3] == ["chatgpt", "model", "medium"]
    args = calls[1]
    assert args[:2] == ["chatgpt", "ask"]
    assert args[args.index("--new") + 1] == "true"
    assert review._REVIEW_REQUEST_RE.search(args[2])
    assert "\n" not in args[2]
    assert command.await_args.kwargs["timeout"] == review.config.DAILY_NEWS_WEB_REVIEW_TIMEOUT + 20
    assert all(call.kwargs["site_session_namespace"].startswith(
        "frontiertechsn-review-"
    ) for call in command.await_args_list)


def test_chatgpt_late_recovery_accepts_only_its_owned_turn():
    owned_prompt = ""

    async def command(args, **kwargs):
        nonlocal owned_prompt
        if args[:2] == ["chatgpt", "model"]:
            return OpenCLIResult(
                tuple(args), 0, '[{"Status":"Success","Model":"Medium"}]', ""
            )
        if args[:2] == ["chatgpt", "ask"]:
            owned_prompt = args[2]
            return OpenCLIResult(
                tuple(args),
                0,
                '[{"response":"💬 [NO RESPONSE] No ChatGPT response within 90s."}]',
                "",
            )
        if args[:2] == ["chatgpt", "read"]:
            turns = [
                {"Index": 1, "Role": "Assistant", "Text": "What's the vibe, Leo?"},
                {"Index": 2, "Role": "User", "Text": owned_prompt},
                {"Index": 3, "Role": "User", "Text": "split evidence fragment"},
                {"Index": 4, "Role": "Assistant", "Text": "💬 `W1P2P`"},
            ]
            return OpenCLIResult(tuple(args), 0, json.dumps(turns), "")
        raise AssertionError(args)

    command_mock = AsyncMock(side_effect=command)
    with patch.object(review, "run_opencli", command_mock):
        payload, raw, _, provider = asyncio.run(
            review._web_story_review("audit", story_numbers=[1, 2], log=None)
        )

    assert payload["approved"] is True
    assert provider == "chatgpt"
    assert "[CHATGPT RECOVERY]" in raw
    assert not any(call.args[0][0] == "gemini" for call in command_mock.await_args_list)
    assert len(
        {
            call.kwargs["site_session_namespace"]
            for call in command_mock.await_args_list
        }
    ) == 1


def test_unowned_chatgpt_greeting_is_rejected_before_fresh_retry():
    chatgpt_asks = 0

    async def command(args, **kwargs):
        nonlocal chatgpt_asks
        if args[:2] == ["chatgpt", "model"]:
            return OpenCLIResult(
                tuple(args), 0, '[{"Status":"Success","Model":"Medium"}]', ""
            )
        if args[:2] == ["chatgpt", "ask"]:
            chatgpt_asks += 1
            response = "[NO RESPONSE]" if chatgpt_asks == 1 else "W1P2P"
            return OpenCLIResult(
                tuple(args),
                0,
                json.dumps([{"response": response}]),
                "",
            )
        if args[:2] == ["chatgpt", "read"]:
            unrelated = [
                {"Index": 1, "Role": "Assistant", "Text": "What's the vibe, Leo?"},
                {"Index": 2, "Role": "User", "Text": "Analyze a Kuala Lumpur B-roll video"},
            ]
            return OpenCLIResult(tuple(args), 0, json.dumps(unrelated), "")
        raise AssertionError(args)

    command_mock = AsyncMock(side_effect=command)
    with patch.object(review, "run_opencli", command_mock):
        payload, raw, _, provider = asyncio.run(
            review._web_story_review("audit", story_numbers=[1, 2], log=None)
        )

    assert payload["approved"] is True
    assert provider == "chatgpt"
    assert chatgpt_asks == 2
    assert "What's the vibe" not in raw
    chatgpt_calls = [
        call.args[0]
        for call in command_mock.await_args_list
        if call.args[0][:2] == ["chatgpt", "ask"]
    ]
    assert all(args[args.index("--new") + 1] == "true" for args in chatgpt_calls)
    assert not any(call.args[0][0] == "gemini" for call in command_mock.await_args_list)


def test_chatgpt_late_recovery_polls_before_resubmitting_the_owned_request():
    owned_prompt = ""
    chatgpt_asks = 0
    chatgpt_reads = 0

    async def command(args, **kwargs):
        nonlocal owned_prompt, chatgpt_asks, chatgpt_reads
        if args[:2] == ["chatgpt", "model"]:
            return OpenCLIResult(
                tuple(args), 0, '[{"Status":"Success","Model":"Medium"}]', ""
            )
        if args[:2] == ["chatgpt", "ask"]:
            chatgpt_asks += 1
            owned_prompt = args[2]
            return OpenCLIResult(
                tuple(args),
                0,
                '[{"response":"[NO RESPONSE]"}]',
                "",
            )
        if args[:2] == ["chatgpt", "read"]:
            chatgpt_reads += 1
            turns = [{"Role": "User", "Text": owned_prompt}]
            if chatgpt_reads == 3:
                turns.append({"Role": "Assistant", "Text": "W1P2P"})
            return OpenCLIResult(tuple(args), 0, json.dumps(turns), "")
        raise AssertionError(args)

    with patch.object(review, "run_opencli", AsyncMock(side_effect=command)):
        payload, _, _, provider = asyncio.run(
            review._web_story_review("audit", story_numbers=[1, 2], log=None)
        )

    assert payload["approved"] is True
    assert provider == "chatgpt"
    assert chatgpt_asks == 1
    assert chatgpt_reads == 3


def test_story_review_retries_fresh_chatgpt_without_gemini_fallback():
    chatgpt_asks = 0

    async def command(args, **kwargs):
        nonlocal chatgpt_asks
        if args[:2] == ["chatgpt", "model"]:
            return OpenCLIResult(
                tuple(args),
                0,
                '[{"Status":"Success","Model":"Medium"}]',
                "",
            )
        if args[:2] == ["chatgpt", "ask"]:
            chatgpt_asks += 1
            response = "[NO RESPONSE]" if chatgpt_asks == 1 else "W1P2P"
            return OpenCLIResult(
                tuple(args),
                0,
                json.dumps([{"response": response, "conversationUrl": ""}]),
                "",
            )
        if args[:2] == ["chatgpt", "read"]:
            return OpenCLIResult(tuple(args), 0, "[]", "")
        raise AssertionError(args)

    command_mock = AsyncMock(side_effect=command)
    with patch.object(review, "run_opencli", command_mock):
        payload, _, url, provider = asyncio.run(
            review._web_story_review("audit", story_numbers=[1, 2], log=None)
        )

    assert payload["approved"] is True
    assert provider == "chatgpt"
    assert url == ""
    assert chatgpt_asks == 2
    calls = [call.args[0] for call in command_mock.await_args_list]
    assert not any(args[0] == "gemini" for args in calls)
    chatgpt_ask = next(args for args in calls if args[:2] == ["chatgpt", "ask"])
    assert chatgpt_ask[chatgpt_ask.index("--new") + 1] == "true"
    assert chatgpt_ask[chatgpt_ask.index("--window") + 1] == "foreground"


def test_chatgpt_fact_check_retries_transient_model_picker_failure():
    model_attempts = 0
    messages: list[str] = []

    async def command(args, **kwargs):
        nonlocal model_attempts
        if args[:2] == ["chatgpt", "model"]:
            model_attempts += 1
            if model_attempts == 1:
                raise OpenCLIError("ChatGPT model picker was still hydrating")
            return OpenCLIResult(
                tuple(args),
                0,
                '[{"Status":"Success","Model":"Medium"}]',
                "",
            )
        if args[:2] == ["chatgpt", "ask"]:
            return OpenCLIResult(
                tuple(args),
                0,
                '[{"response":"W1P2P","conversationUrl":"https://chatgpt.com/c/retry"}]',
                "",
            )
        raise AssertionError(args)

    with (
        patch.object(review, "run_opencli", AsyncMock(side_effect=command)),
        patch.object(review.asyncio, "sleep", AsyncMock()) as sleep,
    ):
        payload, raw, url, provider = asyncio.run(
            review._web_story_review(
                "audit",
                story_numbers=[1, 2],
                log=messages.append,
            )
        )

    assert payload["approved"] is True
    assert provider == "chatgpt"
    assert url.endswith("/retry")
    assert model_attempts == 2
    assert "[CHATGPT MODEL POLICY ERROR 1/3]" in raw
    assert any("model policy check attempt 1/3 failed" in message for message in messages)
    sleep.assert_any_await(review._MODEL_SELECTION_RETRY_DELAY_SECONDS)


def test_chatgpt_fact_check_uses_in_range_model_after_preferred_switch_fails():
    model_attempts = 0
    chatgpt_asks = 0
    messages: list[str] = []

    async def command(args, **kwargs):
        nonlocal model_attempts, chatgpt_asks
        if args[:2] == ["chatgpt", "model"]:
            model_attempts += 1
            return OpenCLIResult(
                tuple(args),
                0,
                '[{"Status":"Policy fallback","Model":"High"}]',
                "",
            )
        if args[:2] == ["chatgpt", "ask"]:
            chatgpt_asks += 1
            return OpenCLIResult(
                tuple(args),
                0,
                '[{"response":"W1P2P","conversationUrl":"https://chatgpt.com/c/current"}]',
                "",
            )
        raise AssertionError(args)

    with (
        patch.object(review, "run_opencli", AsyncMock(side_effect=command)),
        patch.object(review.asyncio, "sleep", AsyncMock()),
    ):
        payload, raw, url, provider = asyncio.run(
            review._web_story_review(
                "audit",
                story_numbers=[1, 2],
                log=messages.append,
            )
        )

    assert payload["approved"] is True
    assert provider == "chatgpt"
    assert url.endswith("/current")
    assert model_attempts == 1
    assert chatgpt_asks == 1
    assert "[CHATGPT MODEL POLICY]" in raw
    assert "Allowed: medium..xhigh" in raw
    assert "Status: Policy fallback" in raw
    assert "Observed: High" in raw
    assert any("Policy fallback at High" in message for message in messages)


@pytest.mark.parametrize("terminal_failure", [False, True])
def test_model_preflight_recovers_stale_page_then_timeout_in_fresh_sessions(terminal_failure):
    sessions = []
    asks = []

    async def command(args, **kwargs):
        namespace = kwargs["site_session_namespace"]
        if args[:2] == ["chatgpt", "model"]:
            assert args[args.index("--timeout") + 1] == "45"
            assert kwargs["timeout"] == 75
            sessions.append(namespace)
            if len(sessions) == 1:
                raise OpenCLIError("Page not found: deadbeef — stale page identity")
            if len(sessions) == 2:
                raise OpenCLIError("OpenCLI command timed out after 75s")
            # Same namespace would hit the daemon lease from attempt two.
            assert namespace not in sessions[:-1]
            if terminal_failure:
                raise OpenCLIError("SESSION_BUSY")
            return OpenCLIResult(tuple(args), 0, '[{"Model":"Medium"}]', "")
        if args[:2] == ["chatgpt", "ask"]:
            asks.append(namespace)
            return OpenCLIResult(tuple(args), 0, '[{"response":"W1P2P"}]', "")
        raise AssertionError(args)

    with (
        patch.object(review, "run_opencli", AsyncMock(side_effect=command)),
        patch.object(review.asyncio, "sleep", AsyncMock()),
        patch.object(review, "close_opencli_site_sessions", AsyncMock()) as close,
    ):
        coroutine = review._web_story_review(
            "audit", story_numbers=[1, 2], log=None,
            site_session_namespace="frontiertechsn-review-original",
        )
        if terminal_failure:
            with pytest.raises(RuntimeError, match="after 3 attempts"):
                asyncio.run(coroutine)
            assert asks == []
        else:
            payload, _, _, _ = asyncio.run(coroutine)
            assert payload["approved"] is True
            assert asks == [sessions[-1]]
    assert len(set(sessions)) == 3
    assert [call.args[0] for call in close.await_args_list] == sessions
    assert all(call.kwargs["sites"] == ("chatgpt",) for call in close.await_args_list)


@pytest.mark.parametrize("outside_level", ["Instant", "Pro"])
def test_chatgpt_fact_check_retries_out_of_range_model_without_submitting(
    outside_level: str,
):
    model_attempts = 0
    chatgpt_asks = 0
    messages: list[str] = []

    async def command(args, **kwargs):
        nonlocal model_attempts, chatgpt_asks
        if args[:2] == ["chatgpt", "model"]:
            model_attempts += 1
            return OpenCLIResult(
                tuple(args),
                0,
                json.dumps(
                    [{"Status": "Policy fallback", "Model": outside_level}]
                ),
                "",
            )
        if args[:2] == ["chatgpt", "ask"]:
            chatgpt_asks += 1
        raise AssertionError(args)

    with (
        patch.object(review, "run_opencli", AsyncMock(side_effect=command)),
        patch.object(review.asyncio, "sleep", AsyncMock()) as sleep,
        pytest.raises(RuntimeError, match="allowed medium..xhigh range after 3 attempts"),
    ):
        asyncio.run(
            review._web_story_review(
                "audit",
                story_numbers=[1, 2],
                log=messages.append,
            )
        )

    assert model_attempts == 3
    assert chatgpt_asks == 0
    assert sleep.await_count == 2
    assert any("model policy check attempt 2/3 failed" in message for message in messages)


def test_chatgpt_fact_check_recovers_target_conversation_after_route_drift():
    chatgpt_prompt = ""
    target_url = "https://chatgpt.com/c/6a868e01-3b9c-83ec-b973-e3aee234afe4"

    async def command(args, **kwargs):
        nonlocal chatgpt_prompt
        if args[:2] == ["chatgpt", "model"]:
            return OpenCLIResult(
                tuple(args), 0, '[{"Status":"Success","Model":"Medium"}]', ""
            )
        if args[:2] == ["chatgpt", "ask"]:
            chatgpt_prompt = args[2]
            raise OpenCLIError(
                "ChatGPT navigated away from the target conversation "
                f"({target_url}); current URL is https://chatgpt.com/new"
            )
        if args[:2] == ["chatgpt", "detail"]:
            turns = [
                {"Index": 1, "Role": "User", "Text": chatgpt_prompt},
                {"Index": 2, "Role": "Assistant", "Text": "W1P2P"},
            ]
            return OpenCLIResult(tuple(args), 0, json.dumps(turns), "")
        raise AssertionError(args)

    command_mock = AsyncMock(side_effect=command)
    with patch.object(review, "run_opencli", command_mock):
        payload, raw, url, provider = asyncio.run(
            review._web_story_review("audit", story_numbers=[1, 2], log=None)
        )

    assert payload["approved"] is True
    assert provider == "chatgpt"
    assert url == target_url
    assert "[CHATGPT RECOVERY]" in raw
    detail = next(
        call
        for call in command_mock.await_args_list
        if call.args[0][:2] == ["chatgpt", "detail"]
    )
    assert detail.args[0][2] == target_url
    assert detail.args[0][detail.args[0].index("--timeout") + 1] == str(
        review.config.DAILY_NEWS_WEB_REVIEW_TIMEOUT
    )
    assert detail.kwargs["timeout"] == review.config.DAILY_NEWS_WEB_REVIEW_TIMEOUT + 15


def test_chatgpt_fact_check_retries_target_until_protocol_is_stable():
    chatgpt_prompt = ""
    detail_reads = 0
    target_url = "https://chatgpt.com/c/6a8efd33-7908-83ec-af9a-5f2f74befe0b"

    async def command(args, **kwargs):
        nonlocal chatgpt_prompt, detail_reads
        if args[:2] == ["chatgpt", "model"]:
            return OpenCLIResult(
                tuple(args), 0, '[{"Status":"Success","Model":"Medium"}]', ""
            )
        if args[:2] == ["chatgpt", "ask"]:
            chatgpt_prompt = args[2]
            response = [{
                "response": "W2P;6D@6.2[citation loading]",
                "conversationUrl": target_url,
            }]
            return OpenCLIResult(tuple(args), 0, json.dumps(response), "")
        if args[:2] == ["chatgpt", "detail"]:
            detail_reads += 1
            answer = (
                "W2P;6D@6.2[citation loading]"
                if detail_reads == 1
                else "W2P;6D@6.2"
            )
            turns = [
                {"Index": 1, "Role": "User", "Text": chatgpt_prompt},
                {"Index": 2, "Role": "Assistant", "Text": answer},
            ]
            return OpenCLIResult(tuple(args), 0, json.dumps(turns), "")
        raise AssertionError(args)

    command_mock = AsyncMock(side_effect=command)
    with (
        patch.object(review, "run_opencli", command_mock),
        patch.object(review.asyncio, "sleep", AsyncMock()) as sleep,
    ):
        payload, raw, url, provider = asyncio.run(
            review._web_story_review(
                "audit",
                story_numbers=[2, 6],
                claim_catalog={
                    2: {"2.1": "Story two claim."},
                    6: {"6.2": "Story six unsupported claim."},
                },
                log=None,
            )
        )

    assert payload["approved"] is False
    assert payload["issues"][0]["evidence_story_numbers"] == [6]
    assert payload["issues"][0]["claim_ids"] == ["6.2"]
    assert provider == "chatgpt"
    assert url == target_url
    assert detail_reads == 2
    assert "[CHATGPT RECOVERY]\nW2P;6D@6.2" in raw
    sleep.assert_any_await(review._RECOVERY_POLL_INTERVAL_SECONDS)


def test_chatgpt_fact_check_waits_for_slow_target_conversation_convergence():
    chatgpt_prompt = ""
    detail_reads = 0
    target_url = "https://chatgpt.com/c/6a9a987d-9b10-83ec-a5fe-a4f892f2dddb"
    messages: list[str] = []

    async def command(args, **kwargs):
        nonlocal chatgpt_prompt, detail_reads
        if args[:2] == ["chatgpt", "model"]:
            return OpenCLIResult(
                tuple(args), 0, '[{"Status":"Success","Model":"Medium"}]', ""
            )
        if args[:2] == ["chatgpt", "ask"]:
            chatgpt_prompt = args[2]
            return OpenCLIResult(
                tuple(args),
                0,
                json.dumps([{
                    "response": "Searching the web before returning W1E@1.3;2P",
                    "conversationUrl": target_url,
                }]),
                "",
            )
        if args[:2] == ["chatgpt", "detail"]:
            detail_reads += 1
            answer = (
                "Checking sources and citations..."
                if detail_reads < 6
                else "W1E@1.3;2P"
            )
            turns = [
                {"Index": 1, "Role": "User", "Text": chatgpt_prompt},
                {"Index": 2, "Role": "Assistant", "Text": answer},
            ]
            return OpenCLIResult(tuple(args), 0, json.dumps(turns), "")
        raise AssertionError(args)

    command_mock = AsyncMock(side_effect=command)
    with (
        patch.object(review, "run_opencli", command_mock),
        patch.object(review.asyncio, "sleep", AsyncMock()) as sleep,
    ):
        payload, raw, url, provider = asyncio.run(
            review._web_story_review(
                "audit",
                story_numbers=[1, 2],
                claim_catalog={
                    1: {"1.3": "Story one unsupported claim."},
                    2: {"2.1": "Story two claim."},
                },
                log=messages.append,
            )
        )

    assert payload["approved"] is False
    assert payload["issues"][0]["claim_ids"] == ["1.3"]
    assert provider == "chatgpt"
    assert url == target_url
    assert detail_reads == 6
    assert len([
        mock_call
        for mock_call in command_mock.await_args_list
        if mock_call.args[0][:2] == ["chatgpt", "ask"]
    ]) == 1
    recovery_attempts = len(review._target_recovery_poll_delays()) + 1
    assert f"[CHATGPT RECOVERY INVALID 5/{recovery_attempts}]" in raw
    assert "[CHATGPT RECOVERY]\nW1E@1.3;2P" in raw
    assert [mock_call.args[0] for mock_call in sleep.await_args_list] == list(
        review._target_recovery_poll_delays()[:5]
    )
    assert sum(review._target_recovery_poll_delays()) == 180
    assert any("rereading its target conversation" in message for message in messages)


def test_chatgpt_fact_check_retries_submission_without_using_gemini():
    chatgpt_asks = 0

    async def command(args, **kwargs):
        nonlocal chatgpt_asks
        if args[:2] == ["chatgpt", "model"]:
            return OpenCLIResult(
                tuple(args), 0, '[{"Status":"Success","Model":"Medium"}]', ""
            )
        if args[:2] == ["chatgpt", "ask"]:
            chatgpt_asks += 1
            if chatgpt_asks == 1:
                raise OpenCLIError("ChatGPT composer submission failed")
            return OpenCLIResult(tuple(args), 0, '[{"response":"W1P2P"}]', "")
        if args[:2] == ["chatgpt", "read"]:
            return OpenCLIResult(tuple(args), 0, "[]", "")
        raise AssertionError(args)

    command_mock = AsyncMock(side_effect=command)
    with patch.object(review, "run_opencli", command_mock):
        payload, _, _, provider = asyncio.run(
            review._web_story_review("audit", story_numbers=[1, 2], log=None)
        )

    chatgpt_calls = [
        call.args[0]
        for call in command_mock.await_args_list
        if call.args[0][:2] == ["chatgpt", "ask"]
    ]
    assert payload["approved"] is True
    assert provider == "chatgpt"
    assert chatgpt_asks == 2
    assert all(args[args.index("--new") + 1] == "true" for args in chatgpt_calls)
    assert not any(call.args[0][0] == "gemini" for call in command_mock.await_args_list)


def test_review_report_records_chatgpt_only(tmp_path: Path):
    articles = [
        research.NewsArticle(
            id=str(index),
            source_id=f"source-{index}",
            source_name=f"Source {index}",
            language="en",
            title=f"Story {index}",
            url=f"https://example.com/{index}",
            published_at="2026-08-20T00:00:00+00:00",
            summary=f"Evidence for story {index}.",
            evidence_text=f"Evidence for story {index}.",
        )
        for index in (1, 2)
    ]
    dossier = research.ResearchDossier(
        "2026-08-20",
        "now",
        36,
        articles,
        articles,
        [],
    )
    payload = {
        "approved": True,
        "confidence": 100,
        "summary": "2/2 story audits passed.",
        "issues": [],
        "corrected_script": "",
    }
    web_review = AsyncMock(return_value=(payload, "[CHATGPT 1]\nW1P;2P", "", "chatgpt"))
    contract = {"passed": True, "failures": []}

    with (
        patch.object(review, "_web_story_review", web_review),
        patch.object(review, "script_contract_report", return_value=contract),
        patch.object(
            review, "close_opencli_site_sessions", AsyncMock()
        ) as close_sessions,
    ):
        result = asyncio.run(
            review.review_daily_script(
                "Opening.\nStory one.\nStory two.\nClosing.",
                dossier,
                date(2026, 8, 20),
                tmp_path,
                language="en",
                closing_remarks="Closing.",
                ai_endpoint=None,
                ai_model=None,
                provider_id=None,
            )
        )

    assert result.report["fallback_used"] is False
    assert result.report["reviewer"] == (
        "ChatGPT Web (current model constrained to medium..xhigh) "
        "via project-local OpenCLI"
    )
    assert result.report["attempts"][0]["providers"] == ["chatgpt"]
    assert web_review.await_args.kwargs["site_session_namespace"].startswith(
        "frontiertechsn-review-"
    )
    close_sessions.assert_awaited_once_with(
        web_review.await_args.kwargs["site_session_namespace"]
    )
    assert (tmp_path / "review" / "story-review-prompt-1-group-1.txt").exists()
    assert (tmp_path / "review" / "story-review-response-1-group-1.txt").exists()


def test_review_cleanup_failure_does_not_mask_the_review_failure(tmp_path: Path):
    review_failure = RuntimeError("review failed")
    messages: list[str] = []

    with (
        patch.object(
            review,
            "_review_daily_script",
            AsyncMock(side_effect=review_failure),
        ),
        patch.object(
            review,
            "close_opencli_site_sessions",
            AsyncMock(side_effect=OpenCLIError("bridge unavailable")),
        ) as close_sessions,
    ):
        with pytest.raises(RuntimeError, match="review failed"):
            asyncio.run(
                review.review_daily_script(
                    "Opening.\nClosing.",
                    object(),
                    date(2026, 8, 26),
                    tmp_path,
                    language="en",
                    closing_remarks="Closing.",
                    ai_endpoint=None,
                    ai_model=None,
                    provider_id=None,
                    log=messages.append,
                )
            )

    close_sessions.assert_awaited_once()
    assert any("browser-session cleanup failed" in message for message in messages)


def test_review_fits_script_to_target_before_web_accuracy_audit(tmp_path: Path):
    articles = [
        research.NewsArticle(
            id=str(index),
            source_id=f"source-{index}",
            source_name=f"Source {index}",
            language="en",
            title=f"Story {index}",
            url=f"https://example.com/{index}",
            published_at="2026-08-24T00:00:00+00:00",
            summary=f"Evidence for story {index}.",
            evidence_text=f"Evidence for story {index}.",
        )
        for index in (1, 2)
    ]
    dossier = research.ResearchDossier(
        "2026-08-24",
        "now",
        36,
        articles,
        articles,
        [],
    )
    repaired = "Opening.\n" + " ".join(["word"] * 360) + "\nClosing."
    duration_contract = daily_script_duration_report(repaired, 3, "en")
    fit = AsyncMock(return_value=(repaired, duration_contract))
    web_review = AsyncMock(
        return_value=(
            {
                "approved": True,
                "confidence": 100,
                "summary": "2/2 story audits passed.",
                "issues": [],
            },
            "[RAW]",
            "",
            "gemini",
        )
    )

    with (
        patch.object(review, "fit_daily_script_duration", fit),
        patch.object(
            review,
            "script_contract_report",
            return_value={"passed": True, "failures": []},
        ),
        patch.object(review, "_web_story_review", web_review),
        patch.object(review, "close_opencli_site_sessions", AsyncMock()),
    ):
        result = asyncio.run(
            review.review_daily_script(
                "Opening.\nToo short.\nClosing.",
                dossier,
                date(2026, 8, 24),
                tmp_path,
                language="en",
                closing_remarks="Closing.",
                ai_endpoint=None,
                ai_model=None,
                provider_id=None,
                target_duration_minutes=3,
            )
        )

    assert result.script == repaired
    assert result.report["attempts"][0]["duration_contract"]["passed"] is True
    assert fit.await_args.kwargs["target_duration_minutes"] == 3
    assert (tmp_path / "review" / "candidate-duration-cycle-1.txt").exists()


def test_review_protects_d_corrected_story_during_next_duration_fit(tmp_path: Path):
    articles = [
        research.NewsArticle(
            id=str(index),
            source_id=f"source-{index}",
            source_name=f"Source {index}",
            language="en",
            title=f"Story {index}",
            url=f"https://example.com/{index}",
            published_at="2026-08-25T00:00:00+00:00",
            summary=f"Evidence for story {index}.",
            evidence_text=f"Evidence for story {index}.",
        )
        for index in (1, 2)
    ]
    dossier = research.ResearchDossier(
        "2026-08-26", "now", 36, articles, articles, [],
    )
    candidate = "Opening.\nStory one.\nStory two.\nClosing."

    async def fit(script, *_args, **_kwargs):
        return script, {"passed": True}

    web_review = AsyncMock(side_effect=[
        (
            {
                "approved": False,
                "issues": [{
                    "severity": "blocking",
                    "claim": "Web audit codes D for story 2 at claims 2.1",
                    "evidence_story_numbers": [2],
                    "claim_ids": ["2.1"],
                    "claim_texts": ["Story two."],
                }],
            },
            "W1P2D@2.1",
            "",
            "gemini",
        ),
        (
            {"approved": True, "issues": []},
            "W2P",
            "",
            "gemini",
        ),
        (
            {"approved": True, "issues": []},
            "W1P2P",
            "",
            "gemini",
        ),
    ])
    duration_fit = AsyncMock(side_effect=fit)

    with (
        patch.object(review, "fit_daily_script_duration", duration_fit),
        patch.object(
            review,
            "script_contract_report",
            return_value={"passed": True, "failures": []},
        ),
        patch.object(review, "_web_story_review", web_review),
        patch.object(review, "revise_daily_script", AsyncMock(return_value=candidate)),
        patch.object(review, "close_opencli_site_sessions", AsyncMock()),
    ):
        result = asyncio.run(
            review.review_daily_script(
                candidate,
                dossier,
                date(2026, 8, 26),
                tmp_path,
                language="en",
                closing_remarks="Closing.",
                ai_endpoint=None,
                ai_model=None,
                provider_id=None,
                target_duration_minutes=3,
            )
        )

    assert result.report["passed"] is True
    assert duration_fit.await_args_list[0].kwargs["protected_story_numbers"] == set()
    assert duration_fit.await_args_list[1].kwargs["protected_story_numbers"] == {2}
    assert duration_fit.await_args_list[2].kwargs["protected_story_numbers"] == {2}
    assert web_review.await_args_list[1].kwargs["story_numbers"] == [2]
    assert web_review.await_args_list[2].kwargs["story_numbers"] == [1, 2]


def test_review_stops_when_the_same_claim_failure_repeats_after_correction(tmp_path: Path):
    article = research.NewsArticle(
        id="robot",
        source_id="qbitai",
        source_name="量子位 QbitAI",
        language="en",
        title="Robot demo",
        url="https://example.com/robot",
        published_at="2026-08-26T00:00:00+00:00",
        summary="A robot demo.",
        evidence_text="A robot demo.",
    )
    dossier = research.ResearchDossier(
        "2026-08-26", "now", 36, [article], [article], [],
    )
    candidate = "Opening.\nQbitAI reports a robot demo.\nClosing."
    issue = {
        "severity": "blocking",
        "claim": "Web audit codes D for story 1 at claims 1.1",
        "evidence_story_numbers": [1],
        "claim_ids": ["1.1"],
        "claim_texts": ["QbitAI reports a robot demo."],
    }
    web_review = AsyncMock(
        side_effect=[
            ({"approved": False, "issues": [issue]}, "W1D@1.1", "", "gemini"),
            ({"approved": False, "issues": [issue]}, "W1D@1.1", "", "gemini"),
        ]
    )
    revise = AsyncMock(return_value=candidate)

    with (
        patch.object(
            review,
            "script_contract_report",
            return_value={"passed": True, "failures": []},
        ),
        patch.object(review, "_web_story_review", web_review),
        patch.object(review, "revise_daily_script", revise),
        patch.object(
            review, "close_opencli_site_sessions", AsyncMock()
        ) as close_sessions,
    ):
        with pytest.raises(RuntimeError, match="same claim-level blocking verdict"):
            asyncio.run(
                review.review_daily_script(
                    candidate,
                    dossier,
                    date(2026, 8, 26),
                    tmp_path,
                    language="en",
                    closing_remarks="Closing.",
                    ai_endpoint=None,
                    ai_model=None,
                    provider_id=None,
                )
            )

    assert web_review.await_count == 2
    assert revise.await_count == 1
    close_sessions.assert_awaited_once()
    report = json.loads((tmp_path / "review" / "fact_check_report.json").read_text())
    assert report["manual_review_required"] is True
    assert report["correction_count"] == 1


@pytest.mark.parametrize("language,repetitions", [("en", 1), ("en", 20), ("zh", 1)])
def test_automatic_report_length_survives_generation_and_review(tmp_path, language, repetitions):
    edition = date(2026, 9, 7)
    evidence = (
        "Source 1 reports a new chip with published specifications."
        if language == "en" else "Source 1 报道了一款新芯片，并公布了技术规格。"
    )
    story = " ".join([evidence] * repetitions)
    article = research.NewsArticle(
        id="1", source_id="source-1", source_name="Source 1", language=language,
        title="New chip", url="https://example.com/chip",
        published_at="2026-09-07T00:00:00+00:00", summary=evidence, evidence_text=story,
    )
    dossier = research.ResearchDossier("2026-09-07", "now", 36, [article], [article], [])
    closing = "Thanks for listening." if language == "en" else "感谢收听。"
    chat = AsyncMock(return_value=story)
    fit = AsyncMock(side_effect=AssertionError("Automatic editions must not fit a duration"))
    web_review = AsyncMock(return_value=(
        {"approved": True, "confidence": 100, "summary": "Verified.", "issues": []},
        "[RAW]", "", "gemini",
    ))

    async def run():
        script = await scriptwriter.generate_daily_script(
            dossier, edition, target_duration_minutes=None, language=language,
            closing_remarks=closing, ai_endpoint=None, ai_model=None, provider_id=None,
        )
        result = await review.review_daily_script(
            script, dossier, edition, tmp_path, target_duration_minutes=None,
            language=language, closing_remarks=closing, ai_endpoint=None,
            ai_model=None, provider_id=None,
        )
        assert result.script == script
        assert result.report["attempts"][0]["duration_contract"] is None
        assert story in result.script

    with (
        patch.object(scriptwriter, "_resolve_provider", AsyncMock(return_value=("https://example.com", "model", "key"))),
        patch.object(scriptwriter, "_chat", chat),
        patch.object(review, "fit_daily_script_duration", fit),
        patch.object(review, "_web_story_review", web_review),
        patch.object(review, "close_opencli_site_sessions", AsyncMock()),
    ):
        asyncio.run(run())
    fit.assert_not_awaited()
    web_review.assert_awaited_once()
    assert "There is no target runtime or total word count" in chat.await_args.args[0]


def test_automatic_audio_continues_to_composition_without_duration_gate():
    task = TaskResponse(
        id="daily-automatic", created_at="now", updated_at="now",
        source_type=SourceType.NEWS_DAILY, status=TaskStatus.TTS,
        config=TaskConfig(target_duration_minutes=None, auto_render=True),
    )
    compose = AsyncMock()
    with (
        patch.object(orchestrator, "narration_duration_report", side_effect=AssertionError("No fixed duration gate")),
        patch.object(orchestrator, "run_compose", compose),
    ):
        asyncio.run(orchestrator._after_audio(
            task, script_path="script.txt", audio_path="audio.wav",
            task_log=lambda _: None, log=None,
        ))
    compose.assert_awaited_once_with(task, log=None)
    assert task.audio_path == "audio.wav"


def test_saved_duration_is_retired_without_changing_other_recipe_fields(tmp_path):
    settings_path = tmp_path / "daily_automation.json"
    original = DailyAutomationSettings(auto_publish=False, max_stories=4).model_dump(mode="json")
    settings_path.write_text(json.dumps({**original, "target_duration_minutes": 3}))
    with patch.object(scheduler, "_settings_path", return_value=settings_path):
        assert scheduler.load_settings().model_dump(mode="json") == original
    assert json.loads(settings_path.read_text()) == original


@pytest.mark.parametrize("test_mode", [False, True])
def test_run_now_api_ignores_legacy_duration_and_queues_automatic_length(tmp_path, test_mode):
    from fastapi.testclient import TestClient
    from backend.main import app

    settings_path = tmp_path / "recipe.json"
    settings_path.write_text(json.dumps({
        **DailyAutomationSettings(auto_publish=False).model_dump(mode="json"),
        "target_duration_minutes": 3,
    }))

    async def create(source_type, payload, task_config, **kwargs):
        return TaskResponse(
            id="automatic-api-test", created_at="now", updated_at="now",
            source_type=source_type, source_url=payload,
            status=TaskStatus.QUEUED, config=task_config,
        )

    with (
        patch.object(scheduler, "_settings_path", return_value=settings_path),
        patch.object(scheduler, "_state_path", return_value=tmp_path / "state.json"),
        patch.object(scheduler.database, "create_task", AsyncMock(side_effect=create)),
    ):
        client = TestClient(app)
        response = client.post(
            "/api/daily-news/run-now",
            params={"test_mode": str(test_mode).lower(), "duration_minutes": 1},
        )
        assert response.status_code == 200
        assert response.json()["config"]["target_duration_minutes"] is None
        assert response.json()["config"]["publish_test_mode"] is test_mode
        assert "target_duration_minutes" not in client.get("/api/daily-news").json()["settings"]


@pytest.mark.parametrize('generation', [False, True, None])
def test_chatgpt_recovers_frozen_partial_only_after_settled_owned_reads(generation):
    prompt = ''
    commands = []
    reads = 0
    target = 'https://chatgpt.com/c/frozen-owned-turn'

    async def command(args, **kwargs):
        nonlocal prompt, reads
        commands.append(args)
        if args[:2] == ['chatgpt', 'model']:
            rows = [{'Model': 'Medium', 'Status': 'Success'}]
        elif args[:2] == ['chatgpt', 'ask']:
            prompt = args[2]
            rows = [{'response': 'W5B@5.', 'conversationUrl': target}]
        else:
            assert args[:3] == ['chatgpt', 'detail', target]
            reads += 1
            refreshing = '--refresh' in args
            if generation is False and reads == 3:
                assert refreshing
            else:
                assert not refreshing
            rows = [
                {'Role': 'User', 'Text': prompt},
                {'Role': 'Assistant', 'Text': 'W5B@5.1;6P' if reads >= 3 or generation is True else 'W5B@5.'},
            ]
            for row in rows:
                if generation is not None:
                    row['Generating'] = generation if reads < 3 else False
        return OpenCLIResult(tuple(args), 0, json.dumps(rows), '')

    with patch.object(review, 'run_opencli', AsyncMock(side_effect=command)), patch.object(review.asyncio, 'sleep', AsyncMock()):
        payload, raw, _, _ = asyncio.run(review._web_story_review(
            'audit', story_numbers=[5, 6], claim_catalog={5: {'5.1': 'Claim'}, 6: {'6.1': 'Claim'}}, log=None,
        ))
    assert payload['approved'] is False
    assert payload['issues'][0]['claim_ids'] == ['5.1']
    assert reads == 3
    assert len([args for args in commands if args[:2] == ['chatgpt', 'ask']]) == 1
    assert ('[CHATGPT SETTLED TARGET REFRESH]' in raw) == (generation is False)


@pytest.mark.parametrize('owned', [True, False])
def test_chatgpt_refresh_is_bounded_and_cannot_approve_invalid_or_unowned_verdicts(owned):
    prompt = ''
    refreshes = 0

    async def command(args, **kwargs):
        nonlocal prompt, refreshes
        if args[:2] == ['chatgpt', 'model']:
            rows = [{'Model': 'Medium'}]
        elif args[:2] == ['chatgpt', 'ask']:
            prompt = args[2]
            rows = [{'response': 'W5B@5.', 'conversationUrl': 'https://chatgpt.com/c/owned'}]
        else:
            refreshes += '--refresh' in args
            rows = [
                {'Role': 'User', 'Text': prompt if owned else 'REVIEW_REQUEST_ID:' + '0' * 32, 'Generating': False},
                {'Role': 'Assistant', 'Text': 'W5B@5.99;6P' if owned else 'W5P;6P', 'Generating': False},
            ]
        return OpenCLIResult(tuple(args), 0, json.dumps(rows), '')

    with (
        patch.object(review, 'run_opencli', AsyncMock(side_effect=command)),
        patch.object(review, '_target_recovery_poll_delays', return_value=(0, 0, 0, 0)),
        patch.object(review.asyncio, 'sleep', AsyncMock()),
        pytest.raises(review._WebStoryReviewError),
    ):
        asyncio.run(review._web_story_review(
            'audit', story_numbers=[5, 6],
            claim_catalog={5: {'5.1': 'Claim'}, 6: {'6.1': 'Claim'}}, log=None,
        ))
    assert refreshes == (2 if owned else 0)  # At most once per submission.


def test_rate_limit_recovers_the_owned_turn_without_resubmitting():
    prompt = ''
    calls = []
    target = 'https://chatgpt.com/c/rate-limited-owned'

    async def command(args, **kwargs):
        nonlocal prompt
        calls.append(args)
        if args[:2] == ['chatgpt', 'model']:
            rows = [{'Model': 'Medium'}]
        elif args[:2] == ['chatgpt', 'ask']:
            prompt = args[2]
            raise review.OpenCLIRateLimitError('CHATGPT_RATE_LIMITED Target: ' + target)
        else:
            assert args[:3] == ['chatgpt', 'detail', target]
            assert '--refresh' not in args
            rows = [{'Role': 'User', 'Text': prompt}, {'Role': 'Assistant', 'Text': 'W5P;6P', 'Generating': False}]
        return OpenCLIResult(tuple(args), 0, json.dumps(rows), '')

    logs = []
    with patch.object(review, 'run_opencli', AsyncMock(side_effect=command)):
        payload, _, url, _ = asyncio.run(review._web_story_review(
            'audit', story_numbers=[5, 6], log=logs.append,
        ))
    assert payload['approved'] is True
    assert url == target
    assert len([args for args in calls if args[:2] == ['chatgpt', 'ask']]) == 1
    assert any('shared cooldown' in line for line in logs)

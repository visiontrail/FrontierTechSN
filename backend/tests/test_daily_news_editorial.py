import asyncio
from datetime import date
from unittest.mock import AsyncMock, patch

import pytest

from backend.daily_news import research, review, scriptwriter
from backend.daily_news.editorial import DAILY_NEWS_EDITORIAL_RULES

EDITION = date(2026, 9, 7)
CLOSING = "Thanks for listening."


def dossier(*sources):
    articles = [
        research.NewsArticle(
            id=str(index), source_id=source_id, source_name=name, language=language,
            title="A company announces a trial", url=f"https://example.com/{index}",
            published_at="2026-09-07T00:00:00+00:00",
            summary="The company says the trial has started; commercial timing is not disclosed.",
            evidence_status="feed_summary",
        )
        for index, (source_id, name, language) in enumerate(sources)
    ]
    return research.ResearchDossier("2026-09-07", "now", 36, articles, articles, [])


def contract(data, *stories, language="en"):
    script = "\n".join([scriptwriter.morning_opening(EDITION, language), *stories, CLOSING])
    return scriptwriter.script_contract_report(
        script, data, EDITION, language=language, closing_remarks=CLOSING,
    )


@pytest.mark.parametrize("story", [
    "QbitAI reports that a Chinese company started a trial in China.",
    "DeepTech China reports a trial. QbitAI also reports it.",
    "Chinese-language outlet Machine Heart reports a trial. QbitAI reports another.",
    "QbitAI reports a trial. Later, the Chinese-language outlet QbitAI adds context.",
])
def test_chinese_provenance_must_describe_this_outlet_at_first_mention(story):
    report = contract(dossier(("qbitai", "量子位 QbitAI", "zh")), story)
    assert not report["passed"]
    assert report["chinese_media_labels"] == {"0": False}


@pytest.mark.parametrize("label", [
    "the Chinese-language outlet QbitAI",
    "Chinese tech publication QbitAI",
    "QbitAI, a Chinese-language news outlet",
    "the China-based technology publication QbitAI",
])
def test_explicit_chinese_outlet_identity_is_accepted(label):
    assert contract(
        dossier(("qbitai", "量子位 QbitAI", "zh")),
        f"According to {label}, the company says its trial has started.",
    )["passed"]


def test_a_source_mentioned_in_another_story_does_not_satisfy_local_attribution():
    report = contract(
        dossier(("bloomberg", "Bloomberg", "en"), ("ft", "Financial Times", "en")),
        "Bloomberg reports the trial; the Financial Times also covers it.",
        "The second company says its trial started.",
    )
    assert not report["passed"]
    assert report["story_source_mentions"]["1"] == []
    assert "story 2 is missing its reporting publication attribution" in report["failures"]


def test_chinese_subject_does_not_make_english_publication_chinese_media():
    report = contract(
        dossier(("bloomberg", "Bloomberg", "en")),
        "Bloomberg reports that a Chinese company has begun a trial in China.",
    )
    assert report["passed"]
    assert report["chinese_media_labels"] == {}


def test_chinese_edition_does_not_require_an_english_media_descriptor():
    assert contract(
        dossier(("qbitai", "量子位 QbitAI", "zh")),
        "量子位 QbitAI 报道，公司表示试验已经开始。", language="zh",
    )["passed"]


def test_legacy_chinese_correction_does_not_insert_an_english_descriptor():
    data = dossier(("qbitai", "量子位 QbitAI", "zh"))
    script = "\n".join([
        scriptwriter.morning_opening(EDITION, "zh"),
        "量子位 QbitAI 报道，公司表示试验已经开始。", CLOSING,
    ])
    revised = scriptwriter._minimalize_unsupported_paragraphs(
        script, [{"claim": "Web audit codes D", "evidence_story_numbers": [1]}],
        data, language="zh",
    )
    assert revised == script


def test_new_chinese_publication_gets_language_guidance_without_catalog_entry():
    data = dossier(("new-outlet", "New Outlet", "zh"))
    assert "Chinese-language source" in research.dossier_markdown(data)
    assert not contract(data, "New Outlet reports a trial.")["passed"]
    assert contract(data, "The Chinese-language outlet New Outlet reports a trial.")["passed"]


def test_generation_retries_missing_provenance_and_supplies_shared_style():
    data = dossier(("qbitai", "量子位 QbitAI", "zh"))
    chat = AsyncMock(side_effect=[
        "QbitAI reports that the company says its trial has started.",
        "The Chinese-language outlet QbitAI reports that the company says its trial has started.",
    ])
    with (
        patch.object(scriptwriter, "_resolve_provider", AsyncMock(return_value=("endpoint", "model", "key"))),
        patch.object(scriptwriter, "_chat", chat),
    ):
        result = asyncio.run(scriptwriter.generate_daily_script(
            data, EDITION, target_duration_minutes=None, language="en",
            closing_remarks=CLOSING, ai_endpoint=None, ai_model=None, provider_id=None,
        ))
    assert chat.await_count == 2
    assert "Chinese-language outlet QbitAI" in result
    assert DAILY_NEWS_EDITORIAL_RULES in chat.await_args_list[0].args[0]
    assert "must explicitly identify" in chat.await_args_list[1].args[0]
    assert "Chinese-language source" in chat.await_args.args[1]


def test_final_review_receives_source_language_and_commentary_attribution_rules():
    data = dossier(("qbitai", "量子位 QbitAI", "zh"))
    script = "\n".join([
        scriptwriter.morning_opening(EDITION),
        "The Chinese-language outlet QbitAI reports a trial.", CLOSING,
    ])
    for prompt in (
        review._review_prompt(script, data, EDITION, 1),
        review._batch_review_prompt(script, data, EDITION, [1]),
    ):
        assert "Spoken provenance: Chinese-language source" in prompt
        assert "actual publication/analyst attribution" in prompt
        assert "commentary-only sources" in prompt
        assert "do not imply the original article was read" in prompt

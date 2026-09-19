import asyncio
import json
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
    "the Chinese-language AI outlet QbitAI",
    "the Chinese-language science publication QbitAI",
    "the Chinese-language science and technology outlet QbitAI",
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
        json.dumps({"1": "The Chinese-language outlet QbitAI reports that the company says its trial has started."}),
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


def test_automatic_bulletin_rejects_a_technical_lecture_but_keeps_fixed_duration_mode():
    data = dossier(("bloomberg", "Bloomberg", "en"))
    story = "Bloomberg reports " + " ".join(["detail"] * 180) + "."
    assert not contract(data, story)["passed"]
    script = "\n".join([scriptwriter.morning_opening(EDITION), story, CLOSING])
    assert scriptwriter._bulletin_length_failures(script, data, "en", 8) == []


def test_generation_retries_overlong_automatic_story():
    data = dossier(("bloomberg", "Bloomberg", "en"))
    chat = AsyncMock(side_effect=[
        "Bloomberg reports " + " ".join(["detail"] * 180) + ".",
        json.dumps({"1": "Bloomberg reports that the company says its trial has started."}),
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
    assert "maximum 180" in chat.await_args.args[0]
    assert "its trial has started" in result


def test_cnbc_section_name_accepts_broadcast_publication_but_not_unrelated_names():
    data = dossier(("cnbc_technology", "CNBC Technology", "en"))
    assert contract(data, "CNBC reports that investigators issued a preliminary report.")["passed"]
    assert not contract(data, "NBC reports that investigators issued a preliminary report.")["passed"]
    assert not contract(data, "CNBCOther reports a trial.")["passed"]


@pytest.mark.parametrize("name", ["BBC", "BBC Technology"])
def test_bbc_technology_accepts_its_spoken_publication_name(name):
    data = dossier(("bbc_technology", "BBC Technology", "en"))
    report = contract(data, f"The {name} reports that pubs can accept digital ID apps.")
    assert report["passed"]
    assert report["story_source_mentions"]["0"] == (["BBC Technology", "BBC"] if name == "BBC Technology" else ["BBC"])
    assert scriptwriter._preferred_spoken_source(data.selected[0]) == "BBC"


@pytest.mark.parametrize("name", ["CNBC", "BBCOther", "OtherBBC", "the government"])
def test_bbc_technology_rejects_unrelated_or_missing_publication(name):
    assert not contract(
        dossier(("bbc_technology", "BBC Technology", "en")),
        f"According to {name}, pubs can accept digital ID apps.",
    )["passed"]


def test_generation_accepts_bbc_attribution_in_story_six_without_retries():
    data = dossier(*[("bloomberg", "Bloomberg", "en")] * 5, ("bbc_technology", "BBC Technology", "en"))
    stories = ["Bloomberg reports that a company started a trial."] * 5 + [
        "The BBC reports that new rules let pubs and shops in England and Wales "
        "accept digital ID apps on phones to prove a customer's age."
    ]
    chat = AsyncMock(return_value="\n\n".join(stories))
    with (
        patch.object(scriptwriter, "_resolve_provider", AsyncMock(return_value=("endpoint", "model", "key"))),
        patch.object(scriptwriter, "_chat", chat),
    ):
        result = asyncio.run(scriptwriter.generate_daily_script(
            data, EDITION, target_duration_minutes=None, language="en",
            closing_remarks=CLOSING, ai_endpoint=None, ai_model=None, provider_id=None,
        ))
    assert chat.await_count == 1
    assert result.splitlines()[1:-1] == stories
    assert scriptwriter.script_contract_report(
        result, data, EDITION, language="en", closing_remarks=CLOSING,
    )["passed"]


@pytest.mark.parametrize("bad_repair", ["not JSON", json.dumps({"1": "Changed passing story", "2": "BBC reports a trial."})])
def test_generation_repairs_all_failures_without_rewriting_passing_stories(bad_repair):
    data = dossier(("cnbc_technology", "CNBC Technology", "en"), ("bbc", "BBC", "en"))
    passing = "CNBC reports that the company says its trial has started."
    # The observed 181-word boundary and missing attribution must be diagnosed together.
    overlong = " ".join(["detail"] * 181) + "."
    repaired = "BBC reports that the company says its trial has started."
    responses = iter([passing + "\n" + overlong, bad_repair, json.dumps({"2": repaired})])
    calls = []

    async def chat(*args, **kwargs):
        calls.append((args, kwargs))
        kwargs["route_selected"]("selected-endpoint", "selected-model", "selected-key")
        return next(responses)

    with (
        patch.object(scriptwriter, "_resolve_provider", AsyncMock(return_value=("endpoint", "model", "key"))),
        patch.object(scriptwriter, "_chat", side_effect=chat),
    ):
        result = asyncio.run(scriptwriter.generate_daily_script(
            data, EDITION, target_duration_minutes=None, language="en",
            closing_remarks=CLOSING, ai_endpoint=None, ai_model=None, provider_id=None,
        ))
    assert result.splitlines() == [scriptwriter.morning_opening(EDITION), passing, repaired, CLOSING]
    for args, kwargs in calls[1:]:
        assert "181 words, maximum 180" in args[0]
        assert "story 2 is missing its reporting publication attribution" in args[0]
        assert "spoken source: BBC" in args[1]
        assert passing not in args[1]
        assert overlong in args[1]
        assert args[2:5] == ("selected-endpoint", "selected-model", "selected-key")
        assert kwargs["disable_thinking"] is True


def test_generation_keeps_newly_repaired_stories_and_still_enforces_word_limit():
    data = dossier(("bloomberg", "Bloomberg", "en"), ("bbc", "BBC", "en"))
    overlong = "BBC reports " + " ".join(["detail"] * 179) + "."
    fixed = "Bloomberg reports that the company says its trial has started."
    chat = AsyncMock(side_effect=[
        "A company says its trial has started.\n" + overlong,
        json.dumps({"1": fixed, "2": overlong}),
        json.dumps({"2": overlong}),
    ])
    with (
        patch.object(scriptwriter, "_resolve_provider", AsyncMock(return_value=("endpoint", "model", "key"))),
        patch.object(scriptwriter, "_chat", chat),
        pytest.raises(RuntimeError, match="exhausted 3 semantic response attempts: news bulletin story 2 is too long: 181 words"),
    ):
        asyncio.run(scriptwriter.generate_daily_script(
            data, EDITION, target_duration_minutes=None, language="en",
            closing_remarks=CLOSING, ai_endpoint=None, ai_model=None, provider_id=None,
        ))
    assert "Repair ONLY story numbers [2]" in chat.await_args.args[0]
    assert fixed not in chat.await_args.args[1]


@pytest.mark.parametrize("duration, expected_calls", [(None, 2), (8, 1)])
def test_audit_correction_obeys_automatic_length_limit_and_preserves_passing_story(duration, expected_calls):
    data = dossier(("bloomberg", "Bloomberg", "en"), ("bbc", "BBC", "en"))
    passing = "BBC reports that the company says its trial has started."
    original = "\n".join([scriptwriter.morning_opening(EDITION), "Bloomberg reports a trial.", passing, CLOSING])
    long = "Bloomberg reports " + " ".join(["detail"] * 182) + "."
    short = "Bloomberg reports that the company says its trial has started."
    chat = AsyncMock(side_effect=[json.dumps({"1": long}), json.dumps({"1": short})])
    with (
        patch.object(scriptwriter, "_resolve_provider", AsyncMock(return_value=("endpoint", "model", "key"))),
        patch.object(scriptwriter, "_chat", chat),
    ):
        result = asyncio.run(scriptwriter.revise_daily_script(
            original, data, [{"evidence_story_numbers": [1]}], EDITION,
            language="en", closing_remarks=CLOSING, ai_endpoint=None, ai_model=None,
            provider_id=None, target_duration_minutes=duration,
        ))
    assert chat.await_count == expected_calls
    assert result.splitlines()[2] == passing
    assert result.splitlines()[1] == (short if duration is None else long)
    if duration is None:
        assert "maximum 180" in chat.await_args.args[0]
        assert "184 words" in chat.await_args.args[0]


def test_audit_correction_exhausts_length_retries_without_truncating_or_switching_provider():
    data = dossier(("bloomberg", "Bloomberg", "en"))
    original = "\n".join([scriptwriter.morning_opening(EDITION), "Bloomberg reports a trial.", CLOSING])
    calls = []

    async def oversized(*args, **kwargs):
        calls.append(args)
        kwargs["route_selected"]("selected-endpoint", "selected-model", "selected-key")
        return json.dumps({"1": "Bloomberg reports " + " ".join(["detail"] * 182) + "."})

    with (
        patch.object(scriptwriter, "_resolve_provider", AsyncMock(return_value=("endpoint", "model", "key"))),
        patch.object(scriptwriter, "_chat", side_effect=oversized),
        pytest.raises(RuntimeError, match="exhausted 3 length-constrained responses"),
    ):
        asyncio.run(scriptwriter.revise_daily_script(
            original, data, [{"evidence_story_numbers": [1]}], EDITION,
            language="en", closing_remarks=CLOSING, ai_endpoint=None, ai_model=None, provider_id=None,
        ))
    assert len(calls) == scriptwriter.DAILY_NEWS_EDIT_RESPONSE_ATTEMPTS
    assert all(call[2:5] == ("selected-endpoint", "selected-model", "selected-key") for call in calls[1:])


def test_generation_spells_grouped_quantity_before_review():
    data = dossier(("bloomberg", "Bloomberg", "en"))
    with (
        patch.object(scriptwriter, "_resolve_provider", AsyncMock(return_value=("endpoint", "model", "key"))),
        patch.object(scriptwriter, "_chat", AsyncMock(return_value="Bloomberg reports capacity for 600,000 homes.")),
    ):
        result = asyncio.run(scriptwriter.generate_daily_script(
            data, EDITION, target_duration_minutes=None, language="en",
            closing_remarks=CLOSING, ai_endpoint=None, ai_model=None, provider_id=None,
        ))
    assert result.splitlines()[1] == "Bloomberg reports capacity for six hundred thousand homes."
    assert len(result.splitlines()) == 3


def test_correction_spells_quantity_only_in_failed_story():
    data = dossier(("bloomberg", "Bloomberg", "en"), ("bbc", "BBC", "en"))
    passing = "BBC reports capacity for 250,000 homes."
    original = "\n".join([scriptwriter.morning_opening(EDITION), "Bloomberg reports a trial.", passing, CLOSING])
    with (
        patch.object(scriptwriter, "_resolve_provider", AsyncMock(return_value=("endpoint", "model", "key"))),
        patch.object(scriptwriter, "_chat", AsyncMock(return_value=json.dumps({"1": "Bloomberg reports capacity for 600,000 homes."}))),
    ):
        result = asyncio.run(scriptwriter.revise_daily_script(
            original, data, [{"evidence_story_numbers": [1]}], EDITION,
            language="en", closing_remarks=CLOSING, ai_endpoint=None, ai_model=None,
            provider_id=None, target_duration_minutes=None,
        ))
    assert result.splitlines()[1] == "Bloomberg reports capacity for six hundred thousand homes."
    assert result.splitlines()[2] == passing
    assert len(result.splitlines()) == 4


@pytest.mark.parametrize("bad_response", [False, True])
def test_review_repairs_contract_before_web_and_records_exhaustion(tmp_path, bad_response):
    data = dossier(("bbc", "BBC", "en"), ("deeptech", "DeepTech China", "zh"))
    passing = "BBC reports that the company says its trial has started."
    fixed = "The Chinese-language outlet DeepTech China reports that the company says its trial has started."
    original = "\n".join([scriptwriter.morning_opening(EDITION), passing,
                          "The company says its trial has started.", CLOSING])
    chat = AsyncMock(return_value="not JSON" if bad_response else json.dumps({"2": fixed}))
    web = AsyncMock(return_value=({"issues": []}, "W1P;2P", "", "chatgpt"))
    with (
        patch.object(scriptwriter, "_resolve_provider", AsyncMock(return_value=("endpoint", "model", "key"))),
        patch.object(scriptwriter, "_chat", chat),
        patch.object(review, "_web_story_review", web),
        patch.object(review, "close_opencli_site_sessions", AsyncMock()),
    ):
        task = review.review_daily_script(
            original, data, EDITION, tmp_path, language="en", closing_remarks=CLOSING,
            ai_endpoint=None, ai_model=None, provider_id=None,
        )
        if bad_response:
            with pytest.raises(RuntimeError, match="Automatic script contract repair exhausted"):
                asyncio.run(task)
        else:
            result = asyncio.run(task)
            assert result.script.splitlines()[1:-1] == [passing, fixed]
            assert result.report["final_contract"]["passed"]
            assert fixed in web.await_args.args[0]
    report = json.loads((tmp_path / "review/fact_check_report.json").read_text())
    assert report["passed"] is not bad_response
    assert report["contract_repairs"][0]["before_contract"]["passed"] is False
    assert chat.await_count == (3 if bad_response else 1)
    assert web.await_count == (0 if bad_response else 1)
    for call in chat.await_args_list:
        assert "Repair ONLY story numbers [2]" in call.args[0]
        assert passing not in call.args[1]


def test_review_bounds_contract_duration_oscillation(tmp_path):
    data = dossier(("bbc", "BBC", "en"))
    original = "\n".join([scriptwriter.morning_opening(EDITION), "BBC reports a trial.", CLOSING])
    with (
        patch.object(review, "script_contract_report", return_value={"passed": False, "failures": ["duration mismatch"]}),
        patch.object(review, "generate_daily_script", AsyncMock(return_value=original)) as repair,
        patch.object(review, "_web_story_review", AsyncMock()) as web,
        pytest.raises(RuntimeError, match="bounded automatic repairs"),
    ):
        asyncio.run(review._review_daily_script(
            original, data, EDITION, tmp_path, language="en", closing_remarks=CLOSING,
            ai_endpoint=None, ai_model=None, provider_id=None, review_session_namespace="test",
        ))
    assert repair.await_count == 3
    web.assert_not_awaited()
    report = json.loads((tmp_path / "review/fact_check_report.json").read_text())
    assert report["manual_review_required"]

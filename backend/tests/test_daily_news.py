import asyncio
import json
import math
import struct
import wave
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from backend import config
from backend.daily_news import scheduler
from backend.daily_news.scheduler import _is_due, create_daily_task, next_run_at
from backend.daily_news import research, review
from backend.daily_news.scriptwriter import (
    SOURCE_SPOKEN_ALIASES,
    _issues_are_unsupported_only,
    _minimalize_unsupported_paragraphs,
    enforce_script_contract,
    morning_opening,
)
from backend.daily_news.source_catalog import load_source_catalog
from backend.models import DailyAutomationSettings, SourceType, TaskConfig, TaskResponse, TaskStatus
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


def test_source_catalog_contains_bilingual_priority_six_and_twelve_total():
    load_source_catalog.cache_clear()
    sources = load_source_catalog()

    assert len(sources) == 12
    assert [source.priority for source in sources[:6]] == [1, 2, 3, 4, 5, 6]
    assert {source.language for source in sources} == {"en", "zh"}


def test_article_page_json_ld_supplies_machine_verifiable_publication_date():
    payload = """<html><head><script type="application/ld+json">
    {"@type":"NewsArticle","datePublished":"2026-08-18T06:30:00+08:00"}
    </script></head><body><article><p>This paragraph contains enough text to be evidence.</p></article></body></html>"""

    published = research._extract_published_at(payload)

    assert published == datetime(2026, 8, 17, 22, 30, tzinfo=timezone.utc)


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


def test_only_pure_d_audits_can_bypass_the_writing_model():
    assert _issues_are_unsupported_only([
        {"claim": "Web audit codes D for story 3"},
    ]) is True
    assert _issues_are_unsupported_only([
        {"claim": "Web audit codes DE for story 3"},
    ]) is False
    assert _issues_are_unsupported_only([
        {"claim": "unstructured reviewer note"},
    ]) is False


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
    assert SOURCE_SPOKEN_ALIASES["量子位 QbitAI"] == ("QbitAI",)


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
    assert settings.tts_model == "orpheus-en"
    assert settings.voice == "leah"
    assert settings.auto_publish is True
    assert settings.publish_targets == ["youtube", "x", "apple_podcast"]
    assert settings.publish_visibility == "public"
    assert settings.news_image_count == 4


def test_daily_desk_recipe_validates_tts_model_and_voice_before_saving():
    with pytest.raises(ValueError, match="Unknown TTS model"):
        DailyAutomationSettings(tts_model="missing-model")

    with pytest.raises(ValueError, match="unavailable"):
        DailyAutomationSettings(tts_model="orpheus-en", voice="Carter")

    settings = DailyAutomationSettings(tts_model="orpheus-en", voice="tara")
    assert settings.voice == "tara"

    with pytest.raises(ValueError, match="greater than or equal to 1"):
        DailyAutomationSettings(footage_clip_count=0)


def test_daily_desk_recipe_backfills_public_footage_count_for_old_settings():
    settings = DailyAutomationSettings.model_validate(
        {
            "tts_model": "vibevoice-0.5b",
            "voice": "Carter",
            "collage_broll_count": 5,
        }
    )

    assert settings.footage_clip_count == 8
    assert settings.collage_broll_count == 5
    assert settings.news_image_count == 4


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

    assert settings.target_duration_minutes == 12
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
        collage_broll_count=7,
        news_image_count=6,
        public_footage_enabled=True,
        footage_clip_count=11,
        background_music_provider="local",
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
    assert task_config.target_duration_minutes == 12
    assert task_config.tts_model == "orpheus-en"
    assert task_config.voice_1 == "tara"
    assert task_config.collage_broll_count == 7
    assert task_config.news_images_enabled is True
    assert task_config.news_image_count == 6
    assert task_config.footage_enabled is True
    assert task_config.footage_clip_count == 11
    assert task_config.background_music_provider == "local"
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
                duration_override=1,
            )
        )

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["last_run_date"] == "2026-08-18"
    assert state["last_task_id"] == "yesterday"
    assert state["last_test_task_id"] == queued.id
    test_config = create.await_args.args[2]
    assert test_config.target_duration_minutes == 1
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
        "feed_title": "Frontier Tech Daily",
        "author": "FrontierTechSN",
    }


def test_review_token_normalization_accepts_only_known_ui_wrappers():
    payload = review._batch_payload(
        "💬️ \ufeff\u200b```text\n1P 2D\n```\u200b",
        [1, 2],
    )

    assert payload["approved"] is False
    assert payload["issues"][0]["evidence_story_numbers"] == [2]
    with pytest.raises(ValueError, match="invalid characters"):
        review._batch_payload("Result: 1P2P", [1, 2])
    with pytest.raises(ValueError, match="invalid characters"):
        review._batch_payload("1P2P because both stories pass", [1, 2])


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


def test_story_review_uses_gemini_primary_without_touching_chatgpt():
    command = AsyncMock(
        return_value=OpenCLIResult(
            ("gemini", "ask"),
            0,
            '[{"response":"💬 P"}]',
            "",
        )
    )
    with patch.object(review, "run_opencli", command):
        payload, _, url, provider = asyncio.run(
            review._web_story_review("audit\nmore", story_number=1, log=None)
        )

    assert payload["approved"] is True
    assert url == ""
    assert provider == "gemini"
    assert command.await_count == 1
    args = command.await_args.args[0]
    assert args[:2] == ["gemini", "ask"]
    assert args[args.index("--model") + 1] == "3.7-flash"
    assert args[args.index("--new") + 1] == "true"
    assert review._REVIEW_REQUEST_RE.search(args[2])
    assert "\n" not in args[2]
    assert command.await_args.kwargs["timeout"] == review.config.DAILY_NEWS_WEB_REVIEW_TIMEOUT + 60
    assert command.await_args.kwargs["site_session_namespace"].startswith(
        "frontiertechsn-review-"
    )


def test_gemini_late_recovery_accepts_only_its_owned_turn():
    owned_prompt = ""

    async def command(args, **kwargs):
        nonlocal owned_prompt
        if args[:2] == ["gemini", "ask"]:
            owned_prompt = args[2]
            return OpenCLIResult(
                tuple(args),
                0,
                '[{"response":"💬 [NO RESPONSE] No Gemini response within 90s."}]',
                "",
            )
        if args[:2] == ["gemini", "read"]:
            turns = [
                {"Index": 1, "Role": "Assistant", "Text": "What's the vibe, Leo?"},
                {"Index": 2, "Role": "User", "Text": owned_prompt},
                {"Index": 3, "Role": "User", "Text": "split evidence fragment"},
                {"Index": 4, "Role": "Assistant", "Text": "💬 `1P2P`"},
            ]
            return OpenCLIResult(tuple(args), 0, json.dumps(turns), "")
        raise AssertionError(args)

    command_mock = AsyncMock(side_effect=command)
    with patch.object(review, "run_opencli", command_mock):
        payload, raw, _, provider = asyncio.run(
            review._web_story_review("audit", story_numbers=[1, 2], log=None)
        )

    assert payload["approved"] is True
    assert provider == "gemini"
    assert "[GEMINI RECOVERY]" in raw
    assert not any(call.args[0][0] == "chatgpt" for call in command_mock.await_args_list)
    assert len(
        {
            call.kwargs["site_session_namespace"]
            for call in command_mock.await_args_list
        }
    ) == 1


def test_unowned_gemini_greeting_is_rejected_before_fresh_retry():
    gemini_asks = 0

    async def command(args, **kwargs):
        nonlocal gemini_asks
        if args[:2] == ["gemini", "ask"]:
            gemini_asks += 1
            response = "[NO RESPONSE]" if gemini_asks == 1 else "1P2P"
            return OpenCLIResult(
                tuple(args),
                0,
                json.dumps([{"response": response}]),
                "",
            )
        if args[:2] == ["gemini", "read"]:
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
    assert provider == "gemini"
    assert gemini_asks == 2
    assert "What's the vibe" not in raw
    gemini_calls = [
        call.args[0]
        for call in command_mock.await_args_list
        if call.args[0][:2] == ["gemini", "ask"]
    ]
    assert all(args[args.index("--new") + 1] == "true" for args in gemini_calls)


def test_gemini_late_recovery_polls_before_resubmitting_the_owned_request():
    owned_prompt = ""
    gemini_asks = 0
    gemini_reads = 0

    async def command(args, **kwargs):
        nonlocal owned_prompt, gemini_asks, gemini_reads
        if args[:2] == ["gemini", "ask"]:
            gemini_asks += 1
            owned_prompt = args[2]
            return OpenCLIResult(
                tuple(args),
                0,
                '[{"response":"[NO RESPONSE]"}]',
                "",
            )
        if args[:2] == ["gemini", "read"]:
            gemini_reads += 1
            turns = [{"Role": "User", "Text": owned_prompt}]
            if gemini_reads == 3:
                turns.append({"Role": "Assistant", "Text": "1P2P"})
            return OpenCLIResult(tuple(args), 0, json.dumps(turns), "")
        raise AssertionError(args)

    with patch.object(review, "run_opencli", AsyncMock(side_effect=command)):
        payload, _, _, provider = asyncio.run(
            review._web_story_review("audit", story_numbers=[1, 2], log=None)
        )

    assert payload["approved"] is True
    assert provider == "gemini"
    assert gemini_asks == 1
    assert gemini_reads == 3


def test_story_review_falls_back_to_fresh_chatgpt_after_gemini_exhaustion():
    gemini_asks = 0

    async def command(args, **kwargs):
        nonlocal gemini_asks
        if args[:2] == ["gemini", "ask"]:
            gemini_asks += 1
            return OpenCLIResult(
                tuple(args),
                0,
                '[{"response":"[NO RESPONSE]"}]',
                "",
            )
        if args[:2] == ["gemini", "read"]:
            return OpenCLIResult(
                tuple(args),
                0,
                '[{"Role":"Assistant","Text":"What is next?"}]',
                "",
            )
        if args[:2] == ["chatgpt", "model"]:
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
                '[{"response":"1P2P","conversationUrl":"https://chatgpt.com/c/fallback"}]',
                "",
            )
        raise AssertionError(args)

    command_mock = AsyncMock(side_effect=command)
    with patch.object(review, "run_opencli", command_mock):
        payload, _, url, provider = asyncio.run(
            review._web_story_review("audit", story_numbers=[1, 2], log=None)
        )

    assert payload["approved"] is True
    assert provider == "chatgpt"
    assert url.endswith("/fallback")
    assert gemini_asks == 2
    calls = [call.args[0] for call in command_mock.await_args_list]
    first_chatgpt = next(index for index, args in enumerate(calls) if args[0] == "chatgpt")
    assert all(args[0] == "gemini" for args in calls[:first_chatgpt])
    chatgpt_ask = next(args for args in calls if args[:2] == ["chatgpt", "ask"])
    assert chatgpt_ask[chatgpt_ask.index("--new") + 1] == "true"


def test_chatgpt_fallback_retries_transient_model_picker_failure():
    model_attempts = 0
    messages: list[str] = []

    async def command(args, **kwargs):
        nonlocal model_attempts
        if args[:2] == ["gemini", "ask"]:
            return OpenCLIResult(tuple(args), 0, '[{"response":"[NO RESPONSE]"}]', "")
        if args[:2] == ["gemini", "read"]:
            return OpenCLIResult(tuple(args), 0, "[]", "")
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
                '[{"response":"1P2P","conversationUrl":"https://chatgpt.com/c/retry"}]',
                "",
            )
        raise AssertionError(args)

    with patch.object(review, "run_opencli", AsyncMock(side_effect=command)):
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
    assert "[CHATGPT MODEL ERROR 1/2]" in raw
    assert any("model selection attempt 1/2 failed" in message for message in messages)


def test_chatgpt_fallback_recovers_the_target_conversation_after_route_drift():
    chatgpt_prompt = ""
    target_url = "https://chatgpt.com/c/6a868e01-3b9c-83ec-b973-e3aee234afe4"

    async def command(args, **kwargs):
        nonlocal chatgpt_prompt
        if args[:2] == ["gemini", "ask"]:
            raise OpenCLIError("Gemini browser lease failed")
        if args[:2] == ["gemini", "read"]:
            return OpenCLIResult(tuple(args), 0, "[]", "")
        if args[:2] == ["chatgpt", "model"]:
            return OpenCLIResult(tuple(args), 0, '[{"Status":"Success"}]', "")
        if args[:2] == ["chatgpt", "ask"]:
            chatgpt_prompt = args[2]
            raise OpenCLIError(
                "ChatGPT navigated away from the target conversation "
                f"({target_url}); current URL is https://chatgpt.com/new"
            )
        if args[:2] == ["chatgpt", "detail"]:
            turns = [
                {"Index": 1, "Role": "User", "Text": chatgpt_prompt},
                {"Index": 2, "Role": "Assistant", "Text": "1P2P"},
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


def test_gemini_retry_uses_current_flash_when_model_picker_is_missing():
    gemini_asks = 0

    async def command(args, **kwargs):
        nonlocal gemini_asks
        if args[:2] == ["gemini", "ask"]:
            gemini_asks += 1
            if gemini_asks == 1:
                raise OpenCLIError("Gemini model picker button was not found")
            return OpenCLIResult(tuple(args), 0, '[{"response":"1P2P"}]', "")
        if args[:2] == ["gemini", "read"]:
            return OpenCLIResult(tuple(args), 0, "[]", "")
        raise AssertionError(args)

    command_mock = AsyncMock(side_effect=command)
    with patch.object(review, "run_opencli", command_mock):
        payload, _, _, provider = asyncio.run(
            review._web_story_review("audit", story_numbers=[1, 2], log=None)
        )

    gemini_calls = [
        call.args[0]
        for call in command_mock.await_args_list
        if call.args[0][:2] == ["gemini", "ask"]
    ]
    assert payload["approved"] is True
    assert provider == "gemini"
    assert "--model" in gemini_calls[0]
    assert "--model" not in gemini_calls[1]
    assert gemini_calls[0][gemini_calls[0].index("--new") + 1] == "true"
    assert gemini_calls[1][gemini_calls[1].index("--new") + 1] == "false"


@pytest.mark.parametrize(
    ("provider", "fallback_used", "reviewer_fragment"),
    [
        ("gemini", False, "Gemini Web (3.7-flash) via"),
        ("chatgpt", True, "with ChatGPT Web (medium) fallback"),
    ],
)
def test_review_report_records_gemini_primary_and_chatgpt_fallback(
    tmp_path: Path,
    provider: str,
    fallback_used: bool,
    reviewer_fragment: str,
):
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
    web_review = AsyncMock(return_value=(payload, "[RAW]", "", provider))
    contract = {"passed": True, "failures": []}

    with (
        patch.object(review, "_web_story_review", web_review),
        patch.object(review, "script_contract_report", return_value=contract),
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

    assert result.report["fallback_used"] is fallback_used
    assert reviewer_fragment in result.report["reviewer"]
    assert web_review.await_args.kwargs["site_session_namespace"].startswith(
        "frontiertechsn-review-"
    )
    assert (tmp_path / "review" / "story-review-prompt-1-group-1.txt").exists()
    assert (tmp_path / "review" / "story-review-response-1-group-1.txt").exists()

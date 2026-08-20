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
        "claim": "ChatGPT audit codes D for story 1",
        "evidence_story_numbers": [1],
    }]

    revised = _minimalize_unsupported_paragraphs(script, issues)

    assert revised.splitlines()[1] == "QbitAI reports that MyContext is open source."


def test_only_pure_d_audits_can_bypass_the_writing_model():
    assert _issues_are_unsupported_only([
        {"claim": "ChatGPT audit codes D for story 3"},
    ]) is True
    assert _issues_are_unsupported_only([
        {"claim": "ChatGPT audit codes DE for story 3"},
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

    with patch.object(config, "OUTPUTS_DIR", tmp_path):
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


def test_chatgpt_review_falls_back_to_gemini_after_a_browser_failure():
    async def command(args, **kwargs):
        if args[:2] == ["chatgpt", "model"]:
            return OpenCLIResult(tuple(args), 0, '[{"Status":"Success","Model":"Balanced"}]', "")
        if args[:2] == ["chatgpt", "ask"]:
            raise OpenCLIError("temporary browser lease failure")
        if args[:2] == ["chatgpt", "read"]:
            raise OpenCLIError("no completed ChatGPT turn")
        if args[:2] == ["gemini", "ask"]:
            return OpenCLIResult(tuple(args), 0, '[{"response":"💬 P"}]', "")
        raise AssertionError(args)

    command_mock = AsyncMock(side_effect=command)
    with patch.object(review, "run_opencli", command_mock):
        payload, _, url, provider = asyncio.run(
            review._chatgpt_review("audit", story_number=1, log=None)
        )

    assert payload["approved"] is True
    assert url == ""
    assert provider == "gemini"
    assert any(call.args[0][:3] == ["chatgpt", "model", "medium"] for call in command_mock.await_args_list)
    gemini_call = next(
        call for call in command_mock.await_args_list
        if call.args[0][:2] == ["gemini", "ask"]
    )
    assert gemini_call.kwargs["timeout"] == review.config.DAILY_NEWS_WEB_REVIEW_TIMEOUT + 60


def test_chatgpt_review_retries_a_truncated_response_until_valid_code():
    model = OpenCLIResult(
        args=("chatgpt", "model"), returncode=0,
        stdout='[{"Status":"Already selected","Model":"Balanced"}]', stderr="",
    )
    first = OpenCLIResult(
        args=("chatgpt", "ask"), returncode=0,
        stdout='[{"response":"{\\"approved\\":false, citations interrupted","conversationUrl":"https://chatgpt.com/c/test"}]',
        stderr="",
    )
    repaired = OpenCLIResult(
        args=("chatgpt", "read"), returncode=0,
        stdout='[{"Index":1,"Role":"Assistant","Text":"D"}]',
        stderr="",
    )
    command = AsyncMock(side_effect=[model, first, repaired])
    with patch.object(review, "run_opencli", command):
        payload, raw, _, provider = asyncio.run(
            review._chatgpt_review("audit\nmore", story_number=2, log=None)
        )

    assert payload["approved"] is False
    assert "extrapolative sentence" in payload["issues"][0]["correction"]
    assert "D" in raw
    assert provider == "chatgpt"
    assert command.await_args_list[2].args[0][1] == "read"
    assert "\n" not in command.await_args_list[1].args[0][2]


def test_gemini_fallback_recovers_a_late_completed_response():
    async def command(args, **kwargs):
        if args[:2] == ["chatgpt", "model"]:
            raise OpenCLIError("model picker unavailable")
        if args[:2] == ["gemini", "ask"]:
            return OpenCLIResult(
                tuple(args), 0,
                '[{"response":"💬 [NO RESPONSE] No Gemini response within 45s."}]',
                "",
            )
        if args[:2] == ["gemini", "read"]:
            return OpenCLIResult(
                tuple(args), 0,
                '[{"Index":1,"Role":"Assistant","Text":"1P2P"}]',
                "",
            )
        raise AssertionError(args)

    with patch.object(review, "run_opencli", AsyncMock(side_effect=command)):
        payload, raw, _, provider = asyncio.run(
            review._chatgpt_review("audit", story_numbers=[1, 2], log=None)
        )

    assert payload["approved"] is True
    assert provider == "gemini"
    assert "[GEMINI RECOVERY]" in raw


def test_gemini_fallback_retries_once_with_a_fresh_conversation():
    gemini_asks = 0

    async def command(args, **kwargs):
        nonlocal gemini_asks
        if args[:2] == ["chatgpt", "model"]:
            raise OpenCLIError("model picker unavailable")
        if args[:2] == ["gemini", "ask"]:
            gemini_asks += 1
            response = "[NO RESPONSE]" if gemini_asks == 1 else "1P2P"
            return OpenCLIResult(tuple(args), 0, f'[{json.dumps({"response": response})}]', "")
        if args[:2] == ["gemini", "read"]:
            return OpenCLIResult(
                tuple(args), 0,
                '[{"Index":1,"Role":"Assistant","Text":"What is next?"}]',
                "",
            )
        raise AssertionError(args)

    command_mock = AsyncMock(side_effect=command)
    with patch.object(review, "run_opencli", command_mock):
        payload, raw, _, provider = asyncio.run(
            review._chatgpt_review("audit", story_numbers=[1, 2], log=None)
        )

    assert payload["approved"] is True
    assert provider == "gemini"
    assert gemini_asks == 2
    assert "[GEMINI 1]" in raw
    assert "[GEMINI 2]" in raw
    assert all(
        call.args[0][call.args[0].index("--new") + 1] == "true"
        for call in command_mock.await_args_list
        if call.args[0][:2] == ["gemini", "ask"]
    )


def test_gemini_retry_uses_current_flash_when_model_picker_is_missing():
    gemini_asks = 0

    async def command(args, **kwargs):
        nonlocal gemini_asks
        if args[:2] == ["chatgpt", "model"]:
            raise OpenCLIError("model picker unavailable")
        if args[:2] == ["gemini", "ask"]:
            gemini_asks += 1
            if gemini_asks == 1:
                raise OpenCLIError("Gemini model picker button was not found")
            return OpenCLIResult(tuple(args), 0, '[{"response":"1P2P"}]', "")
        if args[:2] == ["gemini", "read"]:
            return OpenCLIResult(
                tuple(args), 0,
                '[{"Index":1,"Role":"Assistant","Text":"Your move"}]',
                "",
            )
        raise AssertionError(args)

    command_mock = AsyncMock(side_effect=command)
    with patch.object(review, "run_opencli", command_mock):
        payload, _, _, provider = asyncio.run(
            review._chatgpt_review("audit", story_numbers=[1, 2], log=None)
        )

    gemini_calls = [
        call.args[0] for call in command_mock.await_args_list
        if call.args[0][:2] == ["gemini", "ask"]
    ]
    assert payload["approved"] is True
    assert provider == "gemini"
    assert "--model" in gemini_calls[0]
    assert "--model" not in gemini_calls[1]
    assert gemini_calls[0][gemini_calls[0].index("--new") + 1] == "true"
    assert gemini_calls[1][gemini_calls[1].index("--new") + 1] == "false"

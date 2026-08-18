import asyncio
import json
import math
import struct
import wave
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

from backend import config
from backend.daily_news.scheduler import _is_due, next_run_at
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
    assert settings.auto_publish is True
    assert settings.publish_targets == ["youtube", "x", "apple_podcast"]
    assert settings.publish_visibility == "public"


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
    assert manifest["platforms"]["apple_podcast"]["identity"] is None


def test_chatgpt_review_retries_a_browser_failure():
    success = OpenCLIResult(
        args=("chatgpt", "ask"),
        returncode=0,
        stdout='[{"response":"P","conversationUrl":"https://chatgpt.com/c/test"}]',
        stderr="",
    )
    command = AsyncMock(side_effect=[
        OpenCLIError("temporary browser lease failure"), success,
    ])
    with (
        patch.object(review, "run_opencli", command),
        patch.object(review.asyncio, "sleep", AsyncMock()),
    ):
        payload, _, url = asyncio.run(review._chatgpt_review("audit", story_number=1, log=None))

    assert payload["approved"] is True
    assert url.endswith("/test")
    assert command.await_count >= 2


def test_chatgpt_review_retries_a_truncated_response_until_valid_code():
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
    command = AsyncMock(side_effect=[first, repaired])
    with (
        patch.object(review, "run_opencli", command),
        patch.object(review.asyncio, "sleep", AsyncMock()),
    ):
        payload, raw, _ = asyncio.run(review._chatgpt_review("audit\nmore", story_number=2, log=None))

    assert payload["approved"] is False
    assert "extrapolative sentence" in payload["issues"][0]["correction"]
    assert "D" in raw
    assert command.await_args_list[1].args[0][1] == "read"
    assert "\n" not in command.await_args_list[0].args[0][2]

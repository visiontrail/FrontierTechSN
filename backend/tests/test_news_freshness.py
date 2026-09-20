import asyncio
import hashlib
import json
import sqlite3
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from backend import config
from backend.daily_news import freshness, research, review, scriptwriter
from backend.daily_news.source_catalog import load_source_catalog

NOW = datetime(2026, 9, 20, 3, 36, tzinfo=timezone.utc)
EDITION = NOW.date()
QUOTE = "On September 20, 2026, Lab published a new evaluation of Orbit with reproducible results."
OLD = "On September 15, 2026, Acme launched the Orbit model."


def article(source="nature", **changes):
    row = research.NewsArticle(
        id=source, source_id=source, source_name=source, language="en",
        title="Lab publishes reproducible Orbit evaluation", url=f"https://{source}.test/orbit",
        published_at=(NOW - timedelta(hours=1)).isoformat(), category="science",
        evidence_text=OLD + " " + QUOTE, evidence_status="article_excerpt",
    )
    return replace(row, **changes)


def assessment(row, **changes):
    return {
        "id": row.id, "decision": "include", "reason": "New dated evaluation",
        "event_identity": "Lab publishes Orbit evaluation", "event_date": "2026-09-15",
        "development_date": "2026-09-20", "development_kind": "follow_up",
        "development": "Lab published a new evaluation of Orbit.",
        "development_quote": QUOTE, "date_basis": "explicit", "date_quote": QUOTE,
        "matched_history_keys": [], "new_since_history": "", **changes,
    }


def validate(row, record, history=()):
    return freshness.validate_assessment(row, record, list(history), now=NOW,
                                        window_hours=36, timezone_name="UTC")


def previous(row=None, **changes):
    row = row or article()
    return {"key": "prior", "edition_date": "2026-09-19", "title": row.title,
            "url": row.url, "published_at": "2026-09-15T00:00:00+00:00",
            "script": OLD, "development": {}, **changes}


@pytest.mark.parametrize("source,kind", [
    ("a16z", "analysis"), ("sequoia", "analysis"), ("bloomberg", "news"),
    ("nature", "news"), ("unknown-new-source", "news"),
])
def test_all_sources_share_exact_window_even_with_legacy_override(source, kind):
    row = article(source, content_kind=kind, lookback_hours=168,
                  published_at=(NOW - timedelta(days=5)).isoformat())
    assert not research._within_window(row, NOW, 36)
    assert research._within_window(replace(row, published_at=(NOW - timedelta(hours=36)).isoformat()), NOW, 36)
    assert not research._within_window(replace(row, published_at=(NOW - timedelta(hours=36, seconds=1)).isoformat()), NOW, 36)
    assert not research._within_window(replace(row, published_at=(NOW + timedelta(seconds=1)).isoformat()), NOW, 36)
    assert not research._within_window(replace(row, published_at="invalid"), NOW, 36)


@pytest.mark.parametrize("sid,category", [
    ("a16z", "business"), ("bloomberg", "business"), ("financial_times", "business"),
    ("ieee_spectrum", "robotics"), ("nature", "science"),
])
def test_preferences_only_reward_qualified_relevant_evidence(sid, category):
    source = next(s for s in load_source_catalog() if s.id == sid)
    row = article(sid, category=category, content_kind=source.content_kind)
    rank = lambda a: research.score_and_deduplicate([a], [source], NOW)[0]
    baseline = rank(row).score
    assert row.preference_bonus == 0
    qualified = replace(row, freshness={"decision": "include"})
    assert rank(qualified).score == baseline + 6
    assert qualified.preference_bonus == 6
    assert rank(replace(qualified, evidence_status="feed_summary")).preference_bonus == 3
    assert rank(replace(qualified, evidence_status="headline_only")).preference_bonus == 0
    assert rank(replace(qualified, category="general")).preference_bonus == 0
    assert rank(replace(qualified, freshness={"decision": "exclude"})).preference_bonus == 0


def test_a16z_has_no_seat_but_fresh_supported_analysis_can_win():
    rows = [article(sid, score=60 + index) for index, sid in enumerate(["nature", "bloomberg", "ieee_spectrum"])]
    weak = article("a16z", content_kind="analysis", score=10)
    assert research.select_balanced(rows + [weak], 3) == research.select_balanced(rows, 3)
    assert research.select_balanced(rows + [replace(weak, score=100)], 3)[0].source_id == "a16z"
    assert research.select_balanced(rows + [replace(weak, score=100, freshness={"decision": "exclude"})], 3) == research.select_balanced(rows, 3)
    assert research.select_balanced(rows + [replace(weak, score=100, evidence_status="feed_summary")], 3) == research.select_balanced(rows, 3)


@pytest.mark.parametrize("source", ["a16z", "bloomberg", "jiqizhixin_daily", "nature"])
def test_new_publication_date_cannot_redate_an_old_event(source):
    row = article(source, evidence_text=OLD)
    with pytest.raises(ValueError, match="outside"):
        validate(row, assessment(row, development_quote=OLD, date_quote=OLD, development_date="2026-09-15"))
    with pytest.raises(ValueError, match="date quotation"):
        validate(row, assessment(row, development_quote=OLD, date_quote=OLD))
    with pytest.raises(ValueError, match="Publication date"):
        validate(row, assessment(row, development_quote=OLD, date_basis="publication", development_kind="new_event"))
    with pytest.raises(ValueError, match="date quotation"):
        text = QUOTE.replace("2026", "2025")
        validate(replace(row, evidence_text=text), assessment(row, development_quote=text, date_quote=text))


@pytest.mark.parametrize("source", ["bloomberg", "jiqizhixin_daily", "nature"])
def test_same_event_allows_a_verbatim_dated_followup_not_a_renamed_old_fact(source):
    row = article(source)
    history = [previous(row)]
    with pytest.raises(ValueError, match="omitted"):
        validate(row, assessment(row), history)
    record = assessment(row, matched_history_keys=["prior"], new_since_history="A reproducible evaluation, rather than the launch.")
    assert validate(row, record, history)["decision"] == "include"
    with pytest.raises(ValueError, match="already covered"):
        validate(row, record, [previous(row, script=QUOTE)])
    with pytest.raises(ValueError, match="specific evidenced change"):
        validate(row, {**record, "new_since_history": ""}, history)


def test_tracking_and_source_aliases_cannot_hide_same_article():
    row = article(url="https://www.nature.test/orbit/?utm_source=new&ref=x")
    assert freshness.known_matches(row, [previous(url="http://nature.test/orbit")]) == {"prior"}


def test_invented_quote_and_unrelated_date_cannot_pass():
    row = article()
    with pytest.raises(ValueError, match="verbatim"):
        validate(row, assessment(row, development_quote="Lab obtained one hundred percent accuracy in a new trial."))
    with pytest.raises(ValueError, match="date quotation"):
        validate(row, assessment(row, date_quote="On September 20, the company changed its name."))


@pytest.mark.parametrize("text,basis", [
    ("实验室在 9 月 20 日发布了 Orbit 新评估，提供了可复现的实验结果。", "explicit"),
    ("Today Lab published a new evaluation of Orbit with reproducible results.", "relative"),
])
def test_new_developments_can_use_chinese_or_relative_source_dates(text, basis):
    row = article(evidence_text=text)
    assert validate(row, assessment(row, development_quote=text, date_quote=text, date_basis=basis))["decision"] == "include"


def test_editorial_assessment_compares_all_history_and_repairs_malformed_response(tmp_path):
    row = article("jiqizhixin_daily", url="https://newsite.test/new-url", title="A renamed retrospective")
    history = [previous(title="另一种语言的旧发布报道")]
    rejected = {"id": row.id, "decision": "exclude", "reason": "Rewritten old event without a new development"}
    chat = AsyncMock(side_effect=['{"assessments": []}', json.dumps({"assessments": [rejected]})])
    with patch.object(freshness, "_resolve_provider", AsyncMock(return_value=("e", "m", "k"))), patch.object(freshness, "_chat", chat):
        asyncio.run(freshness.assess_candidates([row], history, now=NOW, window_hours=36, output_dir=tmp_path))
    assert row.freshness["decision"] == "exclude"
    assert chat.await_count == 2
    assert "across languages" in chat.await_args.args[0]
    assert history[0]["title"] in chat.await_args.args[1]
    assert row.url in chat.await_args.args[1]
    assert len(list((tmp_path / "research/freshness").glob("*.txt"))) == 2


def test_assessment_exhaustion_never_falls_back_to_publication_date(tmp_path):
    row = article()
    chat = AsyncMock(return_value="not JSON")
    with patch.object(freshness, "_resolve_provider", AsyncMock(return_value=("e", "m", "k"))), patch.object(freshness, "_chat", chat):
        with pytest.raises(RuntimeError, match="exhausted bounded repairs"):
            asyncio.run(freshness.assess_candidates([row], [], now=NOW, window_hours=36, output_dir=tmp_path))
    assert chat.await_count == 3
    assert not row.freshness


@pytest.fixture
def history_store(tmp_path, monkeypatch):
    db_path = tmp_path / "tasks.db"
    monkeypatch.setattr(config, "DB_PATH", db_path)
    with sqlite3.connect(db_path) as db:
        db.execute("CREATE TABLE tasks (id TEXT, source_type TEXT, status TEXT, source_url TEXT, config_json TEXT, output_dir TEXT, script_path TEXT, updated_at TEXT)")

    def add(task_id, edition="2026-09-19", status="complete", test=False, script=OLD, broken=False, updated_at=None):
        directory = tmp_path / task_id
        (directory / "research").mkdir(parents=True)
        row = article()
        if not broken:
            (directory / "research/dossier.json").write_text(json.dumps({"edition_date": edition, "selected": [row.as_dict()]}))
        (directory / "script.txt").write_text("Opening\n" + script + "\nClosing")
        with sqlite3.connect(db_path) as db:
            db.execute("INSERT INTO tasks VALUES (?,?,?,?,?,?,?,?)", (
                task_id, "news_daily", status, json.dumps({"edition_date": edition, "test_mode": test}),
                "{}", str(directory), str(directory / "script.txt"), updated_at or NOW.isoformat(),
            ))
        return directory
    return add


def test_history_excludes_failed_running_tests_current_and_same_edition(history_store):
    history_store("good")
    history_store("same-previous-edition-copy")
    history_store("failed", status="failed", broken=True)
    history_store("running", status="reviewing", broken=True)
    history_store("test", test=True, broken=True)
    history_store("current", broken=True)
    history_store("same-edition", edition="2026-09-20", broken=True)
    history_store("future-edition", edition="2026-09-21", broken=True)
    history_store("outside", edition="2026-09-05", broken=True)
    history_store("not-yet-completed", updated_at=(NOW + timedelta(seconds=1)).isoformat(), broken=True)
    rows = asyncio.run(freshness.load_history(EDITION, "current", now=NOW))
    assert len(rows) == 1
    assert rows[0]["script"] == OLD
    assert rows == asyncio.run(freshness.load_history(EDITION, "current", now=NOW))


def test_missing_completed_history_fails_closed(history_store):
    history_store("lost-evidence", broken=True)
    with pytest.raises(RuntimeError, match="unreadable"):
        asyncio.run(freshness.load_history(EDITION, "current", now=NOW))


def test_retry_reuses_same_history_but_completion_race_invalidates_review(history_store):
    directory = history_store("current", edition="2026-09-20", status="publishing")
    path = directory / "research/dossier.json"
    raw = json.loads(path.read_text())
    raw.update(freshness_policy_version=1, history=[], generated_at=NOW.isoformat(), window_hours=36)
    path.write_text(json.dumps(raw))
    (directory / "review").mkdir()
    (directory / "review/fact_check_report.json").write_text(json.dumps({
        "passed": True, "freshness_review_key": freshness.review_binding(raw),
        "final_contract": {"script_sha256": hashlib.sha256((directory / "script.txt").read_bytes()).hexdigest()},
    }))
    task = SimpleNamespace(id="current", source_type="news_daily", output_dir=str(directory), script_path=None)

    async def check():
        async with freshness.completion_guard(task):
            return True
    assert asyncio.run(check())
    history_store("concurrent-same-edition", edition="2026-09-20")
    assert asyncio.run(check())
    history_store("new-completed-prior-edition")
    with pytest.raises(RuntimeError, match="history changed"):
        asyncio.run(check())
    (directory / "script.txt").write_text("An unreviewed revised script")
    with pytest.raises(RuntimeError, match="changed after approval"):
        asyncio.run(check())


def test_completion_guard_serializes_concurrent_workers(history_store):
    directory = history_store("current", edition="2026-09-20", status="publishing")
    task = SimpleNamespace(id="current", source_type="news_daily", output_dir=str(directory))

    async def check():
        events = []
        async def one():
            async with freshness.completion_guard(task):
                events.append("one-enter")
                await asyncio.sleep(0.05)
                events.append("one-exit")
        async def two():
            await asyncio.sleep(0.01)
            async with freshness.completion_guard(task):
                events.append("two-enter")
        await asyncio.gather(one(), two())
        return events
    assert asyncio.run(check()) == ["one-enter", "one-exit", "two-enter"]


def test_evidence_binding_and_independent_review_include_new_development():
    row = article()
    row.freshness = validate(row, assessment(row))
    dossier = research.ResearchDossier(EDITION.isoformat(), NOW.isoformat(), 36, [row], [row], [], freshness_policy_version=1)
    assert not freshness.eligibility_failures(dossier)
    script = "Opening\nNature reports that Acme launched Orbit on September 15.\nClosing"
    prompt = review._batch_review_prompt(script, dossier, EDITION, [1])
    assert QUOTE in prompt
    assert '"development_date": "2026-09-20"' in prompt
    assert "if the qualifying new development is missing" in prompt
    assert "Compare completed-history spoken facts" in prompt
    first_key = review._story_evidence_key(script, dossier, EDITION, 1)[0]
    dossier.history = [previous()]
    assert review._story_evidence_key(script, dossier, EDITION, 1)[0] != first_key
    assert freshness.eligibility_failures(dossier)
    dossier.history = []
    row.evidence_text = "Changed source content"
    assert freshness.eligibility_failures(dossier)


def test_missing_update_uses_existing_bounded_claim_correction(tmp_path):
    row = article(source_name="Nature")
    data = research.ResearchDossier(EDITION.isoformat(), NOW.isoformat(), 36, [row], [row], [])
    row.freshness = validate(row, assessment(row))
    opening = scriptwriter.morning_opening(EDITION)
    old = "Nature reports that Acme launched Orbit on September 15."
    new = "Nature reports that Lab published a new evaluation of Orbit on September 20, with reproducible results."
    original = "\n".join([opening, old, "Goodbye."])
    corrected = "\n".join([opening, new, "Goodbye."])
    issue = review._single_line_payload("A", 1, claim_ids=["1.0"])
    web = AsyncMock(side_effect=[(issue, "W1A@1.0", "", "chatgpt"), ({"issues": []}, "W1P", "", "chatgpt")])
    with patch.object(review, "_web_story_review", web), patch.object(review, "revise_daily_script", AsyncMock(return_value=corrected)) as revise:
        result = asyncio.run(review._review_daily_script(
            original, data, EDITION, tmp_path, language="en", closing_remarks="Goodbye.",
            ai_endpoint=None, ai_model=None, provider_id=None, review_session_namespace="test",
        ))
    assert result.script == corrected
    assert revise.await_count == 1
    assert "qualifying new development" in revise.await_args.args[2][0]["correction"]
    assert result.report["correction_count"] == 1


def test_review_resume_refreshes_changed_history_and_keeps_original_dossier(history_store):
    directory = history_store("current", edition="2026-09-20", status="failed")
    row = article()
    row.freshness = validate(row, assessment(row))
    data = research.ResearchDossier(EDITION.isoformat(), NOW.isoformat(), 36, [row], [row], [], freshness_policy_version=1)
    path = directory / "research/dossier.json"
    original = json.dumps(data.as_dict())
    path.write_text(original)
    history_store("new-prior")

    async def assess(rows, history, **kwargs):
        for item in rows:
            item.freshness = validate(item, assessment(item, matched_history_keys=[history[0]["key"]],
                new_since_history="A reproducible evaluation was added after the launch."), history)
    with patch.object(freshness, "assess_candidates", AsyncMock(side_effect=assess)) as judge:
        asyncio.run(freshness.refresh_review_history(data, directory))
        asyncio.run(freshness.refresh_review_history(data, directory))
    assert judge.await_count == 1  # Retry with the same snapshot doesn't re-adjudicate itself.
    assert len(data.history) == 1
    assert not freshness.eligibility_failures(data)
    assert next(path.parent.glob("dossier-before-refresh-*.json")).read_text() == original
    assert json.loads(path.read_text())["history"] == data.history


def test_legacy_review_resume_cannot_restore_seven_day_exception(tmp_path):
    row = article("a16z", content_kind="analysis", lookback_hours=168,
                  published_at=(NOW - timedelta(days=5)).isoformat())
    data = research.ResearchDossier(EDITION.isoformat(), NOW.isoformat(), 36, [row], [row], [])
    with patch.object(freshness, "assess_candidates", AsyncMock()) as judge:
        with pytest.raises(RuntimeError, match="stale/undated"):
            asyncio.run(freshness.refresh_review_history(data, tmp_path, force=True))
    judge.assert_not_awaited()


def test_failed_refresh_does_not_overwrite_original_dossier(history_store):
    directory = history_store("current", edition="2026-09-20", status="failed")
    row = article()
    data = research.ResearchDossier(EDITION.isoformat(), NOW.isoformat(), 36, [row], [row], [], freshness_policy_version=1)
    path = directory / "research/dossier.json"
    original = path.read_bytes()

    async def reject(rows, *args, **kwargs):
        rows[0].freshness = {"decision": "exclude", "reason": "No new fact"}
    with patch.object(freshness, "assess_candidates", AsyncMock(side_effect=reject)):
        with pytest.raises(RuntimeError, match="lack a new development"):
            asyncio.run(freshness.refresh_review_history(data, directory))
    assert path.read_bytes() == original


def test_research_filters_old_sources_then_semantic_repeats_before_selection(tmp_path):
    sources = [s for s in load_source_catalog() if s.id in {"a16z", "bloomberg", "nature", "jiqizhixin_daily"}]
    rows = [article(s.id, source_name=s.name, language=s.language, content_kind=s.content_kind,
                    title=f"{s.id} reports a separate reproducible evaluation", category="ai_models") for s in sources]
    old = next(a for a in rows if a.source_id == "a16z")
    old.published_at = (NOW - timedelta(days=5)).isoformat()
    old.lookback_hours = 168
    duplicate = replace(next(a for a in rows if a.source_id == "nature"), id="reprint", title="Rewritten launch retrospective",
                        url="https://nature.test/a-new-url", evidence_text=OLD)
    rows.append(duplicate)
    history = [previous(duplicate)]
    batches = []

    async def fetch(client, source):
        matches = [a for a in rows if a.source_id == source.id]
        return matches, research.SourceFetch(source.id, source.name, source.homepage, True, 200, len(matches))

    async def chat(system, content, *args, **kwargs):
        payload = json.loads(content)
        batches.extend(a["id"] for a in payload["candidates"])
        assert payload["completed_history"] == history
        decisions = []
        for candidate in payload["candidates"]:
            row = next(a for a in rows if a.id == candidate["id"])
            if row.id == "reprint":
                decisions.append({"id": row.id, "decision": "exclude", "reason": "Old launch, no substantive update"})
            else:
                decisions.append(assessment(row))
        return json.dumps({"assessments": decisions})

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return NOW

    with (
        patch.object(research, "datetime", Clock),
        patch.object(research, "enabled_sources", return_value=sources),
        patch.object(research, "fetch_source", AsyncMock(side_effect=fetch)),
        patch.object(research, "hydrate_evidence", AsyncMock()),
        patch.object(freshness, "load_history", AsyncMock(return_value=history)),
        patch.object(freshness, "_resolve_provider", AsyncMock(return_value=("e", "m", "k"))),
        patch.object(freshness, "_chat", AsyncMock(side_effect=chat)),
    ):
        result = asyncio.run(research.run_research(EDITION, tmp_path, max_stories=3))
    assert len(result.selected) == 3
    assert {a.source_id for a in result.selected} == {"bloomberg", "nature", "jiqizhixin_daily"}
    assert "a16z" not in batches
    assert "reprint" in batches
    assert not freshness.eligibility_failures(result)
    assert next(a for a in result.selected if a.source_id == "bloomberg").preference_bonus == 6
    assert next(a for a in result.candidates if a.id == "reprint").freshness["decision"] == "exclude"

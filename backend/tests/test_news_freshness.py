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


@pytest.mark.parametrize("published,zone,local_day", [
    ("2026-09-19T23:01:57+00:00", "Asia/Singapore", "2026-09-20"),
    ("2026-09-20T02:01:57+00:00", "America/Los_Angeles", "2026-09-19"),
    ("2026-09-20T07:01:57+08:00", "UTC", "2026-09-19"),
])
def test_publication_reporting_uses_edition_day_without_redating_event(published, zone, local_day):
    row = article(published_at=published)
    for model_day in {published[:10], local_day}:
        record = assessment(row, development_kind="new_reporting", date_basis="publication",
                            development_date=model_day, date_quote="")
        result = freshness.validate_assessment(row, record, [], now=NOW,
                                              window_hours=36, timezone_name=zone)
        assert result["development_date"] == local_day
        assert result["event_date"] == "2026-09-15"
        assert record["development_date"] == model_day
        assert result["evidence_sha256"] == freshness.fingerprint(freshness.evidence_payload(row))
    with pytest.raises(ValueError, match="Publication date supports only"):
        freshness.validate_assessment(row, {**record, "development_kind": "new_event"}, [],
                                      now=NOW, window_hours=36, timezone_name=zone)


def test_unrelated_publication_day_gets_precise_repair_feedback():
    row = article(published_at="2026-09-19T23:01:57+00:00")
    record = assessment(row, development_kind="new_reporting", date_basis="publication",
                        development_date="2026-09-18")
    with pytest.raises(ValueError, match="development_date must be 2026-09-20 in Asia/Singapore"):
        freshness.validate_assessment(row, record, [], now=NOW, window_hours=36,
                                      timezone_name="Asia/Singapore")


@pytest.mark.parametrize("published", [NOW - timedelta(hours=36, seconds=1), NOW + timedelta(seconds=1)])
def test_publication_date_normalization_cannot_admit_out_of_window_timestamp(published):
    row = article(published_at=published.isoformat())
    record = assessment(row, development_kind="new_reporting", date_basis="publication",
                        development_date=published.date().isoformat())
    with pytest.raises(ValueError, match="publication timestamp falls outside"):
        validate(row, record)


@pytest.mark.parametrize("changes", [[], [""], [" "], ["New finding", 42], [{"fact": "New finding"}], None])
def test_history_fact_format_normalization_does_not_accept_missing_or_invalid_facts(changes):
    row = article()
    record = assessment(row, matched_history_keys=["prior"], new_since_history=changes)
    with pytest.raises(ValueError, match="specific evidenced change"):
        validate(row, record, [previous(row)])


def test_history_fact_list_keeps_all_facts_and_duplicate_guard():
    row = article()
    facts = [" A reproducible evaluation was published. ", "The new results include error rates."]
    record = assessment(row, matched_history_keys=["prior"], new_since_history=facts)
    result = validate(row, record, [previous(row)])
    assert result["new_since_history"] == "A reproducible evaluation was published.\nThe new results include error rates."
    assert record["new_since_history"] == facts
    with pytest.raises(ValueError, match="already covered"):
        validate(row, record, [previous(row, script=QUOTE)])


def test_cross_timezone_reporting_and_history_list_need_no_semantic_repair(tmp_path):
    row = article("bbc", published_at="2026-09-19T23:01:57+00:00")
    record = assessment(row, development_kind="new_reporting", date_basis="publication",
                        development_date="2026-09-19", date_quote="", matched_history_keys=["prior"],
                        new_since_history=["New interviews and evaluation results absent from prior coverage."])
    chat = AsyncMock(return_value=json.dumps({"assessments": [record]}))
    with patch.object(freshness, "_resolve_provider", AsyncMock(return_value=("e", "m", "k"))), patch.object(freshness, "_chat", chat):
        asyncio.run(freshness.assess_candidates([row], [previous(row)], now=NOW,
                                              window_hours=36, output_dir=tmp_path,
                                              timezone_name="Asia/Singapore"))
    assert chat.await_count == 1
    assert chat.await_args.kwargs["max_tokens"] == freshness.ASSESSMENT_MAX_OUTPUT_TOKENS
    request = json.loads(chat.await_args.args[1])
    candidate = request["candidates"][0]
    assert candidate["published_at"] == row.published_at
    assert candidate["published_at_local"] == "2026-09-20T07:01:57+08:00"
    assert candidate["publication_date"] == "2026-09-20"
    assert row.freshness["decision"] == "include"
    assert row.freshness["development_date"] == "2026-09-20"
    assert row.freshness["new_since_history"] == record["new_since_history"][0]
    saved = next((tmp_path / "research/freshness").glob("*-1.validation.json"))
    assert json.loads(saved.read_text()) == {"accepted_ids": [row.id], "errors": []}


def test_tracking_and_source_aliases_cannot_hide_same_article():
    row = article(url="https://www.nature.test/orbit/?utm_source=new&ref=x")
    assert freshness.known_matches(row, [previous(url="http://nature.test/orbit")]) == {"prior"}


def test_review_feedback_cannot_supply_missing_development_evidence(tmp_path):
    row = article()
    outside_quote = "Outside source reports a different result not present in captured evidence."
    feedback = [{"reason": "The proposed event is old", "correction": outside_quote}]
    record = assessment(row, development_quote=outside_quote)
    chat = AsyncMock(return_value=json.dumps({"assessments": [record]}))
    with (
        patch.object(freshness, "_resolve_provider", AsyncMock(return_value=("e", "m", "k"))),
        patch.object(freshness, "_chat", chat),
    ):
        with pytest.raises(RuntimeError, match="quote the captured evidence"):
            asyncio.run(freshness.assess_candidates(
                [row], [], now=NOW, window_hours=36, output_dir=tmp_path,
                review_feedback=feedback,
            ))
    request = json.loads(chat.await_args.args[1])
    assert request["independent_review_feedback"] == feedback
    assert request["candidates"][0]["evidence_text"] == row.evidence_text
    assert chat.await_count == freshness.ASSESSMENT_ATTEMPTS


def test_invented_quote_and_unrelated_date_cannot_pass():
    row = article()
    with pytest.raises(ValueError, match="verbatim"):
        validate(row, assessment(row, development_quote="Lab obtained one hundred percent accuracy in a new trial."))
    with pytest.raises(ValueError, match="date quotation"):
        validate(row, assessment(row, date_quote="On September 20, the company changed its name."))


def test_typographic_quote_variants_are_bound_back_to_exact_captured_text():
    captured = "On September 20, 2026, Lab’s new evaluation showed “reproducible results” for Orbit."
    row = article(evidence_text=captured)
    generated = captured.replace("’", "'").replace("“", '"').replace("”", '"').lower()
    generated = generated.replace("new evaluation", "new\n  evaluation")
    record = assessment(row, development_quote=generated, date_quote=generated)
    result = validate(row, record)
    assert result["development_quote"] == captured
    assert result["date_quote"] == captured
    assert record["development_quote"] == generated
    with pytest.raises(ValueError, match="already covered"):
        validate(row, {**record, "matched_history_keys": ["prior"], "new_since_history": "Claimed update"},
                 [previous(row, script=captured)])


@pytest.mark.parametrize("generated", [
    "Lab publishes reproducible Orbit evaluation — " + QUOTE,
    "On September 20, 2026, Lab published a new evaluation of Orbit with conclusive results.",
    "On September 20, 2026, Lab published … with reproducible results.",
    QUOTE.replace("September 20", "September 21"),
])
def test_quote_matching_still_rejects_stitched_paraphrased_or_changed_evidence(generated):
    row = article()
    with pytest.raises(ValueError, match="verbatim"):
        validate(row, assessment(row, development_quote=generated))


def test_quotes_cannot_borrow_summary_when_captured_article_text_exists():
    row = article(evidence_text=OLD, summary=QUOTE)
    with pytest.raises(ValueError, match="verbatim"):
        validate(row, assessment(row))
    assert validate(replace(row, evidence_text=""), assessment(row))["development_quote"] == QUOTE


@pytest.mark.parametrize("text,basis", [
    ("实验室在 9 月 20 日发布了 Orbit 新评估，提供了可复现的实验结果。", "explicit"),
    ("Today Lab published a new evaluation of Orbit with reproducible results.", "relative"),
])
def test_new_developments_can_use_chinese_or_relative_source_dates(text, basis):
    row = article(evidence_text=text)
    assert validate(row, assessment(row, development_quote=text, date_quote=text, date_basis=basis))["decision"] == "include"


@pytest.mark.parametrize("day", ["on Monday", "Monday", "周一", "星期一"])
@pytest.mark.parametrize("published", ["2026-09-21T11:10:55+00:00", "2026-09-20T23:10:55+00:00"])
def test_weekday_development_uses_captured_publication_date(day, published):
    now = datetime(2026, 9, 22, 3, tzinfo=timezone.utc)
    text = f"The lab published its new reproducible evaluation {day}."
    row = article(published_at=published, evidence_text=text)
    record = assessment(row, development_date="2026-09-21", development_kind="new_event",
                        development_quote=text, date_quote=text, date_basis="relative")
    result = freshness.validate_assessment(row, record, [], now=now, window_hours=36,
                                          timezone_name="Asia/Singapore")
    assert result["date_basis"] == "relative"
    assert result["development_date"] == "2026-09-21"
    assert result["date_quote"] == text


@pytest.mark.parametrize("day", ["last Monday", "next Monday", "previous Monday",
                                "Monday last week", "Monday next week", "上周一", "下星期一",
                                "Monday, September 14, 2026", "周一（9月14日）",
                                "Monday in January", "Monday in 2025", "this past Monday", "on Tuesday",
                                "on Monday a week ago", "every Monday", "each Monday"])
def test_weekday_cannot_redate_past_future_or_conflicting_source_date(day):
    now = datetime(2026, 9, 21, 12, tzinfo=timezone.utc)
    text = f"The lab published its new reproducible evaluation {day}."
    row = article(published_at="2026-09-21T11:10:55+00:00", evidence_text=text)
    record = assessment(row, development_date="2026-09-21", development_kind="new_event",
                        development_quote=text, date_quote=text, date_basis="relative")
    with pytest.raises(ValueError, match="date quotation"):
        freshness.validate_assessment(row, record, [], now=now, window_hours=36,
                                      timezone_name="Asia/Singapore")


def test_weekday_on_cutoff_day_still_needs_precise_timing():
    text = "On Friday the lab published its new reproducible evaluation."
    row = article(evidence_text=text)
    record = assessment(row, development_date="2026-09-18", development_quote=text,
                        date_quote=text, date_basis="relative")
    with pytest.raises(ValueError, match="cutoff day"):
        validate(row, record)


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
    repair = json.loads(chat.await_args.args[1])["repair"]
    assert repair["previous_response"] == '{"assessments": []}'
    assert repair["validation_errors"] == [{"id": None, "error":
        f"Missing, duplicate or unexpected candidate IDs; expected=['{row.id}'], received=[]"}]
    assert len(list((tmp_path / "research/freshness").glob("*.txt"))) == 2


def test_repairs_report_each_candidate_failure_and_keep_valid_decisions(tmp_path):
    valid, bad_date, bad_quote = [article(name) for name in ("valid", "bad-date", "bad-quote")]
    first = json.dumps({"assessments": [
        assessment(valid),
        assessment(bad_date, development_kind="new_event", date_basis="publication"),
        assessment(bad_quote, development_quote=bad_quote.title + " — " + QUOTE),
    ]})
    second = json.dumps({"assessments": [
        assessment(bad_date),
        {"id": bad_quote.id, "decision": "exclude", "reason": "The captured evidence cannot support the claimed update"},
    ]})
    chat = AsyncMock(side_effect=[first, second])
    logs = []
    with patch.object(freshness, "_resolve_provider", AsyncMock(return_value=("e", "m", "k"))), patch.object(freshness, "_chat", chat):
        asyncio.run(freshness.assess_candidates([valid, bad_date, bad_quote], [], now=NOW,
                                              window_hours=36, output_dir=tmp_path, log=logs.append))
    request = json.loads(chat.await_args.args[1])
    assert [a["id"] for a in request["candidates"]] == [bad_date.id, bad_quote.id]
    assert request["repair"]["previous_response"] == first
    errors = request["repair"]["validation_errors"]
    assert [e["id"] for e in errors] == [bad_date.id, bad_quote.id]
    assert "Publication date" in errors[0]["error"]
    assert "verbatim" in errors[1]["error"]
    assert all(a.freshness["decision"] == "include" for a in (valid, bad_date))
    assert bad_quote.freshness["decision"] == "exclude"
    assert bad_date.id in logs[0] and bad_quote.id in logs[0]
    audit = tmp_path / "research/freshness"
    saved = json.loads(next(audit.glob("*-1.validation.json")).read_text())
    assert saved == {"accepted_ids": [valid.id], "errors": errors}
    saved_request = json.loads(next(audit.glob("*-2.request.json")).read_text())
    assert saved_request == request


def test_failed_repairs_remain_bounded_without_committing_partial_batch(tmp_path):
    good, bad = article("good"), article("bad")
    initial = json.dumps({"assessments": [assessment(good), assessment(bad, development_quote="invented")]})
    invalid = json.dumps({"assessments": [assessment(bad, development_quote="invented")]})
    chat = AsyncMock(side_effect=[initial, "not JSON", invalid])
    with patch.object(freshness, "_resolve_provider", AsyncMock(return_value=("e", "m", "k"))), patch.object(freshness, "_chat", chat):
        with pytest.raises(RuntimeError, match="bad: The new development must quote"):
            asyncio.run(freshness.assess_candidates([good, bad], [], now=NOW, window_hours=36, output_dir=tmp_path))
    assert chat.await_count == 3
    assert not good.freshness and not bad.freshness
    assert all([a["id"] for a in json.loads(call.args[1])["candidates"]] == [bad.id]
               for call in chat.await_args_list[1:])


@pytest.mark.parametrize("duplicate", [False, True])
def test_partial_response_repairs_missing_or_duplicate_ids_and_keeps_valid_decisions(tmp_path, duplicate):
    good, bad = article("good"), article("bad")
    first_rows = [assessment(good)]
    if duplicate:
        first_rows.extend([assessment(bad), {"id": bad.id, "decision": "exclude", "reason": "Conflicting duplicate"}])
    chat = AsyncMock(side_effect=[json.dumps({"assessments": first_rows}),
                                 json.dumps({"assessments": [assessment(bad)]})])
    with (patch.object(freshness, "_resolve_provider", AsyncMock(return_value=("e", "m", "k"))),
          patch.object(freshness, "_chat", chat)):
        asyncio.run(freshness.assess_candidates([good, bad], [], now=NOW, window_hours=36, output_dir=tmp_path))
    request = json.loads(chat.await_args.args[1])
    assert [a["id"] for a in request["candidates"]] == [bad.id]
    assert request["repair"]["validation_errors"][0]["id"] == bad.id
    assert request["repair"]["validation_errors"][0]["error"].startswith("Duplicate" if duplicate else "Missing")
    assert good.freshness["decision"] == bad.freshness["decision"] == "include"
    assert chat.await_count == 2


def test_assessment_exhaustion_never_falls_back_to_publication_date(tmp_path):
    row = article()
    chat = AsyncMock(return_value="not JSON")
    with patch.object(freshness, "_resolve_provider", AsyncMock(return_value=("e", "m", "k"))), patch.object(freshness, "_chat", chat):
        with pytest.raises(RuntimeError, match="exhausted bounded repairs"):
            asyncio.run(freshness.assess_candidates([row], [], now=NOW, window_hours=36, output_dir=tmp_path))
    assert chat.await_count == 3
    assert not row.freshness


def test_discovery_excludes_exhausted_candidate_and_preserves_accepted_batch(tmp_path):
    good, bad, later = [article(name) for name in ("good", "bad", "later")]
    initial = json.dumps({"assessments": [assessment(good), assessment(bad, date_basis="publication")]})
    invalid = json.dumps({"assessments": [assessment(bad, date_basis="publication")]})
    chat = AsyncMock(side_effect=[initial, invalid, invalid,
                                 json.dumps({"assessments": [assessment(later)]})])
    logs = []
    with (patch.object(freshness, "ASSESSMENT_BATCH_SIZE", 2),
          patch.object(freshness, "_resolve_provider", AsyncMock(return_value=("e", "m", "k"))),
          patch.object(freshness, "_chat", chat)):
        asyncio.run(freshness.assess_candidates([good, bad, later], [], now=NOW, window_hours=36,
                                              output_dir=tmp_path, log=logs.append,
                                              exclude_invalid_candidates=True))
    assert chat.await_count == 4
    assert good.freshness["decision"] == later.freshness["decision"] == "include"
    assert bad.freshness["decision"] == "exclude"
    assert bad.freshness["exclusion_basis"] == "validation_exhausted"
    assert "Publication date supports only" in bad.freshness["validation_error"]
    assert "development_date" not in bad.freshness
    assert bad.freshness["evidence_sha256"] == freshness.fingerprint(freshness.evidence_payload(bad))
    assert any("excluded candidate bad" in line for line in logs)
    receipts = [json.loads(p.read_text()) for p in (tmp_path / "research/freshness").glob("*.json")
                if "assessments" in json.loads(p.read_text())]
    assert any(bad.freshness in receipt["assessments"] for receipt in receipts)


@pytest.mark.parametrize("failure", ["not JSON", '{"assessments": []}',
                                     '{"assessments": [{"id": "unexpected"}]}', RuntimeError("provider 429")])
def test_discovery_does_not_hide_response_or_provider_failures(tmp_path, failure):
    row = article()
    chat = AsyncMock(side_effect=failure) if isinstance(failure, Exception) else AsyncMock(return_value=failure)
    with (patch.object(freshness, "_resolve_provider", AsyncMock(return_value=("e", "m", "k"))),
          patch.object(freshness, "_chat", chat)):
        with pytest.raises(RuntimeError, match="provider 429|exhausted bounded repairs"):
            asyncio.run(freshness.assess_candidates([row], [], now=NOW, window_hours=36,
                                                  output_dir=tmp_path, exclude_invalid_candidates=True))
    assert not row.freshness
    assert chat.await_count == (1 if isinstance(failure, Exception) else 3)


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


@pytest.mark.parametrize("token,claim_ids", [
    ("W1A@1.0", ["1.0"]),
    ("W1AF@1.0,1.1", ["1.0", "1.1"]),
])
def test_missing_update_uses_existing_bounded_claim_correction(tmp_path, token, claim_ids):
    row = article(source_name="Nature")
    data = research.ResearchDossier(EDITION.isoformat(), NOW.isoformat(), 36, [row], [row], [])
    row.freshness = validate(row, assessment(row))
    opening = scriptwriter.morning_opening(EDITION)
    old = "Nature reports that Acme launched Orbit on September 15."
    new = "Nature reports that Lab published a new evaluation of Orbit on September 20, with reproducible results."
    original = "\n".join([opening, old, "Goodbye."])
    corrected = "\n".join([opening, new, "Goodbye."])
    issue = review._batch_payload(
        token, [1],
        claim_catalog=review._claim_catalog(original, data), require_web=True,
    )
    web = AsyncMock(side_effect=[(issue, token, "", "chatgpt"), ({"issues": []}, "W1P", "", "chatgpt")])
    with patch.object(review, "_web_story_review", web), patch.object(review, "revise_daily_script", AsyncMock(return_value=corrected)) as revise:
        result = asyncio.run(review._review_daily_script(
            original, data, EDITION, tmp_path, language="en", closing_remarks="Goodbye.",
            ai_endpoint=None, ai_model=None, provider_id=None, review_session_namespace="test",
        ))
    assert result.script == corrected
    assert revise.await_count == 1
    assert "qualifying new development" in revise.await_args.args[2][0]["correction"]
    assert revise.await_args.args[2][0]["claim_ids"] == claim_ids
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


@pytest.mark.parametrize("usable_replacements", [True, False])
def test_research_replaces_exhausted_candidates_without_weakening_minimum(tmp_path, usable_replacements):
    sources = [s for s in load_source_catalog() if s.id in {"nature", "bloomberg", "jiqizhixin_daily"}]
    rows = [article(sources[i % 3].id, id=f"candidate-{i}", title=f"Independent story {i}",
                    url=f"https://example.test/story/{i}") for i in range(27)]
    by_id = {row.id: row for row in rows}

    async def fetch(client, source):
        matches = [a for a in rows if a.source_id == source.id]
        return matches, research.SourceFetch(source.id, source.name, source.homepage, True, 200, len(matches))

    async def chat(system, content, *args, **kwargs):
        decisions = []
        for candidate in json.loads(content)["candidates"]:
            row = by_id[candidate["id"]]
            valid = usable_replacements and row in rows[24:]
            decisions.append(assessment(row, development_quote=QUOTE if valid else "invented"))
        return json.dumps({"assessments": decisions})

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return NOW

    # Keep discovery order deterministic so the first hydration batch consists
    # entirely of invalid candidates, followed by three possible replacements.
    def rank(articles, *args):
        return sorted(articles, key=lambda a: int(a.id.split("-")[1]))

    with (
        patch.object(research, "datetime", Clock),
        patch.object(research, "enabled_sources", return_value=sources),
        patch.object(research, "fetch_source", AsyncMock(side_effect=fetch)),
        patch.object(research, "score_and_deduplicate", side_effect=rank),
        patch.object(research, "hydrate_evidence", AsyncMock()) as hydrate,
        patch.object(freshness, "load_history", AsyncMock(return_value=[])),
        patch.object(freshness, "_resolve_provider", AsyncMock(return_value=("e", "m", "k"))),
        patch.object(freshness, "_chat", AsyncMock(side_effect=chat)),
    ):
        if usable_replacements:
            result = asyncio.run(research.run_research(EDITION, tmp_path, max_stories=3))
            assert {a.id for a in result.selected} == {a.id for a in rows[24:]}
            assert not freshness.eligibility_failures(result)
        else:
            with pytest.raises(RuntimeError, match="only 0 usable stories"):
                asyncio.run(research.run_research(EDITION, tmp_path, max_stories=3))
    assert [len(call.args[1]) for call in hydrate.await_args_list] == [24, 3]
    assert all(a.freshness["decision"] == "exclude" for a in rows[:24])

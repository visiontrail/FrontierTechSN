import asyncio
import json
from datetime import date
from unittest.mock import AsyncMock, patch

import pytest

from backend.daily_news import research, review
from backend.pipeline.opencli import OpenCLIResult


@pytest.mark.parametrize("mutation", ["valid", "no_web", "missing_url", "wrong_claim", "missing_story"])
def test_freshness_diagnosis_is_sourced_context_never_approval(coverage_case, mutation):
    _, _, claims = coverage_case
    payload = {"web_verified": True, "findings": [{
        "story_number": 7, "claim_ids": ["7.0", "7.2"],
        "reason": "The event predates the research window.",
        "correction": "Cover the newly published analysis, not the old launch.",
        "source_urls": ["https://example.com/original"],
    }]}
    if mutation == "no_web":
        payload["web_verified"] = False
    elif mutation == "missing_url":
        payload["findings"][0]["source_urls"] = []
    elif mutation == "wrong_claim":
        payload["findings"][0]["claim_ids"] = ["8.2"]
    elif mutation == "missing_story":
        payload["findings"] = []
    if mutation == "valid":
        result = review._diagnostic_payload(json.dumps(payload), [7], claims)
        assert result["approved"] is False
    else:
        with pytest.raises(ValueError):
            review._diagnostic_payload(json.dumps(payload), [7], claims)


def test_freshness_failure_gets_diagnosis_once_and_still_stops_on_repeat(coverage_case, tmp_path):
    script, dossier, claims = coverage_case
    dossier.freshness_policy_version = 1
    issue = review._batch_payload("W7AF@7.0,7.2", [7], claim_catalog=claims, require_web=True)["issues"][0]
    async def verdict(_prompt, *, story_numbers, **_kwargs):
        issues = [issue] if 7 in story_numbers else []
        return {"approved": not issues, "issues": issues}, "[RAW]", "", "chatgpt"
    diagnose = AsyncMock()
    revise = AsyncMock(return_value=script)
    from backend.daily_news import freshness
    with (
        patch.object(freshness, "refresh_review_history", AsyncMock()),
        patch.object(review, "script_contract_report", return_value={"passed": True, "failures": []}),
        patch.object(review, "_web_story_review", AsyncMock(side_effect=verdict)),
        patch.object(review, "_diagnose_freshness", diagnose),
        patch.object(review, "revise_daily_script", revise),
        patch.object(review, "close_opencli_site_sessions", AsyncMock()),
    ):
        with pytest.raises(RuntimeError, match="same claim-level blocking verdict"):
            asyncio.run(review.review_daily_script(
                script, dossier, date(2026, 9, 21), tmp_path, language="en",
                closing_remarks="Goodbye.", ai_endpoint=None, ai_model=None, provider_id=None,
            ))
    assert diagnose.await_count == revise.await_count == 1
    report = json.loads((tmp_path / "review/fact_check_report.json").read_text())
    assert report["passed"] is False
    assert report["manual_review_required"] is True


def test_diagnostic_preserves_browser_rendered_newline_without_relaxing_web_gate(coverage_case):
    _, _, claims = coverage_case
    raw = '{"web_verified": true, "findings": [{"story_number": 7, "claim_ids": ["7.0"], "reason": "Old event.\nNew article.", "correction": "Cover analysis.", "source_urls": ["https://example.com/source"]}]}'
    result = review._diagnostic_payload(raw, [7], claims)
    assert result["findings"][0]["reason"] == "Old event.\nNew article."
    assert result["approved"] is False
    with pytest.raises(ValueError, match="live Web verification"):
        review._diagnostic_payload(raw.replace('true', 'false'), [7], claims)


@pytest.mark.parametrize("decision", ["include", "exclude"])
def test_diagnosis_reassesses_same_captured_evidence_and_invalidates_receipt(coverage_case, tmp_path, decision):
    from backend.daily_news import freshness
    script, dossier, claims = coverage_case
    issue = review._batch_payload("W7AF@7.0,7.2", [7], claim_catalog=claims, require_web=True)["issues"][0]
    finding = {"story_number": 7, "claim_ids": ["7.0", "7.2"],
               "reason": "Old event", "correction": "Cover the new analysis",
               "source_urls": ["https://example.com/original"]}
    (tmp_path / "review").mkdir()
    (tmp_path / "research").mkdir()
    (tmp_path / "research/dossier.json").write_text(json.dumps(dossier.as_dict()))
    old_key, _ = review._story_evidence_key(script, dossier, date(2026, 9, 21), 7)
    old_evidence = dossier.selected[6].evidence_text
    async def reassess(articles, *_args, **kwargs):
        assert articles == [dossier.selected[6]]
        assert kwargs["review_feedback"] == [finding]
        articles[0].freshness = {"decision": decision, "development": "New analysis"}
    async def run():
        await review._diagnose_freshness(
            script, dossier, date(2026, 9, 21), [issue], tmp_path, cycle=1,
            review_session_namespace="test", ai_endpoint=None, ai_model=None, provider_id=None, log=None,
        )
    with (
        patch.object(review, "_web_story_review", AsyncMock(return_value=(
            {"approved": False, "findings": [finding]}, "diagnostic raw", "", "chatgpt"))),
        patch.object(freshness, "assess_candidates", AsyncMock(side_effect=reassess)),
    ):
        if decision == "exclude":
            with pytest.raises(RuntimeError, match="no evidence-backed fresh development"):
                asyncio.run(run())
        else:
            asyncio.run(run())
    assert dossier.selected[6].evidence_text == old_evidence
    assert issue["claim_ids"] == ["7.0", "7.2"]
    assert issue["freshness_diagnosis"] == [finding]
    new_key, _ = review._story_evidence_key(script, dossier, date(2026, 9, 21), 7)
    assert new_key != old_key
    assert (tmp_path / "research/dossier-before-diagnosis-1.json").exists()


@pytest.fixture
def coverage_case():
    names = ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot", "Gemini", "Quadruped"]
    paragraphs = [
        f"{name} published a report. {name} describes an older development."
        for name in names
    ]
    articles = [
        research.NewsArticle(
            id=name, source_id=name, source_name=name, language="en",
            title=f"{name} report", url=f"https://example.com/{name}",
            published_at="2026-09-21T00:00:00+00:00", evidence_text=paragraph,
        )
        for name, paragraph in zip(names, paragraphs, strict=True)
    ]
    dossier = research.ResearchDossier(
        "2026-09-21", "2026-09-21T04:00:00+00:00", 36, articles, articles, [],
    )
    script = "\n".join(["Good morning.", *paragraphs, "Goodbye."])
    return script, dossier, review._claim_catalog(script, dossier)


def test_existing_background_still_has_independent_coverage_claim(coverage_case):
    script, dossier, claims = coverage_case
    assert "7.0" in claims[7]
    assert claims[7]["7.1"] == "Gemini published a report."
    assert claims[7]["7.2"] == "Gemini describes an older development."
    for prompt in (
        review._batch_review_prompt(script, dossier, date(2026, 9, 21), [7, 8]),
        review._review_prompt(script, dossier, date(2026, 9, 21), 7),
    ):
        assert "CLAIM 7.0:" in prompt
        assert "AND the exact affected sentence IDs" in prompt


@pytest.mark.parametrize("token", ["W7A@7.0;8P", "W7AF@7.0,7.2;8P"])
def test_coverage_failure_remains_blocking_with_exact_claims(coverage_case, token):
    _, _, claims = coverage_case
    payload = review._batch_payload(token, [7, 8], claim_catalog=claims, require_web=True)
    assert payload["approved"] is False
    assert len(payload["issues"]) == 1
    issue = payload["issues"][0]
    assert issue["severity"] == "blocking"
    assert issue["evidence_story_numbers"] == [7]
    expected = ["7.0", "7.2"] if "AF" in token else ["7.0"]
    assert issue["claim_ids"] == expected
    assert issue["claim_texts"] == [claims[7][claim_id] for claim_id in expected]


@pytest.mark.parametrize("token", [
    "W7AF@7.0,",                    # Truncated response.
    "W7AF@7.0,7.99;8P",             # Unknown sentence.
    "W7AF@7.0,8.2;8P",              # Cross-story sentence.
    "W7AF@8.0,7.2;8P",              # Cross-story coverage.
    "W7AF@7.2;8P",                  # A requires coverage ID.
    "W7F@7.0;8P",                   # Sentence failure cannot cite coverage.
    "W7A@7.0,7.2;8P",               # A alone cannot reject a sentence.
    "W7P@7.0;8P",                   # Passing verdict cannot cite failure IDs.
    "W7AF@7.0,7.2",                 # Missing story verdict.
    "W8P;7AF@7.0,7.2",              # Reordered stories.
    "7AF@7.0,7.2;8P",               # No live-search confirmation.
])
def test_coverage_support_does_not_accept_invalid_verdicts(coverage_case, token):
    with pytest.raises(ValueError):
        review._batch_payload(token, [7, 8], claim_catalog=coverage_case[2], require_web=True)


@pytest.mark.parametrize("single", [False, True])
def test_owned_partial_recovers_combined_coverage_failure_without_resubmission(coverage_case, single):
    _, _, claims = coverage_case
    prompt = ""
    target = "https://chatgpt.com/c/coverage-regression"
    final_token = "W7AF@7.0,7.2" + ("" if single else ";8P")

    async def command(args, **kwargs):
        nonlocal prompt
        if args[:2] == ["chatgpt", "model"]:
            rows = [{"Status": "Already selected", "Model": "Medium"}]
        elif args[:2] == ["chatgpt", "ask"]:
            prompt = args[2]
            rows = [{"response": "W7AF@7.0,", "conversationUrl": target}]
        else:
            assert args[:3] == ["chatgpt", "detail", target]
            rows = [
                {"Role": "User", "Text": prompt, "Generating": False},
                {"Role": "Assistant", "Text": final_token, "Generating": False},
            ]
        return OpenCLIResult(tuple(args), 0, json.dumps(rows), "")

    call = AsyncMock(side_effect=command)
    with patch.object(review, "run_opencli", call):
        payload, raw, url, provider = asyncio.run(review._web_story_review(
            "audit", claim_catalog=claims, log=None,
            **({"story_number": 7} if single else {"story_numbers": [7, 8]}),
        ))
    assert payload["approved"] is False
    assert payload["issues"][0]["claim_ids"] == ["7.0", "7.2"]
    assert url == target and provider == "chatgpt"
    assert "[CHATGPT RECOVERY]\n" + final_token in raw
    assert sum(c.args[0][:2] == ["chatgpt", "ask"] for c in call.await_args_list) == 1
    assert sum(c.args[0][:2] == ["chatgpt", "detail"] for c in call.await_args_list) == 1


def test_combined_failure_caches_only_the_passing_story(coverage_case, tmp_path):
    script, dossier, _ = coverage_case
    edition = date(2026, 9, 21)
    review._save_story_evidence(
        tmp_path, script, dossier, edition, [7, 8],
        "[CHATGPT RECOVERY]\nW7AF@7.0,7.2;8P",
    )
    assert review._load_story_evidence(tmp_path, script, dossier, edition, 7) is None
    assert review._load_story_evidence(tmp_path, script, dossier, edition, 8) is not None


@pytest.mark.parametrize("token", ["W7F", "W7F@7.0", "W7P@7.0"])
def test_single_story_keeps_strict_failure_claim_validation(coverage_case, token):
    prompt = ""

    async def command(args, **kwargs):
        nonlocal prompt
        if args[:2] == ["chatgpt", "model"]:
            rows = [{"Model": "Medium"}]
        elif args[:2] == ["chatgpt", "ask"]:
            prompt = args[2]
            rows = [{"response": token, "conversationUrl": "https://chatgpt.com/c/invalid"}]
        else:
            rows = [{"Role": "User", "Text": prompt}, {"Role": "Assistant", "Text": token}]
        return OpenCLIResult(tuple(args), 0, json.dumps(rows), "")

    with (
        patch.object(review, "run_opencli", AsyncMock(side_effect=command)),
        patch.object(review, "_target_recovery_poll_delays", return_value=()),
        pytest.raises(review._WebStoryReviewError),
    ):
        asyncio.run(review._web_story_review(
            "audit", story_number=7, log=None,
            claim_catalog=None if token == "W7F" else coverage_case[2],
        ))

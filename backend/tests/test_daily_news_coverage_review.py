import asyncio
import json
from datetime import date
from unittest.mock import AsyncMock, patch

import pytest

from backend.daily_news import research, review
from backend.pipeline.opencli import OpenCLIResult


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

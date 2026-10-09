import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest

from backend.daily_news import review
from backend.pipeline.opencli import OpenCLIError, OpenCLIResult


TARGET = "https://chatgpt.com/c/6ab86756-1f2c-83ec-ad4c-37a569fe8748"
CLAIMS = {1: {"1.1": "First claim"}, 7: {"7.6": "Dated development"}}


def result(args, rows):
    return OpenCLIResult(tuple(args), 0, json.dumps(rows), "")


def run_review(directory=None, prompt="audit"):
    return asyncio.run(review._web_story_review(
        prompt, story_numbers=[1, 7], claim_catalog=CLAIMS, log=None,
        recovery_dir=directory,
    ))


@pytest.mark.parametrize("generation", [False, True, None])
def test_missing_owner_refreshes_only_settled_saved_target(generation):
    prompt = ""
    calls = []
    reads = 0

    async def command(args, **kwargs):
        nonlocal prompt, reads
        calls.append(args)
        if args[:2] == ["chatgpt", "model"]:
            return result(args, [{"Model": "Medium"}])
        if args[:2] == ["chatgpt", "ask"]:
            prompt = args[2]
            raise OpenCLIError("CHATGPT_TURN_PENDING: Saved conversation: " + TARGET)
        assert args[:3] == ["chatgpt", "detail", TARGET]
        reads += 1
        rows = [{"Role": "Assistant", "Text": "W1P;7P"}]
        if "--refresh" in args:
            assert generation is False and reads == 3
            rows = [{"Role": "User", "Text": prompt},
                    {"Role": "Assistant", "Text": "W1P;7F@7.6"}]
        for row in rows:
            if generation is not None:
                row["Generating"] = generation
        return result(args, rows)

    with (
        patch.object(review, "run_opencli", AsyncMock(side_effect=command)),
        patch.object(review, "_target_recovery_poll_delays", return_value=(0,) * 3),
        patch.object(review.asyncio, "sleep", AsyncMock()),
    ):
        if generation is False:
            payload, raw, url, _ = run_review()
            assert not payload["approved"]
            assert payload["issues"][0]["claim_ids"] == ["7.6"]
            assert "RECOVERY STATE" in raw
            assert url == TARGET
        else:
            with pytest.raises(review._WebStoryReviewError):
                run_review()
            assert not any("--refresh" in args for args in calls)
    assert sum(args[:2] == ["chatgpt", "ask"] for args in calls) == 1


@pytest.mark.parametrize("recovered", [True, False])
def test_restart_recovers_saved_nonce_without_preflight_or_resubmission(tmp_path, recovered):
    prompt = ""

    async def first_run(args, **kwargs):
        nonlocal prompt
        if args[:2] == ["chatgpt", "model"]:
            return result(args, [{"Model": "Medium"}])
        if args[:2] == ["chatgpt", "ask"]:
            prompt = args[2]
            # Ownership is durable before the submission call.
            checkpoint = json.loads(next(tmp_path.glob("*.json")).read_text())
            assert checkpoint["request_id"] in prompt
            raise OpenCLIError("CHATGPT_TURN_PENDING: Saved conversation: " + TARGET)
        return result(args, [{"Role": "Assistant", "Text": "W1P;7P", "Generating": False}])

    with (
        patch.object(review, "run_opencli", AsyncMock(side_effect=first_run)),
        patch.object(review, "_target_recovery_poll_delays", return_value=(0, 0)),
        patch.object(review.asyncio, "sleep", AsyncMock()),
        pytest.raises(review._WebStoryReviewError),
    ):
        run_review(tmp_path)
    path = next(tmp_path.glob("*.json"))
    checkpoint = json.loads(path.read_text())
    assert checkpoint["state"] == "pending"
    assert checkpoint["conversation_url"] == TARGET

    async def resumed(args, **kwargs):
        assert args[:3] == ["chatgpt", "detail", TARGET]
        assert kwargs["site_session_namespace"] == checkpoint["session_namespace"]
        return result(args, [
            {"Role": "User", "Text": prompt if recovered else "Another request", "Generating": False},
            {"Role": "Assistant", "Text": "W1P;7F@7.6", "Generating": False},
        ])

    with (
        patch.object(review, "run_opencli", AsyncMock(side_effect=resumed)),
        patch.object(review, "close_opencli_site_sessions", AsyncMock()) as close,
        patch.object(review, "_target_recovery_poll_delays", return_value=(0, 0)),
        patch.object(review.asyncio, "sleep", AsyncMock()),
    ):
        if recovered:
            payload, _, _, _ = run_review(tmp_path)
            assert payload["issues"][0]["claim_ids"] == ["7.6"]
        else:
            with pytest.raises(review._WebStoryReviewError, match="no duplicate prompt"):
                run_review(tmp_path)
    assert json.loads(path.read_text())["state"] == ("complete" if recovered else "pending")
    if recovered:
        close.assert_awaited_once()
    else:
        close.assert_not_awaited()

    if recovered:
        # A crash between recovery and the caller's response-file write also
        # reuses the original blocking verdict, including strict validation.
        with (
            patch.object(review, "run_opencli", AsyncMock(side_effect=AssertionError("No browser call"))),
            patch.object(review, "close_opencli_site_sessions", AsyncMock()),
        ):
            assert run_review(tmp_path)[0] == payload
            saved = json.loads(path.read_text())
            saved["response"] = "W1P;7F@7.99"
            path.write_text(json.dumps(saved))
            with pytest.raises(ValueError):
                run_review(tmp_path)


def test_durable_verdict_is_bound_to_exact_prompt_and_policy(tmp_path):
    asks = []

    async def command(args, **kwargs):
        if args[:2] == ["chatgpt", "model"]:
            return result(args, [{"Model": "Medium"}])
        assert args[:2] == ["chatgpt", "ask"]
        asks.append(args[2])
        return result(args, [{"response": "W1P;7P", "conversationUrl": TARGET}])

    with (
        patch.object(review, "run_opencli", AsyncMock(side_effect=command)),
        patch.object(review, "close_opencli_site_sessions", AsyncMock()),
    ):
        run_review(tmp_path)
        run_review(tmp_path)
        run_review(tmp_path, "Changed evidence")
        with patch.object(review.config, "DAILY_NEWS_CHATGPT_REVIEW_MAX_LEVEL", "high"):
            run_review(tmp_path)
    assert len(asks) == 3


@pytest.mark.parametrize("state,owner,retained", [
    ("pending", "same", True),
    ("complete", "same", False),
    ("pending", "unrelated", False),
])
def test_daily_review_cleanup_preserves_only_its_unresolved_turn(tmp_path, state, owner, retained):
    from datetime import date

    async def failed_review(*args, review_session_namespace, **kwargs):
        directory = tmp_path / "review/browser-turns"
        directory.mkdir(parents=True)
        (directory / "owned.json").write_text(json.dumps({
            "session_namespace": review_session_namespace if owner == "same" else "unrelated",
            "state": state,
        }))
        raise RuntimeError("original failure")

    with (
        patch.object(review, "_review_daily_script", AsyncMock(side_effect=failed_review)),
        patch.object(review, "close_opencli_site_sessions", AsyncMock()) as close,
        pytest.raises(RuntimeError, match="original failure"),
    ):
        asyncio.run(review.review_daily_script(
            "draft", object(), date(2026, 10, 9), tmp_path, language="en",
            closing_remarks="Closing.", ai_endpoint=None, ai_model=None, provider_id=None,
        ))
    assert close.await_count == (0 if retained else 1)

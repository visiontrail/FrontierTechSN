import asyncio
import copy
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from backend import explainer_visual_review as review
from backend.pipeline.opencli import OpenCLIResult
from backend.pipeline import multimodal_review, opencli_session


SCENE = {"id": "scene-001", "narration": "She lifts the box.",
         "visual_prompt": "A line-drawn person lifting a box."}


@pytest.fixture(autouse=True)
def close_review_sessions(monkeypatch):
    close = AsyncMock()
    monkeypatch.setattr(opencli_session, "close_opencli_site_sessions", close)
    return close


def payload():
    return {"image_received": True, "reviews": [{"id": "scene-001", "checks": [
        {"criterion": name, "verdict": "pass", "evidence": "The box rests against both hands.", "repair": ""}
        for name in review.CHECKS]}]}


@pytest.mark.parametrize("fault", ["missing_image", "wrong_scene", "missing_check", "duplicate_check",
                                  "no_evidence", "contradictory_pass", "no_repair"])
def test_incomplete_or_contradictory_reviews_fail_closed(fault):
    value = payload()
    row = value["reviews"][0]
    if fault == "missing_image": value["image_received"] = "true"
    if fault == "wrong_scene": row["id"] = "scene-002"
    if fault == "missing_check": row["checks"].pop()
    if fault == "duplicate_check": row["checks"][-1] = copy.deepcopy(row["checks"][0])
    if fault == "no_evidence": row["checks"][0]["evidence"] = ""
    if fault == "contradictory_pass": row["checks"][0]["repair"] = "Fix the detached hands"
    if fault == "no_repair": row["checks"][0]["verdict"] = "fail"
    with pytest.raises(ValueError):
        review.validate_review(value, "scene-001")


@pytest.mark.parametrize("verdict", ["fail", "uncertain"])
def test_failure_or_uncertainty_rejects_otherwise_passing_image(verdict):
    value = payload()
    value["reviews"][0]["checks"][2].update(verdict=verdict, repair="Show contact between hands and box")
    assert review.validate_review(value, "scene-001")["passed"] is False


@pytest.mark.parametrize("verdict", ["fail", "uncertain"])
def test_omitted_repair_is_only_allowed_for_pass(verdict):
    value = payload()
    value["reviews"][0]["checks"][0]["verdict"] = verdict
    del value["reviews"][0]["checks"][0]["repair"]
    with pytest.raises(ValueError, match="omitted visible evidence or repair"):
        review.validate_review(value, "scene-001")


@pytest.mark.parametrize("repair", [None, [], 0])
def test_explicit_invalid_pass_repair_remains_invalid(repair):
    value = payload()
    value["reviews"][0]["checks"][0]["repair"] = repair
    with pytest.raises(ValueError, match="omitted visible evidence or repair"):
        review.validate_review(value, "scene-001")


def test_review_cache_is_bound_to_pixels_scene_and_rules(tmp_path, monkeypatch):
    calls = []

    async def provider(args, **kwargs):
        calls.append(args)
        assert "--file" in args and args[args.index("--file") + 1] == str(tmp_path / "image.png")
        assert args[args.index("--site-session") + 1] == "persistent"
        assert kwargs["site_session_namespace"].endswith("-gemini")
        return OpenCLIResult(tuple(args), 0, json.dumps(payload()), "")

    monkeypatch.setattr(review, "run_opencli", provider)
    rules = ["Objects must be coherent."]
    monkeypatch.setattr(review, "visual_logic_rules", lambda: rules[0])

    async def run():
        image = tmp_path / "image.png"
        image.write_bytes(b"test pixels one")
        scene = copy.deepcopy(SCENE)
        first, path = await review.review_image(image, scene, {}, tmp_path, lambda _: None)
        assert first["passed"] and path.is_file()
        assert await review.review_image(image, scene, {}, tmp_path, lambda _: None) == (first, path)
        assert len(calls) == 1
        image.write_bytes(b"different pixels")
        await review.review_image(image, scene, {}, tmp_path, lambda _: None)
        scene["narration"] = "She drops the box."
        await review.review_image(image, scene, {}, tmp_path, lambda _: None)
        rules[0] += " Gravity must be coherent."
        await review.review_image(image, scene, {}, tmp_path, lambda _: None)
        assert len(calls) == 4
        await review.review_image(image, scene, {"character_description": "Thin line limbs"}, tmp_path, lambda _: None)
        assert len(calls) == 5
        scene["on_screen_text"] = "11:47 PM"
        await review.review_image(image, scene, {"character_description": "Thin line limbs"}, tmp_path, lambda _: None)
        assert len(calls) == 6
        # Review revisions must retain every original provider response.
        assert len(list(tmp_path.glob("response-explainer_logic-*-gemini.json"))) == 6
    asyncio.run(run())


def test_valid_rejection_never_asks_fallback_for_a_better_verdict(tmp_path, monkeypatch):
    calls = []
    value = payload()
    value["reviews"][0]["checks"][0].update(verdict="fail", repair="Use a consistent perspective")

    async def provider(args, **kwargs):
        calls.append(args[0])
        return OpenCLIResult(tuple(args), 0, json.dumps(value), "")

    monkeypatch.setattr(review, "run_opencli", provider)
    monkeypatch.setattr(review.config, "AV_SYNC_REVIEW_FALLBACK_PROVIDER", "chatgpt")
    image = tmp_path / "image.png"
    image.write_bytes(b"pixels")
    result, _ = asyncio.run(review.review_image(image, SCENE, {}, tmp_path, lambda _: None))
    assert result["passed"] is False and calls == ["gemini"]


def test_clipboard_review_separates_page_direction_from_surface_visibility():
    scene = {**SCENE, "visual_prompt": "Overhead view of a teacher holding a clipboard."}
    prompt = review.review_prompt(scene, {})
    assert "bottom edge is normally nearer the reader" in prompt
    assert "does not require the holder to read its front" in prompt
    assert "Do not invent a reading action for a blank, unused prop" in prompt
    assert "rotating that sheet 180 degrees would invert it" in prompt
    assert "Still reject a visibly backward surface" in prompt
    assert "use uncertain if the necessary geometry is unreadable" in prompt
    # This is a review correction; unrelated scenes keep their cache contract.
    assert "Clipboard reading geometry:" not in review.review_prompt(SCENE, {})


def test_rejection_with_omitted_pass_repairs_is_cached_without_fallback(tmp_path, monkeypatch):
    value = payload()
    for check in value["reviews"][0]["checks"]:
        del check["repair"]
    value["reviews"][0]["checks"][0].update(
        verdict="fail", repair="Turn the laptop screen toward the character")
    original = copy.deepcopy(value)
    calls = []

    async def provider(args, **kwargs):
        calls.append(args[0])
        return OpenCLIResult(tuple(args), 0, json.dumps(value), "")

    monkeypatch.setattr(review, "run_opencli", provider)
    monkeypatch.setattr(review.config, "AV_SYNC_REVIEW_FALLBACK_PROVIDER", "chatgpt")
    image = tmp_path / "image.png"
    image.write_bytes(b"pixels")

    async def run():
        first = await review.review_image(image, SCENE, {}, tmp_path, lambda _: None)
        assert await review.review_image(image, SCENE, {}, tmp_path, lambda _: None) == first
        return first[0]

    result = asyncio.run(run())
    assert not result["passed"] and calls == ["gemini"]
    assert result["payload"] == original  # Preserve the provider's original evidence.
    assert all(check["repair"] == "" for check in result["checks"][1:])


def test_unavailable_review_blocks_image_instead_of_passing(tmp_path, monkeypatch):
    calls = []

    async def provider(args, **kwargs):
        calls.append(args[0])
        return OpenCLIResult(tuple(args), 0, json.dumps({"image_received": False, "reviews": []}), "")

    monkeypatch.setattr(review, "run_opencli", provider)
    monkeypatch.setattr(review.config, "AV_SYNC_REVIEW_FALLBACK_PROVIDER", "chatgpt")
    image = tmp_path / "image.png"
    image.write_bytes(b"pixels")
    with pytest.raises(RuntimeError, match="cannot be accepted"):
        asyncio.run(review.review_image(image, SCENE, {}, tmp_path, lambda _: None))
    assert calls == ["gemini", "chatgpt"]


@pytest.mark.parametrize("provider", ["gemini", "chatgpt"])
def test_preemption_closes_exact_review_session_and_preserves_candidate(
    tmp_path, monkeypatch, close_review_sessions, provider,
):
    entered = asyncio.Event()
    closed = asyncio.Event()
    release_cleanup = asyncio.Event()
    calls = []
    image = tmp_path / "image.png"
    image.write_bytes(b"unchanged accepted generation")
    generated = tmp_path / "complete.json"
    generated.write_text('{"provider":"generation evidence"}')

    async def command(args, **kwargs):
        calls.append((args, kwargs))
        if args[0] != provider:
            raise RuntimeError("Gemini unavailable")
        assert args[args.index("--site-session") + 1] == "persistent"
        assert kwargs["site_session_namespace"].endswith("-" + provider)
        entered.set()
        await asyncio.Event().wait()

    async def close(namespace, *, sites):
        if sites == (provider,):
            assert namespace == calls[-1][1]["site_session_namespace"]
            closed.set()
            await release_cleanup.wait()

    monkeypatch.setattr(review, "run_opencli", command)
    monkeypatch.setattr(close_review_sessions, "side_effect", close)
    monkeypatch.setattr(review.config, "AV_SYNC_REVIEW_FALLBACK_PROVIDER", "chatgpt")

    async def run():
        task = asyncio.create_task(review.review_image(image, SCENE, {}, tmp_path, lambda _: None))
        await entered.wait()
        task.cancel()
        await closed.wait()
        assert not task.done()  # The worker cannot move on before tab cleanup.
        release_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    assert [args[0] for args, _ in calls] == (["gemini"] if provider == "gemini" else ["gemini", "chatgpt"])
    assert image.read_bytes() == b"unchanged accepted generation"
    assert generated.read_text() == '{"provider":"generation evidence"}'
    journals = list(tmp_path.glob(f"review-session-*-{provider}/browser-session-{provider}.json"))
    assert len(journals) == 1
    assert json.loads(journals[0].read_text())["verified"] is True
    assert not list(tmp_path.glob("visual-review-*.json"))
    response = json.loads(next(tmp_path.glob(f"response-explainer_logic-*-{provider}.json")).read_text())
    assert "Review interrupted" in response["stderr"] or provider == "chatgpt"


def test_preemption_during_original_conversation_read_closes_same_session(
    tmp_path, monkeypatch, close_review_sessions,
):
    entered = asyncio.Event()
    calls = []
    image = tmp_path / "image.png"
    image.write_bytes(b"original candidate")
    conversation_id = "00000000-0000-0000-0000-000000000001"

    async def ask(args, **kwargs):
        if args[0] == "gemini":
            raise RuntimeError("Gemini unavailable")
        calls.append((args, kwargs))
        return OpenCLIResult(tuple(args), 0, json.dumps([{"conversationId": conversation_id}]), "")

    async def read(args, **kwargs):
        calls.append((args, kwargs))
        assert args[:3] == ["chatgpt", "detail", conversation_id]
        assert args[args.index("--site-session") + 1] == "persistent"
        assert kwargs["site_session_namespace"] == calls[0][1]["site_session_namespace"]
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(review, "run_opencli", ask)
    monkeypatch.setattr(multimodal_review, "run_opencli", read)
    monkeypatch.setattr(review.config, "AV_SYNC_REVIEW_FALLBACK_PROVIDER", "chatgpt")

    async def run():
        task = asyncio.create_task(review.review_image(image, SCENE, {}, tmp_path, lambda _: None))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    assert [args[1] for args, _ in calls] == ["ask", "detail"]
    close_review_sessions.assert_any_await(calls[0][1]["site_session_namespace"], sites=("chatgpt",))
    assert image.read_bytes() == b"original candidate"
    assert not list(tmp_path.glob("visual-review-*.json"))
    receipt = json.loads(next(tmp_path.glob("response-explainer_logic-*-chatgpt.json")).read_text())
    assert conversation_id in receipt["stdout"]


def test_preemption_during_cleanup_reuses_completed_verdict_without_new_request(
    tmp_path, monkeypatch, close_review_sessions,
):
    closing = asyncio.Event()
    release_cleanup = asyncio.Event()
    calls = []
    image = tmp_path / "image.png"
    image.write_bytes(b"original candidate")

    async def command(args, **kwargs):
        calls.append(args)
        return OpenCLIResult(tuple(args), 0, json.dumps(payload()), "")

    async def close(namespace, *, sites):
        closing.set()
        await release_cleanup.wait()

    monkeypatch.setattr(review, "run_opencli", command)
    monkeypatch.setattr(close_review_sessions, "side_effect", close)

    async def run():
        task = asyncio.create_task(review.review_image(image, SCENE, {}, tmp_path, lambda _: None))
        await closing.wait()
        report_path = next(tmp_path.glob("visual-review-*.json"))
        assert json.loads(report_path.read_text())["passed"] is True
        task.cancel()
        release_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        result, cached_path = await review.review_image(image, SCENE, {}, tmp_path, lambda _: None)
        assert result["passed"] is True and cached_path == report_path

    asyncio.run(run())
    assert len(calls) == 1

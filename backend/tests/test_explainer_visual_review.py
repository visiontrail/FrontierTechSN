import asyncio
import copy
import json
from pathlib import Path

import pytest

from backend import explainer_visual_review as review
from backend.pipeline.opencli import OpenCLIResult


SCENE = {"id": "scene-001", "narration": "She lifts the box.",
         "visual_prompt": "A line-drawn person lifting a box."}


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

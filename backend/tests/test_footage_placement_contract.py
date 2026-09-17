"""Acquisition must not count Commons clips the compositor cannot place."""
import asyncio
import hashlib
import json
from unittest.mock import AsyncMock, patch

import pytest

from backend.pipeline import composer, footage, visual_plan, web_footage


SCRIPT = (
    "The Wall Street Journal goes inside the White House's debate over artificial intelligence.\n"
    "Hackers demand a ransom after a data breach.\n"
    "Robots navigate obstacles using multimodal sensors."
)
EXCERPT = SCRIPT.splitlines()[0]


def garden_clip():
    return {
        "id": "clip-02", "provider_id": "wikimedia", "query": "White House Washington",
        "script_excerpt": EXCERPT, "purpose": EXCERPT,
        "title": "White House Kitchen Garden, March 2025.webm",
        "description": "Young gardeners visit the White House to plant fruits, veggies and herbs.",
        "source_page_url": "https://commons.example/garden", "license": "Public domain",
    }


def save_manifest(root):
    (root / "footage").mkdir()
    bad = garden_clip()
    good = {**bad, "id": "clip-01", "query": "robot sensors",
            "title": "Robots navigate obstacles", "description": "Multimodal sensors guide robots.",
            "script_excerpt": SCRIPT.splitlines()[2], "purpose": SCRIPT.splitlines()[2],
            "source_page_url": "https://commons.example/robots"}
    for clip in (good, bad):
        clip["local_path"] = f"footage/{clip['id']}.mp4"
        body = clip["id"].encode()
        (root / clip["local_path"]).write_bytes(body)
        clip["sha256"] = hashlib.sha256(body).hexdigest()
    manifest = {
        "provider_id": "hybrid-youtube", "orientation": "landscape", "selection_mode": "ai",
        "binding_version": 2, "status": "ready", "requested_clip_count": 2,
        "script_sha256": hashlib.sha256(SCRIPT.encode()).hexdigest(), "clips": [good, bad],
        "queries": [{k: c[k] for k in ("query", "purpose", "script_excerpt")} for c in (good, bad)],
    }
    footage._write_manifest(root, manifest)
    return manifest


def test_search_match_is_not_narration_grounding():
    clip = garden_clip()
    assert footage._candidate_query_is_specific(clip, clip["query"])
    assert footage._commons_narration_rejection(clip, SCRIPT)
    board = {"scenes": [{"id": str(i), "text": text} for i, text in enumerate(SCRIPT.splitlines())]}
    assert visual_plan.match_footage_scene(clip, board) is None
    suitable = {**clip, "title": "White House artificial intelligence debate",
                "description": "Administration officials discuss potential guardrails."}
    assert footage._commons_narration_rejection(suitable, SCRIPT) == ""
    assert visual_plan.match_footage_scene(suitable, board)["id"] == "0"


def test_ready_manifest_and_resume_recheck_commons_grounding(tmp_path):
    original = save_manifest(tmp_path)
    assert not footage.acquisition_is_complete(tmp_path, original, SCRIPT)
    resumed = footage._resume_web_manifest(tmp_path, provider="hybrid", script=SCRIPT,
                                           orientation="landscape", clip_count=None)
    assert [c["id"] for c in resumed["clips"]] == ["clip-01"]
    assert resumed["clips"][0]["sha256"] == original["clips"][0]["sha256"]
    assert resumed["invalidated_clips"][0]["id"] == "clip-02"
    assert "narration" in resumed["invalidated_clips"][0]["invalidation_reason"]
    assert (tmp_path / "footage/clip-02.mp4").is_file()
    assert list((tmp_path / "footage/history").glob("manifest-*.json"))


def test_recovery_supplements_only_invalidated_shot(tmp_path):
    save_manifest(tmp_path)
    script = tmp_path / "script.txt"
    script.write_text(SCRIPT)
    supplement = AsyncMock(side_effect=lambda **kw: kw["manifest"])
    with patch.object(web_footage, "supplement_web_footage", supplement):
        asyncio.run(footage.acquire_footage(
            media_provider="hybrid", task_id="test", task_dir=tmp_path, title="News",
            script_path=script, clip_count=None, orientation="landscape", license_policy="open_only",
            provider_id=None, ai_endpoint=None, ai_model=None,
        ))
    kw = supplement.call_args.kwargs
    assert kw["target_total"] == 2
    assert [s["query"] for s in kw["query_plan"]] == ["White House Washington"]
    assert len(kw["manifest"]["clips"]) == 1


def test_acquisition_rejects_wrong_scene_before_download(tmp_path):
    script = tmp_path / "script.txt"
    script.write_text(SCRIPT)
    with (
        patch.object(footage, "search_wikimedia", AsyncMock(return_value=[garden_clip()])),
        patch.object(footage, "_download_candidate", AsyncMock()) as download,
    ):
        result = asyncio.run(footage.acquire_public_footage(
            task_id="test", task_dir=tmp_path, title="News", script_path=script,
            clip_count=1, orientation="landscape", license_policy="open_only",
            provider_id=None, ai_endpoint=None, ai_model=None,
            planned_queries=[{"query": "White House Washington", "script_excerpt": EXCERPT}],
        ))
    download.assert_not_awaited()
    assert not result["clips"]
    assert result["errors"][0]["stage"] == "narration-grounding"


@pytest.mark.parametrize("duplicate", [False, True])
def test_placement_gate_reports_missing_or_duplicate_clip_identity(tmp_path, duplicate):
    manifest = save_manifest(tmp_path)
    plans = [{"id": "scene-03", "archetype": "footage",
              "footage_sequence": [{"src": "footage/clip-01.mp4"}]}]
    if duplicate:
        plans[0]["footage_sequence"].append({"src": "footage/clip-01.mp4"})
    with pytest.raises(RuntimeError, match=r"clip-02 \(White House Washington\)"):
        composer._require_public_footage_placement(plans, manifest, tmp_path)
    report = json.loads((tmp_path / "footage-placement.json").read_text())
    assert report["passed"] is False
    assert report["placed"] == (0 if duplicate else 1)


def test_composer_stops_before_collage_generation_for_unplaceable_clip(tmp_path, monkeypatch):
    save_manifest(tmp_path)
    script = tmp_path / "script.txt"
    script.write_text(SCRIPT)
    board = {"scenes": [{"id": str(i), "text": text} for i, text in enumerate(SCRIPT.splitlines())],
             "alignment": {"passed": True}, "total_duration": 30, "scene_count": 3}
    monkeypatch.setattr(composer, "_narration_manifest_failures", lambda *a: [])
    monkeypatch.setattr(composer, "_narration_completeness_failures", lambda *a, **kw: [])
    monkeypatch.setattr(composer, "_resume_unavailable_visual_review", AsyncMock(return_value=None))
    monkeypatch.setattr(composer.sb, "get_audio_duration", lambda *a: 30)
    monkeypatch.setattr(composer.av_sync, "ensure_word_transcript", AsyncMock(return_value=([], {})))
    monkeypatch.setattr(composer, "_detect_silence_boundaries", lambda *a: [])
    monkeypatch.setattr(composer.sb, "build_storyboard", lambda **kw: board)
    monkeypatch.setattr(composer.sb, "write_storyboard", lambda *a: None)
    monkeypatch.setattr(composer, "_load_or_plan_scene_visuals", AsyncMock(return_value=[
        {"id": s["id"], "archetype": "topic"} for s in board["scenes"]
    ]))
    monkeypatch.setattr(footage, "normalize_manifest_clips", AsyncMock(side_effect=lambda root, m, **kw: m))
    collage = AsyncMock()
    monkeypatch.setattr(composer.collage_broll, "generate_collage_broll", collage)
    with pytest.raises(RuntimeError, match="clip-02"):
        asyncio.run(composer.compose_video(str(script), "unused.wav", str(tmp_path),
                                           tts_model="test", collage_broll_enabled=True))
    collage.assert_not_awaited()

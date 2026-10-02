import asyncio
import copy
import hashlib
import json
from pathlib import Path

import pytest

from backend.pipeline import composer, visual_plan
from backend.pipeline.video_format import PORTRAIT


def test_quality_retry_preserves_configured_bookends_but_still_directs_unlocked_scenes():
    plans = [
        {"id": "intro", "archetype": "intro", "footage_src": "intro.mp4"},
        {"id": "news", "archetype": "footage", "footage_src": "news.mp4"},
        {"id": "graphic", "archetype": "topic"},
        {"id": "outro", "archetype": "outro", "footage_src": "outro.mp4"},
    ]
    assert [p["id"] for p in composer._director_scene_plans(plans, quality_retry=False)] == [
        "intro", "graphic", "outro",
    ]
    assert [p["id"] for p in composer._director_scene_plans(plans, quality_retry=True)] == ["graphic"]


def test_rejected_bookend_keeps_locked_preset_and_reaches_overlay_editor():
    feedback = {"issues": ["Closing controls too small"], "suggested_visual": "Large end screen"}
    staged = {"id": "outro", "archetype": "outro", "footage_src": "selected.mp4"}
    repaired = {"id": "outro", "archetype": "statement", "footage_src": "unapproved.mp4",
                "visual_review_feedback": feedback, "review_repair_source_sha256": "candidate"}
    plan = composer._retain_bookend_review_feedback(staged, repaired)
    assert plan["archetype"] == "outro"
    assert plan["footage_src"] == "selected.mp4"
    assert plan["review_repair_source_sha256"] == "candidate"
    assert plan["visual_review_feedback"] == feedback
    assert composer._director_scene_plans([plan], quality_retry=True) == [plan]


@pytest.mark.parametrize("enabled,passed,providers,valid,expected", [
    (True, False, ["chatgpt", "chatgpt"], True, True),
    (True, False, ["gemini", "chatgpt"], True, False),
    (True, False, ["chatgpt"], False, False),
    (True, True, ["chatgpt"], True, False),
    (False, False, ["chatgpt"], True, False),
    (True, False, [], True, False),
])
def test_repaired_pixels_return_to_the_configured_available_reviewer(
    tmp_path, monkeypatch, enabled, passed, providers, valid, expected,
):
    monkeypatch.setattr(composer.config, "AV_SYNC_REVIEW_FALLBACK_PROVIDER", "chatgpt" if enabled else "")
    (tmp_path / "av_sync_report.next.json").write_text(json.dumps({
        "multimodal": {"passed": passed, "calibration": {"batches": [
            {"review_provider": provider, "contract_valid": valid} for provider in providers
        ]}},
    }))
    assert composer._previous_rejection_used_fallback(tmp_path) is expected


def test_collage_planning_respects_existing_media_and_explicit_quantity():
    board = {'scenes': [
        {'id': 'opening', 'program_segment_kind': 'opening'},
        {'id': 'story-one'}, {'id': 'story-two'},
        {'id': 'closing', 'program_segment_kind': 'closing'},
    ]}
    plans = [{'id': 'story-one', 'archetype': 'footage', 'footage_src': 'actual.mp4'}]
    original = copy.deepcopy(board)
    available = composer._available_collage_storyboard(board, plans, None)
    assert [scene['id'] for scene in available['scenes']] == ['story-one', 'story-two']
    assert board == original
    with pytest.raises(RuntimeError, match='3 clips requested but only 2'):
        composer._available_collage_storyboard(board, plans, 3)
    plans.append({'id': 'story-two', 'archetype': 'footage', 'footage_src': 'other.mp4'})
    assert len(composer._available_collage_storyboard(board, plans, None)['scenes']) == 2


def test_verified_orpheus_chunks_define_exact_program_segments(tmp_path: Path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    audio = audio_dir / "tts_input_generated.wav"
    audio.write_bytes(b"wav-placeholder")
    (audio_dir / "part-1.txt").write_text("Good morning.")
    (audio_dir / "part-2.txt").write_text("First story.")
    (audio_dir / "part-3.txt").write_text("Thanks for watching.")
    (audio_dir / "tts_manifest.json").write_text(
        json.dumps(
            {
                "integrity": {"passed": True, "verified_source_coverage": 1.0},
                "parts": [
                    {"input": "part-1.txt", "duration_seconds": 1.25},
                    {"input": "part-2.txt", "duration_seconds": 2.5},
                    {"input": "part-3.txt", "duration_seconds": 1.75},
                ],
            }
        )
    )
    physical = [
        {"text": "Good morning.", "word_count": 2},
        {"text": "First story.", "word_count": 2},
        {"text": "Thanks for watching.", "word_count": 3},
    ]

    segments = composer._manifest_program_segments(audio, physical)
    alignment = composer._apply_manifest_program_alignment(
        {"passed": False, "word_coverage": 0.5, "failure_reasons": ["missed"]},
        {"passed": True},
        segments,
    )

    assert [row["start"] for row in segments] == [0.0, 1.25, 3.75]
    assert [row["duration"] for row in segments] == [1.25, 2.5, 1.75]
    assert alignment["passed"] is True
    assert alignment["method"] == "orpheus_manifest_program_timeline"
    assert alignment["whole_file_asr_observation"]["passed"] is False


def test_program_opening_copy_keeps_exact_show_title_and_date_visible():
    plans = [{"id": "scene-01", "kicker": "MORNING", "headline": "Generic"}]
    board = {
        "program_timeline": True,
        "scenes": [
            {
                "text": (
                    "It's Friday, August 28, 2026, and this is ByteFront Espresso—"
                    "your concise morning briefing."
                )
            }
        ],
    }

    composer._enforce_program_opening_copy(plans, board)

    assert plans[0]["kicker"] == "FRIDAY, AUGUST 28, 2026"
    assert plans[0]["headline"] == "ByteFront Espresso"


def _cache_board() -> dict:
    return {
        "thesis": "Frontier systems are crossing into daily work.",
        "scenes": [
            {
                "id": "scene-01",
                "index": 0,
                "text": "Robots learn from government-backed training centers.",
                "start": 0.0,
                "duration": 7.25,
                "keywords": ["robots", "training"],
            },
            {
                "id": "scene-02",
                "index": 1,
                "text": "Bacteria can form circuit-like computing systems.",
                "start": 7.25,
                "duration": 5.5,
                "keywords": ["bacteria", "circuits"],
            },
        ],
    }


def _cache_plans() -> list[dict]:
    return [
        {
            "id": "scene-01",
            "headline": "Robot training loop",
            "archetype": "footage",
            "footage_src": "../collage.mp4",
            "collage_broll": True,
        },
        {
            "id": "scene-02",
            "archetype": "news_image",
            "headline": "Living circuit boards",
            "news_image": True,
            "news_image_src": "../news_images/bacteria.jpg",
            "news_image_mode": "fullscreen",
            "news_image_original_archetype": "statement",
        },
    ]


def _write_cache(tmp_path, board: dict | None = None, plans: list[dict] | None = None) -> None:
    composer._write_visual_plan_checkpoint(
        tmp_path,
        board or _cache_board(),
        plans or _cache_plans(),
    )


def test_mount_list_starts_with_content_and_has_no_title_card():
    board = {
        "content_start": 0.0,
        "scenes": [
            {"id": "scene-01", "start": 0.0, "duration": 7.0},
            {"id": "scene-02", "start": 7.0, "duration": 5.0},
        ],
    }

    mounts = composer._mount_list(board)

    assert mounts[0] == {"id": "scene-01", "start": 0.0, "duration": 7.0}
    assert [mount["id"] for mount in mounts] == ["scene-01", "scene-02"]


def test_mount_list_marks_the_branded_intro_for_foreground_layering():
    board = {
        "scenes": [
            {
                "id": "scene-intro",
                "start": 0.0,
                "duration": 6.0,
                "scene_kind": "intro",
            },
            {"id": "scene-01", "start": 6.0, "duration": 7.0},
        ]
    }

    mounts = composer._mount_list(board)

    assert mounts[0]["scene_kind"] == "intro"
    assert mounts[1] == {"id": "scene-01", "start": 6.0, "duration": 7.0}


def test_kit_plans_preserve_the_fifth_narrative_item():
    items = [
        "Summarize annual reports across multiple documents",
        "Search and compare company operational data online",
        "Control a desktop browser for retrieval and document generation",
        "Build English-language presentations from data",
        "Generate marketing posters from reference images",
        "A sixth unsupported task",
    ]

    plans = composer._kit_plans(
        [{"id": "scene-11", "archetype": "list", "items": items}],
        [{"id": "scene-11", "duration": 12.0}],
        composer.scene_kit.DEFAULT_THEME,
    )

    assert plans[0].items == tuple(items[: composer.scene_kit.MAX_NARRATIVE_ITEMS])
    assert plans[0].items[-1] == "Generate marketing posters from reference images"


def test_locked_visual_asset_gate_requires_every_media_source_in_scene_html(tmp_path):
    compositions = tmp_path / "compositions"
    compositions.mkdir()
    (compositions / "scene-02.html").write_text(
        '<video src="../footage/clip-01.mp4"></video>'
        '<video src="../footage/clip-02.mp4"></video>'
        '<img src="../news_webpages/page-01.png">',
        encoding="utf-8",
    )
    plans = [
        {
            "id": "scene-02",
            "footage_src": "../footage/clip-01.mp4",
            "footage_sequence": [
                {"src": "../footage/clip-01.mp4"},
                {"src": "../footage/clip-02.mp4"},
            ],
            "news_webpage_src": "../news_webpages/page-01.png",
        }
    ]

    composer._assert_locked_visual_assets(tmp_path, plans)

    (compositions / "scene-02.html").write_text(
        '<video src="../footage/clip-01.mp4"></video>', encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="news_webpages/page-01.png"):
        composer._assert_locked_visual_assets(tmp_path, plans)


def test_cached_scene_plans_require_exact_current_checkpoint_and_strip_placements(tmp_path):
    board = _cache_board()
    _write_cache(tmp_path, board)

    cached = composer._load_cached_scene_plans(tmp_path, board)

    assert cached is not None
    assert [item["id"] for item in cached] == ["scene-01", "scene-02"]
    assert cached[0]["archetype"] == "topic"
    assert "collage_broll" not in cached[0]
    assert "footage_src" not in cached[0]
    assert cached[1] == {
        "id": "scene-02",
        "archetype": "statement",
        "headline": "Living circuit boards",
    }
    assert isinstance(json.loads((tmp_path / "visual_plan.json").read_text()), list)


def _failed_visual_candidate(root):
    (root / "video.next.mp4").write_bytes(b"reviewed candidate")
    digest = hashlib.sha256(b"reviewed candidate").hexdigest()
    (root / "quality_retry_state.json").write_text(json.dumps({
        "status": "retrying", "latest": {
            "rendered_video_sha256": digest, "failed_scene_ids": ["scene-02"],
            "sampling_contract": composer.multimodal_review.SAMPLING_CONTRACT,
            "scene_reviews": [{"id": "scene-02", "review_available": True, "issues": ["Missing quantified context"],
                               "suggested_visual": "Show the narrated figures"}],
        },
    }))
    return digest


def test_quality_retry_replans_only_failed_scenes_and_checkpoints_feedback(tmp_path, monkeypatch):
    board = {**_cache_board(), "scene_count": 2}
    _write_cache(tmp_path, board)
    digest = _failed_visual_candidate(tmp_path)
    before = composer._load_cached_scene_plans(tmp_path, board)
    calls = []

    async def repair(repair_board, **kwargs):
        calls.append(repair_board)
        return [{"id": "scene-02", "archetype": "stat", "headline": "Quantified living circuits",
                 "body": "The measured result and its attribution"}]

    monkeypatch.setattr(visual_plan, "plan_scene_visuals", repair)
    kwargs = dict(ai_endpoint=None, ai_model=None, provider_id=None, log=lambda _: None)
    repaired = asyncio.run(composer._load_or_plan_scene_visuals(tmp_path, board, **kwargs))
    assert repaired[0] == before[0]
    assert repaired[1]["review_repair_source_sha256"] == digest
    assert repaired[1]["visual_review_feedback"] == {
        "issues": ["Missing quantified context"],
        "suggested_visual": "Show the narrated figures",
    }
    assert [s["id"] for s in calls[0]["scenes"]] == ["scene-02"]
    assert calls[0]["visual_review_feedback"][0]["issues"] == ["Missing quantified context"]
    assert asyncio.run(composer._load_or_plan_scene_visuals(tmp_path, board, **kwargs)) == repaired
    assert len(calls) == 1


def test_quality_feedback_from_another_candidate_is_not_reused(tmp_path):
    _failed_visual_candidate(tmp_path)
    (tmp_path / "video.next.mp4").write_bytes(b"different video")
    assert composer._pending_visual_repairs(tmp_path, _cache_plans()) == ("", [])


def test_legacy_midpoint_feedback_cannot_rewrite_a_complete_shot_sequence(tmp_path):
    _failed_visual_candidate(tmp_path)
    path = tmp_path / "quality_retry_state.json"
    state = json.loads(path.read_text())
    state["latest"].pop("sampling_contract")
    path.write_text(json.dumps(state))
    assert composer._pending_visual_repairs(tmp_path, _cache_plans()) == ("", [])


def test_failed_repair_provider_preserves_plan_and_unconsumed_feedback(tmp_path, monkeypatch):
    board = {**_cache_board(), "scene_count": 2}
    _write_cache(tmp_path, board)
    _failed_visual_candidate(tmp_path)
    before = (tmp_path / "visual_plan.json").read_bytes()

    async def unavailable(*args, **kwargs):
        return [{"id": "scene-02", "headline": "First sentence only", "grounding_source": "narration_fallback"}]

    monkeypatch.setattr(visual_plan, "plan_scene_visuals", unavailable)
    with pytest.raises(RuntimeError, match="retaining the previous plan"):
        asyncio.run(composer._load_or_plan_scene_visuals(
            tmp_path, board, ai_endpoint=None, ai_model=None, provider_id=None, log=lambda _: None,
        ))
    assert (tmp_path / "visual_plan.json").read_bytes() == before
    assert composer._pending_visual_repairs(tmp_path, _cache_plans())[1]


def test_legacy_fallback_repair_marker_does_not_consume_feedback(tmp_path):
    digest = _failed_visual_candidate(tmp_path)
    plans = _cache_plans()
    plans[1].update(grounding_source="narration_fallback", review_repair_source_sha256=digest)
    assert [r["id"] for r in composer._pending_visual_repairs(tmp_path, plans)[1]] == ["scene-02"]


def test_unavailable_visual_review_does_not_replan_scene_content(tmp_path):
    _failed_visual_candidate(tmp_path)
    path = tmp_path / "quality_retry_state.json"
    state = json.loads(path.read_text())
    state["latest"]["scene_reviews"][0]["review_available"] = False
    path.write_text(json.dumps(state))
    assert composer._pending_visual_repairs(tmp_path, _cache_plans())[1] == []


@pytest.mark.parametrize("mutation", [None, "video", "script", "setting", "report_hash", "missing_checkpoint", "news_images.py", "collage_broll.py", "visual_plan.py", "composer.py"])
@pytest.mark.parametrize("interrupted", [False, True])
@pytest.mark.parametrize("accepted", [False, True])
def test_visual_review_resume_is_bound_to_candidate_and_render_inputs(tmp_path, monkeypatch, mutation, interrupted, accepted):
    from unittest.mock import AsyncMock
    policy_path = None
    if mutation and mutation.endswith(".py"):
        code_dir = tmp_path / "pipeline_source"
        code_dir.mkdir()
        monkeypatch.setattr(composer, "__file__", str(code_dir / "composer.py"))
        policy_path = code_dir / mutation
        policy_path.write_text("old generation policy")
    script = tmp_path / "script.txt"
    script.write_text("The complete narration")
    candidate = tmp_path / "video.next.mp4"
    candidate.write_bytes(b"rendered video")
    request = {"script_path": str(script), "captions_enabled": False}
    board = {"total_duration": 8.0, "audio_duration": 8.0, "alignment": {"passed": True},
             "scenes": [{"lines": [{"start": 0.0, "duration": 8.0, "text": "The complete narration"}]}]}
    (tmp_path / "storyboard.json").write_text(json.dumps(board))
    digest = composer._sha256_path(candidate)
    (tmp_path / "render_review_checkpoint.json").write_text(json.dumps({
        "input_sha256": composer._review_retry_fingerprint(tmp_path, request), "video_sha256": digest,
    }))
    report = {
        "rendered_video_sha256": digest, "visual_grounding": {"passed": True},
        "multimodal": {"passed": False, "errors": ["provider unavailable"]},
    }
    if interrupted:
        report.pop("rendered_video_sha256")
        report.update(quality_status="pending", delivery_status="pending", multimodal={"status": "pending", "passed": False})
    if mutation == "report_hash": report["rendered_video_sha256"] = "another video"
    (tmp_path / "av_sync_report.next.json").write_text(json.dumps(report))
    review = AsyncMock(return_value={"passed": accepted, "scenes": [], "errors": []})
    monkeypatch.setattr(composer.multimodal_review, "review_video", review)
    monkeypatch.setattr(composer, "_rendered_video_failures", lambda *args, **kwargs: [])
    monkeypatch.setattr(composer.config, "AV_SYNC_GEMINI_REVIEW_ENABLED", True)
    if mutation == "video": candidate.write_bytes(b"changed candidate")
    if mutation == "script": script.write_text("Changed narration")
    if mutation == "setting": request["captions_enabled"] = True
    if mutation == "missing_checkpoint": (tmp_path / "render_review_checkpoint.json").unlink()
    if policy_path is not None: policy_path.write_text("corrected generation policy")
    if mutation is None and not accepted:
        with pytest.raises(composer.QualityGateRetry):
            asyncio.run(composer._resume_unavailable_visual_review(tmp_path, request, composer.LANDSCAPE, lambda _: None))
        assert candidate.is_file()
        assert not (tmp_path / "av_sync_report.json").exists()
        assert json.loads((tmp_path / "av_sync_report.next.json").read_text())["passed"] is False
        return
    result = asyncio.run(composer._resume_unavailable_visual_review(tmp_path, request, composer.LANDSCAPE, lambda _: None))
    if mutation is None:
        assert Path(result).read_bytes() == b"rendered video"
        assert json.loads((tmp_path / "av_sync_report.json").read_text())["passed"] is True
        review.assert_awaited_once()
        assert review.await_args.kwargs["prefer_fallback"] is (not interrupted)
        assert json.loads((tmp_path / "av_sync_report.json").read_text())["rendered_video_sha256"] == digest
    else:
        assert result is None
        review.assert_not_awaited()
        assert candidate.is_file()


def test_quality_retry_stops_before_rendering_unchanged_scene_again(tmp_path, monkeypatch):
    board = {**_cache_board(), "scene_count": 2}
    _write_cache(tmp_path, board)
    _failed_visual_candidate(tmp_path)
    cached = composer._load_cached_scene_plans(tmp_path, board)

    async def unchanged(*args, **kwargs):
        return [cached[1]]

    monkeypatch.setattr(visual_plan, "plan_scene_visuals", unchanged)
    with pytest.raises(RuntimeError, match="unchanged visible content"):
        asyncio.run(composer._load_or_plan_scene_visuals(
            tmp_path, board, ai_endpoint=None, ai_model=None, provider_id=None, log=lambda _: None,
        ))


def _mismatching_media_candidate(root, verdict="mismatch"):
    board = {**_cache_board(), "scene_count": 2}
    plans = _cache_plans()
    asset = root / "historical.jpg"
    asset.write_bytes(b"wrong historical event")
    asset_hash = hashlib.sha256(asset.read_bytes()).hexdigest()
    plans[1]["media_shots"] = [{
        "src": "../historical.jpg", "sha256": asset_hash, "kind": "image",
        "start": 0, "duration": 5.5,
    }]
    composer._write_visual_plan_checkpoint(root, board, plans)
    digest = _failed_visual_candidate(root)
    (root / "av_sync_report.next.json").write_text(json.dumps({
        "rendered_video_sha256": digest, "multimodal": {"scenes": [{
            "id": "scene-02", "passed": False, "verdict": verdict,
            "timestamp": 9.0, "issues": ["Wrong historical event"],
        }]},
    }))
    return board, digest


@pytest.mark.parametrize("verdict", ["mismatch", "partial"])
def test_quality_retry_removes_sampled_wrong_media_without_rewriting_correct_copy(tmp_path, monkeypatch, verdict):
    from unittest.mock import AsyncMock
    board, digest = _mismatching_media_candidate(tmp_path, verdict)
    before = composer._load_cached_scene_plans(tmp_path, board)
    planner = AsyncMock(side_effect=AssertionError("copy already describes the right event"))
    monkeypatch.setattr(visual_plan, "plan_scene_visuals", planner)
    kwargs = dict(ai_endpoint=None, ai_model=None, provider_id=None, log=lambda _: None)
    repaired = asyncio.run(composer._load_or_plan_scene_visuals(tmp_path, board, **kwargs))
    assert repaired[0] == before[0]
    assert repaired[1]["headline"] == before[1]["headline"]
    assert repaired[1]["review_rejected_media"][0]["src"] == "../historical.jpg"
    assert repaired[1]["review_rejected_media"][0]["rendered_video_sha256"] == digest
    assert repaired[1]["review_repair_source_sha256"] == digest
    assert asyncio.run(composer._load_or_plan_scene_visuals(tmp_path, board, **kwargs)) == repaired
    planner.assert_not_awaited()
    assert (tmp_path / "historical.jpg").read_bytes() == b"wrong historical event"


def _misassigned_article_candidate(root):
    from PIL import Image
    board, digest = _mismatching_media_candidate(root)
    plans = json.loads((root / "visual_plan.json").read_text())
    asset = root / "news_webpages/article.png"
    asset.parent.mkdir()
    Image.new("RGB", (2880, 1800), "white").save(asset)
    asset_hash = hashlib.sha256(asset.read_bytes()).hexdigest()
    plans[1].update(news_webpage_src="../news_webpages/article.png", news_webpage_sha256=asset_hash)
    plans[1]["media_shots"] = [{"kind": "article", "src": "../news_webpages/article.png", "sha256": asset_hash}]
    composer._write_visual_plan_checkpoint(root, board, plans)
    article = {"language": "en", "source_name": "Example", "title": "Robots learn in training centers",
               "url": "https://example.com/robots"}
    (root / "research").mkdir()
    (root / "research/dossier.json").write_text(json.dumps({"selected": [article]}))
    page = {"scene_id": "scene-02", "source_url": article["url"], "sha256": asset_hash,
            "local_path": "news_webpages/article.png", "bytes": asset.stat().st_size}
    (root / "news_webpages/manifest.json").write_text(json.dumps({"pages": [page]}))
    path = root / "av_sync_report.next.json"
    report = json.loads(path.read_text())
    report["multimodal"]["scenes"][0].update(rubric_consistent=True, sample_timestamps=[8, 9, 10])
    path.write_text(json.dumps(report))
    return board, digest


def _attributed_sequence_candidate(root):
    board, digest = _mismatching_media_candidate(root)
    plans = json.loads((root / "visual_plan.json").read_text())
    shot = plans[1]["media_shots"][0]
    shot["duration"] = 2
    plans[1]["media_shots"].append({"kind": "editorial", "start": 2, "duration": 3.5})
    composer._write_visual_plan_checkpoint(root, board, plans)
    frame = {"id": "scene-02", "timestamp": 11, "scene": board["scenes"][1], "samples": [
        {"id": "scene-02-shot-01", "timestamp": 8.25,
         "shot_sha256": composer.multimodal_review.shot_fingerprint(shot),
         "media": {key: shot[key] for key in ("src", "sha256", "kind")}},
        {"id": "scene-02-shot-02", "timestamp": 11, "media": {"kind": "editorial"}},
    ]}
    normalized = composer.multimodal_review.normalise_batch({"image_received": True, "reviews": [{
        "id": "scene-02", "score": 30, "verdict": "mismatch", "visual_summary": "Wrong historical event",
        "alignment_reason": "The first shot depicts a different event", "issues": ["Wrong source subject"],
        "suggested_visual": "Use the correct event",
        "source_media_mismatches": [{"sample_id": "scene-02-shot-01", "reason": "Wrong historical event"}],
    }]}, [frame], 70)
    assert normalized["contract_valid"]
    (root / "av_sync_report.next.json").write_text(json.dumps({
        "rendered_video_sha256": digest, "multimodal": {
            "sampling_contract": composer.multimodal_review.SAMPLING_CONTRACT,
            "scenes": normalized["reviews"],
        },
    }))
    return board, digest


def test_sequence_repair_excludes_exact_attributed_media_and_preserves_denial(tmp_path, monkeypatch):
    from unittest.mock import AsyncMock
    board, digest = _attributed_sequence_candidate(tmp_path)
    before = composer._load_cached_scene_plans(tmp_path, board)
    report_bytes = (tmp_path / "av_sync_report.next.json").read_bytes()
    planner = AsyncMock(side_effect=AssertionError("grounded copy does not need another request"))
    monkeypatch.setattr(visual_plan, "plan_scene_visuals", planner)
    repaired = asyncio.run(composer._load_or_plan_scene_visuals(
        tmp_path, board, ai_endpoint=None, ai_model=None, provider_id=None, log=lambda _: None,
    ))
    rejection = repaired[1]["review_rejected_media"][0]
    assert rejection["sample_id"] == "scene-02-shot-01"
    assert rejection["src"] == "../historical.jpg"
    assert rejection["rendered_video_sha256"] == digest
    assert repaired[0] == before[0]
    assert repaired[1]["headline"] == before[1]["headline"]
    assert (tmp_path / "av_sync_report.next.json").read_bytes() == report_bytes
    assert (tmp_path / "historical.jpg").read_bytes() == b"wrong historical event"
    planner.assert_not_awaited()


@pytest.mark.parametrize("outcome", ["mismatch", "match", "unavailable", "changed_video", "changed_plan"])
def test_legacy_attribution_reuses_pixels_and_preserves_original_denial(tmp_path, monkeypatch, outcome):
    from unittest.mock import AsyncMock
    board, digest = _attributed_sequence_candidate(tmp_path)
    path = tmp_path / "av_sync_report.next.json"
    original = json.loads(path.read_text())
    fresh_review = copy.deepcopy(original["multimodal"])
    legacy = original["multimodal"]["scenes"][0]
    legacy.pop("media_sample_contract")
    legacy.pop("rejected_media_samples")
    path.write_text(json.dumps(original))
    before = path.read_bytes()
    if outcome == "match":
        fresh_review["scenes"][0].update(passed=True, verdict="match", rejected_media_samples=[])
    if outcome == "unavailable":
        fresh_review["scenes"][0].update(rubric_consistent=False, rejected_media_samples=[])

    async def review(candidate, scoped, directory, **kwargs):
        assert candidate == tmp_path / "video.next.mp4"
        assert [scene["id"] for scene in scoped["scenes"]] == ["scene-02"]
        if outcome == "changed_video": candidate.write_bytes(b"new render")
        if outcome == "changed_plan":
            plans = json.loads((tmp_path / "visual_plan.json").read_text())
            plans[1]["media_shots"][0]["source_start"] = 1
            (tmp_path / "visual_plan.json").write_text(json.dumps(plans))
        return fresh_review

    reviewer = AsyncMock(side_effect=review)
    monkeypatch.setattr(composer.multimodal_review, "review_video", reviewer)
    result = asyncio.run(composer._legacy_media_attribution(
        tmp_path, board, digest, {"scene-02"}, lambda _: None,
    ))
    assert bool(result) is (outcome == "mismatch")
    assert path.read_bytes() == before
    assert (tmp_path / "historical.jpg").read_bytes() == b"wrong historical event"
    assert len(list((tmp_path / "media_attribution").glob("*/report.json"))) == 1
    if outcome in {"mismatch", "match", "unavailable"}:
        assert asyncio.run(composer._legacy_media_attribution(
            tmp_path, board, digest, {"scene-02"}, lambda _: None,
        )) == result
    reviewer.assert_awaited_once()


def test_interrupted_legacy_attribution_does_not_repeat_browser_request(tmp_path, monkeypatch):
    from unittest.mock import AsyncMock
    board, digest = _attributed_sequence_candidate(tmp_path)
    path = tmp_path / "av_sync_report.next.json"
    report = json.loads(path.read_text())
    report["multimodal"]["scenes"][0].pop("media_sample_contract")
    path.write_text(json.dumps(report))
    reviewer = AsyncMock(side_effect=asyncio.CancelledError())
    monkeypatch.setattr(composer.multimodal_review, "review_video", reviewer)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(composer._legacy_media_attribution(tmp_path, board, digest, {"scene-02"}, lambda _: None))
    with pytest.raises(RuntimeError, match="recover its existing browser request"):
        asyncio.run(composer._legacy_media_attribution(tmp_path, board, digest, {"scene-02"}, lambda _: None))
    reviewer.assert_awaited_once()


def test_legacy_recovery_supplies_exclusion_to_compose_without_copy_retry(tmp_path, monkeypatch):
    from unittest.mock import AsyncMock
    board, digest = _attributed_sequence_candidate(tmp_path)
    path = tmp_path / "av_sync_report.next.json"
    report = json.loads(path.read_text())
    review = copy.deepcopy(report["multimodal"])
    report["multimodal"]["scenes"][0].pop("media_sample_contract")
    report["multimodal"]["scenes"][0].pop("rejected_media_samples")
    path.write_text(json.dumps(report))
    before = path.read_bytes()
    reviewer = AsyncMock(return_value=review)
    planner = AsyncMock(side_effect=AssertionError("copy cannot fix mismatching media"))
    monkeypatch.setattr(composer.multimodal_review, "review_video", reviewer)
    monkeypatch.setattr(visual_plan, "plan_scene_visuals", planner)
    repaired = asyncio.run(composer._load_or_plan_scene_visuals(
        tmp_path, board, ai_endpoint=None, ai_model=None, provider_id=None, log=lambda _: None,
    ))
    assert repaired[1]["review_rejected_media"][0]["sample_id"] == "scene-02-shot-01"
    assert repaired[1]["review_repair_source_sha256"] == digest
    assert path.read_bytes() == before
    planner.assert_not_awaited()


@pytest.mark.parametrize("mutation", ["report_hash", "asset_bytes", "shot_inpoint", "wrong_id", "wrong_src", "wrong_timestamp", "unknown_contract", "missing_sampling_contract", "unavailable", "empty", "no_reason"])
def test_sequence_exclusion_fails_closed_on_stale_or_incomplete_binding(tmp_path, mutation):
    board, digest = _attributed_sequence_candidate(tmp_path)
    path = tmp_path / "av_sync_report.next.json"
    report = json.loads(path.read_text())
    review = report["multimodal"]["scenes"][0]
    sample = review["rejected_media_samples"][0]
    if mutation == "report_hash": report["rendered_video_sha256"] = "old video"
    if mutation == "asset_bytes": (tmp_path / "historical.jpg").write_bytes(b"replacement asset")
    if mutation == "shot_inpoint":
        plans = json.loads((tmp_path / "visual_plan.json").read_text())
        plans[1]["media_shots"][0]["source_start"] = 1
        composer._write_visual_plan_checkpoint(tmp_path, board, plans)
    if mutation == "wrong_id": sample["sample_id"] = "scene-02-shot-02"
    if mutation == "wrong_src": sample["src"] = "../other.jpg"
    if mutation == "wrong_timestamp": sample["timestamp"] = 9
    if mutation == "unknown_contract": review["media_sample_contract"] = "unknown"
    if mutation == "missing_sampling_contract": report["multimodal"].pop("sampling_contract")
    if mutation == "unavailable": review["rubric_consistent"] = False
    if mutation == "empty": review["rejected_media_samples"] = []
    if mutation == "no_reason": sample["reason"] = " "
    path.write_text(json.dumps(report))
    assert composer._review_rejected_media(tmp_path, board, digest) == {}


def test_quality_retry_repairs_proven_article_assignment_without_rewriting_copy(tmp_path, monkeypatch):
    from unittest.mock import AsyncMock
    board, digest = _misassigned_article_candidate(tmp_path)
    before = composer._load_cached_scene_plans(tmp_path, board)
    planner = AsyncMock(side_effect=AssertionError("repair the article inventory, not correct copy"))
    monkeypatch.setattr(visual_plan, "plan_scene_visuals", planner)
    kwargs = dict(ai_endpoint=None, ai_model=None, provider_id=None, log=lambda _: None)
    report_before = (tmp_path / "av_sync_report.next.json").read_bytes()
    repaired = asyncio.run(composer._load_or_plan_scene_visuals(tmp_path, board, **kwargs))
    assert repaired[0] == before[0]
    assert repaired[1]["headline"] == before[1]["headline"]
    assert repaired[1]["review_webpage_assignment_repair"]["source_url"] == "https://example.com/robots"
    assert repaired[1]["review_repair_source_sha256"] == digest
    assert "news_webpage_src" not in repaired[1]
    assert asyncio.run(composer._load_or_plan_scene_visuals(tmp_path, board, **kwargs)) == repaired
    planner.assert_not_awaited()
    assert (tmp_path / "av_sync_report.next.json").read_bytes() == report_before
    assert (tmp_path / "news_webpages/article.png").is_file()


@pytest.mark.parametrize("mutation", ["report_hash", "asset_bytes", "shot_hash", "unavailable", "correct_assignment"])
def test_article_assignment_repair_requires_current_independent_evidence(tmp_path, mutation):
    board, digest = _misassigned_article_candidate(tmp_path)
    path = tmp_path / "av_sync_report.next.json"
    report = json.loads(path.read_text())
    if mutation == "report_hash": report["rendered_video_sha256"] = "stale"
    if mutation == "unavailable": report["multimodal"]["scenes"][0]["rubric_consistent"] = False
    path.write_text(json.dumps(report))
    if mutation == "asset_bytes": (tmp_path / "news_webpages/article.png").write_bytes(b"new image")
    if mutation == "shot_hash":
        path = tmp_path / "visual_plan.json"
        plans = json.loads(path.read_text())
        plans[1]["media_shots"][0]["sha256"] = "stale"
        path.write_text(json.dumps(plans))
    if mutation == "correct_assignment":
        path = tmp_path / "research/dossier.json"
        dossier = json.loads(path.read_text())
        dossier["selected"][0]["title"] = board["scenes"][1]["text"]
        path.write_text(json.dumps(dossier))
    assert composer._review_misassigned_webpages(tmp_path, board, digest) == {}


@pytest.mark.parametrize("verdict", ["mismatch", "partial"])
@pytest.mark.parametrize("mutation", ["report_hash", "asset_bytes", "editorial_sample", "match", "passed_partial", "unknown_verdict", "outside_sample", "unavailable", "sequence"])
def test_media_exclusion_requires_current_mismatching_sample(tmp_path, monkeypatch, mutation, verdict):
    from unittest.mock import AsyncMock
    board, digest = _mismatching_media_candidate(tmp_path, verdict)
    path = tmp_path / "av_sync_report.next.json"
    report = json.loads(path.read_text())
    review = report["multimodal"]["scenes"][0]
    if mutation == "report_hash": report["rendered_video_sha256"] = "different candidate"
    if mutation == "asset_bytes": (tmp_path / "historical.jpg").write_bytes(b"new asset")
    if mutation == "editorial_sample":
        plans = json.loads((tmp_path / "visual_plan.json").read_text())
        plans[1]["media_shots"][0]["kind"] = "editorial"
        composer._write_visual_plan_checkpoint(tmp_path, board, plans)
    if mutation == "match": review.update(passed=True, verdict="match")
    if mutation == "passed_partial": review.update(passed=True, verdict="partial")
    if mutation == "unknown_verdict": review["verdict"] = "unknown"
    if mutation == "outside_sample": review["timestamp"] = 100
    if mutation == "sequence": review["sample_timestamps"] = [8.0, 9.0, 10.0]
    path.write_text(json.dumps(report))
    if mutation == "unavailable":
        path = tmp_path / "quality_retry_state.json"
        state = json.loads(path.read_text())
        state["latest"]["scene_reviews"][0]["review_available"] = False
        path.write_text(json.dumps(state))
    else:
        assert composer._review_rejected_media(tmp_path, board, digest) == {}
    cached = composer._load_cached_scene_plans(tmp_path, board)
    monkeypatch.setattr(visual_plan, "plan_scene_visuals", AsyncMock(return_value=[cached[1]]))
    kwargs = dict(ai_endpoint=None, ai_model=None, provider_id=None, log=lambda _: None)
    if mutation == "unavailable":
        assert asyncio.run(composer._load_or_plan_scene_visuals(tmp_path, board, **kwargs)) == cached
    else:
        with pytest.raises(RuntimeError, match="unchanged visible content"):
            asyncio.run(composer._load_or_plan_scene_visuals(tmp_path, board, **kwargs))


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("thesis", "A different thesis"),
        ("thesis", None),
        ("index", 4),
        ("text", "Unrelated narration with the same scene id"),
        ("start", 0.125),
        ("duration", 8.0),
        ("keywords", ["different", "anchors"]),
    ],
)
def test_cached_scene_plans_reject_changed_storyboard_semantics(
    tmp_path,
    field,
    replacement,
):
    board = _cache_board()
    _write_cache(tmp_path, board)
    changed = copy.deepcopy(board)
    if field == "thesis":
        changed[field] = replacement
    else:
        changed["scenes"][0][field] = replacement

    assert composer._load_cached_scene_plans(tmp_path, changed) is None


def test_cached_scene_plans_reject_missing_and_legacy_sidecars(tmp_path):
    board = _cache_board()
    (tmp_path / "visual_plan.json").write_text(
        json.dumps(_cache_plans()), encoding="utf-8"
    )
    assert composer._load_cached_scene_plans(tmp_path, board) is None

    _write_cache(tmp_path, board)
    (tmp_path / composer.VISUAL_PLAN_CACHE_FILENAME).unlink()
    assert composer._load_cached_scene_plans(tmp_path, board) is None


@pytest.mark.parametrize("sidecar", [b"not-json", b'{"cache_version": NaN}'])
def test_cached_scene_plans_reject_corrupt_or_nonfinite_sidecars(tmp_path, sidecar):
    board = _cache_board()
    _write_cache(tmp_path, board)
    (tmp_path / composer.VISUAL_PLAN_CACHE_FILENAME).write_bytes(sidecar)

    assert composer._load_cached_scene_plans(tmp_path, board) is None


def test_cached_scene_plans_reject_tampered_plan_and_sidecar(tmp_path):
    board = _cache_board()
    _write_cache(tmp_path, board)
    plan_path = tmp_path / "visual_plan.json"
    plan_path.write_bytes(plan_path.read_bytes() + b"\n")
    assert composer._load_cached_scene_plans(tmp_path, board) is None

    _write_cache(tmp_path, board)
    cache_path = tmp_path / composer.VISUAL_PLAN_CACHE_FILENAME
    cache = json.loads(cache_path.read_text())
    cache["visual_plan_sha256"] = "0" * 64
    cache_path.write_text(json.dumps(cache), encoding="utf-8")
    assert composer._load_cached_scene_plans(tmp_path, board) is None


@pytest.mark.parametrize("nonfinite", ["NaN", "Infinity", "1e999"])
def test_cached_scene_plans_reject_nonfinite_plan_even_with_matching_sha(
    tmp_path,
    nonfinite,
):
    board = _cache_board()
    _write_cache(tmp_path, board)
    plan_bytes = (
        '[{"id":"scene-01","stat":'
        + nonfinite
        + '},{"id":"scene-02"}]'
    ).encode()
    plan_path = tmp_path / "visual_plan.json"
    plan_path.write_bytes(plan_bytes)
    cache_path = tmp_path / composer.VISUAL_PLAN_CACHE_FILENAME
    cache = json.loads(cache_path.read_text())
    cache["visual_plan_bytes"] = len(plan_bytes)
    cache["visual_plan_sha256"] = hashlib.sha256(plan_bytes).hexdigest()
    cache_path.write_text(json.dumps(cache), encoding="utf-8")

    assert composer._load_cached_scene_plans(tmp_path, board) is None


def test_cached_scene_plans_reject_prompt_contract_version_change(tmp_path, monkeypatch):
    board = _cache_board()
    _write_cache(tmp_path, board)
    monkeypatch.setattr(
        composer,
        "VISUAL_PLAN_PROMPT_CONTRACT_VERSION",
        composer.VISUAL_PLAN_PROMPT_CONTRACT_VERSION + 1,
    )

    assert composer._load_cached_scene_plans(tmp_path, board) is None


def test_cached_scene_plans_reject_prompt_bytes_change(tmp_path, monkeypatch):
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    source_prompt = composer.config.PROMPTS_DIR / "visual_plan.txt"
    prompt_path = prompt_dir / "visual_plan.txt"
    prompt_path.write_bytes(source_prompt.read_bytes())
    monkeypatch.setattr(composer.config, "PROMPTS_DIR", prompt_dir)
    board = _cache_board()
    _write_cache(tmp_path, board)

    prompt_path.write_text("changed visual planner contract", encoding="utf-8")

    assert composer._load_cached_scene_plans(tmp_path, board) is None


def test_direction_checkpoint_survives_a_downstream_failure_without_attachments(
    tmp_path,
    monkeypatch,
):
    board = {**_cache_board(), "scene_count": 2}
    direction = [
        {"id": "scene-01", "archetype": "topic", "headline": "Robot training loop"},
        {"id": "scene-02", "archetype": "statement", "headline": "Living circuits"},
    ]
    planner_calls = 0

    async def plan_scene_visuals(*_args, **_kwargs):
        nonlocal planner_calls
        planner_calls += 1
        return copy.deepcopy(direction)

    monkeypatch.setattr(composer.visual_plan, "plan_scene_visuals", plan_scene_visuals)

    async def fail_after_direction():
        plans = await composer._load_or_plan_scene_visuals(
            tmp_path,
            board,
            ai_endpoint=None,
            ai_model=None,
            provider_id=None,
            log=lambda _message: None,
        )
        plans[0].update(
            {
                "archetype": "footage",
                "collage_broll": True,
                "footage_src": "../late-stage-collage.mp4",
            }
        )
        raise RuntimeError("downstream collage failed")

    with pytest.raises(RuntimeError, match="downstream collage failed"):
        asyncio.run(fail_after_direction())

    assert planner_calls == 1
    assert (tmp_path / composer.VISUAL_PLAN_CACHE_FILENAME).is_file()
    cached = composer._load_cached_scene_plans(tmp_path, board)
    assert cached == direction

    async def planner_must_not_repeat(*_args, **_kwargs):
        raise AssertionError("planner repeated after downstream failure")

    monkeypatch.setattr(
        composer.visual_plan,
        "plan_scene_visuals",
        planner_must_not_repeat,
    )
    retried = asyncio.run(
        composer._load_or_plan_scene_visuals(
            tmp_path,
            board,
            ai_endpoint=None,
            ai_model=None,
            provider_id=None,
            log=lambda _message: None,
        )
    )
    assert retried == direction


def test_visual_plan_checkpoint_commits_sidecar_last(tmp_path, monkeypatch):
    board = _cache_board()
    _write_cache(tmp_path, board)
    replacement = _cache_plans()
    replacement[0]["headline"] = "Replacement direction"
    real_atomic_write = composer._atomic_write_bytes

    def fail_sidecar(path, payload):
        if path.name == composer.VISUAL_PLAN_CACHE_FILENAME:
            raise RuntimeError("simulated crash before sidecar commit")
        real_atomic_write(path, payload)

    monkeypatch.setattr(composer, "_atomic_write_bytes", fail_sidecar)
    with pytest.raises(RuntimeError, match="simulated crash"):
        composer._write_visual_plan_checkpoint(tmp_path, board, replacement)

    assert not (tmp_path / composer.VISUAL_PLAN_CACHE_FILENAME).exists()
    assert json.loads((tmp_path / "visual_plan.json").read_text())[0]["headline"] == (
        "Replacement direction"
    )
    assert composer._load_cached_scene_plans(tmp_path, board) is None


def test_partial_news_image_error_reports_acquired_attached_and_missing_scenes():
    message = composer._news_image_placement_error(
        {
            "status": "partial",
            "images": [{"scene_id": "scene-02"}, {"scene_id": "scene-04"}, {"scene_id": "scene-05"}],
            "missing_scene_ids": ["scene-03", "scene-07", "scene-09", "scene-10"],
        },
        attached_images=0,
        required_count=7,
    )

    assert "scout acquired 3/7" in message
    assert "manifest status=partial" in message
    assert "missing scenes=scene-03,scene-07,scene-09,scene-10" in message
    assert "0/7 reached final scenes" in message


def test_partial_news_images_are_reported_as_delivery_warning():
    warnings = composer._quality_warnings(
        {"passed": True},
        {"passed": True, "scenes": []},
        news_image_inventory={
            "enabled": True,
            "complete": False,
            "warning": "Licensed image shortfall: 5/8 attached.",
        },
    )

    assert warnings == ["Licensed image shortfall: 5/8 attached."]


def test_quality_failures_become_final_delivery_warnings():
    warnings = composer._quality_warnings(
        {
            "passed": False,
            "failure_reasons": ["word-level transcript unavailable"],
            "transcription_failures": ["mlx-whisper attempt 3: timeout"],
        },
        {
            "passed": False,
            "scenes": [
                {"id": "scene-01", "grounded": True},
                {"id": "scene-02", "grounded": False},
            ],
        },
        {
            "status": "failed",
            "passed": False,
            "failed_scene_ids": ["scene-03"],
            "errors": ["batch 1: malformed response"],
        },
    )

    assert len(warnings) == 3
    assert "estimated timing was used" in warnings[0]
    assert "scene-02" in warnings[1]
    assert "scene-03" in warnings[2]
    assert "malformed response" in warnings[2]


def test_clean_quality_report_has_no_warnings():
    assert composer._quality_warnings(
        {"passed": True},
        {"passed": True, "scenes": []},
        {"status": "passed", "passed": True},
    ) == []


def test_failed_review_defers_delivery_for_retry():
    report = composer._finalize_quality_report(
        {},
        {"passed": True},
        {"passed": True, "scenes": []},
        {
            "status": "failed",
            "passed": False,
            "average_score": 81.42,
            "failed_scene_ids": [],
            "errors": [],
        },
        multimodal_enabled=True,
    )

    assert report["passed"] is False
    assert report["quality_status"] == "retrying"
    assert report["delivery_status"] == "deferred"
    assert "81.42" in report["warnings"][0]


def test_clean_initial_matches_and_grounding_cannot_override_calibration_issues():
    initial_reviews = [
        {
            "id": "scene-01",
            "passed": True,
            "rubric_consistent": True,
            "gemini_verdict": "match",
            "issues": [],
            "score": 76,
        },
        {
            "id": "scene-02",
            "passed": True,
            "rubric_consistent": True,
            "gemini_verdict": "match",
            "issues": [],
            "score": 74,
        },
    ]
    calibrated_reviews = [
        {"id": "scene-01", "passed": False, "score": 78, "issues": ["Add polish."]},
        {"id": "scene-02", "passed": False, "score": 77, "issues": ["Add polish."]},
    ]
    multimodal = {
        "status": "failed",
        "passed": False,
        "scene_count": 2,
        "average_score": 77.5,
        "failed_scene_ids": ["scene-01", "scene-02"],
        "errors": [],
        "batches": [
            {
                "image_received": True,
                "structure_valid": True,
                "contract_valid": True,
            }
        ],
        "calibration": {
            "attempted": True,
            "fallback_to_initial": False,
            "initial_average_score": 75.0,
            "initial_scenes": initial_reviews,
        },
        "scenes": calibrated_reviews,
    }

    report = composer._finalize_quality_report(
        {},
        {"passed": True},
        {
            "passed": True,
            "scenes": [
                {"id": "scene-01", "grounded": True},
                {"id": "scene-02", "grounded": True},
            ],
        },
        multimodal,
        multimodal_enabled=True,
    )

    assert report["passed"] is False
    assert report["quality_status"] == "retrying"
    assert report["delivery_status"] == "deferred"
    assert report["multimodal"]["average_score"] == 77.5
    assert report["multimodal"]["failed_scene_ids"] == ["scene-01", "scene-02"]
    assert report["multimodal"]["scenes"] == calibrated_reviews
    assert "override_reason" not in report["multimodal"]["calibration"]


def test_calibration_disagreement_still_fails_without_clean_initial_evidence():
    multimodal = {
        "status": "failed",
        "passed": False,
        "scene_count": 1,
        "average_score": 76,
        "failed_scene_ids": ["scene-01"],
        "errors": [],
        "batches": [
            {
                "image_received": True,
                "structure_valid": True,
                "contract_valid": True,
            }
        ],
        "calibration": {
            "attempted": True,
            "fallback_to_initial": False,
            "initial_average_score": 72,
            "initial_scenes": [
                {
                    "id": "scene-01",
                    "passed": False,
                    "rubric_consistent": True,
                    "gemini_verdict": "partial",
                    "issues": ["The subject is missing."],
                }
            ],
        },
        "scenes": [{"id": "scene-01", "passed": False, "score": 76}],
    }

    report = composer._finalize_quality_report(
        {},
        {"passed": True},
        {"passed": True, "scenes": [{"id": "scene-01", "grounded": True}]},
        multimodal,
        multimodal_enabled=True,
    )

    assert report["passed"] is False
    assert report["quality_status"] == "retrying"


def test_clean_initial_pixel_matches_still_fail_without_exact_grounding():
    multimodal = {
        "status": "failed",
        "passed": False,
        "scene_count": 1,
        "average_score": 78,
        "failed_scene_ids": ["scene-01"],
        "errors": [],
        "batches": [
            {
                "image_received": True,
                "structure_valid": True,
                "contract_valid": True,
            }
        ],
        "calibration": {
            "attempted": True,
            "fallback_to_initial": False,
            "initial_average_score": 76,
            "initial_scenes": [
                {
                    "id": "scene-01",
                    "passed": True,
                    "rubric_consistent": True,
                    "gemini_verdict": "match",
                    "issues": [],
                }
            ],
        },
        "scenes": [{"id": "scene-01", "passed": False, "score": 78}],
    }

    report = composer._finalize_quality_report(
        {},
        {"passed": True},
        {"passed": False, "scenes": [{"id": "scene-01", "grounded": False}]},
        multimodal,
        multimodal_enabled=True,
    )

    assert report["passed"] is False
    assert report["quality_status"] == "retrying"


def test_failed_quality_report_never_promotes_and_preserves_candidate(tmp_path, monkeypatch):
    previous = tmp_path / "video.mp4"
    staged = tmp_path / "video.next.mp4"
    staged_report = tmp_path / "av_sync_report.next.json"
    previous.write_bytes(b"previous completed cut")
    staged.write_bytes(b"new rejected candidate")
    staged_report.write_text('{"passed": false}', encoding="utf-8")
    promoted = False

    def should_not_promote(*_args, **_kwargs):
        nonlocal promoted
        promoted = True
        raise AssertionError("quality-failed candidate was promoted")

    monkeypatch.setattr(composer, "_promote_render_candidate", should_not_promote)

    with pytest.raises(
        composer.QualityGateRetry,
        match="automatic compose retry 1 requested",
    ):
        composer._promote_quality_gated_candidate(
            staged,
            staged_report,
            tmp_path,
            composer._sha256_path(staged),
            {
                "passed": False,
                "warnings": ["Gemini rendered-frame review average stayed below 82"],
            },
        )

    assert promoted is False
    assert previous.read_bytes() == b"previous completed cut"
    assert staged.read_bytes() == b"new rejected candidate"
    assert staged_report.read_text(encoding="utf-8") == '{"passed": false}'
    retry_state = json.loads(
        (tmp_path / composer.QUALITY_RETRY_STATE_FILENAME).read_text(encoding="utf-8")
    )
    assert retry_state["status"] == "retrying"
    assert retry_state["attempt_count"] == 1


def test_portrait_render_command_uses_task_resolution(tmp_path):
    command = composer._build_render_command(tmp_path, tmp_path / "video.mp4", PORTRAIT)

    assert command[command.index("--resolution") + 1] == "portrait-4k"
    assert command[command.index("--protocol-timeout") + 1] == str(
        composer.config.RENDER_PROTOCOL_TIMEOUT_MS
    )


def test_news_images_exclude_program_bookends_and_existing_footage():
    plans = [
        {'id': 'intro', 'archetype': 'statement'},
        {'id': 'news-video', 'archetype': 'footage'},
        {'id': 'news-image', 'archetype': 'statement'},
        {'id': 'outro', 'archetype': 'statement'},
    ]
    board = {'scenes': [
        {'id': 'intro', 'program_segment_kind': 'opening'},
        {'id': 'news-video', 'program_segment_kind': 'news'},
        {'id': 'news-image', 'program_segment_kind': 'news'},
        {'id': 'outro', 'program_segment_kind': 'closing'},
    ]}
    assert composer._eligible_news_image_scene_ids(plans, board) == {'news-image'}
    assert composer._eligible_news_image_scene_ids(plans, {}) == {'intro', 'news-image', 'outro'}

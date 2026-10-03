import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest
from PIL import Image

from backend.pipeline import multimodal_review, production_plan
from backend.pipeline.opencli import OpenCLIResult


def _shot():
    return {
        "kind": "three_d", "start": 0, "duration": 8,
        "src": "../three_d/derived/clip.mp4", "sha256": "clip-hash",
        "copy": ["A layered processor"], "layout": "focus",
        "direction": {"subject": "Processor", "action": "Separate the illustrative layers",
                      "purpose": "Show the layered structure", "evidence_kind": "illustration"},
        "motion_cues": {"reveal_start": .1, "reveal_end": .65, "emphasis": 1.3, "settle": 1.75},
        "three_d_spec_sha256": "spec-hash", "three_d_renderer_sha256": "renderer-hash",
        "three_d_spec": {
            "contract": "three-d-explainer-v1", "representation": "illustration",
            "fps": 30, "duration_frames": 240,
            "camera": {"position": [0, 3, 8], "target": [0, 0, 0], "end_position": [2, 3, 8]},
            "actors": [{"id": "processor", "kind": "chip", "position": [0, 0, 0],
                        "action": "explode", "explode": 0, "end_explode": 1,
                        "motion_start": .12, "motion_end": .72}],
            "semantic_intent": "Separate layers to explain structure without measured dimensions",
        },
    }


def _scene():
    return {"id": "scene-01", "start": 0, "duration": 8,
            "text": "The processor has a layered structure.",
            "lines": [{"start": 0, "duration": 8, "text": "The processor has a layered structure."}]}


def _payload(score=90, mismatches=None):
    return {"image_received": True, "reviews": [{
        "id": "scene-01", "score": score, "verdict": "match" if score >= 70 else "partial",
        "visual_summary": "A processor with separated layers",
        "alignment_reason": "Visible layers match the narrated structure",
        "issues": [] if score >= 70 else ["scene-01-shot-01: geometry is obscured by the text"],
        "suggested_visual": "" if score >= 70 else "Reframe the geometry above the text",
        "source_media_mismatches": mismatches or [],
    }]}


def _opencli_payload(payload):
    return OpenCLIResult(args=(), returncode=0,
                         stdout=json.dumps([{"response": json.dumps(payload)}]), stderr="")


def _frames(tmp_path, shot):
    (tmp_path / "visual_plan.json").write_text(json.dumps([{"id": "scene-01", "media_shots": [shot]}]))

    async def extract(video, path, timestamp):
        Image.new("RGB", (640, 360), "navy").save(path)

    with patch.object(multimodal_review, "_extract_frame", AsyncMock(side_effect=extract)):
        return asyncio.run(multimodal_review.extract_scene_frames(
            tmp_path / "video.next.mp4", {"scenes": [_scene()]}, tmp_path / "review",
        ))


def test_three_d_timeline_binds_geometry_and_samples_the_actual_motion_window():
    shot = _shot()
    board = {"total_duration": 8, "scenes": [_scene()]}
    timeline = production_plan.build_timeline(board, [{"id": "scene-01", "media_shots": [shot]}], fps=30)
    context = timeline["shots"][0]["three_d"]
    assert timeline["three_d_shot_count"] == 1
    assert context["representation"] == "illustration"
    assert context["camera"]["end_position"] == [2, 3, 8]
    assert context["actors"][0]["end_explode"] == 1
    assert context["spec_sha256"] == "spec-hash"
    phases = dict(production_plan.three_d_sample_times(shot))
    assert phases["onset"] == .25
    assert phases["action"] == pytest.approx(3.346)
    assert phases["settled"] == pytest.approx(5.769333)
    previews = production_plan.preview_times(timeline)
    assert len(previews) == 4
    assert any(value > 5.7 for value in previews)
    assert all(value < 8 for value in previews)
    assert {e["name"] for e in timeline["events"]} >= {"three_d_onset", "three_d_action", "three_d_settled"}
    original_hash = timeline["input_sha256"]
    shot["three_d_spec"]["camera"]["end_position"] = [4, 3, 8]
    changed = production_plan.build_timeline(board, [{"id": "scene-01", "media_shots": [shot]}], fps=30)
    assert changed["input_sha256"] != original_hash
    assert context["camera"]["end_position"] == [2, 3, 8]


def test_three_d_hold_actor_still_controls_camera_settle_and_end_stays_inside_shot():
    shot = _shot()
    shot["three_d_spec"]["actors"][0].update(action="hold", motion_end=1)
    phases = dict(production_plan.three_d_sample_times(shot))
    assert phases["settled"] == pytest.approx(239 / 30, abs=1e-6)
    assert phases["settled"] < shot["duration"]


@pytest.mark.parametrize("defect", ["no_spec", "captured", "no_camera", "frames", "duration", "motion", "stale_cues"])
def test_three_d_missing_or_stale_render_intent_fails_closed(defect):
    shot = _shot()
    if defect == "no_spec": shot.pop("three_d_spec")
    if defect == "captured": shot["three_d_spec"]["representation"] = "captured_evidence"
    if defect == "no_camera": shot["three_d_spec"].pop("camera")
    if defect == "frames": shot["three_d_spec"]["duration_frames"] = True
    if defect == "duration": shot["duration"] = 9
    if defect == "motion": shot["three_d_spec"]["actors"][0]["motion_end"] = float("nan")
    if defect == "stale_cues": shot["three_d_motion_cues"] = {"motion_start": .1, "action": 1, "settle": 1.75}
    with pytest.raises(ValueError):
        production_plan.three_d_sample_times(shot)


def test_three_d_final_samples_keep_midpoint_and_expose_geometry_as_illustration(tmp_path):
    frames = _frames(tmp_path, _shot())
    samples = frames[0]["samples"]
    assert {sample["phase"] for sample in samples} == {"onset", "action", "midpoint", "settled"}
    midpoint = next(sample for sample in samples if sample["phase"] == "midpoint")
    assert midpoint["id"] == "scene-01-shot-01" and midpoint["timestamp"] == 4
    assert all(sample["shot_sha256"] == multimodal_review.shot_fingerprint(_shot()) for sample in samples)
    prompt = multimodal_review._review_prompt("Processor", frames, 70, 82)
    assert '"representation": "illustration"' in prompt
    assert '"end_explode": 1' in prompt
    assert "camera framing" in prompt and "not captured footage" in prompt
    assert "geometry must remain visible behind the final text overlays" in prompt


def test_three_d_failure_cannot_reject_captured_media(tmp_path):
    frames = _frames(tmp_path, _shot())
    invalid = multimodal_review.normalise_batch(_payload(60, [{
        "sample_id": "scene-01-shot-01", "reason": "Layers are not visible",
    }]), frames, 70)
    assert invalid["contract_valid"] is False
    assert invalid["reviews"][0]["rejected_media_samples"] == []
    repair = multimodal_review.normalise_batch(_payload(60), frames, 70)
    assert repair["contract_valid"] is True
    assert repair["reviews"][0]["passed"] is False
    assert repair["reviews"][0]["rejected_media_samples"] == []


def test_three_d_camera_change_invalidates_review_cache_with_identical_pixels(tmp_path):
    frames = _frames(tmp_path, _shot())
    sheet = multimodal_review.create_contact_sheet(frames, tmp_path / "sheet.jpg")
    call = AsyncMock(return_value=_opencli_payload(_payload()))
    args = dict(title="Processor", batch_frames=frames, sheet=sheet, match_floor=70,
                minimum_average_score=82, timeout=120, maximum_retries=0,
                batch_index=1, phase="initial", log=None)
    with patch.object(multimodal_review, "run_opencli", call):
        asyncio.run(multimodal_review._review_batch(**args))
        asyncio.run(multimodal_review._review_batch(**args))
        assert call.await_count == 1
        frames[0]["planned_shots"][0]["three_d"]["camera"]["end_position"] = [4, 3, 8]
        asyncio.run(multimodal_review._review_batch(**args))
        assert call.await_count == 2


def test_final_review_reports_actual_three_d_participation(tmp_path):
    frames = _frames(tmp_path, _shot())
    with (
        patch.object(multimodal_review, "extract_scene_frames", AsyncMock(return_value=frames)),
        patch.object(multimodal_review, "run_opencli", AsyncMock(return_value=_opencli_payload(_payload()))),
    ):
        report = asyncio.run(multimodal_review.review_video(
            tmp_path / "video.next.mp4", {"title": "Processor", "scenes": [_scene()]}, tmp_path,
        ))
    assert report["passed"] is True
    assert report["three_d_sampling_contract"] == multimodal_review.THREE_D_SAMPLING_CONTRACT
    assert report["reviewed_three_d_shots"] == 1
    assert report["three_d_sample_count"] == 4

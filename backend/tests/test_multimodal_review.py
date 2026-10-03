import asyncio
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

import pytest
from PIL import Image

from backend import config
from backend.pipeline import multimodal_review
from backend.pipeline.opencli import OpenCLIError, OpenCLIResult


def test_bookend_review_uses_spoken_scope_without_dropping_score_thresholds():
    prompt = multimodal_review._review_prompt(
        "A headline from a later story",
        [{"id": "scene-01", "scene": {
            "text": "Good morning. This is ByteFront Espresso.",
            "start": 0, "duration": 6, "program_segment_kind": "opening",
        }}],
        70, 82,
    )
    assert '"program_segment_kind": "opening"' in prompt
    assert "not a required claim in every scene" in prompt
    assert "unless that opening narration actually introduces it" in prompt
    assert "missing or contradictory claim" in prompt
    assert "match requires score 70-100 and issues=[]" in prompt
    assert "release average is 82/100" in prompt


def _scene(scene_id: str, start: float, text: str) -> dict:
    return {
        "id": scene_id,
        "start": start,
        "duration": 8.0,
        "text": text,
        "lines": [{"start": start, "duration": 8.0, "text": text}],
    }


def _frames(root: Path, scenes: list[dict]) -> list[dict]:
    frames = []
    for index, scene in enumerate(scenes):
        path = root / f"{scene['id']}.jpg"
        Image.new("RGB", (640, 360), (40 + index * 20, 70, 100)).save(path)
        frames.append(
            {
                "id": scene["id"],
                "timestamp": scene["start"] + scene["duration"] / 2,
                "path": path,
                "scene": scene,
            }
        )
    return frames


def _opencli_result(payload: dict) -> OpenCLIResult:
    return OpenCLIResult(
        args=(),
        returncode=0,
        stdout=json.dumps([{"response": json.dumps(payload)}]),
        stderr="",
    )


def _matching_payload(scenes: list[dict], score: int) -> dict:
    return {
        "image_received": True,
        "reviews": [
            {
                "id": scene["id"],
                "score": score,
                "verdict": "match",
                "visual_summary": f"Visible subject for {scene['id']}",
                "alignment_reason": "The visible subject matches the narration.",
                "issues": [],
                "suggested_visual": "",
            }
            for scene in scenes
        ],
    }


def test_contact_sheet_labels_and_compacts_frames(tmp_path: Path):
    scenes = [
        _scene("scene-01", 0, "Gold reaches a record price."),
        _scene("scene-02", 8, "Central banks increase reserves."),
        _scene("scene-03", 16, "Investors react to fear."),
    ]
    output = multimodal_review.create_contact_sheet(
        _frames(tmp_path, scenes), tmp_path / "sheet.jpg"
    )

    with Image.open(output) as sheet:
        assert sheet.width == 2 * multimodal_review.FRAME_WIDTH + multimodal_review.SHEET_GAP
        assert sheet.height == 2 * (
            multimodal_review.FRAME_HEIGHT + multimodal_review.HEADER_HEIGHT
        ) + multimodal_review.SHEET_GAP
    assert output.stat().st_size <= 750_000


def test_review_samples_every_timed_shot_and_preserves_whole_scene_context(tmp_path):
    scene = _scene("scene-09", 100, "Motor specs. Price and range. Charging and job cuts.")
    opening = _scene("scene-01", 0, "Welcome.")
    (tmp_path / "visual_plan.json").write_text(json.dumps([{
        "id": "scene-09", "media_shots": [
            {"start": 0, "duration": 2}, {"start": 2, "duration": 4},
            {"start": 6, "duration": 2},
        ],
    }]))

    async def extract(video, path, timestamp):
        Image.new("RGB", (640, 360), "white").save(path)

    with patch.object(multimodal_review, "_extract_frame", AsyncMock(side_effect=extract)) as call:
        frames = asyncio.run(multimodal_review.extract_scene_frames(
            tmp_path / "video.next.mp4", {"scenes": [opening, scene]}, tmp_path / "review",
        ))
    assert call.await_count == 4
    assert frames[0]["timestamp"] == 4
    assert [sample["timestamp"] for sample in frames[1]["samples"]] == [101, 104, 107]
    assert frames[1]["samples"][0]["shot_sha256"] == multimodal_review.shot_fingerprint({"start": 0, "duration": 2})
    assert frames[1]["scene"]["text"] == scene["text"]
    sheet = multimodal_review.create_contact_sheet(frames, tmp_path / "sheet.jpg")
    with Image.open(sheet) as image:
        assert image.height == 2 * (multimodal_review.FRAME_HEIGHT + multimodal_review.HEADER_HEIGHT) + multimodal_review.SHEET_GAP
    prompt = multimodal_review._review_prompt("EV news", frames, 70, 82)
    assert scene["text"] in prompt
    assert "scene-09-shot-03" in prompt
    assert "every fact simultaneously in every frame" in prompt
    assert "missing from the entire sequence or contradicted by any frame" in prompt
    assert "source_media_mismatches" in prompt
    assert "incorrect overlay copy" in prompt


@pytest.mark.parametrize("mutation", [None, "unknown_sample", "duplicate", "editorial", "blank_reason", "match", "malformed", "no_image", "missing_scene"])
def test_source_rejection_requires_explicit_valid_sample_attribution(tmp_path, mutation):
    scenes = [_scene("scene-09", 0, "Robot doing kitchen chores.")]
    frames = _frames(tmp_path, scenes)
    sample = {"id": "scene-09-shot-02", "timestamp": 5, "shot_sha256": "local-shot",
              "media": {"src": "../robot.mp4", "sha256": "local-asset", "kind": "public_footage"}}
    frames[0]["samples"] = [sample]
    payload = _matching_payload(scenes, 55)
    row = payload["reviews"][0]
    row.update(verdict="partial", issues=["Warehouse instead of kitchen"], suggested_visual="Kitchen footage",
               source_media_mismatches=[{"sample_id": sample["id"], "reason": "Source shows warehouse boxes",
                                         "src": "model-must-not-select-a-path", "sha256": "invented"}])
    if mutation == "unknown_sample": row["source_media_mismatches"][0]["sample_id"] = "scene-08-shot-02"
    if mutation == "duplicate": row["source_media_mismatches"] *= 2
    if mutation == "editorial": sample["media"]["kind"] = "editorial"
    if mutation == "blank_reason": row["source_media_mismatches"][0]["reason"] = " "
    if mutation == "match": row.update(score=90, verdict="match", issues=[])
    if mutation == "malformed": row["source_media_mismatches"] = "shot 02"
    if mutation == "no_image": payload["image_received"] = False
    if mutation == "missing_scene": frames.append({**frames[0], "id": "scene-10"})
    result = multimodal_review.normalise_batch(payload, frames, 70)
    assert result["contract_valid"] is (mutation is None)
    rejected = result["reviews"][0]["rejected_media_samples"]
    if mutation:
        assert rejected == []
    else:
        assert rejected == [{"sample_id": sample["id"], "timestamp": 5, "shot_sha256": "local-shot",
                             **sample["media"], "reason": "Source shows warehouse boxes"}]
        assert result["reviews"][0]["passed"] is False


@pytest.mark.parametrize("shots", [
    [{"start": 1, "duration": 7}],
    [{"start": 0, "duration": 7}],
    [{"start": 0, "duration": 9}],
    [{"start": 0, "duration": 0}],
    [{"start": 0, "duration": float("nan")}],
])
def test_incomplete_or_invalid_shot_evidence_fails_closed(tmp_path, shots):
    (tmp_path / "visual_plan.json").write_text(json.dumps([{"id": "scene-09", "media_shots": shots}]))
    with patch.object(multimodal_review, "_extract_frame", AsyncMock()):
        with pytest.raises(ValueError, match="review shot coverage"):
            asyncio.run(multimodal_review.extract_scene_frames(
                tmp_path / "video.next.mp4", {"scenes": [_scene("scene-09", 0, "Specs.")]}, tmp_path / "review",
            ))


def test_contact_sheet_budget_never_splits_or_truncates_a_scene_sequence():
    frames = [{"id": f"scene-{index}", "samples": [{}] * count} for index, count in enumerate([1, 4, 7, 9, 1])]
    batches = multimodal_review._frame_batches(frames, 8)
    assert [[frame["id"] for frame in batch] for batch in batches] == [
        ["scene-0", "scene-1"], ["scene-2"], ["scene-3"], ["scene-4"],
    ]
    assert [frame for batch in batches for frame in batch] == frames


def test_editorial_motion_samples_preserve_midpoints_and_source_attribution(tmp_path):
    scene = _scene("scene-09", 100, "A cause and its measured effect.")
    cues = {"reveal_start": 0.2, "reveal_end": 1, "emphasis": 2.3, "settle": 3.4}
    shots = [
        {"start": 0, "duration": 4, "kind": "editorial", "motion_cues": cues},
        {"start": 4, "duration": 4, "kind": "public_footage", "src": "source.mp4",
         "sha256": "source-hash", "motion_cues": cues},
    ]
    (tmp_path / "visual_plan.json").write_text(json.dumps([{"id": "scene-09", "media_shots": shots}]))
    with patch.object(multimodal_review, "_extract_frame", AsyncMock()) as extract:
        frames = asyncio.run(multimodal_review.extract_scene_frames(
            tmp_path / "video.next.mp4", {"scenes": [scene]}, tmp_path / "review",
        ))
    samples = frames[0]["samples"]
    assert extract.await_count == 4
    assert [(sample["id"], sample["timestamp"]) for sample in samples] == [
        ("scene-09-shot-01-action", 100.6), ("scene-09-shot-01", 102),
        ("scene-09-shot-01-settled", 103.4), ("scene-09-shot-02", 106),
    ]
    assert samples[-1]["shot_sha256"] == multimodal_review.shot_fingerprint(shots[-1])
    assert samples[-1]["media"] == {"kind": "public_footage", "src": "source.mp4", "sha256": "source-hash"}
    prompt = multimodal_review._review_prompt("Causes", frames, 70, 82)
    assert "partial reveal" in prompt and "Contradictions at any phase still count" in prompt


def test_review_binds_selected_shot_direction_to_its_actual_sample_window(tmp_path):
    scene = _scene("scene-06", 100, "Blue light steers the swimmer through a maze.")
    direction = {
        "purpose": "Explain how the swimmer is steered", "subject": "The swimmer",
        "action": "Trace the swimmer through an illustrated maze with an external blue beam",
        "continuity": "Keep the same swimmer identifiable", "evidence_kind": "editorial",
        "motion": "reveal",
    }
    shot = {
        "start": 0, "duration": 8, "kind": "editorial", "layout": "focus",
        "copy": ["Blue light steers the swimmer"], "direction": direction,
        "relationship_excerpt": scene["text"],
        "motion_cues": {"reveal_start": .2, "reveal_end": 1, "emphasis": 2, "settle": 3},
    }
    (tmp_path / "visual_plan.json").write_text(json.dumps([
        {"id": "scene-06", "media_shots": [shot]},
        {"id": "unrelated", "media_shots": [{"direction": {"action": "Unrelated intent"}}]},
    ]))
    with patch.object(multimodal_review, "_extract_frame", AsyncMock()):
        frames = asyncio.run(multimodal_review.extract_scene_frames(
            tmp_path / "video.next.mp4", {"scenes": [scene]}, tmp_path / "review",
        ))
    prompt = multimodal_review._review_prompt("Robotics", frames, 70, 82)
    segment = json.loads(prompt.split("Segments: ", 1)[1].split("\n\n", 1)[0])[0]
    assert segment["planned_shots"] == [{
        "id": "scene-06-shot-01", "start_seconds": 100, "end_seconds": 108,
        "direction": direction, "layout": "focus", "selected_copy": shot["copy"],
        "relationship_excerpt": scene["text"],
    }]
    assert {sample["shot_id"] for sample in segment["rendered_samples"]} == {"scene-06-shot-01"}
    assert {sample["phase"] for sample in segment["rendered_samples"]} == {"action", "settled", "midpoint"}
    assert "Unrelated intent" not in prompt
    assert "Accurate text alone cannot substitute" in prompt
    assert "can be satisfied by a legible text card" in prompt
    assert "and the supplied direction is fulfilled" in prompt
    assert "return partial or mismatch" in prompt
    assert "match requires score 70-100 and issues=[]" in prompt
    assert "release average is 82/100" in prompt


def test_unrealized_direction_is_an_editorial_failure_without_rejecting_source_media(tmp_path):
    scenes = [_scene("scene-06", 0, "Blue light steers the swimmer through a maze.")]
    frames = _frames(tmp_path, scenes)
    payload = _matching_payload(scenes, 60)
    payload["reviews"][0].update(
        verdict="partial", issues=["scene-06-shot-01: promised maze and blue beam are absent"],
        suggested_visual="Render the planned maze and blue beam around the swimmer",
        source_media_mismatches=[],
    )
    result = multimodal_review.normalise_batch(payload, frames, 70)
    assert result["contract_valid"] is True
    assert result["reviews"][0]["passed"] is False
    assert result["reviews"][0]["rejected_media_samples"] == []


@pytest.mark.parametrize("cues", [
    {}, [], {"reveal_start": 0, "reveal_end": 2, "emphasis": 1, "settle": 3},
    {"reveal_start": 0, "reveal_end": 1, "emphasis": 2, "settle": 4},
    {"reveal_start": 0, "reveal_end": 1, "emphasis": 2, "settle": float("nan")},
])
def test_review_rejects_malformed_editorial_motion_evidence(cues):
    with pytest.raises(ValueError, match="motion cues"):
        multimodal_review._editorial_sample_times({"kind": "editorial", "duration": 4, "motion_cues": cues})


@pytest.mark.parametrize("verdict", ["match", "partial", "omitted"])
def test_sequence_review_still_requires_complete_contract_valid_scene_results(tmp_path, verdict):
    scenes = [_scene("scene-09", 0, "Motor specs. Price and range. Charging and jobs.")]
    frames = _frames(tmp_path, scenes)
    frames[0]["samples"] = [
        {"id": f"scene-09-shot-{index:02d}", "timestamp": index, "path": frames[0]["path"]}
        for index in range(1, 5)
    ]
    payload = _matching_payload(scenes, 90)
    if verdict == "partial":
        payload["reviews"][0].update(score=55, verdict="partial", issues=["Range absent from every shot"], suggested_visual="Add the narrated range")
    elif verdict == "omitted":
        payload["reviews"] = []
    with (
        patch.object(multimodal_review, "extract_scene_frames", AsyncMock(return_value=frames)),
        patch.object(multimodal_review, "run_opencli", AsyncMock(return_value=_opencli_result(payload))),
        patch.object(config, "AV_SYNC_GEMINI_MAX_RETRIES", 0),
        patch.object(config, "AV_SYNC_REVIEW_FALLBACK_PROVIDER", ""),
    ):
        report = asyncio.run(multimodal_review.review_video(tmp_path / "video.mp4", {"scenes": scenes}, tmp_path))
    assert report["passed"] is (verdict == "match")
    assert report["sampling_contract"] == multimodal_review.SAMPLING_CONTRACT
    assert report["reviewed_scenes"] == 1
    assert report["scenes"][0]["sample_timestamps"] == [1, 2, 3, 4]
    if verdict == "partial":
        assert report["failed_scene_ids"] == ["scene-09"]
        assert report["average_score"] == 55


def test_missing_or_unsubstantiated_gemini_rows_fail_closed(tmp_path: Path):
    scenes = [
        _scene("scene-01", 0, "Gold reaches a record price."),
        _scene("scene-02", 8, "Central banks increase reserves."),
    ]
    frames = _frames(tmp_path, scenes)

    normalized = multimodal_review.normalise_batch(
        {
            "image_received": True,
            "reviews": [
                {
                    "id": "scene-01",
                    "score": 95,
                    "verdict": "match",
                    "visual_summary": "A gold bar and a record-price chart.",
                    "alignment_reason": "The visible chart directly supports the claim.",
                    "issues": [],
                }
            ],
        },
        frames,
        minimum_scene_score=70,
    )

    assert normalized["reviews"][0]["passed"] is True
    assert normalized["reviews"][1]["score"] == 0
    assert normalized["reviews"][1]["verdict"] == "mismatch"
    assert normalized["structure_valid"] is False
    assert normalized["contract_valid"] is False


def test_score_verdict_contradiction_invalidates_review_contract(tmp_path: Path):
    scenes = [_scene("scene-01", 0, "Gold reaches a record price.")]
    normalized = multimodal_review.normalise_batch(
        {
            "image_received": True,
            "reviews": [
                {
                    "id": "scene-01",
                    "score": 68,
                    "verdict": "match",
                    "visual_summary": "A gold bar and a record-price chart.",
                    "alignment_reason": "The visible subject directly matches.",
                    "issues": [],
                    "suggested_visual": "",
                }
            ],
        },
        _frames(tmp_path, scenes),
        minimum_scene_score=70,
    )

    assert normalized["structure_valid"] is True
    assert normalized["contract_valid"] is False
    assert normalized["reviews"][0]["rubric_consistent"] is False


def test_malformed_issue_field_cannot_disappear_into_a_valid_match(tmp_path: Path):
    scenes = [_scene("scene-01", 0, "Two distinct subjects must both be visible.")]
    normalized = multimodal_review.normalise_batch(
        {
            "image_received": True,
            "reviews": [
                {
                    "id": "scene-01",
                    "score": 90,
                    "verdict": "match",
                    "visual_summary": "Only the first subject is visible.",
                    "alignment_reason": "The second subject is missing.",
                    "issues": "Missing second subject",
                    "suggested_visual": "Show both subjects.",
                }
            ],
        },
        _frames(tmp_path, scenes),
        minimum_scene_score=70,
    )

    assert normalized["structure_valid"] is True
    assert normalized["contract_valid"] is False
    assert normalized["reviews"][0]["rubric_consistent"] is False


def test_response_payload_can_recover_json_wrapper_from_stderr():
    payload = _matching_payload([_scene("scene-01", 0, "Gold rises.")], 90)
    wrapped = json.dumps([{"response": json.dumps(payload)}])

    assert multimodal_review._response_payload(f"plain stdout\n{wrapped}") == payload


def test_response_payload_recovers_complete_envelope_after_abandoned_scene_prefix():
    scenes = [_scene("scene-01", 0, "Gold rises."), _scene("scene-02", 8, "Rates fall.")]
    payload = _matching_payload(scenes, 90)
    prefix = '{"image_received":true,"reviews":[' + json.dumps(payload["reviews"][0]) + ",\n"
    wrapped = json.dumps([{"response": prefix + "```json\n" + json.dumps(payload)}])
    assert multimodal_review._response_payload(wrapped) == payload


def test_response_payload_does_not_choose_between_conflicting_complete_reviews():
    scenes = [_scene("scene-01", 0, "Gold rises.")]
    first = _matching_payload(scenes, 30)
    second = _matching_payload(scenes, 90)
    wrapped = json.dumps([{"response": json.dumps(first) + "\n" + json.dumps(second)}])
    with pytest.raises(OpenCLIError, match="exactly one complete review"):
        multimodal_review._response_payload(wrapped)


def test_recovered_review_still_rejects_missing_image_and_scenes(tmp_path):
    scenes = [_scene("scene-01", 0, "Gold rises."), _scene("scene-02", 8, "Rates fall.")]
    payload = _matching_payload(scenes[:1], 90)
    payload["image_received"] = False
    wrapped = json.dumps([{"response": '{"abandoned":' + json.dumps(payload)}])
    recovered = multimodal_review._response_payload(wrapped)
    normalized = multimodal_review.normalise_batch(recovered, _frames(tmp_path, scenes), 70)
    assert not multimodal_review._batch_is_valid(normalized)


class ReviewVideoTests(unittest.IsolatedAsyncioTestCase):
    async def test_keyframe_extraction_retries_transient_failure(self):
        extract_once = AsyncMock(side_effect=[RuntimeError("busy"), None])
        with (
            patch.object(multimodal_review, "_extract_frame_once", extract_once),
            patch.object(config, "AV_SYNC_FRAME_MAX_RETRIES", 1),
        ):
            await multimodal_review._extract_frame(
                Path("video.mp4"), Path("frame.jpg"), 12.5
            )

        self.assertEqual(extract_once.await_count, 2)

    async def test_complete_gemini_review_passes_and_uses_file_upload(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            scenes = [
                _scene("scene-01", 0, "Gold reaches a record price."),
                _scene("scene-02", 8, "Central banks increase reserves."),
            ]
            frames = _frames(root, scenes)
            payload = {
                "image_received": True,
                "reviews": [
                    {
                        "id": scene["id"],
                        "score": 90,
                        "verdict": "match",
                        "visual_summary": f"Visible subject for {scene['id']}",
                        "alignment_reason": "The visible subject matches the narration.",
                        "issues": [],
                        "suggested_visual": "",
                    }
                    for scene in scenes
                ],
            }
            result = OpenCLIResult(
                args=(),
                returncode=0,
                stdout=json.dumps([{"response": "💬 " + json.dumps(payload)}]),
                stderr="",
            )
            opencli = AsyncMock(return_value=result)
            with (
                patch.object(
                    multimodal_review,
                    "extract_scene_frames",
                    AsyncMock(return_value=frames),
                ),
                patch.object(multimodal_review, "run_opencli", opencli),
                patch.object(config, "AV_SYNC_GEMINI_BATCH_SIZE", 8),
                patch.object(config, "AV_SYNC_GEMINI_MIN_SCENE_SCORE", 70),
                patch.object(config, "AV_SYNC_GEMINI_MIN_AVERAGE_SCORE", 82),
                patch.object(config, "AV_SYNC_GEMINI_TIMEOUT", 120),
                patch.object(config, "AV_SYNC_GEMINI_MAX_RETRIES", 1),
                patch.object(config, "AV_SYNC_REVIEW_FALLBACK_PROVIDER", ""),
            ):
                report = await multimodal_review.review_video(
                    root / "video.mp4",
                    {"title": "Gold", "scenes": scenes},
                    root,
                )

            self.assertTrue(report["passed"])
            self.assertEqual(report["average_score"], 90)
            self.assertFalse(report["calibration"]["attempted"])
            self.assertEqual(report["calibration"]["initial_average_score"], 90)
            self.assertEqual(report["calibration"]["match_floor"], 82)
            self.assertEqual(opencli.await_count, 1)
            args = opencli.await_args.args[0]
            self.assertIn("--file", args)
            self.assertIn("foreground", args)
            self.assertTrue((root / "multimodal_review" / "contact-sheet-01.jpg").is_file())

    async def test_branded_bookends_are_excluded_from_narration_match_review(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            narrated = _scene("scene-01", 6, "A narrated technology story.")
            storyboard = {
                "title": "ByteFront Espresso",
                "scenes": [
                    {
                        "id": "scene-intro",
                        "start": 0.0,
                        "duration": 6.0,
                        "text": "Branded intro",
                        "lines": [],
                    },
                    narrated,
                    {
                        "id": "scene-outro",
                        "start": 14.0,
                        "duration": 6.0,
                        "text": "Branded outro",
                        "lines": [],
                    },
                ],
            }
            frames = _frames(root, [narrated])
            extract = AsyncMock(return_value=frames)
            opencli = AsyncMock(return_value=_opencli_result(_matching_payload([narrated], 90)))
            with (
                patch.object(multimodal_review, "extract_scene_frames", extract),
                patch.object(multimodal_review, "run_opencli", opencli),
                patch.object(config, "AV_SYNC_GEMINI_BATCH_SIZE", 8),
                patch.object(config, "AV_SYNC_GEMINI_MIN_SCENE_SCORE", 70),
                patch.object(config, "AV_SYNC_GEMINI_MIN_AVERAGE_SCORE", 82),
                patch.object(config, "AV_SYNC_GEMINI_TIMEOUT", 120),
                patch.object(config, "AV_SYNC_GEMINI_MAX_RETRIES", 0),
                patch.object(config, "AV_SYNC_REVIEW_FALLBACK_PROVIDER", ""),
            ):
                report = await multimodal_review.review_video(
                    root / "video.mp4",
                    storyboard,
                    root,
                )

            review_storyboard = extract.await_args.args[1]
            self.assertEqual([scene["id"] for scene in review_storyboard["scenes"]], ["scene-01"])
            self.assertTrue(report["passed"])
            self.assertEqual(report["scene_count"], 1)

    async def test_malformed_gemini_response_is_retried_once(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            scenes = [_scene("scene-01", 0, "Gold reaches a record price.")]
            frames = _frames(root, scenes)
            valid_payload = {
                "image_received": True,
                "reviews": [
                    {
                        "id": "scene-01",
                        "score": 90,
                        "verdict": "match",
                        "visual_summary": "A gold bar and a rising price chart.",
                        "alignment_reason": "The visible subject matches the narration.",
                        "issues": [],
                    }
                ],
            }
            opencli = AsyncMock(
                side_effect=[
                    OpenCLIResult(
                        args=(), returncode=0, stdout="not JSON", stderr=""
                    ),
                    OpenCLIResult(
                        args=(),
                        returncode=0,
                        stdout=json.dumps([{"response": json.dumps(valid_payload)}]),
                        stderr="",
                    ),
                ]
            )
            with (
                patch.object(
                    multimodal_review,
                    "extract_scene_frames",
                    AsyncMock(return_value=frames),
                ),
                patch.object(multimodal_review, "run_opencli", opencli),
                patch.object(config, "AV_SYNC_GEMINI_BATCH_SIZE", 8),
                patch.object(config, "AV_SYNC_GEMINI_MIN_SCENE_SCORE", 70),
                patch.object(config, "AV_SYNC_GEMINI_MIN_AVERAGE_SCORE", 82),
                patch.object(config, "AV_SYNC_GEMINI_TIMEOUT", 120),
                patch.object(config, "AV_SYNC_GEMINI_MAX_RETRIES", 1),
            ):
                report = await multimodal_review.review_video(
                    root / "video.mp4",
                    {"title": "Gold", "scenes": scenes},
                    root,
                )

            self.assertTrue(report["passed"])
            self.assertEqual(opencli.await_count, 2)
            self.assertEqual(report["batches"][0]["attempts"], 2)
            self.assertFalse(report["calibration"]["attempted"])

    async def test_real_partial_match_never_enters_aggregate_calibration(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            scenes = [
                _scene("scene-01", 0, "Gold reaches a record price."),
                _scene("scene-02", 8, "Central banks increase reserves."),
            ]
            frames = _frames(root, scenes)
            payload = {
                "image_received": True,
                "reviews": [
                    {
                        "id": "scene-01",
                        "score": 75,
                        "verdict": "match",
                        "visual_summary": "Gold bars and a price chart.",
                        "alignment_reason": "Direct match.",
                        "issues": [],
                    },
                    {
                        "id": "scene-02",
                        "score": 65,
                        "verdict": "partial",
                        "visual_summary": "A generic bank building.",
                        "alignment_reason": "The reserve increase is not visible.",
                        "issues": ["The reserve increase is not visible."],
                        "suggested_visual": "Show a reserve ledger rising beside the bank.",
                    },
                ],
            }
            opencli = AsyncMock(return_value=_opencli_result(payload))
            with (
                patch.object(
                    multimodal_review,
                    "extract_scene_frames",
                    AsyncMock(return_value=frames),
                ),
                patch.object(multimodal_review, "run_opencli", opencli),
                patch.object(config, "AV_SYNC_GEMINI_BATCH_SIZE", 8),
                patch.object(config, "AV_SYNC_GEMINI_MIN_SCENE_SCORE", 70),
                patch.object(config, "AV_SYNC_GEMINI_MIN_AVERAGE_SCORE", 82),
                patch.object(config, "AV_SYNC_GEMINI_TIMEOUT", 120),
                patch.object(config, "AV_SYNC_GEMINI_MAX_RETRIES", 0),
                patch.object(config, "AV_SYNC_REVIEW_FALLBACK_PROVIDER", ""),
            ):
                report = await multimodal_review.review_video(
                    root / "video.mp4",
                    {"title": "Gold", "scenes": scenes},
                    root,
                )

            self.assertFalse(report["passed"])
            self.assertEqual(report["average_score"], 70)
            self.assertEqual(report["failed_scene_ids"], ["scene-02"])
            self.assertFalse(report["calibration"]["attempted"])
            self.assertEqual(opencli.await_count, 1)

    async def test_clean_low_average_gets_one_strict_calibration_round_and_passes(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            scenes = [_scene("scene-01", 0, "Gold reaches a record price.")]
            frames = _frames(root, scenes)
            opencli = AsyncMock(
                side_effect=[
                    _opencli_result(_matching_payload(scenes, 80)),
                    _opencli_result(_matching_payload(scenes, 88)),
                ]
            )
            with (
                patch.object(
                    multimodal_review,
                    "extract_scene_frames",
                    AsyncMock(return_value=frames),
                ),
                patch.object(multimodal_review, "run_opencli", opencli),
                patch.object(config, "AV_SYNC_GEMINI_BATCH_SIZE", 8),
                patch.object(config, "AV_SYNC_GEMINI_MIN_SCENE_SCORE", 70),
                patch.object(config, "AV_SYNC_GEMINI_MIN_AVERAGE_SCORE", 82),
                patch.object(config, "AV_SYNC_GEMINI_TIMEOUT", 120),
                patch.object(config, "AV_SYNC_GEMINI_MAX_RETRIES", 0),
                patch.object(config, "AV_SYNC_REVIEW_FALLBACK_PROVIDER", ""),
            ):
                report = await multimodal_review.review_video(
                    root / "video.mp4",
                    {"title": "Gold", "scenes": scenes},
                    root,
                )

            self.assertTrue(report["passed"])
            self.assertEqual(report["average_score"], 88)
            self.assertEqual(opencli.await_count, 2)
            calibration = report["calibration"]
            self.assertTrue(calibration["attempted"])
            self.assertEqual(calibration["initial_average_score"], 80)
            self.assertEqual(calibration["match_floor"], 82)
            self.assertEqual(len(calibration["batches"]), 1)
            self.assertTrue(calibration["batches"][0]["contract_valid"])
            initial_command = opencli.await_args_list[0].args[0]
            calibration_command = opencli.await_args_list[1].args[0]
            self.assertEqual(
                initial_command[initial_command.index("--file") + 1],
                calibration_command[calibration_command.index("--file") + 1],
            )
            self.assertIn("match requires score 70-100", initial_command[2])
            self.assertIn("one aggregate-score calibration pass", calibration_command[2])
            self.assertIn("match requires score 82-100", calibration_command[2])

    async def test_low_calibration_match_below_strict_floor_is_invalid_and_fails(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            scenes = [_scene("scene-01", 0, "Gold reaches a record price.")]
            frames = _frames(root, scenes)
            low_match = _matching_payload(scenes, 80)
            opencli = AsyncMock(
                side_effect=[
                    _opencli_result(low_match),
                    _opencli_result(low_match),
                ]
            )
            with (
                patch.object(
                    multimodal_review,
                    "extract_scene_frames",
                    AsyncMock(return_value=frames),
                ),
                patch.object(multimodal_review, "run_opencli", opencli),
                patch.object(config, "AV_SYNC_GEMINI_BATCH_SIZE", 8),
                patch.object(config, "AV_SYNC_GEMINI_MIN_SCENE_SCORE", 70),
                patch.object(config, "AV_SYNC_GEMINI_MIN_AVERAGE_SCORE", 82),
                patch.object(config, "AV_SYNC_GEMINI_TIMEOUT", 120),
                patch.object(config, "AV_SYNC_GEMINI_MAX_RETRIES", 0),
                patch.object(config, "AV_SYNC_REVIEW_FALLBACK_PROVIDER", ""),
            ):
                report = await multimodal_review.review_video(
                    root / "video.mp4",
                    {"title": "Gold", "scenes": scenes},
                    root,
                )

            self.assertFalse(report["passed"])
            self.assertEqual(report["average_score"], 80)
            self.assertEqual(report["failed_scene_ids"], ["scene-01"])
            self.assertEqual(opencli.await_count, 2)
            calibration = report["calibration"]
            self.assertTrue(calibration["attempted"])
            self.assertFalse(calibration["batches"][0]["contract_valid"])
            self.assertEqual(calibration["batches"][0]["match_floor"], 82)
            self.assertIn("violated the requested review rubric", calibration["errors"][0])
            self.assertIn("calibration batch 1", report["errors"][0])

    async def test_unavailable_calibration_retains_complete_initial_match_evidence(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            scenes = [
                _scene("scene-01", 0, "Gold reaches a record price."),
                _scene("scene-02", 8, "Central banks increase reserves."),
            ]
            opencli = AsyncMock(
                side_effect=[
                    _opencli_result(_matching_payload(scenes, 76)),
                    OpenCLIResult(
                        args=(), returncode=0, stdout="not JSON", stderr=""
                    ),
                    OpenCLIResult(
                        args=(), returncode=0, stdout="still not JSON", stderr=""
                    ),
                ]
            )
            with (
                patch.object(
                    multimodal_review,
                    "extract_scene_frames",
                    AsyncMock(return_value=_frames(root, scenes)),
                ),
                patch.object(multimodal_review, "run_opencli", opencli),
                patch.object(config, "AV_SYNC_GEMINI_BATCH_SIZE", 8),
                patch.object(config, "AV_SYNC_GEMINI_MIN_SCENE_SCORE", 70),
                patch.object(config, "AV_SYNC_GEMINI_MIN_AVERAGE_SCORE", 82),
                patch.object(config, "AV_SYNC_GEMINI_TIMEOUT", 120),
                patch.object(config, "AV_SYNC_GEMINI_MAX_RETRIES", 1),
                patch.object(config, "AV_SYNC_REVIEW_FALLBACK_PROVIDER", ""),
            ):
                report = await multimodal_review.review_video(
                    root / "video.mp4",
                    {"title": "Gold", "scenes": scenes},
                    root,
                )

            self.assertTrue(report["passed"])
            self.assertEqual(report["average_score"], 76)
            self.assertEqual(report["failed_scene_ids"], [])
            self.assertEqual(report["errors"], [])
            self.assertEqual(
                report["release_basis"],
                "clean_initial_matches_after_calibration_unavailable",
            )
            calibration = report["calibration"]
            self.assertTrue(calibration["attempted"])
            self.assertTrue(calibration["fallback_to_initial"])
            self.assertIn("retained the initial review", calibration["fallback_reason"])
            self.assertEqual(opencli.await_count, 3)

    async def test_unavailable_batch_never_erases_another_calibration_rejection(self):
        for available_index in (0, 1):
            for verdict, score in (("match", 88), ("partial", 68), ("mismatch", 40)):
                with self.subTest(available_index=available_index, verdict=verdict):
                    with TemporaryDirectory() as temporary:
                        root = Path(temporary)
                        scenes = [
                            _scene("scene-01", 0, "Gold reaches a record price."),
                            _scene("scene-02", 8, "Central banks increase reserves."),
                        ]
                        payload = _matching_payload([scenes[available_index]], score)
                        row = payload["reviews"][0]
                        row["verdict"] = verdict
                        if verdict != "match":
                            row["issues"] = ["The narrated reserve figure is missing."]
                            row["suggested_visual"] = "Show the reserve figure."
                        calibration_responses = [
                            OpenCLIResult(args=(), returncode=0, stdout="unavailable", stderr="")
                            for _ in scenes
                        ]
                        calibration_responses[available_index] = _opencli_result(payload)
                        opencli = AsyncMock(side_effect=[
                            *[_opencli_result(_matching_payload([scene], 76)) for scene in scenes],
                            *calibration_responses,
                        ])
                        with (
                            patch.object(multimodal_review, "extract_scene_frames", AsyncMock(
                                return_value=_frames(root, scenes)
                            )),
                            patch.object(multimodal_review, "run_opencli", opencli),
                            patch.object(config, "AV_SYNC_GEMINI_BATCH_SIZE", 1),
                            patch.object(config, "AV_SYNC_GEMINI_MIN_SCENE_SCORE", 70),
                            patch.object(config, "AV_SYNC_GEMINI_MIN_AVERAGE_SCORE", 82),
                            patch.object(config, "AV_SYNC_GEMINI_MAX_RETRIES", 0),
                            patch.object(config, "AV_SYNC_REVIEW_FALLBACK_PROVIDER", ""),
                        ):
                            report = await multimodal_review.review_video(
                                root / "video.mp4", {"title": "Gold", "scenes": scenes}, root
                            )
                        self.assertEqual(opencli.await_count, 4)
                        calibration = report["calibration"]
                        self.assertEqual(calibration["scenes"][available_index]["score"], score)
                        self.assertEqual(report["passed"], verdict == "match")
                        self.assertEqual(calibration["fallback_to_initial"], verdict == "match")
                        if verdict != "match":
                            self.assertIn(scenes[available_index]["id"], report["failed_scene_ids"])
                            self.assertEqual(report["scenes"][available_index]["issues"], row["issues"])
                            self.assertTrue(report["errors"])

    async def test_chatgpt_fallback_can_pass_after_unusable_gemini_response(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            scenes = [_scene("scene-01", 0, "Gold reaches a record price.")]
            frames = _frames(root, scenes)
            opencli = AsyncMock(
                side_effect=[
                    OpenCLIResult(args=(), returncode=0, stdout="not JSON", stderr=""),
                    _opencli_result(_matching_payload(scenes, 90)),
                ]
            )
            with (
                patch.object(
                    multimodal_review,
                    "extract_scene_frames",
                    AsyncMock(return_value=frames),
                ),
                patch.object(multimodal_review, "run_opencli", opencli),
                patch.object(config, "AV_SYNC_GEMINI_BATCH_SIZE", 8),
                patch.object(config, "AV_SYNC_GEMINI_MIN_SCENE_SCORE", 70),
                patch.object(config, "AV_SYNC_GEMINI_MIN_AVERAGE_SCORE", 82),
                patch.object(config, "AV_SYNC_GEMINI_TIMEOUT", 120),
                patch.object(config, "AV_SYNC_GEMINI_MAX_RETRIES", 0),
                patch.object(config, "AV_SYNC_REVIEW_FALLBACK_PROVIDER", "chatgpt"),
            ):
                report = await multimodal_review.review_video(
                    root / "video.mp4",
                    {"title": "Gold", "scenes": scenes},
                    root,
                )

            self.assertTrue(report["passed"])
            self.assertEqual(opencli.await_count, 2)
            self.assertEqual(opencli.await_args_list[1].args[0][0], "chatgpt")
            self.assertEqual(report["batches"][0]["review_provider"], "chatgpt")

    async def test_resumed_unavailable_review_tries_fallback_first_without_overriding_rejection(self):
        for outcome in ("match", "partial", "unavailable"):
            with self.subTest(outcome=outcome), TemporaryDirectory() as temporary:
                root = Path(temporary)
                scenes = [_scene("scene-01", 0, "Gold reaches a record price.")]
                payload = _matching_payload(scenes, 90 if outcome == "match" else 65)
                if outcome == "partial":
                    payload["reviews"][0].update(verdict="partial", issues=["Missing price."], suggested_visual="Show the price.")
                first = OpenCLIError("ChatGPT unavailable") if outcome == "unavailable" else _opencli_result(payload)
                call = AsyncMock(side_effect=[first, _opencli_result(_matching_payload(scenes, 90))])
                with patch.object(multimodal_review, "run_opencli", call), patch.object(config, "AV_SYNC_REVIEW_FALLBACK_PROVIDER", "chatgpt"):
                    normalized, attempts, error = await multimodal_review._review_batch(
                        title="Gold", batch_frames=_frames(root, scenes), sheet=root / "sheet.jpg",
                        match_floor=70, minimum_average_score=82, timeout=120,
                        maximum_retries=2, batch_index=1, phase="initial", log=None,
                        prefer_fallback=True,
                    )
                self.assertEqual(call.await_args_list[0].args[0][0], "chatgpt")
                self.assertEqual(attempts, 2 if outcome == "unavailable" else 1)
                self.assertEqual(error, "")
                self.assertEqual(normalized["reviews"][0]["passed"], outcome != "partial")
                self.assertEqual(normalized["review_provider"], "gemini" if outcome == "unavailable" else "chatgpt")

    async def test_unavailable_review_overwrites_stale_response_and_keeps_last_error(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            scenes = [_scene("scene-01", 0, "Gold reaches a record price.")]
            frames = _frames(root, scenes)
            stale = root / "response-initial-01-attempt-01-gemini.json"
            stale.write_text('{"stdout":"previous candidate review"}')
            messages = []
            with (
                patch.object(multimodal_review, "run_opencli", AsyncMock(side_effect=[
                    OpenCLIError("Gemini send rejected: disabled button"),
                    OpenCLIError("ChatGPT upload unavailable"),
                ])),
                patch.object(config, "AV_SYNC_REVIEW_FALLBACK_PROVIDER", "chatgpt"),
            ):
                normalized, attempts, error = await multimodal_review._review_batch(
                    title="Gold", batch_frames=frames, sheet=root / "sheet.jpg",
                    match_floor=70, minimum_average_score=82, timeout=120,
                    maximum_retries=0, batch_index=1, phase="initial", log=messages.append,
                )
            self.assertEqual(attempts, 2)
            self.assertFalse(normalized["image_received"])
            self.assertIn("ChatGPT upload unavailable", error)
            self.assertEqual(json.loads(stale.read_text())["stdout"], "")
            self.assertIn("disabled button", json.loads(stale.read_text())["stderr"])
            fallback = root / "response-initial-01-attempt-02-chatgpt.json"
            self.assertIn("ChatGPT upload unavailable", json.loads(fallback.read_text())["stderr"])
            self.assertTrue(any("exhausted" in message and "disabled button" in message for message in messages))

    async def test_completed_review_is_reused_only_for_identical_pixels_narration_and_direction(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            scenes = [_scene("scene-01", 0, "Gold reaches a record price.")]
            frames = _frames(root, scenes)
            sheet = multimodal_review.create_contact_sheet(frames, root / "sheet.jpg")
            call = AsyncMock(return_value=_opencli_result(_matching_payload(scenes, 90)))
            args = dict(title="Gold", batch_frames=frames, sheet=sheet, match_floor=70,
                        minimum_average_score=82, timeout=120, maximum_retries=0,
                        batch_index=1, phase="initial", log=None)
            with patch.object(multimodal_review, "run_opencli", call):
                first = await multimodal_review._review_batch(**args)
                cached = await multimodal_review._review_batch(**args)
                self.assertEqual(first[0], cached[0])
                self.assertEqual(call.await_count, 1)
                self.assertEqual(cached[1], 0)
                sheet.write_bytes(sheet.read_bytes() + b"changed pixels")
                await multimodal_review._review_batch(**args)
                self.assertEqual(call.await_count, 2)
                scenes[0]["text"] = "A different narrated claim"
                await multimodal_review._review_batch(**args)
                self.assertEqual(call.await_count, 3)
                # Even identical pixels must be reviewed again when their
                # interpretation changes from a midpoint to a motion phase.
                frames[0]["samples"] = [{
                    "id": "scene-01-shot-01", "timestamp": frames[0]["timestamp"],
                    "phase": "action", "media": {"kind": "editorial"},
                }]
                await multimodal_review._review_batch(**args)
                self.assertEqual(call.await_count, 4)
                frames[0]["samples"][0]["phase"] = "settled"
                await multimodal_review._review_batch(**args)
                self.assertEqual(call.await_count, 5)
                planned = {
                    "id": "scene-01-shot-01", "start_seconds": 0, "end_seconds": 8,
                    "direction": {"action": "Reveal the price claim and hold for reading"},
                    "layout": "focus", "selected_copy": [scenes[0]["text"]],
                    "relationship_excerpt": "",
                }
                frames[0]["planned_shots"] = [planned]
                await multimodal_review._review_batch(**args)
                cached = await multimodal_review._review_batch(**args)
                self.assertEqual(call.await_count, 6)
                self.assertEqual(cached[1], 0)
                # Identical pixels and narration are not evidence that a newly
                # promised graphic, emphasis or continuity was implemented.
                planned["direction"]["action"] = "Emphasize the price with a large numeral"
                await multimodal_review._review_batch(**args)
                self.assertEqual(call.await_count, 7)
                for count, (key, value) in enumerate([
                    ("layout", "comparison"), ("selected_copy", ["Changed selected copy"]),
                    ("relationship_excerpt", "Compared with the previous price"),
                    ("end_seconds", 9),
                ], start=8):
                    planned[key] = value
                    await multimodal_review._review_batch(**args)
                    self.assertEqual(call.await_count, count)

    async def test_low_scene_score_rejects_video_even_when_average_is_high(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            scenes = [
                _scene("scene-01", 0, "Gold reaches a record price."),
                _scene("scene-02", 8, "Central banks increase reserves."),
            ]
            frames = _frames(root, scenes)
            payload = {
                "image_received": True,
                "reviews": [
                    {
                        "id": "scene-01",
                        "score": 100,
                        "verdict": "match",
                        "visual_summary": "Gold bars and a price chart.",
                        "alignment_reason": "Direct match.",
                        "issues": [],
                    },
                    {
                        "id": "scene-02",
                        "score": 65,
                        "verdict": "partial",
                        "visual_summary": "A generic beach.",
                        "alignment_reason": "No central-bank or reserve imagery is visible.",
                        "issues": ["No central-bank or reserve imagery is visible."],
                        "suggested_visual": "Show a central bank vault and reserve ledger.",
                    },
                ],
            }
            opencli = AsyncMock(
                return_value=OpenCLIResult(
                    args=(),
                    returncode=0,
                    stdout=json.dumps([{"response": json.dumps(payload)}]),
                    stderr="",
                )
            )
            with (
                patch.object(
                    multimodal_review,
                    "extract_scene_frames",
                    AsyncMock(return_value=frames),
                ),
                patch.object(multimodal_review, "run_opencli", opencli),
                patch.object(config, "AV_SYNC_GEMINI_BATCH_SIZE", 8),
                patch.object(config, "AV_SYNC_GEMINI_MIN_SCENE_SCORE", 70),
                patch.object(config, "AV_SYNC_GEMINI_MIN_AVERAGE_SCORE", 80),
                patch.object(config, "AV_SYNC_GEMINI_TIMEOUT", 120),
                patch.object(config, "AV_SYNC_GEMINI_MAX_RETRIES", 1),
            ):
                report = await multimodal_review.review_video(
                    root / "video.mp4",
                    {"title": "Gold", "scenes": scenes},
                    root,
                )

            self.assertFalse(report["passed"])
            self.assertEqual(report["failed_scene_ids"], ["scene-02"])
            self.assertEqual(report["average_score"], 82.5)

    async def test_inconsistent_score_and_verdict_are_retried(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            scenes = [_scene("scene-01", 0, "Gold reaches a record price.")]
            frames = _frames(root, scenes)
            contradictory = {
                "image_received": True,
                "reviews": [
                    {
                        "id": "scene-01",
                        "score": 68,
                        "verdict": "match",
                        "visual_summary": "Gold bars and a price chart.",
                        "alignment_reason": "Direct match.",
                        "issues": [],
                        "suggested_visual": "",
                    }
                ],
            }
            consistent = {
                "image_received": True,
                "reviews": [
                    {
                        "id": "scene-01",
                        "score": 88,
                        "verdict": "match",
                        "visual_summary": "Gold bars and a price chart.",
                        "alignment_reason": "Direct match.",
                        "issues": [],
                        "suggested_visual": "",
                    }
                ],
            }
            opencli = AsyncMock(
                side_effect=[
                    OpenCLIResult(
                        args=(),
                        returncode=0,
                        stdout=json.dumps([{"response": json.dumps(contradictory)}]),
                        stderr="",
                    ),
                    OpenCLIResult(
                        args=(),
                        returncode=0,
                        stdout=json.dumps([{"response": json.dumps(consistent)}]),
                        stderr="",
                    ),
                ]
            )
            with (
                patch.object(
                    multimodal_review,
                    "extract_scene_frames",
                    AsyncMock(return_value=frames),
                ),
                patch.object(multimodal_review, "run_opencli", opencli),
                patch.object(config, "AV_SYNC_GEMINI_BATCH_SIZE", 8),
                patch.object(config, "AV_SYNC_GEMINI_MIN_SCENE_SCORE", 70),
                patch.object(config, "AV_SYNC_GEMINI_MIN_AVERAGE_SCORE", 82),
                patch.object(config, "AV_SYNC_GEMINI_TIMEOUT", 120),
                patch.object(config, "AV_SYNC_GEMINI_MAX_RETRIES", 1),
            ):
                report = await multimodal_review.review_video(
                    root / "video.mp4",
                    {"title": "Gold", "scenes": scenes},
                    root,
                )

            self.assertTrue(report["passed"])
            self.assertEqual(opencli.await_count, 2)
            self.assertEqual(report["batches"][0]["attempts"], 2)
            prompt = opencli.await_args_list[0].args[0][2]
            self.assertIn("match requires score 70-100 and issues=[]", prompt)
            self.assertIn("release average is 82/100", prompt)

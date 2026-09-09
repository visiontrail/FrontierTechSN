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

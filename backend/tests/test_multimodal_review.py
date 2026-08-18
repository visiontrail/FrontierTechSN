import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

from PIL import Image

from backend import config
from backend.pipeline import multimodal_review
from backend.pipeline.opencli import OpenCLIResult


def _scene(scene_id: str, start: float, text: str) -> dict:
    return {
        "id": scene_id,
        "start": start,
        "duration": 8.0,
        "text": text,
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
                    "visual_summary": "A gold bar and a record-price chart.",
                    "alignment_reason": "The visible chart directly supports the claim.",
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
            ):
                report = await multimodal_review.review_video(
                    root / "video.mp4",
                    {"title": "Gold", "scenes": scenes},
                    root,
                )

            self.assertTrue(report["passed"])
            self.assertEqual(report["average_score"], 90)
            args = opencli.await_args.args[0]
            self.assertIn("--file", args)
            self.assertIn("foreground", args)
            self.assertTrue((root / "multimodal_review" / "contact-sheet-01.jpg").is_file())

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
                        "visual_summary": "A gold bar and a rising price chart.",
                        "alignment_reason": "The visible subject matches the narration.",
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
                        "visual_summary": "Gold bars and a price chart.",
                        "alignment_reason": "Direct match.",
                    },
                    {
                        "id": "scene-02",
                        "score": 65,
                        "visual_summary": "A generic beach.",
                        "alignment_reason": "No central-bank or reserve imagery is visible.",
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

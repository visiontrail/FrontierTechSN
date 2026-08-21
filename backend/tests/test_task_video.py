import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

from backend.models import SourceType, TaskConfig, TaskResponse, TaskStatus
from backend.routers import tasks as tasks_router


def make_video_task(
    task_id: str,
    *,
    status: TaskStatus,
    video_path: str,
) -> TaskResponse:
    return TaskResponse(
        id=task_id,
        created_at="2026-08-21T00:00:00+00:00",
        updated_at="2026-08-21T00:00:00+00:00",
        source_type=SourceType.TOPIC,
        status=status,
        config=TaskConfig(),
        video_path=video_path,
    )


class VideoDownloadRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_complete_task_serves_final_artifact_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "validated.mp4"
            video.write_bytes(b"validated-video")
            task = make_video_task(
                "task-final",
                status=TaskStatus.COMPLETE,
                video_path=str(video),
            )

            with patch.object(
                tasks_router.db,
                "get_task",
                AsyncMock(return_value=task),
            ):
                response = await tasks_router.download_video(task.id)

            self.assertEqual(response.path, video)
            self.assertEqual(response.headers["x-video-artifact-state"], "final")
            self.assertEqual(response.headers["x-task-status"], "complete")
            self.assertIn('filename="task-final.mp4"', response.headers["content-disposition"])

    async def test_failed_task_rejects_default_final_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "previous.mp4"
            video.write_bytes(b"previous-video")
            task = make_video_task(
                "task-failed",
                status=TaskStatus.FAILED,
                video_path=str(video),
            )

            with patch.object(
                tasks_router.db,
                "get_task",
                AsyncMock(return_value=task),
            ):
                with self.assertRaises(HTTPException) as raised:
                    await tasks_router.download_video(task.id)

            self.assertEqual(raised.exception.status_code, 409)
            self.assertIn("artifact=retained", raised.exception.detail)
            self.assertEqual(
                raised.exception.headers,
                {
                    "X-Video-Artifact-State": "retained",
                    "X-Task-Status": "failed",
                },
            )

    async def test_failed_task_serves_explicit_retained_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "previous.mp4"
            video.write_bytes(b"previous-video")
            task = make_video_task(
                "task-retained",
                status=TaskStatus.FAILED,
                video_path=str(video),
            )

            with patch.object(
                tasks_router.db,
                "get_task",
                AsyncMock(return_value=task),
            ):
                response = await tasks_router.download_video(task.id, artifact="retained")

            self.assertEqual(response.path, video)
            self.assertEqual(response.headers["x-video-artifact-state"], "retained")
            self.assertEqual(response.headers["x-task-status"], "failed")
            self.assertIn(
                'filename="task-retained-retained.mp4"',
                response.headers["content-disposition"],
            )

    async def test_complete_task_rejects_retained_artifact_selector(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "validated.mp4"
            video.write_bytes(b"validated-video")
            task = make_video_task(
                "task-current",
                status=TaskStatus.COMPLETE,
                video_path=str(video),
            )

            with patch.object(
                tasks_router.db,
                "get_task",
                AsyncMock(return_value=task),
            ):
                with self.assertRaises(HTTPException) as raised:
                    await tasks_router.download_video(task.id, artifact="retained")

            self.assertEqual(raised.exception.status_code, 409)
            self.assertIn("artifact=final", raised.exception.detail)
            self.assertEqual(
                raised.exception.headers,
                {
                    "X-Video-Artifact-State": "final",
                    "X-Task-Status": "complete",
                },
            )

    async def test_missing_video_path_still_returns_not_found(self):
        task = make_video_task(
            "task-no-video",
            status=TaskStatus.FAILED,
            video_path="",
        )

        with patch.object(
            tasks_router.db,
            "get_task",
            AsyncMock(return_value=task),
        ):
            with self.assertRaises(HTTPException) as raised:
                await tasks_router.download_video(task.id, artifact="retained")

        self.assertEqual(raised.exception.status_code, 404)

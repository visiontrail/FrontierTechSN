import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

from backend import config
from backend.models import TaskConfig, TaskResponse, TaskStatus
from backend.pipeline import orchestrator
from backend.routers import tasks as tasks_router
from backend import worker


def make_task(
    task_id: str,
    *,
    status: TaskStatus,
    script_path: str | None,
    output_dir: str,
) -> TaskResponse:
    return TaskResponse(
        id=task_id,
        created_at="2026-08-20T00:00:00+00:00",
        updated_at="2026-08-20T00:00:00+00:00",
        source_type="news_daily",
        source_title="Saved source title",
        generated_title="Saved publication title",
        status=status,
        output_dir=output_dir,
        script_path=script_path,
        thumbnail_path=str(Path(output_dir) / "thumbnail" / "saved.png"),
        config=TaskConfig(
            auto_render=True,
            auto_publish=True,
            thumbnail_enabled=True,
            footage_enabled=False,
        ),
    )


class ResumeTtsRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_task_with_saved_script_queues_tts_resume_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            task_dir = Path(directory) / "task-route"
            task_dir.mkdir()
            script_path = task_dir / "script.txt"
            script_path.write_text("Persisted approved narration", encoding="utf-8")
            failed = make_task(
                "task-route",
                status=TaskStatus.FAILED,
                script_path=str(script_path),
                output_dir=str(task_dir),
            )
            queued = failed.model_copy(update={"status": TaskStatus.QUEUED, "error_message": None})

            with (
                patch.object(
                    tasks_router.db,
                    "get_task",
                    AsyncMock(side_effect=[failed, queued]),
                ),
                patch.object(tasks_router.db, "update_task", AsyncMock()) as update_task,
            ):
                result = await tasks_router.resume_task_tts(failed.id)

            self.assertEqual(result.status, TaskStatus.QUEUED)
            self.assertTrue((task_dir / ".resume_tts").is_file())
            update_task.assert_awaited_once_with(
                failed.id,
                status=TaskStatus.QUEUED.value,
                error_message=None,
            )

    async def test_resume_tts_rejects_any_non_failed_task(self):
        with tempfile.TemporaryDirectory() as directory:
            task_dir = Path(directory) / "task-active"
            task_dir.mkdir()
            script_path = task_dir / "script.txt"
            script_path.write_text("Narration", encoding="utf-8")
            active = make_task(
                "task-active",
                status=TaskStatus.TTS,
                script_path=str(script_path),
                output_dir=str(task_dir),
            )

            with (
                patch.object(tasks_router.db, "get_task", AsyncMock(return_value=active)),
                patch.object(tasks_router.db, "update_task", AsyncMock()) as update_task,
            ):
                with self.assertRaises(HTTPException) as raised:
                    await tasks_router.resume_task_tts(active.id)

            self.assertEqual(raised.exception.status_code, 409)
            self.assertFalse((task_dir / ".resume_tts").exists())
            update_task.assert_not_awaited()

    async def test_resume_tts_rejects_failed_task_without_script(self):
        with tempfile.TemporaryDirectory() as directory:
            task_dir = Path(directory) / "task-no-script"
            task_dir.mkdir()
            failed = make_task(
                "task-no-script",
                status=TaskStatus.FAILED,
                script_path=None,
                output_dir=str(task_dir),
            )

            with (
                patch.object(tasks_router.db, "get_task", AsyncMock(return_value=failed)),
                patch.object(tasks_router.db, "update_task", AsyncMock()) as update_task,
            ):
                with self.assertRaises(HTTPException) as raised:
                    await tasks_router.resume_task_tts(failed.id)

            self.assertEqual(raised.exception.status_code, 400)
            self.assertFalse((task_dir / ".resume_tts").exists())
            update_task.assert_not_awaited()


class ResumeTtsOrchestratorTests(unittest.IsolatedAsyncioTestCase):
    async def test_resume_only_generates_tts_updates_audio_and_continues_after_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            task_dir = Path(directory) / "task-orchestrator"
            task_dir.mkdir()
            script_path = task_dir / "script.txt"
            script_path.write_text("Persisted approved narration", encoding="utf-8")
            audio_path = str(task_dir / "audio" / "combined.wav")
            task = make_task(
                "task-orchestrator",
                status=TaskStatus.FAILED,
                script_path=str(script_path),
                output_dir=str(task_dir),
            )
            saved_title = task.generated_title
            saved_thumbnail = task.thumbnail_path

            with (
                patch.object(config, "OUTPUTS_DIR", Path(directory)),
                patch.object(orchestrator, "update_task", AsyncMock()) as update_task,
                patch.object(
                    orchestrator,
                    "generate_tts",
                    AsyncMock(return_value=audio_path),
                ) as generate_tts,
                patch.object(orchestrator, "_after_audio", AsyncMock()) as after_audio,
                patch.object(orchestrator, "_generate_task_title", AsyncMock()) as generate_title,
                patch.object(
                    orchestrator,
                    "_generate_task_thumbnail",
                    AsyncMock(),
                ) as generate_thumbnail,
            ):
                await orchestrator.run_tts_resume(task)

            self.assertEqual(task.generated_title, saved_title)
            self.assertEqual(task.thumbnail_path, saved_thumbnail)
            generate_title.assert_not_awaited()
            generate_thumbnail.assert_not_awaited()
            generate_tts.assert_awaited_once()
            self.assertEqual(generate_tts.await_args.args[:4], (
                str(script_path),
                str(task_dir / "audio"),
                [task.config.voice_1],
                task.config.tts_model,
            ))
            self.assertEqual(
                update_task.await_args_list,
                [
                    unittest.mock.call(
                        task.id,
                        status=TaskStatus.TTS.value,
                        error_message=None,
                    ),
                    unittest.mock.call(task.id, audio_path=audio_path),
                ],
            )
            after_audio.assert_awaited_once()
            self.assertEqual(after_audio.await_args.kwargs["script_path"], str(script_path))
            self.assertEqual(after_audio.await_args.kwargs["audio_path"], audio_path)
            self.assertEqual(after_audio.await_args.kwargs["prefix"], "TTS resume: ")


class ResumeTtsWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_worker_claims_resume_tts_marker_without_running_other_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            task_dir = Path(directory) / "task-worker"
            task_dir.mkdir()
            script_path = task_dir / "script.txt"
            script_path.write_text("Persisted approved narration", encoding="utf-8")
            marker = task_dir / worker.TTS_RESUME_MARKER
            marker.write_text("", encoding="utf-8")
            queued = make_task(
                "task-worker",
                status=TaskStatus.QUEUED,
                script_path=str(script_path),
                output_dir=str(task_dir),
            )
            complete = queued.model_copy(update={"status": TaskStatus.COMPLETE})

            with (
                patch.object(config, "OUTPUTS_DIR", Path(directory)),
                patch.object(
                    worker,
                    "get_next_queued_task",
                    AsyncMock(side_effect=[queued, asyncio.CancelledError()]),
                ),
                patch.object(worker, "get_task", AsyncMock(return_value=complete)),
                patch.object(worker, "update_task", AsyncMock()) as update_task,
                patch.object(worker, "run_tts_resume", AsyncMock()) as run_tts_resume,
                patch.object(worker, "run_regenerate", AsyncMock()) as run_regenerate,
                patch.object(worker, "run_daily_review_resume", AsyncMock()) as run_review,
                patch.object(worker, "run_pipeline", AsyncMock()) as run_pipeline,
                patch.object(worker, "run_compose", AsyncMock()) as run_compose,
                patch.object(worker, "run_footage_acquisition", AsyncMock()) as run_footage,
                patch.object(
                    worker,
                    "run_auto_publish_pipeline",
                    AsyncMock(),
                ) as auto_publish,
            ):
                with self.assertRaises(asyncio.CancelledError):
                    await worker._worker_loop()

            self.assertFalse(marker.exists())
            run_tts_resume.assert_awaited_once()
            run_regenerate.assert_not_awaited()
            run_review.assert_not_awaited()
            run_pipeline.assert_not_awaited()
            run_compose.assert_not_awaited()
            run_footage.assert_not_awaited()
            update_task.assert_awaited_once_with(
                queued.id,
                status=TaskStatus.TTS.value,
                error_message=None,
            )
            auto_publish.assert_awaited_once_with(complete)

    async def test_worker_keeps_resume_marker_when_claim_status_cannot_persist(self):
        with tempfile.TemporaryDirectory() as directory:
            task_dir = Path(directory) / "task-claim-failure"
            task_dir.mkdir()
            script_path = task_dir / "script.txt"
            script_path.write_text("Persisted approved narration", encoding="utf-8")
            marker = task_dir / worker.TTS_RESUME_MARKER
            marker.write_text("", encoding="utf-8")
            queued = make_task(
                "task-claim-failure",
                status=TaskStatus.QUEUED,
                script_path=str(script_path),
                output_dir=str(task_dir),
            )

            claim_error = RuntimeError("database unavailable during mode claim")
            with (
                patch.object(config, "OUTPUTS_DIR", Path(directory)),
                patch.object(
                    worker,
                    "get_next_queued_task",
                    AsyncMock(side_effect=[queued, asyncio.CancelledError()]),
                ),
                patch.object(
                    worker,
                    "update_task",
                    AsyncMock(side_effect=[claim_error, None]),
                ) as update_task,
                patch.object(worker, "run_tts_resume", AsyncMock()) as run_tts_resume,
            ):
                with self.assertRaises(asyncio.CancelledError):
                    await worker._worker_loop()

            self.assertTrue(marker.exists())
            run_tts_resume.assert_not_awaited()
            self.assertEqual(update_task.await_count, 2)
            self.assertEqual(
                update_task.await_args_list[0],
                unittest.mock.call(
                    queued.id,
                    status=TaskStatus.TTS.value,
                    error_message=None,
                ),
            )
            self.assertEqual(
                update_task.await_args_list[1],
                unittest.mock.call(
                    queued.id,
                    status=TaskStatus.FAILED.value,
                    error_message=str(claim_error),
                ),
            )


if __name__ == "__main__":
    unittest.main()

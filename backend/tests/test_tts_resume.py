import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

from backend import config
from backend.models import (
    FootageAcquireRequest,
    ScriptUpdate,
    SourceType,
    TaskConfig,
    TaskPublicationRequest,
    TaskPublicationReconcileRequest,
    TaskResponse,
    TaskStatus,
)
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
        source_type=SourceType.NEWS_DAILY,
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
                    AsyncMock(side_effect=[failed, failed, queued]),
                ),
                patch.object(
                    tasks_router.db,
                    "compare_and_set_task_status",
                    AsyncMock(side_effect=[True, True]),
                ) as transition,
            ):
                result = await tasks_router.resume_task_tts(failed.id)

            self.assertEqual(result.status, TaskStatus.QUEUED)
            self.assertTrue((task_dir / ".resume_tts").is_file())
            self.assertEqual(
                transition.await_args_list,
                [
                    unittest.mock.call(
                        failed.id,
                        TaskStatus.FAILED,
                        TaskStatus.TTS,
                        error_message=None,
                        expected_updated_at=failed.updated_at,
                        suppress_next_auto_publish=None,
                    ),
                    unittest.mock.call(
                        failed.id,
                        TaskStatus.TTS,
                        TaskStatus.QUEUED,
                        error_message=None,
                    ),
                ],
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
                patch.object(
                    tasks_router.db,
                    "compare_and_set_task_status",
                    AsyncMock(),
                ) as transition,
            ):
                with self.assertRaises(HTTPException) as raised:
                    await tasks_router.resume_task_tts(active.id)

            self.assertEqual(raised.exception.status_code, 409)
            self.assertFalse((task_dir / ".resume_tts").exists())
            transition.assert_not_awaited()


class ScriptUpdateRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_publication_safety_hold_rejects_script_edits(self):
        with tempfile.TemporaryDirectory() as directory:
            task_dir = Path(directory) / "task-edit-publication-hold"
            task_dir.mkdir()
            script_path = task_dir / "script.txt"
            script_path.write_text("Original", encoding="utf-8")
            failed = make_task(
                "task-edit-publication-hold",
                status=TaskStatus.FAILED,
                script_path=str(script_path),
                output_dir=str(task_dir),
            ).model_copy(update={"publication_safety_hold": True})
            with (
                patch.object(tasks_router.db, "get_task", AsyncMock(return_value=failed)),
                patch.object(tasks_router.db, "update_task", AsyncMock()) as update,
            ):
                with self.assertRaises(HTTPException) as raised:
                    await tasks_router.update_script(
                        failed.id,
                        ScriptUpdate(content="Untraceable replacement"),
                    )

            self.assertEqual(raised.exception.status_code, 409)
            self.assertIn("publication safety hold", raised.exception.detail)
            self.assertEqual(script_path.read_text(encoding="utf-8"), "Original")
            update.assert_not_awaited()

    async def test_processing_task_rejects_script_edits(self):
        with tempfile.TemporaryDirectory() as directory:
            task_dir = Path(directory) / "task-edit-active"
            task_dir.mkdir()
            script_path = task_dir / "script.txt"
            script_path.write_text("Original", encoding="utf-8")
            active = make_task(
                "task-edit-active",
                status=TaskStatus.COMPOSING,
                script_path=str(script_path),
                output_dir=str(task_dir),
            )
            with (
                patch.object(tasks_router.db, "get_task", AsyncMock(return_value=active)),
                patch.object(tasks_router.db, "update_task", AsyncMock()) as update,
            ):
                with self.assertRaises(HTTPException) as raised:
                    await tasks_router.update_script(
                        active.id,
                        ScriptUpdate(content="Racing edit"),
                    )

            self.assertEqual(raised.exception.status_code, 409)
            self.assertEqual(script_path.read_text(encoding="utf-8"), "Original")
            update.assert_not_awaited()

    async def test_stable_task_atomically_replaces_script_and_bumps_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            task_dir = Path(directory) / "task-edit-stable"
            task_dir.mkdir()
            script_path = task_dir / "script.txt"
            script_path.write_text("Original", encoding="utf-8")
            complete = make_task(
                "task-edit-stable",
                status=TaskStatus.COMPLETE,
                script_path=str(script_path),
                output_dir=str(task_dir),
            )
            with (
                patch.object(tasks_router.db, "get_task", AsyncMock(return_value=complete)),
                patch.object(tasks_router.db, "update_task", AsyncMock()) as update,
            ):
                result = await tasks_router.update_script(
                    complete.id,
                    ScriptUpdate(content="Edited narration"),
                )

            self.assertEqual(result, {"ok": True})
            self.assertEqual(
                script_path.read_text(encoding="utf-8"),
                "Edited narration",
            )
            self.assertFalse((task_dir / ".script.txt.edit.tmp").exists())
            update.assert_awaited_once_with(
                complete.id,
                script_path=str(script_path),
                status=TaskStatus.AWAITING_REVIEW.value,
                suppress_next_auto_publish=True,
            )

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
                patch.object(
                    tasks_router.db,
                    "compare_and_set_task_status",
                    AsyncMock(),
                ) as transition,
            ):
                with self.assertRaises(HTTPException) as raised:
                    await tasks_router.resume_task_tts(failed.id)

            self.assertEqual(raised.exception.status_code, 400)
            self.assertFalse((task_dir / ".resume_tts").exists())
            transition.assert_not_awaited()


class RegenerateRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_publication_safety_hold_rejects_worker_mode_queueing(self):
        with tempfile.TemporaryDirectory() as directory:
            task_dir = Path(directory) / "task-regenerate-publication-hold"
            task_dir.mkdir()
            script_path = task_dir / "script.txt"
            script_path.write_text("Persisted narration", encoding="utf-8")
            failed = make_task(
                "task-regenerate-publication-hold",
                status=TaskStatus.FAILED,
                script_path=str(script_path),
                output_dir=str(task_dir),
            ).model_copy(update={"publication_safety_hold": True})
            transition = AsyncMock()

            with (
                patch.object(
                    tasks_router.db,
                    "get_task",
                    AsyncMock(side_effect=[failed, failed]),
                ),
                patch.object(
                    tasks_router.db,
                    "compare_and_set_task_status",
                    transition,
                ),
            ):
                with self.assertRaises(HTTPException) as raised:
                    await tasks_router.regenerate_task(failed.id)

            self.assertEqual(raised.exception.status_code, 409)
            self.assertIn("publication safety hold", raised.exception.detail)
            self.assertFalse((task_dir / worker.REGEN_MARKER).exists())
            transition.assert_not_awaited()

    async def test_complete_task_atomically_queues_regeneration(self):
        with tempfile.TemporaryDirectory() as directory:
            task_dir = Path(directory) / "task-regenerate"
            task_dir.mkdir()
            script_path = task_dir / "script.txt"
            script_path.write_text("Persisted narration", encoding="utf-8")
            complete = make_task(
                "task-regenerate",
                status=TaskStatus.COMPLETE,
                script_path=str(script_path),
                output_dir=str(task_dir),
            )
            queued = complete.model_copy(
                update={
                    "status": TaskStatus.QUEUED,
                    "suppress_next_auto_publish": True,
                }
            )

            with (
                patch.object(
                    tasks_router.db,
                    "get_task",
                    AsyncMock(side_effect=[complete, complete, queued]),
                ),
                patch.object(
                    tasks_router.db,
                    "compare_and_set_task_status",
                    AsyncMock(side_effect=[True, True]),
                ) as transition,
            ):
                result = await tasks_router.regenerate_task(complete.id)

            self.assertEqual(result.status, TaskStatus.QUEUED)
            self.assertTrue((task_dir / ".regenerate").is_file())
            self.assertEqual(
                transition.await_args_list,
                [
                    unittest.mock.call(
                        complete.id,
                        TaskStatus.COMPLETE,
                        TaskStatus.TITLING,
                        error_message=None,
                        expected_updated_at=complete.updated_at,
                        suppress_next_auto_publish=True,
                    ),
                    unittest.mock.call(
                        complete.id,
                        TaskStatus.TITLING,
                        TaskStatus.QUEUED,
                        error_message=None,
                    ),
                ],
            )

    async def test_stale_complete_snapshot_cannot_race_a_render_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            task_dir = Path(directory) / "task-regenerate-race"
            task_dir.mkdir()
            script_path = task_dir / "script.txt"
            script_path.write_text("Persisted narration", encoding="utf-8")
            complete = make_task(
                "task-regenerate-race",
                status=TaskStatus.COMPLETE,
                script_path=str(script_path),
                output_dir=str(task_dir),
            )

            with (
                patch.object(tasks_router.db, "get_task", AsyncMock(return_value=complete)),
                patch.object(
                    tasks_router.db,
                    "compare_and_set_task_status",
                    AsyncMock(return_value=False),
                ) as transition,
            ):
                with self.assertRaises(HTTPException) as raised:
                    await tasks_router.regenerate_task(complete.id)

            self.assertEqual(raised.exception.status_code, 409)
            self.assertFalse((task_dir / ".regenerate").exists())
            transition.assert_awaited_once_with(
                complete.id,
                TaskStatus.COMPLETE,
                TaskStatus.TITLING,
                error_message=None,
                expected_updated_at=complete.updated_at,
                suppress_next_auto_publish=True,
            )


class ResumeReviewRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_daily_review_is_atomically_queued(self):
        with tempfile.TemporaryDirectory() as directory:
            task_dir = Path(directory) / "task-review-resume"
            research_dir = task_dir / "research"
            research_dir.mkdir(parents=True)
            (research_dir / "dossier.json").write_text("{}", encoding="utf-8")
            (task_dir / "script.draft.txt").write_text("Draft", encoding="utf-8")
            failed = make_task(
                "task-review-resume",
                status=TaskStatus.FAILED,
                script_path=str(task_dir / "script.txt"),
                output_dir=str(task_dir),
            )
            queued = failed.model_copy(update={"status": TaskStatus.QUEUED})
            self.assertIs(failed.source_type, SourceType.NEWS_DAILY)

            with (
                patch.object(
                    tasks_router.db,
                    "get_task",
                    AsyncMock(side_effect=[failed, failed, queued]),
                ),
                patch.object(
                    tasks_router.db,
                    "compare_and_set_task_status",
                    AsyncMock(side_effect=[True, True]),
                ) as transition,
            ):
                result = await tasks_router.resume_daily_review(failed.id)

            self.assertEqual(result.status, TaskStatus.QUEUED)
            self.assertTrue((task_dir / ".resume_review").is_file())
            self.assertEqual(
                transition.await_args_list,
                [
                    unittest.mock.call(
                        failed.id,
                        TaskStatus.FAILED,
                        TaskStatus.REVIEWING,
                        error_message=None,
                        expected_updated_at=failed.updated_at,
                        suppress_next_auto_publish=None,
                    ),
                    unittest.mock.call(
                        failed.id,
                        TaskStatus.REVIEWING,
                        TaskStatus.QUEUED,
                        error_message=None,
                    ),
                ],
            )


class FootageRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_complete_task_atomically_queues_footage_without_republishing(self):
        with tempfile.TemporaryDirectory() as directory:
            task_dir = Path(directory) / "task-footage"
            task_dir.mkdir()
            script_path = task_dir / "script.txt"
            script_path.write_text("Persisted narration", encoding="utf-8")
            complete = make_task(
                "task-footage",
                status=TaskStatus.COMPLETE,
                script_path=str(script_path),
                output_dir=str(task_dir),
            )
            queued = complete.model_copy(
                update={
                    "status": TaskStatus.QUEUED,
                    "suppress_next_auto_publish": True,
                }
            )

            with (
                patch.object(
                    tasks_router.db,
                    "get_task",
                    AsyncMock(side_effect=[complete, complete, queued]),
                ),
                patch.object(
                    tasks_router.db,
                    "compare_and_set_task_status",
                    AsyncMock(side_effect=[True, True]),
                ) as transition,
            ):
                result = await tasks_router.acquire_task_footage(
                    complete.id,
                    FootageAcquireRequest(queries=["robotics lab"]),
                )

            self.assertEqual(result.status, TaskStatus.QUEUED)
            self.assertEqual(
                json.loads((task_dir / ".footage").read_text(encoding="utf-8")),
                {
                    "resume_status": TaskStatus.COMPLETE.value,
                    "queries": ["robotics lab"],
                },
            )
            self.assertEqual(
                transition.await_args_list,
                [
                    unittest.mock.call(
                        complete.id,
                        TaskStatus.COMPLETE,
                        TaskStatus.SOURCING,
                        error_message=None,
                        expected_updated_at=complete.updated_at,
                        suppress_next_auto_publish=True,
                    ),
                    unittest.mock.call(
                        complete.id,
                        TaskStatus.SOURCING,
                        TaskStatus.QUEUED,
                        error_message=None,
                    ),
                ],
            )


class RenderRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_render_rejects_script_that_no_longer_matches_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            task_dir = Path(directory) / "task-rerender-stale-audio"
            task_dir.mkdir()
            script_path = task_dir / "script.txt"
            audio_path = task_dir / "audio.wav"
            script_path.write_text("Edited narration", encoding="utf-8")
            audio_path.write_bytes(b"RIFF")
            complete = make_task(
                "task-rerender-stale-audio",
                status=TaskStatus.COMPLETE,
                script_path=str(script_path),
                output_dir=str(task_dir),
            ).model_copy(update={"audio_path": str(audio_path)})

            with (
                patch.object(
                    tasks_router.db,
                    "get_task",
                    AsyncMock(side_effect=[complete, complete]),
                ),
                patch.object(
                    tasks_router.db,
                    "compare_and_set_task_status",
                    AsyncMock(),
                ) as transition,
                patch(
                    "backend.pipeline.composer._narration_manifest_failures",
                    return_value=["the current script changed after narration generation"],
                ),
            ):
                with self.assertRaises(HTTPException) as raised:
                    await tasks_router.render_task(complete.id)

            self.assertEqual(raised.exception.status_code, 409)
            self.assertIn("re-generate audio", raised.exception.detail)
            self.assertFalse((task_dir / worker.RENDER_MARKER).exists())
            transition.assert_not_awaited()

    async def test_complete_task_queues_compose_only_rerender(self):
        with tempfile.TemporaryDirectory() as directory:
            task_dir = Path(directory) / "task-rerender"
            task_dir.mkdir()
            script_path = task_dir / "script.txt"
            audio_path = task_dir / "audio" / "combined.wav"
            audio_path.parent.mkdir()
            script_path.write_text("Persisted narration", encoding="utf-8")
            audio_path.write_bytes(b"RIFF")
            complete = make_task(
                "task-rerender",
                status=TaskStatus.COMPLETE,
                script_path=str(script_path),
                output_dir=str(task_dir),
            ).model_copy(update={"audio_path": str(audio_path)})
            queued = complete.model_copy(
                update={
                    "status": TaskStatus.QUEUED,
                    "error_message": None,
                    "suppress_next_auto_publish": True,
                }
            )

            # An explicit new render choice supersedes every crash-left mode,
            # including partially staged markers, but only after the state CAS
            # succeeds.
            stale_markers = [
                name
                for name in worker.WORKER_MODE_MARKERS
                if name != worker.RENDER_MARKER
            ]
            for stale_name in stale_markers:
                (task_dir / stale_name).write_text("stale", encoding="utf-8")
            for marker_name in worker.WORKER_MODE_MARKERS:
                (task_dir / f"{marker_name}.tmp").write_text(
                    "partially staged",
                    encoding="utf-8",
                )

            with (
                patch.object(
                    tasks_router.db,
                    "get_task",
                    AsyncMock(side_effect=[complete, complete, queued]),
                ),
                patch.object(
                    tasks_router.db,
                    "compare_and_set_task_status",
                    AsyncMock(side_effect=[True, True]),
                ) as transition,
                patch(
                    "backend.pipeline.composer._narration_manifest_failures",
                    return_value=[],
                ),
            ):
                result = await tasks_router.render_task(complete.id)

            self.assertEqual(result.status, TaskStatus.QUEUED)
            self.assertTrue((task_dir / ".render").is_file())
            self.assertEqual(
                json.loads((task_dir / ".render").read_text(encoding="utf-8")),
                {
                    "resume_status": TaskStatus.COMPLETE.value,
                },
            )
            self.assertEqual(
                [
                    name
                    for name in worker.WORKER_MODE_MARKERS
                    if (task_dir / name).exists()
                ],
                [worker.RENDER_MARKER],
            )
            self.assertFalse(
                any(
                    (task_dir / f"{marker_name}.tmp").exists()
                    for marker_name in worker.WORKER_MODE_MARKERS
                )
            )
            self.assertEqual(
                transition.await_args_list,
                [
                    unittest.mock.call(
                        complete.id,
                        TaskStatus.COMPLETE,
                        TaskStatus.COMPOSING,
                        error_message=None,
                        expected_updated_at=complete.updated_at,
                        suppress_next_auto_publish=True,
                    ),
                    unittest.mock.call(
                        complete.id,
                        TaskStatus.COMPOSING,
                        TaskStatus.QUEUED,
                        error_message=None,
                    ),
                ],
            )

    async def test_stale_complete_snapshot_cannot_queue_a_second_render(self):
        with tempfile.TemporaryDirectory() as directory:
            task_dir = Path(directory) / "task-rerender-race"
            task_dir.mkdir()
            script_path = task_dir / "script.txt"
            audio_path = task_dir / "audio.wav"
            script_path.write_text("Persisted narration", encoding="utf-8")
            audio_path.write_bytes(b"RIFF")
            complete = make_task(
                "task-rerender-race",
                status=TaskStatus.COMPLETE,
                script_path=str(script_path),
                output_dir=str(task_dir),
            ).model_copy(update={"audio_path": str(audio_path)})
            stale_marker = task_dir / worker.TTS_RESUME_MARKER
            stale_marker.write_text("winner's intent", encoding="utf-8")
            stale_staged_marker = task_dir / f"{worker.TTS_RESUME_MARKER}.tmp"
            stale_staged_marker.write_text("winner's staged intent", encoding="utf-8")

            with (
                patch.object(tasks_router.db, "get_task", AsyncMock(return_value=complete)),
                patch.object(
                    tasks_router.db,
                    "compare_and_set_task_status",
                    AsyncMock(return_value=False),
                ) as transition,
                patch(
                    "backend.pipeline.composer._narration_manifest_failures",
                    return_value=[],
                ),
            ):
                with self.assertRaises(HTTPException) as raised:
                    await tasks_router.render_task(complete.id)

            self.assertEqual(raised.exception.status_code, 409)
            self.assertFalse((task_dir / ".render").exists())
            self.assertEqual(stale_marker.read_text(encoding="utf-8"), "winner's intent")
            self.assertEqual(
                stale_staged_marker.read_text(encoding="utf-8"),
                "winner's staged intent",
            )
            transition.assert_awaited_once_with(
                complete.id,
                TaskStatus.COMPLETE,
                TaskStatus.COMPOSING,
                error_message=None,
                expected_updated_at=complete.updated_at,
                suppress_next_auto_publish=True,
            )

    async def test_active_task_cannot_queue_an_overlapping_rerender(self):
        with tempfile.TemporaryDirectory() as directory:
            task_dir = Path(directory) / "task-active-render"
            task_dir.mkdir()
            active = make_task(
                "task-active-render",
                status=TaskStatus.COMPOSING,
                script_path=str(task_dir / "script.txt"),
                output_dir=str(task_dir),
            )

            with (
                patch.object(tasks_router.db, "get_task", AsyncMock(return_value=active)),
                patch.object(
                    tasks_router.db,
                    "compare_and_set_task_status",
                    AsyncMock(),
                ) as transition,
            ):
                with self.assertRaises(HTTPException) as raised:
                    await tasks_router.render_task(active.id)

            self.assertEqual(raised.exception.status_code, 409)
            self.assertFalse((task_dir / ".render").exists())
            transition.assert_not_awaited()

    async def test_failed_task_cannot_switch_mode_during_publication_safety_hold(self):
        with tempfile.TemporaryDirectory() as directory:
            task_dir = Path(directory) / "task-rerender-retry"
            task_dir.mkdir()
            script_path = task_dir / "script.txt"
            audio_path = task_dir / "audio.wav"
            script_path.write_text("Persisted narration", encoding="utf-8")
            audio_path.write_bytes(b"RIFF")
            stale_marker = task_dir / worker.REGEN_MARKER
            stale_marker.write_text("stale prior recovery", encoding="utf-8")
            failed = make_task(
                "task-rerender-retry",
                status=TaskStatus.FAILED,
                script_path=str(script_path),
                output_dir=str(task_dir),
            ).model_copy(
                update={
                    "audio_path": str(audio_path),
                    "publication_safety_hold": True,
                }
            )
            with (
                patch.object(
                    tasks_router.db,
                    "get_task",
                    AsyncMock(side_effect=[failed, failed]),
                ),
                patch.object(
                    tasks_router.db,
                    "compare_and_set_task_status",
                    AsyncMock(),
                ) as transition,
                patch(
                    "backend.pipeline.composer._narration_manifest_failures",
                    return_value=[],
                ),
            ):
                with self.assertRaises(HTTPException) as raised:
                    await tasks_router.render_task(failed.id)

            self.assertEqual(raised.exception.status_code, 409)
            self.assertIn("publication safety hold", raised.exception.detail)
            self.assertTrue(stale_marker.exists())
            self.assertFalse((task_dir / worker.RENDER_MARKER).exists())
            transition.assert_not_awaited()

    async def test_awaiting_review_rework_suppression_can_continue_to_render(self):
        with tempfile.TemporaryDirectory() as directory:
            task_dir = Path(directory) / "task-rerender-after-audio-review"
            task_dir.mkdir()
            script_path = task_dir / "script.txt"
            audio_path = task_dir / "audio.wav"
            script_path.write_text("Persisted narration", encoding="utf-8")
            audio_path.write_bytes(b"RIFF")
            awaiting = make_task(
                "task-rerender-after-audio-review",
                status=TaskStatus.AWAITING_REVIEW,
                script_path=str(script_path),
                output_dir=str(task_dir),
            ).model_copy(
                update={
                    "audio_path": str(audio_path),
                    "suppress_next_auto_publish": True,
                }
            )
            queued = awaiting.model_copy(update={"status": TaskStatus.QUEUED})
            transition = AsyncMock(side_effect=[True, True])

            with (
                patch.object(
                    tasks_router.db,
                    "get_task",
                    AsyncMock(side_effect=[awaiting, awaiting, queued]),
                ),
                patch.object(
                    tasks_router.db,
                    "compare_and_set_task_status",
                    transition,
                ),
                patch(
                    "backend.pipeline.composer._narration_manifest_failures",
                    return_value=[],
                ),
            ):
                result = await tasks_router.render_task(awaiting.id)

            self.assertEqual(result.status, TaskStatus.QUEUED)
            self.assertTrue((task_dir / worker.RENDER_MARKER).is_file())
            self.assertEqual(
                transition.await_args_list[0],
                unittest.mock.call(
                    awaiting.id,
                    TaskStatus.AWAITING_REVIEW,
                    TaskStatus.COMPOSING,
                    error_message=None,
                    expected_updated_at=awaiting.updated_at,
                    suppress_next_auto_publish=None,
                ),
            )


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
            configured_output_root = Path(directory) / "configured-output"

            with (
                # The persisted artifact root is authoritative for an existing
                # task even when Admin has since changed OUTPUTS_DIR.
                patch.object(config, "OUTPUTS_DIR", configured_output_root),
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
    async def test_worker_rerenders_complete_task_without_republishing(self):
        with tempfile.TemporaryDirectory() as directory:
            configured_output_root = Path(directory) / "configured-output"
            task_dir = Path(directory) / "persisted-output" / "task-rerender-worker"
            task_dir.mkdir(parents=True)
            marker = task_dir / worker.RENDER_MARKER
            marker.write_text(
                json.dumps({"resume_status": TaskStatus.COMPLETE.value}),
                encoding="utf-8",
            )
            queued = make_task(
                "task-rerender-worker",
                status=TaskStatus.QUEUED,
                script_path=str(task_dir / "script.txt"),
                output_dir=str(task_dir),
            ).model_copy(update={"suppress_next_auto_publish": True})
            publishing = queued.model_copy(
                update={
                    "status": TaskStatus.PUBLISHING,
                    "updated_at": "2026-08-20T00:01:00+00:00",
                }
            )
            final_complete = publishing.model_copy(
                update={
                    "status": TaskStatus.COMPLETE,
                    "updated_at": "2026-08-20T00:01:01+00:00",
                    "suppress_next_auto_publish": False,
                }
            )

            with (
                patch.object(config, "OUTPUTS_DIR", configured_output_root),
                patch.object(
                    worker,
                    "get_next_queued_task",
                    AsyncMock(side_effect=[queued, asyncio.CancelledError()]),
                ),
                patch.object(
                    worker,
                    "compare_and_set_task_status",
                    AsyncMock(side_effect=[True, True]),
                ) as transition,
                patch.object(
                    worker,
                    "get_task",
                    AsyncMock(side_effect=[publishing, final_complete]),
                ),
                patch.object(worker, "update_task", AsyncMock()) as update_task,
                patch.object(worker, "run_compose", AsyncMock()) as run_compose,
                patch.object(worker, "run_pipeline", AsyncMock()) as run_pipeline,
                patch.object(
                    worker,
                    "run_auto_publish_pipeline",
                    AsyncMock(),
                ) as auto_publish,
            ):
                with self.assertRaises(asyncio.CancelledError):
                    await worker._worker_loop()

            self.assertFalse(marker.exists())
            self.assertTrue((task_dir / "logs" / "pipeline.log").is_file())
            self.assertFalse(
                (configured_output_root / queued.id / "logs" / "pipeline.log").exists()
            )
            update_task.assert_not_awaited()
            run_compose.assert_awaited_once_with(queued, log=unittest.mock.ANY)
            run_pipeline.assert_not_awaited()
            auto_publish.assert_not_awaited()
            self.assertEqual(
                transition.await_args_list,
                [
                    unittest.mock.call(
                        queued.id,
                        TaskStatus.QUEUED,
                        TaskStatus.COMPOSING,
                        error_message=None,
                        expected_updated_at=queued.updated_at,
                    ),
                    unittest.mock.call(
                        publishing.id,
                        TaskStatus.PUBLISHING,
                        TaskStatus.COMPLETE,
                        error_message=None,
                        expected_updated_at=publishing.updated_at,
                        suppress_next_auto_publish=False,
                        publication_safety_hold=False,
                    ),
                ],
            )

    async def test_prepublication_rework_failure_remains_retryable(self):
        with tempfile.TemporaryDirectory() as directory:
            task_dir = Path(directory) / "task-rerender-stage-failure"
            task_dir.mkdir()
            marker = task_dir / worker.RENDER_MARKER
            marker.write_text("{}", encoding="utf-8")
            queued = make_task(
                "task-rerender-stage-failure",
                status=TaskStatus.QUEUED,
                script_path=str(task_dir / "script.txt"),
                output_dir=str(task_dir),
            ).model_copy(update={"suppress_next_auto_publish": True})

            with (
                patch.object(
                    worker,
                    "get_next_queued_task",
                    AsyncMock(side_effect=[queued, asyncio.CancelledError()]),
                ),
                patch.object(
                    worker,
                    "compare_and_set_task_status",
                    AsyncMock(return_value=True),
                ),
                patch.object(
                    worker,
                    "run_compose",
                    AsyncMock(side_effect=RuntimeError("render failed before review")),
                ),
                patch.object(worker, "update_task", AsyncMock()) as update_task,
            ):
                with self.assertRaises(asyncio.CancelledError):
                    await worker._worker_loop()

            self.assertFalse(marker.exists())
            update_task.assert_awaited_once_with(
                queued.id,
                status=TaskStatus.FAILED.value,
                error_message="render failed before review",
                publication_safety_hold=False,
            )

            # The retry keeps the completed-edition suppression even though
            # the first render failed before any external publication step.
            marker.write_text("{}", encoding="utf-8")
            retry_queued = queued.model_copy(
                update={"updated_at": "2026-08-20T00:01:00+00:00"}
            )
            publishing = retry_queued.model_copy(
                update={
                    "status": TaskStatus.PUBLISHING,
                    "updated_at": "2026-08-20T00:02:00+00:00",
                }
            )
            final_complete = publishing.model_copy(
                update={
                    "status": TaskStatus.COMPLETE,
                    "updated_at": "2026-08-20T00:03:00+00:00",
                    "suppress_next_auto_publish": False,
                }
            )
            with (
                patch.object(
                    worker,
                    "get_next_queued_task",
                    AsyncMock(side_effect=[retry_queued, asyncio.CancelledError()]),
                ),
                patch.object(
                    worker,
                    "compare_and_set_task_status",
                    AsyncMock(side_effect=[True, True]),
                ) as transition,
                patch.object(
                    worker,
                    "get_task",
                    AsyncMock(side_effect=[publishing, final_complete]),
                ),
                patch.object(worker, "run_compose", AsyncMock()),
                patch.object(worker, "update_task", AsyncMock()) as retry_update,
                patch.object(
                    worker,
                    "run_auto_publish_pipeline",
                    AsyncMock(),
                ) as auto_publish,
            ):
                with self.assertRaises(asyncio.CancelledError):
                    await worker._worker_loop()

            retry_update.assert_not_awaited()
            auto_publish.assert_not_awaited()
            self.assertEqual(
                transition.await_args_list[-1],
                unittest.mock.call(
                    publishing.id,
                    TaskStatus.PUBLISHING,
                    TaskStatus.COMPLETE,
                    error_message=None,
                    expected_updated_at=publishing.updated_at,
                    suppress_next_auto_publish=False,
                    publication_safety_hold=False,
                ),
            )

    async def test_quality_gate_finding_requeues_compose_instead_of_failed(self):
        with tempfile.TemporaryDirectory() as directory:
            task_dir = Path(directory) / "task-quality-retry"
            task_dir.mkdir()
            marker = task_dir / worker.RENDER_MARKER
            marker.write_text("{}", encoding="utf-8")
            queued = make_task(
                "task-quality-retry",
                status=TaskStatus.QUEUED,
                script_path=str(task_dir / "script.txt"),
                output_dir=str(task_dir),
            )

            with (
                patch.object(
                    worker,
                    "get_next_queued_task",
                    AsyncMock(side_effect=[queued, asyncio.CancelledError()]),
                ),
                patch.object(
                    worker,
                    "compare_and_set_task_status",
                    AsyncMock(return_value=True),
                ),
                patch.object(
                    worker,
                    "run_compose",
                    AsyncMock(
                        side_effect=worker.QualityGateRetry(
                            "scene-02 needs a more focused visual"
                        )
                    ),
                ),
                patch.object(worker, "update_task", AsyncMock()) as update_task,
            ):
                with self.assertRaises(asyncio.CancelledError):
                    await worker._worker_loop()

            self.assertTrue(marker.is_file())
            self.assertEqual(
                json.loads(marker.read_text(encoding="utf-8")),
                {"quality_retry": True},
            )
            update_task.assert_awaited_once_with(
                queued.id,
                status=TaskStatus.QUEUED.value,
                error_message=None,
                publication_safety_hold=False,
            )
            pipeline_log = (task_dir / "logs" / "pipeline.log").read_text(
                encoding="utf-8"
            )
            self.assertIn("Quality gate requested another compose cycle", pipeline_log)
            self.assertNotIn("Task failed", pipeline_log)

    async def test_publication_exception_fails_closed_with_durable_suppression(self):
        with tempfile.TemporaryDirectory() as directory:
            task_dir = Path(directory) / "task-publish-failure"
            task_dir.mkdir()
            marker = task_dir / worker.TTS_RESUME_MARKER
            marker.write_text("", encoding="utf-8")
            queued = make_task(
                "task-publish-failure",
                status=TaskStatus.QUEUED,
                script_path=str(task_dir / "script.txt"),
                output_dir=str(task_dir),
            )
            publishing = queued.model_copy(
                update={
                    "status": TaskStatus.PUBLISHING,
                    "updated_at": "2026-08-20T00:02:00+00:00",
                }
            )

            with (
                patch.object(
                    worker,
                    "get_next_queued_task",
                    AsyncMock(side_effect=[queued, asyncio.CancelledError()]),
                ),
                patch.object(
                    worker,
                    "compare_and_set_task_status",
                    AsyncMock(return_value=True),
                ),
                patch.object(worker, "get_task", AsyncMock(return_value=publishing)),
                patch.object(worker, "run_tts_resume", AsyncMock()),
                patch.object(
                    worker,
                    "run_auto_publish_pipeline",
                    AsyncMock(
                        return_value=unittest.mock.Mock(
                            action="failed",
                            reason="receipt write failed",
                        )
                    ),
                ),
                patch.object(worker, "update_task", AsyncMock()) as update_task,
            ):
                with self.assertRaises(asyncio.CancelledError):
                    await worker._worker_loop()

            update_task.assert_awaited_once_with(
                queued.id,
                status=TaskStatus.FAILED.value,
                error_message=(
                    "Automatic publication failed or has an indeterminate "
                    "external result: receipt write failed"
                ),
                publication_safety_hold=True,
            )

    async def test_post_completion_read_failure_does_not_reopen_delivery(self):
        with tempfile.TemporaryDirectory() as directory:
            task_dir = Path(directory) / "task-post-complete-read-failure"
            task_dir.mkdir()
            marker = task_dir / worker.RENDER_MARKER
            marker.write_text("{}", encoding="utf-8")
            queued = make_task(
                "task-post-complete-read-failure",
                status=TaskStatus.QUEUED,
                script_path=str(task_dir / "script.txt"),
                output_dir=str(task_dir),
            ).model_copy(update={"suppress_next_auto_publish": True})
            publishing = queued.model_copy(
                update={
                    "status": TaskStatus.PUBLISHING,
                    "updated_at": "2026-08-20T00:02:00+00:00",
                }
            )

            with (
                patch.object(
                    worker,
                    "get_next_queued_task",
                    AsyncMock(side_effect=[queued, asyncio.CancelledError()]),
                ),
                patch.object(
                    worker,
                    "compare_and_set_task_status",
                    AsyncMock(side_effect=[True, True]),
                ),
                patch.object(
                    worker,
                    "get_task",
                    AsyncMock(
                        side_effect=[
                            publishing,
                            RuntimeError("post-completion read failed"),
                        ]
                    ),
                ),
                patch.object(worker, "run_compose", AsyncMock()),
                patch.object(worker, "update_task", AsyncMock()) as update_task,
            ):
                with self.assertRaises(asyncio.CancelledError):
                    await worker._worker_loop()

            update_task.assert_not_awaited()

    async def test_worker_keeps_render_marker_when_compose_claim_cannot_persist(self):
        with tempfile.TemporaryDirectory() as directory:
            configured_output_root = Path(directory) / "configured-output"
            task_dir = (
                Path(directory)
                / "persisted-output"
                / "task-rerender-claim-failure"
            )
            task_dir.mkdir(parents=True)
            marker = task_dir / worker.RENDER_MARKER
            marker.write_text(
                json.dumps({"resume_status": TaskStatus.COMPLETE.value}),
                encoding="utf-8",
            )
            queued = make_task(
                "task-rerender-claim-failure",
                status=TaskStatus.QUEUED,
                script_path=str(task_dir / "script.txt"),
                output_dir=str(task_dir),
            ).model_copy(update={"suppress_next_auto_publish": True})

            with (
                patch.object(config, "OUTPUTS_DIR", configured_output_root),
                patch.object(
                    worker,
                    "get_next_queued_task",
                    AsyncMock(side_effect=[queued, asyncio.CancelledError()]),
                ),
                patch.object(
                    worker,
                    "compare_and_set_task_status",
                    AsyncMock(return_value=False),
                ) as transition,
                patch.object(worker, "update_task", AsyncMock()) as update_task,
                patch.object(worker, "run_compose", AsyncMock()) as run_compose,
                patch.object(worker, "run_pipeline", AsyncMock()) as run_pipeline,
            ):
                with self.assertRaises(asyncio.CancelledError):
                    await worker._worker_loop()

            self.assertTrue(marker.exists())
            run_compose.assert_not_awaited()
            run_pipeline.assert_not_awaited()
            update_task.assert_not_awaited()
            transition.assert_awaited_once_with(
                queued.id,
                TaskStatus.QUEUED,
                TaskStatus.COMPOSING,
                error_message=None,
                expected_updated_at=queued.updated_at,
            )

    async def test_worker_claims_resume_tts_marker_without_running_other_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            configured_output_root = Path(directory) / "configured-output"
            task_dir = Path(directory) / "persisted-output" / "task-worker"
            task_dir.mkdir(parents=True)
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
            publishing = queued.model_copy(
                update={
                    "status": TaskStatus.PUBLISHING,
                    "updated_at": "2026-08-20T00:01:00+00:00",
                }
            )
            final_complete = publishing.model_copy(
                update={
                    "status": TaskStatus.COMPLETE,
                    "updated_at": "2026-08-20T00:01:01+00:00",
                }
            )

            with (
                patch.object(config, "OUTPUTS_DIR", configured_output_root),
                patch.object(
                    worker,
                    "get_next_queued_task",
                    AsyncMock(side_effect=[queued, asyncio.CancelledError()]),
                ),
                patch.object(
                    worker,
                    "compare_and_set_task_status",
                    AsyncMock(side_effect=[True, True]),
                ) as transition,
                patch.object(
                    worker,
                    "get_task",
                    AsyncMock(side_effect=[publishing, final_complete]),
                ),
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
            update_task.assert_not_awaited()
            self.assertEqual(
                transition.await_args_list,
                [
                    unittest.mock.call(
                        queued.id,
                        TaskStatus.QUEUED,
                        TaskStatus.TTS,
                        error_message=None,
                        expected_updated_at=queued.updated_at,
                    ),
                    unittest.mock.call(
                        publishing.id,
                        TaskStatus.PUBLISHING,
                        TaskStatus.COMPLETE,
                        error_message=None,
                        expected_updated_at=publishing.updated_at,
                        suppress_next_auto_publish=None,
                        publication_safety_hold=False,
                    ),
                ],
            )
            auto_publish.assert_awaited_once_with(publishing)
            self.assertTrue((task_dir / "logs" / "pipeline.log").is_file())
            self.assertFalse(
                (configured_output_root / queued.id / "logs" / "pipeline.log").exists()
            )

    async def test_worker_maps_completed_footage_resume_to_publication_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            configured_output_root = Path(directory) / "configured-output"
            task_dir = Path(directory) / "persisted-output" / "task-footage-worker"
            task_dir.mkdir(parents=True)
            marker = task_dir / worker.FOOTAGE_MARKER
            marker.write_text(
                json.dumps(
                    {
                        "resume_status": TaskStatus.COMPLETE.value,
                        "queries": ["robotics lab"],
                    }
                ),
                encoding="utf-8",
            )
            queued = make_task(
                "task-footage-worker",
                status=TaskStatus.QUEUED,
                script_path=str(task_dir / "script.txt"),
                output_dir=str(task_dir),
            ).model_copy(update={"suppress_next_auto_publish": True})
            publishing = queued.model_copy(
                update={
                    "status": TaskStatus.PUBLISHING,
                    "updated_at": "2026-08-20T00:03:00+00:00",
                }
            )
            final_complete = publishing.model_copy(
                update={
                    "status": TaskStatus.COMPLETE,
                    "updated_at": "2026-08-20T00:03:01+00:00",
                    "suppress_next_auto_publish": False,
                }
            )

            with (
                patch.object(config, "OUTPUTS_DIR", configured_output_root),
                patch.object(
                    worker,
                    "get_next_queued_task",
                    AsyncMock(side_effect=[queued, asyncio.CancelledError()]),
                ),
                patch.object(
                    worker,
                    "compare_and_set_task_status",
                    AsyncMock(side_effect=[True, True]),
                ) as transition,
                patch.object(
                    worker,
                    "get_task",
                    AsyncMock(side_effect=[publishing, final_complete]),
                ),
                patch.object(worker, "update_task", AsyncMock()) as update_task,
                patch.object(
                    worker,
                    "run_footage_acquisition",
                    AsyncMock(),
                ) as run_footage,
                patch.object(
                    worker,
                    "run_auto_publish_pipeline",
                    AsyncMock(),
                ) as auto_publish,
            ):
                with self.assertRaises(asyncio.CancelledError):
                    await worker._worker_loop()

            self.assertFalse(marker.exists())
            run_footage.assert_awaited_once_with(
                queued,
                resume_status=TaskStatus.PUBLISHING.value,
                supplied_queries=["robotics lab"],
                log=unittest.mock.ANY,
            )
            auto_publish.assert_not_awaited()
            update_task.assert_not_awaited()
            self.assertEqual(
                transition.await_args_list,
                [
                    unittest.mock.call(
                        queued.id,
                        TaskStatus.QUEUED,
                        TaskStatus.SOURCING,
                        error_message=None,
                        expected_updated_at=queued.updated_at,
                    ),
                    unittest.mock.call(
                        publishing.id,
                        TaskStatus.PUBLISHING,
                        TaskStatus.COMPLETE,
                        error_message=None,
                        expected_updated_at=publishing.updated_at,
                        suppress_next_auto_publish=False,
                        publication_safety_hold=False,
                    ),
                ],
            )

    async def test_worker_fails_closed_when_multiple_mode_markers_exist(self):
        with tempfile.TemporaryDirectory() as directory:
            configured_output_root = Path(directory) / "configured-output"
            task_dir = Path(directory) / "persisted-output" / "task-conflict"
            task_dir.mkdir(parents=True)
            tts_marker = task_dir / worker.TTS_RESUME_MARKER
            render_marker = task_dir / worker.RENDER_MARKER
            tts_marker.write_text("", encoding="utf-8")
            render_marker.write_text("{}", encoding="utf-8")
            queued = make_task(
                "task-conflict",
                status=TaskStatus.QUEUED,
                script_path=str(task_dir / "script.txt"),
                output_dir=str(task_dir),
            ).model_copy(update={"suppress_next_auto_publish": True})

            with (
                patch.object(config, "OUTPUTS_DIR", configured_output_root),
                patch.object(
                    worker,
                    "get_next_queued_task",
                    AsyncMock(side_effect=[queued, asyncio.CancelledError()]),
                ),
                patch.object(
                    worker,
                    "compare_and_set_task_status",
                    AsyncMock(return_value=True),
                ) as transition,
                patch.object(worker, "update_task", AsyncMock()) as update_task,
                patch.object(worker, "run_tts_resume", AsyncMock()) as run_tts_resume,
                patch.object(worker, "run_compose", AsyncMock()) as run_compose,
                patch.object(worker, "run_pipeline", AsyncMock()) as run_pipeline,
            ):
                with self.assertRaises(asyncio.CancelledError):
                    await worker._worker_loop()

            self.assertTrue(tts_marker.exists())
            self.assertTrue(render_marker.exists())
            run_tts_resume.assert_not_awaited()
            run_compose.assert_not_awaited()
            run_pipeline.assert_not_awaited()
            update_task.assert_not_awaited()
            message = (
                "Conflicting worker mode markers; choose one recovery action: "
                f"{worker.TTS_RESUME_MARKER}, {worker.RENDER_MARKER}"
            )
            transition.assert_awaited_once_with(
                queued.id,
                TaskStatus.QUEUED,
                TaskStatus.FAILED,
                error_message=message,
                expected_updated_at=queued.updated_at,
                publication_safety_hold=False,
            )
            self.assertIn(
                message,
                (task_dir / "logs" / "pipeline.log").read_text(encoding="utf-8"),
            )

    async def test_worker_consumes_suppression_after_cross_mode_tts_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            configured_output_root = Path(directory) / "configured-output"
            task_dir = Path(directory) / "persisted-output" / "task-cross-mode"
            task_dir.mkdir(parents=True)
            marker = task_dir / worker.TTS_RESUME_MARKER
            marker.write_text("", encoding="utf-8")
            queued = make_task(
                "task-cross-mode",
                status=TaskStatus.QUEUED,
                script_path=str(task_dir / "script.txt"),
                output_dir=str(task_dir),
            ).model_copy(update={"suppress_next_auto_publish": True})
            publishing = queued.model_copy(
                update={
                    "status": TaskStatus.PUBLISHING,
                    "updated_at": "2026-08-20T00:02:00+00:00",
                }
            )
            final_complete = publishing.model_copy(
                update={
                    "status": TaskStatus.COMPLETE,
                    "updated_at": "2026-08-20T00:02:01+00:00",
                    "suppress_next_auto_publish": False,
                }
            )

            with (
                patch.object(config, "OUTPUTS_DIR", configured_output_root),
                patch.object(
                    worker,
                    "get_next_queued_task",
                    AsyncMock(side_effect=[queued, asyncio.CancelledError()]),
                ),
                patch.object(
                    worker,
                    "compare_and_set_task_status",
                    AsyncMock(side_effect=[True, True]),
                ) as transition,
                patch.object(
                    worker,
                    "get_task",
                    AsyncMock(side_effect=[publishing, final_complete]),
                ),
                patch.object(worker, "update_task", AsyncMock()) as update_task,
                patch.object(worker, "run_tts_resume", AsyncMock()) as run_tts_resume,
                patch.object(
                    worker,
                    "run_auto_publish_pipeline",
                    AsyncMock(),
                ) as auto_publish,
            ):
                with self.assertRaises(asyncio.CancelledError):
                    await worker._worker_loop()

            self.assertFalse(marker.exists())
            run_tts_resume.assert_awaited_once_with(queued, log=unittest.mock.ANY)
            auto_publish.assert_not_awaited()
            update_task.assert_not_awaited()
            self.assertEqual(
                transition.await_args_list,
                [
                    unittest.mock.call(
                        queued.id,
                        TaskStatus.QUEUED,
                        TaskStatus.TTS,
                        error_message=None,
                        expected_updated_at=queued.updated_at,
                    ),
                    unittest.mock.call(
                        publishing.id,
                        TaskStatus.PUBLISHING,
                        TaskStatus.COMPLETE,
                        error_message=None,
                        expected_updated_at=publishing.updated_at,
                        suppress_next_auto_publish=False,
                        publication_safety_hold=False,
                    ),
                ],
            )


class ManualPublicationRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_publication_reconciliation_can_confirm_deleted_test_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            failed = make_task(
                "task-publication-reconcile-delete",
                status=TaskStatus.FAILED,
                script_path=None,
                output_dir=directory,
            ).model_copy(update={"publication_safety_hold": True})
            complete = failed.model_copy(
                update={
                    "status": TaskStatus.COMPLETE,
                    "publication_safety_hold": False,
                    "updated_at": "2026-08-20T00:01:00+00:00",
                }
            )
            body = TaskPublicationReconcileRequest(
                acknowledge_no_unrecorded_publication=True,
                confirmed_deleted_test_targets=["x"],
            )
            transition = AsyncMock(return_value=True)
            with (
                patch.object(
                    tasks_router.db,
                    "get_task",
                    AsyncMock(side_effect=[failed, failed, complete]),
                ),
                patch.object(
                    tasks_router.db,
                    "compare_and_set_task_status",
                    transition,
                ),
                patch.object(
                    tasks_router,
                    "read_publication_manifest",
                    return_value={"task_id": failed.id, "platforms": {"x": {}}},
                ),
                patch.object(
                    tasks_router,
                    "reconcile_deleted_test_receipts",
                ) as reconcile_receipt,
            ):
                result = await tasks_router.reconcile_task_publication(
                    failed.id,
                    body,
                )

            self.assertEqual(result.status, TaskStatus.COMPLETE)
            reconcile_receipt.assert_called_once_with(failed, ["x"])
            transition.assert_awaited_once_with(
                failed.id,
                TaskStatus.FAILED,
                TaskStatus.COMPLETE,
                error_message=None,
                expected_updated_at=failed.updated_at,
                publication_safety_hold=False,
            )

    async def test_publication_safety_hold_requires_reconciliation_before_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            failed = make_task(
                "task-publish-safety-hold",
                status=TaskStatus.FAILED,
                script_path=None,
                output_dir=directory,
            ).model_copy(update={"publication_safety_hold": True})
            body = TaskPublicationRequest(
                targets=["youtube", "x"],
                test_mode=True,
                visibility="private",
            )
            transition = AsyncMock()
            with (
                patch.object(tasks_router.db, "get_task", AsyncMock(return_value=failed)),
                patch.object(
                    tasks_router.db,
                    "compare_and_set_task_status",
                    transition,
                ),
                patch.object(tasks_router, "plan_publication") as plan,
            ):
                with self.assertRaises(HTTPException) as raised:
                    await tasks_router.publish_task_now(failed.id, body)

            self.assertEqual(raised.exception.status_code, 409)
            self.assertIn("reconcile", raised.exception.detail)
            plan.assert_not_called()
            transition.assert_not_awaited()

    async def test_publication_preflight_refusal_preserves_completed_task(self):
        with tempfile.TemporaryDirectory() as directory:
            complete = make_task(
                "task-publish-preflight",
                status=TaskStatus.COMPLETE,
                script_path=None,
                output_dir=directory,
            )
            body = TaskPublicationRequest(
                targets=["youtube", "x"],
                test_mode=True,
                visibility="private",
            )
            transition = AsyncMock()
            with (
                patch.object(
                    tasks_router.db,
                    "get_task",
                    AsyncMock(side_effect=[complete, complete]),
                ),
                patch.object(
                    tasks_router.db,
                    "compare_and_set_task_status",
                    transition,
                ),
                patch.object(
                    tasks_router,
                    "plan_publication",
                    side_effect=ValueError("YouTube is disabled"),
                ),
            ):
                with self.assertRaises(HTTPException) as raised:
                    await tasks_router.publish_task_now(complete.id, body)

            self.assertEqual(raised.exception.status_code, 409)
            self.assertIn("disabled", raised.exception.detail)
            transition.assert_not_awaited()

    async def test_manual_publication_claims_and_finalizes_around_external_work(self):
        with tempfile.TemporaryDirectory() as directory:
            complete = make_task(
                "task-publish-success",
                status=TaskStatus.COMPLETE,
                script_path=None,
                output_dir=directory,
            )
            publishing = complete.model_copy(
                update={
                    "status": TaskStatus.PUBLISHING,
                    "updated_at": "2026-08-20T00:01:00+00:00",
                }
            )
            body = TaskPublicationRequest(
                targets=["youtube", "x"],
                test_mode=True,
                visibility="private",
            )
            plan = SimpleNamespace(
                pending_targets=("youtube", "x"),
                manifest={"task_id": complete.id, "platforms": {}},
            )
            result = {
                "task_id": complete.id,
                "platforms": {
                    "youtube": {"status": "published"},
                    "x": {"status": "published"},
                },
            }
            transition = AsyncMock(side_effect=[True, True])
            execute = AsyncMock(return_value=result)
            with (
                patch.object(
                    tasks_router.db,
                    "get_task",
                    AsyncMock(side_effect=[complete, complete, publishing]),
                ),
                patch.object(
                    tasks_router.db,
                    "compare_and_set_task_status",
                    transition,
                ),
                patch.object(tasks_router, "plan_publication", return_value=plan),
                patch.object(tasks_router, "execute_publication_plan", execute),
                patch.object(
                    tasks_router,
                    "publication_manifest_view",
                    return_value={"view": "current"},
                ),
            ):
                response = await tasks_router.publish_task_now(complete.id, body)

            self.assertEqual(response, {"view": "current"})
            execute.assert_awaited_once_with(
                publishing,
                plan=plan,
                test_mode=True,
            )
            self.assertEqual(
                transition.await_args_list,
                [
                    unittest.mock.call(
                        complete.id,
                        TaskStatus.COMPLETE,
                        TaskStatus.PUBLISHING,
                        error_message=None,
                        expected_updated_at=complete.updated_at,
                    ),
                    unittest.mock.call(
                        complete.id,
                        TaskStatus.PUBLISHING,
                        TaskStatus.COMPLETE,
                        error_message=None,
                        expected_updated_at=publishing.updated_at,
                        suppress_next_auto_publish=False,
                        publication_safety_hold=False,
                    ),
                ],
            )

    async def test_task_delete_refuses_an_active_apple_feed_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            complete = make_task(
                "task-delete-apple",
                status=TaskStatus.COMPLETE,
                script_path=None,
                output_dir=directory,
            )
            transition = AsyncMock()
            with (
                patch.object(
                    tasks_router.db,
                    "get_task",
                    AsyncMock(side_effect=[complete, complete]),
                ),
                patch.object(
                    tasks_router.db,
                    "compare_and_set_task_status",
                    transition,
                ),
                patch.object(
                    tasks_router,
                    "read_publication_manifest",
                    return_value={
                        "task_id": complete.id,
                        "platforms": {
                            "apple_podcast": {"status": "feed_ready"},
                        },
                    },
                ),
            ):
                with self.assertRaises(HTTPException) as raised:
                    await tasks_router.delete_task(complete.id)

            self.assertEqual(raised.exception.status_code, 409)
            self.assertIn("apple_podcast", raised.exception.detail)
            transition.assert_not_awaited()

    async def test_task_delete_refuses_a_durable_publication_safety_hold(self):
        with tempfile.TemporaryDirectory() as directory:
            task_dir = Path(directory) / "task-delete-safety-hold"
            task_dir.mkdir()
            failed = make_task(
                "task-delete-safety-hold",
                status=TaskStatus.FAILED,
                script_path=None,
                output_dir=str(task_dir),
            ).model_copy(update={"publication_safety_hold": True})
            transition = AsyncMock()
            delete_record = AsyncMock()
            with (
                patch.object(
                    tasks_router.db,
                    "get_task",
                    AsyncMock(side_effect=[failed, failed]),
                ),
                patch.object(
                    tasks_router.db,
                    "compare_and_set_task_status",
                    transition,
                ),
                patch.object(tasks_router.db, "delete_task", delete_record),
            ):
                with self.assertRaises(HTTPException) as raised:
                    await tasks_router.delete_task(failed.id)

            self.assertEqual(raised.exception.status_code, 409)
            self.assertIn("publication safety hold", raised.exception.detail)
            self.assertTrue(task_dir.is_dir())
            transition.assert_not_awaited()
            delete_record.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()


class FootageAcquisitionContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_partial_scout_fails_before_later_pipeline_stages(self):
        task = make_task('partial-footage', status=TaskStatus.FAILED,
                         script_path='/tmp/script.txt', output_dir='/tmp/partial-footage')
        with patch.object(orchestrator, 'acquire_footage', AsyncMock(return_value={
            'requested_clip_count': 6, 'clips': [{'id': 'clip-01'}],
        })):
            with self.assertRaisesRegex(RuntimeError, '1/6 eligible clips'):
                await orchestrator._acquire_task_footage(task, Path(task.output_dir), lambda _: None)

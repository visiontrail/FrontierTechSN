import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from backend import config, explainers as studio


def project_input():
    return studio.ProjectInput(
        title="How the shared worker works",
        research="Supplied research describes pausing a long production at a checkpoint while daily news runs.",
        tts_model="orpheus-en",
        voice="tara",
        scenes=[studio.Scene(id="scene-001", narration="A daily briefing takes priority.",
                             visual_prompt="A line character watches a newsroom clock.")],
    )


class ExplainerPriorityTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        outputs = patch.object(config, "OUTPUTS_DIR", self.directory)
        outputs.start()
        self.addCleanup(outputs.stop)

    async def asyncTearDown(self):
        await studio.stop_jobs()

    async def test_priority_pause_waits_for_cleanup_and_resumes_original_operation(self):
        project = studio.create_project(project_input())
        entered, cleaning, release_cleanup = asyncio.Event(), asyncio.Event(), asyncio.Event()
        directory = studio.project_dir(project["id"])
        checkpoint = directory / "complete-scene.json"
        checkpoint.write_text('{"verified": true}')
        progress = {"completed": 49, "total": 669, "message": "Scene 13"}
        calls = []

        async def render(current, root, update, log):
            calls.append(current["id"])
            self.assertEqual(checkpoint.read_text(), '{"verified": true}')
            if len(calls) == 1:
                update({"status": "generating_images", "progress": progress,
                        "scene_artifacts": [{"id": "scene-001", "image_hash": "saved"}]})
                entered.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    cleaning.set()
                    await release_cleanup.wait()
                    raise
            return {"video_path": "final.mp4"}

        with patch("backend.explainer_pipeline.render_project", side_effect=render):
            original_queue = studio.start(project["id"], "render")["queue"]
            runner = asyncio.create_task(studio.run_next_queued())
            await entered.wait()
            original_claim = studio.read_project(project["id"])["production"]
            pausing = asyncio.create_task(studio.pause_for_morning_desk(project["id"]))
            await cleaning.wait()
            self.assertFalse(pausing.done())
            self.assertFalse(runner.done())
            # Another patrol must wait without cancelling provider cleanup again.
            duplicate = asyncio.create_task(studio.pause_for_morning_desk(project["id"]))
            await asyncio.sleep(0)
            self.assertFalse(duplicate.done())
            release_cleanup.set()
            paused, duplicate_result = await asyncio.gather(pausing, duplicate)
            self.assertTrue(await runner)
            self.assertEqual(paused["status"], "interrupted")
            self.assertEqual(duplicate_result["cancellation_reason"], "morning_desk")
            self.assertEqual(paused["production"], original_claim)
            self.assertEqual(paused["scene_artifacts"], [{"id": "scene-001", "image_hash": "saved"}])
            self.assertEqual(paused["progress"]["completed"], 49)
            self.assertIsNone(studio.production_owner())
            self.assertFalse(studio._jobs)
            self.assertEqual(studio.resume_after_morning_desk(), [project["id"]])
            queued = studio.read_project(project["id"])
            self.assertEqual(queued["production"], original_claim)
            self.assertEqual(queued["queue"]["operation"], "render")
            self.assertEqual(queued["queue"]["enqueued_at"], original_queue["enqueued_at"])
            self.assertEqual(queued["progress"]["completed"], 49)
            self.assertIsNone(queued["priority_pause"])
            self.assertEqual(studio.resume_after_morning_desk(), [])
            self.assertTrue(await studio.run_next_queued())
        self.assertEqual(calls, [project["id"], project["id"]])
        self.assertEqual(studio.read_project(project["id"])["status"], "completed")
        self.assertEqual(checkpoint.read_text(), '{"verified": true}')

    async def test_restart_preserves_priority_pause_and_requeues_original_ahead_of_other_plans(self):
        original = studio.create_project(project_input())
        other = studio.create_project(project_input())
        claim = {"operation": "full", "started_at": "2026-10-08T00:00:00+00:00",
                 "enqueued_at": "2026-10-07T23:00:00+00:00"}
        studio.patch_project(original["id"], {"status": "generating_images", "production": claim})
        studio.start(other["id"], "render")
        paused = await studio.pause_for_morning_desk(original["id"])
        # A restart between marking the priority reason and normal cancellation cleanup.
        studio.patch_project(original["id"], {"status": "generating_images"})
        studio.recover_interrupted()
        recovered = studio.read_project(original["id"])
        self.assertEqual(recovered["status"], "interrupted")
        self.assertEqual(recovered["priority_pause"], paused["priority_pause"])
        self.assertEqual(recovered["cancellation_reason"], "morning_desk")
        self.assertIsNone(studio.production_owner())
        self.assertEqual(studio.resume_after_morning_desk(), [original["id"]])
        self.assertEqual(studio.read_project(original["id"])["queue"]["operation"], "full")
        self.assertEqual(studio.read_project(original["id"])["production"], claim)
        self.assertEqual([p["id"] for p in studio._queued_projects()], [original["id"], other["id"]])

    async def test_reserved_queue_retains_its_original_plan_and_timestamp(self):
        project = studio.create_project(project_input())
        studio.patch_project(project["id"], {"status": "interrupted", "cancellation_reason": "service_interruption",
                                             "production": {"operation": "render", "started_at": "2026-10-08T00:00:00+00:00"}})
        queue = studio.start(project["id"], "render")["queue"]
        await studio.pause_for_morning_desk(project["id"])
        studio.resume_after_morning_desk()
        self.assertEqual(studio.read_project(project["id"])["queue"], queue)
        studio.remove_from_queue(project["id"])
        self.assertIsNone(studio.production_owner())
        self.assertEqual(studio.resume_after_morning_desk(), [])

    async def test_manual_maintenance_and_failed_reservations_remain_manual(self):
        for status, reason in [("interrupted", "maintenance"), ("failed", None)]:
            with self.subTest(status=status):
                project = studio.create_project(project_input())
                progress = {"completed": 49, "total": 669, "message": "Repair needed"}
                studio.patch_project(project["id"], {"status": status, "error": "Manual recovery needed",
                    "cancellation_reason": reason, "progress": progress,
                    "production": {"operation": "render", "started_at": project["created_at"]}})
                await studio.pause_for_morning_desk(project["id"])
                self.assertIsNone(studio.production_owner())
                self.assertEqual(studio.resume_after_morning_desk(), [])
                restored = studio.read_project(project["id"])
                self.assertEqual((restored["status"], restored["cancellation_reason"]), (status, reason))
                self.assertEqual(restored["progress"], progress)
                self.assertEqual(restored["error"], "Manual recovery needed")
                self.assertEqual(studio.production_owner()["id"], project["id"])
                await studio.cancel(project["id"])

    async def test_user_cancellation_removes_priority_recovery(self):
        project = studio.create_project(project_input())
        studio.patch_project(project["id"], {"status": "generating_audio",
            "production": {"operation": "render", "started_at": project["created_at"]}})
        await studio.pause_for_morning_desk(project["id"])
        with self.assertRaisesRegex(studio.ConflictError, "Morning Desk"):
            studio.start(project["id"], "render")
        paused = studio.read_project(project["id"])
        with self.assertRaisesRegex(studio.ConflictError, "Morning Desk"):
            studio.update_project(project["id"], studio.ProjectUpdate(**project_input().model_dump(),
                                                                     revision=paused["revision"]))
        cancelled = await studio.cancel(project["id"])
        self.assertIsNone(cancelled["priority_pause"])
        self.assertEqual(cancelled["cancellation_reason"], "user")
        studio.recover_interrupted()
        self.assertEqual(studio.resume_after_morning_desk(), [])
        self.assertEqual(studio.read_project(project["id"])["status"], "cancelled")

    async def test_manual_pause_during_priority_wait_disables_auto_resume(self):
        project = studio.create_project(project_input())
        studio.patch_project(project["id"], {"status": "generating_audio",
            "production": {"operation": "render", "started_at": project["created_at"]}})
        await studio.pause_for_morning_desk(project["id"])
        await studio.pause(project["id"])
        self.assertIsNone(studio.production_owner())
        self.assertEqual(studio.resume_after_morning_desk(), [])
        restored = studio.read_project(project["id"])
        self.assertEqual(restored["status"], "interrupted")
        self.assertEqual(restored["cancellation_reason"], "maintenance")
        self.assertEqual(studio.production_owner()["id"], project["id"])

    async def test_priority_pause_is_visible_as_original_queue_reservation(self):
        project = studio.create_project(project_input())
        studio.patch_project(project["id"], {"status": "generating_images",
            "production": {"operation": "render", "started_at": project["created_at"]}})
        await studio.pause_for_morning_desk(project["id"])
        news = SimpleNamespace(id="daily-news", status="tts", generated_title="Daily briefing", source_title="Sources")
        with patch("backend.database.list_tasks", AsyncMock(return_value=[news])):
            state = await studio.queue_state()
        self.assertEqual(state["active"]["kind"], "morning_desk")
        self.assertEqual(state["reservation"]["id"], project["id"])
        self.assertEqual(state["reservation"]["operation"], "render")
        self.assertIn("paused for Morning Desk", state["message"])

    async def test_legacy_unknown_operation_is_never_guessed(self):
        project = studio.create_project(project_input())
        studio.patch_project(project["id"], {"status": "generating_images"})
        await studio.pause_for_morning_desk(project["id"])
        self.assertEqual(studio.resume_after_morning_desk(), [])
        restored = studio.read_project(project["id"])
        self.assertIsNone(restored["production"]["operation"])
        self.assertEqual(restored["status"], "interrupted")
        self.assertEqual(restored["cancellation_reason"], "service_interruption")

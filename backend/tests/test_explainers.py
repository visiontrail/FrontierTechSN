import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from backend import config, explainers as studio
from backend.models import TaskStatus
from backend.routers.explainers import router


def project_input(**changes):
    return studio.ProjectInput(**{"title": "How practice works",
                               "research": "Supplied research describes a learner recalling a concept before looking at the answer.",
                               "tts_model": "orpheus-en", "voice": "tara", **changes})


def scene(**changes):
    return {"id": "scene-001", "chapter": "Opening", "narration": "Try remembering before looking at the answer.",
            "visual_prompt": "A line character closes a book and considers a question.", "on_screen_text": "Recall", "motion": False, **changes}


class ExplainerStoreTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.directory = Path(temp.name)
        self.patch = patch.object(config, "OUTPUTS_DIR", self.directory)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_defaults_are_snapshots_and_edits_clear_delivery(self):
        with patch.object(config, "TTS_DEFAULT_MODEL", "vibevoice-0.5b"), patch.object(config, "TTS_DEFAULT_VOICE_1", "Carter"):
            defaults = studio.defaults()
        self.assertEqual((defaults["tts_model"], defaults["voice"]), ("pocket-tts-en", "alba"))
        chatterbox = next(model for model in defaults["tts_models"] if model["id"] == "chatterbox-multilingual")
        self.assertEqual(chatterbox["default_voice"], "space-en-f1")
        self.assertEqual(chatterbox["voices"][0], "space-en-f1")
        self.assertIn("Pocket reference clone", chatterbox["voice_labels"]["jane"])
        original = studio.create_project(studio.ProjectInput(title="How practice works", scenes=[scene()]))
        studio.save_defaults(studio.Defaults(tts_model="orpheus-en", voice="tara", style=studio.Style(accent_color="#112233")))
        snapshot = studio.read_project(original["id"])
        self.assertNotEqual(snapshot["style"]["accent_color"], "#112233")
        self.assertEqual((snapshot["tts_model"], snapshot["voice"]), ("pocket-tts-en", "alba"))
        finished = studio.patch_project(original["id"], {"status": "completed", "artifacts": {"video_path": "final.mp4"}})
        edited = studio.update_project(original["id"], studio.ProjectUpdate(**project_input(scenes=[scene(narration="A changed sentence.")]).model_dump(), revision=finished["revision"]))
        self.assertEqual(edited["status"], "script_ready")
        self.assertEqual(edited["artifacts"], {})
        self.assertEqual(studio.defaults()["style"]["accent_color"], "#112233")
        self.assertEqual((studio.defaults()["tts_model"], studio.defaults()["voice"]), ("orpheus-en", "tara"))

    def test_chatterbox_omitted_voice_defaults_without_migrating_explicit_selections(self):
        selected = studio.ProjectInput(title="An audio test", tts_model="chatterbox-multilingual")
        self.assertEqual(selected.voice, "space-en-f1")
        for model, voice in (("chatterbox-multilingual", "jane"), ("chatterbox-multilingual", "default"), ("pocket-tts-en", "jane")):
            with self.subTest(model=model, voice=voice):
                project = studio.create_project(studio.ProjectInput(title="An audio test", tts_model=model, voice=voice))
                self.assertEqual(studio.read_project(project["id"])["voice"], voice)

    def test_rejects_stale_edits_and_running_edits(self):
        original = studio.create_project(project_input())
        update = studio.ProjectUpdate(**project_input().model_dump(), revision=original["revision"])
        studio.patch_project(original["id"], {"status": "writing_script"})
        with self.assertRaises(studio.ConflictError):
            studio.update_project(original["id"], update)
        studio.patch_project(original["id"], {"status": "failed"})
        with self.assertRaisesRegex(studio.ConflictError, "changed"):
            studio.update_project(original["id"], update)

    def test_paths_and_duplicate_scene_ids_rejected(self):
        with self.assertRaises(FileNotFoundError):
            studio.project_dir("../secret")
        with self.assertRaises(ValueError):
            project_input(scenes=[scene(), scene()])
        project = studio.create_project(project_input())
        with self.assertRaises(FileNotFoundError):
            studio.asset_path(project["id"], "../../defaults.json")
        (studio.project_dir(project["id"]) / "link.json").symlink_to(self.directory / "outside.json")
        with self.assertRaises(ValueError):
            studio.asset_path(project["id"], "link.json")

    def test_restart_marks_interrupted_and_keeps_artifacts(self):
        project = studio.create_project(project_input(scenes=[scene()]))
        directory = studio.project_dir(project["id"])
        (directory / "good.wav").write_bytes(b"saved")
        studio.patch_project(project["id"], {"status": "generating_images"})
        studio.recover_interrupted()
        self.assertEqual(studio.read_project(project["id"])["status"], "interrupted")
        self.assertEqual((directory / "good.wav").read_bytes(), b"saved")

    def test_queue_is_persistent_fifo_and_rejects_edits_or_duplicate_enqueue(self):
        first = studio.create_project(project_input())
        second = studio.create_project(project_input())
        with patch.object(studio, "now", return_value="2026-09-28T00:00:00+00:00"):
            queued = studio.start(first["id"], "full")
            studio.start(second["id"], "script")
        self.assertFalse(studio._jobs)
        with self.assertRaises(studio.ConflictError):
            studio.start(first["id"], "full")
        with self.assertRaisesRegex(studio.ConflictError, "queued"):
            studio.update_project(first["id"], studio.ProjectUpdate(**project_input().model_dump(), revision=queued["revision"]))
        studio.recover_interrupted()
        self.assertEqual([p["id"] for p in studio._queued_projects()], [first["id"], second["id"]])
        public = studio.public_project(studio.read_project(second["id"]))
        self.assertEqual(public["status"], "queued")
        self.assertEqual(public["queue"], {"operation": "script", "enqueued_at": "2026-09-28T00:00:00+00:00", "position": 2})


class ExplainerAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.directory = Path(temp.name)
        self.patch = patch.object(config, "OUTPUTS_DIR", self.directory)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        app = FastAPI()
        app.include_router(router)
        self.client = AsyncClient(transport=ASGITransport(app=app), base_url="http://test")

    async def asyncTearDown(self):
        await studio.stop_jobs()
        await self.client.aclose()

    async def test_api_roundtrip_revision_and_missing_resources(self):
        defaults = (await self.client.get("/api/explainers/defaults")).json()
        self.assertEqual((defaults["tts_model"], defaults["voice"]), ("pocket-tts-en", "alba"))
        response = await self.client.post("/api/explainers", json=project_input().model_dump(exclude={"tts_model", "voice"}))
        self.assertEqual(response.status_code, 201)
        data = response.json()
        self.assertEqual((data["tts_model"], data["voice"]), ("pocket-tts-en", "alba"))
        saved = await self.client.put(f"/api/explainers/{data['id']}", json={**project_input(scenes=[scene()]).model_dump(), "revision": data["revision"]})
        self.assertEqual(saved.status_code, 200)
        stale = await self.client.put(f"/api/explainers/{data['id']}", json={**project_input().model_dump(), "revision": data["revision"]})
        self.assertEqual(stale.status_code, 409)
        listed = (await self.client.get("/api/explainers")).json()["projects"]
        self.assertEqual(listed[0]["scenes"][0]["narration"], scene()["narration"])
        self.assertEqual((await self.client.get("/api/explainers/invalid")).status_code, 404)
        self.assertEqual((await self.client.post(f"/api/explainers/{data['id']}/render")).status_code, 202)
        cancelled = await self.client.post(f"/api/explainers/{data['id']}/cancel")
        self.assertEqual(cancelled.json()["status"], "script_ready")
        self.assertIsNone(cancelled.json()["queue"])

    async def test_generation_single_owner_cancel_and_resume(self):
        project = studio.create_project(project_input())
        started = asyncio.Event()
        async def slow(*args):
            started.set()
            await asyncio.Event().wait()
        with patch.object(studio, "write_script", side_effect=slow):
            studio.start(project["id"], "script")
            runner = asyncio.create_task(studio.run_next_queued())
            await started.wait()
            with self.assertRaises(studio.ConflictError):
                studio.start(project["id"], "script")
            self.assertEqual((await studio.cancel(project["id"]))["status"], "cancelled")
            self.assertTrue(await runner)
        with patch.object(studio, "write_script", AsyncMock(return_value=[scene()])):
            studio.start(project["id"], "script")
            self.assertTrue(await studio.run_next_queued())
        self.assertEqual(studio.read_project(project["id"])["status"], "script_ready")

    async def test_watermark_settings_migrate_persist_and_validate_via_api(self):
        project = studio.create_project(project_input())
        path = studio.project_dir(project["id"]) / "project.json"
        legacy = json.loads(path.read_text())
        legacy["style"].pop("gemini_watermark_removal")
        legacy["style"].pop("gemini_watermark_region")
        path.write_text(json.dumps(legacy))
        url = f"/api/explainers/{project['id']}"
        loaded = (await self.client.get(url)).json()
        self.assertTrue(loaded["style"]["gemini_watermark_removal"])
        values = {key: loaded[key] for key in studio.ProjectInput.model_fields}
        values["style"]["gemini_watermark_removal"] = False
        values["style"]["gemini_watermark_region"]["x"] = 86
        saved = await self.client.put(url, json={**values, "revision": loaded["revision"]})
        self.assertEqual(saved.status_code, 200)
        reloaded = (await self.client.get(url)).json()
        self.assertFalse(reloaded["style"]["gemini_watermark_removal"])
        self.assertEqual(reloaded["style"]["gemini_watermark_region"]["x"], 86)
        defaults = {key: values[key] for key in studio.Defaults.model_fields}
        self.assertEqual((await self.client.put("/api/explainers/defaults", json=defaults)).status_code, 200)
        self.assertFalse((await self.client.get("/api/explainers/defaults")).json()["style"]["gemini_watermark_removal"])
        values["style"]["gemini_watermark_region"]["x"] = 99
        rejected = await self.client.put(url, json={**values, "revision": reloaded["revision"]})
        self.assertEqual(rejected.status_code, 422)

    async def test_failure_retains_existing_script_and_surfaces_error(self):
        project = studio.create_project(project_input(scenes=[scene()]))
        with patch.object(studio, "write_script", AsyncMock(side_effect=RuntimeError("Provider offline"))):
            studio.start(project["id"], "script")
            await studio.run_next_queued()
        result = studio.read_project(project["id"])
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["scenes"], [scene()])
        self.assertIn("Provider offline", result["error"])

    async def test_full_queue_writes_and_renders_in_fifo_order_without_overlap(self):
        first = studio.create_project(project_input(title="First"))
        second = studio.create_project(project_input(title="Second", scenes=[scene()]))
        entered_script, release_script = asyncio.Event(), asyncio.Event()
        events = []

        async def write(project, *args):
            events.append(("script", project["id"]))
            entered_script.set()
            await release_script.wait()
            return [scene()]

        async def render(project, *args):
            self.assertEqual(project["scenes"], [scene()])
            events.append(("render", project["id"]))
            return {"video_path": "final.mp4"}

        with patch.object(studio, "write_script", side_effect=write), patch("backend.explainer_pipeline.render_project", side_effect=render):
            studio.start(first["id"], "full")
            studio.start(second["id"], "full")
            self.assertEqual(events, [])
            runner = asyncio.create_task(studio.run_next_queued())
            await entered_script.wait()
            self.assertFalse(await studio.run_next_queued())
            self.assertEqual(studio.read_project(second["id"])["status"], "queued")
            release_script.set()
            self.assertTrue(await runner)
            self.assertEqual(studio.read_project(first["id"])["status"], "completed")
            self.assertTrue(await studio.run_next_queued())
            self.assertFalse(await studio.run_next_queued())
        self.assertEqual(events, [("script", first["id"]), ("render", first["id"]), ("render", second["id"])])
        self.assertEqual(studio.read_project(second["id"])["artifacts"], {"video_path": "final.mp4"})

    async def test_dequeue_retains_delivery_and_requeue_moves_to_end(self):
        first = studio.create_project(project_input(scenes=[scene()]))
        prior_progress = {"completed": 1, "total": 1, "message": "Completed"}
        studio.patch_project(first["id"], {"status": "completed", "error": None,
                             "artifacts": {"video_path": "final.mp4"}, "progress": prior_progress})
        second = studio.create_project(project_input())
        studio.start(first["id"], "full")
        studio.start(second["id"], "full")
        response = await self.client.post(f"/api/explainers/{first['id']}/dequeue")
        self.assertEqual(response.status_code, 200)
        cancelled = response.json()
        self.assertEqual(cancelled["status"], "completed")
        self.assertEqual(cancelled["progress"], prior_progress)
        self.assertEqual(cancelled["artifacts"]["video_path"], "final.mp4")
        self.assertIsNone(cancelled["queue"])
        studio.start(first["id"], "full")
        self.assertEqual([p["id"] for p in studio._queued_projects()], [second["id"], first["id"]])

    async def test_stale_dequeue_cannot_cancel_a_claimed_generation(self):
        project = studio.create_project(project_input())
        studio.start(project["id"], "script")
        started, release = asyncio.Event(), asyncio.Event()

        async def slow(*args):
            started.set()
            await release.wait()
            return [scene()]

        with patch.object(studio, "write_script", side_effect=slow):
            runner = asyncio.create_task(studio.run_next_queued())
            await started.wait()
            job = studio._jobs[project["id"]]
            response = await self.client.post(f"/api/explainers/{project['id']}/dequeue")
            self.assertEqual(response.status_code, 409)
            self.assertEqual(studio.read_project(project["id"])["status"], "writing_script")
            self.assertFalse(job.done())
            self.assertFalse(job.cancelling())
            release.set()
            self.assertTrue(await runner)
        self.assertEqual(studio.read_project(project["id"])["status"], "script_ready")
        self.assertEqual((await self.client.post(f"/api/explainers/{project['id']}/dequeue")).status_code, 409)

    async def test_failed_job_does_not_block_next_plan(self):
        first = studio.create_project(project_input())
        second = studio.create_project(project_input())
        studio.start(first["id"], "script")
        studio.start(second["id"], "script")
        with patch.object(studio, "write_script", AsyncMock(side_effect=[RuntimeError("Provider offline"), [scene()]])):
            self.assertTrue(await studio.run_next_queued())
            self.assertEqual(studio.read_project(first["id"])["status"], "failed")
            self.assertTrue(await studio.run_next_queued())
        self.assertEqual(studio.read_project(second["id"])["status"], "script_ready")

    async def test_worker_shutdown_propagates_and_retains_waiting_plans(self):
        first = studio.create_project(project_input())
        second = studio.create_project(project_input())
        studio.start(first["id"], "script")
        studio.start(second["id"], "full")
        started = asyncio.Event()

        async def slow(*args):
            started.set()
            await asyncio.Event().wait()

        with patch.object(studio, "write_script", side_effect=slow):
            runner = asyncio.create_task(studio.run_next_queued())
            await started.wait()
            runner.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await runner
        self.assertFalse(studio._jobs)
        self.assertEqual(studio.read_project(first["id"])["status"], "cancelled")
        self.assertEqual(studio.read_project(second["id"])["status"], "queued")

    async def test_queue_api_reports_active_news_and_runs_full_without_existing_scenes(self):
        project = studio.create_project(project_input())
        response = await self.client.post(f"/api/explainers/{project['id']}/queue")
        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json()["queue"]["operation"], "full")
        self.assertEqual((await self.client.post(f"/api/explainers/{project['id']}/queue")).status_code, 409)
        news = SimpleNamespace(id="daily-news", status=TaskStatus.TTS,
                               generated_title="Daily briefing", source_title="Sources")
        with patch("backend.database.list_tasks", AsyncMock(return_value=[news])):
            response = await self.client.get("/api/explainers/queue")
        self.assertEqual(response.status_code, 200)
        state = response.json()
        self.assertEqual(state["active"], {"kind": "morning_desk", "id": "daily-news", "title": "Daily briefing"})
        self.assertEqual([item["id"] for item in state["items"]], [project["id"]])
        self.assertIn("Morning Desk", state["message"])
        for inactive in [TaskStatus.QUEUED, TaskStatus.AWAITING_REVIEW, TaskStatus.COMPLETE, TaskStatus.FAILED]:
            news.status = inactive
            with patch("backend.database.list_tasks", AsyncMock(return_value=[news])):
                self.assertIsNone((await studio.queue_state())["active"])

    async def test_full_queue_checks_research_and_voice_before_enqueue(self):
        empty = studio.create_project(project_input(research="Too short"))
        response = await self.client.post(f"/api/explainers/{empty['id']}/queue")
        self.assertEqual(response.status_code, 422)
        self.assertEqual(studio.read_project(empty["id"])["status"], "draft")
        mismatch = studio.create_project(project_input(language="zh"))
        with self.assertRaisesRegex(ValueError, "language"):
            studio.start(mismatch["id"], "full")

    async def test_bulk_queue_responses_read_each_project_once(self):
        projects = [studio.create_project(project_input(title=f"Topic {index}")) for index in range(3)]
        for project in projects:
            studio.start(project["id"], "full")
        expected_positions = {project["id"]: index for index, project in enumerate(projects, start=1)}
        with patch("backend.database.list_tasks", AsyncMock(return_value=[])):
            for endpoint, key in [("/api/explainers", "projects"), ("/api/explainers/queue", "items")]:
                with patch.object(studio, "read_project", wraps=studio.read_project) as read:
                    response = await self.client.get(endpoint)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(read.call_count, len(projects))
                self.assertEqual({project["id"]: project["queue"]["position"] for project in response.json()[key]}, expected_positions)

    async def test_writing_repairs_invalid_shape_and_reuses_completed_chapter(self):
        project = studio.create_project(project_input(target_minutes=1))
        rows = [scene(id=f"scene-{i + 1:03d}", narration="Close the book and recall the idea before checking your answer.", motion="false") for i in range(12)]
        responses = ["not json", json.dumps({"chapters": [{"title": "Recall", "brief": "A concrete learning situation"}]}), json.dumps({"scenes": rows})]
        with patch("backend.pipeline.digester._resolve_provider", AsyncMock(return_value=("endpoint", "model", "key"))), patch("backend.pipeline.digester._chat", AsyncMock(side_effect=responses)) as chat:
            scenes = await studio.write_script(project, studio.project_dir(project["id"]), lambda message: None)
            self.assertEqual(len(scenes), 12)
            self.assertTrue(all(s["motion"] is False for s in scenes))
            self.assertEqual(chat.await_count, 3)
            reused = await studio.write_script(project, studio.project_dir(project["id"]), lambda message: None)
            self.assertEqual(scenes, reused)
            self.assertEqual(chat.await_count, 3)
            self.assertIn("untrusted source material", chat.call_args.args[0])

    async def test_rejects_language_mismatch_before_media_call(self):
        project = studio.create_project(project_input(language="zh", scenes=[scene()]))
        with self.assertRaisesRegex(ValueError, "language"):
            studio.start(project["id"], "render")
        realtime = studio.create_project(studio.ProjectInput(title="中文", language="zh", tts_model="vibevoice-0.5b", voice="Carter", scenes=[scene()]))
        with self.assertRaisesRegex(ValueError, "language"):
            studio.start(realtime["id"], "render")
        self.assertEqual(studio.tts_language("vibevoice-1.5b"), "multilingual")

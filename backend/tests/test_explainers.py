import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from backend import config, explainers as studio
from backend.routers.explainers import router


def project_input(**changes):
    return studio.ProjectInput(title="How practice works", research="Supplied research describes a learner recalling a concept before looking at the answer.",
                               tts_model="orpheus-en", voice="tara", **changes)


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
        self.assertEqual(cancelled.json()["status"], "cancelled")

    async def test_generation_single_owner_cancel_and_resume(self):
        project = studio.create_project(project_input())
        started = asyncio.Event()
        async def slow(*args):
            started.set()
            await asyncio.Event().wait()
        with patch.object(studio, "write_script", side_effect=slow):
            studio.start(project["id"], "script")
            await started.wait()
            with self.assertRaises(studio.ConflictError):
                studio.start(project["id"], "script")
            self.assertEqual((await studio.cancel(project["id"]))["status"], "cancelled")
        with patch.object(studio, "write_script", AsyncMock(return_value=[scene()])):
            studio.start(project["id"], "script")
            await studio._jobs[project["id"]]
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
            await studio._jobs[project["id"]]
        result = studio.read_project(project["id"])
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["scenes"], [scene()])
        self.assertIn("Provider offline", result["error"])

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

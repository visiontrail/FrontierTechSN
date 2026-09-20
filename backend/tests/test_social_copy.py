import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from twitter_text import parse_tweet

from backend import social_copy as sc
from backend import worker
from backend.models import TaskConfig, TaskResponse
from backend.routers import tasks


def draft(**changes):
    return json.dumps({
        "youtube_title": "Huawei outlines its next AI chips",
        "youtube_show_notes": "Huawei outlined its chip plans. The schedule remains a company target.",
        "x_paragraphs": ["Huawei outlined its next AI chips.", "The schedule is a company target.", "Can it deliver?"],
        **changes,
    }, ensure_ascii=False)


class CopyValidationTests(unittest.TestCase):
    def setUp(self):
        self.inputs = {"final_narration": "Huawei outlined its next AI chips.", "source_references": []}

    def test_preserves_three_semantic_paragraphs(self):
        value = sc._validate(draft(), self.inputs)
        self.assertEqual(value["x_post"], "Huawei outlined its next AI chips.\n\nThe schedule is a company target.\n\nCan it deliver?")
        self.assertEqual(value["x_weighted_length"], parse_tweet(value["x_post"]).weightedLength)

    def test_two_sentence_post_stays_in_one_paragraph(self):
        for paragraph in [
            "Huawei outlined its next AI chips. What would the schedule mean for developers?",
            "华为公布下一代 AI 芯片计划。这一时间表对开发者意味着什么？",
        ]:
            with self.subTest(paragraph=paragraph):
                value = sc._validate(draft(x_paragraphs=[paragraph]), self.inputs)
                self.assertEqual(value["x_post"], paragraph)

    def test_breaks_follow_meaning_instead_of_sentence_pairs(self):
        paragraphs = [
            "  Huawei outlined its next AI chips.  ",
            "The schedule is a company target. What would it mean for developers?",
        ]
        value = sc._validate(draft(x_paragraphs=paragraphs), self.inputs)
        self.assertEqual(value["x_post"], "\n\n".join(p.strip() for p in paragraphs))

    def test_paragraph_breaks_count_toward_platform_limit(self):
        value = sc._validate(draft(x_paragraphs=["a" * 138, "b" * 140]), self.inputs)
        self.assertEqual(value["x_weighted_length"], 280)
        with self.assertRaisesRegex(ValueError, "280"):
            sc._validate(draft(x_paragraphs=["a" * 139, "b" * 140]), self.inputs)

    def test_preserves_numbers_initials_and_chinese_sentences(self):
        value = sc._validate(draft(x_paragraphs=["Dr. Li discussed U.S. investment of $23.5 billion.", "这仍是计划。", "何时落地？"]), self.inputs)
        self.assertIn("$23.5 billion.", value["x_post"])
        self.assertIn("\n\n何时落地？", value["x_post"])

    def test_counts_cjk_emoji_urls_and_normalization(self):
        for text, expected in [("中文", 4), ("👨‍👩‍👧‍👦", 2), ("🙋🏽", 2), ("cafe\u0301", 4), ("https://example.com/a-long-path", 23)]:
            with self.subTest(text=text):
                self.assertEqual(parse_tweet(text).weightedLength, expected)
        with self.assertRaisesRegex(ValueError, "280"):
            sc._validate(draft(x_paragraphs=["中" * 139 + "。", "继续。"]), self.inputs)

    def test_rejects_invalid_platform_content(self):
        invalid = [
            {"youtube_title": "x" * 101}, {"youtube_show_notes": "x" * 5001},
            {"youtube_title": " "}, {"youtube_show_notes": " "},
            {"youtube_title": "A\nB"}, {"youtube_title": "<Title>"},
            {"x_paragraphs": [" ", "A fact."]},
            {"x_paragraphs": []},
            {"x_paragraphs": ["One.", "Two.", "Three.", "Four."]},
            {"x_paragraphs": ["Hello\nworld.", "Next."]},
            {"x_paragraphs": ["Hello\rworld."]},
            {"x_paragraphs": ["#AI #tech #news", "Facts."]},
            {"x_paragraphs": ["Read https://invented.example.com/video"]},
            {"youtube_show_notes": "https://invented.example.com/video"},
            {"youtube_show_notes": "Intro\n00:30 Chips"},
        ]
        for change in invalid:
            with self.subTest(change=list(change)):
                with self.assertRaises(ValueError):
                    sc._validate(draft(**change), self.inputs)

    def test_allows_only_supplied_source_urls(self):
        self.inputs["source_references"] = [{"url": "https://example.com/report"}]
        value = sc._validate(draft(youtube_show_notes="Source: https://example.com/report"), self.inputs)
        self.assertIn("https://example.com/report", value["youtube_show_notes"])


class CopyGenerationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.script = self.root / "script.txt"
        self.script.write_text("Huawei outlined its next AI chips. The schedule remains a target.")
        self.task = TaskResponse(id="copy-test", created_at="2026-09-18", updated_at="2026-09-18",
                                 source_type="topic", status="complete", config=TaskConfig(),
                                 script_path=str(self.script), output_dir=str(self.root))
        self.chat = AsyncMock(return_value=draft())
        for name, value in [
            ("_chat", self.chat),
            ("_resolve_provider", AsyncMock(return_value=("endpoint", "model", "key"))),
        ]:
            patcher = patch.object(sc, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(sc.db, "get_task", AsyncMock(return_value=self.task))
        patcher.start()
        self.addCleanup(patcher.stop)

    async def asyncTearDown(self):
        await sc.stop_generations()

    async def generate(self, regenerate=False):
        response = sc.start_generation(self.task, regenerate=regenerate)
        if self.task.id in sc._running:
            await sc._running[self.task.id]
            await asyncio.sleep(0)
        return sc.read_copy(self.task), response

    async def test_loads_full_pinned_skill_and_persists_copy(self):
        result, started = await self.generate()
        self.assertEqual(started["status"], "generating")
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["humanizer_revision"], sc.HUMANIZER_REVISION)
        prompt = self.chat.await_args.args[0]
        self.assertIn((sc.HUMANIZER_DIR / "SKILL.md").read_text(), prompt)
        self.assertIn((sc.config.PROMPTS_DIR / "title_strategy.txt").read_text().strip(), prompt)
        self.assertEqual(len(result["prompt_sha256"]), 64)
        self.assertFalse(self.chat.await_args.kwargs["enable_skills"])
        payload = json.loads(self.chat.await_args.args[1])
        self.assertEqual(payload["final_narration"], self.script.read_text())
        self.assertNotIn("source_title", payload)
        self.assertTrue((self.root / "social_copy.json").exists())

    async def test_existing_copy_is_idempotent_and_regeneration_is_explicit(self):
        await self.generate()
        await self.generate()
        self.assertEqual(self.chat.await_count, 1)
        await self.generate(regenerate=True)
        self.assertEqual(self.chat.await_count, 2)

    async def test_concurrent_requests_share_one_job(self):
        first = sc.start_generation(self.task)
        job = sc._running[self.task.id]
        second = sc.start_generation(self.task, regenerate=True)
        self.assertIs(job, sc._running[self.task.id])
        self.assertEqual(first["status"], second["status"])
        await job
        self.assertEqual(self.chat.await_count, 1)

    async def test_validation_repair_also_loads_humanizer(self):
        invalid = draft(x_paragraphs=["x" * 280, "More."])
        self.chat.side_effect = [invalid, draft()]
        result, _ = await self.generate()
        self.assertEqual(result["status"], "ready")
        self.assertEqual(self.chat.await_count, 2)
        self.assertIn("validation_feedback", json.loads(self.chat.await_args.args[1]))
        self.assertEqual(json.loads(self.chat.await_args.args[1])["previous_draft"], invalid)
        self.assertIn("<humanizer_skill>", self.chat.await_args.args[0])
        for call in self.chat.await_args_list:
            self.assertIn("<shared_title_strategy>", call.args[0])

    async def test_retry_failure_preserves_last_good_copy(self):
        good, _ = await self.generate()
        self.chat.return_value = "invalid JSON"
        result, _ = await self.generate(regenerate=True)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["copy"], good["copy"])
        self.assertEqual(self.chat.await_count, 4)

    async def test_changed_script_marks_existing_copy_stale(self):
        await self.generate()
        self.script.write_text("A revised narration.")
        self.assertTrue(sc.read_copy(self.task)["stale"])

    async def test_change_during_generation_is_not_promoted(self):
        async def change(*args, **kwargs):
            self.script.write_text("Changed while generating.")
            return draft()
        self.chat.side_effect = change
        result, _ = await self.generate()
        self.assertEqual(result["status"], "failed")
        self.assertIsNone(result["copy"])
        self.assertIn("changed", result["error"])

    async def test_interrupted_run_is_retryable(self):
        sc._write(sc._path(self.task), {"status": "generating", "copy": None})
        self.assertEqual(sc.read_copy(self.task)["status"], "failed")
        result, _ = await self.generate()
        self.assertEqual(result["status"], "ready")

    async def test_routes_generate_read_and_reject_missing_script(self):
        app = FastAPI()
        app.include_router(tasks.router)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            self.assertEqual((await client.get("/api/tasks/copy-test/social-copy")).json()["status"], "missing")
            response = await client.post("/api/tasks/copy-test/social-copy")
            self.assertEqual(response.status_code, 202)
            await sc._running[self.task.id]
            await asyncio.sleep(0)
            self.assertEqual((await client.get("/api/tasks/copy-test/social-copy")).json()["status"], "ready")
            self.script.unlink()
            response = await client.post("/api/tasks/copy-test/social-copy?regenerate=true")
            self.assertEqual(response.status_code, 409)
            with patch.object(sc.db, "get_task", AsyncMock(return_value=None)):
                self.assertEqual((await client.get("/api/tasks/unknown/social-copy")).status_code, 404)

    async def test_running_pipeline_cannot_generate_copy(self):
        self.task.status = "digesting"
        with self.assertRaisesRegex(ValueError, "Wait"):
            sc.start_generation(self.task)

    async def test_worker_generates_copy_after_completion_without_republishing(self):
        queued = self.task.model_copy(update={"status": "queued", "suppress_next_auto_publish": True})
        publishing = self.task.model_copy(update={"status": "publishing", "suppress_next_auto_publish": True})
        for failure in (None, ValueError("script unavailable")):
            with (
                self.subTest(failure=failure),
                patch.object(worker, "get_next_queued_task", AsyncMock(side_effect=[queued, asyncio.CancelledError()])),
                patch.object(worker, "get_task", AsyncMock(side_effect=[publishing, self.task])),
                patch.object(worker, "compare_and_set_task_status", AsyncMock(return_value=True)) as transition,
                patch.object(worker, "run_pipeline", AsyncMock()),
                patch.object(worker, "update_task", AsyncMock()) as update,
                patch.object(worker, "run_auto_publish_pipeline", AsyncMock()) as publish,
                patch.object(sc, "start_generation", side_effect=failure) as start,
            ):
                with self.assertRaises(asyncio.CancelledError):
                    await worker._worker_loop()
                start.assert_called_once_with(self.task)
                publish.assert_not_awaited()
                update.assert_not_awaited()
                self.assertEqual(transition.await_count, 2)
                self.assertEqual(transition.await_args.args[2], "complete")


if __name__ == "__main__":
    unittest.main()

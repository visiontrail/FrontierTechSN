import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from backend import config, prompts_registry
from backend.models import TaskConfig, TaskResponse
from backend.pipeline import orchestrator, title


class TitleCleaningTests(unittest.TestCase):
    def test_platform_limit_is_not_silently_truncated(self):
        self.assertEqual(title.clean_generated_title("中" * 100), "中" * 100)
        for value in ("x" * 101, "First title\nSecond title", "<Title>"):
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                title.clean_generated_title(value)

    def test_removes_fences_label_and_quotes(self):
        value = "```\n标题： “一座城市如何重新夺回街道”\n```"

        self.assertEqual(title.clean_generated_title(value), "一座城市如何重新夺回街道")

    def test_rejects_an_empty_agent_result(self):
        with self.assertRaisesRegex(RuntimeError, "empty title"):
            title.clean_generated_title("  \n ")


class TitleAgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_repairs_invalid_output_with_the_same_shared_strategy(self):
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(title, "_resolve_provider", AsyncMock(return_value=("endpoint", "model", "key"))),
                patch.object(title, "_chat", AsyncMock(side_effect=["x" * 101, "A Supported Title"])) as chat,
            ):
                artifact = await title.generate_title(
                    task_id="repair", task_dir=Path(directory), source_title="Ignored",
                    summary=None, script="The final narration.",
                )
            self.assertEqual(artifact.title, "A Supported Title")
            calls = chat.await_args_list
            self.assertEqual(len(calls), 2)
            strategy = (config.PROMPTS_DIR / "title_strategy.txt").read_text().strip()
            self.assertIn(strategy, calls[0].args[0])
            self.assertEqual(calls[0].args[0], calls[1].args[0])
            self.assertIn("do not truncate", json.loads(calls[1].args[1])["validation_feedback"])
            self.assertEqual(len(json.loads(Path(artifact.manifest_path).read_text())["prompt_sha256"]), 64)

    async def test_invalid_output_exhausts_repairs_without_overwriting_saved_title(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "title" / "title.txt"
            path.parent.mkdir()
            path.write_text("Last good title\n")
            with (
                patch.object(title, "_resolve_provider", AsyncMock(return_value=("endpoint", "model", "key"))),
                patch.object(title, "_chat", AsyncMock(return_value="A\nB")) as chat,
                self.assertRaisesRegex(RuntimeError, "after 3 attempts"),
            ):
                await title.generate_title(task_id="bad", task_dir=Path(directory),
                                           source_title="", summary=None, script="Final narration.")
            self.assertEqual(chat.await_count, 3)
            self.assertEqual(path.read_text(), "Last good title\n")
            self.assertEqual(json.loads(path.with_name("manifest.json").read_text())["status"], "failed")

    async def test_runs_a_dedicated_agent_and_persists_the_result(self):
        with tempfile.TemporaryDirectory() as directory:
            task_dir = Path(directory)
            complete = AsyncMock(return_value="The Quiet Revolution Reclaiming Our Streets")
            with (
                patch.object(
                    title,
                    "_resolve_provider",
                    AsyncMock(return_value=("https://ai.example/v1", "title-model", "secret")),
                ),
                patch("backend.pipeline.agent.agent_complete", complete),
            ):
                artifact = await title.generate_title(
                    task_id="title-test",
                    task_dir=task_dir,
                    source_title="Unreviewed source says 235 billion",
                    summary={"thesis": "Unreviewed brief contains a rejected claim."},
                    script="Cities inherited car-first streets, but residents are changing them.",
                )

            self.assertEqual(artifact.title, "The Quiet Revolution Reclaiming Our Streets")
            self.assertEqual(Path(artifact.title_path).read_text().strip(), artifact.title)
            manifest = json.loads(Path(artifact.manifest_path).read_text())
            self.assertEqual(manifest["status"], "ready")
            self.assertEqual(manifest["executor"], "claude_agent_sdk")
            self.assertEqual(manifest["title"], artifact.title)
            self.assertEqual(complete.await_args.kwargs["label"], "Title agent")
            self.assertEqual(complete.await_args.kwargs["max_tokens"], 1024)
            self.assertFalse(complete.await_args.kwargs["enable_skills"])
            self.assertTrue(complete.await_args.kwargs["disable_thinking"])
            payload = json.loads(complete.await_args.args[1])
            self.assertEqual(payload, {
                "final_audio_script": "Cities inherited car-first streets, but residents are changing them.",
            })
            self.assertNotIn("Unreviewed", complete.await_args.args[1])

    async def test_falls_back_to_http_when_the_title_cli_exits(self):
        with tempfile.TemporaryDirectory() as directory:
            task_dir = Path(directory)
            with (
                patch.object(
                    title,
                    "_resolve_provider",
                    AsyncMock(return_value=("https://ai.example/v1", "title-model", "secret")),
                ),
                patch.object(config, "AI_BACKEND", "agent_sdk"),
                patch.object(config, "AI_HTTP_FALLBACK", True),
                patch(
                    "backend.pipeline.agent.agent_complete",
                    AsyncMock(side_effect=RuntimeError("claude CLI exited 1")),
                ),
                patch(
                    "backend.pipeline.digester._chat_http",
                    AsyncMock(return_value="A Reliable Fallback Title"),
                ) as fallback,
            ):
                artifact = await title.generate_title(
                    task_id="title-fallback",
                    task_dir=task_dir,
                    source_title="Source",
                    summary={"thesis": "Brief"},
                    script="A complete script.",
                )

            self.assertEqual(artifact.title, "A Reliable Fallback Title")
            self.assertEqual(fallback.await_count, 1)

    async def test_composer_receives_the_generated_title(self):
        with tempfile.TemporaryDirectory() as directory:
            task_dir = Path(directory) / "compose-title-test"
            task_dir.mkdir()
            script_path = task_dir / "script.txt"
            audio_path = task_dir / "audio.wav"
            script_path.write_text("Narration", encoding="utf-8")
            audio_path.write_bytes(b"RIFF")
            task = TaskResponse(
                id="compose-title-test",
                created_at="2026-08-09T00:00:00+00:00",
                updated_at="2026-08-09T00:00:00+00:00",
                source_type="youtube",
                source_title="Original Source Title",
                generated_title="Agent Publication Title",
                status="awaiting_review",
                # This test isolates title propagation; the daily-news default
                # intentionally enables a real Gemini music generation step.
                config=TaskConfig(background_music_enabled=False),
                script_path=str(script_path),
                audio_path=str(audio_path),
            )
            compose = AsyncMock(return_value=str(task_dir / "video.mp4"))

            with (
                patch.object(config, "OUTPUTS_DIR", Path(directory)),
                patch.object(orchestrator, "update_task", AsyncMock()),
                patch.object(orchestrator, "compose_video", compose),
            ):
                await orchestrator.run_compose(task)

            self.assertEqual(compose.await_args.kwargs["title"], "Agent Publication Title")


class TitlePromptRegistryTests(unittest.TestCase):
    def test_admin_strategy_edits_are_loaded_at_request_time(self):
        from backend.title_strategy import with_title_strategy

        spec = prompts_registry.get_spec("title_strategy")
        self.assertIsNotNone(spec)
        with tempfile.TemporaryDirectory() as directory, patch.object(config, "PROMPTS_DIR", Path(directory)):
            prompts_registry.write_content(spec, "First editorial rule")
            self.assertIn("First editorial rule", with_title_strategy("Title task"))
            prompts_registry.write_content(spec, "Revised editorial rule")
            self.assertIn("Revised editorial rule", with_title_strategy("Upload task"))
            prompts_registry.write_content(spec, " ")
            with self.assertRaisesRegex(ValueError, "must not be empty"):
                with_title_strategy("Title task")

    def test_title_prompt_is_editable_in_admin(self):
        spec = prompts_registry.get_spec("title")

        self.assertIsNotNone(spec)
        self.assertEqual(spec.file, "title.txt")
        self.assertIn("independent", spec.description.lower())


if __name__ == "__main__":
    unittest.main()

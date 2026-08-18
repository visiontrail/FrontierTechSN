import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from backend import config, prompts_registry
from backend.models import TaskConfig, TaskResponse
from backend.pipeline import orchestrator, title


class TitleCleaningTests(unittest.TestCase):
    def test_removes_fences_label_and_quotes(self):
        value = "```\n标题： “一座城市如何重新夺回街道”\n```"

        self.assertEqual(title.clean_generated_title(value), "一座城市如何重新夺回街道")

    def test_rejects_an_empty_agent_result(self):
        with self.assertRaisesRegex(RuntimeError, "empty title"):
            title.clean_generated_title("  \n ")


class TitleAgentTests(unittest.IsolatedAsyncioTestCase):
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
                    source_title="Urban Mobility Lecture",
                    summary={"thesis": "Streets can be redesigned around people."},
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
    def test_title_prompt_is_editable_in_admin(self):
        spec = prompts_registry.get_spec("title")

        self.assertIsNotNone(spec)
        self.assertEqual(spec.file, "title.txt")
        self.assertIn("independent", spec.description.lower())


if __name__ == "__main__":
    unittest.main()

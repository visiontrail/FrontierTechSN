import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from backend import config, prompts_registry
from backend.models import TaskConfig, TaskResponse
from backend.pipeline.opencli import OpenCLIResult
from backend.pipeline import orchestrator, thumbnail


PNG_FIXTURE = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR"
    b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"
    b"\x00\x00\x00\x00"
)


class ThumbnailPromptTests(unittest.IsolatedAsyncioTestCase):
    async def test_final_prompt_is_derived_from_title_and_audio_script(self):
        captured = {}

        async def fake_chat(system_prompt, user_content, *args, **kwargs):
            captured["system_prompt"] = system_prompt
            captured["source"] = json.loads(user_content)
            return "```text\nMassive glowing book, impossible scale, landscape 16:9.\n```"

        with (
            patch.object(thumbnail, "_resolve_provider", AsyncMock(return_value=("url", "model", "key"))),
            patch.object(thumbnail, "_chat", fake_chat),
        ):
            prompt = await thumbnail.generate_thumbnail_prompt(
                title="The Library That Remembers Everything",
                script="Imagine every highlight becoming a doorway back into the book.",
            )

        self.assertIn("Subject Dominance", captured["system_prompt"])
        self.assertEqual(captured["source"]["video_title"], "The Library That Remembers Everything")
        self.assertIn("every highlight", captured["source"]["audio_script"])
        self.assertFalse(prompt.startswith("```"))
        self.assertIn("landscape 16:9", prompt)

    def test_formula_is_registered_for_live_admin_editing(self):
        spec = prompts_registry.get_spec("thumbnail")
        self.assertIsNotNone(spec)
        self.assertEqual(spec.file, "thumbnail.txt")
        self.assertIn("Mianyang Style", prompts_registry.read_content(spec))


class ThumbnailGenerationTests(unittest.IsolatedAsyncioTestCase):
    async def test_opencli_download_is_saved_with_prompt_and_manifest_without_tts(self):
        with tempfile.TemporaryDirectory() as directory:
            task_dir = Path(directory)
            script_path = task_dir / "script.txt"
            script_path.write_text("A finished narration script about an impossible reading machine.")
            captured_args = []

            async def fake_opencli(args, **kwargs):
                captured_args.extend(args)
                image_dir = Path(args[args.index("--op") + 1])
                image_path = image_dir / "chatgpt_123.png"
                image_path.write_bytes(PNG_FIXTURE)
                return OpenCLIResult(
                    args=tuple(args),
                    returncode=0,
                    stdout=json.dumps(
                        [
                            {
                                "Status": "saved",
                                "File": f"📁 {image_path}",
                                "Link": "🔗 https://chatgpt.com/c/thumbnail-test",
                            }
                        ]
                    ),
                    stderr="",
                )

            with (
                patch.object(
                    thumbnail,
                    "generate_thumbnail_prompt",
                    AsyncMock(return_value="One dramatic story, extreme contrast, landscape 16:9."),
                ),
                patch.object(thumbnail, "run_opencli", fake_opencli),
                patch.object(config, "THUMBNAIL_CHATGPT_TIMEOUT", 90),
            ):
                artifact = await thumbnail.generate_thumbnail(
                    task_id="thumbnail-unit",
                    task_dir=task_dir,
                    title="Impossible Reading Machine",
                    script_path=script_path,
                )

            self.assertEqual(Path(artifact.image_path).name, "chatgpt_123.png")
            self.assertEqual(Path(artifact.prompt_path).read_text().strip(), "One dramatic story, extreme contrast, landscape 16:9.")
            manifest = json.loads(Path(artifact.manifest_path).read_text())
            self.assertEqual(manifest["status"], "ready")
            self.assertEqual(manifest["conversation_url"], "https://chatgpt.com/c/thumbnail-test")
            self.assertEqual(captured_args[:2], ["chatgpt", "image"])
            self.assertNotIn("tts", " ".join(captured_args).lower())


class ThumbnailPipelineOrderTests(unittest.IsolatedAsyncioTestCase):
    async def test_title_and_thumbnail_run_after_script_and_before_tts(self):
        with tempfile.TemporaryDirectory() as directory:
            events = []
            task = TaskResponse(
                id="pipeline-thumbnail-order",
                created_at="2026-08-04T00:00:00+00:00",
                updated_at="2026-08-04T00:00:00+00:00",
                source_type="youtube",
                source_url="https://youtube.example/video",
                status="queued",
                config=TaskConfig(
                    thumbnail_enabled=True,
                    footage_enabled=False,
                    auto_render=False,
                ),
            )
            content = type(
                "Content",
                (),
                {"title": "Order Test", "text": "source words", "metadata": {}},
            )()

            async def fake_script(*args, **kwargs):
                events.append("script")
                self.assertEqual(kwargs["closing_remarks"], task.config.closing_remarks)
                return "Final audio script"

            async def fake_thumbnail(*args, **kwargs):
                events.append("thumbnail")
                self.assertEqual(kwargs["title"], "A Better Video Title")

            async def fake_title(*args, **kwargs):
                events.append("title")
                return SimpleNamespace(title="A Better Video Title")

            async def fake_tts(*args, **kwargs):
                events.append("tts")
                return str(Path(directory) / "audio.wav")

            with (
                patch.object(config, "OUTPUTS_DIR", Path(directory)),
                patch.object(orchestrator, "update_task", AsyncMock()),
                patch.object(orchestrator, "extract_youtube", AsyncMock(return_value=content)),
                patch.object(orchestrator, "summarize", AsyncMock(return_value={"title": "Order Test"})),
                patch.object(orchestrator, "generate_script", fake_script),
                patch.object(orchestrator, "_generate_task_title", fake_title),
                patch.object(orchestrator, "_generate_task_thumbnail", fake_thumbnail),
                patch.object(orchestrator, "generate_tts", fake_tts),
            ):
                await orchestrator.run_pipeline(task)

            self.assertEqual(events, ["script", "title", "thumbnail", "tts"])


if __name__ == "__main__":
    unittest.main()

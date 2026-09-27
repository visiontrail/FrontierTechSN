import asyncio
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from PIL import Image

from backend import config, prompts_registry
from backend.models import TaskConfig, TaskResponse
from backend.pipeline.opencli import OpenCLIError, OpenCLIResult
from backend.pipeline import orchestrator, thumbnail


_png = io.BytesIO()
Image.new("RGB", (16, 9), "navy").save(_png, format="PNG")
PNG_FIXTURE = _png.getvalue()


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
        content = prompts_registry.read_content(spec)
        self.assertIn("Mianyang Style", content)
        self.assertIn("BRAND / PRODUCT FIDELITY", content)
        self.assertIn("Cursor the AI", content)
        self.assertIn("coding product is not a mouse-pointer arrow", content)
        self.assertIn("Story-essential official", content)


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
            self.assertEqual(captured_args[captured_args.index("--site-session") + 1], "ephemeral")
            self.assertEqual(captured_args[captured_args.index("--keep-tab") + 1], "false")
            self.assertNotIn("tts", " ".join(captured_args).lower())


@pytest.mark.parametrize("output", ["malformed", "timeout", "export_failure"])
def test_completed_download_survives_missing_json_or_command_failure(tmp_path, output):
    script = tmp_path / "script.txt"
    script.write_text("Final narration")

    async def command(args, **kwargs):
        image_dir = Path(args[args.index("--op") + 1])
        (image_dir / "finished.png").write_bytes(PNG_FIXTURE)
        result = OpenCLIResult(
            tuple(args), 0 if output == "malformed" else 1, "not JSON",
            '[chatgpt/image] {"stage":"download","conversation_url":"https://chatgpt.com/c/owned"}',
        )
        if output != "malformed":
            raise OpenCLIError(output, result=result)
        return result

    runner = AsyncMock(side_effect=command)
    with (
        patch.object(thumbnail, "generate_thumbnail_prompt", AsyncMock(return_value="Draw a cover")),
        patch.object(thumbnail, "run_opencli", runner),
    ):
        artifact = asyncio.run(thumbnail.generate_thumbnail(
            task_id="recovery", task_dir=tmp_path, title="Title", script_path=script,
        ))
    assert Path(artifact.image_path).read_bytes() == PNG_FIXTURE
    assert artifact.conversation_url == "https://chatgpt.com/c/owned"
    manifest = json.loads(Path(artifact.manifest_path).read_text())
    assert manifest["status"] == "ready"
    assert manifest["last_stage"] == "download"
    assert bool(manifest.get("recovered_download")) == (output != "malformed")
    runner.assert_awaited_once()  # Recovery must never send the paid prompt again.


@pytest.mark.parametrize("invalid", ["missing", "truncated", "old", "outside", "symlink"])
def test_failed_generation_retains_diagnostics_and_rejects_unusable_downloads(tmp_path, invalid):
    script = tmp_path / "script.txt"
    script.write_text("Final narration")
    image_dir = tmp_path / "thumbnail" / "images"
    image_dir.mkdir(parents=True)
    stale = image_dir / "old.png"
    stale.write_bytes(PNG_FIXTURE)
    outside = tmp_path / "outside.png"
    outside.write_bytes(PNG_FIXTURE)

    async def command(args, **kwargs):
        file = image_dir / "new.png"
        if invalid == "truncated":
            file.write_bytes(PNG_FIXTURE[:40])
        elif invalid == "old":
            file = stale
        elif invalid == "outside":
            file = outside
        elif invalid == "symlink":
            file.symlink_to(outside)
        result = OpenCLIResult(
            tuple(args), 1, json.dumps([{"file": str(file)}]),
            '[chatgpt/image] {"stage":"waiting","conversation_url":"https://chatgpt.com/c/owned"}',
        )
        raise OpenCLIError("generation timed out", result=result)

    runner = AsyncMock(side_effect=command)
    with (
        patch.object(thumbnail, "generate_thumbnail_prompt", AsyncMock(return_value="Draw a cover")),
        patch.object(thumbnail, "run_opencli", runner),
        pytest.raises(OpenCLIError, match="generation timed out"),
    ):
        asyncio.run(thumbnail.generate_thumbnail(
            task_id="failed", task_dir=tmp_path, title="Title", script_path=script,
        ))
    manifest = json.loads((tmp_path / "thumbnail" / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["image_path"] == ""
    assert manifest["conversation_url"] == "https://chatgpt.com/c/owned"
    assert manifest["last_stage"] == "waiting"
    assert manifest["command_returncode"] == 1
    assert stale.read_bytes() == PNG_FIXTURE
    runner.assert_awaited_once()


class ThumbnailPipelineOrderTests(unittest.IsolatedAsyncioTestCase):
    async def test_frozen_script_media_overlaps_tts_and_drains_before_delivery(self):
        with tempfile.TemporaryDirectory() as directory:
            events = []
            audio_ready = asyncio.Event()
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
                await asyncio.wait_for(audio_ready.wait(), 1)
                events.append("thumbnail-finished")

            async def fake_title(*args, **kwargs):
                events.append("title")
                return SimpleNamespace(title="A Better Video Title")

            async def fake_tts(*args, **kwargs):
                events.append("tts")
                audio_ready.set()
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

            self.assertEqual(events, ["script", "title", "thumbnail", "tts", "thumbnail-finished"])


if __name__ == "__main__":
    unittest.main()

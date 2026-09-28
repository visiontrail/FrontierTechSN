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
            self.assertEqual(
                captured_args[captured_args.index("--state-file") + 1],
                manifest["request_state_path"],
            )
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


def test_retry_reuses_prompt_and_checkpoint_then_completed_call_is_idempotent(tmp_path):
    script = tmp_path / "script.txt"
    script.write_text("Final narration")
    checkpoints = []

    async def command(args, **kwargs):
        checkpoint = Path(args[args.index("--state-file") + 1])
        checkpoints.append(checkpoint)
        assert args[2] == "Original cover"
        if len(checkpoints) == 1:
            checkpoint.write_text(json.dumps({
                "prompt": "Generate an image of: Original cover",
                "stage": "waiting", "conversationUrl": "https://chatgpt.com/c/owned1234",
                "beforeUrls": [],
            }))
            raise OpenCLIError("response timed out")
        assert json.loads(checkpoint.read_text())["conversationUrl"].endswith("owned1234")
        image = Path(args[args.index("--op") + 1]) / "recovered.png"
        image.write_bytes(PNG_FIXTURE)
        return OpenCLIResult(tuple(args), 0, json.dumps([{"file": str(image)}]), "")

    director = AsyncMock(return_value="Original cover")
    runner = AsyncMock(side_effect=command)
    arguments = dict(task_id="retry", task_dir=tmp_path, title="Title", script_path=script)
    with (
        patch.object(thumbnail, "generate_thumbnail_prompt", director),
        patch.object(thumbnail, "run_opencli", runner),
    ):
        with pytest.raises(OpenCLIError, match="response timed out"):
            asyncio.run(thumbnail.generate_thumbnail(**arguments))
        artifact = asyncio.run(thumbnail.generate_thumbnail(**arguments))
        again = asyncio.run(thumbnail.generate_thumbnail(**arguments))
    assert artifact == again
    assert checkpoints[0] == checkpoints[1]
    director.assert_awaited_once()
    assert runner.await_count == 2


@pytest.mark.parametrize("changed", ["script", "title"])
def test_changed_source_uses_a_new_request_and_preserves_the_old_checkpoint(tmp_path, changed):
    script = tmp_path / "script.txt"
    script.write_text("Original narration")
    checkpoints = []

    async def command(args, **kwargs):
        checkpoint = Path(args[args.index("--state-file") + 1])
        assert not checkpoint.exists()
        checkpoint.write_text(args[2])
        checkpoints.append(checkpoint)
        image = Path(args[args.index("--op") + 1]) / f"cover{len(checkpoints)}.png"
        image.write_bytes(PNG_FIXTURE)
        return OpenCLIResult(tuple(args), 0, json.dumps([{"file": str(image)}]), "")

    director = AsyncMock(side_effect=["Original cover", "Revised cover"])
    with (
        patch.object(thumbnail, "generate_thumbnail_prompt", director),
        patch.object(thumbnail, "run_opencli", command),
    ):
        first = asyncio.run(thumbnail.generate_thumbnail(
            task_id="changed", task_dir=tmp_path, title="Original title", script_path=script,
        ))
        if changed == "script":
            script.write_text("Revised narration")
        second = asyncio.run(thumbnail.generate_thumbnail(
            task_id="changed", task_dir=tmp_path,
            title="Revised title" if changed == "title" else "Original title", script_path=script,
        ))
    assert first.image_path != second.image_path
    assert checkpoints[0] != checkpoints[1]
    assert checkpoints[0].read_text() == "Original cover"
    assert director.await_count == 2


def test_legacy_failure_requires_explicit_recovery_and_keeps_original_prompt(tmp_path):
    script = tmp_path / "script.txt"
    script.write_text("Final narration")
    directory = tmp_path / "thumbnail"
    directory.mkdir()
    (directory / "prompt.txt").write_text("Original legacy cover\n")
    original_manifest = {
        "task_id": "legacy", "status": "failed", "source_script": str(script),
        "conversation_url": "https://chatgpt.com/c/local-chatgpt%3Aunsaved",
    }
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(original_manifest))
    arguments = dict(task_id="legacy", task_dir=tmp_path, title="Title", script_path=script)
    director = AsyncMock()

    async def command(args, **kwargs):
        assert args[2] == "Original legacy cover"
        state = json.loads(Path(args[args.index("--state-file") + 1]).read_text())
        assert state == {
            "prompt": "Generate an image of: Original legacy cover",
            "stage": "recovery_requested",
            "conversationUrl": "https://chatgpt.com/c/verified1234",
            "beforeUrls": [],
        }
        image = Path(args[args.index("--op") + 1]) / "legacy.png"
        image.write_bytes(PNG_FIXTURE)
        return OpenCLIResult(tuple(args), 0, json.dumps([{"file": str(image)}]), "")

    runner = AsyncMock(side_effect=command)
    with (
        patch.object(thumbnail, "generate_thumbnail_prompt", director),
        patch.object(thumbnail, "run_opencli", runner),
    ):
        with pytest.raises(RuntimeError, match="verified resume_conversation_url"):
            asyncio.run(thumbnail.generate_thumbnail(**arguments))
        assert json.loads(manifest_path.read_text()) == original_manifest
        runner.assert_not_awaited()
        artifact = asyncio.run(thumbnail.generate_thumbnail(
            **arguments, resume_conversation_url="https://chatgpt.com/c/verified1234",
        ))
    assert artifact.conversation_url == "https://chatgpt.com/c/verified1234"
    recovered_manifest = json.loads(manifest_path.read_text())
    assert json.loads(Path(recovered_manifest["previous_manifest_path"]).read_text()) == original_manifest
    assert recovered_manifest["recovery_origin"]["previous_request_sha256"] is None
    director.assert_not_awaited()
    runner.assert_awaited_once()


def test_intentional_legacy_regeneration_creates_new_prompt_and_preserves_history(tmp_path):
    script = tmp_path / "script.txt"
    script.write_text("Edited final narration")
    directory = tmp_path / "thumbnail"
    directory.mkdir()
    (directory / "prompt.txt").write_text("Original legacy cover\n")
    original_manifest = {
        "task_id": "legacy-edit", "status": "failed", "source_script": str(script),
        "conversation_url": "https://chatgpt.com/c/original1234",
    }
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(original_manifest))

    async def command(args, **kwargs):
        assert args[2] == "New cover from edited narration"
        state_path = Path(args[args.index("--state-file") + 1])
        assert not state_path.exists()  # OpenCLI must create a fresh request checkpoint.
        state_path.write_text(json.dumps({
            "prompt": "Generate an image of: New cover from edited narration",
            "stage": "saved", "conversationUrl": "https://chatgpt.com/c/newcover1234",
        }))
        image = Path(args[args.index("--op") + 1]) / "new-cover.png"
        image.write_bytes(PNG_FIXTURE)
        return OpenCLIResult(tuple(args), 0, json.dumps([
            {"file": str(image), "link": "https://chatgpt.com/c/newcover1234"},
        ]), "")

    director = AsyncMock(return_value="New cover from edited narration")
    with (
        patch.object(thumbnail, "generate_thumbnail_prompt", director),
        patch.object(thumbnail, "run_opencli", command),
    ):
        artifact = asyncio.run(thumbnail.generate_thumbnail(
            task_id="legacy-edit", task_dir=tmp_path, title="Edited title", script_path=script,
            allow_legacy_regeneration=True,
        ))
    manifest = json.loads(manifest_path.read_text())
    assert json.loads(Path(manifest["previous_manifest_path"]).read_text()) == original_manifest
    assert Path(manifest["previous_prompt_path"]).read_text() == "Original legacy cover\n"
    assert Path(artifact.prompt_path).read_text().strip() == "New cover from edited narration"
    assert artifact.conversation_url == "https://chatgpt.com/c/newcover1234"
    assert director.await_args.kwargs["script"] == "Edited final narration"
    director.assert_awaited_once()


@pytest.mark.parametrize("bad_url", ["https://example.com/c/verified1234", "https://chatgpt.com/new", "https://chatgpt.com/c/local-chatgpt%3A123"])
def test_recovery_rejects_noncanonical_urls_before_modifying_evidence(tmp_path, bad_url):
    script = tmp_path / "script.txt"
    script.write_text("Final narration")
    directory = tmp_path / "thumbnail"
    directory.mkdir()
    (directory / "prompt.txt").write_text("Original cover")
    (directory / "manifest.json").write_text('{"status":"failed"}')
    with pytest.raises(ValueError, match="canonical"):
        asyncio.run(thumbnail.generate_thumbnail(
            task_id="bad-url", task_dir=tmp_path, title="Title", script_path=script,
            resume_conversation_url=bad_url,
        ))
    assert (directory / "manifest.json").read_text() == '{"status":"failed"}'


def test_concurrent_generation_is_rejected_without_submitting_a_second_request(tmp_path):
    script = tmp_path / "script.txt"
    script.write_text("Final narration")
    arguments = dict(task_id="busy", task_dir=tmp_path, title="Title", script_path=script)

    async def exercise():
        entered = asyncio.Event()
        release = asyncio.Event()

        async def command(args, **kwargs):
            entered.set()
            await release.wait()
            raise OpenCLIError("finished original request")

        runner = AsyncMock(side_effect=command)
        with (
            patch.object(thumbnail, "generate_thumbnail_prompt", AsyncMock(return_value="Cover")),
            patch.object(thumbnail, "run_opencli", runner),
        ):
            pending = asyncio.create_task(thumbnail.generate_thumbnail(**arguments))
            await asyncio.wait_for(entered.wait(), 1)
            with pytest.raises(RuntimeError, match="already running"):
                await thumbnail.generate_thumbnail(**arguments)
            release.set()
            with pytest.raises(OpenCLIError, match="finished original request"):
                await pending
        runner.assert_awaited_once()

    asyncio.run(exercise())


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
                self.assertFalse(kwargs.get("allow_legacy_regeneration", False))
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

    async def test_regenerate_explicitly_allows_legacy_thumbnail_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            task_dir = Path(directory)
            script_path = task_dir / "script.txt"
            script_path.write_text("Edited narration")
            task = TaskResponse(
                id="regenerate-legacy-thumbnail",
                created_at="2026-08-04T00:00:00+00:00",
                updated_at="2026-08-04T00:00:00+00:00",
                source_type="youtube", source_url="https://youtube.example/video",
                status="complete", script_path=str(script_path), output_dir=str(task_dir),
                config=TaskConfig(thumbnail_enabled=True, footage_enabled=False, auto_render=False),
            )
            generate = AsyncMock(return_value=thumbnail.ThumbnailArtifact(
                str(task_dir / "prompt.txt"), str(task_dir / "cover.png"), str(task_dir / "manifest.json"),
            ))
            with (
                patch.object(orchestrator, "update_task", AsyncMock()),
                patch.object(orchestrator, "_generate_task_title", AsyncMock(return_value=SimpleNamespace(title="Edited title"))),
                patch.object(orchestrator, "generate_thumbnail", generate),
                patch.object(orchestrator, "generate_tts", AsyncMock(return_value=str(task_dir / "audio.wav"))),
                patch.object(orchestrator, "_after_audio", AsyncMock()),
            ):
                await orchestrator.run_regenerate(task)
            generate.assert_awaited_once()
            self.assertTrue(generate.await_args.kwargs["allow_legacy_regeneration"])
            self.assertEqual(generate.await_args.kwargs["title"], "Edited title")


if __name__ == "__main__":
    unittest.main()

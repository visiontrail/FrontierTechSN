import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from backend import config
from backend.pipeline import voice_previews


class VoicePreviewTests(unittest.IsolatedAsyncioTestCase):
    async def test_generates_remote_preview_once_and_reuses_cache(self):
        async def fake_generate(script_path, output_dir, voice, language, **kwargs):
            self.assertEqual(voice, "tara")
            self.assertEqual(language, "en")
            self.assertEqual(kwargs["max_tokens"], voice_previews.PREVIEW_MAX_TOKENS)
            self.assertEqual(
                Path(script_path).read_text(encoding="utf-8"),
                voice_previews.PREVIEW_TEXT,
            )
            generated = Path(output_dir) / "preview_generated.wav"
            generated.parent.mkdir(parents=True)
            generated.write_bytes(b"RIFF" + b"0" * 64)
            return str(generated)

        with tempfile.TemporaryDirectory() as cache:
            with (
                patch.object(config, "VOICE_PREVIEW_CACHE_DIR", Path(cache)),
                patch.object(
                    voice_previews, "_generate_orpheus", side_effect=fake_generate
                ) as generate,
            ):
                first = await voice_previews.ensure_voice_preview("tara", "orpheus-en")
                second = await voice_previews.ensure_voice_preview("tara", "orpheus-en")

        self.assertEqual(first, second)
        self.assertEqual(generate.await_count, 1)

    async def test_generates_pocket_preview_on_demand(self):
        async def fake_generate(script_path, output_dir, voice, language, **kwargs):
            self.assertEqual((voice, language), ("alba", "en"))
            self.assertNotIn("max_tokens", kwargs)
            generated = Path(output_dir) / "preview_generated.wav"
            generated.parent.mkdir(parents=True)
            generated.write_bytes(b"RIFF" + b"0" * 64)
            return str(generated)

        with tempfile.TemporaryDirectory() as cache:
            with (
                patch.object(config, "VOICE_PREVIEW_CACHE_DIR", Path(cache)),
                patch.object(
                    voice_previews, "_generate_pocket_tts", side_effect=fake_generate
                ) as generate,
            ):
                preview = await voice_previews.ensure_voice_preview(
                    "alba", "pocket-tts-en"
                )

        self.assertEqual(preview.name, "alba.wav")
        self.assertEqual(generate.await_count, 1)

    async def test_preload_skips_network_without_api_key(self):
        ensure = AsyncMock()
        with (
            patch.object(config, "ORPHEUS_TTS_API_KEY", ""),
            patch.object(voice_previews, "ensure_voice_preview", ensure),
        ):
            await voice_previews.preload_remote_voice_previews()
        ensure.assert_not_awaited()

    async def test_chatterbox_preview_is_cached_and_uses_preview_validation(self):
        async def generate(script_path, output_dir, voice, language, **kwargs):
            self.assertEqual((voice, language), ("jane", "en"))
            self.assertIs(kwargs["verify_text"], False)
            target = Path(output_dir) / "preview.wav"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"RIFF" + b"0" * 64)
            return str(target)

        with (
            tempfile.TemporaryDirectory() as cache,
            patch.object(config, "VOICE_PREVIEW_CACHE_DIR", Path(cache)),
            patch("backend.pipeline.chatterbox._generate_chatterbox", side_effect=generate) as request,
        ):
            first = await voice_previews.ensure_voice_preview("jane", "chatterbox-multilingual")
            second = await voice_previews.ensure_voice_preview("jane", "chatterbox-multilingual")
            self.assertEqual(first, second)
            self.assertEqual(request.await_count, 1)
            self.assertEqual(first.parent, config.voice_preview_cache_dir("chatterbox-multilingual"))

    async def test_chatterbox_preview_resumes_job_after_timeout_or_cancellation(self):
        for failure in (TimeoutError("poll timed out"), asyncio.CancelledError()):
            with self.subTest(failure=type(failure).__name__):
                directories = []

                async def generate(script_path, output_dir, voice, language, **kwargs):
                    directory = Path(output_dir)
                    directories.append(directory)
                    directory.mkdir(parents=True, exist_ok=True)
                    checkpoint = directory / "tts_input_part_001.request-0.json"
                    if len(directories) == 1:
                        checkpoint.write_text(json.dumps({"idempotency_key": "original-key", "job_id": "original-job"}))
                        raise failure
                    self.assertEqual(json.loads(checkpoint.read_text()), {
                        "idempotency_key": "original-key", "job_id": "original-job",
                    })
                    target = directory / "preview.wav"
                    target.write_bytes(b"RIFF" + b"0" * 64)
                    return str(target)

                with (
                    tempfile.TemporaryDirectory() as cache,
                    patch.object(config, "VOICE_PREVIEW_CACHE_DIR", Path(cache)),
                    patch("backend.pipeline.chatterbox._generate_chatterbox", side_effect=generate) as request,
                ):
                    with self.assertRaises(type(failure)):
                        await voice_previews.ensure_voice_preview("jane", "chatterbox-multilingual")
                    self.assertIsNone(config.voice_sample_path("jane", "chatterbox-multilingual"))
                    self.assertTrue((directories[0] / "tts_input_part_001.request-0.json").is_file())
                    completed = await voice_previews.ensure_voice_preview("jane", "chatterbox-multilingual")
                    self.assertEqual(directories[0], directories[1])
                    self.assertEqual(request.await_count, 2)
                    self.assertEqual(config.voice_sample_path("jane", "chatterbox-multilingual"), completed)

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

    async def test_preload_skips_network_without_api_key(self):
        ensure = AsyncMock()
        with (
            patch.object(config, "ORPHEUS_TTS_API_KEY", ""),
            patch.object(voice_previews, "ensure_voice_preview", ensure),
        ):
            await voice_previews.preload_remote_voice_previews()
        ensure.assert_not_awaited()

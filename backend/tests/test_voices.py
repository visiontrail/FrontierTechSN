import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend import config


class VoiceSampleTests(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.sample_dir = Path(self._temp.name)
        (self.sample_dir / "en-Carter_man.wav").write_bytes(b"wav")
        (self.sample_dir / "en-Mary_woman_bgm.wav").write_bytes(b"wav")
        self.addCleanup(self._temp.cleanup)

    def _patch_dir(self):
        return patch.object(config, "VOICE_SAMPLE_DIR", self.sample_dir)

    def test_finds_sample_regardless_of_lang_and_suffix(self):
        with self._patch_dir():
            self.assertEqual(
                config.voice_sample_path("Carter"),
                self.sample_dir / "en-Carter_man.wav",
            )
            # The trailing "_bgm" variant must still resolve.
            self.assertEqual(
                config.voice_sample_path("Mary", "vibevoice-1.5b"),
                self.sample_dir / "en-Mary_woman_bgm.wav",
            )

    def test_missing_sample_returns_none(self):
        with self._patch_dir():
            self.assertIsNone(config.voice_sample_path("Alice"))

    def test_resolves_model_voice_aliases(self):
        # The 0.5B model substitutes Alice -> Emma, so a preview must look for
        # the substitute's sample, not Alice's.
        self.assertEqual(config.resolve_voice("Alice", "vibevoice-0.5b"), "Emma")
        self.assertEqual(config.resolve_voice("Alice", "vibevoice-1.5b"), "Alice")
        with self._patch_dir():
            (self.sample_dir / "en-Emma_woman.wav").write_bytes(b"wav")
            self.assertEqual(
                config.voice_sample_path("Alice", "vibevoice-0.5b"),
                self.sample_dir / "en-Emma_woman.wav",
            )

    def test_unknown_model_falls_back_to_no_substitution(self):
        self.assertEqual(config.resolve_voice("Alice", "nope"), "Alice")

    def test_orpheus_has_model_specific_voices_and_uses_generated_cache(self):
        self.assertEqual(
            list(config.voices_for_model("orpheus-en")),
            ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"],
        )
        with tempfile.TemporaryDirectory() as cache:
            with patch.object(config, "VOICE_PREVIEW_CACHE_DIR", Path(cache)):
                self.assertIsNone(config.voice_sample_path("tara", "orpheus-en"))
                sample = Path(cache) / "orpheus-en" / "tara.wav"
                sample.parent.mkdir()
                sample.write_bytes(b"RIFF" + b"0" * 64)
                self.assertEqual(
                    config.voice_sample_path("tara", "orpheus-en"), sample
                )
                self.assertTrue(config.voice_preview_supported("tara", "orpheus-en"))


class VoiceRouteTests(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        from backend.routers import voices

        app = FastAPI()
        app.include_router(voices.router)
        self.client = TestClient(app)

    def test_lists_every_available_voice(self):
        body = self.client.get("/api/voices").json()
        self.assertEqual(
            [v["name"] for v in body], list(config.AVAILABLE_VOICES)
        )

    def test_lists_orpheus_voices_only_for_orpheus(self):
        with tempfile.TemporaryDirectory() as cache:
            with patch.object(config, "VOICE_PREVIEW_CACHE_DIR", Path(cache)):
                body = self.client.get("/api/voices?tts_model=orpheus-en").json()
        self.assertEqual([v["name"] for v in body], list(config.ORPHEUS_EN_VOICES))
        self.assertTrue(all(v["preview_available"] for v in body))

    def test_lists_tts_model_capabilities(self):
        body = self.client.get("/api/voices/models").json()
        orpheus = next(model for model in body if model["id"] == "orpheus-en")
        self.assertEqual(orpheus["provider"], "Orpheus")
        self.assertTrue(orpheus["single_speaker"])

    def test_rejects_unknown_voice_before_touching_the_filesystem(self):
        res = self.client.get("/api/voices/Bogus*/preview")
        self.assertEqual(res.status_code, 404)
        self.assertIn("Unknown voice", res.json()["detail"])

    def test_path_traversal_never_reaches_a_file(self):
        res = self.client.get("/api/voices/..%2F..%2Fetc%2Fpasswd/preview")
        self.assertEqual(res.status_code, 404)

    def test_missing_sample_is_a_404(self):
        with tempfile.TemporaryDirectory() as empty:
            with patch.object(config, "VOICE_SAMPLE_DIR", Path(empty)):
                res = self.client.get("/api/voices/Carter/preview")
        self.assertEqual(res.status_code, 404)
        self.assertIn("No preview sample", res.json()["detail"])

    def test_generates_and_serves_missing_orpheus_preview(self):
        async def fake_ensure(voice, model_id):
            self.assertEqual((voice, model_id), ("tara", "orpheus-en"))
            sample.write_bytes(b"RIFF" + b"0" * 64)
            return sample

        with tempfile.TemporaryDirectory() as cache:
            sample = Path(cache) / "tara.wav"
            with (
                patch.object(config, "VOICE_PREVIEW_CACHE_DIR", Path(cache)),
                patch("backend.routers.voices.ensure_voice_preview", fake_ensure),
            ):
                res = self.client.get(
                    "/api/voices/tara/preview?tts_model=orpheus-en"
                )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.content, b"RIFF" + b"0" * 64)


if __name__ == "__main__":
    unittest.main()

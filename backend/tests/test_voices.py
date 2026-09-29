import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from backend import config
from backend.pipeline import chatterbox, tts, voice_previews


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

    def test_pocket_has_upstream_voice_catalog_and_on_demand_previews(self):
        self.assertIn("alba", config.voices_for_model("pocket-tts-en"))
        self.assertIn("marius", config.voices_for_model("pocket-tts-en"))
        with tempfile.TemporaryDirectory() as cache:
            with patch.object(config, "VOICE_PREVIEW_CACHE_DIR", Path(cache)):
                self.assertIsNone(config.voice_sample_path("alba", "pocket-tts-en"))
                self.assertTrue(config.voice_preview_supported("alba", "pocket-tts-en"))

    def test_chatterbox_preview_identity_tracks_deployment_without_changing_defaults(self):
        self.assertEqual(list(config.voices_for_model("chatterbox-multilingual")), ["space-en-f1", "jane", "default"])
        model = config.TTS_MODELS["chatterbox-multilingual"]
        self.assertEqual(model["language"], "en")
        self.assertFalse(model["preload_previews"])
        self.assertTrue(model["acoustic_integrity"])
        with tempfile.TemporaryDirectory() as cache:
            with patch.object(config, "VOICE_PREVIEW_CACHE_DIR", Path(cache)):
                sample = config.voice_preview_cache_dir("chatterbox-multilingual") / "jane.wav"
                sample.parent.mkdir(parents=True)
                sample.write_bytes(b"RIFF" + b"0" * 64)
                self.assertEqual(config.voice_sample_path("jane", "chatterbox-multilingual"), sample)
                for key in (
                    "CHATTERBOX_TTS_URL", "CHATTERBOX_TTS_SOURCE_REVISION",
                    "CHATTERBOX_TTS_MODEL_REVISION", "CHATTERBOX_TTS_DEPLOYMENT_FINGERPRINT",
                ):
                    with self.subTest(key=key), patch.object(config, key, "changed"):
                        self.assertIsNone(config.voice_sample_path("jane", "chatterbox-multilingual"))
                self.assertEqual(config.voice_sample_path("jane", "chatterbox-multilingual"), sample)

    def test_chatterbox_preview_cache_tracks_synthesis_and_verification_policy(self):
        original = config.voice_preview_cache_dir("chatterbox-multilingual")
        changes = [
            (config, "TTS_RANDOM_SEED", config.TTS_RANDOM_SEED + 1),
            (config, "CHATTERBOX_TTS_CHUNK_WORDS", config.CHATTERBOX_TTS_CHUNK_WORDS + 1),
            (config, "CHATTERBOX_TTS_MAX_CHARS", config.CHATTERBOX_TTS_MAX_CHARS - 1),
            (chatterbox, "VERIFIER_VERSION", chatterbox.VERIFIER_VERSION + 1),
            (tts, "POCKET_TTS_INTEGRITY_VERIFIER_VERSION", tts.POCKET_TTS_INTEGRITY_VERIFIER_VERSION + 1),
            (chatterbox, "SYNTHESIS_PARAMETERS", {**chatterbox.SYNTHESIS_PARAMETERS, "temperature": 0.7}),
            (voice_previews, "PREVIEW_TEXT", "A different preview script."),
        ]
        for module, name, value in changes:
            with self.subTest(setting=name), patch.object(module, name, value):
                self.assertNotEqual(config.voice_preview_cache_dir("chatterbox-multilingual"), original)
        self.assertEqual(config.voice_preview_cache_dir("chatterbox-multilingual"), original)

    def test_model_default_does_not_reuse_a_different_providers_voice(self):
        with patch.object(config, "TTS_DEFAULT_VOICE_1", "jane"):
            self.assertEqual(config.default_voice_for_model("chatterbox-multilingual"), "space-en-f1")
            self.assertEqual(config.default_voice_for_model("pocket-tts-en"), "jane")
        with patch.object(config, "TTS_DEFAULT_VOICE_1", "Carter"):
            self.assertEqual(config.default_voice_for_model("pocket-tts-en"), "alba")
            self.assertEqual(config.default_voice_for_model("vibevoice-0.5b"), "Carter")


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

    def test_lists_pocket_voices_only_for_pocket(self):
        with tempfile.TemporaryDirectory() as cache:
            with patch.object(config, "VOICE_PREVIEW_CACHE_DIR", Path(cache)):
                body = self.client.get("/api/voices?tts_model=pocket-tts-en").json()
        self.assertEqual([v["name"] for v in body], list(config.POCKET_TTS_EN_VOICES))
        self.assertTrue(all(v["preview_available"] for v in body))

    def test_lists_tts_model_capabilities(self):
        body = self.client.get("/api/voices/models").json()
        orpheus = next(model for model in body if model["id"] == "orpheus-en")
        self.assertEqual(orpheus["provider"], "Orpheus")
        self.assertTrue(orpheus["single_speaker"])
        pocket = next(model for model in body if model["id"] == "pocket-tts-en")
        self.assertEqual(pocket["provider"], "Kyutai Pocket TTS")
        self.assertTrue(pocket["single_speaker"])
        chatterbox = next(model for model in body if model["id"] == "chatterbox-multilingual")
        self.assertEqual(chatterbox["provider"], "Resemble AI Chatterbox")
        self.assertTrue(chatterbox["single_speaker"])
        self.assertEqual(chatterbox["default_voice"], "space-en-f1")

    def test_lists_chatterbox_reference_voices(self):
        response = self.client.get("/api/voices?tts_model=chatterbox-multilingual")
        body = response.json()
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertEqual([voice["name"] for voice in body], ["space-en-f1", "jane", "default"])
        self.assertEqual(body[0]["label"], "Official Space English")
        self.assertIn("Pocket clone", body[1]["label"])
        self.assertEqual(body[2]["label"], "Model default voice")
        self.assertTrue(all(voice["preview_version"] for voice in body))
        self.assertTrue(all(voice["preview_available"] for voice in body))

    def test_chatterbox_browser_preview_version_tracks_the_disk_cache(self):
        original = self.client.get("/api/voices?tts_model=chatterbox-multilingual").json()[0]
        for key, value in (
            ("CHATTERBOX_TTS_DEPLOYMENT_FINGERPRINT", "new-reference-deployment"),
            ("TTS_RANDOM_SEED", config.TTS_RANDOM_SEED + 1),
        ):
            with self.subTest(key=key), patch.object(config, key, value):
                updated = self.client.get("/api/voices?tts_model=chatterbox-multilingual").json()[0]
                self.assertNotEqual(updated["preview_version"], original["preview_version"])
                self.assertEqual(updated["preview_version"], config.voice_preview_cache_dir("chatterbox-multilingual").name)

    def test_stale_chatterbox_preview_url_rejects_before_generating(self):
        version = config.voice_preview_version("space-en-f1", "chatterbox-multilingual")
        with (
            patch.object(config, "CHATTERBOX_TTS_DEPLOYMENT_FINGERPRINT", "different-deployment"),
            patch("backend.routers.voices.ensure_voice_preview", new_callable=AsyncMock) as generate,
        ):
            response = self.client.get(f"/api/voices/space-en-f1/preview?tts_model=chatterbox-multilingual&v={version}")
        self.assertEqual(response.status_code, 409)
        generate.assert_not_awaited()

    def test_chatterbox_versioned_url_serves_its_voice_and_legacy_url_is_uncached(self):
        with tempfile.TemporaryDirectory() as cache, patch.object(config, "VOICE_PREVIEW_CACHE_DIR", Path(cache)):
            directory = config.voice_preview_cache_dir("chatterbox-multilingual")
            directory.mkdir(parents=True)
            for index, voice in enumerate(config.CHATTERBOX_TTS_EN_VOICES):
                sample = b"RIFF" + bytes([index]) * 64
                (directory / f"{voice}.wav").write_bytes(sample)
                version = config.voice_preview_version(voice, "chatterbox-multilingual")
                url = f"/api/voices/{voice}/preview?tts_model=chatterbox-multilingual"
                with self.subTest(voice=voice):
                    response = self.client.get(f"{url}&v={version}")
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.content, sample)
                    self.assertEqual(response.headers["cache-control"], "private, max-age=86400, immutable")
                    self.assertEqual(self.client.get(url).headers["cache-control"], "no-store")

    def test_chatterbox_contract_change_during_generation_does_not_cache_wrong_identity(self):
        with tempfile.TemporaryDirectory() as cache, patch.object(config, "VOICE_PREVIEW_CACHE_DIR", Path(cache)):
            sample = Path(cache) / "sample.wav"
            sample.write_bytes(b"RIFF" + b"0" * 64)
            version = config.voice_preview_version("space-en-f1", "chatterbox-multilingual")
            async def generate(*args):
                config.CHATTERBOX_TTS_DEPLOYMENT_FINGERPRINT = "changed-during-request"
                return sample
            with (
                patch.object(config, "CHATTERBOX_TTS_DEPLOYMENT_FINGERPRINT", config.CHATTERBOX_TTS_DEPLOYMENT_FINGERPRINT),
                patch("backend.routers.voices.ensure_voice_preview", side_effect=generate),
            ):
                response = self.client.get(f"/api/voices/space-en-f1/preview?tts_model=chatterbox-multilingual&v={version}")
            self.assertEqual(response.status_code, 409)

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

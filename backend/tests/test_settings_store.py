import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend import config, settings_store


class SettingsStoreTests(unittest.TestCase):
    """The Admin console's settings layer: .env seeds, the store overrides."""

    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.root = Path(self._temp.name)
        self.store = self.root / "settings.json"

        patcher = patch.object(settings_store, "store_path", lambda: self.store)
        patcher.start()
        self.addCleanup(patcher.stop)

        # Every test mutates process-wide config, so snapshot and restore it.
        saved = {spec.key: getattr(config, spec.key) for spec in settings_store.SPECS}
        saved_defaults = settings_store.defaults()

        def restore():
            config.apply_values(saved)
            settings_store._DEFAULTS = saved_defaults

        self.addCleanup(restore)
        # Re-seed defaults from the (restored) env values against an empty store.
        settings_store.apply_saved()

    def stored(self) -> dict:
        return json.loads(self.store.read_text(encoding="utf-8"))["values"]

    def test_defaults_apply_when_store_is_empty(self):
        self.assertFalse(self.store.exists())
        self.assertEqual(config.RENDER_FPS, settings_store.defaults()["RENDER_FPS"])

    def test_update_applies_live_and_survives_a_restart(self):
        settings_store.update({"RENDER_FPS": 30, "RENDER_QUALITY": "high"})

        self.assertEqual(config.RENDER_FPS, 30)
        self.assertEqual(config.RENDER_QUALITY, "high")
        self.assertEqual(self.stored()["RENDER_FPS"], 30)

        # Simulate a fresh process: config reverts to .env, then re-applies.
        config.apply_values(settings_store.defaults())
        self.assertEqual(config.RENDER_FPS, settings_store.defaults()["RENDER_FPS"])
        settings_store.apply_saved()
        self.assertEqual(config.RENDER_FPS, 30)

    def test_moving_aiwork_root_relocates_the_tts_contract(self):
        settings_store.update({"AIWORK_ROOT": str(self.root / "aiwork")})

        self.assertEqual(config.AIWORK_ROOT, self.root / "aiwork")
        self.assertEqual(
            config.TTS_MODELS["vibevoice-1.5b"]["project_dir"],
            self.root / "aiwork" / "VibeVoice-1.5B",
        )
        self.assertEqual(
            config.VOICE_SAMPLE_DIR,
            self.root / "aiwork" / "VibeVoice-1.5B" / "demo" / "voices",
        )

    def test_relative_paths_resolve_under_the_project_root(self):
        settings_store.update({"UPLOADS_DIR": "uploads-test"})
        self.assertEqual(config.UPLOADS_DIR, config.PROJECT_ROOT / "uploads-test")
        self.addCleanup(lambda: (config.PROJECT_ROOT / "uploads-test").rmdir())

    def test_bool_accepts_the_env_style_spellings(self):
        settings_store.update({"DIRECTOR_ENABLED": "0"})
        self.assertIs(config.DIRECTOR_ENABLED, False)
        settings_store.update({"DIRECTOR_ENABLED": "yes"})
        self.assertIs(config.DIRECTOR_ENABLED, True)

    def test_value_matching_the_default_drops_the_override(self):
        default_fps = settings_store.defaults()["RENDER_FPS"]
        settings_store.update({"RENDER_FPS": default_fps + 5})
        self.assertIn("RENDER_FPS", self.stored())

        # Setting it back to the default must not freeze a copy of it, so a
        # later .env change still shows through.
        settings_store.update({"RENDER_FPS": default_fps})
        self.assertNotIn("RENDER_FPS", self.stored())
        self.assertEqual(config.RENDER_FPS, default_fps)

    def test_clearing_a_number_or_choice_restores_its_default(self):
        # The Admin form sends "" when a number input is emptied; that reads as
        # "put it back", not as an invalid number.
        settings_store.update({"RENDER_FPS": 24, "RENDER_QUALITY": "high"})
        settings_store.update({"RENDER_FPS": "", "RENDER_QUALITY": ""})

        self.assertEqual(self.stored(), {})
        self.assertEqual(config.RENDER_FPS, settings_store.defaults()["RENDER_FPS"])
        self.assertEqual(config.RENDER_QUALITY, settings_store.defaults()["RENDER_QUALITY"])

    def test_defaults_are_snapshotted_only_once(self):
        settings_store.update({"RENDER_FPS": 24})
        settings_store.apply_saved()
        # A second apply must not promote the saved override into the default.
        self.assertNotEqual(settings_store.defaults()["RENDER_FPS"], 24)

    def test_rejects_unusable_values_without_touching_config(self):
        before = config.RENDER_FPS
        for values, expected in (
            ({"RENDER_FPS": "many"}, "whole number"),
            ({"RENDER_FPS": 0}, "at least"),
            ({"RENDER_QUALITY": "ultra"}, "must be one of"),
            ({"TTS_DEFAULT_VOICE_1": "Nobody"}, "must be one of"),
            ({"NOT_A_SETTING": "x"}, "Unknown setting"),
        ):
            with self.subTest(values=values):
                with self.assertRaises(settings_store.SettingsError) as ctx:
                    settings_store.update(values)
                self.assertIn(expected, str(ctx.exception))
        self.assertEqual(config.RENDER_FPS, before)
        self.assertFalse(self.store.exists())

    def test_rejects_inverted_chatgpt_review_range_without_persisting(self):
        with self.assertRaises(settings_store.SettingsError) as ctx:
            settings_store.update(
                {
                    "DAILY_NEWS_CHATGPT_REVIEW_MIN_LEVEL": "xhigh",
                    "DAILY_NEWS_CHATGPT_REVIEW_MAX_LEVEL": "medium",
                }
            )

        self.assertIn("minimum level cannot exceed maximum level", str(ctx.exception))
        self.assertFalse(self.store.exists())
        self.assertEqual(config.DAILY_NEWS_CHATGPT_REVIEW_MIN_LEVEL, "medium")
        self.assertEqual(config.DAILY_NEWS_CHATGPT_REVIEW_MAX_LEVEL, "xhigh")

    def test_reset_restores_the_default_and_reports_restart_keys(self):
        live_outputs_dir = config.OUTPUTS_DIR
        settings_store.update({"RENDER_FPS": 42, "OUTPUTS_DIR": str(self.root / "out")})
        self.assertEqual(config.OUTPUTS_DIR, live_outputs_dir)
        self.assertEqual(Path(self.stored()["OUTPUTS_DIR"]), self.root / "out")

        restart = settings_store.reset(["RENDER_FPS", "OUTPUTS_DIR"])

        self.assertEqual(config.RENDER_FPS, settings_store.defaults()["RENDER_FPS"])
        self.assertEqual(config.OUTPUTS_DIR, live_outputs_dir)
        self.assertEqual(self.stored(), {})
        # Only the mount-bound path needs the process restarted.
        self.assertEqual(restart, ["OUTPUTS_DIR"])

    def test_output_root_change_migrates_podcast_and_waits_for_restart(self):
        live_root = self.root / "live-output"
        desired_root = self.root / "next-output"
        podcast = live_root / "podcast"
        (podcast / "media").mkdir(parents=True)
        (podcast / "episodes.json").write_text(
            '[{"guid":"one"}]',
            encoding="utf-8",
        )
        (podcast / "feed.xml").write_text("<rss>one</rss>", encoding="utf-8")
        (podcast / "media" / "one.wav").write_bytes(b"episode one")
        config.apply_values({"OUTPUTS_DIR": live_root})

        restart = settings_store.update({"OUTPUTS_DIR": str(desired_root)})

        self.assertEqual(restart, ["OUTPUTS_DIR"])
        self.assertEqual(config.OUTPUTS_DIR, live_root)
        self.assertEqual(Path(self.stored()["OUTPUTS_DIR"]), desired_root)
        self.assertEqual(
            (desired_root / "podcast" / "episodes.json").read_bytes(),
            (podcast / "episodes.json").read_bytes(),
        )
        self.assertEqual(
            (desired_root / "podcast" / "feed.xml").read_bytes(),
            (podcast / "feed.xml").read_bytes(),
        )
        self.assertEqual(
            (desired_root / "podcast" / "media" / "one.wav").read_bytes(),
            b"episode one",
        )
        settings_store.apply_saved()
        self.assertEqual(config.OUTPUTS_DIR, desired_root)

    def test_output_root_change_refuses_divergent_podcast_state(self):
        live_root = self.root / "live-output"
        desired_root = self.root / "next-output"
        (live_root / "podcast").mkdir(parents=True)
        (desired_root / "podcast").mkdir(parents=True)
        (live_root / "podcast" / "episodes.json").write_bytes(b"source ledger")
        (desired_root / "podcast" / "episodes.json").write_bytes(b"different ledger")
        config.apply_values({"OUTPUTS_DIR": live_root})

        with self.assertRaises(settings_store.SettingsError) as raised:
            settings_store.update({"OUTPUTS_DIR": str(desired_root)})

        self.assertIn("different podcast state", str(raised.exception))
        self.assertEqual(config.OUTPUTS_DIR, live_root)
        self.assertFalse(self.store.exists())

    def test_output_root_change_refuses_nested_podcast_destination(self):
        live_root = self.root / "live-output"
        podcast = live_root / "podcast"
        podcast.mkdir(parents=True)
        (podcast / "episodes.json").write_bytes(b"source ledger")
        desired_root = podcast / "nested-output"
        config.apply_values({"OUTPUTS_DIR": live_root})

        with self.assertRaises(settings_store.SettingsError) as raised:
            settings_store.update({"OUTPUTS_DIR": str(desired_root)})

        self.assertIn("nested output roots", str(raised.exception))
        self.assertFalse(desired_root.exists())
        self.assertFalse(self.store.exists())

    def test_output_root_change_refuses_nested_podcast_source(self):
        desired_root = self.root / "outer-output"
        live_root = desired_root / "podcast" / "nested-live-output"
        podcast = live_root / "podcast"
        podcast.mkdir(parents=True)
        (podcast / "episodes.json").write_bytes(b"source ledger")
        config.apply_values({"OUTPUTS_DIR": live_root})

        with self.assertRaises(settings_store.SettingsError) as raised:
            settings_store.update({"OUTPUTS_DIR": str(desired_root)})

        self.assertIn("nested output roots", str(raised.exception))
        self.assertFalse(self.store.exists())

    def test_update_flags_only_the_settings_that_need_a_restart(self):
        self.assertEqual(settings_store.update({"RENDER_FPS": 24}), [])
        self.assertEqual(
            settings_store.update({"DB_PATH": str(self.root / "tasks.db")}),
            ["DB_PATH"],
        )

    def test_schema_reports_defaults_and_dynamic_options(self):
        settings_store.update({"RENDER_FPS": 24})
        fps = self.field("RENDER_FPS")
        self.assertEqual(fps["value"], 24)
        self.assertEqual(fps["default"], settings_store.defaults()["RENDER_FPS"])
        self.assertTrue(fps["is_overridden"])

        # Choices that depend on runtime config are resolved for the UI.
        self.assertEqual(
            self.field("TTS_DEFAULT_MODEL")["options"], sorted(config.TTS_MODELS)
        )
        expected_voices = (
            list(config.AVAILABLE_VOICES)
            + list(config.ORPHEUS_EN_VOICES)
            + list(config.POCKET_TTS_EN_VOICES)
        )
        self.assertEqual(self.field("TTS_DEFAULT_VOICE_1")["options"], expected_voices)

    def test_orpheus_api_key_is_masked(self):
        settings_store.update({"ORPHEUS_TTS_API_KEY": "orpheus-secret-value"})
        field = self.field("ORPHEUS_TTS_API_KEY")
        self.assertEqual(field["value"], "")
        self.assertEqual(field["masked"], "orph...alue")
        self.assertNotIn("orpheus-secret-value", json.dumps(settings_store.schema()))

    def test_every_spec_maps_to_a_real_config_attribute(self):
        for spec in settings_store.SPECS:
            with self.subTest(key=spec.key):
                self.assertTrue(hasattr(config, spec.key))

    def test_unusable_store_entries_never_block_startup(self):
        self.store.write_text(
            json.dumps({"version": 1, "values": {"RENDER_FPS": "many", "RENDER_QUALITY": "high"}}),
            encoding="utf-8",
        )
        settings_store.apply_saved()

        # The bad entry falls back to the default; the good one still applies.
        self.assertEqual(config.RENDER_FPS, settings_store.defaults()["RENDER_FPS"])
        self.assertEqual(config.RENDER_QUALITY, "high")

    def test_legacy_tts_chunk_override_migrates_to_vibevoice(self):
        self.store.write_text(
            json.dumps({"version": 1, "values": {"TTS_CHUNK_WORDS": 275}}),
            encoding="utf-8",
        )

        settings_store.apply_saved()

        self.assertEqual(config.VIBEVOICE_TTS_CHUNK_WORDS, 275)
        self.assertTrue(self.field("VIBEVOICE_TTS_CHUNK_WORDS")["is_overridden"])

    def test_retired_orpheus_speed_is_pruned_and_cannot_change_narration(self):
        self.store.write_text(
            json.dumps(
                {
                    "version": 1,
                    "values": {
                        "ORPHEUS_TTS_SPEED_PERCENT": 140,
                        "RENDER_QUALITY": "high",
                    },
                }
            ),
            encoding="utf-8",
        )

        settings_store.apply_saved()

        self.assertEqual(config.ORPHEUS_TTS_SPEED_PERCENT, 100)
        self.assertNotIn("ORPHEUS_TTS_SPEED_PERCENT", self.stored())
        self.assertEqual(self.stored()["RENDER_QUALITY"], "high")
        with self.assertRaises(AssertionError):
            self.field("ORPHEUS_TTS_SPEED_PERCENT")

    def test_legacy_daily_news_review_models_are_pruned_and_migrated(self):
        self.store.write_text(
            json.dumps(
                {
                    "version": 1,
                    "values": {
                        "DAILY_NEWS_GEMINI_REVIEW_MODEL": "3.7-flash",
                        "DAILY_NEWS_CHATGPT_REVIEW_MODEL": "high",
                    },
                }
            ),
            encoding="utf-8",
        )

        settings_store.apply_saved()

        self.assertNotIn("DAILY_NEWS_GEMINI_REVIEW_MODEL", self.stored())
        self.assertNotIn("DAILY_NEWS_CHATGPT_REVIEW_MODEL", self.stored())
        self.assertEqual(
            self.stored()["DAILY_NEWS_CHATGPT_REVIEW_MIN_LEVEL"], "high"
        )
        self.assertEqual(config.DAILY_NEWS_CHATGPT_REVIEW_MIN_LEVEL, "high")
        self.assertEqual(config.DAILY_NEWS_CHATGPT_REVIEW_MAX_LEVEL, "xhigh")
        with self.assertRaises(AssertionError):
            self.field("DAILY_NEWS_GEMINI_REVIEW_MODEL")
        with self.assertRaises(AssertionError):
            self.field("DAILY_NEWS_CHATGPT_REVIEW_MODEL")

    def test_provider_routing_settings_are_pruned_from_system_admin(self):
        provider_keys = {
            "AI_ENDPOINT",
            "AI_API_KEY",
            "AI_MODEL",
            "ANTHROPIC_BASE_URL",
            "ANTHROPIC_AUTH_TOKEN",
            "ANTHROPIC_MODEL",
            "ANTHROPIC_DEFAULT_HAIKU_MODEL",
        }
        self.store.write_text(
            json.dumps(
                {
                    "version": 1,
                    "values": {
                        "AI_ENDPOINT": "https://stale.example/v1/chat/completions",
                        "AI_API_KEY": "stale-secret",
                        "AI_MODEL": "stale-model",
                        "ANTHROPIC_MODEL": "stale-sdk-model",
                        "RENDER_QUALITY": "high",
                    },
                }
            ),
            encoding="utf-8",
        )

        settings_store.apply_saved()

        schema_keys = {
            field["key"]
            for group in settings_store.schema()
            for field in group["fields"]
        }
        self.assertTrue(provider_keys.isdisjoint(schema_keys))
        self.assertTrue(provider_keys.isdisjoint(self.stored()))
        self.assertEqual(self.stored()["RENDER_QUALITY"], "high")

    def test_corrupt_store_file_falls_back_to_defaults(self):
        self.store.write_text("{not json", encoding="utf-8")
        settings_store.apply_saved()
        self.assertEqual(config.RENDER_FPS, settings_store.defaults()["RENDER_FPS"])

    def field(self, key: str) -> dict:
        for group in settings_store.schema():
            for entry in group["fields"]:
                if entry["key"] == key:
                    return entry
        raise AssertionError(f"{key} missing from the settings schema")


if __name__ == "__main__":
    unittest.main()

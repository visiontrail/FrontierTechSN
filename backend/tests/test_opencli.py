import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from backend import config, settings_store
from backend.pipeline import opencli as opencli_module
from backend.pipeline.opencli import OpenCLIError, first_json
from backend.pipeline.opencli_rate_limit import (
    is_rate_limited_command,
    normalize_interval,
    wait_for_opencli_web_slot,
)


class OpenCLIOutputTests(unittest.TestCase):
    def test_first_json_recovers_cli_prose_and_fenced_payload(self):
        value = 'Gemini response:\n```json\n{"start_seconds": 4, "end_seconds": 10}\n```'

        self.assertEqual(first_json(value)["start_seconds"], 4)

    def test_first_json_rejects_non_json_output(self):
        with self.assertRaises(OpenCLIError):
            first_json("browser returned no structured payload")


class OpenCLISessionIsolationTests(unittest.TestCase):
    def test_backend_routes_persistent_tabs_through_the_project_namespace(self):
        env = opencli_module._environment()

        self.assertEqual(env["OPENCLI_SITE_SESSION_NAMESPACE"], "frontiertechsn")

    def test_one_review_can_isolate_and_reuse_its_own_persistent_tab(self):
        env = opencli_module._environment(
            site_session_namespace="frontiertechsn-review-request"
        )

        self.assertEqual(
            env["OPENCLI_SITE_SESSION_NAMESPACE"],
            "frontiertechsn-review-request",
        )

    def test_installed_runtime_contains_the_postinstall_namespace_patch(self):
        runtime = (
            config.PROJECT_ROOT
            / "tools"
            / "opencli"
            / "node_modules"
            / "@jackwener"
            / "opencli"
            / "dist"
            / "src"
            / "execution.js"
        )
        patcher = config.PROJECT_ROOT / "tools" / "opencli" / "patch-opencli.mjs"

        self.assertIn("OPENCLI_SITE_SESSION_NAMESPACE", patcher.read_text(encoding="utf-8"))
        self.assertIn("OPENCLI_SITE_SESSION_NAMESPACE", runtime.read_text(encoding="utf-8"))

    def test_isolated_runtime_settings_are_explicit_and_restart_bound(self):
        specs = {item.key: item for item in settings_store.SPECS}

        runtime = specs["OPENCLI_BROWSER_RUNTIME"]
        self.assertEqual(runtime.options, ("bridge", "isolated-headless"))
        self.assertTrue(runtime.restart_required)
        self.assertEqual(config.OPENCLI_BROWSER_RUNTIME, "bridge")
        for key in (
            "OPENCLI_ISOLATED_CHROME_BIN",
            "OPENCLI_ISOLATED_USER_DATA_DIR",
            "OPENCLI_ISOLATED_EXTENSION_PATH",
            "OPENCLI_ISOLATED_PROFILE",
            "OPENCLI_ISOLATED_DEBUG_PORT",
            "OPENCLI_ISOLATED_START_TIMEOUT",
        ):
            self.assertTrue(specs[key].restart_required)


class OpenCLIRateLimitTests(unittest.TestCase):
    def test_only_gemini_and_chatgpt_commands_are_rate_limited(self):
        self.assertTrue(is_rate_limited_command(["chatgpt", "ask", "prompt"]))
        self.assertTrue(is_rate_limited_command(["GEMINI", "ask", "prompt"]))
        self.assertFalse(is_rate_limited_command(["doctor"]))
        self.assertFalse(is_rate_limited_command(["youtube", "search", "query"]))

    def test_interval_is_clamped_to_three_through_ten_minutes(self):
        self.assertEqual(normalize_interval(1), 180)
        self.assertEqual(normalize_interval(240), 240)
        self.assertEqual(normalize_interval(900), 600)
        self.assertEqual(normalize_interval("invalid"), 180)

    def test_admin_setting_accepts_only_three_through_ten_minutes(self):
        spec = next(
            item
            for item in settings_store.SPECS
            if item.key == "OPENCLI_WEB_REQUEST_INTERVAL_SECONDS"
        )

        self.assertEqual(settings_store.coerce(spec, 180), 180)
        self.assertEqual(settings_store.coerce(spec, 600), 600)
        with self.assertRaises(settings_store.SettingsError):
            settings_store.coerce(spec, 179)
        with self.assertRaises(settings_store.SettingsError):
            settings_store.coerce(spec, 601)

    def test_persistent_slot_waits_between_immediate_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "limiter-state"
            now = [1_000.0]
            sleeps: list[float] = []

            def clock() -> float:
                return now[0]

            def sleeper(seconds: float) -> None:
                sleeps.append(seconds)
                now[0] += seconds

            first_delay = wait_for_opencli_web_slot(
                "chatgpt",
                interval=180,
                state_path=state_path,
                clock=clock,
                sleeper=sleeper,
                reporter=lambda _message: None,
            )
            second_delay = wait_for_opencli_web_slot(
                "gemini",
                interval=180,
                state_path=state_path,
                clock=clock,
                sleeper=sleeper,
                reporter=lambda _message: None,
            )

            self.assertEqual(first_delay, 0)
            self.assertEqual(second_delay, 180)
            self.assertEqual(sleeps, [180])
            self.assertEqual(float(state_path.read_text().strip()), 1180)


class OpenCLIRateLimitIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_backend_reserves_slot_before_starting_provider_process(self):
        class SuccessfulProcess:
            returncode = 0

            async def communicate(self):
                return b"ok", b""

        with (
            patch.object(opencli_module.asyncio, "to_thread", AsyncMock()) as to_thread,
            patch.object(
                opencli_module.asyncio,
                "create_subprocess_exec",
                AsyncMock(return_value=SuccessfulProcess()),
            ) as create_process,
            patch.object(config, "OPENCLI_WEB_REQUEST_INTERVAL_SECONDS", 240),
        ):
            result = await opencli_module.run_opencli(
                ["chatgpt", "status"], timeout=5
            )

        self.assertEqual(result.stdout, "ok")
        to_thread.assert_awaited_once_with(
            wait_for_opencli_web_slot,
            "chatgpt",
            interval=240,
        )
        self.assertEqual(
            create_process.await_args.kwargs["env"][
                "OPENCLI_WEB_REQUEST_SLOT_RESERVED"
            ],
            "1",
        )


if __name__ == "__main__":
    unittest.main()

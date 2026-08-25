import json
import os
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from backend import config
from backend.pipeline import opencli as opencli_module
from backend.pipeline import opencli_browser_runtime as browser_runtime
from backend.pipeline.opencli import OpenCLIError


class IsolatedBrowserConfigTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.chrome = self.root / "chrome-for-testing"
        self.chrome.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        self.chrome.chmod(0o755)
        self.extension = self.root / "opencli-extension"
        self.extension.mkdir()
        (self.extension / "manifest.json").write_text(
            json.dumps({"name": "OpenCLI", "version": "1.0.23"}),
            encoding="utf-8",
        )
        self.user_data = self.root / "isolated-profile"

    def tearDown(self):
        self.temporary.cleanup()

    def configured(self, **overrides):
        values = {
            "OPENCLI_BROWSER_RUNTIME": "isolated-headless",
            "OPENCLI_ISOLATED_CHROME_BIN": str(self.chrome),
            "OPENCLI_ISOLATED_USER_DATA_DIR": self.user_data,
            "OPENCLI_ISOLATED_EXTENSION_PATH": str(self.extension),
            "OPENCLI_ISOLATED_PROFILE": "frontiertechsn-headless",
            "OPENCLI_ISOLATED_DEBUG_PORT": 19242,
            "OPENCLI_ISOLATED_START_TIMEOUT": 30,
        }
        values.update(overrides)
        stack = ExitStack()
        for key, value in values.items():
            stack.enter_context(patch.object(config, key, value))
        return stack

    def test_default_bridge_mode_does_not_start_an_isolated_browser(self):
        with patch.object(config, "OPENCLI_BROWSER_RUNTIME", "bridge"):
            self.assertIsNone(browser_runtime.ensure_isolated_headless_browser())

    def test_isolated_environment_pins_the_dedicated_profile(self):
        with (
            patch.object(config, "OPENCLI_BROWSER_RUNTIME", "isolated-headless"),
            patch.object(config, "OPENCLI_PROFILE", "operator-profile"),
            patch.object(
                config,
                "OPENCLI_ISOLATED_PROFILE",
                "frontiertechsn-headless",
            ),
        ):
            environment = opencli_module._environment()

        self.assertEqual(
            environment["OPENCLI_PROFILE"],
            "frontiertechsn-headless",
        )

    def test_bridge_environment_preserves_the_existing_profile(self):
        with (
            patch.object(config, "OPENCLI_BROWSER_RUNTIME", "bridge"),
            patch.object(config, "OPENCLI_PROFILE", "operator-profile"),
            patch.object(config, "OPENCLI_ISOLATED_PROFILE", "isolated-profile"),
        ):
            environment = opencli_module._environment()

        self.assertEqual(environment["OPENCLI_PROFILE"], "operator-profile")

    def test_child_agents_receive_the_live_isolated_runtime(self):
        with (
            patch.object(config, "OPENCLI_BROWSER_RUNTIME", "isolated-headless"),
            patch.object(config, "OPENCLI_ISOLATED_PROFILE", "isolated-profile"),
        ):
            environment = browser_runtime.runtime_subprocess_environment(
                {"SENTINEL": "kept"}
            )

        self.assertEqual(environment["SENTINEL"], "kept")
        self.assertEqual(environment["OPENCLI_BROWSER_RUNTIME"], "isolated-headless")
        self.assertEqual(environment["OPENCLI_PROFILE"], "isolated-profile")

    def test_isolated_mode_requires_an_exact_profile(self):
        with self.configured(OPENCLI_ISOLATED_PROFILE=""):
            with self.assertRaisesRegex(
                browser_runtime.OpenCLIBrowserRuntimeError,
                "refusing to auto-select",
            ):
                browser_runtime.isolated_browser_config()

    def test_human_chrome_profile_directory_is_rejected(self):
        human_profile = (
            Path.home()
            / "Library"
            / "Application Support"
            / "Google"
            / "Chrome"
            / "Default"
        )
        with self.configured(OPENCLI_ISOLATED_USER_DATA_DIR=human_profile):
            with self.assertRaisesRegex(
                browser_runtime.OpenCLIBrowserRuntimeError,
                "must not reuse a normal human browser profile",
            ):
                browser_runtime.isolated_browser_config()

    def test_headless_command_uses_only_isolated_paths_and_loopback(self):
        with self.configured():
            runtime = browser_runtime.isolated_browser_config()
        command = browser_runtime.build_chrome_command(runtime, headless=True)

        self.assertEqual(command[0], str(self.chrome.resolve()))
        self.assertIn("--headless=new", command)
        self.assertIn("--remote-debugging-address=127.0.0.1", command)
        self.assertIn(f"--user-data-dir={self.user_data.resolve()}", command)
        self.assertIn(f"--load-extension={self.extension.resolve()}", command)
        human_chrome = Path.home() / "Library/Application Support/Google/Chrome"
        self.assertNotIn(str(human_chrome), " ".join(command))

    def test_existing_managed_process_must_match_current_settings(self):
        record = browser_runtime.BrowserProcessRecord(
            pid=123,
            chrome_binary="/different/chrome",
            user_data_dir=str(self.user_data),
            extension_path=str(self.extension),
            debug_port=19242,
            headless=True,
            started_at=1.0,
        )
        with (
            self.configured(),
            patch.object(browser_runtime, "_read_record", return_value=record),
            patch.object(browser_runtime, "_pid_running", return_value=True),
            patch.object(
                browser_runtime,
                "_debug_status",
                return_value={"User-Agent": "HeadlessChrome/152"},
            ),
            patch.object(browser_runtime, "_runtime_lock"),
        ):
            browser_runtime._runtime_lock.return_value.__enter__ = Mock(
                return_value=None
            )
            browser_runtime._runtime_lock.return_value.__exit__ = Mock(
                return_value=False
            )
            with self.assertRaisesRegex(
                browser_runtime.OpenCLIBrowserRuntimeError,
                "different settings",
            ):
                browser_runtime.start_isolated_browser()

    def test_start_launches_headless_and_waits_for_exact_profile(self):
        process = Mock(pid=456)
        process.poll.return_value = None
        status = {"User-Agent": "HeadlessChrome/152"}
        with (
            self.configured(),
            patch.object(browser_runtime, "_runtime_lock"),
            patch.object(browser_runtime, "_read_record", return_value=None),
            patch.object(browser_runtime, "_debug_status", side_effect=[None, status]),
            patch.object(browser_runtime, "_ensure_daemon_running") as daemon,
            patch.object(
                browser_runtime.subprocess, "Popen", return_value=process
            ) as popen,
            patch.object(browser_runtime, "_write_record") as write_record,
            patch.object(browser_runtime, "_wait_for_profile") as wait_profile,
            patch.object(
                browser_runtime, "_log_path", return_value=self.root / "chrome.log"
            ),
        ):
            record = browser_runtime.start_isolated_browser()

        daemon.assert_called_once_with(30)
        command = popen.call_args.args[0]
        self.assertIn("--headless=new", command)
        self.assertIn(f"--user-data-dir={self.user_data.resolve()}", command)
        wait_profile.assert_called_once_with("frontiertechsn-headless", 30)
        write_record.assert_called_once()
        self.assertEqual(record.pid, 456)
        self.assertTrue(record.headless)

    def test_stop_refuses_a_pid_that_is_not_the_recorded_browser(self):
        record = browser_runtime.BrowserProcessRecord(
            pid=os.getpid(),
            chrome_binary=str(self.chrome),
            user_data_dir=str(self.user_data),
            extension_path=str(self.extension),
            debug_port=19242,
            headless=True,
            started_at=1.0,
        )
        with (
            patch.object(browser_runtime, "_pid_running", return_value=True),
            patch.object(
                browser_runtime, "_process_command", return_value="python tests"
            ),
        ):
            with self.assertRaisesRegex(
                browser_runtime.OpenCLIBrowserRuntimeError,
                "Refusing to terminate",
            ):
                browser_runtime._terminate_record(record)

    def test_terminate_accepts_process_disappearing_after_sigterm(self):
        record = browser_runtime.BrowserProcessRecord(
            pid=321,
            chrome_binary=str(self.chrome),
            user_data_dir=str(self.user_data),
            extension_path=str(self.extension),
            debug_port=19242,
            headless=True,
            started_at=1.0,
        )
        command = (
            f"{self.chrome} --user-data-dir={self.user_data} "
            "--remote-debugging-port=19242"
        )
        with (
            patch.object(browser_runtime, "_pid_running", return_value=True),
            patch.object(
                browser_runtime,
                "_process_command",
                side_effect=[command, ""],
            ),
            patch.object(browser_runtime.os, "getpgid", return_value=321),
            patch.object(browser_runtime.os, "killpg") as killpg,
        ):
            browser_runtime._terminate_record(record)

        killpg.assert_called_once_with(321, browser_runtime.signal.SIGTERM)


class OpenCLIIsolatedRuntimeIntegrationTests(unittest.IsolatedAsyncioTestCase):
    class SuccessfulProcess:
        returncode = 0

        async def communicate(self):
            return b"ok", b""

    async def test_default_bridge_mode_does_not_touch_runtime_manager(self):
        with (
            patch.object(config, "OPENCLI_BROWSER_RUNTIME", "bridge"),
            patch.object(
                opencli_module,
                "ensure_isolated_headless_browser",
            ) as ensure,
            patch.object(
                opencli_module.asyncio,
                "create_subprocess_exec",
                AsyncMock(return_value=self.SuccessfulProcess()),
            ),
        ):
            result = await opencli_module.run_opencli(["doctor"], timeout=5)

        self.assertEqual(result.stdout, "ok")
        ensure.assert_not_called()

    async def test_isolated_mode_is_ready_before_opencli_is_spawned(self):
        calls: list[str] = []

        async def create_process(*_args, **_kwargs):
            calls.append("opencli")
            return self.SuccessfulProcess()

        async def to_thread(function, *_args, **_kwargs):
            calls.append("runtime")
            return function()

        with (
            patch.object(config, "OPENCLI_BROWSER_RUNTIME", "isolated-headless"),
            patch.object(config, "OPENCLI_ISOLATED_PROFILE", "isolated"),
            patch.object(
                opencli_module,
                "ensure_isolated_headless_browser",
                return_value=Mock(),
            ),
            patch.object(opencli_module.asyncio, "to_thread", side_effect=to_thread),
            patch.object(
                opencli_module.asyncio,
                "create_subprocess_exec",
                side_effect=create_process,
            ) as create_process_mock,
        ):
            await opencli_module.run_opencli(["doctor"], timeout=5)

        self.assertEqual(calls, ["runtime", "opencli"])
        self.assertEqual(
            create_process_mock.await_args.kwargs["env"][
                "OPENCLI_ISOLATED_RUNTIME_READY"
            ],
            "1",
        )

    async def test_isolated_start_failure_never_spawns_opencli(self):
        failure = browser_runtime.OpenCLIBrowserRuntimeError("profile unavailable")
        with (
            patch.object(config, "OPENCLI_BROWSER_RUNTIME", "isolated-headless"),
            patch.object(
                opencli_module,
                "ensure_isolated_headless_browser",
                side_effect=failure,
            ),
            patch.object(
                opencli_module.asyncio,
                "create_subprocess_exec",
                AsyncMock(),
            ) as create_process,
        ):
            with self.assertRaisesRegex(OpenCLIError, "profile unavailable"):
                await opencli_module.run_opencli(["doctor"], timeout=5)

        create_process.assert_not_awaited()

    async def test_invalid_runtime_never_spawns_opencli(self):
        with (
            patch.object(config, "OPENCLI_BROWSER_RUNTIME", "surprise-browser"),
            patch.object(
                opencli_module.asyncio,
                "create_subprocess_exec",
                AsyncMock(),
            ) as create_process,
        ):
            with self.assertRaisesRegex(OpenCLIError, "must be bridge"):
                await opencli_module.run_opencli(["doctor"], timeout=5)

        create_process.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()

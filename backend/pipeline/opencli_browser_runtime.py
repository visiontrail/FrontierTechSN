"""Managed, opt-in isolated browser runtime for OpenCLI.

The production default remains OpenCLI's existing Browser Bridge connection.
When ``OPENCLI_BROWSER_RUNTIME=isolated-headless`` is explicitly selected, this
module starts a separate Chrome for Testing/Chromium process with its own user
data directory and unpacked OpenCLI extension. OpenCLI still talks through the
Browser Bridge, preserving adapter, upload, download, and tab-session semantics.

The isolated profile id/alias is mandatory. That fail-closed requirement keeps
OpenCLI from silently routing a command to an operator's normal Chrome profile
when the isolated extension has not connected yet.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

from backend import config


BRIDGE_RUNTIME = "bridge"
ISOLATED_HEADLESS_RUNTIME = "isolated-headless"
_DAEMON_PORT = 19825


class OpenCLIBrowserRuntimeError(RuntimeError):
    """The isolated OpenCLI browser could not be started safely."""


@dataclass(frozen=True)
class IsolatedBrowserConfig:
    chrome_binary: Path
    user_data_dir: Path
    extension_path: Path
    debug_port: int
    profile: str
    start_timeout: int


@dataclass(frozen=True)
class BrowserProcessRecord:
    pid: int
    chrome_binary: str
    user_data_dir: str
    extension_path: str
    debug_port: int
    headless: bool
    started_at: float


def runtime_mode() -> str:
    mode = str(config.OPENCLI_BROWSER_RUNTIME).strip().lower()
    if mode not in {BRIDGE_RUNTIME, ISOLATED_HEADLESS_RUNTIME}:
        raise OpenCLIBrowserRuntimeError(
            "OPENCLI_BROWSER_RUNTIME must be bridge or isolated-headless; "
            f"got {mode or 'blank'}"
        )
    return mode


def isolated_headless_enabled() -> bool:
    return runtime_mode() == ISOLATED_HEADLESS_RUNTIME


def runtime_subprocess_environment(
    base: dict[str, str] | None = None,
) -> dict[str, str]:
    """Propagate live Admin runtime selection to wrapper calls in child agents."""
    environment = dict(os.environ if base is None else base)
    mode = runtime_mode()
    environment["OPENCLI_BROWSER_RUNTIME"] = mode
    if mode == ISOLATED_HEADLESS_RUNTIME:
        environment["OPENCLI_PROFILE"] = str(config.OPENCLI_ISOLATED_PROFILE).strip()
    return environment


def _configured_path(value: object, *, label: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise OpenCLIBrowserRuntimeError(
            f"{label} is required for the isolated-headless OpenCLI runtime"
        )
    return config.resolve_project_path(text).resolve()


def _same_or_child(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _human_browser_roots() -> tuple[Path, ...]:
    application_support = Path.home() / "Library" / "Application Support"
    return (
        application_support / "Google" / "Chrome",
        application_support / "Google" / "Chrome Beta",
        application_support / "Google" / "Chrome Dev",
        application_support / "Google" / "Chrome Canary",
        application_support / "Google" / "Chrome for Testing",
        application_support / "Chromium",
    )


def isolated_browser_config(*, require_profile: bool = True) -> IsolatedBrowserConfig:
    chrome_binary = _configured_path(
        config.OPENCLI_ISOLATED_CHROME_BIN,
        label="OPENCLI_ISOLATED_CHROME_BIN",
    )
    user_data_dir = Path(config.OPENCLI_ISOLATED_USER_DATA_DIR).expanduser().resolve()
    extension_path = _configured_path(
        config.OPENCLI_ISOLATED_EXTENSION_PATH,
        label="OPENCLI_ISOLATED_EXTENSION_PATH",
    )
    profile = str(config.OPENCLI_ISOLATED_PROFILE).strip()
    if require_profile and not profile:
        raise OpenCLIBrowserRuntimeError(
            "OPENCLI_ISOLATED_PROFILE must name the dedicated connected Browser "
            "Bridge profile; refusing to auto-select an operator profile"
        )

    if not chrome_binary.is_file() or not os.access(chrome_binary, os.X_OK):
        raise OpenCLIBrowserRuntimeError(
            f"Isolated Chrome binary is not executable: {chrome_binary}"
        )
    manifest_path = extension_path / "manifest.json"
    if not manifest_path.is_file():
        raise OpenCLIBrowserRuntimeError(
            f"OpenCLI extension manifest is missing: {manifest_path}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OpenCLIBrowserRuntimeError(
            f"OpenCLI extension manifest is invalid: {manifest_path}"
        ) from exc
    if str(manifest.get("name", "")).strip().lower() != "opencli":
        raise OpenCLIBrowserRuntimeError(
            f"Configured extension is not OpenCLI: {manifest_path}"
        )

    unsafe_roots = (
        Path("/").resolve(),
        Path.home().resolve(),
        config.PROJECT_ROOT.resolve(),
    )
    if user_data_dir in unsafe_roots:
        raise OpenCLIBrowserRuntimeError(
            f"Refusing unsafe isolated Chrome user-data directory: {user_data_dir}"
        )
    for browser_root in _human_browser_roots():
        if _same_or_child(user_data_dir, browser_root.resolve()):
            raise OpenCLIBrowserRuntimeError(
                "OPENCLI_ISOLATED_USER_DATA_DIR must not reuse a normal human "
                f"browser profile: {user_data_dir}"
            )

    debug_port = int(config.OPENCLI_ISOLATED_DEBUG_PORT)
    if not 1024 <= debug_port <= 65535:
        raise OpenCLIBrowserRuntimeError(
            "OPENCLI_ISOLATED_DEBUG_PORT must be between 1024 and 65535"
        )
    start_timeout = int(config.OPENCLI_ISOLATED_START_TIMEOUT)
    if not 5 <= start_timeout <= 120:
        raise OpenCLIBrowserRuntimeError(
            "OPENCLI_ISOLATED_START_TIMEOUT must be between 5 and 120 seconds"
        )
    return IsolatedBrowserConfig(
        chrome_binary=chrome_binary,
        user_data_dir=user_data_dir,
        extension_path=extension_path,
        debug_port=debug_port,
        profile=profile,
        start_timeout=start_timeout,
    )


def _run_dir() -> Path:
    path = config.PROJECT_ROOT / ".run"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _pid_path() -> Path:
    return _run_dir() / "opencli-isolated-browser.json"


def _lock_path() -> Path:
    return _run_dir() / "opencli-isolated-browser.lock"


def _log_path() -> Path:
    return _run_dir() / "opencli-isolated-browser.log"


@contextmanager
def _runtime_lock() -> Iterator[None]:
    with _lock_path().open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _write_record(record: BrowserProcessRecord) -> None:
    path = _pid_path()
    staging = path.with_suffix(".tmp")
    staging.write_text(json.dumps(asdict(record), indent=2), encoding="utf-8")
    os.replace(staging, path)


def _read_record() -> BrowserProcessRecord | None:
    try:
        payload = json.loads(_pid_path().read_text(encoding="utf-8"))
        return BrowserProcessRecord(**payload)
    except (FileNotFoundError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _remove_record() -> None:
    try:
        _pid_path().unlink()
    except FileNotFoundError:
        pass


def _pid_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _tcp_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.4):
            return True
    except OSError:
        return False


def _debug_status(port: int) -> dict[str, object] | None:
    try:
        with urllib.request.urlopen(  # noqa: S310 - fixed loopback URL
            f"http://127.0.0.1:{port}/json/version",
            timeout=1,
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError, urllib.error.URLError):
        return None
    return payload if isinstance(payload, dict) else None


def _node_binary() -> str:
    node = shutil.which("node")
    if not node:
        raise OpenCLIBrowserRuntimeError("Node.js is required to start OpenCLI daemon")
    return node


def _ensure_daemon_running(timeout: int) -> None:
    if _tcp_open(_DAEMON_PORT):
        return
    daemon = (
        config.PROJECT_ROOT
        / "tools"
        / "opencli"
        / "node_modules"
        / "@jackwener"
        / "opencli"
        / "dist"
        / "src"
        / "daemon.js"
    )
    if not daemon.is_file():
        raise OpenCLIBrowserRuntimeError(
            f"Project-local OpenCLI daemon is missing: {daemon}"
        )
    subprocess.Popen(  # noqa: S603 - fixed project-local daemon
        [_node_binary(), str(daemon)],
        cwd=config.PROJECT_ROOT,
        env=os.environ.copy(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _tcp_open(_DAEMON_PORT):
            return
        time.sleep(0.2)
    raise OpenCLIBrowserRuntimeError(
        f"OpenCLI daemon did not start on loopback port {_DAEMON_PORT}"
    )


def build_chrome_command(
    runtime: IsolatedBrowserConfig,
    *,
    headless: bool,
) -> list[str]:
    command = [str(runtime.chrome_binary)]
    if headless:
        command.append("--headless=new")
    command.extend(
        [
            "--remote-debugging-address=127.0.0.1",
            f"--remote-debugging-port={runtime.debug_port}",
            "--remote-allow-origins=*",
            f"--user-data-dir={runtime.user_data_dir}",
            f"--load-extension={runtime.extension_path}",
            f"--disable-extensions-except={runtime.extension_path}",
            "--no-first-run",
            "--no-default-browser-check",
            "--window-size=1280,900",
            "--disable-background-timer-throttling",
            "--disable-backgrounding-occluded-windows",
            "about:blank",
        ]
    )
    return command


def _opencli_binary() -> Path:
    return (
        config.PROJECT_ROOT / "tools" / "opencli" / "node_modules" / ".bin" / "opencli"
    )


def _connected_profiles() -> str:
    binary = _opencli_binary()
    if not binary.is_file():
        raise OpenCLIBrowserRuntimeError(
            f"Project-local OpenCLI binary is missing: {binary}"
        )
    result = subprocess.run(  # noqa: S603 - fixed project-local executable
        [str(binary), "profile", "list"],
        cwd=config.PROJECT_ROOT,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    return f"{result.stdout}\n{result.stderr}"


def _wait_for_profile(profile: str, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    last_output = ""
    while time.monotonic() < deadline:
        try:
            last_output = _connected_profiles()
        except (OSError, subprocess.SubprocessError) as exc:
            last_output = str(exc)
        if profile and any(
            profile in line and "connected" in line.lower()
            for line in last_output.splitlines()
        ):
            return
        time.sleep(0.5)
    detail = " ".join(last_output.split())[-500:]
    raise OpenCLIBrowserRuntimeError(
        f'Isolated Browser Bridge profile "{profile}" did not connect. '
        "Run ./scripts/opencli-browser-runtime.sh bootstrap, sign in using the "
        "dedicated window, then configure its context id or alias. "
        f"Profile output: {detail or 'none'}"
    )


def _record_matches(
    record: BrowserProcessRecord, runtime: IsolatedBrowserConfig
) -> bool:
    return (
        Path(record.chrome_binary).resolve() == runtime.chrome_binary
        and Path(record.user_data_dir).resolve() == runtime.user_data_dir
        and Path(record.extension_path).resolve() == runtime.extension_path
        and record.debug_port == runtime.debug_port
    )


def _tail_log(limit: int = 1500) -> str:
    try:
        return _log_path().read_text(encoding="utf-8", errors="replace")[-limit:]
    except OSError:
        return ""


def start_isolated_browser(
    *,
    headless: bool = True,
    require_profile: bool = True,
) -> BrowserProcessRecord:
    runtime = isolated_browser_config(require_profile=require_profile)
    with _runtime_lock():
        record = _read_record()
        status = _debug_status(runtime.debug_port)
        if record and _pid_running(record.pid) and status:
            if not _record_matches(record, runtime):
                raise OpenCLIBrowserRuntimeError(
                    "An isolated browser is already running with different settings; "
                    "stop it explicitly before changing runtime configuration"
                )
            if record.headless != headless:
                requested = "headless" if headless else "visible bootstrap"
                current = "headless" if record.headless else "visible bootstrap"
                raise OpenCLIBrowserRuntimeError(
                    f"The isolated browser is already running as {current}; stop it "
                    f"explicitly before starting {requested} mode"
                )
            if require_profile:
                _wait_for_profile(runtime.profile, runtime.start_timeout)
            return record
        if status:
            raise OpenCLIBrowserRuntimeError(
                f"Debug port {runtime.debug_port} is already owned by an unmanaged "
                "browser; refusing to attach or fall back"
            )
        if record and _pid_running(record.pid):
            raise OpenCLIBrowserRuntimeError(
                f"Managed browser pid {record.pid} is alive but its debug endpoint is "
                "unavailable; stop it explicitly before retrying"
            )
        _remove_record()
        _ensure_daemon_running(runtime.start_timeout)
        runtime.user_data_dir.mkdir(parents=True, exist_ok=True)
        _log_path().parent.mkdir(parents=True, exist_ok=True)
        # Each launch gets a fresh diagnostic record. Stale Chrome output can
        # otherwise make a current readiness failure point at an older process.
        with _log_path().open("wb") as log:
            process = subprocess.Popen(  # noqa: S603 - validated configured executable
                build_chrome_command(runtime, headless=headless),
                cwd=config.PROJECT_ROOT,
                env=os.environ.copy(),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
        record = BrowserProcessRecord(
            pid=process.pid,
            chrome_binary=str(runtime.chrome_binary),
            user_data_dir=str(runtime.user_data_dir),
            extension_path=str(runtime.extension_path),
            debug_port=runtime.debug_port,
            headless=headless,
            started_at=time.time(),
        )
        _write_record(record)
        deadline = time.monotonic() + runtime.start_timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                _remove_record()
                raise OpenCLIBrowserRuntimeError(
                    f"Isolated Chrome exited with code {process.returncode}: {_tail_log()}"
                )
            status = _debug_status(runtime.debug_port)
            if status:
                break
            time.sleep(0.2)
        else:
            try:
                _terminate_record(record)
            finally:
                _remove_record()
            raise OpenCLIBrowserRuntimeError(
                "Isolated Chrome did not expose its loopback debug endpoint: "
                f"{_tail_log()}"
            )
        if headless and "HeadlessChrome" not in str(status.get("User-Agent", "")):
            try:
                _terminate_record(record)
            finally:
                _remove_record()
            raise OpenCLIBrowserRuntimeError(
                "Configured browser did not start in headless mode; refusing to use it"
            )
        if require_profile:
            try:
                _wait_for_profile(runtime.profile, runtime.start_timeout)
            except Exception:
                try:
                    _terminate_record(record)
                finally:
                    _remove_record()
                raise
        return record


def ensure_isolated_headless_browser() -> BrowserProcessRecord | None:
    if not isolated_headless_enabled():
        return None
    return start_isolated_browser(headless=True, require_profile=True)


def _process_command(pid: int) -> str:
    try:
        result = subprocess.run(  # noqa: S603 - fixed system inspection command
            ["ps", "-p", str(pid), "-o", "command="],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip()


def _terminate_record(record: BrowserProcessRecord) -> None:
    if not _pid_running(record.pid):
        return
    command = _process_command(record.pid)
    required = (
        f"--user-data-dir={record.user_data_dir}",
        f"--remote-debugging-port={record.debug_port}",
    )
    if not command or any(fragment not in command for fragment in required):
        raise OpenCLIBrowserRuntimeError(
            f"Refusing to terminate pid {record.pid}; command line does not match "
            "the managed isolated browser"
        )
    try:
        process_group = os.getpgid(record.pid)
    except ProcessLookupError:
        return
    if process_group != record.pid:
        raise OpenCLIBrowserRuntimeError(
            f"Refusing to terminate unexpected process group {process_group} for pid {record.pid}"
        )
    os.killpg(process_group, signal.SIGTERM)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if (
            not _pid_running(record.pid)
            or not _process_command(record.pid)
            or _debug_status(record.debug_port) is None
        ):
            return
        time.sleep(0.2)
    # Chrome may be reaped between the final poll and the kill. Revalidate the
    # exact target instead of turning a successful SIGTERM into an EPERM error.
    if not _process_command(record.pid):
        return
    try:
        os.killpg(process_group, signal.SIGKILL)
    except ProcessLookupError:
        return
    except PermissionError:
        if not _process_command(record.pid) or _debug_status(record.debug_port) is None:
            return
        raise


def stop_isolated_browser() -> bool:
    with _runtime_lock():
        record = _read_record()
        if not record:
            return False
        _terminate_record(record)
        _remove_record()
        return True


def runtime_status() -> dict[str, object]:
    record = _read_record()
    if not record:
        return {"running": False, "mode": runtime_mode()}
    debug = _debug_status(record.debug_port)
    return {
        "running": _pid_running(record.pid) and debug is not None,
        "mode": runtime_mode(),
        "process": asdict(record),
        "browser": debug or {},
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage FrontierTechSN's isolated OpenCLI browser runtime"
    )
    parser.add_argument(
        "command",
        choices=("start", "bootstrap", "prepare", "status", "stop"),
        help=(
            "start headless, bootstrap a visible isolated profile, prepare a "
            "wrapper call, inspect, or stop"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "start":
            record = start_isolated_browser(headless=True, require_profile=True)
            print(json.dumps(asdict(record), indent=2))
        elif args.command == "bootstrap":
            record = start_isolated_browser(headless=False, require_profile=False)
            print(json.dumps(asdict(record), indent=2))
            print(
                "Dedicated browser started visibly for one-time extension/profile "
                "discovery and provider sign-in. Run './scripts/opencli.sh profile "
                "list', assign an alias, save OPENCLI_ISOLATED_PROFILE, then stop it."
            )
        elif args.command == "prepare":
            ensure_isolated_headless_browser()
            print(str(config.OPENCLI_ISOLATED_PROFILE).strip())
        elif args.command == "status":
            print(json.dumps(runtime_status(), indent=2))
        elif args.command == "stop":
            print("stopped" if stop_isolated_browser() else "not running")
    except OpenCLIBrowserRuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

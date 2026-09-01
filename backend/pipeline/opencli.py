"""Project-local OpenCLI subprocess adapter.

Only the repository wrapper configured by :mod:`backend.config` is executed.
This module never installs a global npm package and never writes Claude skills
outside ``.claude/skills``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend import config
from backend.pipeline.opencli_rate_limit import (
    is_rate_limited_command,
    wait_for_opencli_web_slot,
)
from backend.pipeline.opencli_browser_runtime import (
    ISOLATED_HEADLESS_RUNTIME,
    OpenCLIBrowserRuntimeError,
    ensure_isolated_headless_browser,
    runtime_mode,
    runtime_subprocess_environment,
)

logger = logging.getLogger(__name__)

_SITE_SESSION_COMPONENT_RE = re.compile(r"[^a-z0-9._-]+", re.IGNORECASE)
CLOSE_ADAPTER_SESSIONS_SCRIPT = (
    config.PROJECT_ROOT / "tools" / "opencli" / "close-adapter-sessions.mjs"
)


class OpenCLIError(RuntimeError):
    """An OpenCLI command could not produce a usable result."""


@dataclass(frozen=True)
class OpenCLIResult:
    args: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


def _site_session_name(namespace: str, site: str) -> str:
    """Mirror OpenCLI's project namespace normalization exactly."""
    normalized = _SITE_SESSION_COMPONENT_RE.sub("-", str(namespace).strip())
    normalized = normalized.strip("-")[:64]
    if not normalized:
        raise OpenCLIError("OpenCLI site-session namespace is empty after normalization")
    normalized_site = _SITE_SESSION_COMPONENT_RE.sub("-", str(site).strip())
    normalized_site = normalized_site.strip("-")
    if not normalized_site:
        raise OpenCLIError("OpenCLI site name is empty after normalization")
    return f"site:{normalized}:{normalized_site}"


def _environment(*, site_session_namespace: str | None = None) -> dict[str, str]:
    env = runtime_subprocess_environment()
    path_parts = [
        str(config.PROJECT_ROOT / ".venv" / "bin"),
        str(config.PROJECT_ROOT / "tools" / "opencli" / "node_modules" / ".bin"),
        env.get("PATH", ""),
    ]
    env["PATH"] = os.pathsep.join(part for part in path_parts if part)
    env.setdefault("OPENCLI_BROWSER_CONNECT_TIMEOUT", "20")
    env.setdefault("OPENCLI_BROWSER_COMMAND_TIMEOUT", str(config.OPENCLI_TIMEOUT))
    env["OPENCLI_SITE_SESSION_NAMESPACE"] = (
        site_session_namespace or config.OPENCLI_SITE_SESSION_NAMESPACE
    )
    # The project-local ChatGPT adapter first attempts the requested preferred
    # level, then treats these values as its fallback range. A failed switch can
    # keep Medium/High/Extra High, while Instant/Pro are rejected before a prompt
    # can be sent.
    env["OPENCLI_CHATGPT_MODEL_MIN"] = config.DAILY_NEWS_CHATGPT_REVIEW_MIN_LEVEL
    env["OPENCLI_CHATGPT_MODEL_MAX"] = config.DAILY_NEWS_CHATGPT_REVIEW_MAX_LEVEL
    if runtime_mode() == ISOLATED_HEADLESS_RUNTIME:
        # Never let an isolated run auto-select among connected profiles. The
        # dedicated context id/alias is a fail-closed isolation boundary.
        env["OPENCLI_PROFILE"] = config.OPENCLI_ISOLATED_PROFILE
    elif config.OPENCLI_PROFILE:
        env["OPENCLI_PROFILE"] = config.OPENCLI_PROFILE
    return env


async def run_opencli(
    args: list[str],
    *,
    timeout: int | None = None,
    check: bool = True,
    site_session_namespace: str | None = None,
) -> OpenCLIResult:
    binary = Path(config.OPENCLI_BIN)
    if not binary.is_file():
        raise OpenCLIError(
            f"Project-local OpenCLI wrapper is missing at {binary}. "
            f"Run npm install --prefix {config.PROJECT_ROOT / 'tools' / 'opencli'}"
        )

    try:
        mode = runtime_mode()
    except OpenCLIBrowserRuntimeError as exc:
        raise OpenCLIError(f"OpenCLI browser runtime is invalid: {exc}") from exc
    if mode == ISOLATED_HEADLESS_RUNTIME:
        try:
            await asyncio.to_thread(ensure_isolated_headless_browser)
        except OpenCLIBrowserRuntimeError as exc:
            raise OpenCLIError(f"Isolated OpenCLI browser is not ready: {exc}") from exc

    command = [str(binary), *[str(arg) for arg in args]]
    env = _environment(site_session_namespace=site_session_namespace)
    if mode == ISOLATED_HEADLESS_RUNTIME:
        env["OPENCLI_ISOLATED_RUNTIME_READY"] = "1"
    if is_rate_limited_command(args):
        # Pace before starting the subprocess so the provider-command timeout
        # measures the web operation, not time intentionally spent in queue.
        await asyncio.to_thread(
            wait_for_opencli_web_slot,
            str(args[0]).lower(),
            interval=config.OPENCLI_WEB_REQUEST_INTERVAL_SECONDS,
        )
        env["OPENCLI_WEB_REQUEST_SLOT_RESERVED"] = "1"
    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=str(config.PROJECT_ROOT),
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            process.communicate(), timeout=timeout or config.OPENCLI_TIMEOUT
        )
    except TimeoutError as exc:
        process.kill()
        await process.communicate()
        raise OpenCLIError(
            f"OpenCLI command timed out after {timeout or config.OPENCLI_TIMEOUT}s: "
            f"{' '.join(args[:3])}"
        ) from exc

    result = OpenCLIResult(
        args=tuple(args),
        returncode=process.returncode or 0,
        stdout=stdout_bytes.decode("utf-8", errors="replace").strip(),
        stderr=stderr_bytes.decode("utf-8", errors="replace").strip(),
    )
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout or "unknown OpenCLI failure")[-1200:]
        raise OpenCLIError(
            f"OpenCLI {' '.join(args[:2])} failed with exit {result.returncode}: {detail}"
        )
    return result


async def run_opencli_with_retries(
    args: list[str],
    *,
    timeout: int | None = None,
    attempts: int | None = None,
    check: bool = True,
    log=None,
    label: str = "OpenCLI",
) -> OpenCLIResult:
    """Run a browser-backed command with bounded exponential retries.

    The browser bridge and signed-in web apps can finish late even when the
    first command loses its lease. Each attempt is a fresh command, while the
    site adapter's persistent session keeps authentication and conversation
    state. Ten total attempts is the project default.
    """
    total = max(1, attempts or config.OPENCLI_MAX_ATTEMPTS)
    last_error: Exception | None = None
    for attempt in range(1, total + 1):
        try:
            return await run_opencli(args, timeout=timeout, check=check)
        except Exception as exc:  # noqa: BLE001 - preserve the final adapter error
            last_error = exc
            message = f"{label} attempt {attempt}/{total} failed: {exc}"
            if log:
                log(message)
            else:
                logger.warning(message)
            if attempt < total:
                delay = min(45.0, config.OPENCLI_RETRY_BASE_SECONDS * 2 ** (attempt - 1))
                await asyncio.sleep(delay)
    raise OpenCLIError(f"{label} failed after {total} attempts: {last_error}") from last_error


async def close_opencli_site_sessions(
    site_session_namespace: str,
    *,
    sites: tuple[str, ...] = ("gemini", "chatgpt"),
    timeout: int = 30,
) -> tuple[str, ...]:
    """Release persistent adapter tabs owned by one completed logical run.

    OpenCLI's public ``browser close`` command targets the interactive browser
    surface, so it cannot release adapter leases. This small project-local
    bridge command addresses the adapter surface explicitly and runs after the
    multi-command review has finished using its isolated provider sessions.
    """
    if not CLOSE_ADAPTER_SESSIONS_SCRIPT.is_file():
        raise OpenCLIError(
            "OpenCLI adapter-session cleanup script is missing at "
            f"{CLOSE_ADAPTER_SESSIONS_SCRIPT}"
        )

    env = _environment(site_session_namespace=site_session_namespace)
    node = shutil.which("node", path=env.get("PATH"))
    if not node:
        raise OpenCLIError("Node.js is required to close OpenCLI adapter sessions")

    sessions = tuple(
        _site_session_name(site_session_namespace, site)
        for site in dict.fromkeys(sites)
    )
    process = await asyncio.create_subprocess_exec(
        node,
        str(CLOSE_ADAPTER_SESSIONS_SCRIPT),
        *sessions,
        cwd=str(config.PROJECT_ROOT),
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            process.communicate(), timeout=timeout
        )
    except TimeoutError as exc:
        process.kill()
        await process.communicate()
        raise OpenCLIError(
            f"OpenCLI adapter-session cleanup timed out after {timeout}s"
        ) from exc

    stdout = stdout_bytes.decode("utf-8", errors="replace").strip()
    stderr = stderr_bytes.decode("utf-8", errors="replace").strip()
    if process.returncode:
        detail = (stderr or stdout or "unknown cleanup failure")[-1200:]
        raise OpenCLIError(f"OpenCLI adapter-session cleanup failed: {detail}")
    return sessions


def first_json(value: str) -> Any:
    """Recover the first JSON object/array from CLI prose or fenced output."""
    decoder = json.JSONDecoder()
    for index, character in enumerate(value):
        if character not in "[{":
            continue
        try:
            parsed, _ = decoder.raw_decode(value[index:])
        except json.JSONDecodeError:
            continue
        return parsed
    raise OpenCLIError("Command output did not contain valid JSON")


async def browser_bridge_ready() -> bool:
    result = await run_opencli(["doctor"], timeout=30, check=False)
    combined = f"{result.stdout}\n{result.stderr}"
    return result.returncode == 0 and "Everything looks good" in combined

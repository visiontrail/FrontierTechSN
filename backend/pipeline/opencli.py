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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend import config

logger = logging.getLogger(__name__)


class OpenCLIError(RuntimeError):
    """An OpenCLI command could not produce a usable result."""


@dataclass(frozen=True)
class OpenCLIResult:
    args: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


def _environment() -> dict[str, str]:
    env = os.environ.copy()
    path_parts = [
        str(config.PROJECT_ROOT / ".venv" / "bin"),
        str(config.PROJECT_ROOT / "tools" / "opencli" / "node_modules" / ".bin"),
        env.get("PATH", ""),
    ]
    env["PATH"] = os.pathsep.join(part for part in path_parts if part)
    env.setdefault("OPENCLI_BROWSER_CONNECT_TIMEOUT", "20")
    env.setdefault("OPENCLI_BROWSER_COMMAND_TIMEOUT", str(config.OPENCLI_TIMEOUT))
    if config.OPENCLI_PROFILE:
        env["OPENCLI_PROFILE"] = config.OPENCLI_PROFILE
    return env


async def run_opencli(
    args: list[str],
    *,
    timeout: int | None = None,
    check: bool = True,
) -> OpenCLIResult:
    binary = Path(config.OPENCLI_BIN)
    if not binary.is_file():
        raise OpenCLIError(
            f"Project-local OpenCLI wrapper is missing at {binary}. "
            f"Run npm install --prefix {config.PROJECT_ROOT / 'tools' / 'opencli'}"
        )

    command = [str(binary), *[str(arg) for arg in args]]
    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=str(config.PROJECT_ROOT),
        env=_environment(),
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

"""Keep an explicitly foreground browser usable during a bounded operation."""

from __future__ import annotations

import asyncio
import logging
import os
import plistlib
import subprocess
import sys
from contextlib import asynccontextmanager


logger = logging.getLogger(__name__)


class DesktopBrowserLockedError(RuntimeError):
    """The desktop needs the operator to unlock it before browser use."""


def _mac_screen_locked() -> bool:
    result = subprocess.run(
        ["/usr/sbin/ioreg", "-n", "Root", "-d", "1", "-a"],
        capture_output=True, check=True, timeout=3,
    )
    roots = plistlib.loads(result.stdout)
    if isinstance(roots, dict):
        roots = [roots]
    for root in roots:
        for user in root.get("IOConsoleUsers", []):
            if user.get("kCGSSessionOnConsoleKey") is True:
                return user.get("CGSSessionScreenIsLocked") is True
    return False


async def _assert_unlocked() -> None:
    try:
        locked = await asyncio.to_thread(_mac_screen_locked)
    except (OSError, ValueError, subprocess.SubprocessError):
        logger.warning("Could not read macOS screen-lock state; browser readiness remains unverified")
        return
    if locked:
        raise DesktopBrowserLockedError(
            "BROWSER_SCREEN_LOCKED: macOS is locked. Unlock the Mac before "
            "retrying the foreground browser review."
        )


@asynccontextmanager
async def foreground_browser_session(*, enabled: bool):
    """Prevent idle sleep across pacing and browser work; preserve manual locks.

    The assertion belongs only to this call and is also tied to the backend PID
    so a killed backend cannot leave it active. No system preference is changed.
    The yielded check detects a manual lock during a provider pacing wait.
    """
    async def unchecked() -> None:
        return None

    if not enabled or sys.platform != "darwin":
        yield unchecked
        return

    await _assert_unlocked()
    assertion = None
    try:
        try:
            assertion = subprocess.Popen(
                ["/usr/bin/caffeinate", "-d", "-i", "-w", str(os.getpid())],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, start_new_session=True,
            )
        except OSError:
            logger.warning("Could not prevent idle sleep for this foreground browser operation")
        yield _assert_unlocked
    finally:
        if assertion is not None:
            try:
                assertion.terminate()
                assertion.wait(timeout=3)
            except subprocess.TimeoutExpired:
                assertion.kill()
                assertion.wait(timeout=3)
            except ProcessLookupError:
                pass

import asyncio
import plistlib
import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from backend.pipeline import desktop_browser as desktop


@pytest.fixture
def mac(monkeypatch):
    monkeypatch.setattr(desktop.sys, "platform", "darwin")
    assertion = Mock()
    spawn = Mock(return_value=assertion)
    monkeypatch.setattr(desktop.subprocess, "Popen", spawn)
    monkeypatch.setattr(desktop, "_mac_screen_locked", lambda: False)
    return assertion, spawn


@pytest.mark.asyncio
async def test_locked_screen_blocks_before_power_assertion_or_browser_work(mac, monkeypatch):
    monkeypatch.setattr(desktop, "_mac_screen_locked", lambda: True)
    with pytest.raises(desktop.DesktopBrowserLockedError, match="BROWSER_SCREEN_LOCKED"):
        async with desktop.foreground_browser_session(enabled=True):
            pytest.fail("Locked desktop must not start browser work")
    mac[1].assert_not_called()


@pytest.mark.asyncio
async def test_assertion_covers_wait_and_detects_manual_lock(mac, monkeypatch):
    assertion, spawn = mac
    async with desktop.foreground_browser_session(enabled=True) as check:
        command = spawn.call_args.args[0]
        assert command[:4] == ["/usr/bin/caffeinate", "-d", "-i", "-w"]
        assert "-u" not in command
        assertion.terminate.assert_not_called()
        monkeypatch.setattr(desktop, "_mac_screen_locked", lambda: True)
        with pytest.raises(desktop.DesktopBrowserLockedError):
            await check()
    assertion.terminate.assert_called_once()
    assertion.wait.assert_called_once_with(timeout=3)


@pytest.mark.asyncio
async def test_cancellation_releases_owned_assertion(mac):
    ready = asyncio.Event()

    async def work():
        async with desktop.foreground_browser_session(enabled=True):
            ready.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(work())
    await ready.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    mac[0].terminate.assert_called_once()
    mac[0].wait.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled,platform", [(False, "darwin"), (True, "linux")])
async def test_headless_or_other_platform_never_touches_desktop(mac, monkeypatch, enabled, platform):
    monkeypatch.setattr(desktop.sys, "platform", platform)
    probe = Mock(side_effect=AssertionError("Must not probe desktop"))
    monkeypatch.setattr(desktop, "_mac_screen_locked", probe)
    async with desktop.foreground_browser_session(enabled=enabled) as check:
        await check()
    probe.assert_not_called()
    mac[1].assert_not_called()


@pytest.mark.asyncio
async def test_failed_optional_power_assertion_does_not_change_lock_check(mac, monkeypatch):
    mac[1].side_effect = OSError("unavailable")
    async with desktop.foreground_browser_session(enabled=True) as check:
        monkeypatch.setattr(desktop, "_mac_screen_locked", lambda: True)
        with pytest.raises(desktop.DesktopBrowserLockedError):
            await check()


@pytest.mark.asyncio
async def test_unavailable_lock_probe_is_not_misreported_as_locked(mac, monkeypatch):
    monkeypatch.setattr(desktop, "_mac_screen_locked", Mock(side_effect=subprocess.TimeoutExpired("ioreg", 3)))
    async with desktop.foreground_browser_session(enabled=True) as check:
        await check()
    mac[0].terminate.assert_called_once()


@pytest.mark.parametrize("wrapped", [False, True])
def test_lock_probe_uses_only_active_console_session(monkeypatch, wrapped):
    root = {"IOConsoleUsers": [
        {"kCGSSessionOnConsoleKey": False, "CGSSessionScreenIsLocked": True},
        {"kCGSSessionOnConsoleKey": True, "CGSSessionScreenIsLocked": False},
    ]}
    body = plistlib.dumps([root] if wrapped else root)
    run = Mock(return_value=SimpleNamespace(stdout=body))
    monkeypatch.setattr(desktop.subprocess, "run", run)
    assert desktop._mac_screen_locked() is False
    assert run.call_args.kwargs["timeout"] == 3


@pytest.mark.asyncio
async def test_opencli_checks_foreground_desktop_before_reserving_provider_slot(monkeypatch):
    from backend.pipeline import opencli

    monkeypatch.setattr(opencli, "runtime_mode", lambda: "bridge")
    slot = Mock(side_effect=AssertionError("Must not reserve a provider request while locked"))
    monkeypatch.setattr(opencli, "wait_for_opencli_web_slot", slot)
    monkeypatch.setattr(desktop.sys, "platform", "darwin")
    monkeypatch.setattr(desktop, "_mac_screen_locked", lambda: True)
    with pytest.raises(opencli.OpenCLIError, match="BROWSER_SCREEN_LOCKED"):
        await opencli.run_opencli(["gemini", "ask", "review", "--window", "foreground"])
    slot.assert_not_called()


@pytest.mark.asyncio
async def test_manual_lock_during_pacing_prevents_cli_launch(mac, monkeypatch):
    from backend.pipeline import opencli

    locked = False

    def reserve(*args, **kwargs):
        nonlocal locked
        mac[0].terminate.assert_not_called()
        locked = True

    launch = AsyncMock()
    monkeypatch.setattr(opencli, "runtime_mode", lambda: "bridge")
    monkeypatch.setattr(opencli, "_wait_for_provider_cooldown", AsyncMock())
    monkeypatch.setattr(opencli, "wait_for_opencli_web_slot", reserve)
    monkeypatch.setattr(opencli.asyncio, "create_subprocess_exec", launch)
    monkeypatch.setattr(desktop, "_mac_screen_locked", lambda: locked)
    with pytest.raises(opencli.OpenCLIError, match="BROWSER_SCREEN_LOCKED"):
        await opencli.run_opencli(["gemini", "ask", "review", "--window", "foreground"])
    launch.assert_not_called()
    mac[0].terminate.assert_called_once()

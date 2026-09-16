import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from backend.pipeline import opencli_session as sessions


@pytest.mark.parametrize("outcome", ["success", "failure", "cancel"])
def test_owned_session_closes_on_every_exit_and_records_verification(tmp_path, monkeypatch, outcome):
    close = AsyncMock()
    monkeypatch.setattr(sessions, "close_opencli_site_sessions", close)

    async def run():
        async with sessions.owned_media_session(tmp_path, "gemini") as namespace:
            assert namespace != "frontiertechsn"
            assert json.loads((tmp_path / "browser-session-gemini.json").read_text())["status"] == "active"
            if outcome == "failure":
                raise ValueError("provider failed")
            if outcome == "cancel":
                asyncio.current_task().cancel()
                await asyncio.sleep(0)

    if outcome == "success":
        asyncio.run(run())
    else:
        with pytest.raises(ValueError if outcome == "failure" else asyncio.CancelledError):
            asyncio.run(run())
    close.assert_awaited_once_with(sessions._namespace(tmp_path, "gemini"), sites=("gemini",))
    assert json.loads((tmp_path / "browser-session-gemini.json").read_text())["verified"] is True


def test_recovery_closes_only_derived_owner_and_rejects_live_owner(tmp_path, monkeypatch):
    close = AsyncMock()
    monkeypatch.setattr(sessions, "close_opencli_site_sessions", close)
    owner = tmp_path / "scene-01"
    owner.mkdir()
    journal = owner / "browser-session-gemini.json"
    journal.write_text(json.dumps({"namespace": "another-task", "status": "active"}))

    async def run():
        await sessions.recover_media_sessions(tmp_path)
        assert close.await_args.args == (sessions._namespace(owner, "gemini"),)
        async with sessions.owned_media_session(owner, "gemini"):
            with pytest.raises(sessions.OpenCLIError, match="active operation"):
                await sessions.recover_media_sessions(tmp_path)
            assert close.await_count == 1
    asyncio.run(run())
    assert close.await_count == 2


def test_cleanup_failure_is_durable_and_recovered_without_generation(tmp_path, monkeypatch):
    owner = tmp_path / "scene-01"
    owner.mkdir()
    close = AsyncMock(side_effect=RuntimeError("bridge disconnected"))
    monkeypatch.setattr(sessions, "close_opencli_site_sessions", close)
    monkeypatch.setattr(sessions.asyncio, "sleep", AsyncMock())

    async def run():
        async with sessions.owned_media_session(owner, "gemini"):
            pass
        assert json.loads((owner / "browser-session-gemini.json").read_text())["status"] == "cleanup_failed"
        close.side_effect = None
        await sessions.recover_media_sessions(tmp_path)
    asyncio.run(run())
    assert close.await_count == 3
    assert json.loads((owner / "browser-session-gemini.json").read_text())["status"] == "closed"


def test_namespaces_are_distinct_and_survive_runtime_namespace_changes(tmp_path):
    assert sessions._namespace(tmp_path / "task-a", "gemini") != sessions._namespace(tmp_path / "task-b", "gemini")
    assert len(sessions._namespace(tmp_path, "gemini")) <= 64


def test_second_cancellation_waits_for_cleanup(tmp_path, monkeypatch):
    async def run():
        started = asyncio.Event()
        finish = asyncio.Event()

        async def close(*args, **kwargs):
            started.set()
            await finish.wait()

        monkeypatch.setattr(sessions, "close_opencli_site_sessions", close)

        async def operation():
            async with sessions.owned_media_session(tmp_path, "gemini"):
                raise asyncio.CancelledError

        task = asyncio.create_task(operation())
        await started.wait()
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    asyncio.run(run())
    assert json.loads((tmp_path / "browser-session-gemini.json").read_text())["verified"] is True


def test_startup_recovers_unresumed_tasks_and_skips_symlinked_task_roots(tmp_path, monkeypatch):
    close = AsyncMock()
    monkeypatch.setattr(sessions, "close_opencli_site_sessions", close)
    owner = tmp_path / "finished-task" / "collage_broll" / "01-scene"
    owner.mkdir(parents=True)
    (owner / "browser-session-gemini.json").write_text('{"status":"active"}')
    (tmp_path / "alias").symlink_to(tmp_path / "finished-task", target_is_directory=True)
    asyncio.run(sessions.recover_orphaned_media_sessions(tmp_path))
    close.assert_awaited_once_with(sessions._namespace(owner, "gemini"), sites=("gemini",))


def test_real_subprocess_timeout_reaps_process_before_session_cleanup(tmp_path, monkeypatch):
    import sys
    from backend import config
    from backend.pipeline.opencli import run_opencli

    wrapper = tmp_path / "slow-wrapper"
    wrapper.write_text(f"#!{sys.executable}\nimport time\ntime.sleep(30)\n")
    wrapper.chmod(0o755)
    monkeypatch.setattr(config, "OPENCLI_BIN", str(wrapper))
    close = AsyncMock()
    monkeypatch.setattr(sessions, "close_opencli_site_sessions", close)

    async def run():
        async with sessions.owned_media_session(tmp_path, "gemini") as namespace:
            await run_opencli(["test"], timeout=1, site_session_namespace=namespace)

    with pytest.raises(sessions.OpenCLIError, match="timed out"):
        asyncio.run(run())
    close.assert_awaited_once()
    assert json.loads((tmp_path / "browser-session-gemini.json").read_text())["verified"]

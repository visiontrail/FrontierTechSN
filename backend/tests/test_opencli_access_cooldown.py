import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from backend.pipeline import opencli
from backend.pipeline import opencli_rate_limit as pacing


def test_cooldown_persists_and_escalates_across_calls(tmp_path):
    state = tmp_path / 'pacing'
    clock = [1000.0]
    now = lambda: clock[0]
    until = pacing.record_opencli_rate_limit('chatgpt', state_path=state, clock=now)
    assert until == 2800
    assert pacing.opencli_cooldown_remaining('chatgpt', state_path=state, clock=now) == 1800
    assert pacing.opencli_cooldown_remaining('gemini', state_path=state, clock=now) == 0
    clock[0] = 2801
    assert pacing.opencli_cooldown_remaining('chatgpt', state_path=state, clock=now) == 0
    assert pacing.record_opencli_rate_limit('chatgpt', state_path=state, clock=now) == 6401
    for _ in range(5):
        until = pacing.record_opencli_rate_limit('chatgpt', state_path=state, clock=now)
    assert until == 10001  # bounded at two hours


def test_cooldown_still_escalates_after_a_one_hour_provider_window(tmp_path):
    state = tmp_path / 'pacing'
    clock = [1000.0]
    now = lambda: clock[0]
    assert pacing.record_opencli_rate_limit(
        'chatgpt', state_path=state, clock=now,
    ) == 2800
    clock[0] = 5000
    assert pacing.record_opencli_rate_limit(
        'chatgpt', state_path=state, clock=now,
    ) == 8600


def test_cooldown_wait_is_cancellable_before_browser_launch():
    async def check():
        sleeping = asyncio.Event()

        async def sleep(_):
            sleeping.set()
            await asyncio.Future()

        with (
            patch.object(opencli, 'opencli_cooldown_remaining', return_value=300),
            patch.object(opencli.asyncio, 'sleep', side_effect=sleep),
            patch.object(opencli.asyncio, 'create_subprocess_exec', AsyncMock()) as launch,
        ):
            task = asyncio.create_task(opencli.run_opencli(['chatgpt', 'detail', 'owned']))
            await sleeping.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            launch.assert_not_awaited()
    asyncio.run(check())


@pytest.mark.parametrize('action', ['ask', 'detail', 'read', 'model'])
def test_all_chatgpt_actions_wait_outside_command_timeout(action):
    remaining = iter([60, 30, 0, 0])
    process = AsyncMock()
    process.returncode = 0
    process.communicate.return_value = (b'[]', b'')
    with (
        patch.object(opencli, 'opencli_cooldown_remaining', side_effect=lambda _: next(remaining)),
        patch.object(opencli, 'wait_for_opencli_web_slot'),
        patch.object(opencli, 'wait_for_opencli_generation_quiet_period'),
        patch.object(opencli.asyncio, 'sleep', AsyncMock()) as sleep,
        patch.object(opencli.asyncio, 'create_subprocess_exec', AsyncMock(return_value=process)) as launch,
    ):
        asyncio.run(opencli.run_opencli(['chatgpt', action], timeout=1))
    assert [call.args[0] for call in sleep.await_args_list] == [30, 30]
    launch.assert_awaited_once()


def test_explicit_adapter_limit_opens_breaker_even_with_check_false():
    process = AsyncMock()
    process.returncode = 1
    process.communicate.return_value = (b'', b'CHATGPT_RATE_LIMITED Target: https://chatgpt.com/c/owned')
    with (
        patch.object(opencli, 'opencli_cooldown_remaining', return_value=0),
        patch.object(opencli, 'record_opencli_rate_limit', return_value=1300) as record,
        patch.object(opencli.asyncio, 'create_subprocess_exec', AsyncMock(return_value=process)),
        pytest.raises(opencli.OpenCLIRateLimitError, match='https://chatgpt.com/c/owned'),
    ):
        asyncio.run(opencli.run_opencli(['chatgpt', 'read'], check=False))
    record.assert_called_once_with('chatgpt')

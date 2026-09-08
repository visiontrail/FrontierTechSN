import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest

from backend.pipeline import opencli
from backend.pipeline import opencli_rate_limit as pacing


def test_cooldown_persists_and_increments_by_ten_minutes(tmp_path):
    state = tmp_path / 'pacing'
    clock = [1000.0]
    now = lambda: clock[0]
    until = pacing.record_opencli_rate_limit('chatgpt', state_path=state, clock=now)
    assert until == 1600
    assert pacing.opencli_cooldown_remaining('chatgpt', state_path=state, clock=now) == 600
    assert pacing.opencli_cooldown_remaining('gemini', state_path=state, clock=now) == 0
    clock[0] = 1601
    assert pacing.opencli_cooldown_remaining('chatgpt', state_path=state, clock=now) == 0
    assert pacing.record_opencli_rate_limit('chatgpt', state_path=state, clock=now) == 2801
    assert pacing.record_opencli_rate_limit('chatgpt', state_path=state, clock=now) == 3401


def test_cooldown_caps_at_one_hour(tmp_path):
    state = tmp_path / 'pacing'
    clock = [1000.0]
    now = lambda: clock[0]
    delays = [
        pacing.record_opencli_rate_limit(
            'chatgpt', state_path=state, clock=now,
        ) - clock[0]
        for _ in range(7)
    ]
    assert delays == [600, 1200, 1800, 2400, 3000, 3600, 3600]
    assert pacing.opencli_cooldown_remaining('chatgpt', state_path=state, clock=now) == 3600


def test_cooldown_still_escalates_after_a_one_hour_provider_window(tmp_path):
    state = tmp_path / 'pacing'
    clock = [1000.0]
    now = lambda: clock[0]
    assert pacing.record_opencli_rate_limit(
        'chatgpt', state_path=state, clock=now,
    ) == 1600
    clock[0] = 5000
    assert pacing.record_opencli_rate_limit(
        'chatgpt', state_path=state, clock=now,
    ) == 6200


def test_cooldown_resets_after_six_hours_without_another_limit(tmp_path):
    state = tmp_path / 'pacing'
    clock = [1000.0]
    now = lambda: clock[0]
    assert pacing.record_opencli_rate_limit(
        'chatgpt', state_path=state, clock=now,
    ) == 1600
    clock[0] = 1601
    assert pacing.record_opencli_rate_limit(
        'chatgpt', state_path=state, clock=now,
    ) == 2801
    clock[0] += pacing.PROVIDER_COOLDOWN_ESCALATION_WINDOW_SECONDS + 1
    assert pacing.record_opencli_rate_limit(
        'chatgpt', state_path=state, clock=now,
    ) == clock[0] + 600


def test_existing_two_hour_cooldown_is_clamped_to_new_one_hour_cap(tmp_path):
    state = tmp_path / 'pacing'
    cooldown = tmp_path / 'pacing.chatgpt.cooldown'
    cooldown.write_text(json.dumps({
        'recorded_at': 1000,
        'until': 8200,
        'delay': 7200,
    }))
    clock = [2000.0]
    now = lambda: clock[0]
    assert pacing.opencli_cooldown_remaining(
        'chatgpt', state_path=state, clock=now,
    ) == 2600
    clock[0] = 4601
    assert pacing.opencli_cooldown_remaining(
        'chatgpt', state_path=state, clock=now,
    ) == 0


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

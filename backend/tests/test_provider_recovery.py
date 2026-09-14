from dataclasses import replace
from unittest.mock import AsyncMock, patch

import pytest

from backend.pipeline import agent, model_router


def routes():
    common = dict(provider_id=None, provider_type='test', model='model')
    return (
        model_router.ModelRoute(slot='primary', provider_name='Primary',
            endpoint='https://primary.example', api_key='primary-a',
            api_keys=('primary-a', 'primary-b'), **common),
        model_router.ModelRoute(slot='backup', provider_name='Backup',
            endpoint='https://backup.example', api_key='backup', **common),
    )


@pytest.mark.asyncio
async def test_recent_exhaustion_reorders_routes_without_reducing_retries_or_extending_window():
    primary, backup = routes()
    calls = []
    primary_available = False

    async def complete(*args, route, max_retries, **kwargs):
        calls.append((route.slot, max_retries))
        if route.slot == 'primary' and not primary_available:
            raise RuntimeError('The request queue is full')
        return route.slot, route

    with (
        patch.object(model_router, 'resolve_model_routes', AsyncMock(return_value=(primary, backup))),
        patch.object(agent, '_agent_complete_single', AsyncMock(side_effect=complete)),
        patch.object(agent.time, 'monotonic', return_value=1000) as clock,
    ):
        assert await agent.agent_complete('system', 'first', max_retries=3) == 'backup'
        assert calls == [('primary', 3), ('backup', 3)]
        calls.clear()
        clock.return_value = 1100
        assert await agent.agent_complete('system', 'next', max_retries=3) == 'backup'
        assert calls == [('backup', 3)]
        calls.clear()
        clock.return_value = 1301
        primary_available = True
        assert await agent.agent_complete('system', 'recovered', max_retries=3) == 'primary'
        assert calls == [('primary', 3)]


@pytest.mark.asyncio
async def test_backup_failure_still_tries_primary_during_recovery_window():
    primary, backup = routes()
    calls = []
    agent._recovery_state()[agent._recovery_key(primary)] = 1300

    async def complete(*args, route, **kwargs):
        calls.append(route.slot)
        if route.slot == 'backup':
            raise RuntimeError('Backup failed')
        return 'primary success', route

    with (
        patch.object(model_router, 'resolve_model_routes', AsyncMock(return_value=(primary, backup))),
        patch.object(agent, '_agent_complete_single', AsyncMock(side_effect=complete)),
        patch.object(agent.time, 'monotonic', return_value=1100),
    ):
        assert await agent.agent_complete('system', 'request') == 'primary success'
    assert calls == ['backup', 'primary']
    assert agent._recovery_key(primary) not in agent._recovery_state()


@pytest.mark.asyncio
@pytest.mark.parametrize('change', ['pool_rotation', 'credentials', 'model', 'endpoint', 'standalone'])
async def test_recovery_identity_and_explicit_single_route(change):
    primary, backup = routes()
    agent._recovery_state()[agent._recovery_key(primary)] = 1300
    selected = {
        'pool_rotation': replace(primary, api_key_index=1),
        'credentials': replace(primary, api_keys=('replacement',)),
        'model': replace(primary, model='new-model'),
        'endpoint': replace(primary, endpoint='https://new.example'),
        'standalone': primary,
    }[change]

    async def complete(*args, route, **kwargs):
        return route.slot, route

    with (
        patch.object(model_router, 'resolve_model_routes', AsyncMock(return_value=(selected, backup))),
        patch.object(agent, '_agent_complete_single', AsyncMock(side_effect=complete)),
        patch.object(agent.time, 'monotonic', return_value=1100),
    ):
        options = {'route': selected} if change == 'standalone' else {}
        result = await agent.agent_complete('system', 'request', **options)
    assert result == ('backup' if change == 'pool_rotation' else 'primary')


@pytest.mark.asyncio
@pytest.mark.parametrize('error', [RuntimeError('Invalid JSON'), model_router.ModelOutputCommittedError('Gateway Time-out')])
async def test_content_failures_do_not_create_provider_health_shortcuts(error):
    primary, backup = routes()

    async def complete(*args, route, **kwargs):
        if route.slot == 'primary':
            raise error
        return 'backup', route

    with (
        patch.object(model_router, 'resolve_model_routes', AsyncMock(return_value=(primary, backup))),
        patch.object(agent, '_agent_complete_single', AsyncMock(side_effect=complete)),
    ):
        if isinstance(error, model_router.ModelOutputCommittedError):
            with pytest.raises(model_router.ModelOutputCommittedError):
                await agent.agent_complete('system', 'request')
        else:
            assert await agent.agent_complete('system', 'request') == 'backup'
    assert agent._recovery_state().get(agent._recovery_key(primary)) is None

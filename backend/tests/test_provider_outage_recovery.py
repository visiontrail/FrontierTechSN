"""A successful backup should prevent repeated probes of a known outage."""

import unittest
from unittest.mock import AsyncMock, patch

from backend import config
from backend.pipeline import agent, digester, model_router


class ProviderOutageRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_upstream_500_uses_successful_backup_until_recovery_expires(self):
        routes = (
            model_router.ModelRoute(
                slot="primary", provider_id=1, provider_type="yinhe",
                provider_name="primary", endpoint="https://primary.example",
                model="reasoning", api_key="primary-test",
            ),
            model_router.ModelRoute(
                slot="backup", provider_id=2, provider_type="deepseek",
                provider_name="backup", endpoint="https://backup.example",
                model="backup", api_key="backup-test",
            ),
        )
        seen = []
        primary_failed = False

        async def complete(*args, route, **kwargs):
            nonlocal primary_failed
            seen.append(route.slot)
            if route.slot == "primary" and not primary_failed:
                primary_failed = True
                raise RuntimeError('500 upstream error: do request failed | assistant error: server_error')
            return route.slot + " answer"

        with (
            patch.object(config, "AI_BACKEND", "agent_sdk"),
            patch.object(model_router, "resolve_model_routes", AsyncMock(return_value=routes)),
            patch.object(agent, "agent_complete", complete),
            patch.object(agent.time, "monotonic", return_value=100.0) as now,
        ):
            self.assertEqual(await digester._chat("system", "first"), "backup answer")
            self.assertEqual(await digester._chat("system", "second"), "backup answer")
            now.return_value += agent.PROVIDER_RECOVERY_SECONDS + 1
            self.assertEqual(await digester._chat("system", "third"), "primary answer")

        self.assertEqual(seen, ["primary", "backup", "backup", "primary"])

    def test_budget_and_validation_errors_are_not_provider_outages(self):
        for detail in (
            "Claude Agent SDK response truncated at max_tokens",
            "JSONDecodeError: expecting delimiter",
            "HTTP 400 Bad Request: unsupported model",
            "HTTP 401 Unauthorized",
        ):
            with self.subTest(detail=detail):
                self.assertFalse(agent._provider_outage(RuntimeError(detail)))

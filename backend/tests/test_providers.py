import unittest
from unittest.mock import AsyncMock, patch

from backend.models import ProviderResponse, ProviderTestRequest, ProviderUpdate
from backend.routers import providers


class ProviderRouterTests(unittest.IsolatedAsyncioTestCase):
    async def test_primary_provider_test_checks_every_stored_key(self):
        row = {
            "endpoint": "http://primary.example/v1/chat/completions",
            "model": "yinhe-thinking",
            "api_key": "key-a",
            "api_keys_json": '["key-a", "key-b", "key-c"]',
            "route_role": "primary",
        }
        with (
            patch.object(
                providers.database,
                "get_provider_raw",
                AsyncMock(return_value=row),
            ),
            patch.object(
                providers.digester,
                "test_connection",
                AsyncMock(side_effect=[10, 20, 30]),
            ) as test_connection,
        ):
            result = await providers.test_provider(ProviderTestRequest(provider_id=1))

        self.assertTrue(result.ok)
        self.assertEqual(result.keys_tested, 3)
        self.assertEqual(result.keys_succeeded, 3)
        self.assertEqual(test_connection.await_count, 3)
        self.assertEqual(
            [call.args[2] for call in test_connection.await_args_list],
            ["key-a", "key-b", "key-c"],
        )
        rendered = result.model_dump_json()
        self.assertNotIn('"key-a"', rendered)
        self.assertNotIn('"key-b"', rendered)
        self.assertNotIn('"key-c"', rendered)

    async def test_blank_api_key_on_edit_keeps_the_stored_secret(self):
        captured = {}

        async def fake_update_provider(provider_id: int, **updates):
            captured["provider_id"] = provider_id
            captured["updates"] = updates
            return ProviderResponse(
                id=provider_id,
                provider_type="deepseek",
                name="DeepSeek production",
                endpoint="https://api.deepseek.com/anthropic",
                api_key_masked="secr...-key",
                model="deepseek-v4-flash",
                is_default=False,
                created_at="2026-08-25T00:00:00+00:00",
            )

        with patch.object(providers.database, "update_provider", fake_update_provider):
            response = await providers.edit_provider(
                7,
                ProviderUpdate(
                    provider_type="deepseek",
                    model="deepseek-v4-flash",
                    api_key="",
                ),
            )

        self.assertEqual(response.id, 7)
        self.assertEqual(
            captured,
            {
                "provider_id": 7,
                "updates": {
                    "provider_type": "deepseek",
                    "model": "deepseek-v4-flash",
                },
            },
        )

    async def test_route_probe_reports_the_actual_backup_without_exposing_keys(self):
        primary = {
            "id": 1,
            "provider_type": "yinhe",
            "name": "Yinhe OneAPI",
            "endpoint": "http://primary.example/v1/chat/completions",
            "model": "yinhe-thinking",
            "api_key": "primary-secret",
            "api_keys_json": '["primary-secret", "primary-secret-b"]',
            "route_role": "primary",
        }
        backup = {
            "id": 2,
            "provider_type": "deepseek",
            "name": "DeepSeek backup",
            "endpoint": "http://backup.example/anthropic",
            "model": "deepseek-v4-flash",
            "api_key": "backup-secret",
            "api_keys_json": "[]",
            "route_role": "backup",
        }

        async def fake_chat(*args, **kwargs):
            kwargs["log"]("Model route test: switching to DeepSeek backup")
            kwargs["route_selected"](
                backup["endpoint"],
                backup["model"],
                backup["api_key"],
            )
            return "pong"

        with (
            patch.object(
                providers.database,
                "get_provider_raw",
                AsyncMock(return_value=primary),
            ),
            patch.object(
                providers.database,
                "list_providers_raw",
                AsyncMock(return_value=[primary, backup]),
            ),
            patch.object(providers.digester, "_chat", fake_chat),
        ):
            result = await providers.test_configured_route()

        self.assertTrue(result.ok)
        self.assertEqual(result.route_role, "backup")
        self.assertTrue(result.fallback_used)
        self.assertEqual(result.provider_name, "DeepSeek backup")
        rendered = result.model_dump_json()
        self.assertNotIn("primary-secret", rendered)
        self.assertNotIn("backup-secret", rendered)

    async def test_route_probe_redacts_configured_secrets_from_failures(self):
        primary = {
            "id": 1,
            "provider_type": "yinhe",
            "name": "Yinhe OneAPI",
            "endpoint": "http://primary.example/v1/chat/completions",
            "model": "yinhe-thinking",
            "api_key": "primary-secret",
            "api_keys_json": '["primary-secret"]',
            "route_role": "primary",
        }

        async def fake_chat(*args, **kwargs):
            kwargs["log"]("Model route test failed with primary-secret")
            raise RuntimeError("upstream echoed primary-secret")

        with (
            patch.object(
                providers.database,
                "get_provider_raw",
                AsyncMock(return_value=primary),
            ),
            patch.object(
                providers.database,
                "list_providers_raw",
                AsyncMock(return_value=[primary]),
            ),
            patch.object(providers.digester, "_chat", fake_chat),
        ):
            result = await providers.test_configured_route()

        self.assertFalse(result.ok)
        self.assertIn("<redacted>", result.message)
        self.assertNotIn("primary-secret", result.model_dump_json())

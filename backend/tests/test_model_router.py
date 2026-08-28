import unittest
from unittest.mock import AsyncMock, patch

from backend import config, database
from backend.pipeline import model_router


def provider_row(
    provider_id: int,
    provider_type: str,
    name: str,
    endpoint: str,
    model: str,
    api_key: str,
):
    return {
        "id": provider_id,
        "provider_type": provider_type,
        "name": name,
        "endpoint": endpoint,
        "model": model,
        "api_key": api_key,
    }


class ModelRouterTests(unittest.IsolatedAsyncioTestCase):
    def test_route_specific_retry_limits_keep_primary_and_backup_separate(self):
        primary = model_router.ModelRoute(
            slot="primary",
            provider_id=1,
            provider_type="yinhe",
            provider_name="OneAPI",
            endpoint="http://oneapi.example",
            model="yinhe-thinking",
            api_key="primary-secret",
        )
        backup = model_router.ModelRoute(
            slot="backup",
            provider_id=2,
            provider_type="deepseek",
            provider_name="DeepSeek",
            endpoint="https://api.deepseek.com/anthropic",
            model="deepseek-v4-flash",
            api_key="backup-secret",
        )
        with (
            patch.object(config, "AI_PRIMARY_MAX_RETRIES", 3),
            patch.object(config, "AI_BACKUP_MAX_RETRIES", 2),
        ):
            self.assertEqual(model_router.max_retries_for_route(primary), 3)
            self.assertEqual(model_router.max_retries_for_route(backup), 2)

    async def test_oneapi_route_resolves_deepseek_backup_in_order(self):
        rows = [
            provider_row(
                1,
                "yinhe",
                "OneAPI",
                "http://oneapi.yhroot.com/v1/chat/completions",
                "yinhe-thinking",
                "primary-secret",
            ),
            provider_row(
                2,
                "deepseek",
                "DeepSeek",
                "https://api.deepseek.com/anthropic",
                "deepseek-v4-flash",
                "backup-secret",
            ),
        ]
        with (
            patch.object(config, "AI_PROVIDER_FAILOVER_ENABLED", True),
            patch.object(config, "AI_PRIMARY_PROVIDER_TYPE", "yinhe"),
            patch.object(config, "AI_BACKUP_PROVIDER_TYPE", "deepseek"),
            patch.object(config, "ANTHROPIC_BASE_URL", ""),
            patch.object(config, "ANTHROPIC_MODEL", ""),
            patch.object(database, "list_providers_raw", AsyncMock(return_value=rows)),
        ):
            routes = await model_router.resolve_model_routes(
                endpoint="http://oneapi.yhroot.com/v1/chat/completions",
                model="yinhe-thinking",
                api_key="primary-secret",
            )

        self.assertEqual([route.slot for route in routes], ["primary", "backup"])
        self.assertEqual([route.provider_type for route in routes], ["yinhe", "deepseek"])
        self.assertEqual(routes[1].model, "deepseek-v4-flash")
        self.assertNotIn("primary-secret", repr(routes[0]))
        self.assertNotIn("backup-secret", repr(routes[1]))

    async def test_explicit_deepseek_selection_never_routes_back_to_oneapi(self):
        with patch.object(database, "list_providers_raw", AsyncMock()) as list_rows:
            routes = await model_router.resolve_model_routes(
                endpoint="https://api.deepseek.com/anthropic",
                model="deepseek-v4-flash",
                api_key="backup-secret",
            )

        self.assertEqual(len(routes), 1)
        self.assertEqual(routes[0].endpoint, "https://api.deepseek.com/anthropic")
        list_rows.assert_not_awaited()

    async def test_unsaved_custom_endpoint_never_uses_saved_backup(self):
        with patch.object(database, "list_providers_raw", AsyncMock()) as list_rows:
            routes = await model_router.resolve_model_routes(
                endpoint="https://custom.example/anthropic",
                model="custom-model",
                api_key="custom-secret",
            )

        self.assertEqual(len(routes), 1)
        list_rows.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()

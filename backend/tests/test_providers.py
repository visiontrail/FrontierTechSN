import unittest
from unittest.mock import patch

from backend.models import ProviderResponse, ProviderUpdate
from backend.routers import providers


class ProviderRouterTests(unittest.IsolatedAsyncioTestCase):
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

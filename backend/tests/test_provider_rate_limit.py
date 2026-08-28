import unittest
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from unittest.mock import AsyncMock, patch

import httpx

from backend import config
from backend.pipeline import digester, provider_rate_limit


class ProviderRateLimitTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        provider_rate_limit.reset_for_tests()

    async def test_oneapi_request_starts_are_globally_spaced(self):
        clock = [100.0]
        sleeps: list[float] = []

        async def advance(delay: float):
            sleeps.append(delay)
            clock[0] += delay

        with (
            patch.object(config, "AI_PRIMARY_PROVIDER_TYPE", "yinhe"),
            patch.object(config, "AI_PRIMARY_MIN_REQUEST_INTERVAL_SECONDS", 15.0),
            patch.object(provider_rate_limit, "_proactive_limit_active", return_value=True),
            patch.object(provider_rate_limit.time, "monotonic", side_effect=lambda: clock[0]),
            patch.object(provider_rate_limit.asyncio, "sleep", side_effect=advance),
        ):
            first = await provider_rate_limit.wait_for_request_slot(
                "http://oneapi.yhroot.com/v1/chat/completions"
            )
            second = await provider_rate_limit.wait_for_request_slot(
                "http://oneapi.yhroot.com"
            )

        self.assertEqual(first, 0.0)
        self.assertEqual(second, 15.0)
        self.assertEqual(sleeps, [15.0])

    async def test_backup_and_custom_endpoints_are_not_oneapi_throttled(self):
        with patch.object(provider_rate_limit.asyncio, "sleep", AsyncMock()) as sleep:
            waited = await provider_rate_limit.wait_for_request_slot(
                "https://api.deepseek.com/anthropic"
            )

        self.assertEqual(waited, 0.0)
        sleep.assert_not_awaited()

    async def test_oneapi_429_opens_a_full_shared_cooldown(self):
        clock = [100.0]
        sleeps: list[float] = []
        logs: list[str] = []

        async def advance(delay: float):
            sleeps.append(delay)
            clock[0] += delay

        with (
            patch.object(config, "AI_PRIMARY_PROVIDER_TYPE", "yinhe"),
            patch.object(config, "AI_PRIMARY_MIN_REQUEST_INTERVAL_SECONDS", 15.0),
            patch.object(config, "AI_PRIMARY_RATE_LIMIT_COOLDOWN_SECONDS", 65.0),
            patch.object(provider_rate_limit, "_proactive_limit_active", return_value=True),
            patch.object(provider_rate_limit.time, "monotonic", side_effect=lambda: clock[0]),
            patch.object(provider_rate_limit.asyncio, "sleep", side_effect=advance),
        ):
            cooldown = await provider_rate_limit.record_rate_limit(
                "http://oneapi.yhroot.com/v1/chat/completions",
                log=logs.append,
                label="Duration edit",
            )
            waited = await provider_rate_limit.wait_for_request_slot(
                "http://oneapi.yhroot.com",
                log=logs.append,
                label="Duration edit",
            )

        self.assertEqual(cooldown, 65.0)
        self.assertEqual(waited, 65.0)
        self.assertEqual(sleeps, [65.0])
        self.assertTrue(any("rolling one-minute quota" in line for line in logs))

    async def test_night_and_weekend_skip_proactive_spacing(self):
        # 15:00 UTC is 23:00 in Asia/Singapore on this Friday.
        friday_night = datetime(2026, 8, 28, 15, tzinfo=timezone.utc)
        saturday_day = datetime(2026, 8, 29, 3, tzinfo=timezone.utc)
        with (
            patch.object(config, "AI_PRIMARY_RATE_LIMIT_TIMEZONE", "Asia/Singapore"),
            patch.object(config, "AI_PRIMARY_RATE_LIMIT_START_HOUR", 8),
            patch.object(config, "AI_PRIMARY_RATE_LIMIT_END_HOUR", 20),
            patch.object(provider_rate_limit.asyncio, "sleep", AsyncMock()) as sleep,
        ):
            night_wait = await provider_rate_limit.wait_for_request_slot(
                "http://oneapi.yhroot.com",
                now=friday_night,
            )
            weekend_wait = await provider_rate_limit.wait_for_request_slot(
                "http://oneapi.yhroot.com",
                now=saturday_day,
            )

        self.assertEqual(night_wait, 0.0)
        self.assertEqual(weekend_wait, 0.0)
        sleep.assert_not_awaited()


class RetryAfterTests(unittest.TestCase):
    def test_numeric_retry_after_is_honored_and_bounded(self):
        response = httpx.Response(429, headers={"Retry-After": "75"})
        self.assertEqual(digester._retry_after_seconds(response), 75.0)

        response = httpx.Response(429, headers={"Retry-After": "99999"})
        self.assertEqual(digester._retry_after_seconds(response), 3600.0)

    def test_http_date_retry_after_is_supported(self):
        response = httpx.Response(
            429,
            headers={
                "Retry-After": format_datetime(
                    datetime.now(timezone.utc) + timedelta(seconds=90),
                    usegmt=True,
                )
            },
        )
        delay = digester._retry_after_seconds(response)
        self.assertGreater(delay, 85.0)
        self.assertLessEqual(delay, 90.0)


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import AsyncMock, patch

from backend import config
from backend.pipeline import digester, model_router


def routes() -> tuple[model_router.ModelRoute, model_router.ModelRoute]:
    return (
        model_router.ModelRoute(
            slot="primary",
            provider_id=1,
            provider_type="yinhe",
            provider_name="OneAPI",
            endpoint="http://oneapi.example/v1/chat/completions",
            model="yinhe-thinking",
            api_key="primary-key",
        ),
        model_router.ModelRoute(
            slot="backup",
            provider_id=2,
            provider_type="deepseek",
            provider_name="DeepSeek",
            endpoint="https://api.deepseek.com/anthropic",
            model="deepseek-v4-flash",
            api_key="backup-key",
        ),
    )


class RoutedChatTests(unittest.IsolatedAsyncioTestCase):
    async def test_agent_primary_exhaustion_moves_forward_without_http_failback(self):
        configured_routes = routes()
        agent_complete = AsyncMock(
            side_effect=[RuntimeError("primary exhausted"), "backup answer"]
        )
        chat_http = AsyncMock(return_value="must not run")
        logs: list[str] = []

        with (
            patch.object(config, "AI_BACKEND", "agent_sdk"),
            patch.object(config, "AI_HTTP_FALLBACK", True),
            patch.object(config, "AI_PRIMARY_MAX_RETRIES", 3),
            patch.object(config, "AI_BACKUP_MAX_RETRIES", 2),
            patch.object(
                model_router,
                "resolve_model_routes",
                AsyncMock(return_value=configured_routes),
            ),
            patch("backend.pipeline.agent.agent_complete", agent_complete),
            patch.object(digester, "_chat_http", chat_http),
        ):
            result = await digester._chat(
                "system",
                "content",
                endpoint=configured_routes[0].endpoint,
                model=configured_routes[0].model,
                api_key="primary-key",
                log=logs.append,
            )

        self.assertEqual(result, "backup answer")
        self.assertEqual(agent_complete.await_count, 2)
        self.assertEqual(
            [call.kwargs["endpoint"] for call in agent_complete.await_args_list],
            [configured_routes[0].endpoint, configured_routes[1].endpoint],
        )
        self.assertTrue(
            all(
                call.kwargs["allow_provider_failover"] is False
                for call in agent_complete.await_args_list
            )
        )
        self.assertEqual(
            [call.kwargs["max_retries"] for call in agent_complete.await_args_list],
            [3, 2],
        )
        chat_http.assert_not_awaited()
        self.assertTrue(any("exhausted its 4 attempts" in line for line in logs))
        self.assertIn("completed via DeepSeek backup", logs[-1])

    async def test_http_backend_exhausts_primary_before_backup(self):
        configured_routes = routes()
        chat_http = AsyncMock(
            side_effect=[RuntimeError("primary exhausted"), "backup answer"]
        )

        with (
            patch.object(config, "AI_BACKEND", "http"),
            patch.object(
                model_router,
                "resolve_model_routes",
                AsyncMock(return_value=configured_routes),
            ),
            patch.object(digester, "_chat_http", chat_http),
        ):
            result = await digester._chat(
                "system",
                "content",
                endpoint=configured_routes[0].endpoint,
                model=configured_routes[0].model,
                api_key="primary-key",
            )

        self.assertEqual(result, "backup answer")
        self.assertEqual(
            [call.kwargs["endpoint"] for call in chat_http.await_args_list],
            [configured_routes[0].endpoint, configured_routes[1].endpoint],
        )
        self.assertEqual(
            [call.kwargs["max_retries"] for call in chat_http.await_args_list],
            [config.AI_PRIMARY_MAX_RETRIES, config.AI_BACKUP_MAX_RETRIES],
        )

    async def test_final_http_transport_stays_on_backup(self):
        configured_routes = routes()
        agent_complete = AsyncMock(
            side_effect=[RuntimeError("primary exhausted"), RuntimeError("backup sdk failed")]
        )
        chat_http = AsyncMock(return_value="backup over http")

        with (
            patch.object(config, "AI_BACKEND", "agent_sdk"),
            patch.object(config, "AI_HTTP_FALLBACK", True),
            patch.object(
                model_router,
                "resolve_model_routes",
                AsyncMock(return_value=configured_routes),
            ),
            patch("backend.pipeline.agent.agent_complete", agent_complete),
            patch.object(digester, "_chat_http", chat_http),
        ):
            result = await digester._chat(
                "system",
                "content",
                endpoint=configured_routes[0].endpoint,
                model=configured_routes[0].model,
                api_key="primary-key",
            )

        self.assertEqual(result, "backup over http")
        chat_http.assert_awaited_once()
        self.assertEqual(chat_http.await_args.kwargs["endpoint"], configured_routes[1].endpoint)
        self.assertEqual(chat_http.await_args.kwargs["model"], configured_routes[1].model)
        self.assertEqual(
            chat_http.await_args.kwargs["max_retries"],
            config.AI_BACKUP_MAX_RETRIES,
        )


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from backend import config
from backend.pipeline import digester


class ChatCompletionsUrlTests(unittest.TestCase):
    def test_a_provider_host_is_completed_to_the_chat_route(self):
        # Provider rows only need the gateway host for the Agent SDK backend.
        self.assertEqual(
            digester._chat_completions_url("http://oneapi.example"),
            "http://oneapi.example/v1/chat/completions",
        )
        self.assertEqual(
            digester._chat_completions_url("http://oneapi.example/v1"),
            "http://oneapi.example/v1/chat/completions",
        )

    def test_a_complete_url_is_left_alone(self):
        self.assertEqual(
            digester._chat_completions_url("http://oneapi.example/v1/chat/completions"),
            "http://oneapi.example/v1/chat/completions",
        )

    def test_a_gateway_subpath_keeps_its_prefix(self):
        self.assertEqual(
            digester._chat_completions_url("https://api.deepseek.com/anthropic"),
            "https://api.deepseek.com/anthropic/v1/chat/completions",
        )


class ChatDispatchTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_failed_sdk_call_falls_back_to_the_http_client(self):
        agent_complete = AsyncMock(side_effect=RuntimeError("claude CLI exited 1"))
        chat_http = AsyncMock(return_value="summary")

        with (
            patch.object(config, "AI_BACKEND", "agent_sdk"),
            patch.object(config, "AI_HTTP_FALLBACK", True),
            patch("backend.pipeline.agent.agent_complete", agent_complete),
            patch.object(digester, "_chat_http", chat_http),
        ):
            result = await digester._chat("system", "content", endpoint="http://x/v1")

        self.assertEqual(result, "summary")
        self.assertEqual(chat_http.await_count, 1)

    async def test_the_fallback_can_be_switched_off(self):
        agent_complete = AsyncMock(side_effect=RuntimeError("claude CLI exited 1"))
        chat_http = AsyncMock(return_value="summary")

        with (
            patch.object(config, "AI_BACKEND", "agent_sdk"),
            patch.object(config, "AI_HTTP_FALLBACK", False),
            patch("backend.pipeline.agent.agent_complete", agent_complete),
            patch.object(digester, "_chat_http", chat_http),
        ):
            with self.assertRaisesRegex(RuntimeError, "claude CLI exited 1"):
                await digester._chat("system", "content", endpoint="http://x/v1")

        self.assertEqual(chat_http.await_count, 0)


class ChatHttpReasoningBudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_empty_length_response_doubles_reasoning_budget_immediately(self):
        exhausted = MagicMock()
        exhausted.raise_for_status.return_value = None
        exhausted.json.return_value = {
            "choices": [{
                "finish_reason": "length",
                "message": {"content": "", "reasoning_content": "thinking"},
            }],
            "usage": {"prompt_tokens": 100, "completion_tokens": 4096, "total_tokens": 4196},
        }
        completed = MagicMock()
        completed.raise_for_status.return_value = None
        completed.json.return_value = {
            "choices": [{
                "finish_reason": "stop",
                "message": {"content": "final answer"},
            }],
        }
        client = MagicMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)
        client.post = AsyncMock(side_effect=[exhausted, completed])

        with (
            patch.object(config, "AI_MAX_RETRIES", 2),
            patch.object(digester.httpx, "AsyncClient", return_value=client),
            patch.object(digester.asyncio, "sleep", AsyncMock()) as sleep,
        ):
            result = await digester._chat_http(
                "system",
                "content",
                endpoint="http://x/v1",
                model="yinhe-thinking",
                max_tokens=4096,
            )

        self.assertEqual(result, "final answer")
        self.assertEqual(
            [call.kwargs["json"]["max_tokens"] for call in client.post.await_args_list],
            [4096, 8192],
        )
        self.assertEqual(sleep.await_count, 0)

    async def test_empty_length_response_stops_at_adaptive_budget_ceiling(self):
        exhausted = MagicMock()
        exhausted.raise_for_status.return_value = None
        exhausted.json.return_value = {
            "choices": [{
                "finish_reason": "length",
                "message": {"content": "", "reasoning_content": "thinking"},
            }],
        }
        client = MagicMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)
        client.post = AsyncMock(return_value=exhausted)

        with (
            patch.object(config, "AI_MAX_RETRIES", 9),
            patch.object(digester.httpx, "AsyncClient", return_value=client),
        ):
            with self.assertRaisesRegex(RuntimeError, "maximum adaptive output budget"):
                await digester._chat_http(
                    "system",
                    "content",
                    endpoint="http://x/v1",
                    model="yinhe-thinking",
                    max_tokens=digester.MAX_REASONING_OUTPUT_TOKENS,
                )

        self.assertEqual(client.post.await_count, 1)


class ScriptClosingTests(unittest.IsolatedAsyncioTestCase):
    async def test_configured_closing_is_prompted_and_appended_when_model_omits_it(self):
        resolve_provider = AsyncMock(return_value=("http://x/v1", "model", "key"))
        chat = AsyncMock(return_value="A sharp opening.\nA useful final takeaway.")

        with (
            patch.object(digester, "_resolve_provider", resolve_provider),
            patch.object(digester, "_chat", chat),
        ):
            result = await digester.generate_script(
                {"title": "Test"},
                closing_remarks="Thanks for watching. See you next time.",
            )

        system_prompt = chat.await_args.args[0]
        self.assertIn("End the script with the exact spoken text below", system_prompt)
        self.assertTrue(result.endswith("Thanks for watching. See you next time."))

    async def test_configured_closing_is_not_duplicated_when_model_includes_it(self):
        closing = "Thanks for watching. See you next time."
        resolve_provider = AsyncMock(return_value=("http://x/v1", "model", "key"))
        chat = AsyncMock(return_value=f"A sharp opening.\n{closing}")

        with (
            patch.object(digester, "_resolve_provider", resolve_provider),
            patch.object(digester, "_chat", chat),
        ):
            result = await digester.generate_script(
                {"title": "Test"},
                closing_remarks=closing,
            )

        self.assertEqual(result.count(closing), 1)

    async def test_model_line_break_inside_closing_does_not_duplicate_it(self):
        closing = "Thanks for watching. See you next time."
        resolve_provider = AsyncMock(return_value=("http://x/v1", "model", "key"))
        chat = AsyncMock(return_value="A sharp opening.\nThanks for watching.\nSee you next time.")

        with (
            patch.object(digester, "_resolve_provider", resolve_provider),
            patch.object(digester, "_chat", chat),
        ):
            result = await digester.generate_script(
                {"title": "Test"},
                closing_remarks=closing,
            )

        self.assertEqual(result.count("Thanks for watching."), 1)
        self.assertEqual(result.count("See you next time."), 1)


if __name__ == "__main__":
    unittest.main()

import asyncio
import sys
import types
import unittest
from unittest.mock import AsyncMock, patch

from backend import config, skills_admin
from backend.pipeline import agent, model_router, provider_rate_limit


class FakeTextBlock:
    def __init__(self, text: str):
        self.text = text


class FakeThinkingBlock:
    def __init__(self, thinking: str):
        self.thinking = thinking


class FakeToolUseBlock:
    def __init__(self, name: str):
        self.name = name


class FakeAssistantMessage:
    def __init__(self, content, *, error=None):
        self.content = content
        self.error = error


class FakeResultMessage:
    def __init__(self, result: str = "", *, is_error: bool = False):
        self.result = result
        self.is_error = is_error
        self.errors = []
        self.usage = {}


class FakeClaudeAgentOptions:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.stderr = None


def fake_sdk(query):
    return types.SimpleNamespace(
        AssistantMessage=FakeAssistantMessage,
        ClaudeAgentOptions=FakeClaudeAgentOptions,
        ResultMessage=FakeResultMessage,
        TextBlock=FakeTextBlock,
        ThinkingBlock=FakeThinkingBlock,
        ToolUseBlock=FakeToolUseBlock,
        query=query,
    )


class ProviderBaseUrlTests(unittest.TestCase):
    def test_openai_chat_suffix_is_removed_for_the_agent_sdk(self):
        with patch.object(config, "ANTHROPIC_BASE_URL", ""):
            self.assertEqual(
                agent._derive_base_url("http://oneapi.example/v1/chat/completions"),
                "http://oneapi.example",
            )

    def test_anthropic_provider_subpath_is_preserved(self):
        with patch.object(config, "ANTHROPIC_BASE_URL", ""):
            self.assertEqual(
                agent._derive_base_url("https://api.deepseek.com/anthropic"),
                "https://api.deepseek.com/anthropic",
            )
            self.assertEqual(
                agent._derive_base_url(
                    "https://workspace.example/apps/anthropic/v1/chat/completions"
                ),
                "https://workspace.example/apps/anthropic",
            )


class AgentCompleteTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        provider_rate_limit.reset_for_tests()
        self.addCleanup(provider_rate_limit.reset_for_tests)
        self.enterContext(
            patch.object(config, "AI_YINHE_MIN_REQUEST_INTERVAL_SECONDS", 0.0)
        )
        self.enterContext(
            patch.object(config, "AI_PRIMARY_MIN_REQUEST_INTERVAL_SECONDS", 0.0)
        )

    async def test_primary_429_rotates_to_next_key_before_backup(self):
        attempts: list[str] = []
        selected: list[str] = []
        logs: list[str] = []

        async def query(*, prompt, options):
            key = options.env["ANTHROPIC_API_KEY"]
            attempts.append(key)
            if key == "primary-a":
                options.stderr("[ERROR] 429 rate_limit")
                raise Exception("quota full")
            yield FakeAssistantMessage([FakeTextBlock("primary answer")])

        routes = (
            model_router.ModelRoute(
                slot="primary",
                provider_id=1,
                provider_type="yinhe",
                provider_name="OneAPI",
                endpoint="http://oneapi.example",
                model="yinhe-thinking",
                api_key="primary-a",
                api_keys=("primary-a", "primary-b", "primary-c"),
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
        with (
            patch.dict(sys.modules, {"claude_agent_sdk": fake_sdk(query)}),
            patch.object(
                model_router,
                "resolve_model_routes",
                AsyncMock(return_value=routes),
            ),
            patch.object(config, "AI_PRIMARY_MAX_RETRIES", 0),
            patch.object(config, "AI_PRIMARY_RATE_LIMIT_MAX_WAITS", 0),
            patch.object(config, "ANTHROPIC_BASE_URL", ""),
            patch.object(config, "ANTHROPIC_AUTH_TOKEN", ""),
            patch.object(config, "ANTHROPIC_MODEL", ""),
            patch.object(config, "ANTHROPIC_DEFAULT_HAIKU_MODEL", ""),
            patch.object(skills_admin, "runtime_skill_names", return_value=([], [])),
            patch.object(
                provider_rate_limit,
                "wait_for_request_slot",
                AsyncMock(return_value=0.0),
            ),
            patch.object(
                provider_rate_limit,
                "record_rate_limit",
                AsyncMock(return_value=65.0),
            ) as record_rate_limit,
        ):
            result = await agent.agent_complete(
                "Return text.",
                "content",
                endpoint=routes[0].endpoint,
                model=routes[0].model,
                api_key=routes[0].api_key,
                log=logs.append,
                credential_selected=selected.append,
            )

        self.assertEqual(result, "primary answer")
        self.assertEqual(attempts, ["primary-a", "primary-b"])
        self.assertEqual(selected, ["primary-b"])
        self.assertTrue(any("rotating primary API key" in line for line in logs))
        self.assertFalse(any("switching to DeepSeek" in line for line in logs))
        self.assertEqual(
            record_rate_limit.await_args.kwargs["api_key_id"],
            routes[0].api_key_id,
        )

    async def test_output_started_failure_is_never_replayed_on_backup(self):
        attempts: list[str] = []

        async def query(*, prompt, options):
            attempts.append(options.env["ANTHROPIC_BASE_URL"])
            yield FakeAssistantMessage([FakeTextBlock("partial")])
            raise Exception("stream broke after output for primary-key")

        routes = (
            model_router.ModelRoute(
                slot="primary",
                provider_id=1,
                provider_type="yinhe",
                provider_name="OneAPI",
                endpoint="http://oneapi.example",
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
        with (
            patch.dict(sys.modules, {"claude_agent_sdk": fake_sdk(query)}),
            patch.object(model_router, "resolve_model_routes", AsyncMock(return_value=routes)),
            patch.object(config, "AI_PRIMARY_MAX_RETRIES", 2),
            patch.object(config, "ANTHROPIC_BASE_URL", ""),
            patch.object(config, "ANTHROPIC_AUTH_TOKEN", ""),
            patch.object(config, "ANTHROPIC_MODEL", ""),
            patch.object(config, "ANTHROPIC_DEFAULT_HAIKU_MODEL", ""),
            patch.object(skills_admin, "runtime_skill_names", return_value=([], [])),
        ):
            with self.assertRaises(model_router.ModelOutputCommittedError) as caught:
                await agent.agent_complete(
                    "Return text.",
                    "content",
                    endpoint=routes[0].endpoint,
                    model=routes[0].model,
                    api_key=routes[0].api_key,
                )

        self.assertEqual(attempts, ["http://oneapi.example"])
        self.assertNotIn("primary-key", str(caught.exception))

    async def test_rate_limit_assistant_error_is_not_treated_as_committed_output(self):
        attempts: list[str] = []

        async def query(*, prompt, options):
            key = options.env["ANTHROPIC_API_KEY"]
            attempts.append(key)
            if key == "primary-a":
                yield FakeAssistantMessage(
                    [FakeTextBlock("API Error: 429 request limit")],
                    error="rate_limit",
                )
                yield FakeResultMessage(
                    "API Error: 429 request limit",
                    is_error=True,
                )
                return
            yield FakeAssistantMessage([FakeTextBlock("primary answer")])

        routes = (
            model_router.ModelRoute(
                slot="primary",
                provider_id=1,
                provider_type="yinhe",
                provider_name="Galaxy OneAPI",
                endpoint="http://oneapi.example",
                model="yinhe-thinking",
                api_key="primary-a",
                api_keys=("primary-a", "primary-b"),
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
        with (
            patch.dict(sys.modules, {"claude_agent_sdk": fake_sdk(query)}),
            patch.object(model_router, "resolve_model_routes", AsyncMock(return_value=routes)),
            patch.object(config, "AI_PRIMARY_MAX_RETRIES", 0),
            patch.object(config, "ANTHROPIC_BASE_URL", ""),
            patch.object(config, "ANTHROPIC_AUTH_TOKEN", ""),
            patch.object(config, "ANTHROPIC_MODEL", ""),
            patch.object(config, "ANTHROPIC_DEFAULT_HAIKU_MODEL", ""),
            patch.object(skills_admin, "runtime_skill_names", return_value=([], [])),
            patch.object(
                provider_rate_limit,
                "wait_for_request_slot",
                AsyncMock(return_value=0.0),
            ),
            patch.object(
                provider_rate_limit,
                "record_rate_limit",
                AsyncMock(return_value=65.0),
            ) as record_rate_limit,
        ):
            result = await agent.agent_complete(
                "Return text.",
                "content",
                endpoint=routes[0].endpoint,
                model=routes[0].model,
                api_key=routes[0].api_key,
            )

        self.assertEqual(result, "primary answer")
        self.assertEqual(attempts, ["primary-a", "primary-b"])
        record_rate_limit.assert_awaited_once()
        self.assertEqual(
            record_rate_limit.await_args.kwargs["provider_type"],
            "yinhe",
        )

    async def test_rate_limit_after_skill_load_and_thinking_uses_backup(self):
        attempts: list[str] = []
        logs: list[str] = []

        async def query(*, prompt, options):
            base = options.env["ANTHROPIC_BASE_URL"]
            attempts.append(base)
            if "oneapi" in base:
                yield FakeAssistantMessage(
                    [
                        FakeThinkingBlock("select the relevant writing skill"),
                        FakeToolUseBlock("Skill"),
                    ]
                )
                options.stderr("[ERROR] API error (attempt 1/1): 429 rate_limit")
                yield FakeAssistantMessage(
                    [FakeTextBlock("API Error: 429 request limit")],
                    error="rate_limit",
                )
                raise Exception("primary second turn rejected")
            yield FakeAssistantMessage([FakeTextBlock("backup answer")])

        routes = (
            model_router.ModelRoute(
                slot="primary",
                provider_id=1,
                provider_type="yinhe",
                provider_name="Galaxy OneAPI",
                endpoint="http://oneapi.example",
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
        with (
            patch.dict(sys.modules, {"claude_agent_sdk": fake_sdk(query)}),
            patch.object(model_router, "resolve_model_routes", AsyncMock(return_value=routes)),
            patch.object(config, "AI_PRIMARY_MAX_RETRIES", 0),
            patch.object(config, "AI_PRIMARY_RATE_LIMIT_MAX_WAITS", 0),
            patch.object(config, "AI_BACKUP_MAX_RETRIES", 0),
            patch.object(config, "ANTHROPIC_BASE_URL", ""),
            patch.object(config, "ANTHROPIC_AUTH_TOKEN", ""),
            patch.object(config, "ANTHROPIC_MODEL", ""),
            patch.object(config, "ANTHROPIC_DEFAULT_HAIKU_MODEL", ""),
            patch.object(skills_admin, "runtime_skill_names", return_value=([], [])),
            patch.object(
                provider_rate_limit,
                "wait_for_request_slot",
                AsyncMock(return_value=0.0),
            ),
            patch.object(
                provider_rate_limit,
                "record_rate_limit",
                AsyncMock(return_value=65.0),
            ),
        ):
            result = await agent.agent_complete(
                "Return text.",
                "content",
                endpoint=routes[0].endpoint,
                model=routes[0].model,
                api_key=routes[0].api_key,
                log=logs.append,
            )

        self.assertEqual(result, "backup answer")
        self.assertEqual(
            attempts,
            ["http://oneapi.example", "https://api.deepseek.com/anthropic"],
        )
        self.assertTrue(any("switching to DeepSeek backup" in line for line in logs))

    async def test_primary_429_wait_does_not_consume_attempt_or_use_backup(self):
        attempts: list[str] = []
        logs: list[str] = []

        async def query(*, prompt, options):
            attempts.append(options.env["ANTHROPIC_BASE_URL"])
            if len(attempts) == 1:
                options.stderr("[ERROR] API error (attempt 1/1): 429 rate_limit")
                raise Exception("primary quota full")
            yield FakeAssistantMessage([FakeTextBlock("primary answer")])

        routes = (
            model_router.ModelRoute(
                slot="primary",
                provider_id=1,
                provider_type="yinhe",
                provider_name="OneAPI",
                endpoint="http://oneapi.example",
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
        with (
            patch.dict(sys.modules, {"claude_agent_sdk": fake_sdk(query)}),
            patch.object(model_router, "resolve_model_routes", AsyncMock(return_value=routes)),
            patch.object(config, "AI_PRIMARY_MAX_RETRIES", 0),
            patch.object(config, "AI_PRIMARY_RATE_LIMIT_MAX_WAITS", 1),
            patch.object(config, "ANTHROPIC_BASE_URL", ""),
            patch.object(config, "ANTHROPIC_AUTH_TOKEN", ""),
            patch.object(config, "ANTHROPIC_MODEL", ""),
            patch.object(config, "ANTHROPIC_DEFAULT_HAIKU_MODEL", ""),
            patch.object(skills_admin, "runtime_skill_names", return_value=([], [])),
            patch.object(
                provider_rate_limit,
                "wait_for_request_slot",
                AsyncMock(return_value=0.0),
            ),
            patch.object(
                provider_rate_limit,
                "record_rate_limit",
                AsyncMock(return_value=65.0),
            ) as record_rate_limit,
        ):
            result = await agent.agent_complete(
                "Return text.",
                "content",
                endpoint=routes[0].endpoint,
                model=routes[0].model,
                api_key="primary-key",
                log=logs.append,
            )

        self.assertEqual(result, "primary answer")
        self.assertEqual(attempts, ["http://oneapi.example", "http://oneapi.example"])
        record_rate_limit.assert_awaited_once()
        self.assertTrue(any("without consuming a provider failure" in line for line in logs))
        self.assertFalse(any("switching" in line for line in logs))

    async def test_primary_exhaustion_switches_to_backup_after_all_retries(self):
        attempts: list[str] = []
        logs: list[str] = []

        async def query(*, prompt, options):
            base = options.env["ANTHROPIC_BASE_URL"]
            attempts.append(base)
            if "oneapi" in base:
                raise Exception("primary unavailable")
            yield FakeAssistantMessage([FakeTextBlock("backup answer")])

        routes = (
            model_router.ModelRoute(
                slot="primary",
                provider_id=1,
                provider_type="yinhe",
                provider_name="OneAPI",
                endpoint="http://oneapi.example",
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
        with (
            patch.dict(sys.modules, {"claude_agent_sdk": fake_sdk(query)}),
            patch.object(
                model_router,
                "resolve_model_routes",
                AsyncMock(return_value=routes),
            ),
            patch.object(config, "AI_PRIMARY_MAX_RETRIES", 2),
            patch.object(config, "AI_BACKUP_MAX_RETRIES", 2),
            patch.object(config, "AI_RETRY_BASE_SECONDS", 0),
            patch.object(config, "AI_RETRY_MAX_SECONDS", 0),
            patch.object(config, "ANTHROPIC_BASE_URL", ""),
            patch.object(config, "ANTHROPIC_AUTH_TOKEN", ""),
            patch.object(config, "ANTHROPIC_MODEL", ""),
            patch.object(config, "ANTHROPIC_DEFAULT_HAIKU_MODEL", ""),
            patch.object(skills_admin, "runtime_skill_names", return_value=([], [])),
        ):
            result = await agent.agent_complete(
                "Return text.",
                "content",
                model="yinhe-thinking",
                endpoint="http://oneapi.example",
                api_key="primary-key",
                log=logs.append,
            )

        self.assertEqual(result, "backup answer")
        self.assertEqual(
            attempts,
            ["http://oneapi.example"] * 3 + ["https://api.deepseek.com/anthropic"],
        )
        switch_log = next(line for line in logs if "switching" in line)
        self.assertIn("exhausted its 3 attempts", switch_log)
        self.assertIn("DeepSeek backup", switch_log)
        self.assertIn("completed via DeepSeek backup", logs[-1])

    async def test_primary_retry_success_does_not_touch_backup(self):
        attempts: list[str] = []

        async def query(*, prompt, options):
            base = options.env["ANTHROPIC_BASE_URL"]
            attempts.append(base)
            if len(attempts) == 1:
                raise Exception("temporary primary failure")
            yield FakeAssistantMessage([FakeTextBlock("primary answer")])

        routes = (
            model_router.ModelRoute(
                slot="primary",
                provider_id=1,
                provider_type="yinhe",
                provider_name="OneAPI",
                endpoint="http://oneapi.example",
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
        with (
            patch.dict(sys.modules, {"claude_agent_sdk": fake_sdk(query)}),
            patch.object(
                model_router,
                "resolve_model_routes",
                AsyncMock(return_value=routes),
            ),
            patch.object(config, "AI_PRIMARY_MAX_RETRIES", 2),
            patch.object(config, "AI_BACKUP_MAX_RETRIES", 2),
            patch.object(config, "AI_RETRY_BASE_SECONDS", 0),
            patch.object(config, "AI_RETRY_MAX_SECONDS", 0),
            patch.object(config, "ANTHROPIC_BASE_URL", ""),
            patch.object(config, "ANTHROPIC_AUTH_TOKEN", ""),
            patch.object(config, "ANTHROPIC_MODEL", ""),
            patch.object(config, "ANTHROPIC_DEFAULT_HAIKU_MODEL", ""),
            patch.object(skills_admin, "runtime_skill_names", return_value=([], [])),
        ):
            result = await agent.agent_complete(
                "Return text.",
                "content",
                model="yinhe-thinking",
                endpoint="http://oneapi.example",
                api_key="primary-key",
            )

        self.assertEqual(result, "primary answer")
        self.assertEqual(attempts, ["http://oneapi.example", "http://oneapi.example"])

    async def test_safe_permission_mode_and_provider_environment(self):
        captured = {}

        async def query(*, prompt, options):
            captured["prompt"] = prompt
            captured["options"] = options
            yield FakeAssistantMessage([FakeTextBlock("pong")])
            yield FakeResultMessage("pong")

        with (
            patch.dict(sys.modules, {"claude_agent_sdk": fake_sdk(query)}),
            patch.object(config, "ANTHROPIC_BASE_URL", ""),
            patch.object(config, "ANTHROPIC_AUTH_TOKEN", ""),
            patch.object(config, "ANTHROPIC_MODEL", ""),
            patch.object(config, "ANTHROPIC_DEFAULT_HAIKU_MODEL", ""),
            patch.object(
                skills_admin,
                "runtime_skill_names",
                return_value=(["hyperframes", "animejs"], ["openspec-archive-change"]),
            ),
        ):
            result = await agent.agent_complete(
                "Reply pong.",
                "ping",
                model="yinhe-thinking",
                endpoint="http://oneapi.example/v1/chat/completions",
                api_key="test-key",
            )

        self.assertEqual(result, "pong")
        self.assertEqual(captured["prompt"], "ping")
        self.assertEqual(captured["options"].permission_mode, "default")
        self.assertEqual(captured["options"].cwd, config.PROJECT_ROOT)
        self.assertEqual(captured["options"].setting_sources, ["project"])
        self.assertEqual(captured["options"].tools, ["Skill"])
        self.assertEqual(
            captured["options"].allowed_tools,
            ["Skill(hyperframes)", "Skill(animejs)"],
        )
        self.assertEqual(
            captured["options"].disallowed_tools,
            ["Skill(openspec-archive-change)"],
        )
        self.assertEqual(captured["options"].max_turns, 2)
        self.assertEqual(
            captured["options"].env["ANTHROPIC_BASE_URL"],
            "http://oneapi.example",
        )
        self.assertEqual(captured["options"].env["ANTHROPIC_API_KEY"], "test-key")
        self.assertEqual(
            captured["options"].env["ANTHROPIC_DEFAULT_HAIKU_MODEL"],
            "yinhe-thinking",
        )
        self.assertEqual(captured["options"].env["CLAUDE_CODE_MAX_RETRIES"], "0")

    async def test_transport_stderr_is_returned_to_the_caller(self):
        async def query(*, prompt, options):
            options.stderr("upstream rejected the request")
            raise Exception("CLI failed")
            yield  # pragma: no cover - keeps this an async generator

        with (
            patch.dict(sys.modules, {"claude_agent_sdk": fake_sdk(query)}),
            patch.object(config, "AI_MAX_RETRIES", 0),
            patch.object(config, "ANTHROPIC_BASE_URL", ""),
            patch.object(config, "ANTHROPIC_AUTH_TOKEN", ""),
            patch.object(config, "ANTHROPIC_MODEL", ""),
            patch.object(skills_admin, "runtime_skill_names", return_value=([], [])),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "upstream rejected the request",
            ):
                await agent.agent_complete(
                    "Reply pong.",
                    "ping",
                    model="yinhe-thinking",
                    endpoint="http://oneapi.example",
                    api_key="test-key",
                )

    async def test_retry_log_does_not_name_an_unselected_provider(self):
        attempts = 0
        logs: list[str] = []

        async def query(*, prompt, options):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise Exception("temporary failure")
            yield FakeAssistantMessage([FakeTextBlock("pong")])

        with (
            patch.dict(sys.modules, {"claude_agent_sdk": fake_sdk(query)}),
            patch.object(config, "AI_MAX_RETRIES", 1),
            patch.object(config, "AI_RETRY_BASE_SECONDS", 0),
            patch.object(config, "AI_RETRY_MAX_SECONDS", 0),
            patch.object(config, "ANTHROPIC_BASE_URL", ""),
            patch.object(config, "ANTHROPIC_AUTH_TOKEN", ""),
            patch.object(config, "ANTHROPIC_MODEL", ""),
            patch.object(skills_admin, "runtime_skill_names", return_value=([], [])),
        ):
            result = await agent.agent_complete(
                "Reply pong.",
                "ping",
                model="deepseek-v4-flash",
                endpoint="https://api.deepseek.com/anthropic",
                api_key="test-key",
                log=logs.append,
            )

        self.assertEqual(result, "pong")
        retry_log = next(line for line in logs if "retrying" in line)
        self.assertIn("retrying same provider turn", retry_log)
        self.assertNotIn("yhroot", retry_log)

    async def test_one_shot_call_can_skip_project_skills(self):
        captured = {}

        async def query(*, prompt, options):
            captured["options"] = options
            yield FakeAssistantMessage([FakeTextBlock("title")])

        with (
            patch.dict(sys.modules, {"claude_agent_sdk": fake_sdk(query)}),
            patch.object(config, "ANTHROPIC_BASE_URL", ""),
            patch.object(config, "ANTHROPIC_AUTH_TOKEN", ""),
            patch.object(config, "ANTHROPIC_MODEL", ""),
            patch.object(skills_admin, "runtime_skill_names") as skill_names,
        ):
            result = await agent.agent_complete(
                "Return a title.",
                "content",
                enable_skills=False,
            )

        self.assertEqual(result, "title")
        self.assertEqual(captured["options"].setting_sources, [])
        self.assertEqual(captured["options"].tools, [])
        self.assertEqual(captured["options"].max_turns, 1)
        skill_names.assert_not_called()

    async def test_constrained_one_shot_call_can_disable_hidden_thinking(self):
        captured = {}

        async def query(*, prompt, options):
            captured["options"] = options
            yield FakeAssistantMessage([FakeTextBlock("edited text")])

        with (
            patch.dict(sys.modules, {"claude_agent_sdk": fake_sdk(query)}),
            patch.object(config, "ANTHROPIC_BASE_URL", ""),
            patch.object(config, "ANTHROPIC_AUTH_TOKEN", ""),
            patch.object(config, "ANTHROPIC_MODEL", ""),
            patch.object(skills_admin, "runtime_skill_names") as skill_names,
        ):
            result = await agent.agent_complete(
                "Return edited text.",
                "content",
                enable_skills=False,
                disable_thinking=True,
            )

        self.assertEqual(result, "edited text")
        self.assertEqual(captured["options"].thinking, {"type": "disabled"})
        skill_names.assert_not_called()

    async def test_cli_debug_chatter_is_dropped_from_the_failure_detail(self):
        async def query(*, prompt, options):
            options.stderr("2026-01-01T00:00:00Z [DEBUG] CA certs: system store returned empty")
            options.stderr("2026-01-01T00:00:01Z [WARN] Streaming stall detected: 32.4s gap")
            raise Exception("CLI failed")
            yield  # pragma: no cover - keeps this an async generator

        with (
            patch.dict(sys.modules, {"claude_agent_sdk": fake_sdk(query)}),
            patch.object(config, "AI_MAX_RETRIES", 0),
            patch.object(config, "ANTHROPIC_BASE_URL", ""),
            patch.object(config, "ANTHROPIC_AUTH_TOKEN", ""),
            patch.object(config, "ANTHROPIC_MODEL", ""),
            patch.object(skills_admin, "runtime_skill_names", return_value=([], [])),
        ):
            with self.assertRaises(RuntimeError) as caught:
                await agent.agent_complete(
                    "Reply pong.",
                    "ping",
                    model="yinhe-thinking",
                    endpoint="http://oneapi.example",
                    api_key="test-key",
                )

        self.assertIn("Streaming stall detected", str(caught.exception))
        self.assertNotIn("CA certs", str(caught.exception))

    async def test_a_hung_cli_is_cut_off_at_the_turn_timeout(self):
        closed: list[bool] = []

        async def query(*, prompt, options):
            try:
                await asyncio.sleep(3600)
                yield FakeResultMessage("never")  # pragma: no cover - unreachable
            finally:
                closed.append(True)

        with (
            patch.dict(sys.modules, {"claude_agent_sdk": fake_sdk(query)}),
            # yhroot hard timeouts participate in the configured retry ladder
            # just like transport errors; every subprocess must still close.
            patch.object(config, "AI_MAX_RETRIES", 2),
            patch.object(config, "AI_RETRY_BASE_SECONDS", 0),
            patch.object(config, "AI_RETRY_MAX_SECONDS", 0),
            patch.object(config, "AGENT_TURN_TIMEOUT", 1),
            patch.object(config, "ANTHROPIC_BASE_URL", ""),
            patch.object(config, "ANTHROPIC_AUTH_TOKEN", ""),
            patch.object(config, "ANTHROPIC_MODEL", ""),
            patch.object(skills_admin, "runtime_skill_names", return_value=([], [])),
        ):
            with self.assertRaisesRegex(RuntimeError, "AGENT_TURN_TIMEOUT"):
                await asyncio.wait_for(
                    agent.agent_complete(
                        "Reply pong.",
                        "ping",
                        model="yinhe-thinking",
                        endpoint="http://oneapi.example",
                        api_key="test-key",
                    ),
                    timeout=30,
                )

        # The generator was closed, so the CLI subprocess is not left running.
        self.assertEqual(closed, [True, True, True])


class ConnectionTestTests(unittest.IsolatedAsyncioTestCase):
    async def test_probe_allows_a_reasoning_model_to_emit_text(self):
        captured = {}

        async def complete(system_prompt, user_content, **kwargs):
            captured.update(kwargs)
            return "pong"

        with patch.object(agent, "agent_complete", complete):
            latency = await agent.test_connection(
                "http://oneapi.example",
                "yinhe-thinking",
                "test-key",
            )

        self.assertGreaterEqual(latency, 0)
        self.assertEqual(captured["max_tokens"], 256)


if __name__ == "__main__":
    unittest.main()

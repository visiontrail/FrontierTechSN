"""Claude Agent SDK transport for the digestion/scriptwriting AI calls.

This replaces the direct OpenAI-compatible HTTP client (see ``digester._chat_http``)
with the Claude Agent SDK. The SDK spawns the bundled/system ``claude`` CLI in
process and talks the Anthropic protocol to whatever gateway ``ANTHROPIC_BASE_URL``
points at, so the same provider registry (endpoint/api_key/model) that drove the
HTTP client keeps working — the endpoint host is simply reinterpreted as an
Anthropic base URL.

The two callers (``summarize`` and ``generate_script``) still want a single
plain-text completion for a (system prompt, user content) pair, so this module
exposes exactly that: :func:`agent_complete`. All the chunking, JSON parsing,
retry-on-empty and CJK repair logic stays in ``digester.py`` unchanged.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
from collections import deque
from collections.abc import Callable
from urllib.parse import urlsplit, urlunsplit

from backend import config, skills_admin

logger = logging.getLogger(__name__)
LogCallback = Callable[[str], None]

# The CLI only reports API failures, retries and streaming stalls at debug
# level, so without --debug-to-stderr a failed turn arrives as a bare
# "exit code 1" with an empty stderr — which is exactly how a 57-minute
# digestion failure managed to leave no diagnosis behind. Debug output is
# therefore always requested, and the routine chatter filtered back out.
DIAGNOSTIC_STDERR_LINES = 20
_NOISE_LEVELS = ("[DEBUG]", "[INFO]")


def _log(log: LogCallback | None, message: str) -> None:
    if log is not None:
        log(message)
    else:
        logger.info(message)


def _warn(log: LogCallback | None, message: str) -> None:
    logger.warning(message)
    if log is not None:
        log(message)


def _keep_diagnostic(sink: deque[str], line: str) -> None:
    """Retain the stderr lines worth reporting on a failure.

    ``--debug-to-stderr`` emits a hundred lines of startup chatter per call, so
    DEBUG/INFO is dropped; everything else — WARN and above, and any untagged
    output such as a CLI crash — is kept."""
    text = line.strip()
    if text and not any(level in text for level in _NOISE_LEVELS):
        sink.append(text)


def _diagnostic_tail(sink: deque[str]) -> str:
    if not sink:
        return ""
    return " | claude CLI: " + " ⏎ ".join(sink)[-1500:]


def _derive_base_url(endpoint: str | None) -> str:
    """Map an OpenAI-style endpoint to an Anthropic base URL.

    The Anthropic client appends ``/v1/messages``. Strip an OpenAI chat suffix,
    but preserve provider-specific Anthropic prefixes such as ``/anthropic`` or
    ``/apps/anthropic``. ``ANTHROPIC_BASE_URL`` still overrides this entirely."""
    if config.ANTHROPIC_BASE_URL:
        return config.ANTHROPIC_BASE_URL
    if not endpoint:
        return ""
    parts = urlsplit(endpoint)
    if parts.scheme and parts.netloc:
        path = parts.path.rstrip("/")
        for suffix in ("/v1/chat/completions", "/chat/completions"):
            if path.endswith(suffix):
                path = path.removesuffix(suffix)
                break
        return urlunsplit((parts.scheme, parts.netloc, path, "", ""))
    return endpoint


def build_agent_env(
    model: str | None,
    endpoint: str | None,
    api_key: str | None,
    max_tokens: int | None = None,
) -> dict[str, str]:
    """Provider env vars for a ``ClaudeAgentOptions``.

    Mirrors SmartHRBI's ``build_sdk_provider_env``: point the SDK-spawned CLI at
    the provider gateway via ``ANTHROPIC_BASE_URL`` + auth token, and select the
    model. Explicit ``ANTHROPIC_*`` config always wins over the derived values."""
    env: dict[str, str] = {
        "API_TIMEOUT_MS": str(config.AGENT_REQUEST_TIMEOUT * 1000),
        # Keep these one-shot text transforms off telemetry/update channels.
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    }

    # Skills under .claude/skills reference `opencli` by command name. Put the
    # repository wrapper/runtime first while retaining the launching process'
    # PATH. Nothing is installed into the user's global Claude Code runtime.
    project_bins = [
        str(config.PROJECT_ROOT / "scripts"),
        str(config.PROJECT_ROOT / "tools" / "opencli" / "node_modules" / ".bin"),
    ]
    env["PATH"] = os.pathsep.join([*project_bins, os.environ.get("PATH", "")])

    token = (config.ANTHROPIC_AUTH_TOKEN or (api_key or "")).strip()
    if token:
        env["ANTHROPIC_API_KEY"] = token
        env["ANTHROPIC_AUTH_TOKEN"] = token

    base_url = _derive_base_url(endpoint)
    if base_url:
        env["ANTHROPIC_BASE_URL"] = base_url

    resolved_model = (config.ANTHROPIC_MODEL or model or "").strip()
    if resolved_model:
        env["ANTHROPIC_MODEL"] = resolved_model
        env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = (
            config.ANTHROPIC_DEFAULT_HAIKU_MODEL or resolved_model
        )

    if max_tokens:
        env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(max_tokens)

    return env


async def agent_complete(
    system_prompt: str,
    user_content: str,
    *,
    model: str | None = None,
    endpoint: str | None = None,
    api_key: str | None = None,
    max_tokens: int | None = None,
    enable_skills: bool = True,
    disable_thinking: bool = False,
    log: LogCallback | None = None,
    label: str = "AI call",
) -> str:
    """Run a single Claude Agent SDK turn and return the assistant's text.

    Raises ``RuntimeError`` on empty/failed output after ``AI_MAX_RETRIES``,
    matching the contract of the legacy ``_chat`` so ``digester.py`` can treat
    both backends the same."""
    # Imported lazily so the module (and the HTTP backend) load fine even when
    # the SDK isn't installed.
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        ResultMessage,
        TextBlock,
        query,
    )

    resolved_model = (config.ANTHROPIC_MODEL or model or "").strip() or None
    env = build_agent_env(model, endpoint, api_key, max_tokens)

    if enable_skills:
        enabled_skills, disabled_skills = skills_admin.runtime_skill_names()
    else:
        enabled_skills, disabled_skills = [], []
    skill_tools = [f"Skill({name})" for name in enabled_skills]

    base_options: dict = {
        "system_prompt": system_prompt,
        "model": resolved_model,
        # One turn may load a relevant Skill; the following turn emits the
        # requested text result.
        "max_turns": 2 if enable_skills else 1,
        # Restrict the agent to Admin-managed project Skills. No filesystem,
        # shell, web, or mutation tools are exposed by this pipeline.
        "tools": ["Skill"] if enabled_skills else [],
        "allowed_tools": skill_tools,
        "disallowed_tools": [f"Skill({name})" for name in disabled_skills],
        # One-shot calls such as title generation do not need project Skills.
        # Avoid loading every discovered Skill into the CLI in those sessions.
        "setting_sources": ["project"] if enable_skills else [],
        "cwd": config.PROJECT_ROOT,
        "permission_mode": "default",
        "env": env,
        "extra_args": {"debug-to-stderr": None},
    }
    if disable_thinking:
        # Constrained one-shot transforms need the requested text, not a long
        # hidden reasoning trace. DeepSeek V4 enables thinking by default and
        # can otherwise consume the entire output allowance before emitting
        # any text; the Anthropic-compatible API supports this standard
        # explicit disable control.
        base_options["thinking"] = {"type": "disabled"}
    # Newer SDK releases support an initialize-time Skill context filter.
    # Keep compatibility with the vendored SDK while using the stronger filter
    # automatically after it is upgraded.
    if "skills" in getattr(ClaudeAgentOptions, "__dataclass_fields__", {}):
        base_options["skills"] = enabled_skills
    if config.CLAUDE_CLI_PATH:
        base_options["cli_path"] = config.CLAUDE_CLI_PATH

    skills_summary = (
        "off"
        if not enable_skills
        else f"{len(enabled_skills)} enabled/{len(disabled_skills)} disabled"
    )
    _log(
        log,
        f"{label}: Claude Agent SDK (model={resolved_model or 'default'}, "
        f"base={env.get('ANTHROPIC_BASE_URL', 'default')}, "
        f"skills={skills_summary}, "
        f"thinking={'off' if disable_thinking else 'default'}, "
        f"~{len(user_content.split())} words in, "
        f"request/turn ceiling {config.AGENT_REQUEST_TIMEOUT}s/{config.AGENT_TURN_TIMEOUT}s)",
    )

    last_error: Exception | None = None
    for attempt in range(config.AI_MAX_RETRIES + 1):
        start = time.perf_counter()
        prompt = user_content
        if attempt > 0:
            # Same nudge the HTTP path uses to stop the model burning the turn
            # on hidden reasoning and return the answer directly.
            base_options["system_prompt"] = (
                f"{system_prompt}\n\n"
                "Important retry instruction: output the final answer directly in the "
                "message content. Do not spend tokens on hidden reasoning, analysis, "
                "markdown fences, or explanations. Start immediately with the requested output."
            )

        options = ClaudeAgentOptions(**base_options)
        text_parts: list[str] = []
        result: ResultMessage | None = None
        diagnostics: deque[str] = deque(maxlen=DIAGNOSTIC_STDERR_LINES)
        options.stderr = lambda line: _keep_diagnostic(diagnostics, line)

        try:
            # The CLI retries a failing request on its own, so the process can
            # outlive AGENT_REQUEST_TIMEOUT many times over. Bound the whole
            # turn here and close the generator on the way out — that tears the
            # transport down and kills the CLI instead of leaking it.
            stream = query(prompt=prompt, options=options)
            try:
                async with asyncio.timeout(config.AGENT_TURN_TIMEOUT):
                    async for message in stream:
                        if isinstance(message, AssistantMessage):
                            for block in message.content:
                                if isinstance(block, TextBlock):
                                    text_parts.append(block.text)
                        elif isinstance(message, ResultMessage):
                            result = message
            finally:
                # Cleanup must not be able to hang the stage it is unwinding.
                with contextlib.suppress(Exception):
                    async with asyncio.timeout(30):
                        await stream.aclose()

            content = "".join(text_parts).strip()
            if not content and result is not None and result.result:
                content = result.result.strip()

            elapsed_ms = (time.perf_counter() - start) * 1000
            if content:
                usage = (result.usage if result else None) or {}
                usage_str = f", tokens={usage}" if usage else ""
                _log(
                    log,
                    f"{label}: responded in {elapsed_ms:.0f}ms ({len(content)} chars{usage_str})",
                )
                return content

            detail = (
                f"empty content from Claude Agent SDK (model={resolved_model or 'default'}"
                f", is_error={getattr(result, 'is_error', 'n/a')})"
            )
            if result is not None and result.is_error:
                errs = result.errors or [result.result or "unknown SDK error"]
                detail = f"Claude Agent SDK error: {'; '.join(str(e) for e in errs)}"
            detail += _diagnostic_tail(diagnostics)
            _warn(log, f"{label} attempt {attempt + 1} failed: {detail}")
            last_error = RuntimeError(detail)
        except TimeoutError:
            elapsed_ms = (time.perf_counter() - start) * 1000
            detail = (
                f"the `claude` CLI ran past AGENT_TURN_TIMEOUT "
                f"({config.AGENT_TURN_TIMEOUT}s, {elapsed_ms:.0f}ms elapsed) and was killed. "
                f"The CLI retries a failing request internally, so this usually means the "
                f"gateway kept timing out at AGENT_REQUEST_TIMEOUT "
                f"({config.AGENT_REQUEST_TIMEOUT}s) rather than that one call was slow"
            ) + _diagnostic_tail(diagnostics)
            _warn(log, f"{label} attempt {attempt + 1} failed: {detail}")
            # A hard turn timeout is still a provider failure. Treat it like
            # every other bounded failure so the selected provider receives
            # the configured number of chances. The long worst-case duration
            # is intentional: daytime gateway stalls are common, while the
            # daily worker can safely keep trying in the background.
            last_error = RuntimeError(detail)
        except Exception as e:  # noqa: BLE001 - surface any SDK/transport failure
            # Preserve the SDK callback's stderr in the error returned by the
            # provider-test API instead of only writing it to container logs.
            detail = f"{e.__class__.__name__}: {e}{_diagnostic_tail(diagnostics)}"
            _warn(log, f"{label} attempt {attempt + 1} failed: {detail}")
            last_error = RuntimeError(detail)

        if attempt < config.AI_MAX_RETRIES:
            delay = min(
                config.AI_RETRY_MAX_SECONDS,
                config.AI_RETRY_BASE_SECONDS * 2**attempt,
            )
            _log(
                log,
                f"{label}: retrying same provider turn in {delay:.0f}s "
                f"({attempt + 2}/{config.AI_MAX_RETRIES + 1})",
            )
            await asyncio.sleep(delay)

    raise RuntimeError(f"AI request failed via Claude Agent SDK: {last_error}") from last_error


async def test_connection(endpoint: str, model: str, api_key: str) -> int:
    """Verify the provider is reachable through the Claude Agent SDK.

    Sends a minimal turn and returns round-trip latency in milliseconds, or
    raises on any failure — same contract as ``digester.test_connection``."""
    start = time.perf_counter()
    text = await agent_complete(
        "You are a connectivity probe. Reply with the single word: pong.",
        "ping",
        model=model,
        endpoint=endpoint,
        api_key=api_key,
        # Reasoning models can spend the first ~100 output tokens internally
        # before emitting even a one-word answer. A 16-token probe therefore
        # reports a false "empty response" despite a healthy connection.
        max_tokens=256,
        label="Provider test",
    )
    if not text.strip():
        raise RuntimeError("Empty response from Claude Agent SDK")
    return int((time.perf_counter() - start) * 1000)

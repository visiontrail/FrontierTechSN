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
from backend.provider_credentials import redact_api_keys

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


async def _record_rate_limit_if_present(
    endpoint: str | None,
    detail: str,
    *,
    route_slot: str,
    provider_type: str,
    api_key_id: str,
    log: LogCallback | None,
    label: str,
) -> bool:
    folded = detail.casefold()
    if "429" not in folded and "rate_limit" not in folded and "请求数限制" not in detail:
        return False
    from backend.pipeline import provider_rate_limit

    cooldown = await provider_rate_limit.record_rate_limit(
        endpoint or config.AI_ENDPOINT,
        route_slot=route_slot,
        provider_type=provider_type,
        api_key_id=api_key_id,
        log=log,
        label=label,
    )
    return cooldown > 0


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
        # Disable Claude Code's own 11-request burst. The pipeline's outer
        # retry ladder is observable, bounded, and paced for OneAPI's quota.
        "CLAUDE_CODE_MAX_RETRIES": str(config.AGENT_INTERNAL_MAX_RETRIES),
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


@timed("model_response", "external_response")
async def _agent_complete_single(
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
    max_retries: int | None = None,
    route=None,
) -> tuple[str, object]:
    """Run one provider's Claude Agent SDK turn and return assistant text.

    Raises ``RuntimeError`` on empty/failed output after the configured retry
    limit for this provider route,
    matching the contract of the legacy ``_chat`` so ``digester.py`` can treat
    both backends the same."""
    # Imported lazily so the module (and the HTTP backend) load fine even when
    # the SDK isn't installed.
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        ResultMessage,
        TextBlock,
        ToolUseBlock,
        query,
    )

    from backend.pipeline import model_router

    if route is None:
        route = model_router.ModelRoute(
            slot="standalone",
            provider_id=None,
            provider_type="selected",
            provider_name="Selected provider",
            endpoint=endpoint or config.AI_ENDPOINT,
            model=model or config.AI_MODEL,
            api_key=api_key if api_key is not None else config.AI_API_KEY,
        )
    current_route = route
    resolved_model = (config.ANTHROPIC_MODEL or current_route.model or "").strip() or None
    env = build_agent_env(
        current_route.model,
        current_route.endpoint,
        current_route.api_key,
        max_tokens,
    )
    retry_limit = max(
        0,
        int(config.AI_MAX_RETRIES if max_retries is None else max_retries),
    )

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
    attempt = 0
    rate_limit_waits = 0
    tried_key_ids = {current_route.api_key_id}
    while attempt <= retry_limit:
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

        text_parts: list[str] = []
        result: ResultMessage | None = None
        diagnostics: deque[str] = deque(maxlen=DIAGNOSTIC_STDERR_LINES)
        assistant_errors: set[str] = set()
        rate_limited = False
        attempt_committed = False

        env = build_agent_env(
            current_route.model,
            current_route.endpoint,
            current_route.api_key,
            max_tokens,
        )
        base_options["env"] = env
        _log(
            log,
            f"{label}: using {current_route.api_key_id} "
            f"({current_route.api_key_index + 1}/{current_route.api_key_count})",
        )
        options = ClaudeAgentOptions(**base_options)
        options.stderr = lambda line: _keep_diagnostic(diagnostics, line)

        try:
            from backend.pipeline import provider_rate_limit

            await provider_rate_limit.wait_for_request_slot(
                current_route.endpoint,
                route_slot=current_route.slot,
                provider_type=current_route.provider_type,
                api_key_id=current_route.api_key_id,
                log=log,
                label=label,
            )
            # Bound the whole turn and close the generator on the way out —
            # that tears the transport down and kills the CLI instead of
            # leaking it. Internal CLI retries are disabled so all subsequent
            # attempts return through the shared OneAPI rate gate above.
            stream = query(prompt=prompt, options=options)
            try:
                async with asyncio.timeout(config.AGENT_TURN_TIMEOUT):
                    async for message in stream:
                        if isinstance(message, AssistantMessage):
                            message_error = getattr(message, "error", None)
                            if message_error:
                                # Claude CLI represents rejected requests as an
                                # AssistantMessage whose text is an API-error
                                # diagnostic. It is not model output and is
                                # safe to retry through the Galaxy rate gate.
                                assistant_errors.add(str(message_error))
                                continue
                            for block in message.content:
                                if isinstance(block, TextBlock):
                                    if block.text:
                                        attempt_committed = True
                                        text_parts.append(block.text)
                                elif (
                                    isinstance(block, ToolUseBlock)
                                    and block.name != "Skill"
                                ):
                                    # Project Skills only load read-only prompt
                                    # context.  If the following model turn is
                                    # rejected (for example by a 429), replaying
                                    # that load through another key/provider is
                                    # safe and is required for configured
                                    # failover to work.  Keep the commit boundary
                                    # for every other tool in case this helper is
                                    # granted additional capabilities later.
                                    attempt_committed = True
                        elif isinstance(message, ResultMessage):
                            result = message
            finally:
                # Cleanup must not be able to hang the stage it is unwinding.
                with contextlib.suppress(Exception):
                    async with asyncio.timeout(30):
                        await stream.aclose()

            content = "".join(text_parts).strip()
            if (
                not content
                and result is not None
                and result.result
                and not result.is_error
                and not assistant_errors
            ):
                content = result.result.strip()

            elapsed_ms = (time.perf_counter() - start) * 1000
            if content:
                usage = (result.usage if result else None) or {}
                usage_str = f", tokens={usage}" if usage else ""
                _log(
                    log,
                    f"{label}: responded in {elapsed_ms:.0f}ms ({len(content)} chars{usage_str})",
                )
                return content, current_route

            detail = (
                f"empty content from Claude Agent SDK (model={resolved_model or 'default'}"
                f", is_error={getattr(result, 'is_error', 'n/a')})"
            )
            if result is not None and result.is_error:
                errs = result.errors or [result.result or "unknown SDK error"]
                detail = f"Claude Agent SDK error: {'; '.join(str(e) for e in errs)}"
            detail += _diagnostic_tail(diagnostics)
            if assistant_errors:
                detail += f" | assistant error: {', '.join(sorted(assistant_errors))}"
            detail = redact_api_keys(detail, current_route.api_keys)
            _warn(log, f"{label} attempt {attempt + 1} failed: {detail}")
            rate_limited = await _record_rate_limit_if_present(
                current_route.endpoint,
                detail,
                route_slot=current_route.slot,
                provider_type=current_route.provider_type,
                api_key_id=current_route.api_key_id,
                log=log,
                label=label,
            )
            last_error = RuntimeError(detail)
        except TimeoutError:
            elapsed_ms = (time.perf_counter() - start) * 1000
            detail = (
                f"the `claude` CLI ran past AGENT_TURN_TIMEOUT "
                f"({config.AGENT_TURN_TIMEOUT}s, {elapsed_ms:.0f}ms elapsed) and was killed. "
                f"The underlying request ceiling is AGENT_REQUEST_TIMEOUT "
                f"({config.AGENT_REQUEST_TIMEOUT}s); later attempts are paced and started "
                "by the pipeline rather than retried inside the CLI"
            ) + _diagnostic_tail(diagnostics)
            if assistant_errors:
                detail += f" | assistant error: {', '.join(sorted(assistant_errors))}"
            detail = redact_api_keys(detail, current_route.api_keys)
            _warn(log, f"{label} attempt {attempt + 1} failed: {detail}")
            rate_limited = await _record_rate_limit_if_present(
                current_route.endpoint,
                detail,
                route_slot=current_route.slot,
                provider_type=current_route.provider_type,
                api_key_id=current_route.api_key_id,
                log=log,
                label=label,
            )
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
            if assistant_errors:
                detail += f" | assistant error: {', '.join(sorted(assistant_errors))}"
            detail = redact_api_keys(detail, current_route.api_keys)
            _warn(log, f"{label} attempt {attempt + 1} failed: {detail}")
            rate_limited = await _record_rate_limit_if_present(
                current_route.endpoint,
                detail,
                route_slot=current_route.slot,
                provider_type=current_route.provider_type,
                api_key_id=current_route.api_key_id,
                log=log,
                label=label,
            )
            last_error = RuntimeError(detail)

        if attempt_committed:
            raise model_router.ModelOutputCommittedError(
                f"{label} failed after model/tool output began; refusing to replay "
                f"through another key or provider: {last_error}"
            ) from last_error

        if rate_limited:
            alternate = current_route.next_untried_key(tried_key_ids)
            if alternate is not None:
                previous_key_id = current_route.api_key_id
                current_route = alternate
                tried_key_ids.add(current_route.api_key_id)
                _warn(
                    log,
                    f"{label}: rotating primary API key {previous_key_id} -> "
                    f"{current_route.api_key_id} before provider failover",
                )
                continue

        if (
            rate_limited
            and rate_limit_waits < max(0, int(config.AI_PRIMARY_RATE_LIMIT_MAX_WAITS))
        ):
            rate_limit_waits += 1
            _log(
                log,
                f"{label}: OneAPI quota wait {rate_limit_waits}/"
                f"{config.AI_PRIMARY_RATE_LIMIT_MAX_WAITS}; retrying the primary after "
                "the shared cooldown without consuming a provider failure attempt",
            )
            # The next loop enters wait_for_request_slot, which owns the exact
            # process-wide cooldown and coordinates concurrent task calls.
            tried_key_ids = {current_route.api_key_id}
            continue

        if attempt < retry_limit:
            delay = min(
                config.AI_RETRY_MAX_SECONDS,
                config.AI_RETRY_BASE_SECONDS * 2**attempt,
            )
            _log(
                log,
                f"{label}: retrying same provider turn in {delay:.0f}s "
                f"({attempt + 2}/{retry_limit + 1})",
            )
            with span("retry_backoff", "retry_wait"):
                await asyncio.sleep(delay)
        attempt += 1
        tried_key_ids = {current_route.api_key_id}

    raise RuntimeError(f"AI request failed via Claude Agent SDK: {last_error}") from last_error


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
    allow_provider_failover: bool = True,
    max_retries: int | None = None,
    route=None,
    credential_selected: Callable[[str], None] | None = None,
) -> str:
    """Complete a turn through the ordered OneAPI -> DeepSeek route.

    Each route owns the full retry ladder implemented by
    :func:`_agent_complete_single`.  No partial assistant content leaves that
    function, so moving to the backup cannot duplicate already-committed text
    or tool calls.
    """
    from backend.pipeline import model_router

    resolved_endpoint = endpoint or config.AI_ENDPOINT
    resolved_model = model or config.AI_MODEL
    resolved_api_key = api_key if api_key is not None else config.AI_API_KEY
    routes = (
        (route,)
        if route is not None
        else await model_router.resolve_model_routes(
            endpoint=resolved_endpoint,
            model=resolved_model,
            api_key=resolved_api_key,
            allow_failover=allow_provider_failover,
        )
    )
    if len(routes) > 1:
        primary_retry_limit = (
            max(0, int(max_retries))
            if max_retries is not None
            else model_router.max_retries_for_route(routes[0])
        )
        _log(
            log,
            f"{label}: model route {routes[0].audit_label} -> {routes[1].audit_label}; "
            f"backup activates only after {primary_retry_limit + 1} primary attempts",
        )

    last_error: Exception | None = None
    for index, route in enumerate(routes):
        route_retry_limit = (
            max(0, int(max_retries))
            if max_retries is not None
            else model_router.max_retries_for_route(route)
        )
        try:
            content, completed_route = await _agent_complete_single(
                system_prompt,
                user_content,
                model=route.model,
                endpoint=route.endpoint,
                api_key=route.api_key,
                max_tokens=max_tokens,
                enable_skills=enable_skills,
                disable_thinking=disable_thinking,
                log=log,
                label=label,
                max_retries=route_retry_limit,
                route=route,
            )
            if len(routes) > 1:
                _log(
                    log,
                    f"{label}: completed via {route.audit_label} "
                    f"using {completed_route.api_key_id}",
                )
            if credential_selected is not None:
                credential_selected(completed_route.api_key)
            return content
        except model_router.ModelOutputCommittedError:
            raise
        except Exception as exc:  # noqa: BLE001 - route exhaustion boundary
            last_error = exc
            next_index = index + 1
            if next_index < len(routes):
                next_route = routes[next_index]
                _warn(
                    log,
                    f"{label}: {route.audit_label} exhausted its "
                    f"{route_retry_limit + 1} attempts; switching to "
                    f"{next_route.audit_label}",
                )

    if len(routes) > 1:
        route_labels = " -> ".join(route.audit_label for route in routes)
        raise model_router.ModelRouteExhausted(
            f"All configured model routes failed after bounded retries: {route_labels}; "
            f"last_error={last_error.__class__.__name__ if last_error else 'unknown'}"
        ) from last_error
    if last_error is not None:
        raise last_error
    raise RuntimeError("No model route was available")


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
        enable_skills=False,
        allow_provider_failover=False,
        disable_thinking=True,
        label="Provider test",
    )
    if not text.strip():
        raise RuntimeError("Empty response from Claude Agent SDK")
    return int((time.perf_counter() - start) * 1000)

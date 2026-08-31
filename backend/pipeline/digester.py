import json
import asyncio
import logging
import re
import time
import httpx
from collections.abc import Callable
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urlsplit, urlunsplit
from backend import config
from backend.pipeline.extractors.base import ExtractedContent
from backend.provider_credentials import redact_api_keys

logger = logging.getLogger(__name__)
LogCallback = Callable[[str], None]
RouteSelectedCallback = Callable[[str, str, str], None]

MAX_CHUNK_WORDS = 6000
DEFAULT_MAX_OUTPUT_TOKENS = 4096
MAX_SCRIPT_OUTPUT_TOKENS = 32_768
MAX_REASONING_OUTPUT_TOKENS = 32_768
NON_ENGLISH_SCRIPT_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]")
SPEAKER_LINE_RE = re.compile(r"^\s*Speaker\s*(\d+)\s*[:：\-—–]\s*(.+)$", re.IGNORECASE)


class ReasoningBudgetExhausted(RuntimeError):
    """A reasoning model consumed the full adaptive output allowance."""


def _dlog(log: LogCallback | None, message: str):
    """Info line to the task log when present (start.sh + pipeline.log + UI),
    else the module logger (start.sh only). Prefer the callback to avoid
    double-logging to start.sh."""
    if log is not None:
        log(message)
    else:
        logger.info(message)


def _dwarn(log: LogCallback | None, message: str):
    """Warning that always reaches start.sh at WARNING level, and the in-app
    LogPanel too when a task callback is present."""
    logger.warning(message)
    if log is not None:
        log(message)


async def _resolve_provider(
    provider_id: int | None,
    ai_endpoint: str | None,
    ai_model: str | None,
) -> tuple[str, str, str]:
    """Resolve (endpoint, model, api_key) for a digestion call.

    Explicit ai_endpoint/ai_model overrides take precedence; otherwise the
    provider config is loaded from the database (by id, or the default
    provider when id is None); falling back to the .env defaults."""
    from backend import database

    endpoint = ai_endpoint
    model = ai_model
    api_key = config.AI_API_KEY

    if endpoint is None or model is None:
        row = await database.get_provider_raw(provider_id)
        if row is not None:
            endpoint = endpoint or row["endpoint"]
            model = model or row["model"]
            api_key = row["api_key"] or ""

    return (endpoint or config.AI_ENDPOINT, model or config.AI_MODEL, api_key)


async def _chat(
    system_prompt: str,
    user_content: str,
    endpoint: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    log: LogCallback | None = None,
    label: str = "AI call",
    max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    enable_skills: bool = True,
    disable_thinking: bool = False,
    route_selected: RouteSelectedCallback | None = None,
) -> str:
    """Single completion for a (system prompt, user content) pair.

    Dispatches to the Claude Agent SDK (default) or the legacy OpenAI-compatible
    HTTP client based on ``config.AI_BACKEND``. OneAPI receives the full
    bounded retry ladder before the call moves to the configured DeepSeek
    backup. Partial output is never exposed between routes."""
    from backend.pipeline import model_router

    routes = await model_router.resolve_model_routes(
        endpoint=endpoint or config.AI_ENDPOINT,
        model=model or config.AI_MODEL,
        api_key=api_key if api_key is not None else config.AI_API_KEY,
    )
    if len(routes) > 1:
        _dlog(
            log,
            f"{label}: model route {routes[0].audit_label} -> {routes[1].audit_label}; "
            "backup activates only after "
            f"{model_router.max_retries_for_route(routes[0]) + 1} primary attempts",
        )

    last_error: Exception | None = None
    for index, route in enumerate(routes):
        selected_api_key = route.api_key

        def remember_credential(api_key: str) -> None:
            nonlocal selected_api_key
            selected_api_key = api_key

        try:
            if config.AI_BACKEND == "agent_sdk":
                from backend.pipeline import agent

                try:
                    content = await agent.agent_complete(
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
                        # This dispatcher owns the route so the agent transport
                        # cannot independently perform the same failover twice.
                        allow_provider_failover=False,
                        max_retries=model_router.max_retries_for_route(route),
                        route=route,
                        credential_selected=remember_credential,
                    )
                    if len(routes) > 1:
                        _dlog(log, f"{label}: completed via {route.audit_label}")
                    if route_selected is not None:
                        route_selected(route.endpoint, route.model, selected_api_key)
                    return content
                except model_router.ModelOutputCommittedError:
                    raise
                except Exception as exc:
                    # When a backup exists, move forward after the primary's
                    # SDK retry ladder. HTTP fallback is reserved for the last
                    # route so a failed backup never jumps back to OneAPI.
                    if not config.AI_HTTP_FALLBACK or index + 1 < len(routes):
                        raise
                    _dwarn(
                        log,
                        f"{label}: Claude Agent SDK failed on {route.audit_label} "
                        f"({exc}); trying the OpenAI-compatible transport on "
                        "that same final route",
                    )

            content = await _chat_http(
                system_prompt,
                user_content,
                endpoint=route.endpoint,
                model=route.model,
                api_key=route.api_key,
                log=log,
                label=label,
                max_tokens=max_tokens,
                max_retries=model_router.max_retries_for_route(route),
                route=route,
                credential_selected=remember_credential,
            )
            if len(routes) > 1:
                _dlog(log, f"{label}: completed via {route.audit_label}")
            if route_selected is not None:
                route_selected(route.endpoint, route.model, selected_api_key)
            return content
        except model_router.ModelOutputCommittedError:
            raise
        except Exception as exc:  # noqa: BLE001 - provider route boundary
            last_error = exc
            next_index = index + 1
            if next_index < len(routes):
                next_route = routes[next_index]
                _dwarn(
                    log,
                    f"{label}: {route.audit_label} exhausted its "
                    f"{model_router.max_retries_for_route(route) + 1} attempts; switching to "
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


def _chat_completions_url(endpoint: str) -> str:
    """Normalise a provider endpoint to an OpenAI chat-completions URL.

    A provider row only has to carry the gateway host for the Agent SDK, which
    reinterprets it as an Anthropic base URL. The HTTP client needs the actual
    route, so a host-root (or bare ``/v1``) endpoint gets completed here rather
    than POSTing to the gateway's front page."""
    parts = urlsplit(endpoint.strip())
    if not parts.scheme or not parts.netloc:
        return endpoint
    path = parts.path.rstrip("/")
    if path.endswith("/chat/completions"):
        return endpoint
    suffix = "/chat/completions" if path.endswith("/v1") else "/v1/chat/completions"
    return urlunsplit((parts.scheme, parts.netloc, path + suffix, parts.query, ""))


def _retry_after_seconds(response: httpx.Response | None) -> float:
    """Return a bounded Retry-After delay from seconds or an HTTP date."""
    if response is None:
        return 0.0
    raw = response.headers.get("retry-after", "").strip()
    if not raw:
        return 0.0
    try:
        return max(0.0, min(3600.0, float(raw)))
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(raw)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            return max(
                0.0,
                min(3600.0, (retry_at - datetime.now(timezone.utc)).total_seconds()),
            )
        except (TypeError, ValueError, OverflowError):
            return 0.0


async def _chat_http(
    system_prompt: str,
    user_content: str,
    endpoint: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    log: LogCallback | None = None,
    label: str = "AI call",
    max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    max_retries: int | None = None,
    route=None,
    credential_selected: Callable[[str], None] | None = None,
) -> str:
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
    endpoint = _chat_completions_url(current_route.endpoint)
    model = current_route.model
    retry_limit = max(
        0,
        int(config.AI_MAX_RETRIES if max_retries is None else max_retries),
    )

    _dlog(log, f"{label}: POST {endpoint} (model={model}, ~{len(user_content.split())} words in)")

    async with httpx.AsyncClient(timeout=config.AI_TIMEOUT) as client:
        current_max_tokens = max_tokens
        attempt = 0
        rate_limit_waits = 0
        tried_key_ids = {current_route.api_key_id}
        while attempt <= retry_limit:
            start = time.perf_counter()
            retry_after_delay = 0.0
            quota_wait_requested = False
            _dlog(
                log,
                f"{label}: using {current_route.api_key_id} "
                f"({current_route.api_key_index + 1}/{current_route.api_key_count})",
            )
            try:
                prompt = system_prompt
                if attempt > 0:
                    prompt = (
                        f"{system_prompt}\n\n"
                        "Important retry instruction: output the final answer directly in the message content. "
                        "Do not spend tokens on hidden reasoning, analysis, markdown fences, or explanations. "
                        "Start immediately with the requested output."
                    )
                from backend.pipeline import provider_rate_limit

                await provider_rate_limit.wait_for_request_slot(
                    endpoint,
                    route_slot=current_route.slot,
                    api_key_id=current_route.api_key_id,
                    log=log,
                    label=label,
                )
                resp = await client.post(
                    endpoint,
                    headers={
                        "Authorization": f"Bearer {current_route.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": prompt},
                            {"role": "user", "content": user_content},
                        ],
                        "temperature": 0.3,
                        "max_tokens": current_max_tokens,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                choice = data["choices"][0]
                message = choice.get("message") or {}
                content = message.get("content") or ""
                elapsed_ms = (time.perf_counter() - start) * 1000
                usage = data.get("usage") or {}
                finish_reason = choice.get("finish_reason")
                usage_str = (
                    f", tokens={usage.get('prompt_tokens', '?')}+{usage.get('completion_tokens', '?')}"
                    f"={usage.get('total_tokens', '?')}"
                    if usage
                    else ""
                )
                finish_str = f", finish={finish_reason}" if finish_reason else ""
                _dlog(log, f"{label}: {model} responded in {elapsed_ms:.0f}ms ({len(content)} chars{usage_str}{finish_str})")
                if content.strip():
                    if credential_selected is not None:
                        credential_selected(current_route.api_key)
                    return content

                reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
                detail = (
                    f"empty message content from {model}"
                    f" (finish={finish_reason or 'unknown'}, max_tokens={current_max_tokens}, "
                    f"reasoning_chars={len(reasoning)})"
                )
                _dwarn(log, f"{label} attempt {attempt + 1} failed: {detail}")
                if finish_reason == "length" and reasoning:
                    expanded_max_tokens = min(
                        MAX_REASONING_OUTPUT_TOKENS,
                        current_max_tokens * 2,
                    )
                    if attempt < retry_limit and expanded_max_tokens > current_max_tokens:
                        current_max_tokens = expanded_max_tokens
                        _dwarn(
                            log,
                            f"{label}: hidden reasoning exhausted the output budget; "
                            f"retrying immediately with max_tokens={current_max_tokens}",
                        )
                        continue
                    raise ReasoningBudgetExhausted(
                        f"AI request returned {detail}; hidden reasoning exhausted "
                        "the maximum adaptive output budget"
                    )
                if attempt == retry_limit:
                    raise RuntimeError(f"AI request returned {detail}")
            except httpx.HTTPStatusError as e:
                # raise_for_status()'s message omits the response body, which is
                # where the API's real error detail lives (bad path, unknown
                # model, auth, quota...). Surface status + URL + body.
                body = e.response.text[:1000].strip() if e.response is not None else ""
                body = redact_api_keys(body, current_route.api_keys)
                detail = (
                    f"HTTP {e.response.status_code} {e.response.reason_phrase} "
                    f"from {e.request.method} {e.request.url} (model={model}); "
                    f"response body: {body or '<empty>'}"
                )
                _dwarn(log, f"{label} attempt {attempt + 1} failed: {detail}")
                retry_after_delay = _retry_after_seconds(e.response)
                if e.response is not None and e.response.status_code == 429:
                    from backend.pipeline import provider_rate_limit

                    retry_after_delay = max(
                        retry_after_delay,
                        await provider_rate_limit.record_rate_limit(
                            endpoint,
                            route_slot=current_route.slot,
                            api_key_id=current_route.api_key_id,
                            log=log,
                            label=label,
                        ),
                    )
                    alternate = current_route.next_untried_key(tried_key_ids)
                    if alternate is not None:
                        previous_key_id = current_route.api_key_id
                        current_route = alternate
                        tried_key_ids.add(current_route.api_key_id)
                        _dwarn(
                            log,
                            f"{label}: rotating primary API key {previous_key_id} -> "
                            f"{current_route.api_key_id} before provider failover",
                        )
                        continue
                    quota_wait_requested = (
                        retry_after_delay > 0
                        and rate_limit_waits
                        < max(0, int(config.AI_PRIMARY_RATE_LIMIT_MAX_WAITS))
                    )
                if (
                    attempt < retry_limit
                    and current_max_tokens > DEFAULT_MAX_OUTPUT_TOKENS
                    and e.response is not None
                    and e.response.status_code in (400, 422)
                    and "max_tokens" in body.lower()
                ):
                    current_max_tokens = DEFAULT_MAX_OUTPUT_TOKENS
                    _dwarn(log, f"{label}: retrying with max_tokens={current_max_tokens}")
                    continue
                if attempt == retry_limit and not quota_wait_requested:
                    raise RuntimeError(f"AI request failed: {detail}") from e
            except httpx.RequestError as e:
                # Connection refused, DNS failure, timeout — no response body.
                detail = f"{e.__class__.__name__} connecting to {endpoint} (model={model}): {e}"
                _dwarn(log, f"{label} attempt {attempt + 1} failed: {detail}")
                if attempt == retry_limit:
                    raise RuntimeError(f"AI request failed: {detail}") from e
            except ReasoningBudgetExhausted:
                raise
            except Exception as e:
                _dwarn(log, f"{label} attempt {attempt + 1} failed: {e.__class__.__name__}: {e}")
                if attempt == retry_limit:
                    raise

            if quota_wait_requested:
                rate_limit_waits += 1
                _dlog(
                    log,
                    f"{label}: OneAPI quota wait {rate_limit_waits}/"
                    f"{config.AI_PRIMARY_RATE_LIMIT_MAX_WAITS}; retrying the primary after "
                    "the shared cooldown without consuming a provider failure attempt",
                )
                tried_key_ids = {current_route.api_key_id}
                continue

            if attempt < retry_limit:
                delay = min(
                    config.AI_RETRY_MAX_SECONDS,
                    config.AI_RETRY_BASE_SECONDS * 2**attempt,
                )
                delay = max(delay, retry_after_delay)
                _dlog(
                    log,
                    f"{label}: retrying same provider call in {delay:.0f}s "
                    f"({attempt + 2}/{retry_limit + 1})",
                )
                await asyncio.sleep(delay)
            attempt += 1
            tried_key_ids = {current_route.api_key_id}


async def test_connection(endpoint: str, model: str, api_key: str) -> int:
    """Send a minimal request to verify the provider is reachable and the model
    responds. Returns the round-trip latency in milliseconds, or raises on any
    failure. Routes through whichever backend ``config.AI_BACKEND`` selects."""
    if config.AI_BACKEND == "agent_sdk":
        from backend.pipeline import agent

        return await agent.test_connection(endpoint, model, api_key)

    start = time.perf_counter()
    async with httpx.AsyncClient(timeout=config.AI_TIMEOUT) as client:
        resp = await client.post(
            _chat_completions_url(endpoint),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 1,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        # Validate the response has the expected OpenAI-compatible shape.
        data["choices"][0]["message"]
    return int((time.perf_counter() - start) * 1000)


def _chunk_text(text: str, max_words: int = MAX_CHUNK_WORDS) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text]
    chunks = []
    for i in range(0, len(words), max_words):
        chunks.append(" ".join(words[i:i + max_words]))
    return chunks


def _contains_cjk(text: str) -> bool:
    return bool(NON_ENGLISH_SCRIPT_RE.search(text))


def _script_max_tokens(word_count: int) -> int:
    return max(DEFAULT_MAX_OUTPUT_TOKENS, min(MAX_SCRIPT_OUTPUT_TOKENS, word_count * 4))


def _normalize_script_lines(script: str, script_format: str, log: LogCallback | None = None) -> list[str]:
    clean = script.strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    lines = []
    for raw_line in clean.splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        match = SPEAKER_LINE_RE.match(raw_line)
        if match:
            text = match.group(2).strip()
            if text:
                lines.append(text)
            continue
        lines.append(raw_line)

    if lines and any(SPEAKER_LINE_RE.match(line) for line in clean.splitlines()):
        _dlog(log, "Removed speaker labels from generated script for TTS-safe output")

    return lines


def _ensure_closing_remarks(
    lines: list[str],
    closing_remarks: str,
    script_format: str,
    log: LogCallback | None = None,
) -> list[str]:
    """Guarantee that the configured spoken close is the final script beat."""
    closing_lines = _normalize_script_lines(closing_remarks, script_format)
    if not closing_lines:
        return lines

    spoken_script = " ".join(" ".join(lines).split()).casefold()
    spoken_closing = " ".join(" ".join(closing_lines).split()).casefold()
    if spoken_script.endswith(spoken_closing):
        # Line wrapping is only a pacing hint. Do not duplicate an otherwise
        # verbatim close merely because the model split it across two beats.
        return lines

    _dlog(log, "Added the configured closing remarks to the final narration")
    return [*lines, *closing_lines]


def _summary_from_curated_highlights(content: ExtractedContent) -> dict:
    highlights = content.metadata.get("curated_highlights") or []
    talking_points = []
    key_quotes = []

    for item in highlights:
        if not isinstance(item, dict):
            continue
        quote = str(item.get("highlightText") or item.get("highlight_text") or "").strip()
        if not quote:
            continue
        chapter = str(item.get("chapterTitle") or item.get("chapter_title") or "").strip()
        reason = str(item.get("selectionReason") or item.get("selection_reason") or "").strip()
        note = str(item.get("noteText") or item.get("note_text") or "").strip()
        post_title = str(item.get("postTitle") or item.get("post_title") or "").strip()
        post_description = str(item.get("postDescription") or item.get("post_description") or "").strip()

        talking_points.append(
            {
                "topic": post_title or chapter or "Curated highlight",
                "detail": post_description or note or reason or quote,
                "why_it_matters": reason,
                "chapter": chapter,
            }
        )
        key_quotes.append(quote)

    return {
        "title": content.title,
        "thesis": "This book discussion is based on Isla-Reader's pre-selected highlights, focusing on the most quotable and discussion-worthy ideas.",
        "talking_points": talking_points,
        "key_quotes": key_quotes,
        "discussion_angles": [
            "Open by explaining why these highlights stood out.",
            "Connect quotes across chapters into a coherent argument.",
            "Let the two hosts react naturally to the strongest lines and reader notes.",
        ],
        "source_format": "curated_highlights",
    }


async def summarize(
    content: ExtractedContent,
    ai_endpoint: str | None = None,
    ai_model: str | None = None,
    provider_id: int | None = None,
    log: LogCallback | None = None,
) -> dict:
    _dlog(log, f"Summarizing content: {content.title} ({len(content.text.split())} words)")

    if content.metadata.get("processing_mode") == "curated_highlights":
        _dlog(log, "Using Isla-Reader curated highlights as pre-selected talking points")
        return _summary_from_curated_highlights(content)

    endpoint, model, api_key = await _resolve_provider(provider_id, ai_endpoint, ai_model)
    system_prompt = (config.PROMPTS_DIR / "summarize.txt").read_text()

    chunks = _chunk_text(content.text)
    _dlog(log, f"Summarizing in {len(chunks)} chunk(s)")

    if len(chunks) == 1:
        result = await _chat(system_prompt, content.text, endpoint, model, api_key, log, "Summarize")
    else:
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            partial = await _chat(
                system_prompt, chunk, endpoint, model, api_key, log, f"Summarize chunk {i + 1}/{len(chunks)}"
            )
            chunk_summaries.append(partial)

        merge_prompt = (
            "You are a content analyst. Merge these partial summaries into a single coherent summary. "
            "Output the same JSON format as the individual summaries, combining the best talking points "
            "and key quotes from all parts. Keep only the top 5-8 talking points and 3-5 quotes. "
            "All JSON string values MUST be in English; translate any non-English source material."
        )
        combined = "\n\n---\n\n".join(chunk_summaries)
        result = await _chat(merge_prompt, combined, endpoint, model, api_key, log, "Summarize merge")

    try:
        clean = result.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1].rsplit("```", 1)[0]
        parsed = json.loads(clean)
        _dlog(log, f"Summary parsed: {len(parsed.get('talking_points', []))} talking points, {len(parsed.get('key_quotes', []))} quotes")
        return parsed
    except json.JSONDecodeError:
        _dwarn(log, "Failed to parse summary JSON, returning raw text")
        return {"title": content.title, "thesis": result, "talking_points": [], "key_quotes": [], "discussion_angles": []}


SCRIPT_PROMPT_FILES = {
    "monologue": "scriptwrite_monologue.txt",
    "dialogue": "scriptwrite_dialogue.txt",
}


async def generate_script(
    summary: dict,
    target_duration_minutes: int = 10,
    script_format: str = "monologue",
    ai_endpoint: str | None = None,
    ai_model: str | None = None,
    provider_id: int | None = None,
    closing_remarks: str = "",
    log: LogCallback | None = None,
) -> str:
    word_count = target_duration_minutes * 150
    prompt_file = SCRIPT_PROMPT_FILES.get(script_format, SCRIPT_PROMPT_FILES["monologue"])
    _dlog(
        log,
        f"Generating {script_format} script (~{word_count} words, {target_duration_minutes} min) "
        f"using {prompt_file}",
    )

    endpoint, model, api_key = await _resolve_provider(provider_id, ai_endpoint, ai_model)
    system_prompt = (config.PROMPTS_DIR / prompt_file).read_text()
    system_prompt = system_prompt.replace("{word_count}", str(word_count))
    system_prompt = system_prompt.replace("{duration_minutes}", str(target_duration_minutes))
    if closing_remarks:
        system_prompt += (
            "\n\nClosing Remarks (MANDATORY):\n"
            "- End the script with the exact spoken text below, verbatim.\n"
            "- Treat it as the final beat of the narrative, with no spoken text after it.\n"
            "- The target word count includes this closing text.\n\n"
            f"{closing_remarks}"
        )

    user_content = json.dumps(summary, indent=2, ensure_ascii=False)
    script_tokens = _script_max_tokens(word_count)
    script = await _chat(
        system_prompt,
        user_content,
        endpoint,
        model,
        api_key,
        log,
        "Scriptwrite",
        max_tokens=script_tokens,
    )

    lines = _normalize_script_lines(script, script_format, log)
    if not lines:
        raise RuntimeError("Generated script contains no spoken lines")

    joined = "\n".join(lines)
    if _contains_cjk(joined):
        _dwarn(log, "Generated script contained non-English/CJK text; requesting English-only repair")
        repair_prompt = (
            "You repair podcast scripts for a TTS pipeline. Rewrite the user's script in natural spoken "
            "English only. Translate all Chinese or other non-English text into English. Preserve the same "
            "one-turn-per-line structure. Do not include Speaker labels, host names, prefixes, headings, "
            "markdown, or stage directions. Output only the repaired spoken lines."
        )
        repaired = await _chat(
            repair_prompt,
            joined,
            endpoint,
            model,
            api_key,
            log,
            "Scriptwrite repair",
            max_tokens=script_tokens,
        )
        lines = _normalize_script_lines(repaired, script_format, log)
        joined = "\n".join(lines)
        if not lines:
            raise RuntimeError("English repair produced no spoken lines")
        if _contains_cjk(joined):
            raise RuntimeError("Generated script still contains non-English/CJK text after repair")

    lines = _ensure_closing_remarks(lines, closing_remarks, script_format, log)
    joined = "\n".join(lines)

    _dlog(log, f"Script generated: {len(lines)} speaker turns, {len(' '.join(lines).split())} words")
    return joined

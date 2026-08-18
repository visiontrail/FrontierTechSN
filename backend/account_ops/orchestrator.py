from __future__ import annotations

import json
import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from backend import config, database
from backend.account_ops.opencode import run_content_agent
from backend.account_ops.x_engagement import (
    account_switch_prompt,
    engagement_prompt,
    parse_engagement_result,
    recover_action_urls,
    run_x_operational_agent,
    validate_engagement_result,
)
from backend.models import (
    AccountAutomationExecutor,
    AccountAutomationFeature,
    AccountAutomationResponse,
    AccountRunResponse,
    AccountRunStatus,
)
from backend.pipeline.opencli import OpenCLIError, first_json, run_opencli

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
_CONTENT_FIELDS = (
    "title",
    "year",
    "event_summary",
    "historical_reflection",
    "post_text",
    "image_prompt",
    "source_notes",
)
_X_POST_LIMIT = 280
_X_POST_MINIMUM = 80


def _rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        for key in ("data", "items", "results", "rows"):
            nested = value.get(key)
            if isinstance(nested, list):
                return [item for item in nested if isinstance(item, dict)]
        return [value]
    return []


def _field(row: dict[str, Any], name: str) -> str:
    for key, value in row.items():
        if str(key).lower() == name.lower():
            return str(value or "").strip().lstrip("📁🔗").strip()
    return ""


def _render_prompt(template: str, event_date: str) -> str:
    parsed = date.fromisoformat(event_date)
    replacements = {
        "{date}": event_date,
        "{year}": str(parsed.year),
        "{month}": str(parsed.month),
        "{month_name}": parsed.strftime("%B"),
        "{day}": str(parsed.day),
    }
    result = template
    for token, value in replacements.items():
        result = result.replace(token, value)
    return result


def _json_dicts(value: Any, *, depth: int = 0) -> list[dict[str, Any]]:
    """Recover JSON objects from mixed OpenCode narration and event output."""
    if depth > 2:
        return []
    if isinstance(value, dict):
        candidates = [value]
        for nested in value.values():
            if isinstance(nested, (dict, list, str)):
                candidates.extend(_json_dicts(nested, depth=depth + 1))
        return candidates
    if isinstance(value, list):
        candidates: list[dict[str, Any]] = []
        for nested in value:
            candidates.extend(_json_dicts(nested, depth=depth + 1))
        return candidates
    if not isinstance(value, str):
        return []

    decoder = json.JSONDecoder()
    candidates = []
    for index, character in enumerate(value):
        if character not in "[{":
            continue
        try:
            parsed, _ = decoder.raw_decode(value[index:])
        except json.JSONDecodeError:
            continue
        candidates.extend(_json_dicts(parsed, depth=depth + 1))
    return candidates


def _fit_x_post(value: Any) -> str:
    """Keep a planned post inside X's limit without cutting through a word."""
    post_text = re.sub(r"\s+", " ", str(value)).strip()
    if len(post_text) <= _X_POST_LIMIT:
        return post_text

    prefix = post_text[: _X_POST_LIMIT - 1].rstrip()
    sentence_end = max(prefix.rfind(mark) for mark in ".!?。！？")
    # Only prefer a complete sentence when it uses most of the budget. Cutting
    # at a much earlier period can discard the historical reflection entirely.
    if sentence_end + 1 >= _X_POST_LIMIT - 40:
        return prefix[: sentence_end + 1].rstrip()

    word_end = prefix.rfind(" ")
    if word_end >= _X_POST_MINIMUM:
        prefix = prefix[:word_end]
    return prefix.rstrip(" ,;:—–-") + "…"


def _content_object(value: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, dict):
        parsed = value
    else:
        candidates = _json_dicts(value)
        complete = [
            candidate
            for candidate in candidates
            if all(candidate.get(key) for key in _CONTENT_FIELDS)
        ]
        if complete:
            # OpenCode emits intermediate tool text before its final response.
            # Prefer the last object that actually satisfies the content contract.
            parsed = complete[-1]
        elif candidates:
            parsed = max(
                candidates,
                key=lambda candidate: sum(
                    bool(candidate.get(key)) for key in _CONTENT_FIELDS
                ),
            )
        else:
            parsed = None
    if not isinstance(parsed, dict):
        raise OpenCLIError("Content planner did not return a JSON object")
    missing = [key for key in _CONTENT_FIELDS if not parsed.get(key)]
    if missing:
        raise OpenCLIError(
            "Content planner omitted required fields: " + ", ".join(missing)
        )
    post_text = _fit_x_post(parsed["post_text"])
    if len(post_text) < _X_POST_MINIMUM:
        raise OpenCLIError("Planned X post is too short to carry historical context")
    sources = parsed.get("source_notes")
    if not isinstance(sources, list) or len(sources) < 2:
        raise OpenCLIError("Content planner must provide at least two source notes")
    parsed["post_text"] = post_text
    return parsed


def _conversation_url(text: str) -> str:
    match = re.search(r"https://chatgpt\.com/c/[a-zA-Z0-9-]+", text)
    return match.group(0) if match else ""


async def _plan_with_opencli(prompt: str) -> tuple[dict[str, Any], str, dict[str, Any]]:
    timeout = min(max(config.OPENCLI_TIMEOUT, 180), 480)
    result = await run_opencli(
        [
            "chatgpt",
            "ask",
            prompt,
            "--new",
            "--timeout",
            str(timeout),
            "--window",
            "background",
            "--site-session",
            "persistent",
            "--keep-tab",
            "false",
            "-f",
            "json",
        ],
        timeout=timeout + 45,
    )
    rows = _rows(first_json(result.stdout))
    if not rows:
        raise OpenCLIError("ChatGPT planning command returned no result row")
    response = _field(rows[0], "response")
    if not response:
        raise OpenCLIError("ChatGPT planning command returned an empty response")
    conversation_url = _field(rows[0], "conversationUrl")
    return _content_object(response), conversation_url, {"opencli_stdout": result.stdout}


async def _plan_content(
    automation: AccountAutomationResponse,
    run: AccountRunResponse,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    prompt = _render_prompt(automation.prompt_template, run.event_date)
    prompt += (
        f"\n\nThe operational date is {run.event_date}. Return the post in English. "
        "The post_text value must be 80-260 characters so it remains safely within "
        "X's 280-character limit after normalization. "
        "Treat source_notes as provenance for the operator; do not append a source list "
        "to the X post unless it fits naturally."
    )
    if automation.executor == AccountAutomationExecutor.OPENCODE:
        result = await run_content_agent(
            model=automation.opencode_model,
            content_prompt=prompt,
            event_date=run.event_date,
        )
        content = _content_object(result.text)
        conversation_url = _conversation_url(result.stdout)
        return content, conversation_url, {
            "opencode_session_id": result.session_id,
            "opencode_stdout": result.stdout,
            "opencode_stderr": result.stderr,
        }
    return await _plan_with_opencli(prompt)


def _new_images(image_dir: Path, before: set[Path]) -> list[Path]:
    candidates = [
        path.resolve()
        for path in image_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in _IMAGE_SUFFIXES
        and path.resolve() not in before
        and path.stat().st_size > 32
    ]
    return sorted(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))


def _prepare_publish_image(image_path: Path, output_dir: Path) -> Path:
    """Create a compact upload copy for browser bridges with message-size limits."""
    output_path = output_dir / "publish.jpg"
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        image.thumbnail((1400, 1400), Image.Resampling.LANCZOS)
        for quality in (88, 82, 76, 70):
            image.save(
                output_path,
                format="JPEG",
                quality=quality,
                optimize=True,
                progressive=True,
            )
            if output_path.stat().st_size <= 700_000:
                break
    return output_path


async def _generate_image(
    content: dict[str, Any], image_dir: Path
) -> tuple[Path, str, str]:
    image_dir.mkdir(parents=True, exist_ok=True)
    before = {path.resolve() for path in image_dir.iterdir() if path.is_file()}
    image_prompt = (
        str(content["image_prompt"]).strip()
        + "\nCreate one landscape 3:2 editorial image for a historically serious social "
        "media post. Favor material detail, period-authentic clothing and architecture, "
        "restrained color, and a documentary sensibility. No text anywhere in the image."
    )
    timeout = max(config.THUMBNAIL_CHATGPT_TIMEOUT, 300)
    result = await run_opencli(
        [
            "chatgpt",
            "image",
            image_prompt,
            "--op",
            str(image_dir.resolve()),
            "--timeout",
            str(timeout),
            "--window",
            "background",
            "--site-session",
            "persistent",
            "--keep-tab",
            "false",
            "-f",
            "json",
        ],
        timeout=timeout + 60,
    )
    rows = _rows(first_json(result.stdout))
    reported: list[Path] = []
    root = image_dir.resolve()
    for row in rows:
        raw = _field(row, "file")
        if not raw or raw == "-":
            continue
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = config.PROJECT_ROOT / candidate
        candidate = candidate.resolve()
        if root in candidate.parents and candidate.is_file():
            reported.append(candidate)
    images = list(dict.fromkeys([*reported, *_new_images(image_dir, before)]))
    if not images:
        raise OpenCLIError("ChatGPT image generation produced no downloaded image")
    link = next((_field(row, "link") for row in rows if _field(row, "link")), "")
    return images[0], link, result.stdout


async def _verify_account(expected_handle: str) -> dict[str, Any]:
    result = await run_opencli(
        [
            "twitter",
            "whoami",
            "--window",
            "background",
            "--site-session",
            "ephemeral",
            "--keep-tab",
            "false",
            "-f",
            "json",
        ],
        timeout=60,
    )
    rows = _rows(first_json(result.stdout))
    if not rows:
        raise OpenCLIError("X account check returned no result")
    actual = _field(rows[0], "username").lstrip("@")
    if actual.casefold() != expected_handle.lstrip("@").casefold():
        raise OpenCLIError(
            f"Refusing to publish: expected @{expected_handle}, browser is @{actual or 'unknown'}"
        )
    return rows[0]


async def _ensure_account(
    automation: AccountAutomationResponse,
    *,
    run_id: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Verify the commissioned account, prompting an agent to switch if needed."""
    try:
        return await _verify_account(automation.account_handle), None
    except OpenCLIError as mismatch:
        agent_result = await run_x_operational_agent(
            automation.executor,
            model=automation.opencode_model,
            prompt=account_switch_prompt(automation.account_handle, run_id=run_id),
            title=f"X account switch · @{automation.account_handle}",
        )
        account = await _verify_account(automation.account_handle)
        return account, {
            "initial_check_error": str(mismatch),
            "agent_session_id": agent_result.session_id,
            "agent_result": agent_result.text,
            "agent_raw": agent_result.raw,
        }


def _fingerprint(text: str) -> str:
    """Normalised form used to recognise our own post on the timeline."""
    return re.sub(r"\s+", " ", text).strip().casefold()


async def _find_published_post(handle: str, post_text: str) -> dict[str, Any] | None:
    """Look for ``post_text`` on the account's timeline.

    A ``twitter post`` that dies mid-flight — the composer hanging behind a
    native dialog is the one seen in production — says nothing about whether X
    accepted the tweet. Recording that run as a clean failure is what invites a
    duplicate on the operator's next retry, so the timeline is the tiebreaker.
    Returns the matching row, or ``None`` when the post is provably absent;
    raises when the timeline itself could not be read."""
    result = await run_opencli(
        [
            "twitter",
            "tweets",
            handle.lstrip("@"),
            "--limit",
            "5",
            "--window",
            "background",
            "--site-session",
            "ephemeral",
            "--keep-tab",
            "false",
            "-f",
            "json",
        ],
        timeout=90,
    )
    needle = _fingerprint(post_text)[:40]
    for row in _rows(first_json(result.stdout)):
        if needle and needle in _fingerprint(_field(row, "text")):
            return row
    return None


async def _publish(post_text: str, image_path: Path, handle: str) -> tuple[str, str, str]:
    try:
        result = await run_opencli(
            [
                "twitter",
                "post",
                post_text,
                "--images",
                str(image_path.resolve()),
                "--window",
                "background",
                "--site-session",
                "ephemeral",
                "--keep-tab",
                "false",
                "-f",
                "json",
            ],
            timeout=max(config.OPENCLI_TIMEOUT, 180),
        )
    except OpenCLIError as exc:
        try:
            posted = await _find_published_post(handle, post_text)
        except Exception as check_error:  # noqa: BLE001 - the browser is already unhealthy
            raise OpenCLIError(
                f"{exc}\n\nPublication state is UNKNOWN: the timeline check also failed "
                f"({check_error}). Look at @{handle} before retrying — the post may have "
                f"gone out before the browser stopped responding."
            ) from exc
        if posted is not None:
            url = _field(posted, "url")
            if url:
                return url, _field(posted, "id"), json.dumps(
                    {"recovered_from_error": str(exc), "timeline_row": posted},
                    ensure_ascii=False,
                )
        raise OpenCLIError(
            f"{exc}\n\nThe post is not on @{handle}'s timeline, so nothing was published "
            f"and this run is safe to retry."
        ) from exc

    rows = _rows(first_json(result.stdout))
    if not rows:
        raise OpenCLIError("X publish command returned no result row")
    url = _field(rows[0], "url")
    post_id = _field(rows[0], "id")
    if not url:
        raise OpenCLIError("X publish command did not return a post URL")
    result_handle = re.search(
        r"^https://(?:x\.com|twitter\.com)/([^/]+)/status/\d+", url
    )
    actual_handle = result_handle.group(1) if result_handle else ""
    if actual_handle.casefold() != handle.lstrip("@").casefold():
        raise OpenCLIError(
            "X reported a successful publish from the wrong account "
            f"@{actual_handle or 'unknown'}: {url}. No retry was attempted."
        )
    return url, post_id, result.stdout


def _write_manifest(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def _recent_engaged_urls(runs: list[AccountRunResponse]) -> set[str]:
    urls: set[str] = set()
    for previous in runs:
        if previous.feature_type != AccountAutomationFeature.X_ENGAGEMENT:
            continue
        for key in ("replies", "quote_reposts"):
            rows = previous.content.get(key)
            if not isinstance(rows, list):
                continue
            for row in rows:
                if isinstance(row, dict) and isinstance(row.get("target_url"), str):
                    urls.add(row["target_url"].strip())
    return urls


def _status_id(url: str | None) -> str | None:
    match = re.search(r"/status/(\d+)", url or "")
    return match.group(1) if match else None


async def _execute_engagement_run(
    run: AccountRunResponse,
    automation: AccountAutomationResponse,
) -> None:
    output_dir = config.OUTPUTS_DIR / "account-operations" / run.id
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest: dict[str, Any] = {
        "run_id": run.id,
        "automation_id": automation.id,
        "feature_type": automation.feature_type.value,
        "status": AccountRunStatus.PLANNING.value,
        "event_date": run.event_date,
        "executor": automation.executor.value,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }

    async def log(message: str) -> None:
        timestamped = datetime.now(timezone.utc).isoformat(timespec="seconds") + " " + message
        await database.append_account_run_log(run.id, timestamped)

    try:
        await log(f"Verifying the active X account is @{automation.account_handle}")
        account, switch_audit = await _ensure_account(automation, run_id=run.id)
        manifest["verified_account"] = account
        if switch_audit:
            manifest["account_switch"] = switch_audit
            await log(f"Switched X to @{automation.account_handle} and verified it")

        previous_runs = await database.list_account_runs(limit=200)
        exclusions = _recent_engaged_urls(previous_runs)
        await database.update_account_run(
            run.id,
            status=AccountRunStatus.PUBLISHING.value,
        )
        manifest["status"] = AccountRunStatus.PUBLISHING.value
        await log(
            f"Starting {automation.executor.value} agent on Following; "
            f"limits {automation.max_replies} replies/{automation.max_quote_reposts} quotes"
        )
        agent_result = await run_x_operational_agent(
            automation.executor,
            model=automation.opencode_model,
            prompt=engagement_prompt(
                automation,
                run_id=run.id,
                excluded_urls=exclusions,
            ),
            title=f"X engagement · {run.event_date} · @{automation.account_handle}",
        )
        manifest.update(
            {
                "agent_session_id": agent_result.session_id,
                "agent_text": agent_result.text,
                "agent_raw": agent_result.raw,
            }
        )
        _write_manifest(manifest_path, manifest)
        preliminary = parse_engagement_result(agent_result.text)
        manifest["agent_result"] = preliminary
        _write_manifest(manifest_path, manifest)
        await database.update_account_run(
            run.id,
            title="Engagement · verifying published actions",
            content_json=json.dumps(preliminary, ensure_ascii=False),
        )
        content = await recover_action_urls(preliminary, automation)
        content = validate_engagement_result(
            content,
            automation,
            excluded_urls=exclusions,
        )
        final_account = await _verify_account(automation.account_handle)
        replies = content["replies"]
        quotes = content["quote_reposts"]
        actions = [*replies, *quotes]
        first_action = actions[0] if actions else {}
        first_text = str(
            first_action.get("reply_text") or first_action.get("quote_text") or ""
        ).strip()
        first_url = str(first_action.get("result_url") or "").strip() or None
        title = f"Engagement · {len(replies)} replies · {len(quotes)} quotes"
        completed_at = datetime.now(timezone.utc).isoformat()
        manifest.update(
            {
                "status": AccountRunStatus.PUBLISHED.value,
                "content": content,
                "verified_account_after_run": final_account,
                "completed_at": completed_at,
            }
        )
        _write_manifest(manifest_path, manifest)
        await database.update_account_run(
            run.id,
            status=AccountRunStatus.PUBLISHED.value,
            completed_at=completed_at,
            title=title,
            post_text=first_text or None,
            post_url=first_url,
            external_post_id=_status_id(first_url),
            content_json=json.dumps(content, ensure_ascii=False),
            error_message=None,
        )
        await log(
            f"Engagement completed: {len(replies)} replies and {len(quotes)} quote-reposts"
        )
    except Exception as exc:
        completed_at = datetime.now(timezone.utc).isoformat()
        manifest.update(
            {
                "status": AccountRunStatus.FAILED.value,
                "completed_at": completed_at,
                "error": str(exc),
            }
        )
        _write_manifest(manifest_path, manifest)
        await database.update_account_run(
            run.id,
            status=AccountRunStatus.FAILED.value,
            completed_at=completed_at,
            error_message=str(exc),
        )
        await log(f"Run failed: {exc}")
        raise


async def execute_run(run: AccountRunResponse) -> None:
    automation = await database.get_account_automation(run.automation_id)
    if automation is None:
        raise RuntimeError(f"Automation {run.automation_id} no longer exists")
    if automation.feature_type == AccountAutomationFeature.X_ENGAGEMENT:
        await _execute_engagement_run(run, automation)
        return
    output_dir = config.OUTPUTS_DIR / "account-operations" / run.id
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest: dict[str, Any] = {
        "run_id": run.id,
        "automation_id": automation.id,
        "status": AccountRunStatus.PLANNING.value,
        "event_date": run.event_date,
        "executor": automation.executor.value,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }

    async def log(message: str) -> None:
        timestamped = datetime.now(timezone.utc).isoformat(timespec="seconds") + " " + message
        await database.append_account_run_log(run.id, timestamped)

    try:
        await log(f"Planning Today in History content via {automation.executor.value}")
        content, conversation_url, planner_raw = await _plan_content(automation, run)
        manifest.update({"content": content, "planner": planner_raw})
        await database.update_account_run(
            run.id,
            title=str(content["title"]),
            post_text=str(content["post_text"]),
            content_json=json.dumps(content, ensure_ascii=False),
            chatgpt_conversation_url=conversation_url or None,
            status=AccountRunStatus.GENERATING_IMAGE.value,
        )
        await log("Generating the historically grounded image through ChatGPT Web")
        image_path, image_link, image_raw = await _generate_image(
            content, output_dir / "images"
        )
        publish_image_path = _prepare_publish_image(image_path, output_dir)
        conversation_url = conversation_url or image_link
        manifest.update(
            {
                "status": AccountRunStatus.PUBLISHING.value,
                "image_path": str(image_path),
                "publish_image_path": str(publish_image_path),
                "image_opencli_stdout": image_raw,
                "chatgpt_conversation_url": conversation_url,
            }
        )
        await database.update_account_run(
            run.id,
            status=AccountRunStatus.PUBLISHING.value,
            image_path=str(image_path),
            chatgpt_conversation_url=conversation_url or None,
        )
        await log(f"Verifying the active X account is @{automation.account_handle}")
        account, switch_audit = await _ensure_account(automation, run_id=run.id)
        manifest["verified_account"] = account
        if switch_audit:
            manifest["account_switch"] = switch_audit
            await log(f"Switched X to @{automation.account_handle} and verified it")
        await log("Publishing the post and image to X")
        post_url, post_id, publish_raw = await _publish(
            str(content["post_text"]), publish_image_path, automation.account_handle
        )
        completed_at = datetime.now(timezone.utc).isoformat()
        manifest.update(
            {
                "status": AccountRunStatus.PUBLISHED.value,
                "post_url": post_url,
                "external_post_id": post_id,
                "publish_opencli_stdout": publish_raw,
                "completed_at": completed_at,
            }
        )
        _write_manifest(manifest_path, manifest)
        await database.update_account_run(
            run.id,
            status=AccountRunStatus.PUBLISHED.value,
            completed_at=completed_at,
            post_url=post_url,
            external_post_id=post_id or None,
            error_message=None,
        )
        await log(f"Published successfully: {post_url}")
    except Exception as exc:
        completed_at = datetime.now(timezone.utc).isoformat()
        manifest.update(
            {
                "status": AccountRunStatus.FAILED.value,
                "completed_at": completed_at,
                "error": str(exc),
            }
        )
        _write_manifest(manifest_path, manifest)
        await database.update_account_run(
            run.id,
            status=AccountRunStatus.FAILED.value,
            completed_at=completed_at,
            error_message=str(exc),
        )
        await log(f"Run failed: {exc}")
        raise

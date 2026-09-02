"""Fail-closed distribution for finished daily-news episodes.

The Admin console stores the expected public channel identifiers. The active
browser identity is discovered again at publish time and must match before a
write is attempted; the resolved identity then becomes part of the immutable
per-attempt receipt.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import os
import re
import shutil
import threading
import time
import weakref
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import format_datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

from backend import config, database
from backend.models import TaskResponse, TaskStatus
from backend.pipeline.opencli import OpenCLIError, first_json, run_opencli_with_retries

logger = logging.getLogger(__name__)
_podcast_feed_lock = threading.Lock()
_platform_publish_locks: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop,
    dict[str, asyncio.Lock],
] = weakref.WeakKeyDictionary()


@dataclass(frozen=True)
class PublicationResult:
    action: str
    reason: str
    publication_url: str | None = None


class PublicationLedgerError(RuntimeError):
    """The durable publication ledger exists but cannot be trusted."""


@dataclass(frozen=True)
class PublicationPlan:
    targets: tuple[str, ...]
    pending_targets: tuple[str, ...]
    visibility: str
    manifest: dict[str, Any]


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _publication_marker(task_id: str) -> str:
    """Stable per-task marker; tasks created on the same day must not collide."""
    token = hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:16]
    return f"FTSN-{token}"


def _platform_publish_lock(target: str) -> asyncio.Lock:
    """Serialize one browser account per platform on the service event loop."""
    loop = asyncio.get_running_loop()
    locks = _platform_publish_locks.setdefault(loop, {})
    return locks.setdefault(target, asyncio.Lock())


_PLATFORM_ENABLE_FLAGS = {
    "youtube": "VIDEO_PUBLISH_YOUTUBE_ENABLED",
    "x": "VIDEO_PUBLISH_X_ENABLED",
    "apple_podcast": "VIDEO_PUBLISH_APPLE_PODCAST_ENABLED",
}


def _normalized_identity(value: Any) -> str:
    return " ".join(str(value or "").strip().split()).casefold()


def _platform_enabled(target: str) -> bool:
    flag = _PLATFORM_ENABLE_FLAGS.get(target)
    return bool(flag and getattr(config, flag, False))


def publication_receipt_is_active(platform: str, entry: Any) -> bool:
    if not isinstance(entry, dict):
        return False
    expected = "feed_ready" if platform == "apple_podcast" else "published"
    return entry.get("status") == expected


def _require_receipt_text(
    platform: str,
    entry: dict[str, Any],
    field: str,
) -> str:
    value = entry.get(field)
    if not isinstance(value, str) or not value.strip():
        raise PublicationLedgerError(
            f"Publication ledger {platform} receipt has invalid {field}"
        )
    return value


def _validate_publication_receipt(platform: str, entry: dict[str, Any]) -> None:
    status = entry.get("status")
    sha256 = _require_receipt_text(platform, entry, "sha256")
    if not re.fullmatch(r"[0-9a-f]{64}", sha256):
        raise PublicationLedgerError(
            f"Publication ledger {platform} receipt has invalid sha256"
        )
    timestamp_field = "prepared_at" if platform == "apple_podcast" else "published_at"
    timestamp = _require_receipt_text(platform, entry, timestamp_field)
    try:
        datetime.fromisoformat(timestamp)
    except ValueError as exc:
        raise PublicationLedgerError(
            f"Publication ledger {platform} receipt has invalid {timestamp_field}"
        ) from exc

    url = _require_receipt_text(platform, entry, "url")
    identity = entry.get("identity")
    if not isinstance(identity, dict):
        raise PublicationLedgerError(
            f"Publication ledger {platform} receipt has invalid identity"
        )
    if platform == "youtube":
        video_id = _require_receipt_text(platform, entry, "external_id")
        if not re.fullmatch(r"[A-Za-z0-9_-]{6,}", video_id) or url != (
            f"https://youtu.be/{video_id}"
        ):
            raise PublicationLedgerError(
                "Publication ledger youtube receipt has inconsistent URL/video ID"
            )
        if not isinstance(identity.get("channel_id"), str) or not identity["channel_id"]:
            raise PublicationLedgerError(
                "Publication ledger youtube receipt has invalid channel identity"
            )
        if entry.get("visibility") not in {"private", "unlisted", "public"}:
            raise PublicationLedgerError(
                "Publication ledger youtube receipt has invalid visibility"
            )
    elif platform == "x":
        username = str(identity.get("username") or "").lstrip("@")
        if not username or not re.fullmatch(
            rf"https://(?:x\.com|twitter\.com)/{re.escape(username)}/status/\d+",
            url,
            re.IGNORECASE,
        ):
            raise PublicationLedgerError(
                "Publication ledger x receipt has inconsistent account/URL"
            )
    else:
        _require_receipt_text(platform, identity, "feed_title")
        if not url.endswith("/podcast/feed.xml"):
            raise PublicationLedgerError(
                "Publication ledger apple_podcast receipt has invalid feed URL"
            )

    if platform in {"youtube", "x"}:
        if not isinstance(entry.get("test_mode"), bool):
            raise PublicationLedgerError(
                f"Publication ledger {platform} receipt has invalid test_mode"
            )
        _require_receipt_text(platform, entry, "marker")
    if status == "deleted":
        deleted_at = _require_receipt_text(platform, entry, "deleted_at")
        try:
            datetime.fromisoformat(deleted_at)
        except ValueError as exc:
            raise PublicationLedgerError(
                f"Publication ledger {platform} receipt has invalid deleted_at"
            ) from exc


def automatic_publication_configuration_errors(targets: list[str]) -> list[str]:
    """Return fail-closed configuration errors for an unattended publish.

    Platform enablement and account identities are deliberately read from live
    Admin settings, not copied into a task. This lets an operator stop a queued
    publication or rotate the expected account right up to dispatch time.
    """
    errors: list[str] = []
    unique_targets = list(dict.fromkeys(targets))
    if not unique_targets:
        return ["No publication platform is selected"]
    for target in unique_targets:
        if not _platform_enabled(target):
            errors.append(f"{target} publication is disabled in Admin → Publishing")
    if "youtube" in unique_targets:
        if not config.VIDEO_PUBLISH_YOUTUBE_CHANNEL_NAME:
            errors.append("YouTube channel name is not configured")
        if not config.VIDEO_PUBLISH_YOUTUBE_CHANNEL_ID:
            errors.append("YouTube channel ID is not configured")
    if "x" in unique_targets and not config.VIDEO_PUBLISH_X_HANDLE:
        errors.append("X account handle is not configured")
    return errors


def _assert_publication_dispatch_allowed(
    target: str,
    *,
    require_auto_publish_enabled: bool,
) -> None:
    if require_auto_publish_enabled and not config.VIDEO_AUTO_PUBLISH_ENABLED:
        raise ValueError("Automatic publication was stopped by the master switch")
    errors = automatic_publication_configuration_errors([target])
    if errors:
        raise ValueError("; ".join(errors))


def _assert_expected_x_identity(identity: dict[str, Any]) -> None:
    expected = config.VIDEO_PUBLISH_X_HANDLE.lstrip("@")
    if not expected:
        return
    actual = str(identity.get("username") or "").lstrip("@")
    if actual.casefold() != expected.casefold():
        raise OpenCLIError(
            f"Refusing X publication: configured @{expected}, active account is "
            f"@{actual or 'unknown'}"
        )


def _assert_expected_youtube_identity(identity: dict[str, Any]) -> None:
    expected_id = config.VIDEO_PUBLISH_YOUTUBE_CHANNEL_ID
    expected_name = config.VIDEO_PUBLISH_YOUTUBE_CHANNEL_NAME
    actual_id = str(identity.get("channel_id") or "")
    actual_name = str(identity.get("channel_name") or "")
    if expected_id and actual_id != expected_id:
        raise OpenCLIError(
            "Refusing YouTube publication: configured channel ID "
            f"{expected_id}, active channel ID is {actual_id or 'unknown'}"
        )
    if expected_name and _normalized_identity(actual_name) != _normalized_identity(expected_name):
        raise OpenCLIError(
            "Refusing YouTube publication: configured channel name "
            f"{expected_name!r}, active channel is {actual_name or 'unknown'!r}"
        )


def _task_dir(task: TaskResponse) -> Path:
    return Path(task.output_dir) if task.output_dir else config.OUTPUTS_DIR / task.id


def _publication_dir(task: TaskResponse) -> Path:
    path = _task_dir(task) / "publications"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _manifest_path(task: TaskResponse) -> Path:
    return _publication_dir(task) / "manifest.json"


def read_publication_manifest(task: TaskResponse) -> dict[str, Any]:
    path = _manifest_path(task)
    temporary = path.with_suffix(".tmp")
    if temporary.exists():
        raise PublicationLedgerError(
            "Publication ledger has an unfinished atomic write; verify external "
            f"destinations before continuing: {temporary}"
        )
    if path.exists() or path.is_symlink():
        if not path.is_file():
            raise PublicationLedgerError(
                f"Publication ledger path is not a regular file: {path}"
            )
    else:
        return {"task_id": task.id, "platforms": {}}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PublicationLedgerError(
            f"Publication ledger exists but cannot be read safely: {path}"
        ) from exc
    if not isinstance(value, dict):
        raise PublicationLedgerError("Publication ledger root must be an object")
    recorded_task_id = value.get("task_id")
    if recorded_task_id is None:
        raise PublicationLedgerError("Publication ledger is missing its task_id")
    if recorded_task_id != task.id:
        raise PublicationLedgerError(
            "Publication ledger belongs to a different task"
        )
    if "platforms" not in value:
        raise PublicationLedgerError("Publication ledger is missing its platforms map")
    platforms = value["platforms"]
    if not isinstance(platforms, dict) or any(
        not isinstance(platform, str) or not isinstance(entry, dict)
        for platform, entry in platforms.items()
    ):
        raise PublicationLedgerError(
            "Publication ledger platforms must map names to receipt objects"
        )
    if not platforms:
        raise PublicationLedgerError(
            "Existing publication ledger contains no durable receipts"
        )
    allowed_statuses = {
        "youtube": {"published", "deleted"},
        "x": {"published", "deleted"},
        "apple_podcast": {"feed_ready"},
    }
    for platform, entry in platforms.items():
        if platform not in allowed_statuses:
            raise PublicationLedgerError(
                f"Publication ledger contains unsupported platform {platform!r}"
            )
        status = entry.get("status")
        if status not in allowed_statuses[platform]:
            raise PublicationLedgerError(
                f"Publication ledger contains invalid {platform} status {status!r}"
            )
        _validate_publication_receipt(platform, entry)
    value["platforms"] = platforms
    return value


def publication_manifest_view(
    task: TaskResponse,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Annotate receipts against the task's current immutable media revision."""
    view = copy.deepcopy(manifest if manifest is not None else read_publication_manifest(task))
    video_sha = (
        _file_sha256(task.video_path)
        if task.video_path and Path(task.video_path).is_file()
        else None
    )
    audio_sha = (
        _file_sha256(task.audio_path)
        if task.audio_path and Path(task.audio_path).is_file()
        else None
    )
    view["current_video_sha256"] = video_sha
    view["current_audio_sha256"] = audio_sha
    for platform, entry in (view.get("platforms") or {}).items():
        if not isinstance(entry, dict):
            continue
        current_sha = audio_sha if platform == "apple_podcast" else video_sha
        entry["matches_current_media"] = bool(
            current_sha and entry.get("sha256") == current_sha
        )
    return view


def _write_manifest(task: TaskResponse, manifest: dict[str, Any]) -> None:
    path = _manifest_path(task)
    manifest["task_id"] = task.id
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    temporary = path.with_suffix(".tmp")
    if temporary.exists():
        raise PublicationLedgerError(
            f"Refusing to overwrite unfinished publication ledger write: {temporary}"
        )
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _record(task: TaskResponse, platform: str, payload: dict[str, Any]) -> dict[str, Any]:
    manifest = read_publication_manifest(task)
    manifest.setdefault("platforms", {})[platform] = payload
    _write_manifest(task, manifest)
    return manifest


def reconcile_deleted_test_receipts(
    task: TaskResponse,
    targets: list[str],
) -> dict[str, Any]:
    """Record an operator-verified delete after an indeterminate adapter result."""
    manifest = read_publication_manifest(task)
    reconciled_at = datetime.now(timezone.utc).isoformat()
    for target in dict.fromkeys(targets):
        entry = (manifest.get("platforms") or {}).get(target)
        if not isinstance(entry, dict) or entry.get("status") != "published":
            raise PublicationLedgerError(
                f"No active {target} receipt is available to reconcile as deleted"
            )
        if entry.get("test_mode") is not True:
            raise PublicationLedgerError(
                f"Refusing to reconcile non-test {target} publication as deleted"
            )
        entry["status"] = "deleted"
        entry["deleted_at"] = reconciled_at
        entry["deletion_reconciled_at"] = reconciled_at
    _write_manifest(task, manifest)
    return manifest


def _title(task: TaskResponse) -> str:
    return task.generated_title or task.source_title or f"ByteFront Espresso — {task.origin_id or task.id}"


def _description(task: TaskResponse, marker: str | None = None) -> str:
    dossier = _task_dir(task) / "research" / "dossier.json"
    sources: list[str] = []
    if dossier.is_file():
        try:
            data = json.loads(dossier.read_text(encoding="utf-8"))
            # Current research dossiers store the final shortlist under
            # ``selected``.  Keep ``stories`` as a compatibility fallback for
            # already-generated episodes from the earlier schema.
            selected = data.get("selected")
            if not isinstance(selected, list):
                selected = data.get("stories", [])
            sources = [
                str(item.get("url"))
                for item in selected
                if isinstance(item, dict) and item.get("url")
            ]
        except (OSError, json.JSONDecodeError):
            pass
    lines = [
        "A source-linked daily briefing on AI and frontier technology.",
        "Generated by FrontierTechSN; factual claims received an independent web-model review.",
    ]
    if marker:
        lines.append(f"Test run: {marker}")
    if sources:
        lines.extend(["", "Sources:", *sources[:10]])
    return "\n".join(lines)[:4900]


async def _browser(
    session: str,
    *args: str,
    timeout: int = 180,
    attempts: int | None = None,
    window: str | None = None,
) -> str:
    # YouTube Studio repeatedly stalls when Chrome leaves its upload/edit tab
    # in the background.  Keep those sessions foregrounded; X's dedicated
    # adapter remains safe and faster in the background.
    effective_window = window or (
        "foreground" if session.startswith("ftsn-yt-") else "background"
    )
    result = await run_opencli_with_retries(
        ["browser", session, *args, "--window", effective_window],
        timeout=timeout,
        attempts=attempts,
        label=f"OpenCLI browser {session}",
    )
    return result.stdout


_CHROME_UPLOAD_PERMISSION_HELP = (
    "Chrome blocked local file upload. Open chrome://extensions, click Details "
    "under the ChatGPT browser extension, enable 'Allow access to file URLs', "
    "then retry the publication."
)


async def _upload_local_media(
    session: str,
    *upload_args: str,
    timeout: int,
) -> str:
    """Upload once to detect a permanent Chrome permission failure quickly.

    Transient bridge failures still receive the project's full ten-attempt
    budget. Chrome's explicit ``Not allowed`` response is configuration-bound,
    so retrying it ten times only hides the actionable prerequisite.
    """
    try:
        return await _browser(
            session,
            "upload",
            *upload_args,
            timeout=timeout,
            attempts=1,
        )
    except OpenCLIError as exc:
        if "not allowed" in str(exc).casefold():
            raise OpenCLIError(_CHROME_UPLOAD_PERMISSION_HELP) from exc
    remaining_attempts = max(1, config.OPENCLI_MAX_ATTEMPTS - 1)
    return await _browser(
        session,
        "upload",
        *upload_args,
        timeout=timeout,
        attempts=remaining_attempts,
    )


async def _browser_json(
    session: str,
    *args: str,
    timeout: int = 180,
    attempts: int | None = None,
) -> Any:
    output = await _browser(session, *args, timeout=timeout, attempts=attempts)
    try:
        # ``browser eval`` legitimately emits JSON primitives such as ``true``
        # for wait predicates, not only objects and arrays.
        return json.loads(output)
    except json.JSONDecodeError:
        # Preserve compatibility with OpenCLI commands that prefix a JSON
        # object/array with human-readable status text.
        try:
            return first_json(output)
        except OpenCLIError:
            # String-valued ``browser eval`` output is emitted as plain text,
            # without JSON quotes (for example a YouTube video title).
            return output.strip()


async def _wait_for(
    session: str,
    javascript: str,
    *,
    timeout_seconds: int,
    interval_seconds: float = 3.0,
) -> Any:
    deadline = time.monotonic() + timeout_seconds
    last: Any = None
    while time.monotonic() < deadline:
        last = await _browser_json(session, "eval", javascript, timeout=60)
        if last:
            return last
        await asyncio.sleep(interval_seconds)
    raise OpenCLIError(f"Browser condition did not become true within {timeout_seconds}s: {last!r}")


async def _x_identity() -> dict[str, Any]:
    result = await run_opencli_with_retries(
        [
            "twitter", "whoami", "--window", "background", "--site-session", "ephemeral",
            "--keep-tab", "false", "-f", "json",
        ],
        timeout=90,
        label="X account identity",
    )
    value = first_json(result.stdout)
    rows = value if isinstance(value, list) else [value]
    identity = rows[0] if rows and isinstance(rows[0], dict) else {}
    handle = str(identity.get("username") or "").lstrip("@")
    if not handle:
        raise OpenCLIError("X is not logged in or its current handle could not be identified")
    return identity


async def publish_x(
    task: TaskResponse,
    *,
    test_mode: bool,
    require_auto_publish_enabled: bool = False,
) -> dict[str, Any]:
    _assert_publication_dispatch_allowed(
        "x",
        require_auto_publish_enabled=require_auto_publish_enabled,
    )
    if not task.video_path or not Path(task.video_path).is_file():
        raise FileNotFoundError("Final video is missing")
    identity = await _x_identity()
    _assert_expected_x_identity(identity)
    handle = str(identity["username"]).lstrip("@")
    marker = _publication_marker(task.id)
    text = (
        f"{_title(task)}\n\n"
        "Today’s source-linked briefing on AI and frontier technology. "
        f"{marker}"
    )[:270]
    session = f"ftsn-x-{task.id}-{time.time_ns()}"
    try:
        await _browser(session, "open", "https://x.com/compose/post", timeout=90)
        await _upload_local_media(
            session, "--testid", "fileInput", "--nth", "0",
            str(Path(task.video_path).resolve()), timeout=300,
        )
        await _browser(
            session, "fill", "--testid", "tweetTextarea_0", "--nth", "0", text,
            timeout=90,
        )
        await _wait_for(
            session,
            "(()=>{const b=[...document.querySelectorAll('[data-testid=tweetButton]')].find(x=>x.offsetParent&&!x.disabled&&x.getAttribute('aria-disabled')!=='true');return b?true:false})()",
            timeout_seconds=600,
            interval_seconds=5,
        )
        _assert_publication_dispatch_allowed(
            "x",
            require_auto_publish_enabled=require_auto_publish_enabled,
        )
        fresh_identity = await _x_identity()
        _assert_expected_x_identity(fresh_identity)
        if str(fresh_identity.get("username") or "").lstrip("@").casefold() != (
            handle.casefold()
        ):
            raise OpenCLIError(
                "Refusing X publication because the active account changed during upload"
            )
        _assert_publication_dispatch_allowed(
            "x",
            require_auto_publish_enabled=require_auto_publish_enabled,
        )
        await _browser(
            session,
            "click",
            "--testid",
            "tweetButton",
            "--nth",
            "0",
            timeout=180,
            attempts=1,
        )
    finally:
        try:
            await _browser(session, "close", timeout=30)
        except Exception:
            logger.warning("Could not close X browser session %s", session)

    search = await run_opencli_with_retries(
        [
            "twitter", "search", f'"{marker}"', "--from", handle, "--product", "live",
            "--limit", "5", "--window", "background", "--site-session", "ephemeral",
            "--keep-tab", "false", "-f", "json",
        ],
        timeout=120,
        label="X publication verification",
    )
    value = first_json(search.stdout)
    rows = value if isinstance(value, list) else [value]
    row = next(
        (item for item in rows if isinstance(item, dict) and marker in str(item.get("text") or "")),
        None,
    )
    if not row or not row.get("url"):
        raise OpenCLIError("X composer completed but the exact test marker was not found on the current account")
    url = str(row["url"])
    if not re.match(rf"https://(?:x\.com|twitter\.com)/{re.escape(handle)}/status/\d+", url, re.I):
        raise OpenCLIError(f"X returned a post owned by an unexpected account: {url}")
    return _record(task, "x", {
        "status": "published",
        "published_at": datetime.now(timezone.utc).isoformat(),
        "identity": {"username": handle, "display_name": identity.get("name")},
        "url": url,
        "external_id": row.get("id"),
        "marker": marker,
        "test_mode": test_mode,
        "sha256": _file_sha256(task.video_path),
    })["platforms"]["x"]


async def delete_x(task: TaskResponse) -> dict[str, Any]:
    entry = read_publication_manifest(task).get("platforms", {}).get("x") or {}
    url = str(entry.get("url") or "")
    if entry.get("status") != "published" or not re.fullmatch(
        r"https://(?:x\.com|twitter\.com)/[^/]+/status/\d+", url
    ):
        raise OpenCLIError("Refusing to delete X content without an exact recorded publication URL")
    identity = await _x_identity()
    current = str(identity.get("username") or "").lstrip("@")
    expected = str(entry.get("identity", {}).get("username") or "").lstrip("@")
    if current.casefold() != expected.casefold():
        raise OpenCLIError(f"Refusing X deletion: published as @{expected}, currently signed in as @{current}")
    await run_opencli_with_retries(
        ["twitter", "delete", url, "--window", "background", "--site-session", "ephemeral", "--keep-tab", "false", "-f", "json"],
        timeout=120,
        attempts=1,
        label="X exact-post deletion",
    )
    entry.update({"status": "deleted", "deleted_at": datetime.now(timezone.utc).isoformat()})
    return _record(task, "x", entry)["platforms"]["x"]


async def _youtube_identity(session: str) -> dict[str, str]:
    await _browser(session, "open", "https://studio.youtube.com/", timeout=120)
    value = await _wait_for(
        session,
        "(()=>{const m=location.href.match(/\\/channel\\/([^/]+)/);const n=document.querySelector('#channel-name')?.textContent?.trim()||document.querySelector('ytcp-navigation-drawer #entity-name')?.textContent?.trim()||'';return m?{channel_id:m[1],channel_name:n,url:location.href}:null})()",
        timeout_seconds=120,
    )
    if not isinstance(value, dict) or not value.get("channel_id"):
        raise OpenCLIError("YouTube Studio did not expose a logged-in channel")
    return {key: str(value.get(key) or "") for key in ("channel_id", "channel_name", "url")}


async def publish_youtube(
    task: TaskResponse,
    *,
    visibility: str,
    test_mode: bool,
    require_auto_publish_enabled: bool = False,
) -> dict[str, Any]:
    _assert_publication_dispatch_allowed(
        "youtube",
        require_auto_publish_enabled=require_auto_publish_enabled,
    )
    if not task.video_path or not Path(task.video_path).is_file():
        raise FileNotFoundError("Final video is missing")
    visibility = visibility if visibility in {"private", "unlisted", "public"} else "private"
    # A fresh lease per invocation prevents a previously timed-out Studio tab
    # from poisoning a later retry of the same task.
    session = f"ftsn-yt-{task.id[:8]}-{time.time_ns()}"
    marker = _publication_marker(task.id)
    title = f"{_title(task)}{' [TEST]' if test_mode else ''}"[:95]
    link: Any = ""
    identity: dict[str, str] = {}
    try:
        identity = await _youtube_identity(session)
        _assert_expected_youtube_identity(identity)
        upload_url = f"https://studio.youtube.com/channel/{quote(identity['channel_id'])}/videos/upload?d=ud"
        await _browser(session, "open", upload_url, timeout=120)
        await _upload_local_media(
            session, "input[name=Filedata]", str(Path(task.video_path).resolve()),
            timeout=300,
        )
        await _wait_for(
            session,
            "(()=>document.querySelector('#title-textarea #textbox')?true:false)()",
            timeout_seconds=300,
            interval_seconds=5,
        )
        await _browser(session, "fill", "#title-textarea #textbox", title, timeout=90)
        await _browser(session, "fill", "#description-textarea #textbox", _description(task, marker), timeout=90)
        try:
            await _browser(session, "click", "--name", "No, it's not made for kids", "--role", "radio", timeout=60)
        except Exception:
            await _browser_json(session, "eval", "(()=>{const r=[...document.querySelectorAll('tp-yt-paper-radio-button')].find(x=>/not made for kids/i.test(x.textContent||''));if(r){r.click();return true}return false})()")
        for _ in range(3):
            await _wait_for(
                session,
                "(()=>{const b=document.querySelector('#next-button');return b&&!b.disabled?true:false})()",
                timeout_seconds=600,
                interval_seconds=5,
            )
            await _browser(session, "click", "#next-button", timeout=90)
        labels = {"private": "Private", "unlisted": "Unlisted", "public": "Public"}
        await _browser(session, "click", "--name", labels[visibility], "--role", "radio", timeout=90)
        await _wait_for(
            session,
            "(()=>{const b=document.querySelector('#done-button');return b&&!b.disabled?true:false})()",
            timeout_seconds=1800,
            interval_seconds=8,
        )
        link = await _browser_json(
            session,
            "eval",
            "(()=>document.querySelector('a[href*=\"youtu.be/\"]')?.href||document.querySelector('a[href*=\"watch?v=\"]')?.href||'')()",
            timeout=60,
        )
        _assert_publication_dispatch_allowed(
            "youtube",
            require_auto_publish_enabled=require_auto_publish_enabled,
        )
        guard_session = f"ftsn-yt-guard-{task.id[:8]}-{time.time_ns()}"
        try:
            fresh_identity = await _youtube_identity(guard_session)
            _assert_expected_youtube_identity(fresh_identity)
            if fresh_identity["channel_id"] != identity["channel_id"]:
                raise OpenCLIError(
                    "Refusing YouTube publication because the active channel "
                    "changed during upload"
                )
            _assert_publication_dispatch_allowed(
                "youtube",
                require_auto_publish_enabled=require_auto_publish_enabled,
            )
            await _browser(
                session,
                "click",
                "#done-button",
                timeout=180,
                attempts=1,
            )
        finally:
            try:
                # Close only after the irreversible click so identity verify,
                # live switch check, and commit have no intervening await.
                await _browser(
                    guard_session,
                    "close",
                    timeout=30,
                    attempts=1,
                )
            except Exception:
                logger.warning(
                    "Could not close YouTube identity guard session %s",
                    guard_session,
                )
        if not link:
            link = await _wait_for(
                session,
                "(()=>{const a=[...document.querySelectorAll('a')].find(x=>/youtu\\.be\\/|watch\\?v=/.test(x.href||''));return a?a.href:''})()",
                timeout_seconds=120,
            )
    finally:
        try:
            await _browser(session, "close", timeout=30)
        except Exception:
            logger.warning("Could not close YouTube browser session %s", session)
    link = str(link)
    match = re.search(r"(?:youtu\.be/|[?&]v=)([A-Za-z0-9_-]{6,})", link)
    if not match:
        raise OpenCLIError(f"YouTube upload completed without a canonical video ID: {link!r}")
    video_id = match.group(1)
    return _record(task, "youtube", {
        "status": "published",
        "published_at": datetime.now(timezone.utc).isoformat(),
        "identity": identity,
        "url": f"https://youtu.be/{video_id}",
        "external_id": video_id,
        "visibility": visibility,
        "marker": marker,
        "test_mode": test_mode,
        "sha256": _file_sha256(task.video_path),
    })["platforms"]["youtube"]


async def delete_youtube(task: TaskResponse) -> dict[str, Any]:
    entry = read_publication_manifest(task).get("platforms", {}).get("youtube") or {}
    video_id = str(entry.get("external_id") or "")
    channel_id = str(entry.get("identity", {}).get("channel_id") or "")
    if entry.get("status") != "published" or not re.fullmatch(r"[A-Za-z0-9_-]{6,}", video_id):
        raise OpenCLIError("Refusing to delete YouTube content without an exact recorded video ID")
    session = f"ftsn-yt-delete-{task.id[:8]}-{time.time_ns()}"
    try:
        identity = await _youtube_identity(session)
        if identity["channel_id"] != channel_id:
            raise OpenCLIError("Refusing YouTube deletion because the active channel changed")
        await _browser(session, "open", f"https://studio.youtube.com/video/{video_id}/edit", timeout=120)
        current_title = await _wait_for(
            session,
            "(()=>document.querySelector('#title-textarea #textbox')?.textContent?.trim()||'')()",
            timeout_seconds=180,
        )
        await _wait_for(
            session,
            "(()=>{const b=document.querySelector('#overflow-menu-button')||document.querySelector('#more-options-button')||document.querySelector('[aria-label*=\"More options\"]');return b?true:false})()",
            timeout_seconds=180,
        )
        try:
            await _browser(session, "click", "#overflow-menu-button", timeout=60)
        except Exception:
            try:
                await _browser(session, "click", "#more-options-button", timeout=60)
            except Exception:
                await _browser(session, "click", "--name", "More options", "--role", "button", timeout=60)
        try:
            await _browser(
                session,
                "click",
                "--name",
                "Delete",
                "--role",
                "menuitem",
                timeout=60,
            )
        except Exception:
            await _browser(session, "click", "--text", "Delete forever", timeout=60)
        await _wait_for(
            session,
            "(()=>document.querySelector('ytcp-video-delete-dialog #confirm-checkbox,ytcp-video-delete-dialog input[type=checkbox]')?true:false)()",
            timeout_seconds=90,
        )
        confirm_input = await _browser_json(
            session,
            "eval",
            "(()=>document.querySelector('ytcp-video-delete-dialog #confirm-input textarea')?true:false)()",
        )
        if confirm_input:
            await _browser(
                session,
                "fill",
                "ytcp-video-delete-dialog #confirm-input textarea",
                str(current_title),
                timeout=60,
            )
        await _browser_json(
            session,
            "eval",
            "(()=>{const c=document.querySelector('ytcp-video-delete-dialog #confirm-checkbox #checkbox')||document.querySelector('ytcp-video-delete-dialog [role=checkbox]')||document.querySelector('ytcp-video-delete-dialog input[type=checkbox]');if(c){if(c.getAttribute('aria-checked')!=='true'&&!c.checked)c.click();return true}return false})()",
        )
        await _wait_for(
            session,
            "(()=>{const b=document.querySelector('ytcp-video-delete-dialog #confirm-button')||[...document.querySelectorAll('ytcp-video-delete-dialog ytcp-button,ytcp-video-delete-dialog button')].find(x=>/delete forever/i.test(x.textContent||''));return b&&b.getAttribute('aria-disabled')!=='true'&&!b.disabled?true:false})()",
            timeout_seconds=90,
        )
        await _browser_json(
            session,
            "eval",
            "(()=>{const b=document.querySelector('ytcp-video-delete-dialog #confirm-button')||[...document.querySelectorAll('ytcp-video-delete-dialog ytcp-button,ytcp-video-delete-dialog button')].find(x=>/delete forever/i.test(x.textContent||''));if(b&&b.getAttribute('aria-disabled')!=='true'&&!b.disabled){b.click();return true}return false})()",
            attempts=1,
        )
        await _wait_for(
            session,
            f"(()=>!location.href.includes('/video/{video_id}/edit'))()",
            timeout_seconds=180,
        )
    finally:
        try:
            await _browser(session, "close", timeout=30)
        except Exception:
            logger.warning("Could not close YouTube deletion session %s", session)
    entry.update({"status": "deleted", "deleted_at": datetime.now(timezone.utc).isoformat()})
    return _record(task, "youtube", entry)["platforms"]["youtube"]


def _read_podcast_episodes(path: Path) -> list[dict[str, Any]]:
    temporary = path.with_suffix(".tmp")
    if temporary.exists():
        raise PublicationLedgerError(
            f"Podcast episode ledger has an unfinished atomic write: {temporary}"
        )
    if not path.exists():
        return []
    try:
        episodes = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PublicationLedgerError(
            f"Podcast episode ledger exists but cannot be read safely: {path}"
        ) from exc
    if not isinstance(episodes, list):
        raise PublicationLedgerError("Podcast episode ledger root must be a list")
    required = {
        "guid": str,
        "title": str,
        "published_at": str,
        "duration": int,
        "audio_url": str,
        "audio_bytes": int,
        "summary": str,
    }
    seen_guids: set[str] = set()
    for index, item in enumerate(episodes):
        if not isinstance(item, dict):
            raise PublicationLedgerError(
                f"Podcast episode ledger item {index} must be an object"
            )
        for field, expected_type in required.items():
            value = item.get(field)
            if not isinstance(value, expected_type) or (
                expected_type is str and not value.strip()
            ):
                raise PublicationLedgerError(
                    f"Podcast episode ledger item {index} has invalid {field}"
                )
        guid = item["guid"]
        if guid in seen_guids:
            raise PublicationLedgerError(
                f"Podcast episode ledger contains duplicate guid {guid!r}"
            )
        seen_guids.add(guid)
        try:
            datetime.fromisoformat(item["published_at"])
        except ValueError as exc:
            raise PublicationLedgerError(
                f"Podcast episode ledger item {index} has invalid published_at"
            ) from exc
    return episodes


def prepare_apple_podcast(task: TaskResponse) -> dict[str, Any]:
    """Create a valid episode feed; Podcasts Connect submission is deferred."""
    from backend import settings_store

    if settings_store.restart_required_change_pending("OUTPUTS_DIR"):
        raise PublicationLedgerError(
            "Apple podcast preparation is paused until the pending OUTPUTS_DIR "
            "change is activated by a service restart"
        )
    if not task.audio_path or not Path(task.audio_path).is_file():
        raise FileNotFoundError("Narration audio is missing")
    with _podcast_feed_lock:
        podcast_dir = config.OUTPUTS_DIR / "podcast"
        podcast_dir.mkdir(parents=True, exist_ok=True)
        episodes_path = podcast_dir / "episodes.json"
        feed_path = podcast_dir / "feed.xml"
        if feed_path.with_suffix(".tmp").exists():
            raise PublicationLedgerError(
                "Podcast feed has an unfinished atomic write; verify the feed "
                "before preparing another episode"
            )
        media_dir = podcast_dir / "media"
        try:
            historical_feed_or_media = (
                feed_path.exists()
                or (media_dir.is_dir() and any(media_dir.iterdir()))
            )
            orphaned_feed_state = (
                not episodes_path.exists()
                and historical_feed_or_media
            )
        except OSError as exc:
            raise PublicationLedgerError(
                "Podcast feed state cannot be inspected safely"
            ) from exc
        if orphaned_feed_state:
            raise PublicationLedgerError(
                "Podcast feed or media exists without its episode ledger; "
                "refusing to replace historical feed state"
            )
        # Validate the durable global ledger before copying or replacing any
        # media. A corrupt ledger is evidence, never an empty starting point.
        episodes = _read_podcast_episodes(episodes_path)
        if episodes_path.exists() and not episodes and historical_feed_or_media:
            raise PublicationLedgerError(
                "Podcast episode ledger is empty while historical feed or media "
                "still exists; refusing to erase prior episodes"
            )
        audio = Path(task.audio_path)
        # A task keeps its original artifact root even if Admin later changes
        # OUTPUTS_DIR. Copy immutable episode audio into the current public
        # feed tree rather than assuming the old root is still mounted there.
        media_dir.mkdir(parents=True, exist_ok=True)
        public_audio = media_dir / f"{task.id}.wav"
        staged_audio = media_dir / f".{task.id}.wav.tmp"
        try:
            shutil.copyfile(audio, staged_audio)
            os.replace(staged_audio, public_audio)
        finally:
            staged_audio.unlink(missing_ok=True)
        relative_audio = public_audio.resolve().relative_to(
            config.OUTPUTS_DIR.resolve()
        ).as_posix()
        public_base = config.VIDEO_PUBLISH_APPLE_FEED_PUBLIC_BASE_URL
        audio_url = (
            f"{public_base}/outputs/{relative_audio}"
            if public_base
            else f"/outputs/{relative_audio}"
        )
        episode = {
            "guid": task.id,
            "title": _title(task),
            "published_at": datetime.now(timezone.utc).isoformat(),
            "duration": int(task.duration_seconds or 0),
            "audio_url": audio_url,
            "audio_bytes": public_audio.stat().st_size,
            "summary": _description(task),
        }
        episodes = [
            episode,
            *[item for item in episodes if item.get("guid") != task.id],
        ][:100]
        staged_episodes = episodes_path.with_suffix(".tmp")
        staged_episodes.write_text(
            json.dumps(episodes, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(staged_episodes, episodes_path)

        rss = ET.Element(
            "rss",
            {
                "version": "2.0",
                "xmlns:itunes": "http://www.itunes.com/dtds/podcast-1.0.dtd",
            },
        )
        channel = ET.SubElement(rss, "channel")
        ET.SubElement(channel, "title").text = config.VIDEO_PUBLISH_APPLE_FEED_TITLE
        ET.SubElement(channel, "description").text = (
            "A daily, source-linked frontier technology briefing."
        )
        ET.SubElement(channel, "language").text = "en"
        ET.SubElement(channel, "link").text = "http://localhost:8101/"
        ET.SubElement(channel, "itunes:author").text = (
            config.VIDEO_PUBLISH_APPLE_FEED_AUTHOR
        )
        for item in episodes:
            node = ET.SubElement(channel, "item")
            ET.SubElement(node, "title").text = item["title"]
            ET.SubElement(node, "guid", {"isPermaLink": "false"}).text = item["guid"]
            published = datetime.fromisoformat(item["published_at"])
            ET.SubElement(node, "pubDate").text = format_datetime(published)
            ET.SubElement(node, "description").text = item["summary"]
            ET.SubElement(node, "itunes:duration").text = str(item["duration"])
            ET.SubElement(
                node,
                "enclosure",
                {
                    "url": item["audio_url"],
                    "length": str(item["audio_bytes"]),
                    "type": "audio/wav",
                },
            )
        staged_feed = feed_path.with_suffix(".tmp")
        ET.ElementTree(rss).write(
            staged_feed,
            encoding="utf-8",
            xml_declaration=True,
        )
        os.replace(staged_feed, feed_path)
        payload = {
            "status": "feed_ready",
            "prepared_at": datetime.now(timezone.utc).isoformat(),
            "identity": {
                "feed_title": config.VIDEO_PUBLISH_APPLE_FEED_TITLE,
                "author": config.VIDEO_PUBLISH_APPLE_FEED_AUTHOR,
            },
            "url": "/outputs/podcast/feed.xml",
            "external_submission": (
                "deferred_until_podcasts_connect_account_and_public_https_url"
            ),
            "sha256": _file_sha256(public_audio),
        }
        return _record(task, "apple_podcast", payload)["platforms"]["apple_podcast"]


def plan_publication(
    task: TaskResponse,
    *,
    targets: list[str],
    test_mode: bool,
    visibility: str,
) -> PublicationPlan:
    """Validate a publication request without changing task or external state."""
    unique_targets = list(dict.fromkeys(targets))
    configuration_errors = automatic_publication_configuration_errors(unique_targets)
    if configuration_errors:
        raise ValueError("; ".join(configuration_errors))

    visibility = visibility if visibility in {"private", "unlisted", "public"} else "private"
    manifest = read_publication_manifest(task)
    pending: list[str] = []
    for target in unique_targets:
        if target not in _PLATFORM_ENABLE_FLAGS:
            raise ValueError(f"Unsupported publication target: {target}")
        asset_path = task.audio_path if target == "apple_podcast" else task.video_path
        if not asset_path or not Path(asset_path).is_file():
            asset_label = "Narration audio" if target == "apple_podcast" else "Rendered video"
            raise FileNotFoundError(f"{asset_label} is missing for {target} publication")
        current_sha = _file_sha256(asset_path)
        entry = manifest.get("platforms", {}).get(target) or {}
        if not publication_receipt_is_active(target, entry):
            pending.append(target)
            continue
        same_request = bool(current_sha) and entry.get("sha256") == current_sha
        if target in {"youtube", "x"}:
            same_request = same_request and bool(entry.get("test_mode")) == bool(test_mode)
        if target == "youtube":
            identity = entry.get("identity") or {}
            same_request = (
                same_request
                and entry.get("visibility") == visibility
                and _normalized_identity(identity.get("channel_id"))
                == _normalized_identity(config.VIDEO_PUBLISH_YOUTUBE_CHANNEL_ID)
                and _normalized_identity(identity.get("channel_name"))
                == _normalized_identity(config.VIDEO_PUBLISH_YOUTUBE_CHANNEL_NAME)
            )
        elif target == "x":
            identity = entry.get("identity") or {}
            same_request = same_request and _normalized_identity(
                str(identity.get("username") or "").lstrip("@")
            ) == _normalized_identity(str(config.VIDEO_PUBLISH_X_HANDLE).lstrip("@"))
        elif target == "apple_podcast":
            identity = entry.get("identity") or {}
            same_request = (
                same_request
                and _normalized_identity(identity.get("feed_title"))
                == _normalized_identity(config.VIDEO_PUBLISH_APPLE_FEED_TITLE)
                and _normalized_identity(identity.get("author"))
                == _normalized_identity(config.VIDEO_PUBLISH_APPLE_FEED_AUTHOR)
            )
        if same_request:
            continue
        raise ValueError(
            f"{target} already has an active publication for this task; "
            "delete the recorded test publication or create a new task before republishing"
        )

    return PublicationPlan(
        targets=tuple(unique_targets),
        pending_targets=tuple(pending),
        visibility=visibility,
        manifest=manifest,
    )


async def execute_publication_plan(
    task: TaskResponse,
    *,
    plan: PublicationPlan,
    test_mode: bool,
    require_auto_publish_enabled: bool = False,
) -> dict[str, Any]:
    """Execute a plan that was validated before the task entered PUBLISHING."""
    if not plan.pending_targets:
        return plan.manifest

    for target in plan.pending_targets:
        async with _platform_publish_lock(target):
            _assert_publication_dispatch_allowed(
                target,
                require_auto_publish_enabled=require_auto_publish_enabled,
            )
            if target == "youtube":
                await publish_youtube(
                    task,
                    visibility=plan.visibility,
                    test_mode=test_mode,
                    require_auto_publish_enabled=require_auto_publish_enabled,
                )
            elif target == "x":
                await publish_x(
                    task,
                    test_mode=test_mode,
                    require_auto_publish_enabled=require_auto_publish_enabled,
                )
            elif target == "apple_podcast":
                prepare_apple_podcast(task)
            else:
                raise ValueError(f"Unsupported publication target: {target}")
    return read_publication_manifest(task)


async def publish_task(
    task: TaskResponse,
    *,
    targets: list[str],
    test_mode: bool,
    visibility: str,
    allow_publishing_state: bool = False,
    require_auto_publish_enabled: bool = False,
) -> dict[str, Any]:
    if task.status != TaskStatus.COMPLETE and not (
        allow_publishing_state and task.status == TaskStatus.PUBLISHING
    ):
        raise ValueError("Only complete tasks can be published")
    plan = plan_publication(
        task,
        targets=targets,
        test_mode=test_mode,
        visibility=visibility,
    )
    return await execute_publication_plan(
        task,
        plan=plan,
        test_mode=test_mode,
        require_auto_publish_enabled=require_auto_publish_enabled,
    )


async def delete_test_publications(task: TaskResponse, *, targets: list[str]) -> dict[str, Any]:
    for target in dict.fromkeys(targets):
        async with _platform_publish_lock(target):
            if target == "youtube":
                await delete_youtube(task)
            elif target == "x":
                await delete_x(task)
            elif target == "apple_podcast":
                continue
    return read_publication_manifest(task)


async def run_auto_publish_pipeline(task: TaskResponse) -> PublicationResult:
    """Dispatch only publishing-workflow tasks with every safety gate open."""
    if task.origin_type not in {"daily_news", "content_plan"}:
        return PublicationResult("not_applicable", "Task did not originate from a publishing workflow")
    if not config.VIDEO_AUTO_PUBLISH_ENABLED:
        return PublicationResult("awaiting_review", "Global automatic publication is disabled")

    if task.origin_type == "daily_news":
        if not task.config.auto_publish:
            return PublicationResult("awaiting_review", "Daily automatic publication is disabled")
        configuration_errors = automatic_publication_configuration_errors(
            task.config.publish_targets
        )
        if configuration_errors:
            return PublicationResult(
                "awaiting_review",
                "Automatic publication is not ready: " + "; ".join(configuration_errors),
            )
        try:
            manifest = await publish_task(
                task,
                targets=task.config.publish_targets,
                test_mode=task.config.publish_test_mode,
                visibility=task.config.publish_visibility,
                allow_publishing_state=True,
                require_auto_publish_enabled=True,
            )
        except Exception as exc:
            return PublicationResult("failed", f"Daily automatic publication failed: {exc}")
        urls = [str(item.get("url")) for item in manifest.get("platforms", {}).values() if item.get("url")]
        return PublicationResult("published", "Daily publication completed", urls[0] if urls else None)

    if not task.origin_id:
        return PublicationResult("blocked", "Publishing workflow task has no origin identifier")
    item = await database.get_content_plan_item(task.origin_id)
    if item is None:
        return PublicationResult("blocked", "Originating plan no longer exists")
    if not item.auto_publish_requested:
        return PublicationResult("awaiting_review", "This content plan requires manual publication")
    return PublicationResult(
        "awaiting_review",
        f"No automatic publisher is configured for {item.platform}",
    )

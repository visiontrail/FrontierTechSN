"""Durable ownership and cleanup for one task's generated-media browser tabs."""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from backend.pipeline.opencli import OpenCLIError, close_opencli_site_sessions

logger = logging.getLogger(__name__)


def _namespace(owner: Path, site: str) -> str:
    digest = hashlib.sha256(str(owner.resolve()).encode()).hexdigest()[:32]
    return f"ftsn-media-{digest}-{site}"


def _write(path: Path, state: dict) -> None:
    temporary = path.with_suffix(".tmp")
    for candidate in (path, temporary):
        if candidate.is_symlink():
            raise OpenCLIError(f"Browser ownership record cannot be a symlink: {candidate}")
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temporary.replace(path)


async def _close(path: Path, namespace: str, site: str) -> None:
    for attempt in range(2):
        try:
            await close_opencli_site_sessions(namespace, sites=(site,))
            _write(path, {"namespace": namespace, "status": "closed", "verified": True})
            logger.info("Released and verified media browser session %s", namespace)
            return
        except Exception as exc:
            _write(path, {"namespace": namespace, "status": "cleanup_failed", "error": str(exc)})
            if attempt == 1:
                raise
            await asyncio.sleep(1)


@asynccontextmanager
async def owned_media_session(owner: Path | None, site: str, *, recover_only: bool = False):
    """Hold a cross-process lock through retries and verified tab cleanup.

    A killed worker leaves a journal, not an ambiguous global provider tab.
    Recovery derives the namespace from the owner path; file contents never
    authorize closing an arbitrary session. The lock prevents one invocation
    from reclaiming another live invocation of the same media operation.
    """
    if owner is None:
        yield None
        return
    if site not in ("chatgpt", "gemini"):
        raise ValueError(f"Unsupported media site: {site}")
    path = owner / f"browser-session-{site}.json"
    if recover_only and not path.exists():
        yield None
        return
    if owner.is_symlink() or path.is_symlink():
        raise OpenCLIError("Media session owner or journal cannot be a symlink")
    lock_path = path.with_suffix(".lock")
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    namespace = _namespace(owner, site)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise OpenCLIError(f"Media browser session is still owned by an active operation: {owner}") from exc
        prior = json.loads(path.read_text()) if path.exists() else {}
        if prior and (prior.get("status") != "closed" or prior.get("verified") is not True):
            await _close(path, namespace, site)
        if recover_only:
            yield None
            return
        _write(path, {"namespace": namespace, "status": "active", "pid": os.getpid()})
        try:
            yield namespace
        finally:
            # A second cancellation must not orphan the cleanup subprocess.
            cleanup = asyncio.create_task(_close(path, namespace, site))
            cancelled = False
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    cancelled = True
                except Exception:
                    break
            try:
                cleanup.result()
            except Exception:
                # Preserve the provider result; the journal retries cleanup
                # before this task can reuse a cached success or failure.
                logger.exception("Media browser cleanup remains pending: %s", namespace)
            if cancelled:
                raise asyncio.CancelledError
    finally:
        os.close(descriptor)


async def recover_media_sessions(root: Path) -> None:
    """Reclaim interrupted task sessions even when acquisition is skipped."""
    for owner in root.iterdir():
        if not owner.is_dir() or owner.is_symlink():
            continue
        for site in ("chatgpt", "gemini"):
            async with owned_media_session(owner, site, recover_only=True):
                pass


async def recover_orphaned_media_sessions(outputs_dir: Path) -> None:
    """Run before the worker starts, including tasks that will not be resumed."""
    if not outputs_dir.is_dir():
        return
    for task_dir in outputs_dir.iterdir():
        root = task_dir / "collage_broll"
        if task_dir.is_symlink() or root.is_symlink() or not root.is_dir():
            continue
        try:
            await recover_media_sessions(root)
        except Exception:
            logger.exception("Could not reclaim interrupted media sessions in %s", root)

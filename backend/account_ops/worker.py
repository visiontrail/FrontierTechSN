from __future__ import annotations

import asyncio
import logging
import traceback
from datetime import datetime, timezone

from backend import config, database
from backend.account_ops.orchestrator import execute_run

logger = logging.getLogger("account_ops.worker")
_worker_task: asyncio.Task | None = None
_last_tick_at: str | None = None


def get_worker_status() -> dict[str, object]:
    alive = _worker_task is not None and not _worker_task.done()
    return {
        "worker_alive": alive,
        "last_tick_at": _last_tick_at,
        "poll_interval": max(config.ACCOUNT_OPS_POLL_SECONDS, 2),
    }


async def _worker_loop() -> None:
    global _last_tick_at
    logger.info("Account-operations scheduler started")
    while True:
        try:
            _last_tick_at = datetime.now(timezone.utc).isoformat()
            enqueued = await database.enqueue_due_account_runs()
            if enqueued:
                logger.info("Enqueued %d scheduled account operation(s)", enqueued)
            run = await database.claim_next_account_run()
            if run is None:
                await asyncio.sleep(max(config.ACCOUNT_OPS_POLL_SECONDS, 2))
                continue
            logger.info("Executing account operation %s", run.id)
            try:
                await execute_run(run)
            except Exception as exc:
                logger.error(
                    "Account operation %s failed: %s\n%s",
                    run.id,
                    exc,
                    traceback.format_exc(),
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(
                "Account-operations worker error: %s\n%s",
                exc,
                traceback.format_exc(),
            )
            await asyncio.sleep(5)


def start_account_worker() -> None:
    global _worker_task
    if _worker_task is None or _worker_task.done():
        _worker_task = asyncio.create_task(_worker_loop())
        logger.info("Account-operations worker task created")


async def stop_account_worker() -> None:
    global _worker_task
    if _worker_task is None:
        return
    _worker_task.cancel()
    try:
        await _worker_task
    except asyncio.CancelledError:
        pass
    finally:
        _worker_task = None
    logger.info("Account-operations worker stopped")

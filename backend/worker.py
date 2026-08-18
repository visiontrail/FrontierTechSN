import asyncio
import json
import logging
import traceback
from pathlib import Path
from backend import config
from backend.database import get_next_queued_task, get_task, update_task
from backend.models import TaskResponse, TaskStatus
from backend.pipeline.orchestrator import (
    append_pipeline_log,
    format_pipeline_log,
    run_pipeline,
    run_daily_review_resume,
    run_regenerate,
    run_compose,
    run_footage_acquisition,
)
from backend.publishing import run_auto_publish_pipeline

# A queued task carrying this marker file in its output dir should re-run only
# the TTS stage (from an edited script) rather than the full pipeline.
REGEN_MARKER = ".regenerate"
# This marker resumes a reviewed task from the compose stage only.
RENDER_MARKER = ".render"
# Retry only the footage scout. The marker stores the stable status to restore
# after the worker briefly moves the task through QUEUED/SOURCING.
FOOTAGE_MARKER = ".footage"
REVIEW_RESUME_MARKER = ".resume_review"

logger = logging.getLogger("worker")

_worker_task: asyncio.Task | None = None
LogEvent = dict[str, str]
_active_log_queues: dict[str, set[asyncio.Queue[LogEvent]]] = {}
_active_task_ids: set[str] = set()


def _task_dir(task_id: str) -> Path:
    return config.OUTPUTS_DIR / task_id


def pipeline_log_file(task_id: str, output_dir: str | None = None) -> Path:
    task_dir = Path(output_dir) if output_dir else _task_dir(task_id)
    return task_dir / "logs" / "pipeline.log"


def subscribe_task_logs(task_id: str) -> asyncio.Queue[LogEvent]:
    queue: asyncio.Queue[LogEvent] = asyncio.Queue()
    _active_log_queues.setdefault(task_id, set()).add(queue)
    return queue


def unsubscribe_task_logs(task_id: str, queue: asyncio.Queue[LogEvent]):
    queues = _active_log_queues.get(task_id)
    if not queues:
        return
    queues.discard(queue)
    if not queues and task_id not in _active_task_ids:
        _active_log_queues.pop(task_id, None)


def is_task_logging_active(task_id: str) -> bool:
    return task_id in _active_task_ids


def publish_task_log(task_id: str, message: str):
    for queue in list(_active_log_queues.get(task_id, ())):
        queue.put_nowait({"event": "log", "data": message})


def finish_task_logs(task_id: str, status: str):
    for queue in list(_active_log_queues.get(task_id, ())):
        queue.put_nowait({"event": "status", "data": status})
    _active_task_ids.discard(task_id)
    _active_log_queues.pop(task_id, None)


def _persist_and_publish(task: TaskResponse, message: str):
    formatted = format_pipeline_log(task.id, message)
    append_pipeline_log(_task_dir(task.id), formatted)
    publish_task_log(task.id, formatted)


async def _worker_loop():
    logger.info("Background worker started")
    while True:
        try:
            task = await get_next_queued_task()
            if task:
                _active_task_ids.add(task.id)
                _active_log_queues.setdefault(task.id, set())
                regen_marker = config.OUTPUTS_DIR / task.id / REGEN_MARKER
                render_marker = config.OUTPUTS_DIR / task.id / RENDER_MARKER
                footage_marker = config.OUTPUTS_DIR / task.id / FOOTAGE_MARKER
                review_resume_marker = config.OUTPUTS_DIR / task.id / REVIEW_RESUME_MARKER
                regenerate = regen_marker.exists()
                render = render_marker.exists()
                footage = footage_marker.exists()
                review_resume = review_resume_marker.exists()
                footage_options = {}
                if regenerate:
                    regen_marker.unlink()
                if render:
                    render_marker.unlink()
                if footage:
                    try:
                        footage_options = json.loads(footage_marker.read_text(encoding="utf-8") or "{}")
                    finally:
                        footage_marker.unlink()
                if review_resume:
                    review_resume_marker.unlink()
                mode = (
                    " [resume-review]" if review_resume
                    else " [regenerate]" if regenerate
                    else " [render]" if render
                    else " [footage]" if footage
                    else ""
                )
                held = f" [scheduled for {task.scheduled_at}]" if task.scheduled_at else ""
                logger.info(f"Processing task {task.id} ({task.source_type}){mode}{held}")
                _persist_and_publish(task, f"Processing task ({task.source_type}){mode}{held}")
                try:
                    if review_resume:
                        await run_daily_review_resume(
                            task,
                            log=lambda message: publish_task_log(task.id, message),
                        )
                    elif regenerate:
                        await run_regenerate(task, log=lambda message: publish_task_log(task.id, message))
                    elif render:
                        await run_compose(task, log=lambda message: publish_task_log(task.id, message))
                    elif footage:
                        await run_footage_acquisition(
                            task,
                            resume_status=footage_options.get("resume_status", TaskStatus.AWAITING_REVIEW.value),
                            supplied_queries=footage_options.get("queries") or None,
                            log=lambda message: publish_task_log(task.id, message),
                        )
                    else:
                        await run_pipeline(task, log=lambda message: publish_task_log(task.id, message))
                    refreshed = await get_task(task.id)
                    if refreshed and refreshed.status == TaskStatus.COMPLETE:
                        publication = await run_auto_publish_pipeline(refreshed)
                        _persist_and_publish(
                            refreshed,
                            f"Publication pipeline: {publication.action} — {publication.reason}",
                        )
                    finish_task_logs(task.id, refreshed.status.value if refreshed else TaskStatus.COMPLETE.value)
                except Exception as e:
                    logger.error(f"Task {task.id} failed: {e}\n{traceback.format_exc()}")
                    _persist_and_publish(task, f"Task failed: {e}")
                    await update_task(task.id, status=TaskStatus.FAILED.value, error_message=str(e))
                    finish_task_logs(task.id, TaskStatus.FAILED.value)
            else:
                # Keep the clock-to-execution gap tight for scheduled plans.
                # Due scheduled tasks are also prioritized by the database.
                await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"Worker error: {e}\n{traceback.format_exc()}")
            await asyncio.sleep(5)


def start_worker():
    global _worker_task
    if _worker_task is None or _worker_task.done():
        _worker_task = asyncio.create_task(_worker_loop())
        logger.info("Worker task created")


async def stop_worker() -> None:
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
    logger.info("Background worker stopped")

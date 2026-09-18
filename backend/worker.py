import asyncio
import json
import logging
import traceback
from pathlib import Path
from backend import config
from backend.database import (
    compare_and_set_task_status,
    get_next_queued_task,
    get_task,
    update_task,
)
from backend.models import TaskResponse, TaskStatus
from backend.pipeline.composer import QualityGateRetry
from backend.pipeline.orchestrator import (
    append_pipeline_log,
    format_pipeline_log,
    run_pipeline,
    run_daily_review_resume,
    run_regenerate,
    run_tts_resume,
    run_compose,
    run_footage_acquisition,
)
from backend.publishing import run_auto_publish_pipeline

# A queued task carrying this marker file in its output dir should re-run only
# the TTS stage (from an edited script) rather than the full pipeline.
REGEN_MARKER = ".regenerate"
# Resume only TTS from a persisted script, preserving title and thumbnail.
TTS_RESUME_MARKER = ".resume_tts"
# This marker resumes a reviewed task from the compose stage only.
RENDER_MARKER = ".render"
# Retry only the footage scout. The marker stores the stable status to restore
# after the worker briefly moves the task through QUEUED/SOURCING.
FOOTAGE_MARKER = ".footage"
REVIEW_RESUME_MARKER = ".resume_review"
WORKER_MODE_MARKERS = (
    REVIEW_RESUME_MARKER,
    TTS_RESUME_MARKER,
    REGEN_MARKER,
    RENDER_MARKER,
    FOOTAGE_MARKER,
)

logger = logging.getLogger("worker")

_worker_task: asyncio.Task | None = None
LogEvent = dict[str, str]
_active_log_queues: dict[str, set[asyncio.Queue[LogEvent]]] = {}
_active_task_ids: set[str] = set()


def _task_dir(task_id: str, output_dir: str | None = None) -> Path:
    return Path(output_dir) if output_dir else config.OUTPUTS_DIR / task_id


def pipeline_log_file(task_id: str, output_dir: str | None = None) -> Path:
    task_dir = _task_dir(task_id, output_dir)
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
    append_pipeline_log(_task_dir(task.id, task.output_dir), formatted)
    publish_task_log(task.id, formatted)


def _read_marker_options(marker: Path) -> dict:
    try:
        raw = marker.read_text(encoding="utf-8")
        value = json.loads(raw) if raw.strip() else {}
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Worker mode marker {marker.name} is unreadable: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Worker mode marker {marker.name} must contain a JSON object")
    if marker.name == FOOTAGE_MARKER:
        resume_status = value.get(
            "resume_status",
            TaskStatus.AWAITING_REVIEW.value,
        )
        allowed_statuses = {
            TaskStatus.AWAITING_REVIEW.value,
            TaskStatus.COMPLETE.value,
            TaskStatus.FAILED.value,
        }
        if resume_status not in allowed_statuses:
            raise RuntimeError(
                f"Worker mode marker {marker.name} has invalid resume_status"
            )
        queries = value.get("queries", [])
        if not isinstance(queries, list) or any(
            not isinstance(query, str) for query in queries
        ):
            raise RuntimeError(
                f"Worker mode marker {marker.name} has invalid queries"
            )
    elif marker.name == RENDER_MARKER:
        resume_status = value.get("resume_status")
        if resume_status is not None and resume_status not in {
            TaskStatus.AWAITING_REVIEW.value,
            TaskStatus.COMPLETE.value,
            TaskStatus.FAILED.value,
        }:
            raise RuntimeError(
                f"Worker mode marker {marker.name} has invalid resume_status"
            )
    return value


async def _worker_loop():
    logger.info("Background worker started")
    while True:
        try:
            task = await get_next_queued_task()
            if not task:
                # Keep the clock-to-execution gap tight for scheduled plans.
                # Due scheduled tasks are also prioritized by the database.
                await asyncio.sleep(1)
                continue

            task_dir = _task_dir(task.id, task.output_dir)
            marker_paths = {
                marker_name: task_dir / marker_name
                for marker_name in WORKER_MODE_MARKERS
            }
            active_markers = [
                name for name, marker in marker_paths.items() if marker.is_file()
            ]
            if len(active_markers) > 1:
                message = (
                    "Conflicting worker mode markers; choose one recovery action: "
                    + ", ".join(active_markers)
                )
                failed = await compare_and_set_task_status(
                    task.id,
                    TaskStatus.QUEUED,
                    TaskStatus.FAILED,
                    error_message=message,
                    expected_updated_at=task.updated_at,
                    publication_safety_hold=False,
                )
                if failed:
                    _active_task_ids.add(task.id)
                    _active_log_queues.setdefault(task.id, set())
                    _persist_and_publish(task, f"Task failed: {message}")
                    finish_task_logs(task.id, TaskStatus.FAILED.value)
                continue

            selected_marker_name = active_markers[0] if active_markers else None
            selected_marker = (
                marker_paths[selected_marker_name] if selected_marker_name else None
            )
            mode_by_marker = {
                REVIEW_RESUME_MARKER: ("resume-review", TaskStatus.REVIEWING),
                TTS_RESUME_MARKER: ("resume-tts", TaskStatus.TTS),
                REGEN_MARKER: ("regenerate", TaskStatus.TITLING),
                RENDER_MARKER: ("render", TaskStatus.COMPOSING),
                FOOTAGE_MARKER: ("footage", TaskStatus.SOURCING),
            }
            mode_name, claim_status = mode_by_marker.get(
                selected_marker_name,
                ("", TaskStatus.EXTRACTING),
            )
            options: dict = {}
            if selected_marker is not None:
                try:
                    options = _read_marker_options(selected_marker)
                except RuntimeError as exc:
                    failed = await compare_and_set_task_status(
                        task.id,
                        TaskStatus.QUEUED,
                        TaskStatus.FAILED,
                        error_message=str(exc),
                        expected_updated_at=task.updated_at,
                        publication_safety_hold=False,
                    )
                    if failed:
                        _active_task_ids.add(task.id)
                        _active_log_queues.setdefault(task.id, set())
                        _persist_and_publish(task, f"Task failed: {exc}")
                        finish_task_logs(task.id, TaskStatus.FAILED.value)
                    continue

            claimed = await compare_and_set_task_status(
                task.id,
                TaskStatus.QUEUED,
                claim_status,
                error_message=None,
                expected_updated_at=task.updated_at,
            )
            if not claimed:
                # Another worker/process owns the snapshot. Never remove its
                # marker or overwrite its state with a failure.
                continue

            _active_task_ids.add(task.id)
            _active_log_queues.setdefault(task.id, set())
            mode = f" [{mode_name}]" if mode_name else ""
            held = f" [scheduled for {task.scheduled_at}]" if task.scheduled_at else ""
            logger.info(f"Processing task {task.id} ({task.source_type}){mode}{held}")
            publication_attempted = False
            completion_committed = False
            try:
                _persist_and_publish(
                    task,
                    f"Processing task ({task.source_type}){mode}{held}",
                )
                if selected_marker is not None:
                    selected_marker.unlink(missing_ok=True)

                if selected_marker_name == REVIEW_RESUME_MARKER:
                    await run_daily_review_resume(
                        task,
                        log=lambda message: publish_task_log(task.id, message),
                    )
                elif selected_marker_name == TTS_RESUME_MARKER:
                    await run_tts_resume(
                        task,
                        log=lambda message: publish_task_log(task.id, message),
                    )
                elif selected_marker_name == REGEN_MARKER:
                    await run_regenerate(
                        task,
                        log=lambda message: publish_task_log(task.id, message),
                    )
                elif selected_marker_name == RENDER_MARKER:
                    await run_compose(
                        task,
                        log=lambda message: publish_task_log(task.id, message),
                    )
                elif selected_marker_name == FOOTAGE_MARKER:
                    resume_status = options.get(
                        "resume_status",
                        TaskStatus.AWAITING_REVIEW.value,
                    )
                    if resume_status == TaskStatus.COMPLETE.value:
                        resume_status = TaskStatus.PUBLISHING.value
                    await run_footage_acquisition(
                        task,
                        resume_status=resume_status,
                        supplied_queries=options.get("queries") or None,
                        log=lambda message: publish_task_log(task.id, message),
                    )
                else:
                    await run_pipeline(
                        task,
                        log=lambda message: publish_task_log(task.id, message),
                    )

                refreshed = await get_task(task.id)
                if refreshed and refreshed.status == TaskStatus.COMPLETE:
                    raise RuntimeError(
                        "Worker stage exposed COMPLETE before the publication gate"
                    )
                if refreshed and refreshed.status == TaskStatus.PUBLISHING:
                    if refreshed.suppress_next_auto_publish:
                        _persist_and_publish(
                            refreshed,
                            "Publication pipeline: skipped — completed task was "
                            "updated without republishing",
                        )
                    else:
                        # Unexpected failures from this point may occur after a
                        # destination accepted the media but before its receipt
                        # was committed. No-external-result actions clear this
                        # provisional risk again before finalization.
                        publication_attempted = True
                        publication = await run_auto_publish_pipeline(refreshed)
                        if publication.action in {
                            "not_applicable",
                            "awaiting_review",
                            "blocked",
                        }:
                            publication_attempted = False
                        _persist_and_publish(
                            refreshed,
                            f"Publication pipeline: {publication.action} — "
                            f"{publication.reason}",
                        )
                        if publication.action == "failed":
                            raise RuntimeError(
                                "Automatic publication failed or has an "
                                f"indeterminate external result: {publication.reason}"
                            )
                    finalized = await compare_and_set_task_status(
                        refreshed.id,
                        TaskStatus.PUBLISHING,
                        TaskStatus.COMPLETE,
                        error_message=None,
                        expected_updated_at=refreshed.updated_at,
                        suppress_next_auto_publish=(
                            False
                            if refreshed.suppress_next_auto_publish
                            else None
                        ),
                        publication_safety_hold=False,
                    )
                    if not finalized:
                        raise RuntimeError(
                            "Task state changed before publication could be finalized"
                        )
                    completion_committed = True
                final_task = await get_task(task.id)
                if final_task and final_task.status == TaskStatus.COMPLETE:
                    # Copy is a separate output; a writing failure must not
                    # invalidate a finished cut or repeat any social publish.
                    from backend.social_copy import start_generation
                    try:
                        start_generation(final_task)
                    except (ValueError, OSError) as exc:
                        logger.warning("Upload copy could not start for %s: %s", task.id, exc)
                finish_task_logs(
                    task.id,
                    final_task.status.value if final_task else TaskStatus.COMPLETE.value,
                )
            except QualityGateRetry as retry:
                # Semantic/visual quality findings are recoverable pipeline
                # feedback, not a terminal task failure. Preserve the rejected
                # candidate and report, enqueue another compose-only cycle,
                # and keep the task out of FAILED while the worker continues.
                retry_marker = _task_dir(task.id, task.output_dir) / RENDER_MARKER
                temporary_marker = retry_marker.with_suffix(".tmp")
                temporary_marker.write_text(
                    json.dumps({"quality_retry": True}), encoding="utf-8"
                )
                temporary_marker.replace(retry_marker)
                _persist_and_publish(
                    task,
                    f"Quality gate requested another compose cycle: {retry}",
                )
                await update_task(
                    task.id,
                    status=TaskStatus.QUEUED.value,
                    error_message=None,
                    publication_safety_hold=False,
                )
                finish_task_logs(task.id, TaskStatus.QUEUED.value)
            except Exception as exc:
                logger.error(f"Task {task.id} failed: {exc}\n{traceback.format_exc()}")
                try:
                    _persist_and_publish(task, f"Task failed: {exc}")
                except Exception as log_exc:  # noqa: BLE001 - state safety first
                    logger.error(
                        "Could not persist failure log for task %s: %s",
                        task.id,
                        log_exc,
                    )
                if completion_committed:
                    # Publication/skip and PUBLISHING->COMPLETE are already
                    # durable. A later log/read failure must not roll the task
                    # back to FAILED and make a duplicate publication possible.
                    finish_task_logs(task.id, TaskStatus.COMPLETE.value)
                else:
                    failure_fields = {
                        "status": TaskStatus.FAILED.value,
                        "error_message": str(exc),
                    }
                    if publication_attempted:
                        # The external destination may have accepted the post
                        # before the exception. All recovery modes must skip
                        # automatic publication until an operator verifies it.
                        failure_fields["publication_safety_hold"] = True
                    else:
                        # Render/marker failures and the skip-publication path
                        # have no uncertain external side effect. Preserve any
                        # rework suppression so a later retry still cannot
                        # republish the completed edition.
                        failure_fields["publication_safety_hold"] = False
                    await update_task(task.id, **failure_fields)
                    finish_task_logs(task.id, TaskStatus.FAILED.value)
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

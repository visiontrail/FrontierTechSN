import asyncio
import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from sse_starlette.sse import EventSourceResponse
from backend import database as db
from backend.models import (
    TaskListResponse,
    TaskResponse,
    SourceType,
    TaskStatus,
    ScriptUpdate,
    FootageAcquireRequest,
    TaskSchedule,
    TaskPublicationRequest,
    TaskPublicationDeleteRequest,
    TaskPublicationReconcileRequest,
    normalize_schedule,
)
# Imported as a module, not by name: the Admin console rebinds these paths at
# runtime.
from backend import config as app_config
from backend.pipeline.footage import read_manifest
from backend.publishing import (
    PublicationLedgerError,
    delete_test_publications,
    execute_publication_plan,
    plan_publication,
    publication_manifest_view,
    publication_receipt_is_active,
    read_publication_manifest,
    reconcile_deleted_test_receipts,
)
from backend.worker import (
    FOOTAGE_MARKER,
    REGEN_MARKER,
    RENDER_MARKER,
    REVIEW_RESUME_MARKER,
    TTS_RESUME_MARKER,
    WORKER_MODE_MARKERS,
    is_task_logging_active,
    pipeline_log_file,
    subscribe_task_logs,
    unsubscribe_task_logs,
)

router = APIRouter(prefix="/api/tasks", tags=["tasks"])
_task_operation_locks: dict[str, asyncio.Lock] = {}


def _task_operation_lock(task_id: str) -> asyncio.Lock:
    # The service intentionally runs one ASGI process because its worker and
    # log subscribers are in-process. Serialize script writes and mode claims
    # within that same process so a render cannot race an editor save.
    return _task_operation_locks.setdefault(task_id, asyncio.Lock())


async def _fail_closed_publication_state(
    task_id: str,
    *,
    expected_updated_at: str,
    error_message: str,
) -> bool:
    """Persist an indeterminate external outcome despite benign DB metadata churn."""
    failed = await db.compare_and_set_task_status(
        task_id,
        TaskStatus.PUBLISHING,
        TaskStatus.FAILED,
        error_message=error_message,
        expected_updated_at=expected_updated_at,
        publication_safety_hold=True,
    )
    if failed:
        return True
    latest = await db.get_task(task_id)
    if latest is None or latest.status != TaskStatus.PUBLISHING:
        return False
    return await db.compare_and_set_task_status(
        task_id,
        TaskStatus.PUBLISHING,
        TaskStatus.FAILED,
        error_message=error_message,
        expected_updated_at=latest.updated_at,
        publication_safety_hold=True,
    )


async def _queue_worker_mode(
    task: TaskResponse,
    *,
    out_dir: Path,
    marker_name: str,
    claim_status: TaskStatus,
    marker_payload: str,
    action_label: str,
    preclaim_validate: Callable[[TaskResponse], None] | None = None,
) -> None:
    """Atomically supersede any stale recovery mode and expose one marker.

    A completed task also stores its one-shot publication suppression in the
    database as part of the first claim, so a crash before the marker write
    cannot turn a later cross-mode recovery into a duplicate publication.
    """
    async with _task_operation_lock(task.id):
        current = await db.get_task(task.id)
        if (
            current is None
            or current.status != task.status
            or current.updated_at != task.updated_at
        ):
            raise HTTPException(
                409,
                f"Task state changed before {action_label} could be queued",
            )
        if current.publication_safety_hold:
            raise HTTPException(
                409,
                "Task is locked by a publication safety hold; verify external "
                "destinations and reconcile the hold before changing its media",
            )
        if preclaim_validate is not None:
            preclaim_validate(current)

        claimed = await db.compare_and_set_task_status(
            task.id,
            task.status,
            claim_status,
            error_message=None,
            expected_updated_at=task.updated_at,
            suppress_next_auto_publish=(
                True if task.status == TaskStatus.COMPLETE else None
            ),
        )
        if not claimed:
            raise HTTPException(
                409,
                f"Task state changed before {action_label} could be queued",
            )

        marker = out_dir / marker_name
        staged_marker = out_dir / f"{marker_name}.tmp"
        try:
            # The new explicit user choice supersedes crash-left mode markers.
            # Do this only after the CAS claim so a losing request cannot
            # delete the winner's intent.
            for stale_name in WORKER_MODE_MARKERS:
                (out_dir / stale_name).unlink(missing_ok=True)
                (out_dir / f"{stale_name}.tmp").unlink(missing_ok=True)
            staged_marker.write_text(marker_payload, encoding="utf-8")
            staged_marker.replace(marker)
            queued = await db.compare_and_set_task_status(
                task.id,
                claim_status,
                TaskStatus.QUEUED,
                error_message=None,
            )
            if not queued:
                raise HTTPException(
                    409,
                    f"Task state changed while {action_label} was being queued",
                )
        except BaseException:
            staged_marker.unlink(missing_ok=True)
            marker.unlink(missing_ok=True)
            await db.compare_and_set_task_status(
                task.id,
                claim_status,
                task.status,
                error_message=task.error_message,
                suppress_next_auto_publish=task.suppress_next_auto_publish,
            )
            raise


@router.post("/{task_id}/schedule", response_model=TaskResponse)
async def reschedule_task(task_id: str, body: TaskSchedule):
    """Move a queued task's start time, or clear it (null) to start now."""
    task = await db.get_task(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    if task.status != TaskStatus.QUEUED:
        raise HTTPException(409, "Only a queued task's start time can be changed")

    try:
        start_at = normalize_schedule(body.scheduled_at)
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    await db.reschedule_task(task_id, start_at)
    return await db.get_task(task_id)


@router.get("", response_model=TaskListResponse)
async def list_tasks():
    tasks = await db.list_tasks()
    return TaskListResponse(tasks=tasks)


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str):
    task = await db.get_task(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    return task


@router.get("/{task_id}/publications")
async def get_task_publications(task_id: str):
    task = await db.get_task(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    try:
        return publication_manifest_view(task)
    except PublicationLedgerError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post("/{task_id}/publications")
async def publish_task_now(task_id: str, body: TaskPublicationRequest):
    snapshot = await db.get_task(task_id)
    if not snapshot:
        raise HTTPException(404, "Task not found")
    if snapshot.status != TaskStatus.COMPLETE:
        raise HTTPException(
            409,
            "Only a completed task can be published; reconcile any publication "
            "safety hold before starting a new attempt",
        )
    async with _task_operation_lock(task_id):
        task = await db.get_task(task_id)
        if (
            task is None
            or task.status != snapshot.status
            or task.updated_at != snapshot.updated_at
        ):
            raise HTTPException(409, "Task state changed before publication started")
        try:
            plan = plan_publication(
                task,
                targets=body.targets,
                test_mode=body.test_mode,
                visibility=body.visibility,
            )
        except (PublicationLedgerError, ValueError, FileNotFoundError) as exc:
            # A pure preflight refusal has not touched an external destination;
            # preserve the completed cut and its stable UI state.
            raise HTTPException(409, str(exc)) from exc
        if not plan.pending_targets:
            return publication_manifest_view(task, plan.manifest)
        claimed = await db.compare_and_set_task_status(
            task.id,
            task.status,
            TaskStatus.PUBLISHING,
            error_message=None,
            expected_updated_at=task.updated_at,
        )
        if not claimed:
            raise HTTPException(409, "Task state changed before publication started")
        publishing_task = await db.get_task(task.id)
        if publishing_task is None:
            raise HTTPException(404, "Task disappeared before publication started")
        try:
            result = await execute_publication_plan(
                publishing_task,
                plan=plan,
                test_mode=body.test_mode,
            )
        except Exception as exc:
            await _fail_closed_publication_state(
                task.id,
                expected_updated_at=publishing_task.updated_at,
                error_message=(
                    "Manual publication failed or has an indeterminate external "
                    f"result: {exc}"
                ),
            )
            if isinstance(exc, (ValueError, FileNotFoundError)):
                raise HTTPException(409, str(exc)) from exc
            raise HTTPException(502, f"Publication failed: {exc}") from exc
        completed = await db.compare_and_set_task_status(
            task.id,
            TaskStatus.PUBLISHING,
            TaskStatus.COMPLETE,
            error_message=None,
            expected_updated_at=publishing_task.updated_at,
            suppress_next_auto_publish=False,
            publication_safety_hold=False,
        )
        if not completed:
            await _fail_closed_publication_state(
                task.id,
                expected_updated_at=publishing_task.updated_at,
                error_message=(
                    "Publication completed externally but local finalization "
                    "could not be confirmed; verify destinations before retrying"
                ),
            )
            raise HTTPException(
                409,
                "Publication finished but task state changed before finalization",
            )
        return publication_manifest_view(publishing_task, result)


@router.post("/{task_id}/publications/delete-test")
async def delete_task_test_publications(task_id: str, body: TaskPublicationDeleteRequest):
    snapshot = await db.get_task(task_id)
    if not snapshot:
        raise HTTPException(404, "Task not found")
    recovering_deletion = bool(
        snapshot.status == TaskStatus.FAILED
        and snapshot.publication_safety_hold
    )
    if snapshot.status != TaskStatus.COMPLETE and not recovering_deletion:
        raise HTTPException(
            409,
            "Only a completed task or a publication-safety recovery can delete publications",
        )
    async with _task_operation_lock(task_id):
        task = await db.get_task(task_id)
        if (
            task is None
            or task.status != snapshot.status
            or task.updated_at != snapshot.updated_at
        ):
            raise HTTPException(409, "Task state changed before deletion started")
        try:
            manifest = read_publication_manifest(task)
        except PublicationLedgerError as exc:
            raise HTTPException(409, str(exc)) from exc
        platforms = manifest.get("platforms", {})
        selected_targets = [
            target
            for target in body.targets
            if (platforms.get(target) or {}).get("status") == "published"
        ]
        selected = [platforms[target] for target in selected_targets]
        if any(not item.get("test_mode") for item in selected):
            raise HTTPException(
                409,
                "Refusing to delete a publication that was not recorded as test_mode",
            )
        if not selected_targets:
            return publication_manifest_view(task, manifest)
        claimed = await db.compare_and_set_task_status(
            task.id,
            task.status,
            TaskStatus.PUBLISHING,
            error_message=None,
            expected_updated_at=task.updated_at,
        )
        if not claimed:
            raise HTTPException(409, "Task state changed before deletion started")
        publishing_task = await db.get_task(task.id)
        if publishing_task is None:
            raise HTTPException(404, "Task disappeared before deletion started")
        try:
            result = await delete_test_publications(
                publishing_task,
                targets=selected_targets,
            )
        except Exception as exc:
            await _fail_closed_publication_state(
                task.id,
                expected_updated_at=publishing_task.updated_at,
                error_message=(
                    "Publication deletion failed or has an indeterminate external "
                    f"result: {exc}"
                ),
            )
            raise HTTPException(502, f"Deletion failed: {exc}") from exc
        completed = await db.compare_and_set_task_status(
            task.id,
            TaskStatus.PUBLISHING,
            TaskStatus.FAILED if recovering_deletion else TaskStatus.COMPLETE,
            error_message=(
                "Recorded test publications were deleted; verify whether any "
                "unrecorded external result exists before resuming publication"
                if recovering_deletion
                else None
            ),
            expected_updated_at=publishing_task.updated_at,
            publication_safety_hold=True if recovering_deletion else None,
        )
        if not completed:
            await _fail_closed_publication_state(
                task.id,
                expected_updated_at=publishing_task.updated_at,
                error_message=(
                    "Publication deletion completed externally but local "
                    "finalization could not be confirmed; verify destinations"
                ),
            )
            raise HTTPException(
                409,
                "Deletion finished but task state changed before finalization",
            )
        return publication_manifest_view(publishing_task, result)


@router.post("/{task_id}/publications/reconcile", response_model=TaskResponse)
async def reconcile_task_publication(
    task_id: str,
    body: TaskPublicationReconcileRequest,
):
    """Clear a durable publication hold after explicit operator verification."""
    snapshot = await db.get_task(task_id)
    if not snapshot:
        raise HTTPException(404, "Task not found")
    if not (
        snapshot.status == TaskStatus.FAILED
        and snapshot.publication_safety_hold
    ):
        raise HTTPException(409, "Task has no publication safety hold to reconcile")
    if not body.acknowledge_no_unrecorded_publication:
        raise HTTPException(
            400,
            "Explicitly acknowledge that all configured destinations were verified",
        )
    async with _task_operation_lock(task_id):
        task = await db.get_task(task_id)
        if (
            task is None
            or task.status != snapshot.status
            or task.updated_at != snapshot.updated_at
            or not task.publication_safety_hold
        ):
            raise HTTPException(409, "Task state changed before reconciliation")
        try:
            # A missing ledger is valid here: it is precisely the indeterminate
            # case the operator is acknowledging. Any existing ledger must pass
            # the strict schema before its safety hold can be cleared.
            read_publication_manifest(task)
            if body.confirmed_deleted_test_targets:
                reconcile_deleted_test_receipts(
                    task,
                    body.confirmed_deleted_test_targets,
                )
        except PublicationLedgerError as exc:
            raise HTTPException(409, str(exc)) from exc
        reconciled = await db.compare_and_set_task_status(
            task.id,
            TaskStatus.FAILED,
            TaskStatus.COMPLETE,
            error_message=None,
            expected_updated_at=task.updated_at,
            publication_safety_hold=False,
        )
        if not reconciled:
            raise HTTPException(409, "Task state changed before reconciliation")
        current = await db.get_task(task.id)
        if current is None:
            raise HTTPException(404, "Task disappeared after reconciliation")
        return current


@router.get("/{task_id}/footage")
async def get_task_footage(task_id: str):
    task = await db.get_task(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    out_dir = Path(task.output_dir) if task.output_dir else app_config.OUTPUTS_DIR / task_id
    manifest = read_manifest(out_dir)
    if manifest is not None:
        return manifest
    provider_id = task.config.footage_provider
    uses_web = provider_id in {"hybrid", "opencli_web"}
    provider = {
        "hybrid": "Hybrid: Wikimedia Commons + YouTube",
        "opencli_web": "YouTube",
    }.get(provider_id, "Wikimedia Commons")
    return {
        "task_id": task_id,
        "status": "not_started",
        "provider": provider,
        "provider_id": provider_id,
        "license_policy": "review_required" if uses_web else "open_only",
        "license_allowlist": (
            [] if provider_id == "opencli_web"
            else ["Public Domain", "CC0", "CC BY", "CC BY-SA"]
        ),
        "rights_review_required": uses_web,
        "publication_blockers": (
            ["Review reuse rights for every YouTube clip before publication"]
            if uses_web else []
        ),
        "requested_clip_count": task.config.footage_clip_count,
        "planner": "",
        "queries": [],
        "clips": [],
        "errors": [],
    }


@router.get("/{task_id}/footage/{clip_id}/file")
async def get_task_footage_file(task_id: str, clip_id: str):
    task = await db.get_task(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    out_dir = Path(task.output_dir) if task.output_dir else app_config.OUTPUTS_DIR / task_id
    manifest = read_manifest(out_dir)
    if manifest is None:
        raise HTTPException(404, "Footage manifest not available")
    clip = next((item for item in manifest.get("clips", []) if item.get("id") == clip_id), None)
    if clip is None:
        raise HTTPException(404, "Footage clip not found")

    path = (out_dir / str(clip.get("local_path") or "")).resolve()
    out_dir_resolved = out_dir.resolve()
    if path != out_dir_resolved and out_dir_resolved not in path.parents:
        raise HTTPException(400, "Invalid footage path")
    if not path.is_file():
        raise HTTPException(404, "Footage file missing")
    return FileResponse(
        path,
        media_type=str(clip.get("mime_type") or "video/webm"),
        filename=path.name,
    )


@router.post("/{task_id}/footage/acquire", response_model=TaskResponse)
async def acquire_task_footage(task_id: str, body: FootageAcquireRequest):
    task = await db.get_task(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    if task.status not in (TaskStatus.AWAITING_REVIEW, TaskStatus.COMPLETE, TaskStatus.FAILED):
        raise HTTPException(409, "Task must be paused, complete, or failed before footage can be retried")
    if not task.script_path or not Path(task.script_path).exists():
        raise HTTPException(400, "No script available for footage planning")

    out_dir = Path(task.output_dir) if task.output_dir else app_config.OUTPUTS_DIR / task_id
    out_dir.mkdir(parents=True, exist_ok=True)
    marker_payload = {
        "resume_status": task.status.value,
        "queries": body.queries,
    }
    await _queue_worker_mode(
        task,
        out_dir=out_dir,
        marker_name=FOOTAGE_MARKER,
        claim_status=TaskStatus.SOURCING,
        marker_payload=json.dumps(marker_payload),
        action_label="footage acquisition",
    )
    return await db.get_task(task_id)


@router.get("/{task_id}/video")
async def download_video(
    task_id: str,
    artifact: Literal["final", "retained"] = "final",
):
    task = await db.get_task(task_id)
    if not task or not task.video_path:
        raise HTTPException(404, "Video not available")
    if task.video_artifact_state != artifact:
        raise HTTPException(
            409,
            f"Requested {artifact} video is unavailable while task status is "
            f"{task.status.value}; request artifact={task.video_artifact_state} instead",
            headers={
                "X-Video-Artifact-State": task.video_artifact_state,
                "X-Task-Status": task.status.value,
            },
        )
    path = Path(task.video_path)
    if not path.exists():
        raise HTTPException(404, "Video file missing")
    filename = (
        f"{task_id}.mp4"
        if artifact == "final"
        else f"{task_id}-retained.mp4"
    )
    return FileResponse(
        path,
        media_type="video/mp4",
        filename=filename,
        headers={
            "X-Video-Artifact-State": artifact,
            "X-Task-Status": task.status.value,
        },
    )


@router.get("/{task_id}/thumbnail")
async def download_thumbnail(task_id: str):
    task = await db.get_task(task_id)
    if not task or not task.thumbnail_path:
        raise HTTPException(404, "Thumbnail not available")
    path = Path(task.thumbnail_path)
    if not path.is_file():
        raise HTTPException(404, "Thumbnail file missing")
    media_type = {
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(path.suffix.lower(), "image/jpeg")
    return FileResponse(path, media_type=media_type, filename=f"{task_id}_thumbnail{path.suffix}")


@router.get("/{task_id}/thumbnail/prompt")
async def download_thumbnail_prompt(task_id: str):
    task = await db.get_task(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    out_dir = Path(task.output_dir) if task.output_dir else app_config.OUTPUTS_DIR / task_id
    path = out_dir / "thumbnail" / "prompt.txt"
    if not path.is_file():
        raise HTTPException(404, "Thumbnail prompt not available")
    return FileResponse(path, media_type="text/plain", filename=f"{task_id}_thumbnail_prompt.txt")


@router.get("/{task_id}/audio")
async def download_audio(task_id: str):
    task = await db.get_task(task_id)
    if not task or not task.audio_path:
        raise HTTPException(404, "Audio not available")
    path = Path(task.audio_path)
    if not path.exists():
        raise HTTPException(404, "Audio file missing")
    return FileResponse(path, media_type="audio/wav", filename=f"{task_id}.wav")


@router.get("/{task_id}/script")
async def get_script(task_id: str):
    task = await db.get_task(task_id)
    if not task or not task.script_path:
        raise HTTPException(404, "Script not available")
    path = Path(task.script_path)
    if not path.exists():
        raise HTTPException(404, "Script file missing")
    return FileResponse(path, media_type="text/plain", filename=f"{task_id}_script.txt")


@router.get("/{task_id}/logs/stream")
async def stream_task_logs(task_id: str):
    task = await db.get_task(task_id)
    if not task:
        raise HTTPException(404, "Task not found")

    async def event_stream():
        active = is_task_logging_active(task_id)
        queue = subscribe_task_logs(task_id) if active else None
        log_path = pipeline_log_file(task_id, task.output_dir)

        if log_path.exists():
            for line in log_path.read_text(encoding="utf-8").splitlines():
                yield {"event": "log", "data": line}

        if not active:
            refreshed = await db.get_task(task_id)
            status = refreshed.status.value if refreshed else task.status.value
            yield {"event": "status", "data": status}
            return

        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield {"event": "ping", "data": ""}
                    continue

                yield event
                if event.get("event") == "status":
                    return
        finally:
            if queue:
                unsubscribe_task_logs(task_id, queue)

    return EventSourceResponse(event_stream())


@router.put("/{task_id}/script")
async def update_script(task_id: str, body: ScriptUpdate):
    async with _task_operation_lock(task_id):
        task = await db.get_task(task_id)
        if not task:
            raise HTTPException(404, "Task not found")
        if task.status not in {
            TaskStatus.AWAITING_REVIEW,
            TaskStatus.COMPLETE,
            TaskStatus.FAILED,
        }:
            raise HTTPException(409, "Script cannot be edited while the task is processing")
        if task.publication_safety_hold:
            raise HTTPException(
                409,
                "Script cannot be edited while a publication safety hold is active",
            )

        out_dir = Path(task.output_dir) if task.output_dir else app_config.OUTPUTS_DIR / task_id
        out_dir.mkdir(parents=True, exist_ok=True)
        path = Path(task.script_path) if task.script_path else out_dir / "script.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        staged = path.with_name(f".{path.name}.edit.tmp")
        staged.write_text(body.content, encoding="utf-8")
        try:
            # Always bump updated_at before publishing the atomic file replace.
            # A route holding an older task snapshot will then lose its CAS.
            update_fields: dict[str, object] = {"script_path": str(path)}
            if task.status == TaskStatus.COMPLETE:
                # The prior video/audio revision is no longer publishable once
                # its source script changes. Preserve the artifacts for audit,
                # but return the task to review until audio and video are rebuilt.
                update_fields["status"] = TaskStatus.AWAITING_REVIEW.value
                update_fields["suppress_next_auto_publish"] = True
            await db.update_task(task_id, **update_fields)
            staged.replace(path)
        finally:
            staged.unlink(missing_ok=True)
    return {"ok": True}


@router.post("/{task_id}/regenerate", response_model=TaskResponse)
async def regenerate_task(task_id: str):
    task = await db.get_task(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    if task.status not in (
        TaskStatus.AWAITING_REVIEW,
        TaskStatus.COMPLETE,
        TaskStatus.FAILED,
    ):
        raise HTTPException(409, "Task is not available for audio regeneration")
    if not task.script_path or not Path(task.script_path).exists():
        raise HTTPException(400, "No script available to regenerate from")

    out_dir = Path(task.output_dir) if task.output_dir else app_config.OUTPUTS_DIR / task_id
    out_dir.mkdir(parents=True, exist_ok=True)
    await _queue_worker_mode(
        task,
        out_dir=out_dir,
        marker_name=REGEN_MARKER,
        claim_status=TaskStatus.TITLING,
        marker_payload="",
        action_label="regeneration",
    )

    return await db.get_task(task_id)


@router.post("/{task_id}/resume-tts", response_model=TaskResponse)
async def resume_task_tts(task_id: str):
    task = await db.get_task(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    if task.status != TaskStatus.FAILED:
        raise HTTPException(409, "Task must be failed before TTS can be resumed")

    out_dir = Path(task.output_dir) if task.output_dir else app_config.OUTPUTS_DIR / task_id
    script_path = Path(task.script_path) if task.script_path else out_dir / "script.txt"
    if not script_path.is_file():
        raise HTTPException(400, "No script available to resume TTS from")

    out_dir.mkdir(parents=True, exist_ok=True)
    await _queue_worker_mode(
        task,
        out_dir=out_dir,
        marker_name=TTS_RESUME_MARKER,
        claim_status=TaskStatus.TTS,
        marker_payload="",
        action_label="TTS resume",
    )
    return await db.get_task(task_id)


@router.post("/{task_id}/resume-review", response_model=TaskResponse)
async def resume_daily_review(task_id: str):
    task = await db.get_task(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    if task.source_type != SourceType.NEWS_DAILY:
        raise HTTPException(400, "Review resume is only available for daily-news tasks")
    if task.status != TaskStatus.FAILED:
        raise HTTPException(409, "Task must be failed before its saved review can be resumed")

    out_dir = Path(task.output_dir) if task.output_dir else app_config.OUTPUTS_DIR / task_id
    if not (out_dir / "research" / "dossier.json").is_file() or not (out_dir / "script.draft.txt").is_file():
        raise HTTPException(400, "Saved research dossier or yhroot draft is missing")
    await _queue_worker_mode(
        task,
        out_dir=out_dir,
        marker_name=REVIEW_RESUME_MARKER,
        claim_status=TaskStatus.REVIEWING,
        marker_payload="",
        action_label="review resume",
    )
    return await db.get_task(task_id)


@router.post("/{task_id}/render", response_model=TaskResponse)
async def render_task(task_id: str):
    task = await db.get_task(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    if task.status not in (
        TaskStatus.AWAITING_REVIEW,
        TaskStatus.COMPLETE,
        TaskStatus.FAILED,
    ):
        raise HTTPException(
            409,
            "Task is neither awaiting review, complete, nor resumable after failure",
        )
    if not task.audio_path or not Path(task.audio_path).exists():
        raise HTTPException(400, "No audio available to render from")

    out_dir = Path(task.output_dir) if task.output_dir else app_config.OUTPUTS_DIR / task_id
    out_dir.mkdir(parents=True, exist_ok=True)

    def validate_narration(current: TaskResponse) -> None:
        from backend.pipeline.composer import _narration_manifest_failures

        failures = _narration_manifest_failures(
            current.script_path or out_dir / "script.txt",
            current.audio_path or "",
            current.config.tts_model,
        )
        if failures:
            raise HTTPException(
                409,
                "The saved script no longer matches the generated audio; "
                "re-generate audio before rendering. " + "; ".join(failures),
            )

    await _queue_worker_mode(
        task,
        out_dir=out_dir,
        marker_name=RENDER_MARKER,
        claim_status=TaskStatus.COMPOSING,
        marker_payload=json.dumps({"resume_status": task.status.value}),
        action_label="render",
        preclaim_validate=validate_narration,
    )

    return await db.get_task(task_id)


@router.delete("/{task_id}")
async def delete_task(task_id: str):
    snapshot = await db.get_task(task_id)
    if not snapshot:
        raise HTTPException(404, "Task not found")
    if snapshot.status not in {
        TaskStatus.QUEUED,
        TaskStatus.AWAITING_REVIEW,
        TaskStatus.COMPLETE,
        TaskStatus.FAILED,
    }:
        raise HTTPException(409, "Task cannot be deleted while it is processing")
    async with _task_operation_lock(task_id):
        task = await db.get_task(task_id)
        if (
            task is None
            or task.status != snapshot.status
            or task.updated_at != snapshot.updated_at
        ):
            raise HTTPException(409, "Task state changed before deletion started")
        if task.publication_safety_hold:
            raise HTTPException(
                409,
                "Task deletion is blocked by a durable publication safety hold; "
                "verify external destinations or complete the recovery first",
            )
        try:
            manifest = read_publication_manifest(task)
        except PublicationLedgerError as exc:
            raise HTTPException(409, str(exc)) from exc
        active_receipts = [
            platform
            for platform, entry in (manifest.get("platforms") or {}).items()
            if publication_receipt_is_active(platform, entry)
        ]
        if active_receipts:
            raise HTTPException(
                409,
                "Delete the active external publication receipt(s) first: "
                + ", ".join(sorted(active_receipts)),
            )
        claimed = await db.compare_and_set_task_status(
            task.id,
            task.status,
            TaskStatus.FAILED,
            error_message="Task deletion claimed",
            expected_updated_at=task.updated_at,
        )
        if not claimed:
            raise HTTPException(409, "Task state changed before deletion started")
        if task.output_dir:
            out = Path(task.output_dir)
            if out.exists():
                shutil.rmtree(out)
        await db.delete_task(task_id)
    return {"ok": True}

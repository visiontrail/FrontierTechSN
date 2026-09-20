"""A resumed task must deliver its enabled footage before rendering."""

import asyncio
import hashlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from backend.models import TaskConfig, TaskResponse
from backend.pipeline import composer, footage, orchestrator
from backend.daily_news import freshness


def task_and_manifest(root):
    script = root / "script.txt"
    script.write_text("Jensen Huang took a call from President Trump at the All-In Summit.")
    audio = root / "audio.wav"
    audio.write_bytes(b"audio")
    folder = root / "footage"
    folder.mkdir(exist_ok=True)
    clip = folder / "clip.mp4"
    clip.write_bytes(b"verified footage")
    task = TaskResponse(
        id="resumed-media", created_at="2026-09-15T00:00:00Z",
        updated_at="2026-09-15T00:00:00Z", source_type="news_daily",
        source_url='{"edition_date":"2026-09-15"}', status="queued",
        output_dir=str(root), script_path=str(script), audio_path=str(audio),
        config=TaskConfig(footage_enabled=True, footage_provider="hybrid",
                          target_duration_minutes=None, thumbnail_enabled=False, auto_render=True),
    )
    manifest = {
        "status": "ready", "requested_clip_count": 1, "selection_mode": "ai",
        "clips": [{"acquisition_profile": footage.acquisition_profile(), "local_path": "footage/clip.mp4", "sha256": hashlib.sha256(clip.read_bytes()).hexdigest()}],
    }
    return task, manifest


@pytest.mark.parametrize("entry", ["run_regenerate", "run_tts_resume", "run_compose", "run_daily_review_resume"])
def test_all_resume_paths_acquire_enabled_footage_before_composition(tmp_path, entry):
    task, manifest = task_and_manifest(tmp_path)
    if entry == "run_daily_review_resume":
        (tmp_path / "research").mkdir()
        (tmp_path / "research/dossier.json").write_text(
            '{"edition_date":"2026-09-15","generated_at":"2026-09-15T00:00:00Z","window_hours":36}'
        )
        (tmp_path / "script.draft.txt").write_text("Draft.")

    async def compose(**kwargs):
        assert kwargs["footage_enabled"] is True
        assert footage.acquisition_is_complete(tmp_path, footage.read_manifest(tmp_path),
                                               (tmp_path / "script.txt").read_text())
        return str(tmp_path / "future.mp4")

    with (
        patch.object(freshness, "refresh_review_history", AsyncMock()),
        patch.object(orchestrator, "update_task", AsyncMock()),
        patch.object(orchestrator, "generate_tts", AsyncMock(return_value=task.audio_path)),
        patch.object(orchestrator, "_generate_task_title", AsyncMock(return_value=SimpleNamespace(title="News"))),
        patch.object(orchestrator, "review_daily_script", AsyncMock(return_value=SimpleNamespace(script=(tmp_path / "script.txt").read_text()))),
        patch.object(orchestrator, "acquire_footage", AsyncMock(return_value=manifest)) as acquire,
        patch.object(orchestrator, "compose_video", AsyncMock(side_effect=compose)) as render,
    ):
        asyncio.run(getattr(orchestrator, entry)(task))
    acquire.assert_awaited_once()
    render.assert_awaited_once()


@pytest.mark.parametrize("change", ["none", "script", "file", "missing", "count", "orientation", "policy", "partial", "resolution"])
def test_resume_reuses_only_current_complete_intact_footage(tmp_path, change):
    task, manifest = task_and_manifest(tmp_path)
    with patch.object(orchestrator, "acquire_footage", AsyncMock(return_value=manifest)):
        asyncio.run(orchestrator._acquire_task_footage(task, tmp_path, lambda _: None))
    if change == "script":
        (tmp_path / "script.txt").write_text("A changed story.")
    elif change == "file":
        (tmp_path / "footage/clip.mp4").write_bytes(b"changed")
    elif change == "missing":
        (tmp_path / "footage/clip.mp4").unlink()
    elif change == "count":
        task.config.footage_clip_count = 2
    elif change == "orientation":
        task.config.video_orientation = "portrait"
    elif change == "policy":
        task.config.footage_provider = "wikimedia"
    elif change == "resolution":
        manifest["clips"][0].pop("acquisition_profile")
        footage._write_manifest(tmp_path, manifest)
    elif change == "partial":
        manifest["status"] = "partial"
        footage._write_manifest(tmp_path, manifest)
    with (
        patch.object(orchestrator, "update_task", AsyncMock()),
        patch.object(orchestrator, "_acquire_task_footage", AsyncMock()) as acquire,
    ):
        asyncio.run(orchestrator._ensure_task_footage(task, tmp_path, lambda _: None))
    assert acquire.await_count == (0 if change == "none" else 1)


def test_incomplete_footage_blocks_resume_render(tmp_path):
    task, manifest = task_and_manifest(tmp_path)
    manifest.update(status="partial", requested_clip_count=2)
    with (
        patch.object(orchestrator, "update_task", AsyncMock()),
        patch.object(orchestrator, "acquire_footage", AsyncMock(return_value=manifest)),
        patch.object(orchestrator, "compose_video", AsyncMock()) as render,
    ):
        with pytest.raises(RuntimeError, match="acquisition incomplete"):
            asyncio.run(orchestrator.run_compose(task))
    render.assert_not_awaited()


def test_disabled_footage_does_not_acquire(tmp_path):
    task, _ = task_and_manifest(tmp_path)
    task.config.footage_enabled = False
    with patch.object(orchestrator, "_acquire_task_footage", AsyncMock()) as acquire:
        asyncio.run(orchestrator._ensure_task_footage(task, tmp_path, lambda _: None))
    acquire.assert_not_awaited()


def test_explicit_ai_zero_selection_is_audited_and_reused(tmp_path):
    task, manifest = task_and_manifest(tmp_path)
    manifest.update(status="no_results", requested_clip_count=0, clips=[])
    with patch.object(orchestrator, "acquire_footage", AsyncMock(return_value=manifest)):
        asyncio.run(orchestrator._acquire_task_footage(task, tmp_path, lambda _: None))
    with patch.object(orchestrator, "_acquire_task_footage", AsyncMock()) as acquire:
        asyncio.run(orchestrator._ensure_task_footage(task, tmp_path, lambda _: None))
    acquire.assert_not_awaited()


def test_direct_composer_cannot_treat_absent_manifest_as_zero_requested(tmp_path):
    task, _ = task_and_manifest(tmp_path)
    with pytest.raises(RuntimeError, match="Public-footage delivery blocked"):
        asyncio.run(composer.compose_video(task.script_path, task.audio_path, str(tmp_path),
                                           footage_enabled=True))

"""Series contracts exercised through API, persistent worker, and real FFmpeg output."""
import asyncio
from unittest.mock import AsyncMock

import pytest
from PIL import Image

from backend import config, explainers as studio, explainer_pipeline as pipeline
from backend.explainer_availability import GeminiVideoDeferred, availability_reason, next_retry_at
from backend.explainer_series import validate_motion_budget


def project_input(count=1, **changes):
    return studio.ProjectInput(title="Learning with AI", research="Use an independent attempt to check what the learner can actually do.",
        scenes=[{"id": f"scene-{i:03d}", "narration": "Try a changed problem.", "visual_prompt": "A learner works with a pencil.",
                 "motion": True} for i in range(count)], **changes)


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "OUTPUTS_DIR", tmp_path)
    monkeypatch.setattr(config, "AI_PRIMARY_RATE_LIMIT_TIMEZONE", "Asia/Singapore")
    return tmp_path


def test_manual_and_interval_motion_budget_preflight(isolated):
    project_input(20)
    with pytest.raises(ValueError, match="at most 20"):
        project_input(21)
    scenes = [{"id": f"scene-{i}", "narration": "Try it.", "visual_prompt": "A learner writes.", "motion": False} for i in range(167)]
    with pytest.raises(ValueError, match="41 are selected"):
        validate_motion_budget(scenes, {"motion_mode": "gemini", "motion_every_n_scenes": 4})
    validate_motion_budget(scenes, {"motion_mode": "gemini", "motion_every_n_scenes": 10})
    # Legacy/hand-authored projects cannot bypass the production preflight.
    project = project_input(20).model_dump()
    project["scenes"].append({**project["scenes"][0], "id": "scene-extra"})
    with pytest.raises(ValueError, match="at most 20"):
        pipeline._validate(project, isolated / "legacy")


def test_only_availability_evidence_defers():
    assert availability_reason(RuntimeError("GEMINI_VIDEO_QUOTA_EXHAUSTED: limit")) == "quota_exhausted"
    assert availability_reason(RuntimeError("GEMINI_VIDEO_UNAVAILABLE: disabled")) == "video_unavailable"
    assert availability_reason(RuntimeError("GEMINI_VIDEO_GENERATION_FAILED: explicit terminal reply")) == "generation_failed"
    for code in ("SUBMISSION_UNCERTAIN", "GENERATION_TIMEOUT", "COMPOSER_NOT_READY"):
        assert availability_reason(RuntimeError("GEMINI_VIDEO_" + code)) is None


def test_next_day_is_timezone_aware_and_gives_rolling_limit_a_day(isolated):
    assert next_retry_at("2026-10-15T15:55:00+00:00") == "2026-10-16T15:55:00+00:00"


def test_persistent_wait_yields_worker_and_resumes_once_without_extending_other_waits(isolated, monkeypatch):
    first = studio.create_project(project_input())
    second = studio.create_project(project_input())
    claim = {"operation": "render", "started_at": "2026-10-14T03:00:00+00:00", "sequence": 7}
    for project in (first, second):
        studio.patch_project(project["id"], {"production": claim, "artifacts": {"thumbnail_path": "cover.png"},
                                             "scene_artifacts": [{"id": "scene-000", "motion_path": "real.mp4"}]})
    monkeypatch.setattr(studio, "now", lambda: "2026-10-15T15:55:00+00:00")
    saved = studio.defer_for_gemini(first["id"], "render", GeminiVideoDeferred("quota_exhausted", "observed limit", "scene-000"))
    deadline = saved["provider_wait"]["retry_at"]
    monkeypatch.setattr(studio, "now", lambda: "2026-10-15T16:10:00+00:00")
    later = studio.defer_for_gemini(second["id"], "render", GeminiVideoDeferred("quota_exhausted", "same limit"))
    assert later["provider_wait"]["retry_at"] == deadline
    studio.recover_interrupted()
    assert studio.production_owner() is None
    assert studio.resume_provider_waits() == []
    assert studio.read_project(first["id"])["scene_artifacts"][0]["motion_path"] == "real.mp4"
    assert studio.read_project(first["id"])["production"] == claim
    with pytest.raises(studio.ConflictError, match="scheduled automatically"):
        studio.start(first["id"], "render")
    monkeypatch.setattr(studio, "now", lambda: deadline)
    assert set(studio.resume_provider_waits()) == {first["id"], second["id"]}
    assert studio.resume_provider_waits() == []
    recovered = studio.read_project(first["id"])
    assert recovered["status"] == "queued"
    assert recovered["queue"]["sequence"] == 7
    assert recovered["production"] == claim
    assert recovered["artifacts"]["thumbnail_path"] == "cover.png"


def test_cancel_and_maintenance_prevent_automatic_recovery(isolated, monkeypatch):
    for action in (studio.cancel, studio.pause):
        project = studio.create_project(project_input())
        studio.patch_project(project["id"], {"production": {"operation": "render", "started_at": studio.now()}})
        studio.defer_for_gemini(project["id"], "render", GeminiVideoDeferred("video_unavailable", "disabled"))
        asyncio.run(action(project["id"]))
        monkeypatch.setattr(studio, "now", lambda: "2030-01-01T00:00:00+00:00")
        assert studio.resume_provider_waits() == []
        assert studio.read_project(project["id"])["provider_wait"] is None


def test_worker_catches_deferred_motion_without_marking_task_failed(isolated, monkeypatch):
    project = studio.create_project(project_input())
    studio.patch_project(project["id"], {"production": {"operation": "render", "started_at": studio.now()}})
    monkeypatch.setattr(pipeline, "render_project", AsyncMock(side_effect=GeminiVideoDeferred("video_unavailable", "disabled", "scene-000")))
    asyncio.run(studio._run(project["id"], "render"))
    saved = studio.read_project(project["id"])
    assert saved["status"] == "waiting_provider" and saved["error"] is None
    assert saved["production"]["operation"] == "render"
    assert saved["provider_wait"]["scene_id"] == "scene-000"
    assert studio.production_owner() is None


def test_thumbnail_reused_and_invalidated_by_episode_content(isolated, monkeypatch):
    async def generate(prompt, stage):
        target = stage / "provider.png"
        Image.new("RGB", (1280, 720), "#182f40").save(target)
        return target, "https://chatgpt.com/c/owned-cover"
    generator = AsyncMock(side_effect=generate)
    monkeypatch.setattr(pipeline, "_generate_still", generator)
    project = project_input().model_dump()
    root = isolated / "cover-test"
    async def run():
        first = await pipeline._thumbnail(project, root, lambda _: None)
        assert await pipeline._thumbnail(project, root, lambda _: None) == first
        assert generator.await_count == 1
        project["scenes"][0]["narration"] = "Check a different idea."
        second = await pipeline._thumbnail(project, root, lambda _: None)
        assert second["thumbnail_path"] != first["thumbnail_path"]
        assert (root / first["thumbnail_path"]).is_file()
        assert generator.await_count == 2
    asyncio.run(run())
    saved = studio.create_project(project_input())
    public = studio.public_project({**saved, "artifacts": {"thumbnail_path": "artifacts/series/cover.png"}})
    assert public["artifacts"]["thumbnail_url"].endswith("/assets/artifacts/series/cover.png")

"""3D inserts must survive assembly as bounded, silent video clips."""

import copy

import pytest

from backend import config
from backend.pipeline import composer, media_shots, multimodal_review, scene_kit, three_d
from backend.pipeline.video_format import LANDSCAPE, PORTRAIT


@pytest.mark.parametrize("frame", [LANDSCAPE, PORTRAIT])
def test_three_d_clip_is_visible_silent_and_locked(frame):
    shot = {
        "kind": "three_d", "start": 0, "duration": 8, "source_duration": 8,
        "source_start": 0, "src": "../three_d/content-address/clip.mp4",
        "copy": ["Compute capacity", "A rack contains multiple compute modules"],
        "layout": "split", "motion_cues": media_shots.motion_cues(8),
    }
    plan = {"id": "scene-02", "media_shots": [shot]}
    html = scene_kit.render_scene(scene_kit.ScenePlan.from_dict(
        plan, duration=8, scene_id="scene-02", frame=frame,
    ))
    media_shots.assert_rendered_shots(plan, html)
    document = media_shots._Elements(html)
    tag, attrs = document.elements["scene-02-shot-1-media"]
    assert tag == "video"
    assert "muted" in attrs and "loop" not in attrs
    assert "three-d-panel" in document.elements["scene-02-shot-1-panel"][1]["class"]
    assert "3D illustration · Not to scale" in html
    assert ".shot-media.three_d { width:58%; object-fit:contain;" in html
    if frame.is_portrait:
        assert ".shot-media.three_d { width:100%; height:55%; }" in html
    for invalid in (
        html.replace(shot["src"], "../wrong.mp4"),
        html.replace('data-media-start="0.00"', 'data-media-start="1.00"'),
        html.replace("muted playsinline", "loop muted playsinline"),
        html.replace('data-duration="8.00"', 'data-duration="8.01"'),
    ):
        with pytest.raises(ValueError):
            media_shots.assert_rendered_shots(plan, invalid)


@pytest.mark.parametrize("status,passed", [
    ({"enabled": True, "passed": True, "rendered_count": 0}, False),
    ({"enabled": True, "passed": False, "rendered_count": 3}, False),
    ({"enabled": True, "passed": True, "rendered_count": 3}, True),
    ({"enabled": False, "passed": True, "rendered_count": 0}, True),
])
def test_final_gate_requires_actual_three_d_renders(status, passed):
    report = {"three_d": {"contract": three_d.CONTRACT, **copy.deepcopy(status)}}
    composer._finalize_quality_report(
        report, {"passed": True}, {"passed": True}, {
            "passed": True,
            "three_d_sampling_contract": multimodal_review.THREE_D_SAMPLING_CONTRACT,
            "reviewed_three_d_shots": status["rendered_count"],
            "three_d_sample_count": status["rendered_count"] * 3,
        },
        multimodal_enabled=True,
    )
    assert report["passed"] is passed


def test_final_gate_cannot_approve_three_d_without_actual_pixel_review():
    report = {"three_d": {"contract": three_d.CONTRACT, "enabled": True, "passed": True, "rendered_count": 1}}
    composer._finalize_quality_report(
        report, {"passed": True}, {"passed": True}, {"passed": True},
        multimodal_enabled=True,
    )
    assert report["passed"] is False


def test_final_gate_rejects_missing_three_d_report_for_a_three_d_timeline():
    report = {
        "production_method": {"timeline_sha256": "same", "three_d_shot_count": 1},
        "production_preflight": {"passed": True},
        "delivery_timeline": {"passed": True, "timeline_sha256": "same"},
    }
    composer._finalize_quality_report(
        report, {"passed": True}, {"passed": True}, {"passed": True},
        multimodal_enabled=True,
    )
    assert report["passed"] is False


def test_review_cache_binds_three_d_code_setting_and_clip(tmp_path, monkeypatch):
    renderer = tmp_path / "renderer"
    (renderer / "src").mkdir(parents=True)
    source = renderer / "src" / "Scene.tsx"
    source.write_text("first")
    monkeypatch.setattr(config, "REMOTION_DIR", renderer)
    monkeypatch.setattr(config, "THREE_D_ENABLED", True)
    before = composer._review_retry_fingerprint(tmp_path, {})
    source.write_text("different camera")
    after_code = composer._review_retry_fingerprint(tmp_path, {})
    assert after_code != before
    (tmp_path / "three_d").mkdir()
    (tmp_path / "three_d" / "clip.mp4").write_bytes(b"new-geometry")
    after_clip = composer._review_retry_fingerprint(tmp_path, {})
    assert after_clip != after_code
    monkeypatch.setattr(config, "THREE_D_ENABLED", False)
    assert composer._review_retry_fingerprint(tmp_path, {}) != after_clip

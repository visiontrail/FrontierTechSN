"""3D inserts must survive assembly as bounded, silent video clips."""

import copy
import json

import pytest

from backend import config
from backend.pipeline import composer, media_shots, multimodal_review, production_plan, scene_kit, three_d
from backend.pipeline.video_format import LANDSCAPE, PORTRAIT, resolve_render_spec


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


@pytest.mark.parametrize("duration,count", [(10.23, 307), (14.56, 437), (3, 90)])
@pytest.mark.parametrize("transition", ["wipe", "iris"])
def test_three_d_boundary_covers_cannot_hide_real_motion_phases(duration, count, transition):
    # The 307/437-frame production shots settle on their last source frame:
    # hold actors also control the moving camera's endpoint. Sampling earlier
    # would hide that endpoint instead of fixing the outgoing cover collision.
    def editorial(start):
        return {"kind": "editorial", "start": start, "duration": 4,
                "copy": ["A grounded claim"], "layout": "focus",
                "transition": transition, "motion_cues": media_shots.motion_cues(4)}

    shot = {
        "kind": "three_d", "start": 4, "duration": duration,
        "source_duration": count / 30, "source_start": 0,
        "src": "../three_d/content-address/clip.mp4", "sha256": "verified-clip",
        "copy": ["A spatial explanation"], "layout": "focus",
        "transition": transition, "motion_cues": media_shots.motion_cues(duration),
        "three_d_spec": {
            "contract": three_d.CONTRACT, "representation": "illustration",
            "fps": 30, "duration_frames": count,
            "camera": {"position": [6, 4, 9], "end_position": [7, 4, 9], "target": [0, 0, 0]},
            "actors": [{"id": "ground", "kind": "object", "action": "hold",
                        "motion_start": 0, "motion_end": 1}],
        },
    }
    shots = [editorial(0), shot, editorial(4 + duration), editorial(8 + duration)]
    plan = {"id": "scene-02", "media_shots": shots}
    board = {"total_duration": duration + 12, "scenes": [
        {"id": "scene-02", "start": 0, "duration": duration + 12},
    ]}
    original = copy.deepcopy(plan)
    timeline = production_plan.build_timeline(board, [plan], fps=30)
    phases = dict(production_plan.three_d_sample_times(shot))
    assert phases["settled"] == pytest.approx((count - 1) / 30, abs=1e-6)
    assert phases["settled"] > duration - .18
    if duration == 3:
        # A short 3D shot's onset falls inside the old incoming cover too.
        assert phases["onset"] < .18

    html = scene_kit.render_scene(scene_kit.ScenePlan.from_dict(
        plan, duration=duration + 12, scene_id="scene-02", frame=LANDSCAPE,
    ))
    media_shots.assert_rendered_shots(plan, html)
    elements = media_shots._Elements(html).elements
    covers = {name: attrs for name, (_, attrs) in elements.items()
              if "shot-wipe" in attrs.get("class", "").split()}
    # Keep the ordinary editorial-to-editorial transition, with the same
    # 180ms cover on each side of its cut and the chosen animation shape.
    assert set(covers) == {"scene-02-shot-4-wipe"}
    cover = covers["scene-02-shot-4-wipe"]
    assert float(cover["data-start"]) == pytest.approx(duration + 8 - .18)
    assert float(cover["data-duration"]) == .36
    assert ('clipPath:"circle(0% at 50% 50%)"' in html) == (transition == "iris")
    for phase in phases.values():
        sample_time = shot["start"] + phase
        assert all(not float(c["data-start"]) <= sample_time
                   < float(c["data-start"]) + float(c["data-duration"]) for c in covers.values())
    # Geometry, source ownership, copy, cues and shared preview/final sample
    # events retain their original timing; only the occluding layer is removed.
    assert plan == original
    assert production_plan.build_timeline(board, [plan], fps=30) == timeline


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


@pytest.fixture
def failed_three_d_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "THREE_D_ENABLED", True)
    board = {"scenes": [
        {"id": "scene-01", "duration": 2, "text": "Welcome", "program_segment_kind": "opening"},
        {"id": "scene-02", "duration": 8, "text": "The chip remains in service."},
        {"id": "scene-03", "duration": 8, "text": "The rack contains several modules."},
    ]}
    plans = [{"id": scene["id"], "archetype": "topic"} for scene in board["scenes"]]
    saved = []
    for scene in board["scenes"][1:]:
        shot = {"kind": "editorial", "duration": 8, "layout": "focus",
                "copy": [scene["text"]], "script_excerpt": scene["text"]}
        saved.append({"id": scene["id"], "duration": 8, "passed": True, "coverage": 1,
                      "shots": [shot]})
    spec = {
        "id": "scene-02-shot-01", "representation": "illustration",
        "narration_excerpt": board["scenes"][1]["text"],
        "semantic_intent": "Show the chip structure as a schematic illustration",
        "camera": {"position": [6, 4, 9], "end_position": [7, 4, 9], "target": [0, 0, 0]},
        "actors": [{"id": "chip", "kind": "chip", "position": [0, 0, 0]}],
    }
    payloads = {
        "three_d_render_report.json": {"contract": three_d.CONTRACT, "enabled": True,
            "status": "failed", "passed": False, "required_shots": 2, "rendered_count": 0},
        "media_shots.json": {"contract_version": media_shots.CONTRACT_VERSION,
                             "passed": True, "scenes": saved},
        "three_d_plan.json": {"contract": three_d.CONTRACT,
                               "shots": [{"id": spec["id"], "spec": spec}]},
    }
    for name, payload in payloads.items():
        (tmp_path / name).write_text(json.dumps(payload))
    return board, plans, resolve_render_spec(), payloads


@pytest.mark.parametrize("status", ["failed", "interrupted"])
def test_three_d_resume_preserves_inventory_after_partial_valid_direction(
    tmp_path, failed_three_d_checkpoint, status,
):
    board, plans, render, payloads = failed_three_d_checkpoint
    report = payloads["three_d_render_report.json"]
    report["status"] = status
    (tmp_path / "three_d_render_report.json").write_text(json.dumps(report))
    before = copy.deepcopy((board, plans))
    assert composer._resuming_failed_three_d(tmp_path, board, plans, render)
    assert (board, plans) == before
    assert json.loads((tmp_path / "media_shots.json").read_text()) == payloads["media_shots.json"]


@pytest.mark.parametrize("mutation", [
    "missing", "malformed", "ready", "disabled", "contract", "unfinished_shots", "partial_coverage",
    "changed_duration", "changed_narration", "missing_scene", "missing_spec", "duplicate_spec",
    "invalid_geometry", "required_count",
])
def test_three_d_recovery_does_not_reuse_failures_without_current_valid_checkpoint(
    tmp_path, monkeypatch, failed_three_d_checkpoint, mutation,
):
    board, plans, render, payloads = failed_three_d_checkpoint
    report = payloads["three_d_render_report.json"]
    shots = payloads["media_shots.json"]
    direction = payloads["three_d_plan.json"]
    if mutation == "ready":
        report["status"] = "ready"
    elif mutation == "disabled":
        monkeypatch.setattr(config, "THREE_D_ENABLED", False)
    elif mutation == "contract":
        shots["contract_version"] = -1
    elif mutation == "unfinished_shots":
        shots["passed"] = False
    elif mutation == "partial_coverage":
        shots["scenes"][0]["coverage"] = .5
    elif mutation == "changed_duration":
        board["scenes"][1]["duration"] = 9
    elif mutation == "changed_narration":
        board["scenes"][1]["text"] = "A new unrelated story."
    elif mutation == "missing_scene":
        shots["scenes"].pop()
    elif mutation == "missing_spec":
        direction["shots"] = []
    elif mutation == "duplicate_spec":
        direction["shots"].append(copy.deepcopy(direction["shots"][0]))
    elif mutation == "invalid_geometry":
        direction["shots"][0]["spec"]["actors"][0]["kind"] = "unsupported_mesh"
    elif mutation == "required_count":
        report["required_shots"] = 1
    for name, payload in payloads.items():
        (tmp_path / name).write_text(json.dumps(payload))
    if mutation == "missing":
        (tmp_path / "three_d_render_report.json").unlink()
    elif mutation == "malformed":
        (tmp_path / "three_d_plan.json").write_text("[]")
    assert not composer._resuming_failed_three_d(tmp_path, board, plans, render)

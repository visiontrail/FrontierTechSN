import copy
import json
import subprocess
from types import SimpleNamespace

import pytest

from backend.pipeline import production_plan
from backend.pipeline import composer
from backend.pipeline import storyboard


def inputs():
    board = {"total_duration": 2.02, "scenes": [
        {"id": "scene-01", "start": 0, "duration": 1.01},
        {"id": "scene-02", "start": 1.01, "duration": 1.01},
    ]}
    plans = [{"id": "scene-01", "media_shots": [
        {"kind": "editorial", "start": 0, "duration": .51,
         "motion_cues": {"reveal_start": .05, "reveal_end": .1, "emphasis": .2, "settle": .4}},
        {"kind": "image", "start": .51, "duration": .5},
    ]}, {"id": "scene-02", "archetype": "outro"}]
    return board, plans


def test_timeline_projects_shared_boundaries_without_retiming():
    board, plans = inputs()
    original = copy.deepcopy((board, plans))
    timeline = production_plan.build_timeline(board, plans, fps=30)
    assert [(s["start"], s["end"]) for s in timeline["shots"]] == [(0, 16), (16, 31), (31, 61)]
    assert timeline["duration_frames"] == 61
    assert (board, plans) == original
    assert production_plan.preview_times(timeline) == [0.2, 0.233333, 0.4, 0.766667, 1.5]


@pytest.mark.parametrize("duration,fps,frames", [(37.58, 30, 1128), (1.01, 30, 31), (.35, 30, 11), (8.3, 30, 250)])
def test_frame_count_matches_hyperframes_partial_last_frame(duration, fps, frames):
    board = {"total_duration": duration, "scenes": [{"id": "s", "start": 0, "duration": duration}]}
    timeline = production_plan.build_timeline(board, [{"id": "s"}], fps=fps)
    assert timeline["duration_frames"] == frames
    assert timeline["shots"][-1]["end"] == frames


def test_asr_fractional_boundaries_remain_contiguous_without_retiming_speech():
    scenes = [{"id": "a", "start": 0, "duration": 14.55, "lines": [{"start": .2}]},
              {"id": "b", "start": 14.55, "duration": 15.45, "lines": [{"start": 14.555}]},
              {"id": "c", "start": 30.01, "duration": 14.99, "lines": [{"start": 30.006}]}]
    before = copy.deepcopy([s["lines"] for s in scenes])
    storyboard.normalize_scene_intervals(scenes, 45)
    assert scenes[1]["duration"] == 15.46
    assert [s["lines"] for s in scenes] == before
    timeline = production_plan.build_timeline({"scenes": scenes, "total_duration": 45},
                                             [{"id": s["id"]} for s in scenes], fps=30)
    assert timeline["shots"][-1]["end"] == 1350


@pytest.mark.parametrize("mutation", ["gap", "overlap", "missing_shot", "outside_event", "nonfinite"])
def test_rejects_incomplete_or_invalid_authoritative_timing(mutation):
    board, plans = inputs()
    if mutation == "gap": board["scenes"][1]["start"] = 1.02
    if mutation == "overlap": plans[0]["media_shots"][1]["start"] = .5
    if mutation == "missing_shot": plans[0]["media_shots"].pop()
    if mutation == "outside_event": plans[0]["media_shots"][0]["motion_cues"]["settle"] = .51
    if mutation == "nonfinite": board["scenes"][0]["duration"] = float("nan")
    with pytest.raises(ValueError):
        production_plan.build_timeline(board, plans, fps=30)


@pytest.mark.parametrize("defect", [None, "aac_priming", "vfr", "count", "origin", "missing", "probe_failure"])
def test_delivery_verifies_every_decoded_presentation_time(tmp_path, monkeypatch, defect):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"delivery bytes")
    timeline = {"fps": 30, "duration_frames": 3, "duration_seconds": .1, "input_sha256": "timeline"}
    stamps = [0, 1 / 30, 2 / 30]
    if defect == "vfr": stamps[1] += .01
    if defect == "count": stamps.pop()
    if defect == "origin": stamps = [stamp + .5 for stamp in stamps]
    if defect == "aac_priming": stamps = [stamp + .021029 for stamp in stamps]
    payload = {"streams": [{"time_base": "1/15360", "codec_type": "video"},
                           {"codec_type": "audio", "codec_name": "aac", "sample_rate": "48000", "start_time": "0"}],
               "frames": [{"best_effort_timestamp_time": value} for value in stamps]}
    if defect == "missing": payload["frames"][1] = {}

    def probe(*args, **kwargs):
        if defect == "probe_failure": raise subprocess.CalledProcessError(1, "ffprobe")
        return SimpleNamespace(stdout=json.dumps(payload))

    monkeypatch.setattr(production_plan.subprocess, "run", probe)
    report = production_plan.verify_delivery(video, timeline, tmp_path / "report.json")
    assert report["passed"] is (defect in {None, "aac_priming"})
    assert json.loads((tmp_path / "report.json").read_text()) == report
    assert report["timeline_sha256"] == "timeline"


@pytest.mark.parametrize("defect", [None, "missing_preflight", "missing_delivery", "wrong_timeline"])
def test_directed_production_cannot_pass_without_bound_evidence(defect):
    report = {"production_method": {"contract": production_plan.CONTRACT, "timeline_sha256": "current"},
              "production_preflight": {"passed": True},
              "delivery_timeline": {"passed": True, "timeline_sha256": "current"}}
    if defect == "missing_preflight": report.pop("production_preflight")
    if defect == "missing_delivery": report.pop("delivery_timeline")
    if defect == "wrong_timeline": report["delivery_timeline"]["timeline_sha256"] = "stale"
    composer._finalize_quality_report(report, {"passed": True}, {"passed": True},
                                     {"passed": True}, multimodal_enabled=True)
    assert report["passed"] is (defect is None)

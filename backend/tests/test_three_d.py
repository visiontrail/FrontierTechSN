import asyncio
import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from PIL import Image, ImageDraw

from backend.pipeline import three_d as td


def context():
    return {"id": "scene-02-shot-01", "duration": 8.01,
            "copy": ["Reported chip transaction", "Chips remain in service"],
            "narration_excerpt": "The proposed transaction would leave the chips in service.",
            "relationship_excerpt": "", "scene_id": "scene-02", "shot_index": 0}


def raw_spec():
    return {"id": context()["id"], "representation": "illustration",
            "narration_excerpt": context()["narration_excerpt"],
            "semantic_intent": "Rotate a schematic chip to show its physical structure while retaining the proposal qualifier",
            "background": "#101820", "accent": "#59C4D8",
            "camera": {"position": [6, 4, 9], "target": [0, 0, 0]},
            "actors": [{"id": "chip", "kind": "chip", "position": [0, 0, 0],
                        "end_rotation": [0, .6, 0]}]}


def test_spec_owns_frames_pixels_copy_and_excludes_executable_model_output():
    raw = raw_spec()
    raw.update(width=100, fps=1000, duration_frames=1, copy=["Invented announcement"],
               script="fetch('https://example.com')", src="https://example.com/model.glb")
    spec = td.validate_spec(raw, context(), width=3840, height=2160, fps=30)
    assert spec["duration_frames"] == 241
    assert (spec["width"], spec["height"], spec["fps"]) == (3840, 2160, 30)
    assert spec["copy"] == context()["copy"]
    assert "script" not in spec and "src" not in spec
    assert spec["actors"][0]["end_rotation"] == [0, .6, 0]


@pytest.mark.parametrize("mutation,error", [
    (lambda s: s.update(id="different"), "different editorial"),
    (lambda s: s.update(representation="captured_evidence"), "illustration"),
    (lambda s: s.update(narration_excerpt="The deal is confirmed"), "selected shot"),
    (lambda s: s["actors"][0].update(position=[0, float("nan"), 1]), "numeric value"),
    (lambda s: s["actors"][0].update(kind="remote_model"), "actor kind"),
    (lambda s: s["actors"][0].update(id="../../model"), "safe identifiers"),
    (lambda s: s["actors"][0].update(motion_start=.9, motion_end=.2), "must precede"),
    (lambda s: s["actors"][0].update(end_rotation=[0, 0, 0]), "spatial motion"),
    (lambda s: s["actors"][0].update(action="hold"), "spatial motion"),
    (lambda s: s["actors"].append(copy.deepcopy(s["actors"][0])), "unique"),
    (lambda s: s["camera"].update(position=[0, 0, 3], end_position=[0, 0, -3]), "path crosses"),
    (lambda s: s["actors"][0].update(kind="robot", end_explode=.4), "exploded views"),
    (lambda s: s.update(links=[{"from": "chip", "to": "chip"}]), "grounded relationship"),
])
def test_invalid_or_ungrounded_3d_never_reaches_executable_renderer(mutation, error):
    raw = raw_spec()
    mutation(raw)
    with pytest.raises(ValueError, match=error):
        td.validate_spec(raw, context(), width=1920, height=1080, fps=30)


def test_robot_spec_omits_unsupported_exploded_properties():
    raw = raw_spec()
    raw["actors"][0]["kind"] = "robot"
    actor = td.validate_spec(raw, context(), width=1920, height=1080, fps=30)["actors"][0]
    assert "explode" not in actor and "end_explode" not in actor


@pytest.mark.parametrize("field,maximum", [("semantic_intent", 600), ("narration_excerpt", 5000)])
def test_text_bounds_are_exact_and_report_length_without_truncation(field, maximum):
    raw = raw_spec()
    supplied = context()
    raw[field] = "x" * maximum
    if field == "narration_excerpt":
        supplied[field] = raw[field]
    assert td.validate_spec(raw, supplied, width=3840, height=2160, fps=30)[field] == raw[field]
    raw[field] += "x"
    original = copy.deepcopy(raw)
    with pytest.raises(ValueError, match=rf"{field} length={maximum + 1}.*at most {maximum} characters"):
        td.validate_spec(raw, supplied, width=3840, height=2160, fps=30)
    assert raw == original
    for invalid in (None, "", "   "):
        raw[field] = invalid
        with pytest.raises(ValueError, match=rf"{field} must be nonempty text.*{maximum} characters"):
            td.validate_spec(raw, supplied, width=3840, height=2160, fps=30)


@pytest.mark.parametrize("kind", ["chip", "rack", "satellite", "robot", "data"])
def test_primitive_cannot_override_a_semantic_mesh(kind):
    raw = raw_spec()
    raw["actors"][0].update(kind=kind, primitive="sphere")
    with pytest.raises(ValueError, match=f"actor chip .*kind={kind}.*omit primitive"):
        td.validate_spec(raw, context(), width=3840, height=2160, fps=30)


@pytest.mark.parametrize("primitive", ["box", "sphere", "cylinder", "torus"])
def test_generic_object_retains_its_requested_primitive(primitive):
    raw = raw_spec()
    raw["actors"][0].update(kind="object", primitive=primitive)
    spec = td.validate_spec(raw, context(), width=3840, height=2160, fps=30)
    assert spec["actors"][0]["primitive"] == primitive


def test_unsafe_context_scene_id_is_rejected_before_renderer():
    supplied = context()
    supplied["id"] = "../../scene"
    raw = {**raw_spec(), "id": supplied["id"]}
    with pytest.raises(ValueError, match="scene ID"):
        td.validate_spec(raw, supplied, width=1920, height=1080, fps=30)


def test_full_scene_context_cannot_move_another_narration_claim_into_selected_shot():
    supplied = {**context(), "scene_text": context()["narration_excerpt"] + " A different mechanism follows."}
    raw = {**raw_spec(), "narration_excerpt": "A different mechanism follows."}
    with pytest.raises(ValueError, match="selected shot"):
        td.validate_spec(raw, supplied, width=1920, height=1080, fps=30)


@pytest.mark.parametrize("scale,expected", [
    ([1.6, 2, .08], ["actors[0](report-sheet).scale[2]=0.08"]),
    ([8.6, .14, 4.6], ["actors[0](report-sheet).scale[0]=8.6",
                       "actors[0](report-sheet).scale[2]=4.6"]),
])
def test_invalid_scale_identifies_every_bad_coordinate_without_clamping(scale, expected):
    raw = raw_spec()
    raw["actors"][0].update(id="report-sheet", kind="object", primitive="box", scale=scale)
    original = copy.deepcopy(raw)
    with pytest.raises(ValueError) as caught:
        td.validate_spec(raw, context(), width=3840, height=2160, fps=30)
    for field in expected:
        assert field in str(caught.value)
    assert str(caught.value).count("within 0.1..4") == len(expected)
    assert raw == original


def inputs():
    scene = {"id": "scene-02", "duration": 18.01, "text": context()["narration_excerpt"],
             "program_segment_kind": "news"}
    editorial = {"kind": "editorial", "start": 0, "duration": 8.01, "layout": "focus",
                 "copy": context()["copy"], "script_excerpt": scene["text"],
                 "direction": {"purpose": "Explain the proposed transaction", "evidence_kind": "editorial"}}
    footage = {"kind": "public_footage", "start": 8.01, "duration": 10,
               "src": "../footage/verified.mp4", "sha256": "verified source", "copy": ["Captured evidence"]}
    return [{"id": scene["id"], "media_shots": [editorial, footage]}], {"scenes": [scene], "director_brief": {"theme": "swiss"}}


def test_selection_preserves_all_source_assets_and_reports_ineligible_stories():
    plans, board = inputs()
    plans[0]["media_shots"].append({**plans[0]["media_shots"][0], "layout": "flow", "duration": 4,
                                   "relationship_excerpt": "then"})
    original = copy.deepcopy((plans, board))
    contexts, skipped = td.select_shots(plans, board)
    assert contexts[0]["shot_index"] == 2 and not skipped
    assert (plans, board) == original
    plans[0]["media_shots"] = [plans[0]["media_shots"][1]]
    assert td.select_shots(plans, board)[0] == []
    assert "source media retained" in td.select_shots(plans, board)[1][0]["reason"]


def _renderer(root):
    folder = root / "remotion"
    (folder / "scripts").mkdir(parents=True)
    (folder / "src").mkdir()
    (folder / "package.json").write_text('{}')
    (folder / "package-lock.json").write_text('{}')
    (folder / "scripts/render.mjs").write_text('// fixed local renderer')
    (folder / "src/Scene.tsx").write_text('// R3F scene')
    return folder


def test_live_path_uses_configured_provider_and_reuses_only_bound_render(tmp_path, monkeypatch):
    renderer = _renderer(tmp_path)
    monkeypatch.setattr(td.config, "PROJECT_ROOT", tmp_path)
    provider = AsyncMock(return_value=("configured endpoint", "configured model", "test key"))

    async def completion(*args, **kwargs):
        kwargs["route_selected"]("https://user:secret@backup.example/v1?token=secret#secret",
                                 "actual backup model", "private API credential")
        return json.dumps([raw_spec()])

    chat = AsyncMock(side_effect=completion)
    commands = []

    async def command(name, args, **kwargs):
        commands.append(args)
        Path(args[args.index("--output") + 1]).write_bytes(b"valid local 3d")
        return "rendered"

    async def verify(output, spec, **kwargs):
        return {"passed": True, "sha256": td._file_hash(output),
                "duration": spec["duration_frames"] / spec["fps"], "frames": spec["duration_frames"],
                "width": spec["width"], "height": spec["height"], "fps": spec["fps"], "stills": []}

    monkeypatch.setattr("backend.pipeline.digester._resolve_provider", provider)
    monkeypatch.setattr("backend.pipeline.digester._chat", chat)
    monkeypatch.setattr(td, "_command", command)
    monkeypatch.setattr(td, "validate_render", verify)
    plans, board = inputs()
    footage = copy.deepcopy(plans[0]["media_shots"][1])
    directory = tmp_path / "task"
    report = asyncio.run(td.render_story_shots(plans, board, directory, width=3840, height=2160, fps=30))
    shot = plans[0]["media_shots"][0]
    assert report["passed"] and report["rendered_count"] == 1
    assert shot["kind"] == "three_d" and shot["three_d_spec"]["width"] == 3840
    assert shot["duration"] == 8.01 and shot["start"] == 0 and shot["copy"] == context()["copy"]
    assert shot["three_d_motion_cues"]["settle"] > 5
    assert plans[0]["media_shots"][1] == footage
    assert chat.call_args.args[2:5] == ("configured endpoint", "configured model", "test key")
    assert chat.call_args.kwargs["enable_skills"] is False
    saved = json.loads((directory / "three_d_plan.json").read_text())
    assert saved["shots"][0]["planner_model"] == "actual backup model"
    assert saved["shots"][0]["planner_route"] == {
        "endpoint": "https://backup.example/v1", "model": "actual backup model"}
    assert saved["shots"][0]["requested_route"]["model"] == "configured model"
    assert "secret" not in json.dumps(saved) and "private API credential" not in json.dumps(saved)
    asyncio.run(td.render_story_shots(plans, board, directory, width=3840, height=2160, fps=30))
    assert chat.await_count == 1 and len(commands) == 1
    assert json.loads((directory / "three_d_plan.json").read_text())["shots"][0]["planner_model"] == "actual backup model"
    # Renderer source changes invalidate clip bytes without repeating a valid model plan.
    (renderer / "src/Scene.tsx").write_text('// updated R3F scene')
    asyncio.run(td.render_story_shots(plans, board, directory, width=3840, height=2160, fps=30))
    assert chat.await_count == 1 and len(commands) == 2
    plans[0]["visual_review_feedback"] = {"issues": ["Wrong object framing"]}
    asyncio.run(td.render_story_shots(plans, board, directory, width=3840, height=2160, fps=30))
    assert chat.await_count == 2
    # Identical feedback from a new rejected candidate still needs fresh geometry direction.
    plans[0]["review_repair_source_sha256"] = "later rejected candidate"
    asyncio.run(td.render_story_shots(plans, board, directory, width=3840, height=2160, fps=30))
    assert chat.await_count == 3
    # Changing configured selection invalidates the creative plan while valid identical clip bytes may be reused.
    provider.return_value = ("new configured endpoint", "new configured model", "new key")
    asyncio.run(td.render_story_shots(plans, board, directory, width=3840, height=2160, fps=30))
    assert chat.await_count == 4


@pytest.mark.parametrize("defect,error", [("primitive", "primitives are only permitted"),
                                         ("nonfinite", "numeric value"),
                                         ("thin", "numeric value"),
                                         ("long_intent", "semantic_intent length=603")])
def test_invalid_scene_repair_is_targeted_and_resume_preserves_valid_plans(tmp_path, monkeypatch, defect, error):
    _renderer(tmp_path)
    monkeypatch.setattr(td.config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("backend.pipeline.digester._resolve_provider",
                        AsyncMock(return_value=("endpoint", "configured model", "key")))
    plans, board = [], {"scenes": []}
    for index in range(2, 9):
        p, b = inputs()
        p[0]["id"] = b["scenes"][0]["id"] = f"scene-{index:02d}"
        plans.extend(p)
        board["scenes"].extend(b["scenes"])
    failed_id = "scene-08-shot-01"
    repair_requests = []

    def assert_original_rejected_spec(spec):
        actor = spec["actors"][0]
        if defect == "primitive":
            assert actor["primitive"] == "sphere"
        elif defect == "nonfinite":
            assert actor["scale"][0] == "NaN"
        elif defect == "thin":
            assert actor["scale"][2] == .08
        else:
            assert spec["semantic_intent"] == "x" * 603
            assert actor == raw_spec()["actors"][0]

    async def completion(_prompt, payload, *args, **kwargs):
        request = json.loads(payload)
        kwargs["route_selected"]("endpoint", "configured model", "key")
        assert "Omit primitive entirely" in request["actor_contract"]["primitive"]
        assert request["text_contract"]["maximum_characters"] == {
            "semantic_intent": 600, "narration_excerpt": 5000}
        if request["validation_feedback"]:
            repair_requests.append(request)
            assert [row["id"] for row in request["shots"]] == [failed_id]
            assert set(request["validation_feedback"]) == {failed_id}
            rejected = request["rejected_scenes"][0]
            assert rejected["id"] == failed_id
            assert_original_rejected_spec(rejected["spec"])
        results = []
        for row in request["shots"]:
            raw = {**raw_spec(), "id": row["id"]}
            if row["id"] == failed_id:
                if defect == "primitive":
                    raw["actors"][0].update(kind="data", primitive="sphere")
                elif defect == "nonfinite":
                    raw["actors"][0]["scale"] = [float("nan"), 1, 1]
                elif defect == "thin":
                    raw["actors"][0]["scale"] = [1, 1, .08]
                else:
                    raw["semantic_intent"] = "x" * 603
            results.append(raw)
        return json.dumps(results)

    chat = AsyncMock(side_effect=completion)
    monkeypatch.setattr("backend.pipeline.digester._chat", chat)
    command = AsyncMock()
    monkeypatch.setattr(td, "_command", command)
    directory = tmp_path / "task"
    with pytest.raises(RuntimeError, match=error):
        asyncio.run(td.render_story_shots(plans, board, directory))
    command.assert_not_awaited()
    assert len(repair_requests) == 1
    saved = json.loads((directory / "three_d_plan.json").read_text())
    assert len(saved["shots"]) == 6
    saved_hashes = {row["id"]: row["input_sha256"] for row in saved["shots"]}
    evidence = [json.loads(path.read_text()) for path in (directory / "three_d_planning").glob("*.json")]
    assert len(evidence) == 4
    assert sum(failed_id in row["validation_errors"] for row in evidence) == 2
    assert all(row["response"] and row["planner_route"]["model"] == "configured model" for row in evidence)
    latest = max((directory / "three_d_planning").glob("*.json"), key=lambda path: path.stat().st_mtime_ns)
    persisted = json.loads(latest.read_text())
    # One malformed model row must not hide the matching, still-invalid scene.
    persisted["response"] = json.dumps([{"id": []}, *json.loads(persisted["response"])])
    persisted["request"]["shots"].append({"id": []})
    if defect == "long_intent":
        # Old runs persisted a vague error; resume must recompute the exact
        # observed length and unchanged bound before asking the model to repair.
        persisted["validation_errors"][failed_id] = "3D semantic_intent must be concise nonempty text"
    latest.write_text(json.dumps(persisted))
    # Incomplete/corrupt diagnostic envelopes are ignored while older valid
    # matching evidence still seeds the repair request.
    for index, mutation in enumerate([
        {"request": []}, {"request": {"shots": None}},
        {"validation_errors": []}, {"response": []},
    ]):
        (directory / "three_d_planning" / f"malformed-{index}.json").write_text(
            json.dumps({**persisted, **mutation}))

    async def repair(_prompt, payload, *args, **kwargs):
        request = json.loads(payload)
        assert [row["id"] for row in request["shots"]] == [failed_id]
        assert set(request["validation_feedback"]) == {failed_id}
        assert error in request["validation_feedback"][failed_id]
        assert request["rejected_scenes"][0]["id"] == failed_id
        assert_original_rejected_spec(request["rejected_scenes"][0]["spec"])
        if defect == "long_intent":
            assert "at most 600 characters" in request["validation_feedback"][failed_id]
        kwargs["route_selected"]("endpoint", "configured model", "key")
        return json.dumps([{**raw_spec(), "id": failed_id}])

    async def render(_name, args, **kwargs):
        Path(args[args.index("--output") + 1]).write_bytes(b"real renderer boundary mocked")

    async def verify(output, spec, **kwargs):
        return {"passed": True, "sha256": td._file_hash(output),
                "duration": spec["duration_frames"] / spec["fps"], "frames": spec["duration_frames"]}

    chat.side_effect = repair
    command.side_effect = render
    monkeypatch.setattr(td, "validate_render", verify)
    report = asyncio.run(td.render_story_shots(plans, board, directory))
    assert report["passed"] and report["rendered_count"] == 7
    assert chat.await_count == 5  # Three batches, one invalid-scene retry, one resumed missing scene.
    resumed = json.loads((directory / "three_d_plan.json").read_text())
    reused = [row for row in resumed["shots"] if row["planner"] == "cache"]
    assert {row["id"]: row["input_sha256"] for row in reused} == saved_hashes


def _mock_successful_render_boundary(monkeypatch):
    async def render(_name, args, **kwargs):
        Path(args[args.index("--output") + 1]).write_bytes(b"mocked local render")

    async def verify(output, spec, **kwargs):
        return {"passed": True, "sha256": td._file_hash(output),
                "duration": spec["duration_frames"] / spec["fps"], "frames": spec["duration_frames"]}

    command = AsyncMock(side_effect=render)
    monkeypatch.setattr(td, "_command", command)
    monkeypatch.setattr(td, "validate_render", verify)
    return command


@pytest.mark.parametrize("changed_input", ["context", "route", "review_feedback", "candidate", "prompt"])
def test_resume_does_not_reuse_rejection_from_changed_input(tmp_path, monkeypatch, changed_input):
    _renderer(tmp_path)
    monkeypatch.setattr(td.config, "PROJECT_ROOT", tmp_path)
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    (prompt_dir / "three_d.txt").write_text("original scene direction")
    monkeypatch.setattr(td.config, "PROMPTS_DIR", prompt_dir)
    provider = AsyncMock(return_value=("endpoint", "model", "key"))
    monkeypatch.setattr("backend.pipeline.digester._resolve_provider", provider)
    invalid = raw_spec()
    invalid["actors"][0]["scale"] = [1, 1, .08]
    chat = AsyncMock(return_value=json.dumps([invalid]))
    monkeypatch.setattr("backend.pipeline.digester._chat", chat)
    command = _mock_successful_render_boundary(monkeypatch)
    plans, board = inputs()
    directory = tmp_path / "task"
    with pytest.raises(RuntimeError, match="scale"):
        asyncio.run(td.render_story_shots(plans, board, directory))
    command.assert_not_awaited()
    assert chat.await_count == 2

    if changed_input == "context":
        plans[0]["media_shots"][0]["copy"] = ["A different approved framing"]
    elif changed_input == "route":
        provider.return_value = ("another endpoint", "another model", "key")
    elif changed_input == "review_feedback":
        plans[0]["visual_review_feedback"] = {"issues": ["Keep the full report sheet visible"]}
    elif changed_input == "candidate":
        plans[0]["review_repair_source_sha256"] = "a different reviewed candidate"
    else:
        (prompt_dir / "three_d.txt").write_text("changed scene direction")

    async def fresh_plan(_prompt, payload, *args, **kwargs):
        request = json.loads(payload)
        assert request["validation_feedback"] == {}
        assert request["rejected_scenes"] == []
        kwargs["route_selected"](*provider.return_value)
        return json.dumps([raw_spec()])

    chat.side_effect = fresh_plan
    report = asyncio.run(td.render_story_shots(plans, board, directory))
    assert report["passed"] and report["rendered_count"] == 1
    assert chat.await_count == 3
    command.assert_awaited_once()


def test_repair_can_fix_quote_then_new_scale_error_within_three_attempts(tmp_path, monkeypatch):
    _renderer(tmp_path)
    monkeypatch.setattr(td.config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("backend.pipeline.digester._resolve_provider",
                        AsyncMock(return_value=("endpoint", "model", "key")))
    plans, board = inputs()
    source = "The proposed transaction's chips remain in service."
    board["scenes"][0]["text"] = source
    plans[0]["media_shots"][0]["script_excerpt"] = source
    command = _mock_successful_render_boundary(monkeypatch)
    requests = []

    async def completion(_prompt, payload, *args, **kwargs):
        request = json.loads(payload)
        requests.append(request)
        command.assert_not_awaited()
        kwargs["route_selected"]("endpoint", "model", "key")
        raw = raw_spec()
        raw["narration_excerpt"] = source
        if len(requests) == 1:
            raw["narration_excerpt"] = source.replace("'", "’")
        elif len(requests) == 2:
            assert "exact characters" in request["validation_feedback"][raw["id"]]
            assert request["rejected_scenes"][0]["spec"]["narration_excerpt"] != source
            raw["actors"][0]["scale"] = [1, 1, .08]
        else:
            assert len(requests) == 3
            assert "actors[0](chip).scale[2]=0.08" in request["validation_feedback"][raw["id"]]
            assert request["rejected_scenes"][0]["spec"]["actors"][0]["scale"][2] == .08
            assert "Keep valid fields unchanged" in request["repair_contract"]
        return json.dumps([raw])

    chat = AsyncMock(side_effect=completion)
    monkeypatch.setattr("backend.pipeline.digester._chat", chat)
    directory = tmp_path / "task"
    report = asyncio.run(td.render_story_shots(plans, board, directory))
    assert report["passed"] and report["rendered_count"] == 1
    assert chat.await_count == 3
    command.assert_awaited_once()
    saved = json.loads((directory / "three_d_plan.json").read_text())
    assert saved["shots"][0]["spec"]["narration_excerpt"] == source


@pytest.mark.parametrize("thicknesses,expected_calls", [([.08, .07, .06], 3), ([.08, .08], 2)])
def test_invalid_repair_attempts_are_bounded_and_never_render(tmp_path, monkeypatch, thicknesses, expected_calls):
    _renderer(tmp_path)
    monkeypatch.setattr(td.config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("backend.pipeline.digester._resolve_provider",
                        AsyncMock(return_value=("endpoint", "model", "key")))
    attempts = []

    async def invalid_plan(_prompt, payload, *args, **kwargs):
        request = json.loads(payload)
        attempts.append(request)
        assert len(attempts) <= expected_calls
        raw = raw_spec()
        raw["actors"][0]["scale"] = [1, 1, thicknesses[len(attempts) - 1]]
        return json.dumps([raw])

    chat = AsyncMock(side_effect=invalid_plan)
    monkeypatch.setattr("backend.pipeline.digester._chat", chat)
    command = AsyncMock()
    monkeypatch.setattr(td, "_command", command)
    plans, board = inputs()
    directory = tmp_path / "task"
    with pytest.raises(RuntimeError, match="Configured provider did not return valid 3D direction.*scale"):
        asyncio.run(td.render_story_shots(plans, board, directory))
    assert chat.await_count == expected_calls
    command.assert_not_awaited()
    assert plans[0]["media_shots"][0]["kind"] == "editorial"
    report = json.loads((directory / "three_d_render_report.json").read_text())
    assert report["status"] == "failed" and report["passed"] is False and report["rendered_count"] == 0
    evidence = [json.loads(path.read_text()) for path in (directory / "three_d_planning").glob("*.json")]
    assert len(evidence) == expected_calls
    if expected_calls == 2:
        assert any(row["unchanged_rejected_ids"] == [context()["id"]] for row in evidence)


def test_malformed_model_id_becomes_bounded_missing_scene_repair(tmp_path, monkeypatch):
    _renderer(tmp_path)
    monkeypatch.setattr(td.config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("backend.pipeline.digester._resolve_provider",
                        AsyncMock(return_value=("endpoint", "model", "key")))
    requests = []

    async def malformed_plan(_prompt, payload, *args, **kwargs):
        request = json.loads(payload)
        requests.append(request)
        if len(requests) > 1:
            assert "scene spec must be an object" in request["validation_feedback"][context()["id"]]
            assert request["rejected_scenes"][0]["spec"] is None
        return json.dumps([{**raw_spec(), "id": []}])

    chat = AsyncMock(side_effect=malformed_plan)
    monkeypatch.setattr("backend.pipeline.digester._chat", chat)
    command = AsyncMock()
    monkeypatch.setattr(td, "_command", command)
    plans, board = inputs()
    with pytest.raises(RuntimeError, match="Configured provider did not return valid 3D direction"):
        asyncio.run(td.render_story_shots(plans, board, tmp_path / "task"))
    assert chat.await_count == 2
    command.assert_not_awaited()


@pytest.mark.parametrize("failure", ["model", "render", "cancel"])
def test_3d_stage_fails_explicitly_and_never_claims_a_2d_fallback(tmp_path, monkeypatch, failure):
    _renderer(tmp_path)
    monkeypatch.setattr(td.config, "PROJECT_ROOT", tmp_path)
    provider = AsyncMock(return_value=("endpoint", "model", "key"))
    chat = AsyncMock(return_value="[]" if failure == "model" else json.dumps([raw_spec()]))
    monkeypatch.setattr("backend.pipeline.digester._resolve_provider", provider)
    monkeypatch.setattr("backend.pipeline.digester._chat", chat)
    error = asyncio.CancelledError() if failure == "cancel" else RuntimeError("local render failed")
    monkeypatch.setattr(td, "_command", AsyncMock(side_effect=error))
    plans, board = inputs()
    directory = tmp_path / "task"
    with pytest.raises(asyncio.CancelledError if failure == "cancel" else RuntimeError):
        asyncio.run(td.render_story_shots(plans, board, directory))
    report = json.loads((directory / "three_d_render_report.json").read_text())
    assert report["passed"] is False and report["rendered_count"] == 0
    assert report["status"] == ("interrupted" if failure == "cancel" else "failed")
    assert plans[0]["media_shots"][0]["kind"] == "editorial"


def test_no_eligible_interval_is_an_explicit_failure(tmp_path):
    plans, board = inputs()
    plans[0]["media_shots"] = [plans[0]["media_shots"][1]]
    with pytest.raises(RuntimeError, match="No eligible"):
        asyncio.run(td.render_story_shots(plans, board, tmp_path))
    assert json.loads((tmp_path / "three_d_render_report.json").read_text())["status"] == "no_eligible_shots"


@pytest.mark.parametrize("defect", [None, "audio", "frames", "dimensions", "spec_hash", "different_spec", "video_hash", "blank"])
def test_delivery_checks_actual_video_and_three_preview_pixels(tmp_path, monkeypatch, defect):
    spec = td.validate_spec(raw_spec(), context(), width=320, height=320, fps=30)
    output = tmp_path / "clip.mp4"
    output.write_bytes(b"encoded")
    td._write(tmp_path / "scene.json", spec)
    if defect == "different_spec":
        td._write(tmp_path / "scene.json", {**spec, "semantic_intent": "Different geometry requested"})
    stills = []
    for frame in [0, spec["duration_frames"] // 2, spec["duration_frames"] - 1]:
        image = Image.new("RGB", (320, 320), "navy")
        if defect != "blank":
            ImageDraw.Draw(image).rectangle((30, 30, 270, 270), fill="cyan")
        path = tmp_path / f"frame-{frame}.png"
        image.save(path)
        stills.append({"frame": frame, "path": str(path), "sha256": td._file_hash(path)})
    sidecar = {"contract": td.RENDER_CONTRACT, "passed": True, "stills": stills,
               "input_sha256": td._file_hash(tmp_path / "scene.json"), "output_sha256": td._file_hash(output)}
    if defect == "spec_hash": sidecar["input_sha256"] = "old spec"
    if defect == "video_hash": sidecar["output_sha256"] = "old bytes"
    td._write(Path(str(output) + ".render.json"), sidecar)
    stream = {"codec_type": "video", "width": 640 if defect == "dimensions" else 320, "height": 320,
              "avg_frame_rate": "30/1", "duration": str(spec["duration_frames"] / 30),
              "nb_read_frames": str(spec["duration_frames"] - (1 if defect == "frames" else 0))}
    streams = [stream, {"codec_type": "audio"}] if defect == "audio" else [stream]
    monkeypatch.setattr(td, "_command", AsyncMock(return_value=json.dumps({"streams": streams})))
    if defect:
        with pytest.raises(ValueError):
            asyncio.run(td.validate_render(output, spec))
    else:
        report = asyncio.run(td.validate_render(output, spec))
        assert report["passed"] and len(report["stills"]) == 3

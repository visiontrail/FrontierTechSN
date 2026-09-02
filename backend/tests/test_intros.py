import json
from pathlib import Path

import pytest

from backend.pipeline import assembler, composer, director, intros, outros, scene_kit


def _install_fake_library(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    library = tmp_path / "library"
    library.mkdir()
    monkeypatch.setattr(outros, "OUTRO_LIBRARY_DIR", library)
    monkeypatch.setattr(outros, "_verified_asset", lambda path, *_args: path)
    for preset in outros.OUTRO_PRESETS:
        (library / preset["filename"]).write_bytes(b"video")
    (library / outros.OUTRO_LOGO_FILENAME).write_bytes(b"logo")


def _storyboard() -> dict:
    return {
        "title": "ByteFront Espresso",
        "audio_duration": 12.0,
        "content_start": 0.0,
        "outro_start": 12.0,
        "total_duration": 12.0,
        "scene_count": 2,
        "scenes": [
            {
                "id": "scene-01",
                "index": 0,
                "start": 0.0,
                "duration": 6.0,
                "lines": [{"start": 0.0, "duration": 5.0, "text": "Opening"}],
                "text": "Opening",
                "program_segment_kind": "opening",
            },
            {
                "id": "scene-02",
                "index": 1,
                "start": 6.0,
                "duration": 6.0,
                "lines": [{"start": 6.0, "duration": 5.0, "text": "Story"}],
                "text": "Story",
                "program_segment_kind": "news",
            },
        ],
    }


def test_stage_intro_prepends_bookend_shifts_narration_and_writes_variables(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _install_fake_library(tmp_path, monkeypatch)
    board = _storyboard()

    plan = intros.stage_intro(
        tmp_path / "task",
        board,
        "morning-brief",
        edition_date="2026-09-02",
    )

    assert plan["archetype"] == "intro"
    assert plan["edition_weekday"] == "WEDNESDAY"
    assert plan["edition_date"] == "SEPTEMBER 02, 2026"
    assert set(board["intro_variables"]) == {"edition_date", "edition_weekday"}
    assert board["scenes"][0]["id"] == intros.INTRO_SCENE_ID
    assert board["scenes"][1]["start"] == 6.0
    assert board["scenes"][1]["lines"][0]["start"] == 6.0
    assert board["scenes"][2]["start"] == 12.0
    assert board["content_start"] == 6.0
    assert board["total_duration"] == 18.0
    assert board["edition_date"] == "2026-09-02"
    assert (tmp_path / "task/assets/intro/background.mp4").read_bytes() == b"video"
    assert (tmp_path / "task/assets/intro/bytefront-logo.png").read_bytes() == b"logo"
    assert json.loads(
        (tmp_path / "task" / intros.INTRO_VARIABLES_FILENAME).read_text()
    ) == board["intro_variables"]


def test_intro_scene_reads_declared_hyperframes_variables_and_passes_agent_gate():
    plan = scene_kit.ScenePlan(
        id="scene-intro",
        duration=6.0,
        archetype="intro",
        kicker="YOUR DAILY SHOT OF FRONTIER TECH",
        body="Fresh signals, served daily.",
        footage_src="assets/intro/background.mp4",
        footage_kind="video",
        intro_logo_src="assets/intro/bytefront-logo.png",
        intro_style="morning-brief",
        edition_date="SEPTEMBER 02, 2026",
        edition_weekday="WEDNESDAY",
    )

    html = scene_kit.render_scene(plan)

    assert director.validate_scene_html(
        html,
        "scene-intro",
        plan={
            "archetype": "intro",
            "footage_src": "assets/intro/background.mp4",
            "intro_logo_src": "assets/intro/bytefront-logo.png",
        },
    ) == []
    assert "window.__hyperframes.getVariables()" in html
    assert 'data-intro-field="date"' in html
    assert 'data-intro-field="weekday"' in html
    assert 'data-intro-field="label"' not in html
    assert 'data-intro-field="story-count"' not in html
    assert "DAILY TECH BRIEFING" not in html
    assert "intro-meta-chip" not in html
    assert "font:700 66px/1.2" in html
    assert "Date.now" not in html
    assert "muted playsinline" in html


def test_spine_declares_intro_variables_for_hyperframes_studio_and_render():
    board = _storyboard()
    values = intros.edition_variables(board, "2026-09-02")
    board["composition_variables"] = intros.composition_variable_specs(values)

    html = assembler.build_spine(board, audio_src="audio.wav", mounts=[])

    assert "data-composition-variables=" in html
    assert "edition_date" in html
    assert "SEPTEMBER 02, 2026" in html


def test_render_command_passes_strict_runtime_variable_file(tmp_path: Path):
    variables = tmp_path / intros.INTRO_VARIABLES_FILENAME
    variables.write_text('{"edition_date":"SEPTEMBER 02, 2026"}')

    command = composer._build_render_command(tmp_path, tmp_path / "video.mp4")

    assert command[-3:] == ["--variables-file", str(variables), "--strict-variables"]


def test_intro_agent_contract_rejects_static_or_incomplete_metadata():
    broken = '<div data-intro-role="brand"><span data-intro-field="date"></span></div>'

    problems = intros.intro_overlay_problems(broken)

    assert "missing editable intro edition overlay" in problems
    assert "intro does not read HyperFrames variables" in problems


def test_intro_agent_contract_rejects_removed_metadata_chips():
    broken = """
    <div data-intro-role="brand"></div>
    <div data-intro-role="edition">
      <span data-intro-field="date"></span>
      <span data-intro-field="weekday"></span>
      <span data-intro-field="label">DAILY TECH BRIEFING</span>
      <span data-intro-field="story-count">01 STORY</span>
    </div>
    <script>window.__hyperframes.getVariables()</script>
    """

    problems = intros.intro_overlay_problems(broken)

    assert "removed briefing-label field is still present" in problems
    assert "removed story-count field is still present" in problems
    assert "removed Daily Tech Briefing label is still present" in problems
    assert "removed story-count label is still present" in problems

from pathlib import Path

import pytest

from backend.pipeline import director, outros, scene_kit


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
        "total_duration": 42.5,
        "outro_duration": 0.0,
        "outro_start": 42.5,
        "scene_count": 1,
        "scenes": [
            {
                "id": "scene-01",
                "index": 0,
                "start": 0.0,
                "duration": 42.5,
                "lines": [],
                "text": "Narrated content",
                "program_segment_kind": "closing",
            }
        ],
    }


def test_outro_catalog_has_three_presets_and_morning_default():
    presets = outros.list_outro_presets()

    assert [preset["id"] for preset in presets] == [
        "data-extraction",
        "morning-brief",
        "signal-shot",
    ]
    assert next(preset for preset in presets if preset["is_default"])["label"] == (
        "2. 晨间简报｜温暖、编辑感"
    )


def test_stage_outro_replaces_spoken_closing_without_extending_storyboard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _install_fake_library(tmp_path, monkeypatch)
    board = _storyboard()

    plan = outros.stage_outro(tmp_path / "task", board, "signal-shot")

    assert plan["archetype"] == "outro"
    assert plan["outro_style"] == "signal-shot"
    assert plan["id"] == "scene-01"
    assert plan["duration"] == 42.5
    assert board["outro_start"] == 0.0
    assert board["outro_duration"] == 42.5
    assert board["total_duration"] == 42.5
    assert len(board["scenes"]) == 1
    assert board["scenes"][-1]["scene_kind"] == "outro"
    assert (tmp_path / "task/assets/outro/background.mp4").read_bytes() == b"video"
    assert (tmp_path / "task/assets/outro/bytefront-logo.png").read_bytes() == b"logo"


def test_stage_outro_rejects_storyboard_without_timed_scenes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _install_fake_library(tmp_path, monkeypatch)
    board = _storyboard()
    board["scenes"] = []

    with pytest.raises(ValueError, match="no timed scene"):
        outros.stage_outro(tmp_path / "task", board, "morning-brief")


def test_outro_scene_is_a_valid_editable_hyperframes_composition():
    plan = scene_kit.ScenePlan(
        id="scene-09",
        duration=6.0,
        archetype="outro",
        kicker="SEE YOU IN THE NEXT SHOT",
        headline="THANKS FOR WATCHING",
        body="Stay curious.",
        footage_src="assets/outro/background.mp4",
        footage_kind="video",
        outro_logo_src="assets/outro/bytefront-logo.png",
        outro_style="morning-brief",
    )

    html = scene_kit.render_scene(plan)

    assert director.validate_scene_html(
        html,
        "scene-09",
        plan={
            "archetype": "outro",
            "footage_src": "assets/outro/background.mp4",
            "outro_logo_src": "assets/outro/bytefront-logo.png",
        },
    ) == []
    assert "感谢观看" not in html
    assert 'data-outro-role="brand"' in html
    assert 'data-outro-brand-part="espresso"' in html
    assert "outro-badge" not in html
    assert "font:700 66px/1.2" in html
    assert 'data-outro-role="thanks"' in html
    assert html.count("data-outro-action=") == 3
    assert "muted playsinline" in html
    assert 'data-bookend-layer="background" style="z-index:0"' in html
    assert 'data-bookend-layer="overlay" style="z-index:2"' in html


def test_outro_agent_contract_rejects_removed_phrase_and_missing_actions():
    broken = '<div data-outro-role="brand" data-outro-role="thanks">感谢观看</div>'

    problems = outros.outro_overlay_problems(broken)

    assert "the removed Chinese closing phrase is still present" in problems
    assert "missing editable engagement-actions overlay" in problems
    assert "outro background stacking contract is missing" in problems

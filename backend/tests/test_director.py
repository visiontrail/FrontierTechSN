import asyncio

import pytest

from backend.pipeline import assembler, director, scene_kit


GOOD = """<template id="scene-01-template">
  <div id="scene-01" data-composition-id="scene-01" data-width="1920" data-height="1080">
    <style>#scene-01 { background:#0B0D17; }</style>
    <div id="scene-01-head">Hello</div>
    <script src="../vendor/gsap.min.js"></script>
    <script>
      window.__timelines = window.__timelines || {};
      (function () {
        const tl = gsap.timeline({ paused: true });
        const q = gsap.utils.selector("#scene-01");
        tl.fromTo(q("#scene-01-head"), { opacity: 0 }, { opacity: 1, duration: .5 }, 0.2);
        window.__timelines["scene-01"] = tl;
      })();
    </script>
  </div>
</template>
""" + "<!-- padding to clear the minimum length gate -->" * 8


def test_a_well_formed_scene_passes():
    assert validate(GOOD) == []


def validate(text: str) -> list[str]:
    return director.validate_scene_html(text, "scene-01")


@pytest.mark.parametrize(
    "removed, expected",
    [
        ("<template id=\"scene-01-template\">", "missing <template> wrapper"),
        ('data-composition-id="scene-01"', 'missing data-composition-id="scene-01"'),
        ('window.__timelines["scene-01"] = tl;', "missing window.__timelines"),
        ("../vendor/gsap.min.js", "does not load ../vendor/gsap.min.js"),
    ],
)
def test_missing_contract_pieces_are_caught(removed, expected):
    problems = validate(GOOD.replace(removed, ""))
    assert any(expected in p for p in problems), problems


@pytest.mark.parametrize(
    "injected",
    [
        "const x = Math.random();",
        "const t = Date.now();",
        "const p = performance.now();",
        "tl.to(el, { repeat: -1 });",
        '<script src="https://cdn.jsdelivr.net/npm/gsap@3.14.2/dist/gsap.min.js"></script>',
        "@import url('https://fonts.googleapis.com/css2?family=Inter');",
        "video.play();",
    ],
)
def test_render_breaking_patterns_are_rejected(injected):
    assert validate(GOOD.replace("<div id=\"scene-01-head\">Hello</div>", injected + "<div id=\"scene-01-head\">Hello</div>"))


def test_a_stub_file_is_rejected():
    assert validate("<template></template>")


def test_scene_root_declaring_its_own_timing_is_rejected():
    bad = GOOD.replace(
        'data-composition-id="scene-01" data-width',
        'data-composition-id="scene-01" data-start="0" data-width',
    )
    assert any("data-start" in p for p in validate(bad))


def test_the_deterministic_kit_always_passes_its_own_gate():
    for archetype in scene_kit.ARCHETYPES:
        plan = scene_kit.ScenePlan(
            id="scene-01",
            duration=10.0,
            archetype=archetype,
            headline="Headline",
            body="Body copy",
            items=("a", "b"),
            left_text="l",
            right_text="r",
            stat="42",
            quote="q",
            footage_src="footage/x.jpg",
        )
        assert director.validate_scene_html(scene_kit.render_scene(plan), "scene-01") == [], archetype


def test_shanshui_director_prompt_preserves_the_selected_style():
    prompt = director._system_prompt(scene_kit.THEMES["shanshui"])

    assert "background:#F7F0E4" in prompt
    assert "ink #263A30" in prompt
    assert "layered organic terrain" in prompt
    assert "Avoid neon" in prompt


def test_director_brief_uses_content_context_without_exposing_working_title():
    storyboard = {
        "title": "VIDEO 042",
        "thesis": "A content-specific thesis",
    }
    scene = {"id": "scene-01", "duration": 8.0, "text": "The actual opening idea."}
    prompt = director._batch_prompt(
        storyboard,
        [(scene, {"headline": "The actual opening idea"})],
        batch_no=1,
        batch_total=1,
        theme=scene_kit.DEFAULT_THEME,
    )

    assert "VIDEO 042" not in prompt
    assert "Overall thesis: A content-specific thesis" in prompt
    assert "narration spoken over this scene" in prompt


def test_revert_restores_the_deterministic_draft(tmp_path):
    plan = scene_kit.ScenePlan(id="scene-01", duration=8.0, headline="Kept")
    assembler.write_scene_files(tmp_path, [plan])
    target = tmp_path / "compositions" / "scene-01.html"
    target.write_text("garbage")
    assert director.revert_scenes(tmp_path, ["scene-01"], [plan]) == ["scene-01"]
    assert "Kept" in target.read_text()


def test_scenes_named_in_only_reports_error_lines():
    output = "\n".join(
        [
            "  ✗ [compositions/scene-03.html] missing_timeline_registry: ...",
            "  ⚠ [compositions/scene-07.html] studio_missing_editable_id: ...",
        ]
    )
    assert director.scenes_named_in(output, ["scene-03", "scene-07"]) == ["scene-03"]


def test_inspect_findings_map_to_scenes_by_id_and_by_timestamp():
    mounts = [
        {"id": "scene-04", "start": 30.0, "duration": 15.0},
        {"id": "scene-05", "start": 45.0, "duration": 20.0},
        {"id": "scene-06", "start": 65.0, "duration": 10.0},
    ]
    findings = [
        # No scene id in the text — only the timestamp links it to scene-04.
        '✗ t=37.36-44.03s content_overlap div.s-top inside div.s-mid "FU LU SHOU"',
        "✗ t=50.05-60.72s container_overflow div.stage inside #mount-scene-05",
    ]
    candidates = ["scene-04", "scene-05", "scene-06"]
    assert director.scenes_named_in_findings(findings, candidates, mounts) == [
        "scene-04",
        "scene-05",
    ]
    # Without the mount table only the explicitly-named scene can be blamed.
    assert director.scenes_named_in_findings(findings, candidates) == ["scene-05"]


def test_findings_only_blame_candidate_scenes():
    mounts = [{"id": "scene-04", "start": 30.0, "duration": 15.0}]
    findings = ["✗ t=37.0-38.0s content_overlap"]
    assert director.scenes_named_in_findings(findings, [], mounts) == []


def test_untouched_drafts_are_not_counted_as_authored(tmp_path, monkeypatch):
    """A crew that fails before writing must not be reported as a success."""
    plan = scene_kit.ScenePlan(id="scene-01", duration=8.0, headline="Draft")
    assembler.write_scene_files(tmp_path, [plan])

    async def fake_agent(*args, **kwargs):
        return (["scene-01"], "crew 1 failed: boom")

    monkeypatch.setattr(director, "_run_agent", fake_agent)
    monkeypatch.setattr(
        "backend.pipeline.agent.build_agent_env", lambda *a, **k: {}, raising=False
    )

    storyboard = {"title": "T", "scenes": [{"id": "scene-01", "text": "x", "duration": 8.0}]}
    outcome = asyncio.run(
        director.direct_scenes(tmp_path, storyboard, [{"id": "scene-01"}], [plan])
    )
    assert outcome.authored == []
    assert outcome.rejected == ["scene-01"]
    assert outcome.failures


def test_director_retries_selected_provider_until_a_later_attempt_succeeds(
    tmp_path, monkeypatch
):
    attempts = 0
    logs: list[str] = []

    async def fake_attempt(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        return (1, "temporary gateway timeout") if attempts < 3 else (2, None)

    monkeypatch.setattr(director, "_run_agent_attempt", fake_attempt)
    monkeypatch.setattr(director.config, "AI_MAX_RETRIES", 2)
    monkeypatch.setattr(director.config, "AI_RETRY_BASE_SECONDS", 0)

    ids, error = asyncio.run(
        director._run_agent(
            tmp_path,
            "prompt",
            ["scene-01"],
            batch_no=1,
            batch_total=1,
            model="model",
            env={},
            log=logs.append,
        )
    )

    assert ids == ["scene-01"]
    assert error is None
    assert attempts == 3
    assert any("attempt 3/3" in line for line in logs)
    assert all("yhroot" not in line for line in logs)

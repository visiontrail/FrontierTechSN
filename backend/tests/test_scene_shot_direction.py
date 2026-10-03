"""Rendering checks for the shared director shot contract."""

from html.parser import HTMLParser
import re

import pytest

from backend.pipeline import media_shots, scene_kit
from backend.pipeline.director import validate_scene_html
from backend.pipeline.video_format import LANDSCAPE, PORTRAIT


class VisibleCopy(HTMLParser):
    def __init__(self, source):
        super().__init__()
        self.hidden = False
        self.text = []
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        if tag in {"style", "script"}:
            self.hidden = True

    def handle_endtag(self, tag):
        if tag in {"style", "script"}:
            self.hidden = False

    def handle_data(self, data):
        if not self.hidden and data.strip():
            self.text.append(data.strip())


def shot(layout="flow", **overrides):
    value = {
        "kind": "editorial", "start": 0, "duration": 7,
        "copy": ["A grounded explanation", "Input from the source", "Output from the source"],
        "layout": layout, "transition": "wipe",
        "motion_cues": {"reveal_start": .1, "reveal_end": 2.4, "emphasis": 3.8, "settle": 5.2},
    }
    return {**value, **overrides}


def render(shots, frame=LANDSCAPE):
    plan = scene_kit.ScenePlan(
        id="scene-02", duration=sum(s["duration"] for s in shots),
        headline="Unselected scene headline", media_shots=tuple(shots), frame=frame,
        theme=scene_kit.THEMES["swiss"],
    )
    result = scene_kit.render_scene(plan)
    media_shots.assert_rendered_shots({"id": plan.id, "media_shots": shots}, result)
    assert validate_scene_html(result, plan.id, frame=frame) == []
    return result


@pytest.mark.parametrize("layout", ["flow", "comparison", "focus", "cards", "split"])
@pytest.mark.parametrize("frame", [LANDSCAPE, PORTRAIT])
def test_directed_layouts_preserve_factual_copy_and_runtime_contract(layout, frame):
    data = shot(layout)
    output = render([data], frame)
    visible = VisibleCopy(output).text
    for text in data["copy"]:
        assert visible.count(text) == 1
    assert "Unselected scene headline" not in visible
    assert "shot-orbit" not in output
    assert output == render([data], frame)
    assert 'data-cue-reveal-start="0.100"' in output
    assert 'data-cue-settle="5.200"' in output


def test_flow_links_connect_only_supplied_nodes_and_comparison_has_no_arrows():
    flow = render([shot()])
    elements = media_shots._Elements(flow).elements
    assert "scene-02-shot-1-link-1" in elements
    assert "scene-02-shot-1-link-2" not in elements
    comparison = render([shot("comparison")])
    assert not any("-link-" in name for name in media_shots._Elements(comparison).elements)
    assert 'class="shot-grid comparison"' in comparison
    assert 'id="scene-02-shot-1-emphasis-0"' in comparison
    assert 'id="scene-02-shot-1-emphasis-1"' in comparison


def test_one_copy_focus_does_not_duplicate_title_or_invent_support():
    output = render([shot(copy=["One supplied claim", "One supplied claim"])])
    visible = VisibleCopy(output).text
    assert visible == ["One supplied claim"]
    assert 'class="shot-grid focus"' in output
    assert not any("-card-" in name for name in media_shots._Elements(output).elements)


def test_cues_are_offset_once_and_all_authored_tweens_finish_before_settle():
    shots = [shot(), shot(start=7, layout="comparison")]
    output = render(shots)
    assert 'data-cue-reveal-start="7.100"' in output
    assert 'data-cue-settle="12.200"' in output
    for index, data in enumerate(shots, start=1):
        for line in output.splitlines():
            if f'inAt("#scene-02-shot-{index}-' not in line or "-wipe" in line:
                continue
            duration = float(re.search(r"duration:([0-9.]+)", line)[1])
            start = float(re.search(r", ([0-9.]+)\);$", line)[1])
            assert start >= data["start"]
            assert start + duration <= data["start"] + data["motion_cues"]["settle"] + .002


def test_short_title_reveal_does_not_collapse_flow_or_comparison_attention():
    # Exercise the actual program-generated schedule, whose title reveal is
    # shorter than the action window. Every process node must get its turn.
    cues = media_shots.motion_cues(7)
    flow = render([shot(copy=["Title", "Input", "Processing", "Output"], motion_cues=cues)])
    def positions(source, name):
        return [float(re.search(r", ([0-9.]+)\);$", line)[1])
                for line in source.splitlines() if re.search(rf'inAt\("#scene-02-shot-1-{name}-\d+"', line)]
    node_starts = positions(flow, "card")
    assert len(node_starts) == len(set(node_starts)) == 3
    assert node_starts[0] == cues["reveal_start"]
    assert cues["reveal_end"] < node_starts[-1] < cues["emphasis"]
    assert positions(flow, "link") == node_starts[1:]
    comparison = render([shot("comparison", motion_cues=cues)])
    emphasis = positions(comparison, "emphasis")
    assert emphasis[0] == cues["emphasis"] < emphasis[1] < cues["settle"]


def test_short_legacy_shot_has_bounded_cues_and_no_overrunning_reveal():
    data = shot(duration=.2, motion_cues=None, copy=["Brief"])
    cues = scene_kit._shot_cue_times(data)
    assert 0 <= cues["reveal_start"] < cues["reveal_end"] < cues["emphasis"] < cues["settle"] < .2
    render([data])


@pytest.mark.parametrize("cues", [
    {"reveal_start": .1, "reveal_end": float("nan"), "emphasis": 3, "settle": 5},
    {"reveal_start": .1, "reveal_end": 4, "emphasis": 3, "settle": 5},
    {"reveal_start": .1, "reveal_end": 2, "emphasis": 3, "settle": 8},
])
def test_renderer_refuses_invalid_cue_schedule(cues):
    with pytest.raises(ValueError, match="shot motion cues"):
        render([shot(motion_cues=cues)])


def test_directed_editorial_does_not_retime_or_replace_locked_video():
    source = shot(kind="public_footage", src="../footage/source.mp4", source_start=2,
                  source_duration=12, credit="Source credit")
    output = render([source, shot("comparison", start=7)])
    _, media = media_shots._Elements(output).elements["scene-02-shot-1-media"]
    assert media["src"] == "../footage/source.mp4"
    assert media["data-media-start"] == "2.00"
    assert media["data-duration"] == "7.00"
    assert "loop" not in media
    assert "muted" in media and "playsinline" in media


def test_portrait_flow_changes_both_connector_geometry_and_animation_axis():
    output = render([shot()], PORTRAIT)
    assert '.shot-grid.flow { flex-direction:column; }' in output
    assert 'inAt("#scene-02-shot-1-link-1", {scaleY:0}, {scaleY:1' in output
    assert 'inAt("#scene-02-shot-1-emphasis-0", {scaleY:0}, {scaleY:1' in output
    assert 'padding:130px 64px 300px' in output

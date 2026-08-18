import re

import pytest

from backend.pipeline import scene_kit as sk
from backend.pipeline.director import validate_scene_html
from backend.pipeline.video_format import PORTRAIT


def plan(**kwargs) -> sk.ScenePlan:
    base = {"id": "scene-01", "duration": 12.0, "headline": "A headline that reads well"}
    base.update(kwargs)
    return sk.ScenePlan(**base)


@pytest.mark.parametrize("archetype", ["title", "statement", "topic", "outro"])
def test_text_archetypes_satisfy_the_runtime_contract(archetype):
    html = sk.render_scene(plan(archetype=archetype, kicker="KICK", body="Some support copy."))
    assert validate_scene_html(html, "scene-01") == []


def test_outro_marks_its_intentional_exit_overflow():
    html = sk.render_scene(plan(archetype="outro"))

    assert '<div class="stage" data-layout-allow-overflow>' in html


def test_contrast_and_list_render_when_given_their_content():
    contrast = sk.render_scene(
        plan(archetype="contrast", left_label="Then", left_text="A", right_label="Now", right_text="B")
    )
    assert validate_scene_html(contrast, "scene-01") == []
    assert "Then" in contrast and "Now" in contrast

    listed = sk.render_scene(plan(archetype="list", items=("One", "Two", "Three")))
    assert validate_scene_html(listed, "scene-01") == []
    assert "Three" in listed


def test_archetypes_degrade_when_their_required_content_is_missing():
    # A contrast with only one side would render an empty panel; fall back instead.
    assert 'class="cols"' not in sk.render_scene(plan(archetype="contrast", left_text="only one"))
    assert 'class="rows"' not in sk.render_scene(plan(archetype="list", items=("just one",)))
    assert 'class="frame"' not in sk.render_scene(plan(archetype="footage"))


def test_scene_root_does_not_declare_its_own_timing():
    # The spine owns scene timing; a scene declaring data-start would double-schedule.
    html = sk.render_scene(plan(archetype="topic"))
    root = re.search(r"<div id=\"scene-01\"[^>]*>", html).group(0)
    assert "data-start" not in root
    assert "data-track-index" not in root
    assert 'data-width="1920"' in root and 'data-height="1080"' in root


def test_every_selector_is_scoped_to_the_scene():
    html = sk.render_scene(plan(archetype="topic", kicker="K", body="B"))
    style = re.search(r"<style>(.*?)</style>", html, re.S).group(1)
    selectors = [
        line.split("{")[0].strip()
        for line in style.splitlines()
        if "{" in line and not line.strip().startswith(("/*", "*"))
    ]
    assert selectors, "expected scoped rules"
    assert all(sel.startswith("#scene-01") for sel in selectors), selectors


def test_motifs_are_deterministic_and_seeded_per_scene():
    first = sk.render_motif("sunburst", "amber", "scene-01")
    assert first == sk.render_motif("sunburst", "amber", "scene-01")
    assert first != sk.render_motif("sunburst", "amber", "scene-02")
    assert sk.render_motif("none", "amber", "scene-01") == ""
    assert sk.render_motif("not-a-motif", "amber", "scene-01") == ""


@pytest.mark.parametrize("motif", [m for m in sk.MOTIFS if m != "none"])
def test_every_motif_produces_svg(motif):
    svg = sk.render_motif(motif, "teal", "seed")
    assert svg.startswith("<svg") and svg.endswith("</svg>")


def test_headline_size_shrinks_as_copy_grows():
    sizes = [sk.headline_size("x" * n) for n in (10, 30, 50, 80, 200)]
    assert sizes == sorted(sizes, reverse=True)
    assert sizes[-1] >= 56


def test_theme_selection_changes_surface_colours():
    dark = sk.render_scene(plan(theme=sk.THEMES["podcast"]))
    light = sk.render_scene(plan(theme=sk.THEMES["swiss"]))
    assert sk.THEMES["swiss"].bg in light
    assert sk.THEMES["swiss"].bg not in dark
    assert sk.resolve_theme("nope") is sk.DEFAULT_THEME


def test_shanshui_theme_uses_the_account_art_visual_language():
    theme = sk.resolve_theme("shanshui")
    html = sk.render_scene(
        plan(
            theme=theme,
            archetype="topic",
            kicker="CONNECTIONS",
            body="Ideas become paths through a changing landscape.",
            motif="arcs",
            accent="sky",
        )
    )

    assert theme is sk.THEMES["shanshui"]
    assert 'class="shanshui-backdrop"' in html
    assert "paper-grain" in html
    assert 'class="shanshui-terrain"' in html
    assert 'class="shanshui-routes"' in html
    assert 'class="shanshui-nodes"' in html
    assert "strokeDashoffset" in html
    assert sk.accent_hex("sky", theme) in html
    assert sk.ACCENTS["sky"] not in html
    assert validate_scene_html(html, "scene-01") == []


def test_shanshui_accents_stay_inside_the_muted_banner_palette():
    theme = sk.THEMES["shanshui"]
    assert sk.accent_hex("amber", theme) == "#B46F35"
    assert sk.accent_hex("teal", theme) == "#4E695A"
    assert sk.accent_hex("unknown", theme) == "#B46F35"


def test_copy_is_html_escaped():
    html = sk.render_scene(plan(headline='Rats & <script>alert("x")</script>'))
    assert "<script>alert" not in html
    assert "&lt;script&gt;" in html


def test_footage_scene_references_its_asset_relatively():
    html = sk.render_scene(
        plan(archetype="footage", footage_src="footage/paris.jpg", footage_credit="CC BY")
    )
    assert validate_scene_html(html, "scene-01") == []
    assert 'src="footage/paris.jpg"' in html
    assert "CC BY" in html


def test_video_footage_declares_hyperframes_media_timing():
    html = sk.render_scene(
        plan(
            archetype="footage",
            footage_src="footage/city.mp4",
            footage_kind="video",
            footage_credit="Review",
        )
    )
    assert validate_scene_html(html, "scene-01") == []
    video = re.search(r"<video[^>]*>", html).group(0)
    assert 'data-start="0"' in video
    assert 'data-duration="12.00"' in video
    assert 'data-track-index="0"' in video
    assert "muted" in video and "playsinline" in video
    assert " loop" in video


def test_collage_footage_is_clean_locked_off_full_bleed():
    html = sk.render_scene(
        plan(
            archetype="footage",
            footage_src="collage_broll/01/video/final-5s-noaudio.mp4",
            footage_kind="video",
            collage_broll=True,
        )
    )

    assert 'class="frame collage-frame"' in html
    assert 'class="scrim"' not in html
    assert 'class="stage"' not in html
    assert "scale: 1.16" not in html
    video = re.search(r"<video[^>]*>", html).group(0)
    assert " loop" in video


def test_portrait_scene_uses_portrait_root_contract():
    html = sk.render_scene(plan(frame=PORTRAIT, archetype="topic", body="Support"))

    assert 'data-width="1080"' in html
    assert 'data-height="1920"' in html
    assert "width:1080px; height:1920px" in html
    assert validate_scene_html(html, "scene-01", PORTRAIT) == []

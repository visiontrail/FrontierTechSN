import re

import pytest

from backend.pipeline import scene_kit as sk
from backend.pipeline.director import validate_scene_html
from backend.pipeline.video_format import PORTRAIT


def plan(**kwargs) -> sk.ScenePlan:
    base = {"id": "scene-01", "duration": 12.0, "headline": "A headline that reads well"}
    base.update(kwargs)
    return sk.ScenePlan(**base)


@pytest.mark.parametrize("archetype", ["title", "statement", "topic"])
def test_text_archetypes_satisfy_the_runtime_contract(archetype):
    html = sk.render_scene(plan(archetype=archetype, kicker="KICK", body="Some support copy."))
    assert validate_scene_html(html, "scene-01") == []

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


def test_footage_preserves_both_sides_of_the_planned_comparison():
    html = sk.render_scene(plan(
        archetype="footage", footage_src="verified.mp4", footage_kind="video",
        left_label="ONE YEAR AGO", left_text="Devin wrote 13%",
        right_label="TODAY", right_text="More than 90%",
        body="Claim by an investor backing Cognition.",
    ))
    assert validate_scene_html(html, "scene-01") == []
    assert 'class="footage-comparison"' in html
    assert "Devin wrote 13%" in html and "More than 90%" in html
    assert "Claim by an investor backing Cognition." in html
    assert "verified.mp4" in html


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


def test_inline_news_image_is_part_of_the_text_flow_with_seekable_motion():
    html = sk.render_scene(
        plan(
            archetype="topic",
            kicker="NVIDIA VERA",
            body="The CPU is now in production and the body remains visible.",
            items=(
                "MIT bacteria act as <circuit> boards & living computers",
                'Perfect World reports "H1" revenue and a net loss',
            ),
            news_image_src="../news_images/nvidia.png",
            news_image_mode="inline",
            news_image_kind="logo",
            news_image_fit="contain",
            news_image_credit="Image: NVIDIA · Public domain",
            news_image_caption="NVIDIA logo",
        )
    )

    assert validate_scene_html(html, "scene-01") == []
    assert 'class="stage news-inline-stage"' in html
    assert '<div class="news-inline-window" id="scene-01-image-frame">' in html
    assert '<div class="news-inline-media-crop" data-layout-allow-overflow>' in html
    assert 'src="../news_images/nvidia.png"' in html
    assert 'id="scene-01-body"' in html
    assert 'class="news-inline-items"' in html
    assert 'class="news-inline-index"' not in html
    assert '>IMG</div>' not in html
    assert "MIT bacteria act as &lt;circuit&gt; boards &amp; living computers" in html
    assert "Perfect World reports &quot;H1&quot; revenue and a net loss" in html
    assert "<circuit>" not in html
    assert '#scene-01-items .news-inline-item' in html
    assert "stagger: .1" in html
    assert "rotationY" in html and "transformPerspective" in html
    assert "duration: 11.65" in html
    assert "gsap.to(" not in html


def test_inline_news_image_renders_stat_value_and_label():
    html = sk.render_scene(
        plan(
            archetype="stat",
            kicker="GAMING / AI",
            headline="Perfect World's first-half results",
            stat="¥2.751B revenue · ¥118M loss",
            stat_label="First-half 2026, per semiannual report filed Aug 19",
            news_image_src="../news_images/perfect-world.png",
            news_image_mode="inline",
            news_image_kind="logo",
            news_image_fit="contain",
        )
    )

    assert validate_scene_html(html, "scene-01") == []
    assert 'class="news-inline-stat" id="scene-01-stat"' in html
    assert 'class="news-inline-stat-value" id="scene-01-stat-value"' in html
    assert "¥2.751B revenue · ¥118M loss" in html
    assert 'class="news-inline-stat-label" id="scene-01-stat-label"' in html
    assert "First-half 2026, per semiannual report filed Aug 19" in html
    assert 'inAt("#scene-01-stat"' in html


def test_inline_news_image_uses_compact_layout_and_keeps_fifth_item():
    items = [
        "Summarize annual reports across multiple documents",
        "Search and compare company operational data online",
        "Control a desktop browser for retrieval and document generation",
        "Build English-language presentations from data",
        "Generate marketing posters from reference images",
        "A sixth unsupported task",
    ]
    scene_plan = sk.ScenePlan.from_dict(
        {
            "archetype": "list",
            "kicker": "AI AGENTS",
            "headline": "Five real office tasks put to the test",
            "items": items,
            "news_image_src": "../news_images/annual-reports.jpg",
            "news_image_mode": "inline",
            "news_image_kind": "event",
            "news_image_fit": "cover",
        },
        duration=12.0,
        scene_id="scene-11",
    )

    html = sk.render_scene(scene_plan)

    assert validate_scene_html(html, "scene-11") == []
    assert 'class="news-inline-copy news-inline-copy-five"' in html
    assert 'class="news-inline-items news-inline-items-five"' in html
    assert 'id="scene-11-item-5"' in html
    assert "Generate marketing posters from reference images" in html
    assert "A sixth unsupported task" not in html
    assert ".news-inline-items-five { gap:8px; }" in html
    assert "gap:10px; padding:9px 12px;" in html


def test_fullscreen_news_image_uses_a_masked_reveal_and_separate_ken_burns_layer():
    html = sk.render_scene(
        plan(
            archetype="news_image",
            kicker="LAUNCH",
            body="Falcon 9 cleared the tower.",
            news_image_src="../news_images/launch.jpg",
            news_image_mode="fullscreen",
            news_image_kind="event",
            news_image_fit="cover",
            news_image_credit="Image: NASA · Public domain",
            news_image_caption="Falcon 9 launch",
        )
    )

    assert validate_scene_html(html, "scene-01") == []
    assert 'class="news-full-frame"' in html
    assert "clipPath" in html
    assert f'#{"scene-01"}-image-frame' in html
    assert f'#{"scene-01"}-image' in html
    assert "scale: 1.025" in html


@pytest.mark.parametrize("mode", ["inline", "fullscreen"])
def test_contained_photo_is_not_zoomed_or_panned_out_of_its_frame(mode):
    html = sk.render_scene(plan(
        news_image_src="portrait.jpg", news_image_mode=mode,
        news_image_kind="person", news_image_fit="contain",
    ))
    tween = next(line for line in html.splitlines() if 'inAt("#scene-01-image",' in line)
    assert "scale: 1, x: 0" in tween
    assert "scale: 1.0" not in tween
    assert "object-fit:contain" in html


def test_collage_preserves_each_images_fit_without_zooming_portrait():
    html = sk.render_scene(plan(
        news_image_src="portrait.jpg", news_image_mode="fullscreen",
        news_image_srcs=("portrait.jpg", "landscape.jpg", "logo.png"),
        news_image_fits=("contain", "cover", "contain"),
    ))
    for index, expected in [(1, "contain"), (2, "cover"), (3, "contain")]:
        tag = re.search(rf'<img[^>]+id="scene-01-collage-image-{index}"[^>]*>', html).group()
        assert f"object-fit:{expected}" in tag
        tween = next(line for line in html.splitlines() if f'inAt("#scene-01-collage-image-{index}"' in line)
        assert ("scale: 1, x: 0" in tween) == (expected == "contain")


@pytest.mark.parametrize("theme", list(sk.THEMES.values()))
def test_fullscreen_news_image_text_contrasts_with_its_theme_scrim(theme):
    html = sk.render_scene(plan(
        archetype="news_image", news_image_src="logo.png", news_image_mode="fullscreen",
        theme=theme, body="The named source reports the supporting facts.",
    ))
    for cls in ("news-full-headline", "news-full-body"):
        rules = re.search(rf"\.{cls} \{{(.*?)\}}", html, re.S).group(1)
        assert f"color:{theme.ink}" in rules
        assert sk._rgba(theme.bg, .9) in rules


def test_fullscreen_news_image_can_be_a_copy_free_three_image_collage():
    html = sk.render_scene(
        plan(
            archetype="news_image",
            kicker="THIS COPY MUST NOT RENDER",
            body="Nor should this narration copy.",
            news_image_src="../news_images/hero.jpg",
            news_image_srcs=(
                "../news_images/hero.jpg",
                "../news_images/support-1.jpg",
                "../news_images/support-2.jpg",
            ),
            news_image_mode="fullscreen",
            news_image_credits=("Source one", "Source two", "Source three"),
        )
    )

    assert validate_scene_html(html, "scene-01") == []
    assert 'class="news-collage-canvas"' in html
    assert html.count('class="news-collage-panel') == 3
    assert 'class="news-collage-glow"' in html
    assert 'data-layout-allow-overflow' in html
    assert 'src="../news_images/hero.jpg"' in html
    assert 'src="../news_images/support-1.jpg"' in html
    assert 'src="../news_images/support-2.jpg"' in html
    assert 'width:800px; height:400px; z-index:6;' in html
    assert 'top:338px; width:1080px; height:632px; z-index:4;' in html
    assert "THIS COPY MUST NOT RENDER" not in html
    assert "Nor should this narration copy" not in html
    assert "news-full-headline" not in html
    assert "power4.out" in html and "expo.out" in html and "circ.out" in html
    assert " }}, " not in html


def test_english_news_webpage_capture_overlays_image_background():
    html = sk.render_scene(
        plan(
            archetype="news_image",
            news_image_src="../news_images/chip-factory.jpg",
            news_image_mode="fullscreen",
            news_webpage_src="../news_webpages/page-01.png",
            news_webpage_url="https://example.com/english-story",
            news_webpage_source="Example News",
            news_webpage_headline="A chip factory breaks ground",
        )
    )

    assert validate_scene_html(html, "scene-01") == []
    assert 'class="news-web-card"' in html
    assert 'src="../news_images/chip-factory.jpg"' in html
    assert 'src="../news_webpages/page-01.png"' in html
    assert "English news · Example News" in html
    assert "news-full-headline" not in html
    assert "rotationY" in html and "transformPerspective" in html


def test_english_news_webpage_capture_overlays_public_video():
    html = sk.render_scene(
        plan(
            archetype="footage",
            footage_src="../footage/clip-01.mp4",
            footage_kind="video",
            news_webpage_src="../news_webpages/page-01.png",
            news_webpage_source="IEEE Spectrum",
            news_webpage_headline="Simulation software brings monsters to life",
        )
    )

    assert validate_scene_html(html, "scene-01") == []
    video = re.search(r"<video[^>]*>", html).group(0)
    assert 'src="../footage/clip-01.mp4"' in video
    assert 'data-duration="5.00"' in video
    assert "muted" in video and "playsinline" in video and " loop" not in video
    assert 'src="../news_webpages/page-01.png"' in html


def test_news_webpage_overlay_preserves_ai_public_footage_sequence():
    html = sk.render_scene(
        plan(
            archetype="footage",
            footage_src="../footage/clip-01.mp4",
            footage_kind="video",
            footage_sequence=(
                {"src": "../footage/clip-01.mp4", "duration_seconds": 4.0},
                {"src": "../footage/clip-02.mp4", "duration_seconds": 3.0},
            ),
            news_webpage_src="../news_webpages/page-01.png",
            news_webpage_source="IEEE Spectrum",
        )
    )

    videos = re.findall(r"<video[^>]*>", html)
    assert len(videos) == 2
    assert 'data-start="0.00"' in videos[0]
    assert 'data-duration="4.00"' in videos[0]
    assert 'data-start="4.00"' in videos[1]
    assert 'data-duration="3.00"' in videos[1]
    assert all(" loop" not in video for video in videos)


@pytest.mark.parametrize("webpage", [False, True])
def test_media_scenes_preserve_independent_body_and_structured_facts(webpage):
    html = sk.render_scene(plan(
        archetype="footage", footage_src="clip.mp4", footage_kind="video",
        headline="NASA food safety", body="Paul Lachance developed HACCP.",
        items=("1993: 700+ illnesses & four deaths", "1996: FSIS mandate"),
        news_webpage_src="page.png" if webpage else "",
    ))
    assert validate_scene_html(html, "scene-01") == []
    assert "Paul Lachance developed HACCP." in html
    assert "<li>1993: 700+ illnesses &amp; four deaths</li>" in html
    assert "<li>1996: FSIS mandate</li>" in html
    if webpage:
        assert 'src="page.png"' in html
        assert 'class="news-web-editorial"' in html


def test_media_scene_does_not_repeat_a_body_flattened_from_its_items():
    html = sk.render_scene(plan(
        archetype="footage", footage_src="clip.mp4", footage_kind="video",
        body="First finding · Second finding", items=("First finding", "Second finding"),
    ))
    assert html.count("First finding") == 1
    assert html.count("Second finding") == 1


def test_fullscreen_news_image_uses_escaped_quote_when_body_is_empty():
    html = sk.render_scene(
        plan(
            archetype="news_image",
            body="",
            quote='Promising ideas come from people without a <formal> role & "authority".',
            news_image_src="../news_images/university.png",
            news_image_mode="fullscreen",
            news_image_kind="logo",
            news_image_fit="contain",
            news_image_caption="University logo",
        )
    )

    assert validate_scene_html(html, "scene-01") == []
    assert 'class="body news-full-body news-full-quote-support"' in html
    assert (
        "Promising ideas come from people without a &lt;formal&gt; role &amp; "
        "&quot;authority&quot;." in html
    )
    assert "<formal>" not in html
    assert 'inAt("#scene-01-body"' in html


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
    assert 'data-start="0.00"' in video
    assert 'data-duration="5.00"' in video
    assert 'data-track-index="0"' in video
    assert "muted" in video and "playsinline" in video
    assert " loop" not in video
    assert "public-footage-fallback" in html


def test_public_footage_sequence_plays_each_clip_once_then_reveals_hyperframe():
    html = sk.render_scene(
        plan(
            archetype="footage",
            footage_src="footage/first.mp4",
            footage_kind="video",
            footage_sequence=(
                {"src": "footage/first.mp4", "duration_seconds": 4.0},
                {"src": "footage/second.mp4", "duration_seconds": 3.5},
            ),
        )
    )

    videos = re.findall(r"<video[^>]*>", html)
    assert len(videos) == 2
    assert 'data-start="0.00"' in videos[0]
    assert 'data-duration="4.00"' in videos[0]
    assert 'data-start="4.00"' in videos[1]
    assert 'data-duration="3.50"' in videos[1]
    assert all(" loop" not in video for video in videos)
    assert "public-footage-fallback" in html


def test_public_footage_credits_follow_each_clip_and_end_before_fallback():
    html = sk.render_scene(
        plan(
            archetype="footage",
            duration=12.0,
            footage_src="footage/second.mp4",
            footage_kind="video",
            footage_credit="Last clip credit must not label the whole scene",
            footage_sequence=(
                {"src": "footage/first.mp4", "duration_seconds": 4.015, "credit": "Archive & Museum"},
                {"src": "footage/second.mp4", "duration_seconds": 3.5, "credit": "Port authority"},
            ),
        )
    )
    assert validate_scene_html(html, "scene-01") == []
    videos = re.findall(r"<video[^>]*>", html)
    credits = re.findall(r'<div[^>]*class="clip credit"[^>]*>[^<]*</div>', html)
    assert len(credits) == len(videos) == 2
    for video, credit in zip(videos, credits):
        for field in ("data-start", "data-duration"):
            assert re.search(fr'{field}="([^"]*)"', credit).group(1) == re.search(fr'{field}="([^"]*)"', video).group(1)
        assert 'data-track-index="1"' in credit
    assert "Archive &amp; Museum" in credits[0]
    assert "Port authority" in credits[1]
    assert "Last clip credit" not in html


@pytest.mark.parametrize("theme", [sk.THEMES["swiss"], sk.THEMES["podcast"]])
def test_footage_text_uses_a_dark_backing_even_in_light_themes(theme):
    html = sk.render_scene(plan(archetype="footage", footage_src="image.jpg", theme=theme, body="Supporting facts"))
    scrim = re.search(r"\.scrim \{(.*?)\}", html, re.S).group(1)
    body_rules = re.findall(r"\.body \{(.*?)\}", html, re.S)
    assert "rgba(17,19,24,.94)" in scrim
    assert "color:rgba(245,242,234,.9)" in body_rules[-1]
    assert 'inAt("#scene-01-media"' in html


def test_collage_footage_keeps_media_contract_and_adds_seekable_lower_third():
    html = sk.render_scene(
        plan(
            archetype="footage",
            kicker="SUPPLY & <CHAIN>",
            headline="China restricts germanium and quartz exports to Taiwan",
            footage_src="collage_broll/01/video/final-8s-noaudio.mp4",
            footage_kind="video",
            collage_broll=True,
            collage_hold_src="collage_broll/01/frames/last-frame.jpg",
            collage_target_duration_seconds=8.0,
        )
    )

    assert 'class="frame collage-frame"' in html
    assert 'class="scrim"' not in html
    assert 'class="stage"' not in html
    assert 'class="collage-caption"' in html
    assert "SUPPLY &amp; &lt;CHAIN&gt;" in html
    assert "<CHAIN>" not in html
    assert "China restricts germanium and quartz exports to Taiwan" in html
    assert "background:rgba(11,13,23,.90)" in html
    assert 'inAt("#scene-01-collage-caption"' in html
    assert 'inAt("#scene-01-collage-headline"' in html
    assert "gsap.timeline({ paused: true })" in html
    assert "tl.fromTo" in html
    assert "scale: 1.16" not in html
    video = re.search(r"<video[^>]*>", html).group(0)
    assert " loop" not in video
    assert "muted" in video and "playsinline" in video
    assert 'data-duration="8.00"' in video
    hold = re.search(r'<img id="scene-01-hold"[^>]*>', html).group(0)
    assert 'data-start="8.00"' in hold
    assert 'data-duration="4.00"' in hold
    assert 'data-track-index="0"' in hold
    assert '.frame > .clip { position:absolute; inset:0; }' in html


def test_portrait_scene_uses_portrait_root_contract():
    html = sk.render_scene(plan(frame=PORTRAIT, archetype="topic", body="Support"))

    assert 'data-width="1080"' in html
    assert 'data-height="1920"' in html
    assert "width:1080px; height:1920px" in html
    assert validate_scene_html(html, "scene-01", PORTRAIT) == []


@pytest.mark.parametrize("frame", [sk.LANDSCAPE, PORTRAIT])
def test_collage_preserves_reviewed_facts_and_attribution_in_both_formats(frame):
    html = sk.render_scene(plan(
        frame=frame, duration=40, archetype="footage", collage_broll=True,
        footage_src="verified.mp4", footage_kind="video",
        collage_hold_src="verified-hold.jpg", collage_target_duration_seconds=8,
        body="MIT's Codex agent calibrated six qubits.",
        items=("Measured transition frequencies", "Noise still requires human guidance"),
        quote="Safety & <timing> matter", attribution="Named source",
        stat="6", stat_label="Qubits", left_text="Before", right_text="After",
    ))
    assert validate_scene_html(html, "scene-01", frame) == []
    details = re.search(r'<aside class="collage-details"[^>]*>(.*?)</aside>', html, re.S).group(1)
    for expected in (
        "MIT&#x27;s Codex agent calibrated six qubits.", "Measured transition frequencies",
        "Noise still requires human guidance", "Safety &amp; &lt;timing&gt; matter",
        "Named source", "Qubits", "Before", "After",
    ):
        assert expected in details
    assert 'data-duration="8.00"' in html
    assert 'data-start="8.00" data-duration="32.00"' in html
    assert 'inAt("#scene-01-collage-details"' in html
    assert 'ease: "power3.out" }, 8.00)' in html


def test_short_collage_reveals_facts_before_midpoint_without_duplicate_list():
    html = sk.render_scene(plan(
        archetype="footage", collage_broll=True, footage_src="verified.mp4",
        footage_kind="video", collage_target_duration_seconds=8,
        body="One · Two", items=("One", "Two"),
    ))
    assert 'id="scene-01-body"' not in html
    assert '<li>One</li><li>Two</li>' in html
    assert 'ease: "power3.out" }, 3.60)' in html


@pytest.mark.parametrize("webpage", [False, True])
def test_fractional_footage_intervals_do_not_overlap_or_exceed_their_sources(webpage):
    from decimal import Decimal

    durations = [33.867, 15.017, 15.0, 15.0]
    html = sk.render_scene(plan(
        archetype="footage", duration=72.669, footage_src="one.mp4", footage_kind="video",
        footage_sequence=tuple({"src": f"{i}.mp4", "duration_seconds": d} for i, d in enumerate(durations)),
        news_webpage_src="page.png" if webpage else "",
    ))
    previous_end = Decimal(0)
    videos = re.findall(r"<video[^>]*>", html)
    assert len(videos) == 4
    for video, source_duration in zip(videos, durations):
        start = Decimal(re.search(r'data-start="([^"]+)"', video).group(1))
        duration = Decimal(re.search(r'data-duration="([^"]+)"', video).group(1))
        assert start == previous_end
        assert duration <= Decimal(str(source_duration))
        previous_end = start + duration
    assert previous_end <= Decimal("72.669")

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

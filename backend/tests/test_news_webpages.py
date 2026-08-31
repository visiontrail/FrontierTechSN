import hashlib
import io

from PIL import Image

from backend.pipeline import news_webpages


def _board():
    return {
        "scenes": [
            {
                "id": "scene-01",
                "text": "Good morning and welcome to Frontier Tech Daily.",
            },
            {
                "id": "scene-02",
                "text": (
                    "SK Hynix has broken ground on its advanced memory chip packaging "
                    "facility in Indiana, according to Bloomberg."
                ),
            },
            {
                "id": "scene-03",
                "text": "A Chinese laboratory announced a new robotics system.",
            },
        ]
    }


def test_overlay_assignments_require_english_url_and_existing_visual_scene():
    dossier = {
        "selected": [
            {
                "language": "en",
                "source_name": "Techmeme",
                "title": "SK Hynix breaks ground on an advanced memory chip plant",
                "summary": "The Indiana packaging facility will make HBM memory.",
                "url": "https://www.techmeme.com/260829/p5",
            },
            {
                "language": "zh",
                "source_name": "中文科技",
                "title": "机器人实验室发布新系统",
                "summary": "Chinese laboratory robotics system",
                "url": "https://example.cn/story",
            },
            {
                "language": "en",
                "source_name": "Broken",
                "title": "No usable scheme",
                "url": "javascript:alert(1)",
            },
        ]
    }
    plans = [
        {"id": "scene-01", "archetype": "topic"},
        {
            "id": "scene-02",
            "archetype": "footage",
            "footage_src": "../footage/clip-01.mp4",
        },
        {
            "id": "scene-03",
            "archetype": "news_image",
            "news_image_src": "../news_images/robot.jpg",
        },
    ]

    assignments = news_webpages.select_overlay_assignments(
        dossier, _board(), plans, limit=2
    )

    assert assignments == [
        {
            "scene_id": "scene-02",
            "source_url": "https://www.techmeme.com/260829/p5",
            "source_name": "Techmeme",
            "expected_headline": "SK Hynix breaks ground on an advanced memory chip plant",
            "dossier_language": "en",
        }
    ]


def test_page_language_gate_requires_headline_article_copy_and_english_words():
    assert news_webpages._page_is_english(
        {
            "document_language": "en-US",
            "headline": "A real English headline",
            "paragraph_characters": 420,
            "english_word_count": 80,
            "cjk_character_count": 0,
            "headline_match_words": 4,
        }
    )
    assert not news_webpages._page_is_english(
        {
            "document_language": "zh-CN",
            "headline": "English-looking navigation label",
            "paragraph_characters": 420,
            "english_word_count": 18,
            "cjk_character_count": 160,
            "headline_match_words": 1,
        }
    )


def test_page_language_gate_allows_visible_title_revision_grounded_by_metadata():
    assert news_webpages._page_is_english(
        {
            "document_language": "en",
            "headline": "Inside the Physics Engine Making Kong's Muscles Feel So Real",
            "headline_match_words": 1,
            "metadata_match_words": 6,
            "paragraph_characters": 680,
            "english_word_count": 110,
            "cjk_character_count": 0,
        }
    )
    assert not news_webpages._page_is_english(
        {
            "document_language": "en",
            "headline": "Headline but no story body",
            "paragraph_characters": 0,
            "english_word_count": 55,
            "cjk_character_count": 0,
            "headline_match_words": 3,
        }
    )


def test_focused_screenshot_makes_story_block_legible_on_fixed_canvas(tmp_path):
    source = Image.new("RGB", (1440, 900), "#eeeeee")
    source.paste("#123456", (100, 120, 500, 320))
    payload = io.BytesIO()
    source.save(payload, format="PNG")
    destination = tmp_path / "focused.png"

    result = news_webpages._focused_screenshot(
        payload.getvalue(),
        {"x": 100, "y": 120, "width": 400, "height": 200},
        destination,
    )

    with Image.open(destination) as rendered:
        assert rendered.size == (1440, 900)
        assert rendered.getpixel((720, 450)) == (18, 52, 86)
        assert rendered.getpixel((10, 10)) == (255, 255, 255)
    assert result["focus_width"] == 400
    assert result["focused_render_width"] == 1320


def test_attach_news_webpage_preserves_underlying_public_footage(tmp_path):
    root = tmp_path / "news_webpages"
    root.mkdir()
    image_path = root / "page-01.png"
    Image.new("RGB", (1440, 900), "white").save(image_path)
    digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
    manifest = {
        "status": "ready",
        "pages": [
            {
                "scene_id": "scene-02",
                "source_url": "https://example.com/story",
                "source_name": "Example News",
                "dossier_language": "en",
                "captured_headline": "A real English headline",
                "local_path": "news_webpages/page-01.png",
                "bytes": image_path.stat().st_size,
                "sha256": digest,
            }
        ],
    }
    plans = [
        {
            "id": "scene-02",
            "archetype": "footage",
            "footage_src": "../footage/clip-01.mp4",
            "footage_kind": "video",
        }
    ]

    assert news_webpages.attach_news_webpages(plans, manifest, tmp_path) == 1
    assert plans[0]["footage_src"] == "../footage/clip-01.mp4"
    assert plans[0]["news_webpage_src"] == "../news_webpages/page-01.png"
    assert plans[0]["news_webpage_language"] == "en"


def test_attach_news_webpage_rejects_collage_background(tmp_path):
    plans = [
        {
            "id": "scene-01",
            "archetype": "footage",
            "footage_src": "../collage_broll/opening.mp4",
            "collage_broll": True,
        }
    ]
    assert news_webpages.attach_news_webpages(
        plans, {"status": "ready", "pages": []}, tmp_path
    ) == 0


def test_overlay_assignment_preserves_copy_free_three_image_collage():
    dossier = {
        "selected": [
            {
                "language": "en",
                "source_name": "Example News",
                "title": "Chip factory expansion reaches a new milestone",
                "summary": "The chip factory expanded production capacity.",
                "url": "https://example.com/chip-factory",
            }
        ]
    }
    storyboard = {
        "scenes": [
            {
                "id": "scene-02",
                "text": "The chip factory expansion reached a new production milestone.",
            }
        ]
    }
    plans = [
        {
            "id": "scene-02",
            "archetype": "news_image",
            "news_image_src": "../news_images/hero.jpg",
            "news_image_collage": True,
            "news_image_srcs": [
                "../news_images/hero.jpg",
                "../news_images/support-one.jpg",
                "../news_images/support-two.jpg",
            ],
        }
    ]

    assert news_webpages.select_overlay_assignments(dossier, storyboard, plans) == []

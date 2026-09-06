import hashlib
import io
import json

import pytest

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


def test_page_language_gate_accepts_strong_english_page_without_lang_attribute():
    assert news_webpages._page_is_english(
        {
            "document_language": "",
            "headline": "Nvidia plans to bring DLSS 5 to older RTX graphics cards",
            "headline_match_words": 14,
            "metadata_match_words": 18,
            "paragraph_characters": 250,
            "english_word_count": 48,
            "cjk_character_count": 0,
        }
    )
    assert not news_webpages._page_is_english(
        {
            "document_language": "",
            "headline": "Unrelated navigation headline",
            "headline_match_words": 1,
            "metadata_match_words": 1,
            "paragraph_characters": 250,
            "english_word_count": 48,
            "cjk_character_count": 0,
        }
    )


def test_page_language_gate_rejects_human_verification_page():
    assert not news_webpages._page_is_english(
        {
            "document_language": "en",
            "headline": "Let's confirm you are human",
            "headline_match_words": 4,
            "metadata_match_words": 4,
            "paragraph_characters": 250,
            "english_word_count": 48,
            "cjk_character_count": 0,
            "blocking_reason": "human verification challenge",
        }
    )


def test_techmeme_capture_relays_to_matched_original_publisher():
    assignment = {"source_url": "https://www.techmeme.com/260904/p33"}
    info = {
        "page_url": "https://www.techmeme.com/260904/p33#a260904p33",
        "headline_href": "https://www.theverge.com/games/story?utm_medium=gift-link",
        "headline_match_words": 14,
        "metadata_match_words": 18,
    }

    assert news_webpages._article_relay_url(info, assignment) == (
        "https://www.theverge.com/games/story?utm_medium=gift-link"
    )
    info["headline_href"] = "javascript:alert(1)"
    assert news_webpages._article_relay_url(info, assignment) == ""


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
                "captured_url": "https://publisher.example/story",
                "source_name": "Example News",
                "captured_source_name": "Publisher Example",
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
    assert plans[0]["news_webpage_url"] == "https://publisher.example/story"
    assert plans[0]["news_webpage_source"] == "Publisher Example"


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


@pytest.mark.asyncio
async def test_capture_uses_later_english_story_when_a_publisher_blocks(
    monkeypatch, tmp_path
):
    dossier = {
        "selected": [
            {
                "language": "en",
                "source_name": f"Source {index}",
                "title": f"Grounded story {index} alpha beta",
                "summary": f"Grounded story {index} alpha beta details",
                "url": f"https://example.com/story-{index}",
            }
            for index in range(1, 4)
        ]
    }
    storyboard = {
        "scenes": [
            {
                "id": f"scene-0{index}",
                "text": f"Grounded story {index} alpha beta details",
            }
            for index in range(1, 4)
        ]
    }
    plans = [
        {
            "id": f"scene-0{index}",
            "news_image_src": f"../news_images/image-{index}.png",
        }
        for index in range(1, 4)
    ]
    research = tmp_path / "research"
    research.mkdir()
    (research / "dossier.json").write_text(
        json.dumps(dossier), encoding="utf-8"
    )

    async def fake_start_chrome():
        return object(), "ws://127.0.0.1:1/devtools/browser/test", tmp_path / "profile"

    async def fake_stop_chrome(_process, _profile):
        return None

    attempted = []

    async def fake_capture(_browser_ws_url, assignment, destination):
        attempted.append(assignment["source_url"])
        if assignment["source_url"].endswith("story-1"):
            raise RuntimeError(
                "Rendered source blocked article capture: human verification challenge"
            )
        Image.new("RGB", (1440, 900), "white").save(destination)
        return {
            "document_language": "en",
            "headline": assignment["expected_headline"],
            "headline_match_words": 4,
            "metadata_match_words": 4,
            "paragraph_characters": 250,
            "english_word_count": 48,
        }

    monkeypatch.setattr(news_webpages, "_start_chrome", fake_start_chrome)
    monkeypatch.setattr(news_webpages, "_stop_chrome", fake_stop_chrome)
    monkeypatch.setattr(news_webpages, "_capture_page", fake_capture)

    manifest = await news_webpages.acquire_news_webpages(
        storyboard, plans, tmp_path, limit=2
    )

    assert manifest["status"] == "ready"
    assert manifest["requested_count"] == 2
    assert manifest["candidate_count"] == 3
    assert len(manifest["pages"]) == 2
    assert attempted == [
        "https://example.com/story-1",
        "https://example.com/story-2",
        "https://example.com/story-3",
    ]

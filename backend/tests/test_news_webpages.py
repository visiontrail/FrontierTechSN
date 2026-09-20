import asyncio
import base64
import hashlib
import io
import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest

from PIL import Image

from backend.pipeline import news_webpages


@pytest.mark.asyncio
async def test_cdp_deadline_names_the_stalled_command(monkeypatch):
    async def stalled_recv():
        await asyncio.Event().wait()

    socket = AsyncMock()
    socket.recv.side_effect = stalled_recv
    monkeypatch.setattr(news_webpages, "CDP_COMMAND_TIMEOUT_SECONDS", 0.01)
    with pytest.raises(news_webpages.CaptureCommandTimeout, match="Page.navigate exceeded"):
        await news_webpages._cdp_command(socket, [0], "Page.navigate")


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True])
async def test_capture_closes_failed_targets_and_starts_with_blank_page(monkeypatch, tmp_path, cancelled):
    client = AsyncMock()
    response = client.put.return_value
    response.raise_for_status = lambda: None
    response.json = lambda: {"id": "owned-target"}
    client.__aenter__.return_value = client
    monkeypatch.setattr(news_webpages.httpx, "AsyncClient", lambda **_: client)
    error = asyncio.CancelledError if cancelled else RuntimeError
    monkeypatch.setattr(news_webpages, "_capture_target", AsyncMock(side_effect=error))
    with pytest.raises(error):
        await news_webpages._capture_page(
            "ws://127.0.0.1:1234/devtools/browser/test", {}, tmp_path / "page.png"
        )
    assert client.put.call_args.kwargs["params"] == {"url": "about:blank"}
    client.get.assert_awaited_once_with("http://127.0.0.1:1234/json/close/owned-target")


@pytest.mark.asyncio
@pytest.mark.parametrize("source_has_prose,changed_after_capture", [(True, False), (False, False), (True, True)])
async def test_capture_keeps_valid_source_and_only_relays_missing_prose(
    monkeypatch, tmp_path, source_has_prose, changed_after_capture
):
    source = {
        "ready_state": "complete", "page_url": "https://www.techmeme.com/story",
        "headline_href": "https://publisher.example/story", "document_language": "en",
        "headline": "Grounded English news headline", "headline_match_words": 4,
        "paragraph_characters": 120, "english_word_count": 45,
    }
    publisher = {**source, "page_url": source["headline_href"]}
    final = source if source_has_prose else publisher
    values = [source] if source_has_prose else [{**source, "paragraph_characters": 0}, publisher]
    values.append({**final, "blocking_reason": "human verification challenge"} if changed_after_capture else final)
    command = AsyncMock(return_value={})

    @asynccontextmanager
    async def connect(*args, **kwargs):
        yield object()

    def save_image(raw, focus, destination):
        Image.new("RGB", (2880, 1800)).save(destination)
        return {}

    monkeypatch.setattr(news_webpages.websockets, "connect", connect)
    monkeypatch.setattr(news_webpages, "_cdp_command", command)
    monkeypatch.setattr(news_webpages, "_runtime_value", AsyncMock(side_effect=values))
    monkeypatch.setattr(news_webpages, "_capture_stable_article", AsyncMock(return_value=(b"pixels", {})))
    monkeypatch.setattr(news_webpages, "_focused_screenshot", save_image)
    capture = news_webpages._capture_target(
        {"webSocketDebuggerUrl": "ws://test"},
        {"source_url": source["page_url"]}, tmp_path / "page.png",
    )
    if changed_after_capture:
        with pytest.raises(RuntimeError, match="failed validation after screenshot"):
            await capture
    else:
        assert (await capture)["page_url"] == final["page_url"]
    navigations = [call.args[3]["url"] for call in command.call_args_list if call.args[2] == "Page.navigate"]
    assert navigations == ([source["page_url"]] if source_has_prose else [source["page_url"], publisher["page_url"]])


@pytest.mark.asyncio
async def test_techmeme_dom_counts_only_matched_story_prose():
    """Exercise the actual DOM script without any network or publisher dependency."""
    try:
        news_webpages._chrome_binary()
    except RuntimeError:
        pytest.skip("Local Chromium is required for DOM integration coverage")
    process, browser_url, profile = await news_webpages._start_chrome()
    try:
        from urllib.parse import urlsplit

        parsed = urlsplit(browser_url)
        async with news_webpages.httpx.AsyncClient() as client:
            target = (await client.put(f"http://{parsed.hostname}:{parsed.port}/json/new", params={"url": "about:blank"})).json()
        async with news_webpages.websockets.connect(target["webSocketDebuggerUrl"]) as socket:
            counter = [0]
            tree = await news_webpages._cdp_command(socket, counter, "Page.getFrameTree")
            headline = "Cybersecurity stocks outperform chip stocks amid escalating artificial intelligence fears"
            prose = "The software sector outperformed the chip sector to a historic degree as investors responded to escalating risks from artificial intelligence systems, lifting several major cybersecurity companies during Monday trading."
            script = news_webpages._PAGE_INFO_SCRIPT.replace("__EXPECTED_HEADLINE__", json.dumps(headline)).replace("location.hostname", "'www.techmeme.com'")
            for summary in [prose, ""]:
                await news_webpages._cdp_command(socket, counter, "Page.setDocumentContent", {
                    "frameId": tree["frameTree"]["frame"]["id"],
                    "html": f'<style>.ii {{width:650px;font-size:22px}} strong {{display:block}}</style><div class="ii"><strong>{headline}</strong> — {summary}</div><div class="ii"><strong>Unrelated headline</strong>{prose}</div><aside><p>{prose * 5}</p></aside>',
                })
                info = await news_webpages._runtime_value(socket, counter, script)
                assert info["paragraph_characters"] == len(summary)
                assert info["content_kind"] == "aggregator_excerpt"
                assert news_webpages._page_is_english(info) is bool(summary)
    finally:
        await news_webpages._stop_chrome(process, profile)


@pytest.mark.parametrize("change", ["shift", "offscreen", "outside_crop", "missing"])
def test_headline_capture_rejects_pixels_not_matching_measured_crop(change):
    rect = {"x": 100, "y": 100, "width": 500, "height": 120}
    prepared = {"headline_rect": rect, "focus_rect": {"x": 80, "y": 80, "width": 600, "height": 300}}
    after = dict(rect)
    assert news_webpages._headline_capture_is_stable(prepared, after)
    if change == "shift":
        after["y"] = 350
    elif change == "offscreen":
        prepared["headline_rect"] = after = {**rect, "y": -10}
    elif change == "outside_crop":
        prepared["focus_rect"]["x"] = 300
    else:
        after = None
    assert not news_webpages._headline_capture_is_stable(prepared, after)


@pytest.mark.asyncio
@pytest.mark.parametrize("settles", [True, False])
async def test_capture_retries_layout_shift_and_never_returns_stale_pixels(monkeypatch, settles):
    rect = {"x": 100, "y": 100, "width": 500, "height": 120}
    prepared = {"headline_rect": rect, "focus_rect": {"x": 80, "y": 80, "width": 600, "height": 300}}
    shots = []

    async def runtime(_socket, _counter, expression):
        if expression == "prepare":
            return prepared
        return rect if settles and len(shots) == 2 else {**rect, "y": 350}

    async def command(_socket, _counter, _method, _params):
        payload = f"pixels-{len(shots)}".encode()
        shots.append(payload)
        return {"data": base64.b64encode(payload).decode()}

    monkeypatch.setattr(news_webpages, "_runtime_value", runtime)
    monkeypatch.setattr(news_webpages, "_cdp_command", command)
    if settles:
        raw, measured = await news_webpages._capture_stable_article(None, [0], "prepare")
        assert raw == b"pixels-1" and measured == prepared and len(shots) == 2
    else:
        with pytest.raises(RuntimeError, match="headline moved"):
            await news_webpages._capture_stable_article(None, [0], "prepare")
        assert len(shots) == 3


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
    source = Image.new("RGB", (2880, 1800), "#eeeeee")
    source.paste("#123456", (200, 240, 1000, 640))
    payload = io.BytesIO()
    source.save(payload, format="PNG")
    destination = tmp_path / "focused.png"

    result = news_webpages._focused_screenshot(
        payload.getvalue(),
        {"x": 100, "y": 120, "width": 400, "height": 200},
        destination,
    )

    with Image.open(destination) as rendered:
        assert rendered.size == (2880, 1800)
        assert rendered.getpixel((1440, 900)) == (18, 52, 86)
        assert rendered.getpixel((10, 10)) == (255, 255, 255)
    assert result["focus_width"] == 800
    assert result["focused_render_width"] == 2640


@pytest.mark.asyncio
async def test_real_chromium_capture_and_crop_use_physical_pixels(tmp_path):
    """Exercise the real screenshot API; a DPR setting alone is not proof."""
    try:
        news_webpages._chrome_binary()
    except RuntimeError:
        pytest.skip("Local Chromium is required for screenshot integration coverage")
    process, browser_url, profile = await news_webpages._start_chrome()
    try:
        from urllib.parse import urlsplit

        parsed = urlsplit(browser_url)
        async with news_webpages.httpx.AsyncClient() as client:
            target = (await client.put(
                f"http://{parsed.hostname}:{parsed.port}/json/new",
                params={"url": "about:blank"},
            )).json()
        async with news_webpages.websockets.connect(target["webSocketDebuggerUrl"]) as socket:
            counter = [0]
            await news_webpages._cdp_command(socket, counter, "Emulation.setDeviceMetricsOverride", {
                "width": news_webpages.VIEWPORT_WIDTH, "height": news_webpages.VIEWPORT_HEIGHT,
                "deviceScaleFactor": news_webpages.CAPTURE_SCALE, "mobile": False,
            })
            await news_webpages._runtime_value(socket, counter,
                "document.body.style='margin:0;background:#123456'; true")
            shot = await news_webpages._cdp_command(socket, counter, "Page.captureScreenshot", {
                "format": "png", "fromSurface": True, "captureBeyondViewport": False,
            })
        raw = base64.b64decode(shot["data"])
        with Image.open(io.BytesIO(raw)) as captured:
            assert captured.size == (2880, 1800)
        destination = tmp_path / "article.png"
        info = news_webpages._focused_screenshot(
            raw, {"x": 100, "y": 120, "width": 400, "height": 200}, destination,
        )
        assert (info["focus_x"], info["focus_y"]) == (200, 240)
        with Image.open(destination) as image:
            assert image.size == (2880, 1800)
            assert image.getpixel((1440, 900)) == (18, 52, 86)
    finally:
        await news_webpages._stop_chrome(process, profile)


def test_attach_news_webpage_preserves_underlying_public_footage(tmp_path):
    root = tmp_path / "news_webpages"
    root.mkdir()
    image_path = root / "page-01.png"
    Image.new("RGB", (2880, 1800), "white").save(image_path)
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


def test_overlay_assignment_rejects_generic_reporting_word_overlap():
    dossier = {
        "selected": [
            {
                "language": "en",
                "source_name": "Techmeme",
                "title": "Google patches a Chrome flaw that could allow code execution",
                "summary": "A security publication reports that Google shipped the fix.",
                "evidence_text": (
                    "This archive page shows how the site appeared in September. "
                    "The current version is available at the home page."
                ),
                "url": "https://www.techmeme.com/example",
            }
        ]
    }
    storyboard = {
        "scenes": [
            {
                "id": "scene-05",
                "text": (
                    "DeepTech China reports that the Commerce Department could seek "
                    "input on artificial intelligence export controls."
                ),
            }
        ]
    }
    plans = [
        {
            "id": "scene-05",
            "news_image_src": "../news_images/the-information.png",
        }
    ]

    assert news_webpages.select_overlay_assignments(dossier, storyboard, plans) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_kind", ["blocked", "hung_command"])
async def test_capture_uses_later_english_story_when_a_publisher_blocks(
    monkeypatch, tmp_path, failure_kind
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
    cancelled = []

    async def fake_capture(_browser_ws_url, assignment, destination):
        attempted.append(assignment["source_url"])
        if assignment["source_url"].endswith("story-1"):
            if failure_kind == "hung_command":
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.append(assignment["source_url"])
            raise RuntimeError(
                "Rendered source blocked article capture: human verification challenge"
            )
        saved = json.loads((tmp_path / "news_webpages/manifest.json").read_text())
        assert saved["errors"][0]["source_url"].endswith("story-1")
        Image.new("RGB", (2880, 1800), "white").save(destination)
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
    monkeypatch.setattr(news_webpages, "PAGE_CAPTURE_TIMEOUT_SECONDS", 0.05)

    manifest = await news_webpages.acquire_news_webpages(
        storyboard, plans, tmp_path, limit=2
    )

    assert manifest["status"] == "ready"
    assert manifest["requested_count"] == 2
    assert manifest["candidate_count"] == 3
    assert len(manifest["pages"]) == 2
    if failure_kind == "hung_command":
        assert cancelled == ["https://example.com/story-1"]
        assert manifest["errors"][0]["message"] == "Article capture exceeded 0.05s deadline"
    assert attempted == [
        "https://example.com/story-1",
        "https://example.com/story-2",
        "https://example.com/story-3",
    ]

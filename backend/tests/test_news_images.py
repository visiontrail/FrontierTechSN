import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from PIL import Image

from backend.pipeline import news_images, visual_plan


def board() -> dict:
    return {
        "title": "Frontier Tech Daily",
        "scenes": [
            {
                "id": "scene-01",
                "duration": 8.0,
                "text": "NVIDIA announced that its Vera CPU has entered production.",
                "keywords": ["nvidia", "vera", "cpu"],
            },
            {
                "id": "scene-02",
                "duration": 9.0,
                "text": "SpaceX successfully launched a Falcon 9 rocket from Florida.",
                "keywords": ["spacex", "falcon", "rocket"],
            },
        ],
    }


def extended_board() -> dict:
    data = board()
    data["scenes"].extend(
        [
            {
                "id": "scene-03",
                "duration": 10.0,
                "text": "MIT researchers demonstrated a new robotics system in Cambridge.",
                "keywords": ["mit", "robotics", "cambridge"],
            },
            {
                "id": "scene-04",
                "duration": 7.0,
                "text": "Apple introduced a new chip for its Mac product line.",
                "keywords": ["apple", "chip", "mac"],
            },
        ]
    )
    return data


def failure_board() -> dict:
    return {
        "title": "Bacteria, innovation, and Qwen Office",
        "scenes": [
            {
                "id": "scene-02",
                "duration": 8.0,
                "text": (
                    "DeepTech China reports that MIT researchers are exploring living bacteria "
                    "as circuit boards. QbitAI reports that Perfect World's revenue rose."
                ),
                "keywords": ["bacteria", "boards", "revenue"],
            },
            {
                "id": "scene-03",
                "duration": 8.0,
                "text": (
                    "IEEE Spectrum has published an article with the IEEE Technology and "
                    "Engineering Management Society about engineers winning leadership backing."
                ),
                "keywords": ["ieee", "engineering", "leadership"],
            },
            {
                "id": "scene-04",
                "duration": 8.0,
                "text": (
                    "The article, authored by Alexander Brem, editor in chief of IEEE Engineering "
                    "Management Review and a professor at the University of Stuttgart, describes product ideas."
                ),
                "keywords": ["alexander", "brem", "product"],
            },
            {
                "id": "scene-05",
                "duration": 8.0,
                "text": (
                    "IEEE Spectrum notes that organizations including Google and 3M allocate work "
                    "time for employee ideas, according to Brem's research."
                ),
                "keywords": ["employee", "ideas", "innovation"],
            },
            {
                "id": "scene-07",
                "duration": 8.0,
                "text": (
                    "QbitAI reports that Jefferies tested global AI agents, with results showing "
                    "that Alibaba's Qwen Office ranked first ahead of Claude Cowork and Codex."
                ),
                "keywords": ["alibaba", "office", "agents"],
            },
            {
                "id": "scene-09",
                "duration": 8.0,
                "text": (
                    "QbitAI reports that Qwen Office alone scored above 90 in every dimension, "
                    "including browser control and multimodal content generation."
                ),
                "keywords": ["qwen", "office", "browser"],
            },
            {
                "id": "scene-10",
                "duration": 8.0,
                "text": (
                    "The Jefferies report, as described by QbitAI, evaluated model performance and "
                    "the Harness around each model. Qwen Office's Harness score ranked highest."
                ),
                "keywords": ["jefferies", "qwen", "harness"],
            },
        ],
    }


def _primary_plan() -> list[dict]:
    return [
        {
            "scene_id": "scene-01",
            "search_query": "NVIDIA logo",
            "news_query": "NVIDIA Vera CPU",
            "expected_subject": "NVIDIA",
            "kind": "logo",
            "display_mode": "inline",
            "purpose": "Show the company",
            "caption": "NVIDIA",
        },
        {
            "scene_id": "scene-02",
            "search_query": "SpaceX Falcon 9 launch",
            "news_query": "SpaceX Falcon 9",
            "expected_subject": "SpaceX Falcon 9",
            "kind": "event",
            "display_mode": "fullscreen",
            "purpose": "Show the launch",
            "caption": "Falcon 9",
        },
    ]


def _image_bytes(format_name: str = "JPEG") -> bytes:
    payload = io.BytesIO()
    Image.new("RGB", (800, 450), "#336699").save(payload, format=format_name)
    return payload.getvalue()


def _commons_candidate(scene_id: str, subject: str, kind: str = "event") -> dict:
    slug = scene_id.removeprefix("scene-")
    return {
        "provider": "Wikimedia Commons",
        "provider_id": "wikimedia",
        "title": f"{subject} {'logo' if kind == 'logo' else 'photograph'}",
        "source_page_url": f"https://commons.example/source-{slug}",
        "download_url": f"https://upload.example/source-{slug}.jpg",
        "creator": "Test photographer",
        "license": "CC BY-SA 4.0",
        "license_code": "CC-BY-SA-4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "attribution": "Test photographer",
        "description": f"A {'logo' if kind == 'logo' else 'photograph'} of {subject}",
        "width": 1600,
        "height": 900,
        "mime_type": "image/jpeg",
        "kind": kind,
    }


def _mock_acquisition_boundaries(
    monkeypatch,
    successful_scene_ids: set[str],
    source_aliases: dict[str, str] | None = None,
) -> list[str]:
    researched: list[str] = []
    monkeypatch.setattr(
        news_images,
        "plan_news_images",
        AsyncMock(return_value=(_primary_plan(), "mock-planner", "")),
    )

    async def fake_research(shot: dict) -> tuple[list[dict], str]:
        researched.append(shot["scene_id"])
        return [], "mock-reference-search"

    async def fake_search(_client, *, shot: dict, limit: int = 30) -> list[dict]:
        del limit
        scene_id = shot["scene_id"]
        if scene_id not in successful_scene_ids:
            return []
        candidate = _commons_candidate(scene_id, shot["expected_subject"], str(shot.get("kind") or "event"))
        source_alias = (source_aliases or {}).get(scene_id)
        if source_alias:
            candidate["source_page_url"] = f"https://commons.example/{source_alias}"
            candidate["download_url"] = f"https://upload.example/{source_alias}.jpg"
        return [candidate]

    async def fake_download(_client, *, candidate: dict, destination: Path):
        payload = _image_bytes()
        destination.write_bytes(payload)
        return len(payload), hashlib.sha256(payload).hexdigest()

    monkeypatch.setattr(news_images, "research_references", fake_research)
    monkeypatch.setattr(news_images, "search_wikimedia_images", fake_search)
    monkeypatch.setattr(news_images, "_download_candidate", fake_download)
    return researched


def test_normalise_plan_requires_exact_scenes_and_both_modes():
    value = {
        "images": [
            {
                "scene_id": "scene-01",
                "search_query": "NVIDIA logo",
                "expected_subject": "NVIDIA",
                "kind": "logo",
                "display_mode": "inline",
            },
            {
                "scene_id": "scene-02",
                "search_query": "SpaceX Falcon 9 launch",
                "expected_subject": "SpaceX Falcon 9",
                "kind": "event",
                "display_mode": "inline",
            },
            {
                "scene_id": "scene-99",
                "search_query": "unrelated",
                "expected_subject": "unrelated",
            },
        ]
    }

    plan = news_images._normalise_plan(
        value, eligible_scene_ids=["scene-01", "scene-02"], count=2
    )

    assert [item["scene_id"] for item in plan] == ["scene-01", "scene-02"]
    assert {item["display_mode"] for item in plan} == {"inline", "fullscreen"}


@pytest.mark.asyncio
async def test_partial_planner_rows_are_preserved_and_filled_from_unused_scenes(
    monkeypatch,
):
    partial = {
        "images": [
            {
                "scene_id": "scene-02",
                "search_query": "SpaceX Falcon 9 launch",
                "news_query": "SpaceX Falcon 9",
                "expected_subject": "SpaceX Falcon 9",
                "kind": "event",
                "display_mode": "inline",
                "purpose": "Planner-selected launch",
                "caption": "Falcon 9",
            }
        ]
    }
    result = SimpleNamespace(
        stdout=json.dumps(
            [
                {
                    "Response": json.dumps(partial),
                    "ConversationUrl": "https://chat.example/conversation",
                }
            ]
        )
    )
    monkeypatch.setattr(
        news_images,
        "run_opencli_with_retries",
        AsyncMock(return_value=result),
    )
    data = extended_board()

    plan, planner, conversation_url = await news_images.plan_news_images(
        data,
        eligible_scene_ids=["scene-01", "scene-02", "scene-03"],
        count=3,
    )

    assert len(plan) == 3
    assert plan[0]["scene_id"] == "scene-02"
    assert plan[0]["purpose"] == "Planner-selected launch"
    assert len({item["scene_id"] for item in plan}) == 3
    assert {item["scene_id"] for item in plan} == {
        "scene-01",
        "scene-02",
        "scene-03",
    }
    assert {item["display_mode"] for item in plan} == {"inline", "fullscreen"}
    assert planner == "opencli:chatgpt-picture-editor+deterministic-fill"
    assert conversation_url == "https://chat.example/conversation"


@pytest.mark.asyncio
async def test_planner_timeout_uses_narrated_entities_and_unique_qwen_story_owners(
    monkeypatch,
):
    monkeypatch.setattr(
        news_images,
        "run_opencli_with_retries",
        AsyncMock(side_effect=TimeoutError("chatgpt planning timed out after 45s")),
    )
    data = failure_board()
    hints = {scene["id"]: {"kicker": "AI AGENTS TESTED", "headline": "World Google and"} for scene in data["scenes"]}

    plan, planner, conversation_url = await news_images.plan_news_images(
        data,
        eligible_scene_ids=[scene["id"] for scene in data["scenes"]],
        count=7,
        scene_hints=hints,
    )

    subjects = {shot["scene_id"]: shot["expected_subject"] for shot in plan}
    assert planner == "deterministic-fallback"
    assert conversation_url == ""
    assert len(plan) == 7
    assert subjects["scene-02"] == "Perfect World"
    assert subjects["scene-04"] == "Alexander Brem"
    assert subjects["scene-05"] == "Google"
    assert subjects["scene-07"] == "Alibaba"
    assert subjects["scene-09"] == "Qwen Office"
    assert subjects["scene-10"] == "Jefferies"
    assert len({news_images._subject_identity_key(shot) for shot in plan}) == 7
    assert {shot["display_mode"] for shot in plan} == {"inline", "fullscreen"}
    assert not {"World", "Google and", "AI AGENTS TESTED"} & set(subjects.values())


def test_real_failure_story_entities_keep_possessives_and_strip_caption_residue():
    entities = {
        scene["id"]: news_images._entity_candidates(
            scene,
            {"kicker": "AI AGENTS TESTED", "headline": "World Google and"},
        )
        for scene in failure_board()["scenes"]
    }

    assert entities["scene-02"][:2] == ["Perfect World", "MIT"]
    assert entities["scene-04"][0] == "Alexander Brem"
    assert entities["scene-05"][:2] == ["Google", "3M"]
    assert "Qwen Office" in entities["scene-07"]
    assert entities["scene-09"][:2] == ["Qwen Office", "Qwen"]
    assert entities["scene-10"][0] == "Qwen Office"
    assert "Jefferies" in entities["scene-10"][:3]
    assert all(not entity.endswith((" and", " of", " the")) for rows in entities.values() for entity in rows)


def test_fallback_subject_ignores_only_generic_uppercase_keyword_kickers():
    scene = {
        "id": "scene-01",
        "text": "MIT researchers are teaching a robot to adapt inside a laboratory.",
        "keywords": ["mit", "robot", "laboratory"],
    }

    generic = news_images._fallback_subject(
        scene,
        {"kicker": "ROBOT", "headline": "MIT researchers train adaptable robots"},
    )
    acronym = news_images._fallback_subject(
        scene,
        {"kicker": "MIT", "headline": "Researchers train adaptable robots"},
    )

    assert generic == "MIT"
    assert acronym == "MIT"


def test_wikimedia_candidate_is_license_and_resolution_gated():
    page = {
        "title": "File:NVIDIA logo.svg",
        "imageinfo": [
            {
                "mime": "image/svg+xml",
                "thumbmime": "image/png",
                "url": "https://upload.wikimedia.org/logo.svg",
                "thumburl": "https://upload.wikimedia.org/logo.png",
                "descriptionurl": "https://commons.wikimedia.org/wiki/File:NVIDIA_logo.svg",
                "width": 656,
                "height": 120,
                "thumbwidth": 1600,
                "thumbheight": 293,
                "extmetadata": {
                    "LicenseShortName": {"value": "Public domain"},
                    "Artist": {"value": "NVIDIA Corporation"},
                },
            }
        ],
    }
    shot = {
        "kind": "logo",
        "display_mode": "inline",
        "search_query": "NVIDIA logo",
        "expected_subject": "NVIDIA",
        "_scene_text": "NVIDIA announced its Vera CPU.",
        "_scene_keywords": ["nvidia", "vera", "cpu"],
    }

    candidate = news_images._candidate_from_page(page, shot)

    assert candidate is not None
    assert candidate["download_url"].endswith("logo.png")
    assert candidate["license"] == "Public domain"
    assert news_images._extension_for(candidate) == ".png"

    page["imageinfo"][0]["extmetadata"]["LicenseShortName"]["value"] = "All rights reserved"
    assert news_images._candidate_from_page(page, shot) is None


def test_three_way_grounding_rejects_world_aquatics_chemical_agents_and_google_loon():
    world = news_images._grounding_evidence(
        {"expected_subject": "Perfect World", "kind": "logo"},
        {
            "text": "Perfect World's semiannual revenue report was published.",
            "keywords": [],
        },
        {
            "title": "World Aquatics logo",
            "description": "Logo of the international swimming federation",
        },
    )
    chemical = news_images._grounding_evidence(
        {"expected_subject": "AI AGENTS TESTED", "kind": "event"},
        {"text": "Eight global AI agents were tested on office tasks.", "keywords": []},
        {
            "title": "Chemical and biological agents decontamination test",
            "description": "A hazardous-site response simulation",
        },
    )
    loon = news_images._grounding_evidence(
        {"expected_subject": "Google", "kind": "event"},
        {
            "text": "Google allocates employee work time to develop ideas.",
            "keywords": [],
        },
        {
            "title": "Google Loon launch event",
            "description": "A balloon launch in 2013",
        },
    )

    assert world["grounding_passed"] is False
    assert "perfect" not in world["grounding_distinctive_anchors"]
    assert chemical["grounding_passed"] is False
    assert chemical["grounding_distinctive_anchors"] == []
    assert loon["grounding_passed"] is False
    assert "only one organization" in loon["grounding_reason"]


def test_logo_ranking_prefers_corporate_mark_over_product_logo():
    shot = {
        "kind": "logo",
        "display_mode": "inline",
        "search_query": "NVIDIA logo",
        "expected_subject": "NVIDIA",
    }
    corporate = {
        "title": "NVIDIA logo.svg",
        "description": "Corporate logo of NVIDIA",
        "mime_type": "image/svg+xml",
        "width": 656,
        "height": 120,
        "source_page_url": "https://commons.example/nvidia",
    }
    product = {
        "title": "Nvidia Shield Portable logo.png",
        "description": "Logo for the Nvidia Shield Portable product",
        "mime_type": "image/png",
        "width": 3840,
        "height": 2160,
        "source_page_url": "https://commons.example/nvidia-shield",
    }
    faux_transparent = {
        "title": "Logo-nvidia-transparent-PNG.png",
        "description": "Nvidia logo",
        "mime_type": "image/png",
        "width": 1600,
        "height": 1200,
        "source_page_url": "https://commons.example/nvidia-checkerboard",
    }

    ranked = sorted(
        [product, faux_transparent, corporate],
        key=lambda candidate: news_images._candidate_rank(candidate, shot),
    )

    assert ranked[0] == corporate


def test_download_extension_uses_rasterized_commons_thumbnail_suffix():
    candidate = {
        "download_url": "https://upload.wikimedia.org/thumb/a/ab/Alibaba.svg/1600px-Alibaba.svg.png",
        "mime_type": "image/svg+xml",
    }

    assert news_images._extension_for(candidate) == ".png"


@pytest.mark.asyncio
async def test_wikimedia_429_honors_retry_after_and_retries(monkeypatch):
    request = news_images.httpx.Request("GET", news_images.WIKIMEDIA_API)

    class FakeClient:
        def __init__(self):
            self.calls = 0

        async def get(self, _url, params):
            del params
            self.calls += 1
            if self.calls == 1:
                return news_images.httpx.Response(
                    429,
                    headers={"retry-after": "2"},
                    request=request,
                )
            return news_images.httpx.Response(
                200,
                json={"query": {"pages": []}},
                request=request,
            )

    client = FakeClient()
    sleep = AsyncMock()
    monkeypatch.setattr(news_images.asyncio, "sleep", sleep)

    candidates = await news_images.search_wikimedia_images(
        client,
        shot={
            "scene_id": "scene-01",
            "search_query": "NVIDIA logo",
            "expected_subject": "NVIDIA",
            "kind": "logo",
            "display_mode": "inline",
            "_scene_text": "NVIDIA announced the Vera CPU.",
            "_scene_keywords": ["nvidia"],
        },
    )

    assert candidates == []
    assert client.calls == 2
    sleep.assert_awaited_once_with(2.0)


def test_attach_news_images_preserves_inline_archetype_and_promotes_fullscreen(
    tmp_path: Path,
):
    data = board()
    plans = visual_plan.fallback_plan(data)
    asset_dir = tmp_path / "news_images"
    asset_dir.mkdir()
    (asset_dir / "inline.png").write_bytes(b"image")
    (asset_dir / "full.jpg").write_bytes(b"image")
    manifest = {
        "images": [
            {
                "scene_id": "scene-01",
                "local_path": "news_images/inline.png",
                "display_mode": "inline",
                "kind": "logo",
                "fit": "contain",
                "expected_subject": "NVIDIA",
                "search_query": "NVIDIA logo",
                "creator": "NVIDIA",
                "license": "Public domain",
                "source_page_url": "https://commons.example/nvidia",
                "match_terms": ["nvidia"],
                "grounding_policy_version": news_images.QUERY_SEMANTICS_VERSION,
                "grounding_passed": True,
                "grounding_distinctive_anchors": ["nvidia"],
                "grounding_reason": "test proof",
                "references": [{"url": "https://news.example/nvidia"}],
            },
            {
                "scene_id": "scene-02",
                "local_path": "news_images/full.jpg",
                "display_mode": "fullscreen",
                "kind": "event",
                "fit": "cover",
                "expected_subject": "Falcon 9",
                "search_query": "SpaceX Falcon 9 launch",
                "creator": "NASA",
                "license": "Public domain",
                "source_page_url": "https://commons.example/falcon",
                "match_terms": ["falcon", "spacex"],
                "grounding_policy_version": news_images.QUERY_SEMANTICS_VERSION,
                "grounding_passed": True,
                "grounding_distinctive_anchors": ["falcon", "spacex"],
                "grounding_reason": "test proof",
                "references": [],
            },
        ]
    }

    summary = news_images.attach_news_images(plans, data, manifest, tmp_path)

    assert summary == {"attached": 2, "placement_modes": {"inline": 1, "fullscreen": 1}}
    assert plans[0]["archetype"] != "news_image"
    assert plans[0]["news_image_src"] == "../news_images/inline.png"
    assert plans[1]["archetype"] == "news_image"
    assert plans[1]["news_image_original_archetype"] == "topic"


def test_cached_manifest_requires_same_storyboard_and_materialized_assets(
    tmp_path: Path,
):
    data = board()
    asset_dir = tmp_path / "news_images"
    asset_dir.mkdir()
    payload = _image_bytes("PNG")
    (asset_dir / "image-01.png").write_bytes(payload)
    storyboard_sha256 = news_images.storyboard_fingerprint(data)
    contract_sha256 = news_images.acquisition_contract_fingerprint(
        storyboard_sha256=storyboard_sha256,
        requested_count=1,
        target_count=1,
        eligible_scene_ids=["scene-01", "scene-02"],
        excluded_scene_ids=set(),
    )
    manifest = {
        "manifest_version": news_images.MANIFEST_VERSION,
        "query_semantics_version": news_images.QUERY_SEMANTICS_VERSION,
        "grounding_policy_version": news_images.QUERY_SEMANTICS_VERSION,
        "cache_contract_sha256": contract_sha256,
        "status": "ready",
        "storyboard_sha256": storyboard_sha256,
        "planned_image_count": 1,
        "placement_modes": {"inline": 1, "fullscreen": 0},
        "images": [
            {
                "scene_id": "scene-01",
                "source_page_url": "https://commons.example/nvidia",
                "local_path": "news_images/image-01.png",
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "grounding_policy_version": news_images.QUERY_SEMANTICS_VERSION,
                "grounding_passed": True,
                "grounding_distinctive_anchors": ["nvidia"],
            }
        ],
    }
    (asset_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    assert news_images._cached_manifest(tmp_path, manifest["storyboard_sha256"], contract_sha256) == manifest
    assert news_images._cached_manifest(tmp_path, "stale", contract_sha256) is None
    assert news_images._cached_manifest(tmp_path, storyboard_sha256, "stale") is None

    manifest["manifest_version"] = news_images.MANIFEST_VERSION - 1
    (asset_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert news_images._cached_manifest(tmp_path, manifest["storyboard_sha256"], contract_sha256) is None


def test_cached_manifest_rejects_duplicate_identity_stale_policy_and_bad_hash(
    tmp_path: Path,
):
    data = board()
    asset_dir = tmp_path / "news_images"
    asset_dir.mkdir()
    payload = _image_bytes("PNG")
    for name in ("image-01.png", "image-02.png"):
        (asset_dir / name).write_bytes(payload)
    fingerprint = news_images.storyboard_fingerprint(data)
    contract = news_images.acquisition_contract_fingerprint(
        storyboard_sha256=fingerprint,
        requested_count=2,
        target_count=2,
        eligible_scene_ids=["scene-01", "scene-02"],
        excluded_scene_ids=set(),
    )

    def image(index: int, scene_id: str, source: str, anchor: str) -> dict:
        return {
            "scene_id": scene_id,
            "source_page_url": source,
            "local_path": f"news_images/image-{index:02d}.png",
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "grounding_policy_version": news_images.QUERY_SEMANTICS_VERSION,
            "grounding_passed": True,
            "grounding_distinctive_anchors": [anchor],
        }

    manifest = {
        "manifest_version": news_images.MANIFEST_VERSION,
        "query_semantics_version": news_images.QUERY_SEMANTICS_VERSION,
        "grounding_policy_version": news_images.QUERY_SEMANTICS_VERSION,
        "cache_contract_sha256": contract,
        "status": "ready",
        "storyboard_sha256": fingerprint,
        "planned_image_count": 2,
        "placement_modes": {"inline": 1, "fullscreen": 1},
        "images": [
            image(1, "scene-01", "https://commons.example/nvidia", "nvidia"),
            image(2, "scene-02", "https://commons.example/spacex", "spacex"),
        ],
    }

    def write() -> None:
        (asset_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    write()
    assert news_images._cached_manifest(tmp_path, fingerprint, contract) == manifest

    manifest["images"][1]["source_page_url"] = manifest["images"][0]["source_page_url"]
    write()
    assert news_images._cached_manifest(tmp_path, fingerprint, contract) is None
    manifest["images"][1]["source_page_url"] = "https://commons.example/spacex"
    manifest["images"][1]["scene_id"] = "scene-01"
    write()
    assert news_images._cached_manifest(tmp_path, fingerprint, contract) is None
    manifest["images"][1]["scene_id"] = "scene-02"
    manifest["query_semantics_version"] -= 1
    write()
    assert news_images._cached_manifest(tmp_path, fingerprint, contract) is None
    manifest["query_semantics_version"] = news_images.QUERY_SEMANTICS_VERSION
    manifest["images"][1]["sha256"] = "0" * 64
    write()
    assert news_images._cached_manifest(tmp_path, fingerprint, contract) is None


def test_wikimedia_queries_broaden_without_dropping_the_scene_subject():
    variants = news_images._wikimedia_query_variants(
        {
            "search_query": "NVIDIA VERA logo",
            "expected_subject": "NVIDIA VERA",
            "kind": "logo",
        },
        board()["scenes"][0],
    )

    assert variants[0] == "NVIDIA VERA logo"
    assert "NVIDIA logo" in variants


@pytest.mark.asyncio
async def test_failed_primary_uses_alternate_entity_query_for_the_same_scene(
    tmp_path: Path,
    monkeypatch,
):
    data = {"title": "Vera", "scenes": [board()["scenes"][0]]}
    monkeypatch.setattr(
        news_images,
        "plan_news_images",
        AsyncMock(return_value=([_primary_plan()[0]], "mock-planner", "")),
    )
    monkeypatch.setattr(
        news_images,
        "research_references",
        AsyncMock(return_value=([], "mock-reference-search")),
    )
    attempts: list[tuple[str, str]] = []

    async def fake_search(_client, *, shot: dict, limit: int = 30) -> list[dict]:
        del limit
        attempts.append((shot["scene_id"], shot["expected_subject"]))
        if shot["expected_subject"] != "Vera CPU":
            return []
        return [_commons_candidate("scene-01", "Vera CPU", "logo")]

    async def fake_download(_client, *, candidate: dict, destination: Path):
        payload = _image_bytes()
        destination.write_bytes(payload)
        return len(payload), hashlib.sha256(payload).hexdigest()

    monkeypatch.setattr(news_images, "search_wikimedia_images", fake_search)
    monkeypatch.setattr(news_images, "_download_candidate", fake_download)

    manifest = await news_images.acquire_news_images(data, tmp_path, count=1)

    assert manifest["status"] == "ready"
    assert attempts[0] == ("scene-01", "NVIDIA")
    assert ("scene-01", "Vera CPU") in attempts
    assert manifest["images"][0]["scene_id"] == "scene-01"
    assert manifest["images"][0]["expected_subject"] == "Vera CPU"
    assert manifest["images"][0]["candidate_role"] == "reserve"


@pytest.mark.asyncio
async def test_seven_scene_fallback_reaches_seven_unique_images_with_mode_mix(
    tmp_path: Path,
    monkeypatch,
):
    data = failure_board()
    fallback = news_images._fallback_plan(data["scenes"], 7)
    monkeypatch.setattr(
        news_images,
        "plan_news_images",
        AsyncMock(return_value=(fallback, "deterministic-fallback", "")),
    )
    monkeypatch.setattr(
        news_images,
        "research_references",
        AsyncMock(return_value=([], "mock-reference-search")),
    )

    async def fake_search(_client, *, shot: dict, limit: int = 30) -> list[dict]:
        del limit
        identity = news_images._subject_identity_key(shot)
        candidate = _commons_candidate(shot["scene_id"], shot["expected_subject"], shot["kind"])
        # Every Qwen query intentionally resolves to one Commons page. The
        # allocator must leave only the scarce scene-09 on that identity.
        source_slug = "shared-qwen" if identity == "qwen" else identity
        candidate["source_page_url"] = f"https://commons.example/{source_slug}"
        candidate["download_url"] = f"https://upload.example/{source_slug}.jpg"
        return [candidate]

    async def fake_download(_client, *, candidate: dict, destination: Path):
        payload = _image_bytes()
        destination.write_bytes(payload)
        return len(payload), hashlib.sha256(payload).hexdigest()

    monkeypatch.setattr(news_images, "search_wikimedia_images", fake_search)
    monkeypatch.setattr(news_images, "_download_candidate", fake_download)
    old_asset = tmp_path / "news_images" / "image-99.jpg"
    old_asset.parent.mkdir(parents=True)
    old_asset.write_bytes(b"stale partial asset")
    keep = old_asset.parent / "operator-note.txt"
    keep.write_text("keep", encoding="utf-8")

    manifest = await news_images.acquire_news_images(data, tmp_path, count=7)

    assert manifest["status"] == "ready"
    assert len(manifest["images"]) == 7
    assert len({image["scene_id"] for image in manifest["images"]}) == 7
    assert len({image["source_page_url"] for image in manifest["images"]}) == 7
    assert manifest["placement_modes"]["inline"] > 0
    assert manifest["placement_modes"]["fullscreen"] > 0
    subjects = {image["scene_id"]: image["expected_subject"] for image in manifest["images"]}
    assert subjects["scene-07"] == "Alibaba"
    assert subjects["scene-09"] == "Qwen Office"
    assert subjects["scene-10"] == "Jefferies"
    assert not old_asset.exists()
    assert keep.read_text(encoding="utf-8") == "keep"


@pytest.mark.asyncio
async def test_acquisition_uses_reserves_continuous_names_and_stops_at_target(
    tmp_path: Path,
    monkeypatch,
):
    researched = _mock_acquisition_boundaries(
        monkeypatch,
        {"scene-01", "scene-02", "scene-03", "scene-04"},
        {"scene-01": "shared-primary", "scene-02": "shared-primary"},
    )

    manifest = await news_images.acquire_news_images(
        extended_board(),
        tmp_path,
        count=2,
    )

    assert manifest["status"] == "ready"
    assert manifest["primary_query_count"] == 2
    assert manifest["reserve_query_count"] >= 2
    assert [image["scene_id"] for image in manifest["images"]] == [
        "scene-01",
        "scene-03",
    ]
    assert [image["id"] for image in manifest["images"]] == ["image-01", "image-02"]
    assert [image["local_path"] for image in manifest["images"]] == [
        "news_images/image-01.jpg",
        "news_images/image-02.jpg",
    ]
    assert len({image["scene_id"] for image in manifest["images"]}) == 2
    assert len({image["source_page_url"] for image in manifest["images"]}) == 2
    assert [image["display_mode"] for image in manifest["images"]] == [
        "inline",
        "fullscreen",
    ]
    assert manifest["placement_modes"] == {"inline": 1, "fullscreen": 1}
    selected_queries = {
        image["scene_id"]: next(
            query
            for query in manifest["queries"]
            if query["scene_id"] == image["scene_id"] and query["expected_subject"] == image["expected_subject"]
        )
        for image in manifest["images"]
    }
    assert selected_queries["scene-01"]["planned_display_mode"] == "inline"
    assert selected_queries["scene-01"]["display_mode"] == "inline"
    assert selected_queries["scene-03"]["candidate_role"] == "reserve"
    assert selected_queries["scene-03"]["planned_display_mode"] == "inline"
    assert selected_queries["scene-03"]["display_mode"] == "fullscreen"
    assert researched[:2] == ["scene-01", "scene-02"]
    assert researched.count("scene-02") >= 2
    assert researched[-1] == "scene-03"
    assert "scene-04" not in researched


@pytest.mark.asyncio
async def test_acquisition_tries_next_candidate_after_download_failure(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(
        news_images,
        "plan_news_images",
        AsyncMock(return_value=(_primary_plan(), "mock-planner", "")),
    )
    research = AsyncMock(return_value=([], "mock-reference-search"))
    monkeypatch.setattr(news_images, "research_references", research)

    failed = _commons_candidate("scene-01", "NVIDIA", "logo")
    failed["source_page_url"] = "https://commons.example/a-download-fails"
    failed["download_url"] = "https://upload.example/a-download-fails.jpg"
    succeeds = _commons_candidate("scene-01", "NVIDIA", "logo")
    succeeds["source_page_url"] = "https://commons.example/b-download-succeeds"
    succeeds["download_url"] = "https://upload.example/b-download-succeeds.jpg"

    async def fake_search(_client, *, shot: dict, limit: int = 30) -> list[dict]:
        del limit
        return [failed, succeeds] if shot["scene_id"] == "scene-01" else []

    attempts: list[str] = []

    async def fake_download(_client, *, candidate: dict, destination: Path):
        source = candidate["source_page_url"]
        attempts.append(source)
        if source == failed["source_page_url"]:
            raise RuntimeError("mock download failure")
        payload = _image_bytes()
        destination.write_bytes(payload)
        return len(payload), hashlib.sha256(payload).hexdigest()

    monkeypatch.setattr(news_images, "search_wikimedia_images", fake_search)
    monkeypatch.setattr(news_images, "_download_candidate", fake_download)

    manifest = await news_images.acquire_news_images(
        extended_board(),
        tmp_path,
        count=1,
    )

    assert manifest["status"] == "ready"
    assert attempts == [failed["source_page_url"], succeeds["source_page_url"]]
    assert manifest["images"][0]["source_page_url"] == succeeds["source_page_url"]
    assert manifest["images"][0]["id"] == "image-01"
    assert research.await_count == 1
    assert any(
        error["stage"] == "download" and error["source_page_url"] == failed["source_page_url"]
        for error in manifest["errors"]
    )


@pytest.mark.asyncio
async def test_acquisition_remains_partial_after_all_reserves_are_exhausted(
    tmp_path: Path,
    monkeypatch,
):
    researched = _mock_acquisition_boundaries(monkeypatch, {"scene-01"})

    manifest = await news_images.acquire_news_images(
        extended_board(),
        tmp_path,
        count=2,
    )

    assert manifest["status"] == "partial"
    assert manifest["planned_image_count"] == 2
    assert len(manifest["images"]) == 1
    assert manifest["images"][0]["id"] == "image-01"
    assert set(researched) == {"scene-01", "scene-02", "scene-03", "scene-04"}
    assert researched.count("scene-02") >= 2
    fingerprint = news_images.storyboard_fingerprint(extended_board())
    contract = news_images.acquisition_contract_fingerprint(
        storyboard_sha256=fingerprint,
        requested_count=2,
        target_count=2,
        eligible_scene_ids=["scene-01", "scene-02", "scene-03", "scene-04"],
        excluded_scene_ids=set(),
    )
    assert (
        news_images._cached_manifest(
            tmp_path,
            fingerprint,
            contract,
        )
        is None
    )

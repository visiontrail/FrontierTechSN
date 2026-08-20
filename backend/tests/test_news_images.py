import hashlib
import io
import json
import struct
import zlib
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


def _image_bytes(
    format_name: str = "JPEG",
    seed: str = "default",
    size: tuple[int, int] = (800, 450),
) -> bytes:
    payload = io.BytesIO()
    color = tuple(hashlib.sha256(seed.encode("utf-8")).digest()[:3])
    Image.new("RGB", size, color).save(payload, format=format_name)
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


def _grounded_image_record(
    scene: dict,
    *,
    local_path: str,
    payload: bytes,
    source: str,
    subject: str,
    kind: str,
    mode: str,
) -> dict:
    shot = {
        "scene_id": scene["id"],
        "expected_subject": subject,
        "kind": kind,
        "display_mode": mode,
        "search_query": f"{subject} {'logo' if kind == 'logo' else 'photograph'}",
    }
    candidate = {
        "title": f"{subject} {'logo' if kind == 'logo' else 'photograph'}",
        "description": f"A {'logo' if kind == 'logo' else 'photograph'} of {subject}",
        "attribution": "Test photographer",
        "license": "CC BY-SA 4.0",
        "license_code": "CC-BY-SA-4.0",
        "source_page_url": source,
    }
    evidence = news_images._grounding_evidence(shot, scene, candidate)
    assert evidence["grounding_passed"] is True
    return {
        **shot,
        **candidate,
        **evidence,
        "match_terms": list(evidence["grounding_distinctive_anchors"]),
        "local_path": local_path,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "references": [],
    }


def _cached_news_manifest(
    task_dir: Path,
    data: dict,
    fingerprint: str,
    contract: str,
    *,
    requested_count: int,
    excluded: set[str] | None = None,
) -> dict | None:
    excluded_ids = excluded or set()
    eligible = [
        str(scene.get("id") or "")
        for scene in data.get("scenes") or []
        if str(scene.get("id") or "") not in excluded_ids
    ]
    return news_images._cached_manifest(
        task_dir,
        fingerprint,
        contract,
        data,
        expected_requested_count=requested_count,
        expected_target=min(max(0, requested_count), len(eligible)),
        expected_eligible_scene_ids=eligible,
        expected_excluded_scene_ids=excluded_ids,
    )


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
        payload = _image_bytes(seed=candidate["source_page_url"])
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
    assert candidate["mime_type"] == "image/png"
    assert candidate["license"] == "Public domain"
    assert news_images._extension_for(candidate) == ".png"

    page["imageinfo"][0].pop("thumbmime")
    page["imageinfo"][0].pop("thumburl")
    assert news_images._candidate_from_page(page, shot) is None

    page["imageinfo"][0]["thumbmime"] = "image/png"
    page["imageinfo"][0]["thumburl"] = "https://upload.wikimedia.org/logo.png"

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
    assert "complete identity" in loon["grounding_reason"]


@pytest.mark.parametrize(
    ("shot", "scene", "candidate"),
    [
        (
            {"expected_subject": "University of Stuttgart", "kind": "logo"},
            {"text": "The University of Stuttgart published the study.", "keywords": []},
            {"title": "Uni Stuttgart logo English", "description": ""},
        ),
        (
            {"expected_subject": "Google DeepMind", "kind": "logo"},
            {"text": "Google DeepMind released the model.", "keywords": []},
            {"title": "Google DeepMind logo", "description": ""},
        ),
        (
            {"expected_subject": "Alexander Brem", "kind": "person"},
            {"text": "Alexander Brem authored the article.", "keywords": []},
            {"title": "Editorial portrait", "description": "Portrait of Alexander Brem"},
        ),
        (
            {"expected_subject": "ChatGPT", "kind": "product"},
            {"text": "ChatGPT released a product update.", "keywords": []},
            {"title": "ChatGPT product", "description": ""},
        ),
        (
            {"expected_subject": "CES", "kind": "event"},
            {"text": "CES opened today in Las Vegas.", "keywords": []},
            {"title": "CES 2026 event", "description": ""},
        ),
        (
            {"expected_subject": "3M", "kind": "logo"},
            {"text": "3M allocates employee time to ideas.", "keywords": []},
            {"title": "3M logo", "description": ""},
        ),
    ],
)
def test_complete_identity_grounding_positive_controls(shot, scene, candidate):
    evidence = news_images._grounding_evidence(shot, scene, candidate)

    assert evidence["grounding_passed"] is True
    assert evidence["grounding_identity_field"] in {"title", "description"}
    assert evidence["grounding_distinctive_anchors"] == news_images._identity_tokens(
        shot["expected_subject"]
    )


@pytest.mark.parametrize(
    ("shot", "scene", "candidate"),
    [
        (
            {"expected_subject": "University of Stuttgart", "kind": "logo"},
            {"text": "The University of Stuttgart published the study.", "keywords": []},
            {"title": "University of Oxford logo", "description": ""},
        ),
        (
            {"expected_subject": "Google DeepMind", "kind": "logo"},
            {"text": "Google DeepMind released the model.", "keywords": []},
            {"title": "Google logo", "description": ""},
        ),
        (
            {"expected_subject": "Qwen Office", "kind": "logo"},
            {"text": "Qwen Office ranked first.", "keywords": []},
            {"title": "Qwen Audio logo", "description": ""},
        ),
        (
            {"expected_subject": "Perfect World", "kind": "logo"},
            {"text": "Perfect World's revenue rose.", "keywords": []},
            {"title": "Perfect 10 logo", "description": ""},
        ),
        (
            {"expected_subject": "Alexander Brem", "kind": "person"},
            {"text": "Alexander Brem authored the article.", "keywords": []},
            {"title": "Alexander", "description": "Brem portrait"},
        ),
        (
            {"expected_subject": "Alexander Brem", "kind": "person"},
            {"text": "Alexander Brem authored the article.", "keywords": []},
            {"title": "Editorial portrait", "description": "Professor", "attribution": "Alexander Brem"},
        ),
        (
            {"expected_subject": "Google DeepMind", "kind": "logo"},
            {"text": "The lab released a model.", "keywords": ["Google DeepMind"]},
            {"title": "Google DeepMind logo", "description": ""},
        ),
        (
            {"expected_subject": "AI Agents Tested", "kind": "event"},
            {"text": "AI Agents Tested office workflows.", "keywords": []},
            {"title": "AI Agents Tested event", "description": ""},
        ),
    ],
)
def test_complete_identity_grounding_negative_controls(shot, scene, candidate):
    evidence = news_images._grounding_evidence(shot, scene, candidate)

    assert evidence["grounding_passed"] is False
    assert evidence["grounding_distinctive_anchors"] == []


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
    inline_payload = _image_bytes("PNG", "inline")
    fullscreen_payload = _image_bytes("JPEG", "fullscreen")
    (asset_dir / "image-01.png").write_bytes(inline_payload)
    (asset_dir / "image-02.jpg").write_bytes(fullscreen_payload)
    manifest = {
        "manifest_version": news_images.MANIFEST_VERSION,
        "query_semantics_version": news_images.QUERY_SEMANTICS_VERSION,
        "grounding_policy_version": news_images.QUERY_SEMANTICS_VERSION,
        "status": "ready",
        "storyboard_sha256": news_images.storyboard_fingerprint(data),
        "license_policy": "open_only",
        "requested_image_count": 2,
        "planned_image_count": 2,
        "eligible_scene_count": 2,
        "eligible_scene_ids": ["scene-01", "scene-02"],
        "excluded_scene_ids": [],
        "placement_modes": {"inline": 1, "fullscreen": 1},
        "images": [
            _grounded_image_record(
                data["scenes"][0],
                local_path="news_images/image-01.png",
                payload=inline_payload,
                source="https://commons.example/nvidia",
                subject="NVIDIA",
                kind="logo",
                mode="inline",
            ),
            _grounded_image_record(
                data["scenes"][1],
                local_path="news_images/image-02.jpg",
                payload=fullscreen_payload,
                source="https://commons.example/falcon",
                subject="Falcon 9",
                kind="event",
                mode="fullscreen",
            ),
        ]
    }

    summary = news_images.attach_news_images(plans, data, manifest, tmp_path)

    assert summary == {"attached": 2, "placement_modes": {"inline": 1, "fullscreen": 1}}
    assert plans[0]["archetype"] != "news_image"
    assert plans[0]["news_image_src"] == "../news_images/image-01.png"
    assert plans[1]["archetype"] == "news_image"
    assert plans[1]["news_image_original_archetype"] == "topic"


@pytest.mark.parametrize(
    ("manifest_mutation", "image_mutation"),
    [
        ({"status": "partial"}, {}),
        ({"grounding_policy_version": 999}, {}),
        ({"requested_image_count": 1.9}, {}),
        ({}, {"sha256": "0" * 64}),
        ({}, {"license": "All rights reserved", "license_code": "copyright"}),
        ({}, {"title": "Google logo"}),
        ({}, {"match_terms": ["bogus"]}),
        ({}, {"display_mode": "sideways"}),
        ({}, {"local_path": "news_images/../outside.png"}),
    ],
)
def test_attach_news_images_rejects_forged_or_stale_assets(
    tmp_path: Path,
    manifest_mutation: dict,
    image_mutation: dict,
):
    data = {"title": "NVIDIA", "scenes": [board()["scenes"][0]]}
    plans = visual_plan.fallback_plan(data)
    asset_dir = tmp_path / "news_images"
    asset_dir.mkdir()
    payload = _image_bytes("PNG", "attach-negative")
    (asset_dir / "image-01.png").write_bytes(payload)
    image = _grounded_image_record(
        data["scenes"][0],
        local_path="news_images/image-01.png",
        payload=payload,
        source="https://commons.example/nvidia",
        subject="NVIDIA",
        kind="logo",
        mode="inline",
    )
    image.update(image_mutation)
    manifest = {
        "manifest_version": news_images.MANIFEST_VERSION,
        "query_semantics_version": news_images.QUERY_SEMANTICS_VERSION,
        "grounding_policy_version": news_images.QUERY_SEMANTICS_VERSION,
        "status": "ready",
        "storyboard_sha256": news_images.storyboard_fingerprint(data),
        "license_policy": "open_only",
        "requested_image_count": 1,
        "planned_image_count": 1,
        "eligible_scene_count": 1,
        "eligible_scene_ids": ["scene-01"],
        "excluded_scene_ids": [],
        "placement_modes": {"inline": 1, "fullscreen": 0},
        "images": [image],
        **manifest_mutation,
    }

    assert news_images.attach_news_images(plans, data, manifest, tmp_path)["attached"] == 0


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
        "requested_image_count": 1,
        "eligible_scene_count": 2,
        "eligible_scene_ids": ["scene-01", "scene-02"],
        "excluded_scene_ids": [],
        "planned_image_count": 1,
        "license_policy": "open_only",
        "placement_modes": {"inline": 1, "fullscreen": 0},
        "images": [
            _grounded_image_record(
                data["scenes"][0],
                local_path="news_images/image-01.png",
                payload=payload,
                source="https://commons.example/nvidia",
                subject="NVIDIA",
                kind="logo",
                mode="inline",
            )
        ],
    }
    (asset_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    assert (
        _cached_news_manifest(
            tmp_path,
            data,
            storyboard_sha256,
            contract_sha256,
            requested_count=1,
        )
        == manifest
    )
    assert _cached_news_manifest(tmp_path, data, "stale", contract_sha256, requested_count=1) is None
    assert _cached_news_manifest(tmp_path, data, storyboard_sha256, "stale", requested_count=1) is None

    manifest["manifest_version"] = news_images.MANIFEST_VERSION - 1
    (asset_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert (
        _cached_news_manifest(
            tmp_path,
            data,
            storyboard_sha256,
            contract_sha256,
            requested_count=1,
        )
        is None
    )


def test_cached_asset_rejects_all_raw_svg_encodings_and_active_content(
    tmp_path: Path,
):
    asset_dir = tmp_path / "news_images"
    asset_dir.mkdir()
    path = asset_dir / "image-01.svg"
    payloads = [
        b'<svg xmlns="http://www.w3.org/2000/svg"><rect width="10" height="10"/></svg>',
        b'<svg xmlns="http://www.w3.org/2000/svg" onload="alert(1)"><script>alert(1)</script></svg>',
        b'<svg xmlns="http://www.w3.org/2000/svg"><image href="https://attacker.example/x"/></svg>',
        (
            '<?xml version="1.0" encoding="UTF-16"?>'
            '<!DOCTYPE svg [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>'
            '<svg xmlns="http://www.w3.org/2000/svg">&xxe;</svg>'
        ).encode("utf-16"),
    ]

    for payload in payloads:
        path.write_bytes(payload)
        record = {
            "local_path": "news_images/image-01.svg",
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        assert news_images._cached_asset_is_intact(tmp_path, record) is False


def test_cached_asset_rejects_tiny_and_decompression_bomb_rasters(tmp_path: Path):
    asset_dir = tmp_path / "news_images"
    asset_dir.mkdir()
    path = asset_dir / "image-01.png"

    wordmark = _image_bytes("PNG", "wordmark", size=(656, 120))
    path.write_bytes(wordmark)
    wordmark_record = {
        "local_path": "news_images/image-01.png",
        "bytes": len(wordmark),
        "sha256": hashlib.sha256(wordmark).hexdigest(),
    }
    assert news_images._cached_asset_is_intact(
        tmp_path,
        {**wordmark_record, "kind": "logo"},
    ) is True
    assert news_images._cached_asset_is_intact(
        tmp_path,
        {**wordmark_record, "kind": "event"},
    ) is False

    tiny = io.BytesIO()
    Image.new("RGB", (1, 1), "red").save(tiny, format="PNG")
    tiny_payload = tiny.getvalue()
    path.write_bytes(tiny_payload)
    assert (
        news_images._cached_asset_is_intact(
            tmp_path,
            {
                "local_path": "news_images/image-01.png",
                "bytes": len(tiny_payload),
                "sha256": hashlib.sha256(tiny_payload).hexdigest(),
                "kind": "logo",
            },
        )
        is False
    )

    ihdr = struct.pack(">IIBBBBB", 20_000, 20_000, 8, 2, 0, 0, 0)
    bomb = (
        b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I", len(ihdr))
        + b"IHDR"
        + ihdr
        + struct.pack(">I", zlib.crc32(b"IHDR" + ihdr) & 0xFFFFFFFF)
        + b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    path.write_bytes(bomb)
    assert (
        news_images._cached_asset_is_intact(
            tmp_path,
            {
                "local_path": "news_images/image-01.png",
                "bytes": len(bomb),
                "sha256": hashlib.sha256(bomb).hexdigest(),
                "kind": "event",
            },
        )
        is False
    )


def test_license_gate_rejects_mixed_or_proprietary_markers():
    assert news_images._is_open_license("CC BY-SA 4.0") is True
    assert news_images._is_open_license("All rights reserved; CC BY", "copyright") is False
    assert news_images._is_open_license("Proprietary", "CC-BY") is False


@pytest.mark.asyncio
async def test_acquisition_rejects_symlinked_asset_directory_without_touching_external_files(
    tmp_path: Path,
):
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    victim = external / "image-01.jpg"
    victim.write_bytes(b"operator-owned external file")
    (task_dir / "news_images").symlink_to(external, target_is_directory=True)

    with pytest.raises(ValueError, match="symbolic links"):
        await news_images.acquire_news_images(board(), task_dir, count=1)

    assert victim.read_bytes() == b"operator-owned external file"
    assert not (external / "manifest.json").exists()
    assert news_images._generated_asset_path(task_dir, "news_images/image-01.jpg") is None


def test_cached_manifest_rejects_duplicate_identity_stale_policy_and_bad_hash(
    tmp_path: Path,
):
    data = board()
    asset_dir = tmp_path / "news_images"
    asset_dir.mkdir()
    payloads = [_image_bytes("PNG", "nvidia"), _image_bytes("PNG", "spacex")]
    for index, payload in enumerate(payloads, start=1):
        (asset_dir / f"image-{index:02d}.png").write_bytes(payload)
    fingerprint = news_images.storyboard_fingerprint(data)
    contract = news_images.acquisition_contract_fingerprint(
        storyboard_sha256=fingerprint,
        requested_count=2,
        target_count=2,
        eligible_scene_ids=["scene-01", "scene-02"],
        excluded_scene_ids=set(),
    )

    manifest = {
        "manifest_version": news_images.MANIFEST_VERSION,
        "query_semantics_version": news_images.QUERY_SEMANTICS_VERSION,
        "grounding_policy_version": news_images.QUERY_SEMANTICS_VERSION,
        "cache_contract_sha256": contract,
        "status": "ready",
        "storyboard_sha256": fingerprint,
        "requested_image_count": 2,
        "eligible_scene_count": 2,
        "eligible_scene_ids": ["scene-01", "scene-02"],
        "excluded_scene_ids": [],
        "planned_image_count": 2,
        "license_policy": "open_only",
        "placement_modes": {"inline": 1, "fullscreen": 1},
        "images": [
            _grounded_image_record(
                data["scenes"][0],
                local_path="news_images/image-01.png",
                payload=payloads[0],
                source="https://commons.example/nvidia",
                subject="NVIDIA",
                kind="logo",
                mode="inline",
            ),
            _grounded_image_record(
                data["scenes"][1],
                local_path="news_images/image-02.png",
                payload=payloads[1],
                source="https://commons.example/spacex",
                subject="Falcon 9",
                kind="event",
                mode="fullscreen",
            ),
        ],
    }

    baseline = json.loads(json.dumps(manifest))

    def write(value: dict | None = None) -> None:
        (asset_dir / "manifest.json").write_text(
            json.dumps(manifest if value is None else value),
            encoding="utf-8",
        )

    write()
    assert _cached_news_manifest(tmp_path, data, fingerprint, contract, requested_count=2) == manifest

    manifest["images"][1]["source_page_url"] = manifest["images"][0]["source_page_url"]
    write()
    assert _cached_news_manifest(tmp_path, data, fingerprint, contract, requested_count=2) is None
    manifest["images"][1]["source_page_url"] = "https://commons.example/spacex"
    manifest["images"][1]["scene_id"] = "scene-01"
    write()
    assert _cached_news_manifest(tmp_path, data, fingerprint, contract, requested_count=2) is None
    manifest["images"][1]["scene_id"] = "scene-02"
    manifest["query_semantics_version"] -= 1
    write()
    assert _cached_news_manifest(tmp_path, data, fingerprint, contract, requested_count=2) is None
    manifest["query_semantics_version"] = news_images.QUERY_SEMANTICS_VERSION
    manifest["images"][1]["sha256"] = "0" * 64
    write()
    assert _cached_news_manifest(tmp_path, data, fingerprint, contract, requested_count=2) is None

    forged = json.loads(json.dumps(baseline))
    forged["images"][0]["match_terms"] = ["bogus"]
    write(forged)
    assert _cached_news_manifest(tmp_path, data, fingerprint, contract, requested_count=2) is None

    forged = json.loads(json.dumps(baseline))
    forged["images"][0]["title"] = "Google logo"
    write(forged)
    assert _cached_news_manifest(tmp_path, data, fingerprint, contract, requested_count=2) is None

    forged = json.loads(json.dumps(baseline))
    forged["images"][0]["license"] = "All rights reserved"
    forged["images"][0]["license_code"] = "copyright"
    write(forged)
    assert _cached_news_manifest(tmp_path, data, fingerprint, contract, requested_count=2) is None

    forged = json.loads(json.dumps(baseline))
    forged["placement_modes"] = {"inline": 2, "fullscreen": 0}
    write(forged)
    assert _cached_news_manifest(tmp_path, data, fingerprint, contract, requested_count=2) is None

    forged = json.loads(json.dumps(baseline))
    forged["images"][0]["display_mode"] = "sideways"
    write(forged)
    assert _cached_news_manifest(tmp_path, data, fingerprint, contract, requested_count=2) is None

    forged = json.loads(json.dumps(baseline))
    forged["planned_image_count"] = 1
    forged["images"] = forged["images"][:1]
    forged["placement_modes"] = {"inline": 1, "fullscreen": 0}
    write(forged)
    assert _cached_news_manifest(tmp_path, data, fingerprint, contract, requested_count=2) is None

    forged = json.loads(json.dumps(baseline))
    forged["requested_image_count"] = 2.9
    write(forged)
    assert _cached_news_manifest(tmp_path, data, fingerprint, contract, requested_count=2) is None

    forged = json.loads(json.dumps(baseline))
    forged["images"][0]["local_path"] = "news_images/../outside.png"
    write(forged)
    assert _cached_news_manifest(tmp_path, data, fingerprint, contract, requested_count=2) is None

    forged = json.loads(json.dumps(baseline))
    forged["grounding_policy_version"] = 999
    write(forged)
    assert _cached_news_manifest(tmp_path, data, fingerprint, contract, requested_count=2) is None

    (asset_dir / "image-02.png").write_bytes(payloads[0])
    forged = json.loads(json.dumps(baseline))
    forged["images"][1]["bytes"] = len(payloads[0])
    forged["images"][1]["sha256"] = hashlib.sha256(payloads[0]).hexdigest()
    write(forged)
    assert _cached_news_manifest(tmp_path, data, fingerprint, contract, requested_count=2) is None


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


def test_acquisition_contract_binds_scene_order_exclusions_and_hints():
    common = {
        "storyboard_sha256": "story",
        "requested_count": 2,
        "target_count": 2,
        "eligible_scene_ids": ["scene-01", "scene-02"],
        "excluded_scene_ids": {"scene-03"},
        "scene_hints": {"scene-01": {"headline": "Vera"}},
    }
    baseline = news_images.acquisition_contract_fingerprint(**common)

    assert baseline != news_images.acquisition_contract_fingerprint(
        **{**common, "eligible_scene_ids": ["scene-02", "scene-01"]}
    )
    assert baseline != news_images.acquisition_contract_fingerprint(
        **{**common, "excluded_scene_ids": {"scene-04"}}
    )
    assert baseline != news_images.acquisition_contract_fingerprint(
        **{**common, "scene_hints": {"scene-01": {"headline": "Blackwell"}}}
    )


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
        payload = _image_bytes(seed=candidate["source_page_url"], size=(656, 120))
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
async def test_global_matching_moves_flexible_scene_to_reserve_source(
    tmp_path: Path,
    monkeypatch,
):
    data = {
        "title": "Matching",
        "scenes": [
            {"id": "scene-a", "text": "Alpha and Alpha Reserve launched.", "keywords": []},
            {"id": "scene-b", "text": "Beta launched.", "keywords": []},
        ],
    }
    primary = [
        {
            "scene_id": "scene-a",
            "search_query": "Alpha logo",
            "news_query": "Alpha",
            "expected_subject": "Alpha",
            "kind": "logo",
            "display_mode": "inline",
            "purpose": "",
            "caption": "Alpha",
        },
        {
            "scene_id": "scene-b",
            "search_query": "Beta logo",
            "news_query": "Beta",
            "expected_subject": "Beta",
            "kind": "logo",
            "display_mode": "fullscreen",
            "purpose": "",
            "caption": "Beta",
        },
    ]
    reserve = {
        **primary[0],
        "search_query": "Alpha Reserve logo",
        "expected_subject": "Alpha Reserve",
        "caption": "Alpha Reserve",
    }
    monkeypatch.setattr(
        news_images,
        "plan_news_images",
        AsyncMock(return_value=(primary, "mock-planner", "")),
    )
    monkeypatch.setattr(news_images, "_prepare_primary_plan", lambda plan, *_args, **_kwargs: plan)
    monkeypatch.setattr(news_images, "_reserve_plan", lambda *_args, **_kwargs: [reserve])
    monkeypatch.setattr(
        news_images,
        "research_references",
        AsyncMock(return_value=([], "mock-reference-search")),
    )

    async def fake_search(_client, *, shot: dict, limit: int = 30) -> list[dict]:
        del limit
        subject = shot["expected_subject"]
        candidate = _commons_candidate(shot["scene_id"], subject, "logo")
        slug = "shared-s" if subject in {"Alpha", "Beta"} else "reserve-t"
        candidate["source_page_url"] = f"https://commons.example/{slug}"
        candidate["download_url"] = f"https://upload.example/{slug}.jpg"
        return [candidate]

    async def fake_download(_client, *, candidate: dict, destination: Path):
        payload = _image_bytes(seed=candidate["source_page_url"])
        destination.write_bytes(payload)
        return len(payload), hashlib.sha256(payload).hexdigest()

    monkeypatch.setattr(news_images, "search_wikimedia_images", fake_search)
    monkeypatch.setattr(news_images, "_download_candidate", fake_download)

    manifest = await news_images.acquire_news_images(data, tmp_path, count=2)

    assert manifest["status"] == "ready"
    selected = {image["scene_id"]: image["source_page_url"] for image in manifest["images"]}
    assert selected == {
        "scene-a": "https://commons.example/reserve-t",
        "scene-b": "https://commons.example/shared-s",
    }


@pytest.mark.asyncio
async def test_download_failure_rolls_back_batch_and_globally_rematches(
    tmp_path: Path,
    monkeypatch,
):
    data = {
        "title": "Rematch",
        "scenes": [
            {"id": "scene-a", "text": "Alpha launched.", "keywords": []},
            {"id": "scene-b", "text": "Beta launched.", "keywords": []},
        ],
    }
    primary = [
        {
            "scene_id": scene_id,
            "search_query": f"{subject} logo",
            "news_query": subject,
            "expected_subject": subject,
            "kind": "logo",
            "display_mode": mode,
            "purpose": "",
            "caption": subject,
        }
        for scene_id, subject, mode in (
            ("scene-a", "Alpha", "inline"),
            ("scene-b", "Beta", "fullscreen"),
        )
    ]
    monkeypatch.setattr(
        news_images,
        "plan_news_images",
        AsyncMock(return_value=(primary, "mock-planner", "")),
    )
    monkeypatch.setattr(news_images, "_prepare_primary_plan", lambda plan, *_args, **_kwargs: plan)
    monkeypatch.setattr(news_images, "_reserve_plan", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        news_images,
        "research_references",
        AsyncMock(return_value=([{"url": "https://news.example", "_cookie": "private"}], "mock")),
    )

    def candidate(shot: dict, slug: str) -> dict:
        value = _commons_candidate(shot["scene_id"], shot["expected_subject"], "logo")
        value.update(
            {
                "source_page_url": f"https://commons.example/{slug}",
                "download_url": f"https://upload.example/{slug}.jpg",
                "_scene_text": "private",
                "api_key": "private",
            }
        )
        return value

    async def fake_search(_client, *, shot: dict, limit: int = 30) -> list[dict]:
        del limit
        if shot["scene_id"] == "scene-a":
            return [candidate(shot, "m-s"), candidate(shot, "z-u")]
        return [candidate(shot, "a-t"), candidate(shot, "m-s")]

    attempts: list[str] = []

    async def fake_download(_client, *, candidate: dict, destination: Path):
        slug = candidate["source_page_url"].rsplit("/", 1)[-1]
        attempts.append(slug)
        if slug == "a-t":
            raise RuntimeError("T download failed")
        payload = _image_bytes(seed=slug)
        destination.write_bytes(payload)
        return len(payload), hashlib.sha256(payload).hexdigest()

    monkeypatch.setattr(news_images, "search_wikimedia_images", fake_search)
    monkeypatch.setattr(news_images, "_download_candidate", fake_download)

    manifest = await news_images.acquire_news_images(data, tmp_path, count=2)

    assert manifest["status"] == "ready"
    assert attempts == ["m-s", "a-t", "z-u", "m-s"]
    selected = {image["scene_id"]: image["source_page_url"] for image in manifest["images"]}
    assert selected == {
        "scene-a": "https://commons.example/z-u",
        "scene-b": "https://commons.example/m-s",
    }
    serialized = json.dumps(manifest)
    assert "api_key" not in serialized
    assert "_scene_text" not in serialized
    assert "_cookie" not in serialized
    assert "private" not in serialized


@pytest.mark.asyncio
async def test_fresh_acquisition_rejects_duplicate_content_hashes(
    tmp_path: Path,
    monkeypatch,
):
    data = {
        "title": "Duplicate bytes",
        "scenes": [
            {"id": "scene-a", "text": "Alpha launched.", "keywords": []},
            {"id": "scene-b", "text": "Beta launched.", "keywords": []},
        ],
    }
    primary = [
        {
            "scene_id": scene_id,
            "search_query": f"{subject} logo",
            "news_query": subject,
            "expected_subject": subject,
            "kind": "logo",
            "display_mode": mode,
            "purpose": "",
            "caption": subject,
        }
        for scene_id, subject, mode in (
            ("scene-a", "Alpha", "inline"),
            ("scene-b", "Beta", "fullscreen"),
        )
    ]
    monkeypatch.setattr(
        news_images,
        "plan_news_images",
        AsyncMock(return_value=(primary, "mock-planner", "")),
    )
    monkeypatch.setattr(news_images, "_prepare_primary_plan", lambda plan, *_args, **_kwargs: plan)
    monkeypatch.setattr(news_images, "_reserve_plan", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        news_images,
        "research_references",
        AsyncMock(return_value=([], "mock-reference-search")),
    )

    async def fake_search(_client, *, shot: dict, limit: int = 30) -> list[dict]:
        del limit
        candidate = _commons_candidate(shot["scene_id"], shot["expected_subject"], "logo")
        candidate["source_page_url"] = f"https://commons.example/{shot['scene_id']}"
        return [candidate]

    async def fake_download(_client, *, candidate: dict, destination: Path):
        del candidate
        payload = _image_bytes(seed="identical-content")
        destination.write_bytes(payload)
        return len(payload), hashlib.sha256(payload).hexdigest()

    monkeypatch.setattr(news_images, "search_wikimedia_images", fake_search)
    monkeypatch.setattr(news_images, "_download_candidate", fake_download)

    manifest = await news_images.acquire_news_images(data, tmp_path, count=2)

    assert manifest["status"] == "partial"
    assert len(manifest["images"]) == 1
    assert any(error["stage"] == "duplicate_content" for error in manifest["errors"])
    assert len({image["sha256"] for image in manifest["images"]}) == len(manifest["images"])


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
        payload = _image_bytes(seed=candidate["source_page_url"])
        destination.write_bytes(payload)
        return len(payload), hashlib.sha256(payload).hexdigest()

    monkeypatch.setattr(news_images, "search_wikimedia_images", fake_search)
    monkeypatch.setattr(news_images, "_download_candidate", fake_download)
    old_asset = tmp_path / "news_images" / "image-99.jpg"
    old_asset.parent.mkdir(parents=True)
    stale_payload = b"stale partial asset"
    old_asset.write_bytes(stale_payload)
    replacement_victim = old_asset.parent / "image-88.jpg"
    replacement_victim.write_bytes(b"operator replacement")
    previous_pipeline_payload = b"previous pipeline bytes"
    (old_asset.parent / "manifest.json").write_text(
        json.dumps(
            {
                "manifest_version": 8,
                "status": "partial",
                "storyboard_sha256": news_images.storyboard_fingerprint(data),
                "images": [
                    {
                        "id": "image-99",
                        "local_path": "news_images/image-99.jpg",
                        "bytes": len(stale_payload),
                        "sha256": hashlib.sha256(stale_payload).hexdigest(),
                    },
                    {
                        "id": "image-88",
                        "local_path": "news_images/image-88.jpg",
                        "bytes": len(previous_pipeline_payload),
                        "sha256": hashlib.sha256(previous_pipeline_payload).hexdigest(),
                    },
                    {"local_path": "news_images/image-operator-original.jpg"},
                    {"local_path": "news_images/../../outside.jpg"},
                ],
            }
        ),
        encoding="utf-8",
    )
    operator_image = old_asset.parent / "image-operator-original.jpg"
    operator_image.write_bytes(b"operator source")
    unowned_generated = old_asset.parent / "image-77.jpg"
    unowned_generated.write_bytes(b"unowned source")
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
    assert replacement_victim.read_bytes() == b"operator replacement"
    assert operator_image.read_bytes() == b"operator source"
    assert unowned_generated.read_bytes() == b"unowned source"
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
        _cached_news_manifest(
            tmp_path,
            extended_board(),
            fingerprint,
            contract,
            requested_count=2,
        )
        is None
    )

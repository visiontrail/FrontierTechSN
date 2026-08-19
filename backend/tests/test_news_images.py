import json
from pathlib import Path

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
    }

    candidate = news_images._candidate_from_page(page, shot)

    assert candidate is not None
    assert candidate["download_url"].endswith("logo.png")
    assert candidate["license"] == "Public domain"
    assert news_images._extension_for(candidate) == ".png"

    page["imageinfo"][0]["extmetadata"]["LicenseShortName"]["value"] = "All rights reserved"
    assert news_images._candidate_from_page(page, shot) is None


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


def test_attach_news_images_preserves_inline_archetype_and_promotes_fullscreen(tmp_path: Path):
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


def test_cached_manifest_requires_same_storyboard_and_materialized_assets(tmp_path: Path):
    data = board()
    asset_dir = tmp_path / "news_images"
    asset_dir.mkdir()
    (asset_dir / "image-01.png").write_bytes(b"image")
    manifest = {
        "manifest_version": news_images.MANIFEST_VERSION,
        "status": "ready",
        "storyboard_sha256": news_images.storyboard_fingerprint(data),
        "planned_image_count": 1,
        "placement_modes": {"inline": 1, "fullscreen": 0},
        "images": [{"local_path": "news_images/image-01.png"}],
    }
    (asset_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    assert news_images._cached_manifest(tmp_path, manifest["storyboard_sha256"]) == manifest
    assert news_images._cached_manifest(tmp_path, "stale") is None


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

import asyncio
import json
from pathlib import Path

import pytest

from backend.pipeline import composer, digester, news_images, scene_kit, visual_plan


def board(n: int = 3) -> dict:
    return {
        "title": "Episode",
        "thesis": "A thesis sentence about the episode.",
        "scenes": [
            {
                "id": f"scene-{i + 1:02d}",
                "index": i,
                "start": 5.0 + i * 10,
                "duration": 10.0,
                "text": f"Sentence {i} opens the scene. It continues with more detail after that.",
                "keywords": ["sunflowers", "paris"],
                "lines": [],
            }
            for i in range(n)
        ],
    }


def test_fallback_plan_covers_every_scene_with_usable_direction():
    plans = visual_plan.fallback_plan(board(4))
    assert [p["id"] for p in plans] == ["scene-01", "scene-02", "scene-03", "scene-04"]
    for plan in plans:
        assert plan["headline"]
        assert plan["accent"] in scene_kit.ACCENTS
        assert plan["motif"] in scene_kit.MOTIFS


def test_fallback_headlines_end_on_a_clause_or_an_ellipsis():
    long_text = (
        "You haven't really been somewhere until you go back and realize it has become "
        "a completely different place while you were not looking."
    )
    headline = visual_plan._headline_from(long_text)
    assert len(headline) <= 68
    # Never leave the reader mid-thought without a signal.
    assert headline.endswith("…") or not long_text.startswith(headline + " and")


def test_consecutive_fallback_scenes_do_not_share_an_accent():
    plans = visual_plan.fallback_plan(board(6))
    accents = [p["accent"] for p in plans]
    assert all(a != b for a, b in zip(accents, accents[1:]))


def test_json_array_is_recovered_from_prose_wrapped_replies():
    reply = 'Here is the plan:\n```json\n[{"id": "scene-01"}]\n```\nHope that helps.'
    assert visual_plan._first_json_array(reply) == [{"id": "scene-01"}]

    bare = 'Thinking... [{"id": "scene-02"}] done'
    assert visual_plan._first_json_array(bare) == [{"id": "scene-02"}]

    wrapped = json.dumps({"scenes": [{"id": "scene-03"}]})
    assert visual_plan._first_json_array(wrapped) == [{"id": "scene-03"}]

    assert visual_plan._first_json_array("no json at all") is None


def test_normalise_rejects_spine_owned_archetypes_and_bad_enums():
    scene = board(1)["scenes"][0]
    plan = visual_plan._normalise(
        {"archetype": "title", "accent": "chartreuse", "motif": "spirograph"}, scene, 0
    )
    # title/outro/footage belong to the spine and the footage matcher.
    assert plan["archetype"] == "topic"
    assert plan["accent"] in scene_kit.ACCENTS
    assert plan["motif"] in scene_kit.MOTIFS


def test_normalise_fills_missing_copy_from_the_narration():
    scene = board(1)["scenes"][0]
    plan = visual_plan._normalise({}, scene, 0)
    assert plan["headline"]
    assert plan["kicker"]


def test_missing_keywords_do_not_create_a_generic_chapter_label():
    scene = board(1)["scenes"][0]
    scene["keywords"] = []

    assert visual_plan._normalise({}, scene, 0)["kicker"] == ""
    assert visual_plan.fallback_plan({"scenes": [scene]})[0]["kicker"] == ""


def test_normalise_clamps_copy_to_what_fits_a_frame():
    scene = board(1)["scenes"][0]
    items = [
        "Summarize annual reports across multiple documents",
        "Search and compare company operational data online",
        "Control a desktop browser for retrieval and document generation",
        "Build English-language presentations from data",
        "Generate marketing posters from reference images",
        "This sixth item must not reach the renderer",
    ]
    plan = visual_plan._normalise(
        {
            "headline": "h" * 400,
            "body": "b" * 900,
            "kicker": "k" * 90,
            "items": items,
        },
        scene,
        0,
    )
    assert len(plan["headline"]) <= 110
    assert len(plan["body"]) <= 260
    assert len(plan["kicker"]) <= 30
    assert plan["items"] == items[: scene_kit.MAX_NARRATIVE_ITEMS]
    assert plan["items"][-1] == "Generate marketing posters from reference images"


def test_normalise_restores_exact_scaled_number_precision_from_narration():
    scene = board(1)["scenes"][0]
    scene["text"] = (
        "Perfect World's 2026 semiannual report recorded first-half revenue of "
        "2.751 billion yuan and a net loss of 118 million yuan."
    )

    plan = visual_plan._normalise(
        {
            "archetype": "stat",
            "headline": "Perfect World's first-half results",
            "stat": "¥2.75B revenue · ¥118M loss",
            "stat_label": "First-half 2026",
        },
        scene,
        0,
    )

    assert plan["stat"] == "¥2.751B revenue · ¥118M loss"


def test_scaled_number_precision_restoration_fails_closed_when_ambiguous():
    narration = "The estimates were 2.751 billion and 2.749 billion yuan."

    assert (
        visual_plan._restore_scaled_number_precision("About 2.75B", narration)
        == "About 2.75B"
    )


def test_footage_is_attached_to_the_scene_whose_keywords_match(tmp_path: Path):
    data = board(2)
    data["scenes"][0]["keywords"] = ["hotpot", "sichuan"]
    data["scenes"][1]["keywords"] = ["sunflowers", "fields"]
    plans = visual_plan.fallback_plan(data)

    clip_dir = tmp_path / "footage"
    clip_dir.mkdir()
    (clip_dir / "field.jpg").write_bytes(b"x")
    manifest = {
        "clips": [
            {
                "local_path": "footage/field.jpg",
                "query": "sunflower fields countryside",
                "title": "Sunflowers",
                "attribution": "CC BY-SA",
            }
        ]
    }

    assert visual_plan.attach_footage(plans, data, manifest, tmp_path) == 1
    assert plans[1]["archetype"] == "footage"
    assert plans[1]["footage_src"] == "footage/field.jpg"
    assert plans[1]["footage_kind"] == "image"
    assert plans[0]["archetype"] != "footage"


def test_external_video_credit_uses_creator_and_title_not_acquisition_tool(
    tmp_path: Path,
):
    data = board(1)
    data["scenes"][0]["keywords"] = ["hannibal", "carthage"]
    plans = visual_plan.fallback_plan(data)
    (tmp_path / "footage").mkdir()
    (tmp_path / "footage" / "hannibal.mp4").write_bytes(b"x")
    manifest = {
        "clips": [
            {
                "local_path": "footage/hannibal.mp4",
                "query": "hannibal carthage",
                "title": "Hannibal's Greatest Victory",
                "creator": "Kings and Generals",
                "provider": "YouTube via yt-dlp",
                "platform": "youtube",
                "review_required": True,
            }
        ]
    }

    assert visual_plan.attach_footage(plans, data, manifest, tmp_path) == 1
    assert plans[0]["footage_credit"] == (
        "Source: Kings and Generals · Hannibal's Greatest Victory"
    )
    assert "yt-dlp" not in plans[0]["footage_credit"]


def test_open_license_credit_remains_unchanged(tmp_path: Path):
    data = board(1)
    plans = visual_plan.fallback_plan(data)
    (tmp_path / "footage").mkdir()
    (tmp_path / "footage" / "field.jpg").write_bytes(b"x")
    manifest = {
        "clips": [
            {
                "local_path": "footage/field.jpg",
                "query": "sunflowers paris",
                "attribution": "Vincent Archive · CC BY-SA 4.0",
            }
        ]
    }

    assert visual_plan.attach_footage(plans, data, manifest, tmp_path) == 1
    assert plans[0]["footage_credit"] == "Vincent Archive · CC BY-SA 4.0"


def test_footage_with_no_keyword_overlap_is_not_forced_onto_a_scene(tmp_path: Path):
    data = board(1)
    plans = visual_plan.fallback_plan(data)
    (tmp_path / "footage").mkdir()
    (tmp_path / "footage" / "x.webm").write_bytes(b"x")
    manifest = {"clips": [{"local_path": "footage/x.webm", "query": "unrelated subject matter"}]}
    assert visual_plan.attach_footage(plans, data, manifest, tmp_path) == 0
    assert plans[0]["archetype"] != "footage"


def test_missing_clip_files_are_skipped(tmp_path: Path):
    data = board(1)
    plans = visual_plan.fallback_plan(data)
    manifest = {"clips": [{"local_path": "footage/gone.jpg", "query": "sunflowers paris"}]}
    assert visual_plan.attach_footage(plans, data, manifest, tmp_path) == 0


def test_low_confidence_or_weakly_grounded_footage_is_not_attached(tmp_path: Path):
    data = board(2)
    data["scenes"][0]["text"] = "Central banks are hoarding reserves in secure vaults."
    data["scenes"][1]["text"] = "A supernova forged the gold inside neutron stars."
    plans = visual_plan.fallback_plan(data)
    (tmp_path / "footage").mkdir()
    (tmp_path / "footage" / "bad.mp4").write_bytes(b"x")
    manifest = {
        "clips": [
            {
                "local_path": "footage/bad.mp4",
                "query": "gold market",
                "purpose": "generic gold imagery",
                "analysis": {
                    "confidence": 0.5,
                    "reason": "Talking-head filler with no visual depiction of the requested subject.",
                },
            }
        ]
    }

    assert visual_plan.attach_footage(plans, data, manifest, tmp_path) == 0


def test_footage_uses_purpose_and_two_distinctive_narration_terms(tmp_path: Path):
    data = board(2)
    data["scenes"][0]["text"] = "Gold prices rose, and fear spread through markets."
    data["scenes"][1]["text"] = "Central banks accumulated reserves and hoarded bullion."
    data["scenes"][0]["keywords"] = ["gold", "fear"]
    data["scenes"][1]["keywords"] = ["central", "banks", "reserves", "bullion"]
    plans = visual_plan.fallback_plan(data)
    (tmp_path / "footage").mkdir()
    (tmp_path / "footage" / "vault.mp4").write_bytes(b"x")
    manifest = {
        "clips": [
            {
                "local_path": "footage/vault.mp4",
                "query": "gold bars vault",
                "purpose": "central bank reserves and bullion hoarding",
                "title": "Inside a gold vault",
                "analysis": {"confidence": 0.9, "reason": "Bullion bars in a central bank vault."},
            }
        ]
    }

    assert visual_plan.attach_footage(plans, data, manifest, tmp_path) == 1
    assert plans[1]["archetype"] == "footage"
    assert len(plans[1]["footage_match_terms"]) >= 2


def test_footage_script_excerpt_prevents_query_based_reassignment(tmp_path: Path):
    data = board(2)
    data["scenes"][0]["text"] = (
        "Central banks say fiat money does not need gold backing."
    )
    data["scenes"][1]["text"] = (
        "Gold survived every empire, every war, and every currency collapse in history."
    )
    plans = visual_plan.fallback_plan(data)
    (tmp_path / "footage").mkdir()
    (tmp_path / "footage" / "history.mp4").write_bytes(b"x")
    manifest = {
        "clips": [
            {
                "local_path": "footage/history.mp4",
                "query": "fiat money gold backing",
                "purpose": "paper currency history",
                "script_excerpt": (
                    "They choose the thing that survived every empire, every war, "
                    "and every currency collapse in human history."
                ),
                "analysis": {
                    "confidence": 0.92,
                    "reason": "Historical commodity trading across ancient civilizations.",
                },
            }
        ]
    }

    assert visual_plan.attach_footage(plans, data, manifest, tmp_path) == 1
    assert plans[0]["archetype"] != "footage"
    assert plans[1]["archetype"] == "footage"
    assert len(plans[1]["footage_script_match_terms"]) >= 3


def test_fallback_analyzed_excerpt_clips_are_all_placed_before_generic_results(
    tmp_path: Path,
):
    data = board(4)
    excerpts = [
        "Copper gears reveal the hidden harbor mechanism.",
        "Paper wings cross the midnight carrier deck.",
        "Factory sparks ignite an industrial furnace.",
    ]
    for index, scene in enumerate(data["scenes"]):
        scene["keywords"] = ["shared", f"beat-{index}"]
        if index < len(excerpts):
            scene["text"] = excerpts[index]
    plans = visual_plan.fallback_plan(data)
    footage_dir = tmp_path / "footage"
    footage_dir.mkdir()
    clips = [
        {
            "local_path": "footage/generic.mp4",
            "query": "shared scene",
            "title": "Generic scene",
        }
    ]
    (footage_dir / "generic.mp4").write_bytes(b"x")
    for index, excerpt in enumerate(excerpts, start=1):
        name = f"selected-{index}.mp4"
        (footage_dir / name).write_bytes(b"x")
        clips.append(
            {
                "local_path": f"footage/{name}",
                "query": f"shared beat-{index - 1}",
                "script_excerpt": excerpt,
                "analysis": {
                    "confidence": 0.25,
                    "status": "fallback",
                    "reason": "visual analyzer unavailable",
                },
            }
        )

    assert visual_plan.attach_footage(plans, data, {"clips": clips}, tmp_path) == 3
    placed = [plan for plan in plans if plan["archetype"] == "footage"]
    assert [plan["id"] for plan in placed] == ["scene-01", "scene-02", "scene-03"]
    assert all(plan["footage_confidence"] == 0.65 for plan in placed)
    assert all(plan["footage_analysis_confidence"] == 0.25 for plan in placed)


def test_visual_grounding_report_requires_every_scene_and_grounded_footage():
    data = board(1)
    plans = visual_plan.fallback_plan(data)
    report = visual_plan.visual_grounding_report(plans, data)
    assert report["passed"] is True

    plans[0].update(
        {
            "archetype": "footage",
            "footage_src": "footage/x.mp4",
            "footage_match_terms": ["sunflower"],
            "footage_confidence": 0.9,
        }
    )
    report = visual_plan.visual_grounding_report(plans, data)
    assert report["passed"] is False


def test_visual_grounding_report_accepts_verified_collage_on_requested_scene():
    data = board(1)
    plans = visual_plan.fallback_plan(data)
    plans[0].update(
        {
            "archetype": "footage",
            "footage_src": "../collage_broll/scene-01/final.mp4",
            "collage_broll": True,
            "collage_source_scene_id": "scene-01",
            "collage_placed_scene_id": "scene-01",
            "collage_qa": {"passed": True},
        }
    )

    report = visual_plan.visual_grounding_report(plans, data)

    assert report["passed"] is True
    assert "paper-collage passed" in report["scenes"][0]["reason"]


def test_visual_grounding_report_rejects_collage_moved_to_another_scene():
    data = board(1)
    plans = visual_plan.fallback_plan(data)
    plans[0].update(
        {
            "archetype": "footage",
            "footage_src": "../collage_broll/scene-02/final.mp4",
            "collage_broll": True,
            "collage_source_scene_id": "scene-02",
            "collage_placed_scene_id": "scene-01",
            "collage_qa": {"passed": True},
        }
    )

    report = visual_plan.visual_grounding_report(plans, data)

    assert report["passed"] is False


def test_visual_grounding_report_accepts_exact_licensed_news_image():
    data = board(1)
    data["scenes"][0]["text"] = "NVIDIA announced the Vera CPU."
    plans = visual_plan.fallback_plan(data)
    evidence = news_images._grounding_evidence(
        {"expected_subject": "NVIDIA", "kind": "logo"},
        data["scenes"][0],
        {"title": "NVIDIA logo", "description": "Official NVIDIA logo"},
    )
    plans[0].update(
        {
            "news_image": True,
            "news_image_source_scene_id": "scene-01",
            "news_image_expected_subject": "NVIDIA",
            "news_image_caption": "NVIDIA",
            "news_image_kind": "logo",
            "news_image_title": "NVIDIA logo",
            "news_image_description": "Official NVIDIA logo",
            "news_image_categories": "",
            "news_image_object_name": "",
            "news_image_creator": "",
            "news_image_match_terms": evidence["grounding_distinctive_anchors"],
            "news_image_grounding_policy_version": news_images.QUERY_SEMANTICS_VERSION,
            "news_image_grounding_passed": True,
            "news_image_grounding_distinctive_anchors": evidence["grounding_distinctive_anchors"],
            "news_image_grounding_identity_field": evidence["grounding_identity_field"],
            "news_image_grounding_identity_phrase": evidence["grounding_identity_phrase"],
            "news_image_grounding_identity_field_terms": evidence["grounding_identity_field_terms"],
            "news_image_grounding_context_conflicts": evidence["grounding_context_conflicts"],
            "news_image_license": "Public domain",
            "news_image_license_code": "Public domain",
            "news_image_source_mime_type": "image/png",
        }
    )

    report = visual_plan.visual_grounding_report(plans, data)

    assert report["passed"] is True
    assert "licensed news image" in report["scenes"][0]["reason"]

    plans[0]["news_image_caption"] = "Five logo"
    tampered = visual_plan.visual_grounding_report(plans, data)
    assert tampered["passed"] is False
    assert "lacked a current exact-scene grounding proof" in tampered["scenes"][0]["reason"]


def _verified_news_plan() -> tuple[dict, list[dict]]:
    data = board(1)
    data["scenes"][0]["text"] = "NVIDIA announced the Vera CPU."
    plans = visual_plan.fallback_plan(data)
    candidate = {
        "title": "NVIDIA logo",
        "description": "Official NVIDIA logo",
        "categories": "NVIDIA|Technology company logos",
        "object_name": "NVIDIA logo",
        "creator": "NVIDIA Corporation",
    }
    evidence = news_images._grounding_evidence(
        {"expected_subject": "NVIDIA", "kind": "logo"},
        data["scenes"][0],
        candidate,
    )
    plans[0].update(
        {
            "news_image": True,
            "news_image_source_scene_id": "scene-01",
            "news_image_expected_subject": "NVIDIA",
            "news_image_caption": "NVIDIA",
            "news_image_kind": "logo",
            "news_image_title": "NVIDIA logo",
            "news_image_description": "Official NVIDIA logo",
            "news_image_categories": candidate["categories"],
            "news_image_object_name": candidate["object_name"],
            "news_image_creator": candidate["creator"],
            "news_image_match_terms": evidence["grounding_distinctive_anchors"],
            "news_image_grounding_policy_version": news_images.QUERY_SEMANTICS_VERSION,
            "news_image_grounding_passed": True,
            "news_image_grounding_distinctive_anchors": evidence["grounding_distinctive_anchors"],
            "news_image_grounding_identity_field": evidence["grounding_identity_field"],
            "news_image_grounding_identity_phrase": evidence["grounding_identity_phrase"],
            "news_image_grounding_identity_field_terms": evidence["grounding_identity_field_terms"],
            "news_image_grounding_context_conflicts": evidence["grounding_context_conflicts"],
            "news_image_license": "Public domain",
            "news_image_license_code": "Public domain",
        }
    )
    return data, plans


def test_visual_grounding_report_rejects_category_only_media_identity_conflict():
    data, plans = _verified_news_plan()
    plans[0]["news_image_categories"] = "A Perfect World (film)|Movie logos"

    report = visual_plan.visual_grounding_report(plans, data)

    assert report["passed"] is False
    assert "current exact-scene grounding proof" in report["scenes"][0]["reason"]


def test_visual_grounding_report_requires_complete_v4_context_contract():
    required_context_keys = (
        "news_image_categories",
        "news_image_object_name",
        "news_image_creator",
        "news_image_grounding_context_conflicts",
    )

    for missing_key in required_context_keys:
        data, plans = _verified_news_plan()
        del plans[0][missing_key]

        report = visual_plan.visual_grounding_report(plans, data)

        assert report["passed"] is False, missing_key


def test_visual_grounding_report_recomputes_news_image_policy_and_license():
    mutations = [
        {"news_image_grounding_policy_version": 999},
        {"news_image_license": "All rights reserved", "news_image_license_code": "copyright"},
        {"news_image_match_terms": ["bogus"]},
        {"news_image_grounding_identity_phrase": "bogus"},
        {"news_image_title": "Google logo"},
        {"news_image_source_scene_id": "scene-99"},
    ]

    for mutation in mutations:
        data, plans = _verified_news_plan()
        plans[0].update(mutation)
        report = visual_plan.visual_grounding_report(plans, data)
        assert report["passed"] is False, mutation


def test_visual_grounding_report_rejects_legacy_news_image_without_policy_proof():
    data = board(1)
    data["scenes"][0]["text"] = "NVIDIA announced the Vera CPU."
    plans = visual_plan.fallback_plan(data)
    plans[0].update(
        {
            "news_image": True,
            "news_image_source_scene_id": "scene-01",
            "news_image_match_terms": ["nvidia"],
            "news_image_license": "Public domain",
        }
    )

    report = visual_plan.visual_grounding_report(plans, data)

    assert report["passed"] is False
    assert "current exact-scene grounding proof" in report["scenes"][0]["reason"]


def test_outro_plan_is_spine_owned():
    data = board(1)
    outro = visual_plan.outro_plan(data)
    assert outro["archetype"] == "outro"
    assert outro["body"] == ""


def test_visual_plan_payload_does_not_expose_the_working_title_as_scene_copy():
    data = board(1)
    data["title"] = "VIDEO 042"
    payload = json.loads(visual_plan._batch_prompt_payload(data, data["scenes"]))

    assert "episode_title" not in payload
    assert "VIDEO 042" not in json.dumps(payload)


def test_visual_planner_does_not_load_unrelated_project_skills(monkeypatch):
    observed: dict = {}

    async def fake_resolve_provider(*_args, **_kwargs):
        return "http://provider.test", "model", "key"

    async def fake_chat(*_args, **kwargs):
        observed.update(kwargs)
        return "[]"

    monkeypatch.setattr(digester, "_resolve_provider", fake_resolve_provider)
    monkeypatch.setattr(digester, "_chat", fake_chat)

    plans = asyncio.run(visual_plan.plan_scene_visuals(board(1)))

    assert observed["enable_skills"] is False
    assert observed["disable_thinking"] is True
    assert observed["max_tokens"] == visual_plan.VISUAL_PLAN_MAX_TOKENS
    assert len(plans) == 1


@pytest.mark.parametrize("nonfinite", ["NaN", "Infinity", "1e999"])
def test_visual_planner_drops_unknown_nonfinite_and_arbitrary_fields(
    tmp_path,
    monkeypatch,
    nonfinite,
):
    async def fake_resolve_provider(*_args, **_kwargs):
        return "http://provider.test", "model", "key"

    async def fake_chat(*_args, **_kwargs):
        return (
            '[{"id":"scene-01","archetype":"topic",'
            '"headline":"Sentence 0 opens the scene",'
            '"body":"It continues with more detail",'
            f'"debug":{nonfinite},'
            '"arbitrary_extra":{"value":"discard me"},'
            '"collage_broll":true,"news_image":true,'
            '"footage_src":"../untrusted-runtime.mp4",'
            '"left":{"label":"Before","text":"Sentence 0",'
            f'"debug":{nonfinite}'
            "}}]"
        )

    monkeypatch.setattr(digester, "_resolve_provider", fake_resolve_provider)
    monkeypatch.setattr(digester, "_chat", fake_chat)
    data = board(1)

    plans = asyncio.run(visual_plan.plan_scene_visuals(data))

    assert plans[0]["grounding_source"] == "model"
    assert "debug" not in plans[0]
    assert "arbitrary_extra" not in plans[0]
    assert "collage_broll" not in plans[0]
    assert "news_image" not in plans[0]
    assert "footage_src" not in plans[0]
    assert "debug" not in plans[0]["left"]
    composer._write_visual_plan_checkpoint(
        tmp_path,
        data,
        [*plans, visual_plan.outro_plan(data)],
    )
    encoded = (tmp_path / "visual_plan.json").read_text(encoding="utf-8")
    assert nonfinite not in encoded
    assert "arbitrary_extra" not in encoded
    assert composer._load_cached_scene_plans(tmp_path, data) is not None


@pytest.mark.parametrize(
    "invalid_fields",
    [
        '"headline":NaN',
        '"items":[Infinity]',
        '"items":5',
        '"left":{"label":1e999,"text":"Before"}',
    ],
)
def test_visual_planner_falls_back_for_known_nonfinite_fields_and_checkpoints(
    tmp_path,
    monkeypatch,
    invalid_fields,
):
    async def fake_resolve_provider(*_args, **_kwargs):
        return "http://provider.test", "model", "key"

    async def fake_chat(*_args, **_kwargs):
        return (
            f'[{{"id":"scene-01",{invalid_fields}}},'
            '{"id":"scene-02","headline":"Sentence 1 opens the scene"}]'
        )

    monkeypatch.setattr(digester, "_resolve_provider", fake_resolve_provider)
    monkeypatch.setattr(digester, "_chat", fake_chat)
    data = board(2)

    plans = asyncio.run(visual_plan.plan_scene_visuals(data))

    assert plans[0]["grounding_source"] == "narration_fallback"
    assert plans[1]["grounding_source"] == "model"
    composer._write_visual_plan_checkpoint(
        tmp_path,
        data,
        [*plans, visual_plan.outro_plan(data)],
    )
    assert composer._load_cached_scene_plans(tmp_path, data) == plans

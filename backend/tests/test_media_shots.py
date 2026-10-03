import asyncio
import hashlib
import json
import subprocess
from unittest.mock import AsyncMock, patch

import pytest

from backend.pipeline import media_shots as ms, scene_kit, composer
from backend.pipeline.video_format import LANDSCAPE, PORTRAIT
from PIL import Image


def test_reviewed_wrong_media_is_replaced_by_complete_editorial_shots(tmp_path):
    (tmp_path / "compositions").mkdir()
    asset = tmp_path / "historical.jpg"
    Image.new("RGB", (160, 90)).save(asset)
    rejection = {"src": "../historical.jpg", "sha256": hashlib.sha256(asset.read_bytes()).hexdigest(),
                 "rendered_video_sha256": "reviewed-candidate", "issues": ["Wrong event"]}
    plan = {"id": "scene-02", "headline": "Current event", "archetype": "news_image",
            "news_image_src": "historical.jpg", "review_rejected_media": [rejection]}
    scene = {"id": "scene-02", "duration": 19.7, "text": "The current event has been reported."}
    assert ms.inventory(plan, tmp_path) == []
    assert ms.inventory({**plan, "review_rejected_media": []}, tmp_path)
    # Even a provider outage must not bring the rejected required asset back.
    with patch("backend.pipeline.digester._resolve_provider", AsyncMock(side_effect=RuntimeError("offline"))):
        report = asyncio.run(ms.plan_media_shots([plan], {"scenes": [scene]}, tmp_path))
    assert report["scenes"][0]["review_rejected_media"] == [rejection]
    assert sum(ms.ticks(shot["duration"]) for shot in plan["media_shots"]) == 1970
    assert all(shot["kind"] == "editorial" for shot in plan["media_shots"])
    kit = scene_kit.ScenePlan.from_dict(plan, duration=19.7, scene_id="scene-02")
    html = scene_kit.render_scene(kit)
    output = tmp_path / "compositions" / "scene-02.html"
    output.write_text(html)
    composer._assert_locked_visual_assets(tmp_path, [plan])
    assert "historical.jpg" not in html
    output.write_text(html + '<img src="../historical.jpg">')
    with pytest.raises(RuntimeError, match="rejected media still rendered"):
        composer._assert_locked_visual_assets(tmp_path, [plan])
    # A replacement at the same path is new evidence, not permanently banned.
    Image.new("RGB", (160, 90), color="red").save(asset)
    assert ms.inventory(plan, tmp_path)
    assert ms.review_rejected_sources(plan, tmp_path) == {}


def test_final_footage_audit_distinguishes_review_rejection_from_delivery(tmp_path):
    asset = tmp_path / "wrong.mp4"
    asset.write_bytes(b"historical event")
    plan = {"id": "scene-02", "archetype": "footage", "footage_src": "wrong.mp4",
            "media_shots": [{"kind": "editorial", "duration": 10}],
            "review_rejected_media": [{"src": "../wrong.mp4",
                "sha256": hashlib.sha256(asset.read_bytes()).hexdigest(),
                "rendered_video_sha256": "candidate", "issues": ["Wrong event"]}]}
    manifest = {"requested_clip_count": 1, "clips": [{"id": "clip-01", "path": str(asset)}]}
    composer._require_public_footage_placement([plan], manifest, tmp_path, rendered_shots=True)
    report = json.loads((tmp_path / "footage-placement.json").read_text())
    assert report["passed"] and report["placed"] == 0 and report["review_rejected"] == 1
    assert report["clips"][0]["review_rejected"][0]["scene_id"] == "scene-02"
    plan["media_shots"].append({"kind": "public_footage", "src": "../wrong.mp4"})
    with pytest.raises(RuntimeError, match="delivery blocked"):
        composer._require_public_footage_placement([plan], manifest, tmp_path, rendered_shots=True)
    plan["media_shots"].pop()
    asset.write_bytes(b"changed bytes")
    with pytest.raises(RuntimeError, match="delivery blocked"):
        composer._require_public_footage_placement([plan], manifest, tmp_path, rendered_shots=True)


@pytest.mark.parametrize("size,orientation,expected", [
    ((1821, 3221), 1, "contain"), ((3221, 1821), 6, "contain"),
    ((1920, 1080), 1, "cover"),
])
def test_image_inventory_measures_pixels_and_preserves_portrait_fit(tmp_path, size, orientation, expected):
    exif = Image.Exif()
    exif[274] = orientation
    Image.new("RGB", size).save(tmp_path / "photo.jpg", exif=exif)
    assets = ms.inventory({"news_image_src": "photo.jpg", "news_image_fit": "cover"}, tmp_path)
    assert assets[0]["fit"] == expected
    assert (assets[0]["height"] > assets[0]["width"]) == (expected == "contain")
    scene = {"id": "scene-02", "text": "An event photo.", "duration": 8}
    shots = ms.validate({"shots": [{"asset_id": "asset-1", "duration": 8,
        "copy_ids": ["copy-1"], "script_excerpt": scene["text"], "fit": "cover"}]},
        scene, assets, [{"id": "copy-1", "text": "An event photo"}])
    assert shots[0]["fit"] == expected  # The model cannot override measured layout.
    plan = {"id": scene["id"], "media_shots": shots}
    html = scene_kit.render_scene(scene_kit.ScenePlan.from_dict(plan, duration=8, scene_id=scene["id"]))
    ms.assert_rendered_shots(plan, html)
    media_tween = next(line for line in html.splitlines() if 'inAt("#scene-02-shot-1-media"' in line)
    if expected == "contain":
        assert 'class="clip shot-media image contained contained-photo"' in html
        assert 'class="clip shot-panel visual contained-photo"' in html
        assert "scale" not in media_tween
        with pytest.raises(ValueError, match="containment changed"):
            ms.assert_rendered_shots(plan, html.replace('style="object-fit:contain"', 'style="object-fit:cover"'))
    else:
        assert 'class="clip shot-media image"' in html


def test_logo_fit_survives_mixed_shot_inventory(tmp_path):
    Image.new("RGB", (432, 212)).save(tmp_path / "logo.png")
    assert ms.inventory({"news_image_src": "logo.png", "news_image_fit": "contain"}, tmp_path)[0]["fit"] == "contain"


def test_editorial_copy_preserves_initials_titles_and_decimals():
    text = "Gen. Joshua M. Rudd leads the agency. GLM-6.0 follows U.S. research. Another sentence."
    words = ms.editorial_copy({}, {"text": text})
    assert [word["text"] for word in words] == [
        "Gen. Joshua M. Rudd leads the agency.",
        "GLM-6.0 follows U.S. research.", "Another sentence.",
    ]


def test_decimal_cuts_do_not_create_same_track_float_overlaps():
    scene = {"id": "scene-02", "duration": 70.26, "text": "Grounded news."}
    copy = [{"id": "copy-1", "text": "Grounded news"}]
    raw = {"shots": [{"asset_id": "editorial", "duration": duration,
                      "copy_ids": ["copy-1"], "script_excerpt": scene["text"]}
                     for duration in [53.6, 8.66, 8.0]]}
    plan = {"id": scene["id"], "headline": "Grounded news",
            "media_shots": ms.validate(raw, scene, [], copy)}
    kit = scene_kit.ScenePlan.from_dict(plan, duration=70.26, scene_id=scene["id"])
    html = scene_kit.render_scene(kit)
    ms.assert_rendered_shots(plan, html)
    tracks = {}
    for _, attrs in ms._Elements(html).elements.values():
        if "data-track-index" in attrs:
            tracks.setdefault(attrs["data-track-index"], []).append(
                (float(attrs["data-start"]), float(attrs["data-duration"])))
    for intervals in tracks.values():
        ordered = sorted(intervals)
        assert all(start + duration <= next_start
                   for (start, duration), (next_start, _) in zip(ordered, ordered[1:]))


def context():
    scene = {"id": "scene-02", "duration": 26.5, "text": "The agency has five priorities. AI and cybersecurity are among them."}
    assets = [
        {"id": "asset-1", "kind": "public_footage", "src": "../footage/hearing.mp4", "duration": 15.01, "credit": "Hearing"},
        {"id": "asset-2", "kind": "paper_collage", "src": "../collage_broll/agency.mp4", "duration": 8.0},
    ]
    copy = [{"id": "copy-1", "text": "Five priorities"}, {"id": "copy-2", "text": "Artificial intelligence"}]
    return scene, assets, copy


def raw_plan(order=("asset-2", "asset-1", "editorial")):
    duration = {"asset-1": 15.01, "asset-2": 8, "editorial": 3.49}
    return {"id": "scene-02", "shots": [
        {"asset_id": key, "duration": duration[key], "copy_ids": ["copy-1", "copy-2"],
         "script_excerpt": "The agency has five priorities.", "layout": "cards",
         "direction": {"purpose": "Explain the agency's priorities", "subject": "The agency",
                       "action": "Reveal its named priorities", "continuity": "Agency priorities",
                       "evidence_kind": {"asset-1": "captured_evidence", "asset-2": "illustration",
                                         "editorial": "editorial"}[key], "motion": "reveal"}}
        for key in order
    ]}


@pytest.mark.parametrize("order", [
    ("asset-2", "asset-1", "editorial"),
    ("asset-1", "editorial", "asset-2"),
    ("editorial", "asset-2", "asset-1"),
])
def test_editor_order_is_preserved_and_covers_the_original_blank_interval(order):
    shots = ms.validate(raw_plan(order), *context())
    assert [shot["asset_id"] for shot in shots] == list(order)
    for centisecond in range(2650):
        active = [s for s in shots if ms.ticks(s["start"]) <= centisecond < ms.ticks(s["start"]) + ms.ticks(s["duration"])]
        assert len(active) == 1


@pytest.mark.parametrize("mutation,error", [
    (lambda p: p["shots"].pop(), "coverage"),
    (lambda p: p["shots"][1].update(duration=16), "measured duration"),
    (lambda p: p["shots"][0].update(asset_id="missing"), "unknown asset"),
    (lambda p: p["shots"][0].update(script_excerpt="Another unrelated story"), "narration"),
    (lambda p: p["shots"][0].update(copy_ids=["invented"]), "copy ids"),
    (lambda p: p["shots"][0].update(duration=float("nan")), "finite"),
    (lambda p: p["shots"][0].update(duration=-1), "nonnegative"),
    (lambda p: p["shots"][0].update(duration=True), "boolean"),
    (lambda p: p["shots"][2].update(asset_id="asset-2"), "repeated"),
    (lambda p: (p["shots"][0].update(asset_id="editorial"),
                p["shots"][0]["direction"].update(evidence_kind="editorial")), "omitted"),
])
def test_editor_cannot_ship_gaps_invented_assets_or_repetition(mutation, error):
    raw = raw_plan()
    mutation(raw)
    with pytest.raises(ValueError, match=error):
        ms.validate(raw, *context())


def test_recovery_keeps_assets_and_fills_long_tail_with_real_information():
    scene, assets, copy = context()
    scene["duration"] = 70.26
    raw = ms.fallback(scene, assets, copy)
    shots = ms.validate(raw, scene, assets, copy)
    assert {s["kind"] for s in shots} == {"public_footage", "paper_collage", "editorial"}
    assert all(s["duration"] <= 12 for s in shots if s["kind"] == "editorial")
    assert round(sum(s["duration"] for s in shots), 2) == 70.26


def test_recovery_reserves_reading_time_for_the_editorial_tail():
    scene = {"id": "scene-01", "duration": 16, "text": "The system processes a single request."}
    assets = [{"id": "asset-1", "kind": "public_footage", "duration": 15.01}]
    copy = [{"id": "copy-1", "text": scene["text"]}]
    shots = ms.validate(ms.fallback(scene, assets, copy), scene, assets, copy, require_direction=True)
    assert shots[-1]["kind"] == "editorial" and shots[-1]["duration"] >= 2
    assert sum(ms.ticks(shot["duration"]) for shot in shots) == 1600


def test_recovery_rebudgets_copy_when_trimming_the_previous_shot_for_a_tail():
    long_copy = "The system processes a request and returns the result while preserving every important detail of the original submitted input data"
    scene = {"id": "scene-01", "duration": 9, "text": long_copy}
    assets = [{"id": "asset-1", "kind": "public_footage", "duration": 8.9}]
    copy = [{"id": "copy-1", "text": long_copy}, {"id": "copy-2", "text": "Request processing"}]
    raw = ms.fallback(scene, assets, copy)
    shots = ms.validate(raw, scene, assets, copy, require_direction=True)
    assert shots[0]["copy_ids"] == ["copy-2"]
    assert sum(ms.ticks(shot["duration"]) for shot in shots) == 900


def test_no_asset_explainer_is_directed_and_brief_changes_invalidate_checkpoint(tmp_path):
    scene = {"id": "scene-01", "duration": 9, "text": "A request enters the system. Then a response returns."}
    plan = {"id": scene["id"], "headline": "Request and response", "body": "A response returns"}
    opening = {"id": "intro", "duration": 2, "text": "Welcome", "program_segment_kind": "opening"}
    board = {"scenes": [opening, scene], "director_brief": {"takeaway": "Follow one request"}}
    raw = {"id": scene["id"], "shots": [{"asset_id": "editorial", "duration": 9,
        "copy_ids": ["copy-1", "copy-2"], "script_excerpt": scene["text"], "layout": "cards",
        "direction": {"purpose": "Follow a request", "subject": "Request",
                      "action": "Reveal the response", "continuity": "The same request",
                      "evidence_kind": "editorial", "motion": "reveal"}}]}
    chat = AsyncMock(return_value=json.dumps([raw]))
    provider = AsyncMock(return_value=("local-endpoint", "configured-model", "not-a-real-key"))
    with patch("backend.pipeline.digester._resolve_provider", provider), patch("backend.pipeline.digester._chat", chat):
        report = asyncio.run(ms.plan_media_shots([plan], board, tmp_path))
        assert report["contract_version"] == 3
        assert [row["id"] for row in report["scenes"]] == [scene["id"]]
        assert report["scenes"][0]["planner"] == "ai"
        payload = json.loads(chat.call_args.args[1])["scenes"][0]
        assert payload["assets"] == [] and payload["director_brief"] == board["director_brief"]
        assert chat.call_args.args[2:5] == ("local-endpoint", "configured-model", "not-a-real-key")
        assert chat.call_args.kwargs == {
            "max_tokens": 28000, "enable_skills": False, "disable_thinking": True,
        }
        asyncio.run(ms.plan_media_shots([plan], board, tmp_path))
        assert chat.await_count == 1
        board["director_brief"]["takeaway"] = "Explain the response"
        asyncio.run(ms.plan_media_shots([plan], board, tmp_path))
        assert chat.await_count == 2


@pytest.mark.parametrize("duration", [.2, .37, 1, 3.49, 15.01, 120])
def test_shared_motion_cues_are_deterministic_ordered_and_inside_the_shot(duration):
    cues = ms.motion_cues(duration)
    assert list(cues) == ["reveal_start", "reveal_end", "emphasis", "settle"]
    assert list(cues.values()) == sorted(cues.values())
    assert 0 < cues["reveal_start"] <= cues["settle"] < duration
    assert cues == ms.motion_cues(duration)


def _relationship(layout="flow"):
    text = ("A request enters the system. Then a response returns." if layout == "flow" else
            "The old system uses five steps, whereas the new system uses two steps.")
    scene = {"id": "scene-02", "text": text, "duration": 12}
    copy = [{"id": "title", "text": "Request sequence" if layout == "flow" else "Two systems"},
            {"id": "left", "text": "Request enters" if layout == "flow" else "Old: five steps"},
            {"id": "right", "text": "Response returns" if layout == "flow" else "New: two steps"}]
    raw = {"shots": [{"asset_id": "editorial", "duration": 12, "copy_ids": ["title", "left", "right"],
        "script_excerpt": text, "relationship_excerpt": text, "layout": layout,
        "motion_cues": {"reveal_start": 999, "settle": -1},
        "direction": {"purpose": "Explain the stated relationship", "subject": "Request",
            "action": "Reveal the stated stages", "continuity": "The same system",
            "evidence_kind": "editorial", "motion": "trace" if layout == "flow" else "compare"}}]}
    return raw, scene, [], copy


@pytest.mark.parametrize("layout", ["flow", "comparison"])
def test_relationship_layout_uses_grounded_excerpt_and_program_owned_motion(layout):
    raw, scene, assets, copy = _relationship(layout)
    shot = ms.validate(raw, scene, assets, copy, require_direction=True)[0]
    assert shot["layout"] == layout and shot["relationship_excerpt"] == scene["text"]
    assert shot["motion_cues"] == ms.motion_cues(12)


@pytest.mark.parametrize("layout", ["flow", "comparison"])
def test_relationship_layout_cannot_replace_the_captured_media_frame(layout):
    raw, scene, _, copy = _relationship(layout)
    raw["shots"][0]["asset_id"] = "asset-1"
    raw["shots"][0]["direction"]["evidence_kind"] = "captured_evidence"
    with pytest.raises(ValueError, match="require an editorial shot"):
        ms.validate(raw, scene, [{"id": "asset-1", "kind": "image"}], copy, require_direction=True)


@pytest.mark.parametrize("mutation,error", [
    (lambda shot: shot.pop("direction"), "direction object"),
    (lambda shot: shot["direction"].update(purpose=""), "purpose"),
    (lambda shot: shot["direction"].update(evidence_kind="captured_evidence"), "actual asset"),
    (lambda shot: shot.update(relationship_excerpt="A request enters the system."), "grounded relationship"),
    (lambda shot: shot.update(relationship_excerpt="An invented claim causes a response."), "grounded relationship"),
    (lambda shot: shot["direction"].update(motion="compare"), "match the grounded layout"),
    (lambda shot: shot.update(copy_ids=["title", "left", "left"]), "repeats"),
])
def test_direction_and_relationship_cannot_invent_evidence_or_duplicate_copy(mutation, error):
    raw, scene, assets, copy = _relationship()
    mutation(raw["shots"][0])
    with pytest.raises(ValueError, match=error):
        ms.validate(raw, scene, assets, copy, require_direction=True)


def test_long_copy_must_have_a_readable_hold_and_legacy_direction_stays_compatible():
    scene = {"id": "scene-02", "duration": 1, "text": "The system has many important details to explain."}
    copy = [{"id": "copy-1", "text": scene["text"]}]
    raw = {"shots": [{"asset_id": "editorial", "duration": 1,
                     "script_excerpt": scene["text"], "copy_ids": ["copy-1"]}]}
    assert ms.validate(raw, scene, [], copy)[0]["direction"]["motion"] == "reveal"
    raw["shots"][0]["direction"] = ms._default_direction("editorial", [scene["text"]])
    with pytest.raises(ValueError, match="readable attention window"):
        ms.validate(raw, scene, [], copy, require_direction=True)


def test_a_short_cut_cannot_claim_a_three_word_reading_window():
    scene = {"id": "scene-01", "duration": .2, "text": "Request processing"}
    copy = [{"id": "copy-1", "text": scene["text"]}]
    raw = {"shots": [{"asset_id": "editorial", "duration": .2, "copy_ids": ["copy-1"],
                     "script_excerpt": scene["text"],
                     "direction": ms._default_direction("editorial", [scene["text"]])}]}
    with pytest.raises(ValueError, match="readable attention window"):
        ms.validate(raw, scene, [], copy, require_direction=True)


def test_editorial_copy_deduplicates_case_and_punctuation_variants():
    copy = ms.editorial_copy({"headline": "A request arrives", "body": "A REQUEST ARRIVES."},
                             {"text": "A request arrives. A response follows."})
    assert [row["text"] for row in copy] == ["A request arrives", "A response follows."]


@pytest.mark.parametrize("theme", list(scene_kit.THEMES))
def test_rendered_mixed_scene_has_timed_media_and_substantive_editorial_panels(theme):
    shots = ms.validate(raw_plan(), *context())
    plan = {"id": "scene-02", "headline": "Five priorities", "media_shots": shots}
    kit = scene_kit.ScenePlan.from_dict(plan, duration=26.5, scene_id="scene-02", theme=scene_kit.THEMES[theme])
    html = scene_kit.render_scene(kit)
    ms.assert_rendered_shots(plan, html)
    assert 'class="clip shot-panel editorial"' in html
    assert 'class="shot-card ' in html
    assert "public-footage-fallback" not in html
    assert " loop" not in html
    assert 'data-start="8.00" data-duration="15.01"' in html
    with pytest.raises(ValueError, match="timing"):
        ms.assert_rendered_shots(plan, html.replace('data-duration="15.01"', 'data-duration="26.50"'))
    with pytest.raises(ValueError, match="looped"):
        ms.assert_rendered_shots(plan, html.replace("muted playsinline", "loop muted playsinline"))


@pytest.mark.parametrize("layout", ["cards", "split"])
@pytest.mark.parametrize("kind", ["editorial", "public_footage", "image"])
@pytest.mark.parametrize("orientation", ["landscape", "portrait"])
def test_two_selected_claims_render_as_distinct_panels(layout, kind, orientation):
    copy = ["First supported assertion", "Second supported assertion"]
    shot = {"start": 0, "duration": 9, "kind": kind, "layout": layout, "copy": copy,
            "motion_cues": ms.motion_cues(9), "source_start": 2, "source_duration": 15,
            "src": "../verified.mp4" if kind == "public_footage" else "../verified.png",
            "credit": "Source: Verified source"}
    plan = {"id": "scene-02", "headline": "Scene context", "media_shots": [shot]}
    kit = scene_kit.ScenePlan.from_dict(
        plan, duration=9, scene_id="scene-02",
        frame=PORTRAIT if orientation == "portrait" else LANDSCAPE,
    )
    html = scene_kit.render_scene(kit)
    ms.assert_rendered_shots(plan, html)
    doc = ms._Elements(html)
    prefix = "scene-02-shot-1"
    assert doc.parents[prefix + "-title-panel"] == prefix + "-pair"
    assert doc.parents[prefix + "-card-0"] == prefix + "-pair"
    assert doc.parents[prefix + "-title"] == prefix + "-title-panel"
    assert doc.elements[prefix + "-panel"][1]["data-shot-layout"] == layout
    assert all(html.count(text) == 1 for text in copy)
    assert html.count("Source: Verified source") == 1
    assert f'inAt("#{prefix}-title",' in html
    assert f'inAt("#{prefix}-card-0",' in html
    assert 'data-cue-reveal-start="0.100"' in html
    assert 'data-cue-settle="1.750"' in html
    assert '.visual .shot-grid.shot-pair { display:grid; max-width:1400px; }' in html
    assert '.shot-pair > .shot-card { min-width:0;' in html
    if orientation == "portrait":
        assert '.shot-grid.shot-pair, #scene-02 .shot-grid.shot-pair.split { grid-template-columns:1fr; }' in html
    if kind != "editorial":
        media = doc.elements[prefix + "-media"][1]
        assert media["src"] == shot["src"]
        assert media["data-duration"] == "9.00"
        if kind == "public_footage":
            assert media["data-media-start"] == "2.00"
    # A renamed attribute or two nested nodes is not proof of two rendered panels.
    for invalid in [
        html.replace(f'data-shot-layout="{layout}"', 'data-shot-layout="focus"'),
        html.replace(f'class="shot-grid {layout} shot-pair"', 'class="shot-grid focus"'),
        html.replace(f'<div id="{prefix}-pair"', f'<div><div id="{prefix}-pair"', 1),
        html.replace(f'<div class="shot-card " id="{prefix}-card-0"',
                     f'<div><div class="shot-card " id="{prefix}-card-0"', 1),
    ]:
        with pytest.raises(ValueError, match="paired"):
            ms.assert_rendered_shots(plan, invalid)


@pytest.mark.parametrize("layout,copy", [
    ("cards", ["Only one assertion"]),
    ("split", ["Only one assertion"]),
    ("focus", ["The title", "One support"]),
    ("cards", ["The title", "First support", "Second support"]),
    ("split", ["The title", "First support", "Second support"]),
])
def test_other_selected_copy_layouts_keep_existing_structure(layout, copy):
    shot = {"start": 0, "duration": 12, "kind": "editorial", "layout": layout,
            "copy": copy, "motion_cues": ms.motion_cues(12)}
    plan = {"id": "scene-02", "media_shots": [shot]}
    html = scene_kit.render_scene(scene_kit.ScenePlan.from_dict(plan, duration=12, scene_id="scene-02"))
    ms.assert_rendered_shots(plan, html)
    doc = ms._Elements(html)
    assert "scene-02-shot-1-pair" not in doc.elements
    assert doc.parents["scene-02-shot-1-title"] == "scene-02-shot-1-panel"
    assert doc.elements["scene-02-shot-1-panel"][1]["data-shot-layout"] == ("focus" if len(copy) < 3 else layout)
    assert all(html.count(text) == 1 for text in copy)


def test_pair_guard_parser_does_not_nest_following_elements_under_void_tags():
    doc = ms._Elements('<div id="root"><img id="image" src="x"><br/><source src="y">'
                       '<div id="first"></div><span id="empty"/><div id="second"></div></div>')
    assert doc.parents["first"] == doc.parents["second"] == "root"


def test_live_planner_repairs_invalid_output_and_reuses_only_valid_checkpoint(tmp_path):
    scene, assets, _ = context()
    plan = {"id": "scene-02", "headline": "Five priorities", "body": "Artificial intelligence"}
    bad = raw_plan()
    bad["shots"][0]["duration"] = 99
    provider = AsyncMock(return_value=("endpoint", "model", "key"))
    chat = AsyncMock(side_effect=[json.dumps([bad]), json.dumps([raw_plan()])])
    with patch.object(ms, "inventory", return_value=assets), patch("backend.pipeline.digester._resolve_provider", provider), patch("backend.pipeline.digester._chat", chat):
        report = asyncio.run(ms.plan_media_shots([plan], {"scenes": [scene]}, tmp_path))
        assert chat.await_count == 2
        assert all(call.kwargs == {
            "max_tokens": 28000, "enable_skills": False, "disable_thinking": True,
        } for call in chat.await_args_list)
        assert "measured duration" in chat.call_args_list[1].args[1]
        assert report["scenes"][0]["planner"] == "ai"
        assert plan["media_shots"][0]["kind"] == "paper_collage"
        asyncio.run(ms.plan_media_shots([plan], {"scenes": [scene]}, tmp_path))
        assert chat.await_count == 2
        # Actual media changes must invalidate a formerly successful plan.
        assets[0]["duration"] = 14
        chat.side_effect = RuntimeError("provider offline")
        report = asyncio.run(ms.plan_media_shots([plan], {"scenes": [scene]}, tmp_path))
        assert chat.await_count == 4
        assert report["scenes"][0]["planner"] == "recovery"
        assert all(s["duration"] <= 14 for s in plan["media_shots"] if s["kind"] == "public_footage")


def _batch_contexts(count):
    scene, assets, _ = context()
    scenes = [{**scene, "id": f"scene-{i:02d}"} for i in range(count)]
    plans = [{"id": scene["id"], "headline": "Five priorities", "body": "Artificial intelligence"}
             for scene in scenes]
    return plans, {"scenes": scenes}, assets


def _batch_answer(content):
    return json.dumps([{**raw_plan(), "id": scene["id"]}
                       for scene in json.loads(content)["scenes"]])


def test_direction_constraints_repair_only_current_rejected_candidate(tmp_path):
    plans, board, assets = _batch_contexts(2)
    for plan, digest in zip(plans, ["historical-video", "current-video"]):
        plan["review_repair_source_sha256"] = digest
        plan["visual_review_feedback"] = {
            "issues": ["The second clause appears before its promised reveal."],
            "suggested_visual": "Use a supported staged explanation.",
        }
    batches = []

    async def chat(prompt, content, *args, **kwargs):
        batches.append(json.loads(content)["scenes"])
        return _batch_answer(content)

    provider = AsyncMock(return_value=("endpoint", "model", "key"))
    with patch.object(ms, "inventory", return_value=assets), patch(
        "backend.pipeline.digester._resolve_provider", provider
    ), patch("backend.pipeline.digester._chat", side_effect=chat):
        original = asyncio.run(ms.plan_media_shots(plans, board, tmp_path))
        report_path = tmp_path / "av_sync_report.next.json"
        report_path.write_text(json.dumps({
            "rendered_video_sha256": "current-video",
            "multimodal": {"failed_scene_ids": ["scene-01"]},
        }))
        repaired = asyncio.run(ms.plan_media_shots(plans, board, tmp_path))
        assert [scene["id"] for scene in batches[-1]] == ["scene-01"]
        feedback = batches[-1][0]["visual_review_feedback"]
        assert len(feedback["production_constraints"]) == 3
        assert "indivisible" in feedback["production_constraints"][0]
        assert "actual selected asset" in feedback["production_constraints"][1]
        assert "Do not repeat exact copy-id groups" in feedback["production_constraints"][2]
        assert all("production_constraints" not in plan["visual_review_feedback"] for plan in plans)
        assert [row["input_sha256"] for row in repaired["scenes"]] == [
            row["input_sha256"] for row in original["scenes"]
        ]
        assert repaired["scenes"][1]["direction_repair"]["video_sha256"] == "current-video"
        asyncio.run(ms.plan_media_shots(plans, board, tmp_path))
        assert len(batches) == 2  # Reuse this exact constrained repair on resume.

        # A later candidate rejects another scene. Historical feedback and the
        # repaired scene's unchanged context must not force both through the model.
        plans[0]["review_repair_source_sha256"] = "next-video"
        report_path.write_text(json.dumps({
            "rendered_video_sha256": "next-video",
            "multimodal": {"failed_scene_ids": ["scene-00"]},
        }))
        asyncio.run(ms.plan_media_shots(plans, board, tmp_path))
        assert len(batches) == 3
        assert [scene["id"] for scene in batches[-1]] == ["scene-00"]


def test_editor_batches_overlap_with_a_two_request_cap_and_reuse_results(tmp_path):
    async def exercise():
        plans, board, assets = _batch_contexts(9)
        started, release = asyncio.Event(), asyncio.Event()
        active = maximum = 0
        batches = []

        async def chat(prompt, content, *args, **kwargs):
            nonlocal active, maximum
            batches.append([scene["id"] for scene in json.loads(content)["scenes"]])
            active += 1
            maximum = max(maximum, active)
            if len(batches) == 2:
                started.set()
            try:
                await release.wait()
                return _batch_answer(content)
            finally:
                active -= 1

        provider = AsyncMock(return_value=("endpoint", "model", "key"))
        with patch.object(ms, "inventory", return_value=assets), patch(
            "backend.pipeline.digester._resolve_provider", provider
        ), patch("backend.pipeline.digester._chat", side_effect=chat):
            task = asyncio.create_task(ms.plan_media_shots(plans, board, tmp_path))
            try:
                await asyncio.wait_for(started.wait(), 2)
                await asyncio.sleep(0)
                assert len(batches) == 2  # The third batch cannot start yet.
            finally:
                release.set()
            report = await asyncio.wait_for(task, 2)
            assert maximum == 2 and len(batches) == 3 and active == 0
            assert [row["id"] for row in report["scenes"]] == [scene["id"] for scene in board["scenes"]]
            assert all(row["coverage"] == 1 and row["passed"] for row in report["scenes"])
            await ms.plan_media_shots(plans, board, tmp_path)
            assert len(batches) == 3
            assert provider.await_count == 1

    asyncio.run(exercise())


@pytest.mark.parametrize("failure", ["provider", "invalid_plan"])
def test_editor_parallel_failure_retries_only_unaccepted_scenes(tmp_path, failure):
    plans, board, assets = _batch_contexts(8)
    batches = []

    async def chat(prompt, content, *args, **kwargs):
        scenes = json.loads(content)["scenes"]
        ids = [scene["id"] for scene in scenes]
        batches.append(ids)
        if ids[0] == "scene-04" and sum(batch[0] == "scene-04" for batch in batches) == 1:
            if failure == "provider":
                raise RuntimeError("temporary provider failure")
            response = json.loads(_batch_answer(content))
            response[0]["shots"][0]["duration"] = 99
            return json.dumps(response)
        return _batch_answer(content)

    with patch.object(ms, "inventory", return_value=assets), patch(
        "backend.pipeline.digester._resolve_provider", AsyncMock(return_value=("endpoint", "model", "key"))
    ), patch("backend.pipeline.digester._chat", side_effect=chat):
        report = asyncio.run(ms.plan_media_shots(plans, board, tmp_path))
    assert len(batches) == 3
    assert batches[2] == (["scene-04", "scene-05", "scene-06", "scene-07"]
                          if failure == "provider" else ["scene-04"])
    assert len(report["scenes"]) == 8
    assert all(row["passed"] and row["planner"] == "ai" for row in report["scenes"])


def test_editor_cancellation_drains_inflight_batches_without_retrying(tmp_path):
    async def exercise():
        plans, board, assets = _batch_contexts(9)
        ready, never = asyncio.Event(), asyncio.Event()
        started = drained = 0

        async def chat(*args, **kwargs):
            nonlocal started, drained
            started += 1
            if started == 2:
                ready.set()
            try:
                await never.wait()
            finally:
                drained += 1

        with patch.object(ms, "inventory", return_value=assets), patch(
            "backend.pipeline.digester._resolve_provider", AsyncMock(return_value=("endpoint", "model", "key"))
        ), patch("backend.pipeline.digester._chat", side_effect=chat):
            task = asyncio.create_task(ms.plan_media_shots(plans, board, tmp_path))
            await asyncio.wait_for(ready.wait(), 2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert started == drained == 2
        assert not (tmp_path / "media_shots.json").exists()
        assert all("media_shots" not in plan for plan in plans)

    asyncio.run(exercise())


def test_locked_gate_inspects_real_scheduled_elements(tmp_path):
    scene, assets, _ = context()
    shots = ms.validate(raw_plan(), *context())
    plan = {"id": "scene-02", "headline": "Five priorities", "footage_src": "footage/hearing.mp4",
            "collage_src": "../collage_broll/agency.mp4", "media_shots": shots}
    kit = scene_kit.ScenePlan.from_dict(plan, duration=26.5, scene_id="scene-02")
    folder = tmp_path / "compositions"
    folder.mkdir()
    path = folder / "scene-02.html"
    html = scene_kit.render_scene(kit)
    path.write_text(html)
    composer._assert_locked_visual_assets(tmp_path, [plan])
    path.write_text(html.replace('id="scene-02-shot-1-media"', 'id="removed"'))
    with pytest.raises(RuntimeError, match="missing rendered shot"):
        composer._assert_locked_visual_assets(tmp_path, [plan])


def test_inventory_uses_actual_video_duration_and_keeps_mixed_sources(tmp_path):
    folder = tmp_path / "footage"
    folder.mkdir()
    video = folder / "short.mp4"
    subprocess.run([
        "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=blue:s=64x64:r=24",
        "-t", "1", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video),
    ], check=True)
    plan = {"footage_sequence": [{"src": "footage/short.mp4", "duration_seconds": 99}],
            "collage_broll": True, "collage_src": "../footage/short.mp4"}
    assets = ms.inventory(plan, tmp_path)
    assert [asset["kind"] for asset in assets] == ["public_footage", "paper_collage"]
    assert all(asset["duration"] == 1 for asset in assets)
    plan["footage_sequence"][0]["src"] = "../outside.mp4"
    with pytest.raises(ValueError, match="Missing or out-of-project"):
        ms.inventory(plan, tmp_path)


def test_failed_coverage_cannot_be_overruled_by_other_quality_checks():
    report = {"visual_coverage": {"passed": False}}
    composer._finalize_quality_report(report, {"passed": True}, {"passed": True},
                                     {"passed": True}, multimodal_enabled=True)
    assert report["passed"] is False


def test_short_hold_is_optional_bounded_and_cannot_restart_footage():
    raw = raw_plan(("asset-2", "asset-1", "editorial"))
    raw["shots"][2]["duration"] = 2
    raw["shots"].insert(2, {**raw["shots"][1], "treatment": "hold", "duration": 1.49})
    shots = ms.validate(raw, *context())
    assert shots[2]["kind"] == "hold"
    assert shots[2]["source_start"] == 14.96
    raw["shots"][2]["duration"] = 3
    with pytest.raises(ValueError, match="at most 2"):
        ms.validate(raw, *context())
    raw["shots"][2]["duration"] = 1.49
    raw["shots"][2]["asset_id"] = "asset-2"
    with pytest.raises(ValueError, match="follow its video"):
        ms.validate(raw, *context())


def test_video_in_point_is_owned_by_the_editor_and_bounded_by_the_source():
    raw = raw_plan()
    raw["shots"][1].update(duration=10.01, source_start=5)
    raw["shots"][2]["duration"] = 8.49
    shots = ms.validate(raw, *context())
    plan = {"id": "scene-02", "media_shots": shots}
    html = scene_kit.render_scene(scene_kit.ScenePlan.from_dict(plan, duration=26.5, scene_id="scene-02"))
    assert 'data-media-start="5.00"' in html
    ms.assert_rendered_shots(plan, html)
    with pytest.raises(ValueError, match="in-point"):
        ms.assert_rendered_shots(plan, html.replace('data-media-start="5.00"', 'data-media-start="0.00"'))
    raw["shots"][1]["source_start"] = 6
    with pytest.raises(ValueError, match="measured duration"):
        ms.validate(raw, *context())

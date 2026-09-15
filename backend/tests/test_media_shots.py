import asyncio
import json
import subprocess
from unittest.mock import AsyncMock, patch

import pytest

from backend.pipeline import media_shots as ms, scene_kit, composer
from PIL import Image


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
         "script_excerpt": "The agency has five priorities.", "layout": "cards"}
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
    (lambda p: p["shots"][0].update(asset_id="editorial"), "omitted"),
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

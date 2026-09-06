import asyncio
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from backend import config
from backend.pipeline import footage


def wikimedia_page(*, license_name="CC BY-SA 4.0", license_code="cc-by-sa-4.0", size=2_000_000):
    return {
        "title": "File:City cyclists.webm",
        "imageinfo": [
            {
                "mime": "video/webm",
                "width": 1280,
                "height": 720,
                "duration": 12.5,
                "size": size,
                "url": "https://upload.wikimedia.org/city.webm",
                "descriptionurl": "https://commons.wikimedia.org/wiki/File:City_cyclists.webm",
                "extmetadata": {
                    "LicenseShortName": {"value": license_name},
                    "License": {"value": license_code},
                    "LicenseUrl": {"value": "https://creativecommons.org/licenses/by-sa/4.0/"},
                    "Artist": {"value": '<a href="/wiki/User:Scout">Open Scout</a>'},
                    "ImageDescription": {"value": "<p>Cyclists cross a city street.</p>"},
                    "AttributionRequired": {"value": "true"},
                },
            }
        ],
    }


class FootagePlanTests(unittest.TestCase):
    def test_plan_parser_accepts_fenced_json_and_deduplicates_queries(self):
        raw = """```json
{"queries":[
  {"query":"city cyclists commuting!", "purpose":"Urban movement"},
  {"query":"city cyclists commuting", "purpose":"duplicate"},
  {"query":"solar panels rooftop", "purpose":"Clean energy"}
]}
```"""
        parsed = footage._parse_plan(raw, 4)

        self.assertEqual([item["query"] for item in parsed], [
            "city cyclists commuting",
            "solar panels rooftop",
        ])

    def test_license_gate_allows_only_explicit_open_licenses(self):
        self.assertTrue(footage._is_open_license("Public domain"))
        self.assertTrue(footage._is_open_license("CC BY-SA 4.0"))
        self.assertFalse(footage._is_open_license(""))
        self.assertFalse(footage._is_open_license("CC BY-NC 4.0"))
        self.assertFalse(footage._is_open_license("All rights reserved"))

    def test_manual_query_is_grounded_in_matching_script_sentence(self):
        script = (
            "Shein pivoted to a Hong Kong IPO. "
            "A robot duck can walk and right itself."
        )

        self.assertEqual(
            footage._script_purpose_for_query("robot", script),
            "A robot duck can walk and right itself.",
        )
        self.assertEqual(footage._script_purpose_for_query("computer screen", script), "")

    def test_collage_reserved_purpose_detects_same_story_only(self):
        reserved = ["Shein pivoted to a Hong Kong IPO after Beijing's approval."]

        self.assertTrue(
            footage._purpose_conflicts_with_reserved(
                "Reuters says Shein pivoted to a Hong Kong IPO after Beijing's approval.",
                reserved,
            )
        )
        self.assertFalse(
            footage._purpose_conflicts_with_reserved(
                "A robot duck can walk and right itself.", reserved
            )
        )

    def test_sentence_purpose_expands_to_full_storyboard_scene(self):
        scenes = {
            "scene-02": "Shein moved to Hong Kong. Kaiser will keynote in Beijing.",
            "scene-03": "A robot duck can walk and right itself.",
        }

        self.assertEqual(
            footage._closest_storyboard_purpose("Kaiser keynote Beijing", scenes),
            scenes["scene-02"],
        )

    def test_footage_plan_discards_two_queries_for_the_same_story(self):
        script = (
            "Taiwanese prosecutors raided a printed circuit board factory. "
            "AQuA agents improve research using a factor loop and model loop."
        )
        plan = [
            {
                "query": "printed circuit board factory",
                "purpose": "The Taiwan factory raid",
            },
            {
                "query": "Taipei city skyline",
                "purpose": "The Taiwan prosecutor raid and chip supply story",
            },
            {
                "query": "computer research agents",
                "purpose": "AQuA agents improve research with two loops",
            },
        ]

        grounded = footage._distinct_grounded_plan(plan, script, 2)

        self.assertEqual(
            [item["query"] for item in grounded],
            ["printed circuit board factory", "computer research agents"],
        )
        self.assertEqual(
            grounded[0]["script_excerpt"],
            "Taiwanese prosecutors raided a printed circuit board factory.",
        )
        self.assertEqual(
            grounded[1]["script_excerpt"],
            "AQuA agents improve research using a factor loop and model loop.",
        )

    def test_ai_quantity_plan_can_select_two_sequential_shots_for_one_long_beat(self):
        script = "Engineers assemble and test a reusable lunar rocket engine."
        answer = json.dumps(
            {
                "queries": [
                    {"query": "rocket engine assembly", "purpose": script},
                    {"query": "rocket engine test fire", "purpose": script},
                ]
            }
        )
        with (
            patch.object(
                footage,
                "_resolve_provider",
                AsyncMock(return_value=("https://example.test", "model", "key")),
            ),
            patch.object(footage, "_chat", AsyncMock(return_value=answer)) as chat,
        ):
            plan, planner = asyncio.run(
                footage.plan_footage_queries(
                    title="Lunar engine",
                    script=script,
                    count=None,
                    provider_id=None,
                    ai_endpoint=None,
                    ai_model=None,
                )
            )

        self.assertEqual(len(plan), 2)
        self.assertEqual({item["script_excerpt"] for item in plan}, {script})
        self.assertEqual(planner, "ai:model")
        self.assertIn("no quota or fixed", chat.await_args.args[1])

    def test_hybrid_web_plan_skips_scene_already_filled_by_commons(self):
        rocket = "NASA engineers test a new rocket engine concept."
        factory = "Prosecutors raided a printed circuit board factory."

        remaining = footage._unoccupied_web_plan(
            [
                {
                    "query": "rocket engine test",
                    "purpose": rocket,
                    "script_excerpt": rocket,
                },
                {
                    "query": "printed circuit board factory",
                    "purpose": factory,
                    "script_excerpt": factory,
                },
            ],
            [{"provider_id": "wikimedia", "purpose": rocket}],
        )

        self.assertEqual(
            [shot["query"] for shot in remaining],
            ["printed circuit board factory"],
        )

    def test_fallback_plan_uses_distinct_story_entities_not_reporting_boilerplate(self):
        script = """Good morning. This is Frontier Tech Daily—your concise briefing.
The Wall Street Journal reports that Anthropic signed a cloud deal with Nvidia-backed Lambda for a Texas data center.
QbitAI reports that Didi Autonomous Driving began passenger tests with its Robotaxi R2 in Beijing and Guangzhou.
IEEE Spectrum revisits the first battery at the Faraday Museum in London, built after work by Alessandro Volta.
DeepTech reports that Chinese researchers developed a light-driven soft robot that can jump repeatedly.
Alex Konrad reports that AI startup Mirage streamed a live news show on X.
Thanks for watching, and subscribe for more."""

        plan = footage._fallback_plan("Frontier Tech Daily", script, 12)

        self.assertEqual(len(plan), 5)
        self.assertTrue(all(len(item["query"].split()) >= 2 for item in plan))
        self.assertEqual(len({item["purpose"] for item in plan}), len(plan))
        combined = " ".join(item["query"].casefold() for item in plan)
        self.assertNotIn("and reports", combined)
        self.assertNotIn("reports can", combined)
        self.assertIn("anthropic", combined)
        self.assertIn("didi", combined)
        self.assertIn("battery", combined)
        self.assertIn("robot", combined)

    def test_planner_fallback_keeps_alternate_story_queries_for_web_fill(self):
        script = "\n".join(
            [
                "Anthropic signed a Lambda cloud agreement in Texas.",
                "Didi tested the Robotaxi R2 in Beijing.",
                "The Faraday Museum displays Alessandro Volta's battery.",
                "Chinese researchers built a light-driven soft robot.",
                "Mirage streamed an AI news show.",
                "Students earned ham radio licenses in New Jersey.",
            ]
        )

        async def fail_chat(*args, **kwargs):
            raise ValueError("malformed planner JSON")

        async def provider(*args, **kwargs):
            return "https://example.invalid", "test-model", "test-key"

        with (
            patch.object(footage, "_chat", fail_chat),
            patch.object(footage, "_resolve_provider", provider),
        ):
            plan, planner = __import__("asyncio").run(
                footage.plan_footage_queries(
                    title="Frontier Tech Daily",
                    script=script,
                    count=2,
                    provider_id=None,
                    ai_endpoint=None,
                    ai_model=None,
                )
            )

        self.assertEqual(planner, "deterministic-fallback")
        self.assertGreater(len(plan), 2)
        self.assertEqual(len({item["script_excerpt"] for item in plan}), len(plan))

    def test_manual_plan_keeps_supplied_alternates_beyond_clip_target(self):
        script = "\n".join(
            [
                "Anthropic signed a Lambda cloud agreement in Texas.",
                "Didi tested the Robotaxi R2 in Beijing.",
                "The Faraday Museum displays Alessandro Volta's battery.",
                "Chinese researchers built a light-driven soft robot.",
            ]
        )

        plan, planner = __import__("asyncio").run(
            footage.plan_footage_queries(
                title="Frontier Tech Daily",
                script=script,
                count=2,
                provider_id=None,
                ai_endpoint=None,
                ai_model=None,
                supplied_queries=[
                    "Anthropic Lambda Texas",
                    "Didi Robotaxi Beijing",
                    "Faraday Volta battery",
                    "light-driven soft robot",
                ],
            )
        )

        self.assertEqual(planner, "user")
        self.assertEqual(len(plan), 4)


class WikimediaCandidateTests(unittest.TestCase):
    def test_two_term_query_requires_both_terms_in_candidate_metadata(self):
        self.assertFalse(
            footage._candidate_query_is_specific(
                {
                    "title": "Melosira Research Vessel.webm",
                    "description": "A university research vessel",
                },
                "student research",
            )
        )
        self.assertTrue(
            footage._candidate_query_is_specific(
                {
                    "title": "Student research presentation.webm",
                    "description": "An engineering symposium",
                },
                "student research",
            )
        )

    def test_multi_term_query_requires_two_concrete_metadata_anchors(self):
        self.assertFalse(
            footage._candidate_query_is_specific(
                {
                    "title": "General technology documentary.webm",
                    "description": "A broad report about future technology.",
                },
                "Anthropic Lambda Texas data center",
            )
        )
        self.assertTrue(
            footage._candidate_query_is_specific(
                {
                    "title": "Texas data center construction.webm",
                    "description": "Workers install cooling equipment.",
                },
                "Anthropic Lambda Texas data center",
            )
        )

    def test_candidate_retains_provenance_and_cleans_creator_markup(self):
        candidate = footage._candidate_from_page(wikimedia_page(), "landscape")

        self.assertIsNotNone(candidate)
        self.assertEqual(candidate["creator"], "Open Scout")
        self.assertEqual(candidate["license"], "CC BY-SA 4.0")
        self.assertTrue(candidate["attribution_required"])
        self.assertEqual(candidate["width"], 1280)

    def test_candidate_rejects_unknown_license_and_large_file(self):
        self.assertIsNone(
            footage._candidate_from_page(
                wikimedia_page(license_name="Copyrighted", license_code="copyright"),
                "landscape",
            )
        )
        with patch.object(config, "FOOTAGE_MAX_BYTES", 1_000):
            self.assertIsNone(
                footage._candidate_from_page(wikimedia_page(size=2_000), "landscape")
            )

    def test_next_clip_id_never_reuses_manifest_or_audit_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            footage_dir = Path(temp_dir)
            (footage_dir / "clip-01.webm").write_bytes(b"old-audit")

            clip_id = footage._next_clip_id(
                footage_dir,
                [{"id": "clip-02"}, {"id": "clip-03"}],
            )

            self.assertEqual(clip_id, "clip-04")


class AcquireFootageTests(unittest.IsolatedAsyncioTestCase):
    async def test_acquisition_writes_download_and_auditable_manifest(self):
        candidate = footage._candidate_from_page(wikimedia_page(), "landscape")

        async def fake_search(client, *, query, orientation, limit=16):
            self.assertEqual(query, "city cyclists")
            self.assertEqual(orientation, "landscape")
            return [candidate]

        async def fake_download(client, *, candidate, destination):
            destination.write_bytes(b"public-video")
            return len(b"public-video"), "demo-sha256"

        async def fake_normalize(source, destination):
            destination.write_bytes(b"render-safe-video")
            return {
                "bytes": len(b"render-safe-video"),
                "sha256": "render-sha256",
                "local_path": destination,
                "render_safe": True,
                "render_profile": {"video_codec": "h264"},
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            task_dir = Path(temp_dir)
            script_path = task_dir / "script.txt"
            script_path.write_text("A story about cycling through a changing city.")

            with (
                patch.object(footage, "search_wikimedia", fake_search),
                patch.object(footage, "_download_candidate", fake_download),
                patch.object(footage, "_normalize_render_clip", fake_normalize),
            ):
                manifest = await footage.acquire_public_footage(
                    task_id="task-demo",
                    task_dir=task_dir,
                    title="City mobility",
                    script_path=script_path,
                    clip_count=1,
                    orientation="landscape",
                    license_policy="open_only",
                    provider_id=None,
                    ai_endpoint=None,
                    ai_model=None,
                    supplied_queries=["city cyclists"],
                )

            saved = json.loads(
                (task_dir / "footage" / "manifest.json").read_text()
            )
            self.assertEqual(manifest["status"], "ready")
            self.assertEqual(saved["planner"], "user")
            self.assertEqual(saved["clips"][0]["license"], "CC BY-SA 4.0")
            self.assertEqual(saved["clips"][0]["source_sha256"], "demo-sha256")
            self.assertEqual(saved["clips"][0]["sha256"], "render-sha256")
            self.assertTrue(saved["clips"][0]["render_safe"])
            self.assertTrue(
                (task_dir / saved["clips"][0]["local_path"]).is_file()
            )

    async def test_rescout_preserves_verified_clip_and_only_plans_the_gap(self):
        candidate = footage._candidate_from_page(wikimedia_page(), "landscape")

        async def fake_search(client, *, query, orientation, limit=16):
            self.assertEqual(query, "solar panels")
            return [
                {
                    **candidate,
                    "title": "File:Solar panels.webm",
                    "source_page_url": "https://commons.wikimedia.org/wiki/File:Solar_panels.webm",
                    "download_url": "https://upload.wikimedia.org/solar.webm",
                }
            ]

        async def fake_download(client, *, candidate, destination):
            destination.write_bytes(b"second-public-video")
            return len(b"second-public-video"), hashlib.sha256(
                b"second-public-video"
            ).hexdigest()

        async def fake_normalize(source, destination):
            destination.write_bytes(b"render-safe-video")
            return {
                "bytes": len(b"render-safe-video"),
                "sha256": hashlib.sha256(b"render-safe-video").hexdigest(),
                "local_path": destination,
                "render_safe": True,
                "render_profile": {"video_codec": "h264"},
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            task_dir = Path(temp_dir)
            footage_dir = task_dir / "footage"
            footage_dir.mkdir()
            first = footage_dir / "clip-01.webm"
            first.write_bytes(b"first-public-video")
            first_sha = hashlib.sha256(first.read_bytes()).hexdigest()
            (footage_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "provider_id": "wikimedia",
                        "orientation": "landscape",
                        "created_at": "2026-08-28T00:00:00+00:00",
                        "clips": [
                            {
                                "id": "clip-01",
                                "query": "visual stories",
                                "purpose": "Two visual stories.",
                                "license": "CC0",
                                "source_page_url": "https://commons.wikimedia.org/wiki/File:First.webm",
                                "local_path": "footage/clip-01.webm",
                                "sha256": first_sha,
                            }
                        ],
                    }
                )
            )
            script_path = task_dir / "script.txt"
            script_path.write_text("Two visual stories.")

            with (
                patch.object(footage, "search_wikimedia", fake_search),
                patch.object(footage, "_download_candidate", fake_download),
                patch.object(footage, "_normalize_render_clip", fake_normalize),
            ):
                manifest = await footage.acquire_public_footage(
                    task_id="task-rescout",
                    task_dir=task_dir,
                    title="Visual stories",
                    script_path=script_path,
                    clip_count=2,
                    orientation="landscape",
                    license_policy="open_only",
                    provider_id=None,
                    ai_endpoint=None,
                    ai_model=None,
                    supplied_queries=["solar panels"],
                )

            self.assertEqual(manifest["status"], "ready")
            self.assertEqual([clip["id"] for clip in manifest["clips"]], ["clip-01", "clip-02"])
            self.assertEqual(manifest["created_at"], "2026-08-28T00:00:00+00:00")

    async def test_rescout_drops_two_term_clip_that_matches_only_generic_half(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            task_dir = Path(temp_dir)
            footage_dir = task_dir / "footage"
            footage_dir.mkdir()
            clip_path = footage_dir / "clip-01.mp4"
            clip_path.write_bytes(b"research-vessel")
            digest = hashlib.sha256(clip_path.read_bytes()).hexdigest()
            previous = {
                "provider_id": "wikimedia",
                "orientation": "landscape",
                "clips": [
                    {
                        "id": "clip-01",
                        "query": "student research",
                        "purpose": "Students present original research.",
                        "title": "Research vessel",
                        "description": "A university research ship",
                        "license": "CC BY 4.0",
                        "source_page_url": "https://commons.example/vessel",
                        "local_path": "footage/clip-01.mp4",
                        "sha256": digest,
                    }
                ],
            }

            reused = footage._reusable_manifest_clips(
                task_dir,
                previous,
                orientation="landscape",
                script="Students present original research.",
            )

            self.assertEqual(reused, [])

    def test_explicit_rescout_replaces_only_clip_on_same_narration_scene(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            task_dir = Path(temp_dir)
            footage_dir = task_dir / "footage"
            footage_dir.mkdir()
            rows = []
            for index, (query, purpose) in enumerate(
                [
                    ("robot", "A robot duck can walk and self-right."),
                    ("student research", "Students present original research."),
                ],
                start=1,
            ):
                path = footage_dir / f"clip-{index:02d}.mp4"
                path.write_bytes(query.encode())
                rows.append(
                    {
                        "id": f"clip-{index:02d}",
                        "query": query,
                        "purpose": purpose,
                        "license": "CC BY 4.0",
                        "source_page_url": f"https://commons.example/{index}",
                        "local_path": f"footage/clip-{index:02d}.mp4",
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                )
            previous = {
                "provider_id": "wikimedia",
                "orientation": "landscape",
                "clips": rows,
            }

            reused = footage._reusable_manifest_clips(
                task_dir,
                previous,
                orientation="landscape",
                script=(
                    "A robot duck can walk and self-right. "
                    "Students present original research."
                ),
                replacement_purposes=["Students present original research."],
            )

            self.assertEqual([clip["id"] for clip in reused], ["clip-01"])

    async def test_rescout_drops_clip_reserved_for_ready_collage(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            task_dir = Path(temp_dir)
            footage_dir = task_dir / "footage"
            footage_dir.mkdir()
            clip_path = footage_dir / "clip-01.webm"
            clip_path.write_bytes(b"hong-kong-video")
            clip_sha = hashlib.sha256(clip_path.read_bytes()).hexdigest()
            (footage_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "provider_id": "wikimedia",
                        "orientation": "landscape",
                        "clips": [
                            {
                                "id": "clip-01",
                                "query": "Hong Kong",
                                "purpose": "Shein pivoted to a Hong Kong IPO after Beijing's approval.",
                                "license": "CC BY-SA 4.0",
                                "source_page_url": "https://commons.wikimedia.org/wiki/File:HK.webm",
                                "local_path": "footage/clip-01.webm",
                                "sha256": clip_sha,
                            }
                        ],
                    }
                )
            )
            collage_dir = task_dir / "collage_broll"
            collage_dir.mkdir()
            (collage_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "items": [
                            {
                                "status": "ready",
                                "scene_id": "scene-02",
                                "spec": {
                                    "script_meaning": "Shein pivoted to a Hong Kong IPO after Beijing's approval."
                                },
                            }
                        ]
                    }
                )
            )
            (task_dir / "storyboard.json").write_text(
                json.dumps(
                    {
                        "scenes": [
                            {
                                "id": "scene-02",
                                "text": (
                                    "Shein pivoted to a Hong Kong IPO after Beijing's approval. "
                                    "Kaiser will keynote a conference."
                                ),
                            }
                        ]
                    }
                )
            )

            reserved = footage._reserved_collage_purposes(task_dir)
            reusable = footage._reusable_manifest_clips(
                task_dir,
                footage.read_manifest(task_dir),
                orientation="landscape",
                script="Shein pivoted to a Hong Kong IPO. A robot duck can walk.",
                reserved_purposes=reserved,
            )

            self.assertEqual(reusable, [])
            self.assertEqual(reserved, [
                "Shein pivoted to a Hong Kong IPO after Beijing's approval. "
                "Kaiser will keynote a conference."
            ])


if __name__ == "__main__":
    unittest.main()


def test_bytefront_fallback_excludes_bookends_and_preserves_short_entities():
    script = "\n\n".join([
        "It's Sunday, September 6. This is ByteFront Espresso, your frontier-tech signal.",
        "QbitAI reports that Bilibili has closed its first AI Creation Open Competition.",
        "DeepTech reports the Ig Nobel Prizes have been announced.",
        "That's today's ByteFront Espresso. Subscribe to stay ahead of the next signal.",
    ])
    plan = footage._fallback_plan("ByteFront Espresso", script, None)
    assert len(plan) == 2
    assert "Bilibili" in plan[0]["query"] and "AI" in plan[0]["query"]
    assert "Ig Nobel" in plan[1]["query"]


def test_query_binding_does_not_switch_to_longer_unrelated_story_sentence():
    paragraph = ("World Labs released a spatial intelligence model. "
                 "The same outlet reports that Feidu Technology won a summit prize for "
                 "flood simulation that helps cities make emergency decisions before rain falls.")
    plan = footage._distinct_grounded_plan(
        [{"query": "World Labs spatial intelligence", "purpose": paragraph}], paragraph, None,
    )
    assert plan[0]["script_excerpt"] == "World Labs released a spatial intelligence model."

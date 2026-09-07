import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

from backend import config
from backend.pipeline import web_footage
from backend.pipeline.opencli import OpenCLIResult


class WebFootageAnalysisTests(unittest.IsolatedAsyncioTestCase):
    async def test_youtube_search_ranks_specific_result_beyond_first_four(self):
        rows = [
            {
                "title": f"Generic Didi travel guide {index}",
                "channel": "Travel Guide",
                "webpage_url": f"https://www.youtube.com/watch?v=generic{index}",
            }
            for index in range(1, 5)
        ]
        rows.append(
            {
                "title": "China Self-Driving DiDi RoboTaxi Fleet in Shanghai",
                "channel": "Road Test",
                "webpage_url": "https://www.youtube.com/watch?v=robotaxi",
            }
        )
        runner = AsyncMock(
            return_value=(
                0,
                "\n".join(json.dumps(row) for row in rows),
                "",
            )
        )

        with patch.object(web_footage, "_run_command", runner):
            ranked = await web_footage.search_youtube("Didi Robotaxi Beijing")

        command = runner.await_args.args[0]
        self.assertIn("ytsearch8:Didi Robotaxi Beijing", command)
        self.assertEqual(
            ranked[0]["source_page_url"],
            "https://www.youtube.com/watch?v=robotaxi",
        )

    def test_youtube_candidates_prefer_specific_story_terms_over_first_result(self):
        candidates = [
            {
                "title": "How to Use Didi in China",
                "creator": "Travel Guide",
                "description": "Book taxis and rideshares with the Didi app.",
                "source_page_url": "https://www.youtube.com/watch?v=generic",
            },
            {
                "title": "Driverless Robotaxis Begin Taking Customers in Beijing",
                "creator": "City News",
                "description": "A new autonomous taxi fleet enters service.",
                "source_page_url": "https://www.youtube.com/watch?v=wrong-brand",
            },
            {
                "title": "China Self-Driving DiDi RoboTaxi Fleet in Shanghai",
                "creator": "Road Test",
                "description": "Autonomous vehicles on public roads.",
                "source_page_url": "https://www.youtube.com/watch?v=robotaxi",
            },
        ]

        ranked = web_footage._rank_youtube_candidates(
            candidates,
            "Didi Robotaxi Beijing",
        )

        self.assertEqual(
            ranked[0]["source_page_url"],
            "https://www.youtube.com/watch?v=robotaxi",
        )
        self.assertGreater(
            ranked[0]["query_relevance_score"],
            ranked[1]["query_relevance_score"],
        )

    def test_youtube_candidate_ranking_preserves_provider_order_for_ties(self):
        candidates = [
            {
                "title": "First unrelated clip",
                "creator": "One",
                "description": "",
                "source_page_url": "https://www.youtube.com/watch?v=first",
            },
            {
                "title": "Second unrelated clip",
                "creator": "Two",
                "description": "",
                "source_page_url": "https://www.youtube.com/watch?v=second",
            },
        ]

        ranked = web_footage._rank_youtube_candidates(candidates, "Didi Robotaxi")

        self.assertEqual(
            [candidate["source_page_url"] for candidate in ranked],
            [
                "https://www.youtube.com/watch?v=first",
                "https://www.youtube.com/watch?v=second",
            ],
        )

    def test_youtube_candidate_coverage_beats_one_ambiguous_early_term(self):
        candidates = [
            {
                "title": "Faraday Motor: How It Works",
                "creator": "Electronics Tutorial",
                "description": "Build a small spinning motor.",
                "source_page_url": "https://www.youtube.com/watch?v=motor",
            },
            {
                "title": "Alessandro Volta and Making a Battery",
                "creator": "Science Museum",
                "description": "The first electric battery and Luigi Galvani.",
                "source_page_url": "https://www.youtube.com/watch?v=battery",
            },
        ]

        ranked = web_footage._rank_youtube_candidates(
            candidates,
            "Faraday Museum Volta battery",
        )

        self.assertEqual(
            ranked[0]["source_page_url"],
            "https://www.youtube.com/watch?v=battery",
        )

    async def test_non_youtube_candidate_is_rejected(self):
        candidate = {
            "platform": "bilibili",
            "source_page_url": "https://www.bilibili.com/video/BV1demo",
            "duration_seconds": 100,
        }

        with self.assertRaisesRegex(web_footage.WebFootageError, "only accepts YouTube"):
            await web_footage.analyze_candidate_link(candidate, "city narration")

    async def test_late_gemini_response_is_recovered_before_fallback(self):
        source_url = "https://www.youtube.com/watch?v=demo"
        candidate = {
            "platform": "youtube",
            "source_page_url": source_url,
            "duration_seconds": 100,
        }
        no_response = OpenCLIResult((), 0, "[NO RESPONSE] timed out", "")
        recovered = OpenCLIResult(
            (),
            0,
            (
                '[{"Role":"User","Text":"Analyze '
                + source_url
                + '"},{"Role":"Assistant","Text":"JSON{\\"start_seconds\\":66,'
                '\\"end_seconds\\":73,\\"confidence\\":0.95,\\"reason\\":'
                '\\"Italian states\\"}"}]'
            ),
            "",
        )

        with (
            patch.object(config, "WEB_FOOTAGE_CLIP_SECONDS", 8),
            patch.object(config, "WEB_FOOTAGE_CLIP_MIN_SECONDS", 5),
            patch.object(
                web_footage, "run_opencli", AsyncMock(side_effect=[no_response, recovered])
            ),
        ):
            analysis = await web_footage.analyze_candidate_link(candidate, "narration")

        self.assertEqual(analysis["start_seconds"], 66)
        self.assertEqual(analysis["end_seconds"], 73)
        self.assertEqual(analysis["analyzer"], "gemini-web-via-opencli")
        self.assertEqual(analysis["status"], "analyzed_after_timeout")

    def test_recovery_does_not_reuse_answer_after_another_video_request(self):
        source_url = "https://www.youtube.com/watch?v=first"
        turns = (
            '[{"Role":"User","Text":"'
            + source_url
            + '"},{"Role":"User","Text":"https://www.youtube.com/watch?v=second"},'
            '{"Role":"Assistant","Text":"{\\"start_seconds\\":10,\\"end_seconds\\":18}"}]'
        )

        self.assertIsNone(web_footage._analysis_from_gemini_turns(turns, source_url))

    async def test_disabled_gemini_is_recorded_as_fallback(self):
        candidate = {
            "platform": "youtube",
            "source_page_url": "https://www.youtube.com/watch?v=demo",
            "duration_seconds": 30,
        }

        with patch.object(config, "WEB_FOOTAGE_GEMINI_ENABLED", False):
            analysis = await web_footage.analyze_candidate_link(candidate, "narration")

        self.assertEqual(analysis["status"], "fallback")
        self.assertIn("disabled", analysis["reason"])

    def test_unknown_duration_safe_offset_is_fitted_after_probe(self):
        analysis = {
            "start_seconds": 0,
            "end_seconds": 8,
            "analyzer": "deterministic-safe-offset",
        }

        with (
            patch.object(config, "WEB_FOOTAGE_CLIP_SECONDS", 8),
            patch.object(config, "WEB_FOOTAGE_CLIP_MIN_SECONDS", 5),
        ):
            fitted = web_footage._fit_analysis_to_media(analysis, 100)

        self.assertEqual(fitted["start_seconds"], 10)
        self.assertEqual(fitted["end_seconds"], 18)

    def test_model_interval_is_clamped_to_configured_clip_length(self):
        candidate = {"duration_seconds": 100}
        with (
            patch.object(config, "WEB_FOOTAGE_CLIP_SECONDS", 8),
            patch.object(config, "WEB_FOOTAGE_CLIP_MIN_SECONDS", 5),
        ):
            result = web_footage._normalise_analysis(
                {
                    "start_seconds": 12,
                    "end_seconds": 40,
                    "confidence": 1.8,
                    "reason": "wide skyline",
                },
                candidate,
            )

        self.assertEqual(result["start_seconds"], 12)
        self.assertEqual(result["end_seconds"], 20)
        self.assertEqual(result["confidence"], 1)

    def test_short_gemini_interval_is_extended_to_minimum(self):
        """Gemini frequently returns ~6-second fragments; the normaliser must
        extend them to at least WEB_FOOTAGE_CLIP_MIN_SECONDS."""
        candidate = {"duration_seconds": 120}
        with (
            patch.object(config, "WEB_FOOTAGE_CLIP_SECONDS", 15),
            patch.object(config, "WEB_FOOTAGE_CLIP_MIN_SECONDS", 10),
        ):
            result = web_footage._normalise_analysis(
                {
                    "start_seconds": 20,
                    "end_seconds": 26,
                    "confidence": 0.8,
                    "reason": "city panorama",
                },
                candidate,
            )

        self.assertEqual(result["start_seconds"], 20)
        self.assertEqual(result["end_seconds"], 30)
        self.assertGreaterEqual(
            result["end_seconds"] - result["start_seconds"], 10
        )

    def test_fit_analysis_extends_short_interval_to_minimum(self):
        analysis = {
            "start_seconds": 5,
            "end_seconds": 8,
            "analyzer": "gemini-web-via-opencli",
        }
        with (
            patch.object(config, "WEB_FOOTAGE_CLIP_SECONDS", 15),
            patch.object(config, "WEB_FOOTAGE_CLIP_MIN_SECONDS", 10),
        ):
            fitted = web_footage._fit_analysis_to_media(analysis, 100)

        self.assertEqual(fitted["start_seconds"], 5)
        self.assertEqual(fitted["end_seconds"], 15)

    async def test_trim_passes_recovered_source_interval_to_ffmpeg(self):
        runner = AsyncMock(return_value=(0, "", ""))
        analysis = {"start_seconds": 66.0, "end_seconds": 73.0}

        with patch.object(web_footage, "_run_command", runner):
            await web_footage._trim(
                Path("raw.mp4"), Path("clip.mp4"), analysis, "landscape"
            )

        command = runner.await_args.args[0]
        self.assertEqual(command[command.index("-ss") + 1], "66.000")
        self.assertEqual(command[command.index("-t") + 1], "7.000")

    async def test_youtube_download_fetches_only_the_analyzed_video_section(self):
        candidate = {
            "source_page_url": "https://www.youtube.com/watch?v=demo",
            "duration_seconds": 100,
        }
        analysis = {"start_seconds": 66.0, "end_seconds": 73.0}

        with TemporaryDirectory() as directory:
            raw_dir = Path(directory)

            async def fake_run(command, **_kwargs):
                (raw_dir / "demo.mp4").write_bytes(b"video")
                return 0, "", ""

            with (
                patch.object(web_footage, "_yt_dlp_bin", return_value="yt-dlp"),
                patch.object(web_footage, "_run_command", AsyncMock(side_effect=fake_run)) as runner,
            ):
                _path, sectioned = await web_footage._download_youtube(
                    candidate, raw_dir, analysis
                )

        command = runner.await_args.args[0]
        self.assertTrue(sectioned)
        self.assertEqual(
            command[command.index("--download-sections") + 1], "*66.000-73.000"
        )
        self.assertNotIn("bestaudio", command[command.index("-f") + 1])

    async def test_youtube_section_403_retries_with_embedded_player(self):
        candidate = {
            "source_page_url": "https://www.youtube.com/watch?v=demo",
            "duration_seconds": 900,
        }
        analysis = {"start_seconds": 66.0, "end_seconds": 81.0}

        with TemporaryDirectory() as directory:
            raw_dir = Path(directory)

            async def fake_run(command, **_kwargs):
                if "--extractor-args" not in command:
                    raise web_footage.WebFootageError("ffmpeg returned 403 Forbidden")
                (raw_dir / "demo.mp4").write_bytes(b"video")
                return 0, "", ""

            with (
                patch.object(web_footage, "_yt_dlp_bin", return_value="yt-dlp"),
                patch.object(
                    web_footage,
                    "_run_command",
                    AsyncMock(side_effect=fake_run),
                ) as runner,
            ):
                path, sectioned = await web_footage._download_youtube(
                    candidate, raw_dir, analysis
                )

        self.assertEqual(path.name, "demo.mp4")
        self.assertTrue(sectioned)
        self.assertEqual(runner.await_count, 2)
        embedded_command = runner.await_args_list[1].args[0]
        self.assertIn("--download-sections", embedded_command)
        self.assertEqual(
            embedded_command[embedded_command.index("--extractor-args") + 1],
            "youtube:player_client=web_embedded",
        )

    async def test_youtube_embedded_section_failure_uses_bounded_native_download(self):
        candidate = {
            "source_page_url": "https://www.youtube.com/watch?v=demo",
            "duration_seconds": 900,
        }
        analysis = {"start_seconds": 66.0, "end_seconds": 81.0}

        with TemporaryDirectory() as directory:
            raw_dir = Path(directory)

            async def fake_run(command, **_kwargs):
                if "--download-sections" in command:
                    raise web_footage.WebFootageError("403 Forbidden")
                (raw_dir / "demo-full.mp4").write_bytes(b"video")
                return 0, "", ""

            with (
                patch.object(web_footage, "_yt_dlp_bin", return_value="yt-dlp"),
                patch.object(
                    web_footage,
                    "_run_command",
                    AsyncMock(side_effect=fake_run),
                ) as runner,
            ):
                path, sectioned = await web_footage._download_youtube(
                    candidate, raw_dir, analysis
                )

        self.assertEqual(path.name, "demo-full.mp4")
        self.assertFalse(sectioned)
        self.assertEqual(runner.await_count, 3)
        fallback_command = runner.await_args_list[2].args[0]
        self.assertNotIn("--download-sections", fallback_command)
        self.assertIn("--max-filesize", fallback_command)
        self.assertIn(
            "bestvideo[height<=360]",
            fallback_command[fallback_command.index("-f") + 1],
        )

    async def test_supplement_retries_rejected_candidate_before_download(self):
        with TemporaryDirectory() as directory:
            task_dir = Path(directory)
            (task_dir / "footage").mkdir()
            manifest = {"provider_id": "hybrid", "clips": [], "errors": []}
            candidates = [
                {
                    "platform": "youtube",
                    "provider": "YouTube",
                    "provider_id": "youtube-ytdlp",
                    "title": "Blocked candidate",
                    "creator": "Channel A",
                    "source_page_url": "https://www.youtube.com/watch?v=blocked",
                    "duration_seconds": 90,
                },
                {
                    "platform": "youtube",
                    "provider": "YouTube",
                    "provider_id": "youtube-ytdlp",
                    "title": "Working candidate",
                    "creator": "Channel B",
                    "source_page_url": "https://www.youtube.com/watch?v=working",
                    "duration_seconds": 90,
                },
            ]

            async def fake_download(candidate, raw_dir, _analysis):
                if candidate["source_page_url"].endswith("blocked"):
                    raise web_footage.WebFootageError("403 Forbidden")
                path = raw_dir / "working.mp4"
                path.write_bytes(b"raw-video")
                return path, True

            async def fake_trim(_raw, destination, _analysis, _orientation):
                destination.write_bytes(b"trimmed-video")

            with (
                patch.object(
                    web_footage,
                    "search_youtube",
                    AsyncMock(return_value=candidates),
                ) as search,
                patch.object(
                    web_footage,
                    "analyze_candidate_link",
                    AsyncMock(side_effect=[
                        {"suitable": False, "confidence": 0.5, "reason": "no visual connection"},
                        {"suitable": True, "confidence": 0.9, "start_seconds": 4,
                         "end_seconds": 14, "analyzer": "test"},
                    ]),
                ) as analyze,
                patch.object(
                    web_footage,
                    "_download_youtube",
                    AsyncMock(side_effect=fake_download),
                ) as download,
                patch.object(
                    web_footage,
                    "_probe",
                    AsyncMock(return_value={"duration_seconds": 10, "width": 1280, "height": 720}),
                ),
                patch.object(web_footage, "_trim", AsyncMock(side_effect=fake_trim)),
                patch.object(
                    web_footage,
                    "_evidence_frames",
                    AsyncMock(return_value=["frame-01.jpg", "frame-02.jpg"]),
                ),
            ):
                result = await web_footage.supplement_web_footage(
                    task_dir=task_dir,
                    manifest=manifest,
                    query_plan=[
                        {
                            "query": "chip factory",
                            "purpose": "story",
                            "script_excerpt": "Exact persisted narration segment.",
                        }
                    ],
                    target_total=1,
                    orientation="landscape",
                    script="The chip factory is expanding production.",
                )

        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["clips"][0]["title"], "Working candidate")
        self.assertEqual(download.await_count, 1)
        self.assertEqual(len(result["rejected_candidates"]), 1)
        self.assertEqual(search.await_count, 2)
        self.assertEqual(
            analyze.await_args.args[1], "Exact persisted narration segment."
        )
        self.assertEqual(result["errors"][0]["candidate_attempt"], 1)

    async def test_supplement_retries_next_unique_candidate_after_download_failure(self):
        with TemporaryDirectory() as directory:
            task_dir = Path(directory)
            (task_dir / "footage").mkdir()
            manifest = {"provider_id": "hybrid", "clips": [], "errors": []}
            candidates = [
                {
                    "platform": "youtube",
                    "provider": "YouTube",
                    "provider_id": "youtube-ytdlp",
                    "title": "Blocked candidate",
                    "creator": "Channel A",
                    "source_page_url": "https://www.youtube.com/watch?v=blocked",
                    "duration_seconds": 90,
                },
                {
                    "platform": "youtube",
                    "provider": "YouTube",
                    "provider_id": "youtube-ytdlp",
                    "title": "Working candidate",
                    "creator": "Channel B",
                    "source_page_url": "https://www.youtube.com/watch?v=working",
                    "duration_seconds": 90,
                },
            ]

            async def fake_download(candidate, raw_dir, _analysis):
                if candidate["source_page_url"].endswith("blocked"):
                    raise web_footage.WebFootageError("403 Forbidden")
                path = raw_dir / "working.mp4"
                path.write_bytes(b"raw-video")
                return path, True

            async def fake_trim(_raw, destination, _analysis, _orientation):
                destination.write_bytes(b"trimmed-video")

            with (
                patch.object(
                    web_footage,
                    "search_youtube",
                    AsyncMock(return_value=candidates),
                ) as search,
                patch.object(
                    web_footage,
                    "analyze_candidate_link",
                    AsyncMock(
                        return_value={
                            "start_seconds": 4,
                            "end_seconds": 14,
                            "analyzer": "test",
                            "confidence": 0.9,
                        }
                    ),
                ) as analyze,
                patch.object(
                    web_footage,
                    "_download_youtube",
                    AsyncMock(side_effect=fake_download),
                ) as download,
                patch.object(
                    web_footage,
                    "_probe",
                    AsyncMock(return_value={"duration_seconds": 10, "width": 1280, "height": 720}),
                ),
                patch.object(web_footage, "_trim", AsyncMock(side_effect=fake_trim)),
                patch.object(
                    web_footage,
                    "_evidence_frames",
                    AsyncMock(return_value=["frame-01.jpg", "frame-02.jpg"]),
                ),
            ):
                result = await web_footage.supplement_web_footage(
                    task_dir=task_dir,
                    manifest=manifest,
                    query_plan=[
                        {
                            "query": "chip factory",
                            "purpose": "story",
                            "script_excerpt": "Exact persisted narration segment.",
                        }
                    ],
                    target_total=1,
                    orientation="landscape",
                    script="The chip factory is expanding production.",
                )

        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["clips"][0]["title"], "Working candidate")
        self.assertEqual(download.await_count, 2)
        self.assertEqual(search.await_count, 2)
        self.assertEqual(
            analyze.await_args.args[1], "Exact persisted narration segment."
        )
        self.assertEqual(result["errors"][0]["candidate_attempt"], 1)

    async def test_supplement_uses_short_stock_variant_after_primary_candidates_fail(self):
        with TemporaryDirectory() as directory:
            task_dir = Path(directory)
            (task_dir / "footage").mkdir()
            manifest = {"provider_id": "hybrid", "clips": [], "errors": []}
            primary = [
                {
                    "platform": "youtube",
                    "provider": "YouTube",
                    "provider_id": "youtube-ytdlp",
                    "title": f"Blocked {index}",
                    "creator": "Channel",
                    "source_page_url": f"https://www.youtube.com/watch?v=blocked{index}",
                    "duration_seconds": 900,
                }
                for index in range(1, 4)
            ]
            stock = [{
                "platform": "youtube",
                "provider": "YouTube",
                "provider_id": "youtube-ytdlp",
                "title": "Short stock clip",
                "creator": "Stock channel",
                "source_page_url": "https://www.youtube.com/watch?v=stock",
                "duration_seconds": 20,
            }]

            async def fake_search(query):
                return stock if query.endswith("stock footage") else primary

            async def fake_download(candidate, raw_dir, _analysis):
                if "blocked" in candidate["source_page_url"]:
                    raise web_footage.WebFootageError("403 Forbidden")
                path = raw_dir / "stock.mp4"
                path.write_bytes(b"raw-video")
                return path, True

            async def fake_trim(_raw, destination, _analysis, _orientation):
                destination.write_bytes(b"trimmed-video")

            with (
                patch.object(
                    web_footage,
                    "search_youtube",
                    AsyncMock(side_effect=fake_search),
                ) as search,
                patch.object(
                    web_footage,
                    "analyze_candidate_link",
                    AsyncMock(return_value={"start_seconds": 1, "end_seconds": 11, "analyzer": "test", "confidence": 0.9}),
                ),
                patch.object(
                    web_footage,
                    "_download_youtube",
                    AsyncMock(side_effect=fake_download),
                ) as download,
                patch.object(
                    web_footage,
                    "_probe",
                    AsyncMock(return_value={"duration_seconds": 10, "width": 1280, "height": 720}),
                ),
                patch.object(web_footage, "_trim", AsyncMock(side_effect=fake_trim)),
                patch.object(
                    web_footage,
                    "_evidence_frames",
                    AsyncMock(return_value=["frame-01.jpg", "frame-02.jpg"]),
                ),
            ):
                result = await web_footage.supplement_web_footage(
                    task_dir=task_dir,
                    manifest=manifest,
                    query_plan=[{"query": "chip factory", "purpose": "story"}],
                    target_total=1,
                    orientation="landscape",
                    script="The chip factory is expanding production.",
                )

        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["clips"][0]["query"], "chip factory stock footage")
        self.assertEqual(download.await_count, 4)
        self.assertEqual(search.await_count, 4)


if __name__ == "__main__":
    unittest.main()


def test_rejection_without_timestamps_survives_normalization_and_late_read():
    verdict = {"suitable": False, "confidence": 0.9, "reason": "Unrelated drama"}
    result = web_footage._normalise_analysis(verdict, {"duration_seconds": 300})
    assert result["status"] == "rejected"
    assert "start_seconds" not in result
    turns = [{"Role": "user", "Text": "https://youtu.be/example"},
             {"Role": "assistant", "Text": json.dumps(verdict)}]
    assert web_footage._analysis_from_gemini_turns(json.dumps(turns), "https://youtu.be/example") == verdict


def test_legacy_irrelevant_and_low_confidence_analysis_is_rejected():
    for verdict in [
        {"confidence": 0.9, "reason": "no visual connection to AI development"},
        {"confidence": 0.45, "reason": "biblical themes"},
        {"suitable": "false", "confidence": 0.99},
        {"status": "fallback", "confidence": 0.25},
    ]:
        assert web_footage.analysis_rejection(verdict)


class WebFootageEmptySearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_empty_search_broadens_subject_without_dropping_narration(self):
        with TemporaryDirectory() as directory:
            with patch.object(web_footage, 'search_youtube', AsyncMock(return_value=[])) as search:
                result = await web_footage.supplement_web_footage(
                    task_dir=Path(directory), manifest={'clips': []},
                    query_plan=[{'query': 'Bilibili AI Creation Competition Project NEKO',
                                 'script_excerpt': 'Bilibili held its AI competition.'}],
                    target_total=1, orientation='landscape', script='Bilibili held its AI competition.',
                )
        self.assertEqual([call.args[0] for call in search.await_args_list],
                         ['Bilibili AI Creation Competition Project NEKO', 'Bilibili AI Creation'])
        self.assertEqual(result['requested_clip_count'], 1)
        self.assertEqual(result['status'], 'no_results')


class WebFootagePreviewTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_preview_requires_explicit_received_image_and_suitability(self):
        candidate = {'source_page_url': 'https://youtu.be/test', 'duration_seconds': 60,
                     'title': 'Test footage'}
        for verdict, accepted in [
            ({'image_received': False, 'suitable': True, 'confidence': 0.99}, False),
            ({'image_received': True, 'suitable': False, 'confidence': 0.99}, False),
            ({'image_received': True, 'suitable': True, 'confidence': 0.9,
              'visible_content': 'Robotaxi driving on city streets', 'reason': 'Relevant vehicle demo', 'selected_window': 1}, True),
        ]:
            with self.subTest(verdict=verdict), TemporaryDirectory() as directory:
                root = Path(directory)
                with (
                    patch.object(web_footage, '_download_youtube', AsyncMock(return_value=(root / 'raw.mp4', True))),
                    patch.object(web_footage, '_probe', AsyncMock(return_value={'duration_seconds': 15})),
                    patch.object(web_footage, '_trim', AsyncMock()),
                    patch.object(web_footage, '_run_command', AsyncMock()),
                    patch.object(web_footage, 'run_opencli', AsyncMock(return_value=OpenCLIResult(
                        args=[], returncode=0, stdout=json.dumps(verdict), stderr='',
                    ))) as ask,
                ):
                    if accepted:
                        result = await web_footage._analyze_candidate_preview(candidate, 'Robotaxi narration', root)
                        self.assertEqual(result['analyzer'], 'gemini-web-contact-sheet')
                        self.assertAlmostEqual(result['start_seconds'], 19.8)
                        self.assertIn('--file', ask.await_args.args[0])
                    else:
                        with self.assertRaises(web_footage.WebFootageError):
                            await web_footage._analyze_candidate_preview(candidate, 'Robotaxi narration', root)

    async def test_publisher_file_uses_original_timestamps_without_redownloading(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'publisher.mp4'
            source.touch()
            verdict = {'image_received': True, 'suitable': True, 'confidence': 0.95,
                       'visible_content': 'Actual product demonstration',
                       'reason': 'The demonstrated product matches the narration',
                       'selected_window': 2}
            with (
                patch.object(web_footage, '_download_youtube', AsyncMock()) as download,
                patch.object(web_footage, '_probe', AsyncMock(return_value={'duration_seconds': 100})),
                patch.object(web_footage, '_trim', AsyncMock()) as trim,
                patch.object(web_footage, '_run_command', AsyncMock()),
                patch.object(web_footage, 'run_opencli', AsyncMock(return_value=OpenCLIResult(
                    args=[], returncode=0, stdout=json.dumps(verdict), stderr='',
                ))),
            ):
                result = await web_footage._analyze_candidate_preview(
                    {'source_page_url': 'https://publisher.example/demo', 'duration_seconds': 100},
                    'Product demonstration', root, source_path=source,
                )
            download.assert_not_awaited()
            self.assertEqual(result['start_seconds'], 67)
            self.assertEqual(trim.await_args_list[2].args[0], source.resolve())
            self.assertEqual(trim.await_args_list[2].args[2]['start_seconds'], 67)

    async def test_publisher_file_must_belong_to_task(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                await web_footage._analyze_candidate_preview({}, '', root / 'task', source_path=root / 'outside.mp4')
            with self.assertRaises(web_footage.WebFootageError):
                await web_footage._analyze_candidate_preview({}, '', root, source_path=root / 'missing.mp4')

import io
import json
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx

from backend import config
from backend.pipeline import tts


def wav_bytes(*, frames: int = 24_000, sample_rate: int = 24_000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\0\0" * frames)
    return buffer.getvalue()


def write_wav(path: Path, *, frames: int = 24_000, sample_rate: int = 24_000) -> None:
    path.write_bytes(wav_bytes(frames=frames, sample_rate=sample_rate))


class GenerateTtsTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def verified_report(_path, text, _verification_dir, **_kwargs):
        count = len(tts._lexical_tokens(text))
        return {
            "verified": True,
            "expected_words": count,
            "transcript_words": count,
            "matched_exact_words": count,
            "exact_asr_word_coverage": 1.0,
            "leading_anchor": True,
            "trailing_anchor": True,
            "failure_reasons": [],
        }

    def test_split_tts_text_prefers_sentence_boundaries(self):
        text = "One two three. Four five. Six seven eight."

        chunks = tts._split_tts_text(text, max_words=5)

        self.assertEqual(chunks, ["One two three.\nFour five.", "Six seven eight."])
        self.assertEqual(" ".join(" ".join(chunks).split()), text)

    def test_split_tts_text_repeats_dialogue_metadata_without_repeating_words(self):
        text = "Speaker 1: One two three. Four five six.\nSpeaker 2: Seven eight."

        chunks = tts._split_tts_text(
            text, max_words=4, preserve_speaker_labels=True
        )

        self.assertEqual(
            chunks,
            [
                "Speaker 1: One two three.",
                "Speaker 1: Four five six.",
                "Speaker 2: Seven eight.",
            ],
        )
        self.assertEqual(
            " ".join(tts._strip_speaker_labels("\n".join(chunks)).split()),
            "One two three. Four five six. Seven eight.",
        )

    def test_split_tts_text_avoids_dangling_boundary(self):
        text = "One two three four five the six seven eight."

        chunks = tts._split_tts_text(text, max_words=6)

        self.assertEqual(chunks, ["One two three four five the six seven eight."])
        self.assertEqual(" ".join(" ".join(chunks).split()), text)

    def test_split_tts_text_keeps_fully_with_the_modified_verb(self):
        text = (
            "He knew he had to gain the upper hand before America fully committed, "
            "because once it did, the math changed forever."
        )

        chunks = tts._split_tts_text(text, max_words=12)

        self.assertEqual(
            chunks,
            [
                "He knew he had to gain the upper hand before America",
                "fully committed, because once it did, the math changed forever.",
            ],
        )
        self.assertEqual(" ".join(" ".join(chunks).split()), text)

    def test_split_tts_text_attaches_short_sentence_tail(self):
        text = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen."

        chunks = tts._split_tts_text(text, max_words=12)

        self.assertEqual(
            chunks,
            [
                "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen.",
            ],
        )
        self.assertEqual(" ".join(" ".join(chunks).split()), text)

    def test_split_tts_text_avoids_dangling_quantifier_after_clause(self):
        text = (
            "Japan took tea ceremony, calligraphy, the game of go, flower "
            "arrangement, all of it, absorbed it, made it their own."
        )

        chunks = tts._split_tts_text(text, max_words=12)

        self.assertEqual(
            chunks,
            [
                "Japan took tea ceremony, calligraphy, the game of go, flower arrangement,",
                "all of it, absorbed it, made it their own.",
            ],
        )
        self.assertEqual(" ".join(" ".join(chunks).split()), text)

    def test_split_tts_text_rechecks_dangling_tail_after_moving_leading_of(self):
        text = (
            "Nikkei Asia reports that China is restricting or delaying exports "
            "to Taiwan of germanium-based and quartz-based materials used in "
            "fiber optics, photonics, and chip manufacturing."
        )

        chunks = tts._split_tts_text(text, max_words=12)

        self.assertEqual(
            chunks[:2],
            [
                "Nikkei Asia reports that China is restricting or delaying exports",
                "to Taiwan of germanium-based and quartz-based materials used in "
                "fiber optics, photonics, and chip manufacturing.",
            ],
        )
        self.assertFalse(
            any(
                chunk.rstrip(".,!?;:\"'’”()[]{}").split()[-1].casefold()
                in tts.DANGLING_CHUNK_WORDS
                for chunk in chunks[:-1]
            )
        )
        self.assertEqual(" ".join(" ".join(chunks).split()), text)

    def test_split_tts_text_does_not_start_chunk_with_attached_preposition(self):
        text = (
            "So what you have is a country that imported the aesthetics of "
            "Chinese civilization but rejected its control mechanisms."
        )

        chunks = tts._split_tts_text(text, max_words=12)

        self.assertEqual(
            chunks,
            [
                "So what you have is a country that imported",
                "the aesthetics of Chinese civilization but rejected its control mechanisms.",
            ],
        )
        self.assertEqual(" ".join(" ".join(chunks).split()), text)

    def test_split_tts_text_separates_mirrored_parallel_clauses(self):
        text = (
            "Japanese go to war prepared to die, "
            "Chinese go to war prepared to win."
        )

        chunks = tts._split_tts_text(text, max_words=12)

        self.assertEqual(
            chunks,
            [
                "Japanese go to war prepared to die,",
                "Chinese go to war prepared to win.",
            ],
        )
        self.assertEqual(" ".join(" ".join(chunks).split()), text)

    def test_split_tts_text_keeps_preposition_with_proper_noun_clause_object(self):
        text = (
            "Now here's where the comparison gets really uncomfortable, because "
            "you look at Germany, same era, same kind of disciplined, formidable, "
            "militaristic society, and you'd think, same story, right?"
        )

        chunks = tts._split_tts_text(text, max_words=12)

        self.assertEqual(
            chunks,
            [
                "Now here's where the comparison gets really uncomfortable, "
                "because you look at Germany,",
                "same era, same kind of disciplined, formidable, militaristic "
                "society, and you'd think, same story, right?",
            ],
        )
        self.assertEqual(" ".join(" ".join(chunks).split()), text)

    def test_orpheus_prompt_adds_only_unspoken_terminal_punctuation(self):
        self.assertEqual(tts._orpheus_prompt_text("A short open phrase"), "A short open phrase.")
        self.assertEqual(tts._orpheus_prompt_text("Already complete!"), "Already complete!")
        self.assertEqual(tts._orpheus_prompt_text("A complete clause,"), "A complete clause.")

    def test_orpheus_prompt_articulates_qbitai_publication_name(self):
        text = "QbitAI reports the result"

        self.assertEqual(
            tts._orpheus_prompt_text(text),
            "Q-bit A-I reports the result.",
        )

    def test_orpheus_prompt_articulates_leading_describes_inflection(self):
        text = "describes situations where employees have promising ideas"

        self.assertEqual(
            tts._orpheus_prompt_text(text),
            "describes. Situations where employees have promising ideas.",
        )
        self.assertEqual(
            tts._orpheus_prompt_text("describes outcomes from the research"),
            "describes outcomes from the research.",
        )

    def test_orpheus_prompt_articulates_failed_opening_months_inflection(self):
        text = (
            "Months of obsessive engineering, intelligence gathering, training, "
            "and planning so meticulous it's almost uncomfortable to admire."
        )

        prompt = tts._orpheus_prompt_text(text)

        self.assertEqual(
            prompt,
            "Month-s of obsessive engineering, intelligence gathering, training, "
            "and planning so meticulous it's almost uncomfortable to admire.",
        )

    def test_orpheus_prompt_separates_declaring_from_noun_list_prosody(self):
        text = "Proxy conflicts, aid without troops, arming without declaring."

        prompt = tts._orpheus_prompt_text(text)

        self.assertEqual(
            prompt,
            "Proxy conflicts. Aid without troops. Arming without declaring.",
        )
        self.assertEqual(tts._lexical_tokens(prompt), tts._lexical_tokens(text))

    def test_orpheus_prompt_adds_pause_for_fragile_disciplined_inflection(self):
        text = "same kind of disciplined, formidable, militaristic society,"

        self.assertEqual(
            tts._orpheus_prompt_text(text),
            "same kind of disciplined. Formidable, militaristic society.",
        )

    def test_orpheus_prompt_articulates_fragile_passed_inflection(self):
        text = "and they passed that love to their kids."

        self.assertEqual(
            tts._orpheus_prompt_text(text),
            "and they passed. That love to their kids.",
        )

    def test_orpheus_prompt_exposes_disproportionate_morpheme_boundary(self):
        text = "Why does this island produce disproportionate art,"

        self.assertEqual(
            tts._orpheus_prompt_text(text),
            "Why does this island produce dis-proportionate art.",
        )

    def test_orpheus_transcript_accepts_only_exact_disproportionate_morphemes(self):
        expected = "This island produces disproportionate art."

        correct = "This island produces dis proportionate art".split()
        wrong = "This island produces dis precautionate art".split()
        correct_words = [
            {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
            for index, word in enumerate(correct)
        ]
        wrong_words = [
            {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
            for index, word in enumerate(wrong)
        ]

        self.assertTrue(tts._orpheus_transcript_report(expected, correct_words)["verified"])
        self.assertFalse(tts._orpheus_transcript_report(expected, wrong_words)["verified"])

    def test_split_tts_text_separates_repeated_clause_openings(self):
        text = (
            "no matter how suicidal, no matter how strategically insane.\n"
            "Same discipline."
        )

        self.assertEqual(
            tts._split_tts_text(text, max_words=12),
            [
                "no matter how suicidal,",
                "no matter how strategically insane.\nSame discipline.",
            ],
        )
        self.assertEqual(
            " ".join(" ".join(tts._split_tts_text(text, max_words=12)).split()),
            " ".join(text.split()),
        )

    def test_split_tts_text_keeps_past_tense_continuation_with_subject(self):
        text = (
            "They grew up on Japanese animation, and they love it, and they "
            "passed that love to their kids."
        )

        chunks = tts._split_tts_text(text, max_words=12)

        self.assertEqual(
            chunks,
            [
                "They grew up on Japanese animation, and they love it,",
                "and they passed that love to their kids.",
            ],
        )
        self.assertEqual(" ".join(" ".join(chunks).split()), text)

    def test_split_tts_text_separates_repeated_adjective_items(self):
        text = (
            "Why does this one island produce disproportionate art, "
            "disproportionate culture, disproportionate aesthetic influence?"
        )

        chunks = tts._split_tts_text(text, max_words=12)

        self.assertEqual(
            chunks,
            [
                "Why does this one island produce disproportionate art,",
                "disproportionate culture, disproportionate aesthetic influence?",
            ],
        )
        self.assertEqual(" ".join(" ".join(chunks).split()), text)

    def test_split_tts_text_keeps_battle_ready_death_cult_together(self):
        text = (
            "Japan is simultaneously the most reserved society on earth, hidden "
            "chimneys, kneeling on floors, whispering on trains, and the most "
            "unrestrained, a battle-ready death cult, sexually open, transformed "
            "by drink, willing to cross any line in art or war."
        )

        chunks = tts._split_tts_text(text, max_words=12)

        self.assertEqual(
            chunks,
            [
                "Japan is simultaneously the most reserved society on earth, "
                "hidden chimneys, kneeling",
                "on floors, whispering on trains, and the most unrestrained,",
                "a battle-ready death cult, sexually open, transformed by drink,",
                "willing to cross any line in art or war.",
            ],
        )
        self.assertEqual(" ".join(" ".join(chunks).split()), text)

    def test_split_tts_text_separates_repeated_moderation_items(self):
        text = (
            "And moderation is beautiful, moderation is stable, moderation builds "
            "enduring civilizations."
        )

        chunks = tts._split_tts_text(text, max_words=12)

        self.assertEqual(
            chunks,
            [
                "And moderation is beautiful,",
                "moderation is stable,",
                "moderation builds enduring civilizations.",
            ],
        )
        self.assertEqual(" ".join(" ".join(chunks).split()), text)

    async def test_resolves_application_paths_before_changing_cwd(self):
        captured = {}

        async def fake_stream_subprocess(**kwargs):
            captured.update(kwargs)
            command = kwargs["command"][2]
            output_dir = Path(command.split('--output_dir "', 1)[1].split('"', 1)[0])
            input_path = Path(command.split('--txt_path "', 1)[1].split('"', 1)[0])
            write_wav(output_dir / f"{input_path.stem}_generated.wav", frames=10)
            return 0, ""

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            root = Path(temp_dir)
            script_path = root / "script.txt"
            script_path.write_text("Speaker 1: Hello")
            output_dir = root / "audio"

            runtime = {
                "env_script": root / "env.sh",
                "project_dir": root / "vibevoice",
                "inference_script": root / "vibevoice" / "infer.py",
                "speaker_flag": "--speaker_name",
                "single_speaker": True,
            }
            runtime["env_script"].write_text("")
            runtime["project_dir"].mkdir()
            runtime["inference_script"].write_text("")

            with (
                patch.dict(config.TTS_MODELS, {"test-model": runtime}, clear=True),
                patch.object(tts, "stream_subprocess", fake_stream_subprocess),
            ):
                result = await tts.generate_tts(
                    str(script_path.relative_to(Path.cwd())),
                    str(output_dir.relative_to(Path.cwd())),
                    voices=["Alice"],
                    tts_model="test-model",
                )

        command = captured["command"][2]
        self.assertIn(f'--txt_path "{script_path.resolve().parent / "audio" / "tts_input.txt"}"', command)
        self.assertIn(f'--output_dir "{output_dir.resolve()}"', command)
        self.assertEqual(captured["cwd"], runtime["project_dir"])
        self.assertTrue(Path(result).is_absolute())

    async def test_reports_missing_runtime_before_spawning(self):
        runtime = {
            "env_script": Path("/missing/env.sh"),
            "project_dir": Path("/missing/vibevoice"),
            "inference_script": Path("/missing/vibevoice/infer.py"),
            "speaker_flag": "--speaker_name",
            "single_speaker": True,
        }
        with patch.dict(config.TTS_MODELS, {"missing": runtime}, clear=True):
            with self.assertRaisesRegex(RuntimeError, r"Run the local app"):
                await tts.generate_tts(
                    "script.txt",
                    "audio",
                    voices=["Alice"],
                    tts_model="missing",
                )

    async def test_applies_model_specific_voice_alias(self):
        captured = {}

        async def fake_stream_subprocess(**kwargs):
            captured.update(kwargs)
            command = kwargs["command"][2]
            output_dir = Path(command.split('--output_dir "', 1)[1].split('"', 1)[0])
            input_path = Path(command.split('--txt_path "', 1)[1].split('"', 1)[0])
            write_wav(output_dir / f"{input_path.stem}_generated.wav", frames=10)
            return 0, ""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            script_path = root / "script.txt"
            script_path.write_text("Hello")
            runtime = {
                "env_script": root / "env.sh",
                "project_dir": root / "vibevoice",
                "inference_script": root / "vibevoice" / "infer.py",
                "speaker_flag": "--speaker_name",
                "single_speaker": True,
                "voice_aliases": {"Alice": "Emma"},
            }
            runtime["env_script"].write_text("")
            runtime["project_dir"].mkdir()
            runtime["inference_script"].write_text("")

            with (
                patch.dict(config.TTS_MODELS, {"test-model": runtime}, clear=True),
                patch.object(tts, "stream_subprocess", fake_stream_subprocess),
            ):
                await tts.generate_tts(
                    str(script_path),
                    str(root / "audio"),
                    voices=["Alice"],
                    tts_model="test-model",
                )

        self.assertIn('--speaker_name "Emma"', captured["command"][2])

    async def test_chunks_long_scripts_and_concatenates_outputs(self):
        calls = []

        async def fake_stream_subprocess(**kwargs):
            calls.append(kwargs)
            command = kwargs["command"][2]
            output_dir = Path(command.split('--output_dir "', 1)[1].split('"', 1)[0])
            input_path = Path(command.split('--txt_path "', 1)[1].split('"', 1)[0])
            write_wav(output_dir / f"{input_path.stem}_generated.wav", frames=10)
            return 0, ""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            script_path = root / "script.txt"
            script_path.write_text("One two three. Four five six. Seven eight nine.")
            runtime = {
                "env_script": root / "env.sh",
                "project_dir": root / "vibevoice",
                "inference_script": root / "vibevoice" / "infer.py",
                "speaker_flag": "--speaker_name",
                "single_speaker": True,
            }
            runtime["env_script"].write_text("")
            runtime["project_dir"].mkdir()
            runtime["inference_script"].write_text("")

            with (
                patch.dict(config.TTS_MODELS, {"test-model": runtime}, clear=True),
                patch.object(config, "VIBEVOICE_TTS_CHUNK_WORDS", 3),
                patch.object(config, "TTS_RANDOM_SEED", 42),
                patch.object(tts, "stream_subprocess", fake_stream_subprocess),
            ):
                result = await tts.generate_tts(
                    str(script_path),
                    str(root / "audio"),
                    voices=["Alice"],
                    tts_model="test-model",
                )

            self.assertEqual(tts._read_pcm_wav(Path(result)).frame_count, 30)
            self.assertEqual(
                (root / "audio" / "tts_input.txt").read_text(),
                script_path.read_text(),
            )
            self.assertEqual(
                [call["name"] for call in calls],
                ["TTS part 1/3", "TTS part 2/3", "TTS part 3/3"],
            )
            self.assertIn("tts_seeded_runner.py", calls[0]["command"][2])
            manifest = json.loads((root / "audio" / "tts_manifest.json").read_text())
            self.assertEqual(manifest["chunk_count"], 3)
            self.assertEqual(manifest["output_wav"]["frame_count"], 30)
            self.assertEqual(
                [
                    len(path.read_text().split())
                    for path in sorted(
                        (root / "audio").glob("tts_input_part_*.txt")
                    )
                ],
                [3, 3, 3],
            )

    async def test_orpheus_submits_polls_and_downloads_wav(self):
        requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            if request.method == "POST":
                return httpx.Response(202, json={"id": "job-1", "status": "queued"})
            if request.url.path.endswith("/audio"):
                return httpx.Response(200, content=wav_bytes())
            return httpx.Response(200, json={"id": "job-1", "status": "completed"})

        original_client = httpx.AsyncClient

        def client_factory(**kwargs):
            return original_client(transport=httpx.MockTransport(handler), **kwargs)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            script = root / "script.txt"
            script.write_text("Speaker 1: Remote narration.")
            with (
                patch.object(config, "ORPHEUS_TTS_API_KEY", "test-secret"),
                patch.object(tts.httpx, "AsyncClient", client_factory),
                patch.object(
                    tts,
                    "_verify_orpheus_part",
                    AsyncMock(side_effect=self.verified_report),
                ),
            ):
                result = await tts.generate_tts(
                    str(script), str(root / "audio"), ["tara"], "orpheus-en"
                )

            self.assertEqual(tts._read_pcm_wav(Path(result)).duration_seconds, 1.0)
            submitted = __import__("json").loads(requests[0].content)
            self.assertEqual(submitted["voice_id"], "tara")
            self.assertEqual(submitted["input"], "Remote narration.")
            self.assertEqual(submitted["temperature"], 0.8)
            self.assertEqual(submitted["top_p"], 0.95)
            self.assertEqual(submitted["top_k"], 40)
            self.assertEqual(submitted["min_p"], 0.05)
            self.assertEqual(requests[0].headers["X-API-Key"], "test-secret")
            self.assertEqual(
                [request.url.path for request in requests],
                [
                    "/v1/audio/jobs",
                    "/v1/audio/jobs/job-1",
                    "/v1/audio/jobs/job-1/audio",
                ],
            )

    async def test_orpheus_poll_recovers_from_read_timeout_without_resubmitting(self):
        requests = []
        poll_attempts = 0
        messages = []

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal poll_attempts
            requests.append(request)
            if request.method == "POST":
                return httpx.Response(202, json={"id": "job-poll-retry"})
            if request.url.path.endswith("/audio"):
                return httpx.Response(200, content=wav_bytes())
            poll_attempts += 1
            if poll_attempts == 1:
                raise httpx.ReadTimeout("", request=request)
            if poll_attempts == 2:
                return httpx.Response(429, json={"detail": "retry later"})
            return httpx.Response(200, json={"status": "completed"})

        original_client = httpx.AsyncClient

        def client_factory(**kwargs):
            return original_client(transport=httpx.MockTransport(handler), **kwargs)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            script = root / "script.txt"
            script.write_text("Speaker 1: Retry the existing job.")
            sleeper = AsyncMock()
            with (
                patch.object(config, "ORPHEUS_TTS_API_KEY", "test-secret"),
                patch.object(tts.httpx, "AsyncClient", client_factory),
                patch.object(tts, "asyncio", SimpleNamespace(sleep=sleeper)),
                patch.object(
                    tts,
                    "_verify_orpheus_part",
                    AsyncMock(side_effect=self.verified_report),
                ),
            ):
                result = await tts.generate_tts(
                    str(script), str(root / "audio"), ["tara"], "orpheus-en",
                    log=messages.append,
                )
                result_exists = Path(result).is_file()

        self.assertTrue(result_exists)
        self.assertEqual(sum(request.method == "POST" for request in requests), 1)
        self.assertEqual(
            [request.url.path for request in requests if request.method == "GET"],
            [
                "/v1/audio/jobs/job-poll-retry",
                "/v1/audio/jobs/job-poll-retry",
                "/v1/audio/jobs/job-poll-retry",
                "/v1/audio/jobs/job-poll-retry/audio",
            ],
        )
        self.assertTrue(any("ReadTimeout" in message for message in messages))
        self.assertTrue(any("HTTPStatusError" in message for message in messages))
        self.assertEqual(sleeper.await_count, 2)

    async def test_orpheus_poll_fails_only_after_continuous_stall_timeout(self):
        requests = []
        messages = []
        poll_attempts = 0

        class FakeClock:
            def __init__(self):
                self.now = 0.0

            def monotonic(self):
                return self.now

            async def sleep(self, seconds):
                self.now += seconds

        clock = FakeClock()

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal poll_attempts
            requests.append(request)
            if request.method == "POST":
                return httpx.Response(202, json={"id": "job-poll-stall"})
            poll_attempts += 1
            if poll_attempts % 2 == 0:
                return httpx.Response(200, json={})
            raise httpx.ReadTimeout("", request=request)

        original_client = httpx.AsyncClient

        def client_factory(**kwargs):
            return original_client(transport=httpx.MockTransport(handler), **kwargs)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            script = root / "script.txt"
            script.write_text("Speaker 1: Keep polling the existing job.")
            with (
                patch.object(config, "ORPHEUS_TTS_API_KEY", "test-secret"),
                patch.object(config, "TTS_TIMEOUT", 60),
                patch.object(config, "TTS_STALL_TIMEOUT", 3),
                patch.object(config, "ORPHEUS_TTS_POLL_SECONDS", 1),
                patch.object(tts.httpx, "AsyncClient", client_factory),
                patch.object(tts, "time", clock),
                patch.object(tts, "asyncio", clock),
            ):
                with self.assertRaises(TimeoutError) as caught:
                    await tts.generate_tts(
                        str(script), str(root / "audio"), ["tara"], "orpheus-en",
                        log=messages.append,
                    )

        error = str(caught.exception)
        self.assertIn("job-poll-stall", error)
        self.assertIn("no successful poll for 3s", error)
        self.assertIn("ReadTimeout", error)
        self.assertEqual(sum(request.method == "POST" for request in requests), 1)
        self.assertEqual(
            {request.url.path for request in requests if request.method == "GET"},
            {"/v1/audio/jobs/job-poll-stall"},
        )
        self.assertTrue(any("ReadTimeout" in message for message in messages))
        self.assertTrue(any("ValueError" in message for message in messages))

    async def test_orpheus_poll_fails_fast_for_permanent_4xx(self):
        for status_code in (302, 401, 404):
            with self.subTest(status_code=status_code):
                requests = []

                def handler(request: httpx.Request) -> httpx.Response:
                    requests.append(request)
                    if request.method == "POST":
                        return httpx.Response(202, json={"id": "job-poll-4xx"})
                    return httpx.Response(status_code, json={"detail": "denied"})

                original_client = httpx.AsyncClient

                def client_factory(**kwargs):
                    return original_client(
                        transport=httpx.MockTransport(handler), **kwargs
                    )

                with tempfile.TemporaryDirectory() as temp_dir:
                    root = Path(temp_dir)
                    script = root / "script.txt"
                    script.write_text("Speaker 1: Fail fast for this response.")
                    sleeper = AsyncMock()
                    with (
                        patch.object(config, "ORPHEUS_TTS_API_KEY", "test-secret"),
                        patch.object(tts.httpx, "AsyncClient", client_factory),
                        patch.object(
                            tts,
                            "asyncio",
                            SimpleNamespace(sleep=sleeper),
                        ),
                    ):
                        with self.assertRaises(RuntimeError) as caught:
                            await tts.generate_tts(
                                str(script),
                                str(root / "audio"),
                                ["tara"],
                                "orpheus-en",
                            )

                error = str(caught.exception)
                self.assertIn("HTTPStatusError", error)
                self.assertIn(f"HTTP {status_code}", error)
                self.assertEqual(len(requests), 2)
                self.assertEqual(
                    sum(request.method == "POST" for request in requests), 1
                )
                sleeper.assert_not_awaited()

    async def test_orpheus_audio_download_retries_without_resubmitting(self):
        requests = []
        download_attempts = 0
        messages = []

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal download_attempts
            requests.append(request)
            if request.method == "POST":
                return httpx.Response(202, json={"id": "job-download-retry"})
            if request.url.path.endswith("/audio"):
                download_attempts += 1
                if download_attempts == 1:
                    raise httpx.ReadTimeout("", request=request)
                return httpx.Response(200, content=wav_bytes())
            return httpx.Response(200, json={"status": "completed"})

        original_client = httpx.AsyncClient

        def client_factory(**kwargs):
            return original_client(transport=httpx.MockTransport(handler), **kwargs)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            script = root / "script.txt"
            script.write_text("Speaker 1: Retry only the audio download.")
            sleeper = AsyncMock()
            with (
                patch.object(config, "ORPHEUS_TTS_API_KEY", "test-secret"),
                patch.object(tts.httpx, "AsyncClient", client_factory),
                patch.object(tts, "asyncio", SimpleNamespace(sleep=sleeper)),
                patch.object(
                    tts,
                    "_verify_orpheus_part",
                    AsyncMock(side_effect=self.verified_report),
                ),
            ):
                result = await tts.generate_tts(
                    str(script), str(root / "audio"), ["tara"], "orpheus-en",
                    log=messages.append,
                )
                result_exists = Path(result).is_file()

        self.assertTrue(result_exists)
        self.assertEqual(sum(request.method == "POST" for request in requests), 1)
        self.assertEqual(download_attempts, 2)
        self.assertTrue(any("ReadTimeout" in message for message in messages))
        sleeper.assert_awaited_once()

    async def test_orpheus_audio_download_retries_corrupt_200_without_resubmitting(self):
        requests = []
        download_attempts = 0
        messages = []

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal download_attempts
            requests.append(request)
            if request.method == "POST":
                return httpx.Response(202, json={"id": "job-corrupt-download"})
            if request.url.path.endswith("/audio"):
                download_attempts += 1
                if download_attempts == 1:
                    return httpx.Response(200, content=b"not a wav")
                if download_attempts == 2:
                    return httpx.Response(200, content=wav_bytes()[:-100])
                return httpx.Response(200, content=wav_bytes())
            return httpx.Response(200, json={"status": "completed"})

        original_client = httpx.AsyncClient

        def client_factory(**kwargs):
            return original_client(transport=httpx.MockTransport(handler), **kwargs)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            script = root / "script.txt"
            script.write_text("Speaker 1: Retry the corrupt audio download.")
            sleeper = AsyncMock()
            with (
                patch.object(config, "ORPHEUS_TTS_API_KEY", "test-secret"),
                patch.object(tts.httpx, "AsyncClient", client_factory),
                patch.object(tts, "asyncio", SimpleNamespace(sleep=sleeper)),
                patch.object(
                    tts,
                    "_verify_orpheus_part",
                    AsyncMock(side_effect=self.verified_report),
                ),
            ):
                result = await tts.generate_tts(
                    str(script), str(root / "audio"), ["tara"], "orpheus-en",
                    log=messages.append,
                )
                result_exists = Path(result).is_file()

        self.assertTrue(result_exists)
        self.assertEqual(sum(request.method == "POST" for request in requests), 1)
        self.assertEqual(download_attempts, 3)
        self.assertTrue(any("TtsIntegrityError" in message for message in messages))
        self.assertEqual(sleeper.await_count, 2)

    async def test_orpheus_uses_token_budget_chunks_and_lossless_join(self):
        requests = []
        audio = wav_bytes(frames=1_000)

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            if request.method == "POST":
                job_id = f"job-{sum(r.method == 'POST' for r in requests)}"
                return httpx.Response(202, json={"id": job_id})
            if request.url.path.endswith("/audio"):
                return httpx.Response(200, content=audio)
            return httpx.Response(200, json={"status": "completed"})

        original_client = httpx.AsyncClient

        def client_factory(**kwargs):
            return original_client(transport=httpx.MockTransport(handler), **kwargs)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            script = root / "script.txt"
            script.write_text(" ".join(f"word{i}" for i in range(20)))
            with (
                patch.object(config, "ORPHEUS_TTS_API_KEY", "test-secret"),
                patch.object(config, "ORPHEUS_TTS_MAX_TOKENS", 512),
                patch.object(tts.httpx, "AsyncClient", client_factory),
                patch.object(
                    tts,
                    "_verify_orpheus_part",
                    AsyncMock(side_effect=self.verified_report),
                ),
                patch.object(
                    tts,
                    "_validate_wav_part",
                    side_effect=lambda path, *_args, **_kwargs: tts._read_pcm_wav(path),
                ),
            ):
                result = await tts.generate_tts(
                    str(script), str(root / "audio"), ["tara"], "orpheus-en"
                )

            posts = [request for request in requests if request.method == "POST"]
            self.assertGreater(len(posts), 1)
            submitted_text = " ".join(
                json.loads(request.content)["input"].rstrip(".") for request in posts
            )
            self.assertEqual(submitted_text, script.read_text())
            self.assertEqual(
                tts._read_pcm_wav(Path(result)).frame_count,
                len(posts) * 1_000,
            )
            manifest = json.loads((root / "audio" / "tts_manifest.json").read_text())
            self.assertEqual(manifest["source_word_count"], 20)
            self.assertEqual(manifest["chunk_count"], len(posts))
            self.assertEqual(manifest["integrity"]["verified_source_coverage"], 1.0)

    async def test_verified_orpheus_audio_reaching_ceiling_gets_acoustic_check(self):
        requests = []
        audio = wav_bytes(frames=12 * 24_000)

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            if request.method == "POST":
                return httpx.Response(202, json={"id": "job-ceiling"})
            if request.url.path.endswith("/audio"):
                return httpx.Response(200, content=audio)
            return httpx.Response(200, json={"status": "completed"})

        original_client = httpx.AsyncClient

        def client_factory(**kwargs):
            return original_client(transport=httpx.MockTransport(handler), **kwargs)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            script = root / "script.txt"
            script.write_text("One complete short sentence reaches the model token ceiling.")
            verifier = AsyncMock(side_effect=self.verified_report)
            with (
                patch.object(config, "ORPHEUS_TTS_API_KEY", "test-secret"),
                patch.object(config, "ORPHEUS_TTS_MAX_TOKENS", 1_024),
                patch.object(tts.httpx, "AsyncClient", client_factory),
                patch.object(tts, "_verify_orpheus_part", verifier),
            ):
                await tts.generate_tts(
                    str(script), str(root / "audio"), ["tara"], "orpheus-en"
                )

            verifier.assert_awaited_once()

    def test_orpheus_transcript_report_rejects_audio_that_skips_the_opening(self):
        expected = (
            "The opening sentence must be present. "
            "The middle sentence must also be present. "
            "The closing sentence must be present."
        )
        words = [
            {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
            for index, word in enumerate(
                "Middle sentence must also be present The closing sentence must be present".split()
            )
        ]

        report = tts._orpheus_transcript_report(expected, words)

        self.assertFalse(report["verified"])
        self.assertIn("opening words", " ".join(report["failure_reasons"]))

    def test_orpheus_transcript_report_accepts_complete_short_utterance(self):
        expected = "Every requested word remains in this finished audio sentence."
        words = [
            {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
            for index, word in enumerate(expected.rstrip(".").split())
        ]

        report = tts._orpheus_transcript_report(expected, words)

        self.assertTrue(report["verified"])
        self.assertEqual(report["exact_asr_word_coverage"], 1.0)

    async def test_orpheus_verifier_accepts_exact_two_word_transcript(self):
        words = [
            {"text": "to", "start": 0.0, "end": 0.2},
            {"text": "die", "start": 0.2, "end": 0.5},
        ]
        with patch(
            "backend.pipeline.av_sync.ensure_word_transcript",
            AsyncMock(return_value=(words, {"passed": False, "failure_reasons": ["only 2 words"]})),
        ):
            report = await tts._verify_orpheus_part(
                Path("short.wav"),
                "to die.",
                Path("verification"),
                emit=lambda _message: None,
            )

        self.assertTrue(report["verified"])
        self.assertEqual(report["matched_exact_words"], 2)

    def test_orpheus_transcript_normalizes_numeric_ordinals(self):
        expected = "A scrap of land one-thirtieth the size."
        spoken = "A scrap of land 1 30th the size".split()
        words = [
            {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
            for index, word in enumerate(spoken)
        ]

        report = tts._orpheus_transcript_report(expected, words)

        self.assertTrue(report["verified"])
        self.assertEqual(report["exact_asr_word_coverage"], 1.0)

    def test_orpheus_transcript_normalizes_numeric_calendar_ordinal(self):
        expected = (
            "QbitAI reports that Perfect World's 2026 semiannual report, "
            "published August 19, shows"
        )
        observed = (
            "Qubit AI reports that Perfect World's 2026 semi -annual report "
            "published August 19th shows"
        ).split()
        words = [
            {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
            for index, word in enumerate(observed)
        ]

        report = tts._orpheus_transcript_report(expected, words)

        self.assertTrue(report["verified"])
        self.assertEqual(report["exact_asr_word_coverage"], 1.0)

    def test_orpheus_transcript_does_not_normalize_non_date_ordinal(self):
        expected = "Version 19 remains stable."
        observed = "Version 19th remains stable".split()
        words = [
            {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
            for index, word in enumerate(observed)
        ]

        report = tts._orpheus_transcript_report(expected, words)

        self.assertFalse(report["verified"])
        self.assertEqual(report["exact_asr_word_coverage"], 0.75)

    def test_orpheus_transcript_normalizes_spoken_year(self):
        expected = "First gamble, eighteen ninety-five. Japan goes to war."
        observed = "First gamble 1895 Japan goes to war".split()
        words = [
            {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
            for index, word in enumerate(observed)
        ]

        report = tts._orpheus_transcript_report(expected, words)

        self.assertTrue(report["verified"])
        self.assertEqual(report["exact_asr_word_coverage"], 1.0)

    def test_orpheus_transcript_normalizes_oh_year(self):
        expected = "Second gamble, nineteen oh five. Japan goes to war."
        observed = "Second gamble 1905 Japan goes to war".split()
        words = [
            {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
            for index, word in enumerate(observed)
        ]

        report = tts._orpheus_transcript_report(expected, words)

        self.assertTrue(report["verified"])
        self.assertEqual(report["exact_asr_word_coverage"], 1.0)

    def test_orpheus_transcript_accepts_exact_homophone_but_not_near_homophone(self):
        expected = "The ground under your feet has killed people."

        feat_words = [
            {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
            for index, word in enumerate(
                "The ground under your feat has killed people".split()
            )
        ]
        feed_words = [
            {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
            for index, word in enumerate(
                "The ground under your feed has killed people".split()
            )
        ]

        self.assertTrue(tts._orpheus_transcript_report(expected, feat_words)["verified"])
        self.assertFalse(tts._orpheus_transcript_report(expected, feed_words)["verified"])

    def test_orpheus_transcript_accepts_its_contraction_homophone(self):
        expected = "The things we admire in a culture, its art."
        observed = "The things we admire in a culture it's art".split()
        words = [
            {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
            for index, word in enumerate(observed)
        ]

        report = tts._orpheus_transcript_report(expected, words)

        self.assertTrue(report["verified"])
        self.assertEqual(report["exact_asr_word_coverage"], 1.0)

    def test_orpheus_transcript_accepts_your_contraction_homophone_only(self):
        expected = (
            "It's Thursday, August 20, 2026, and this is Frontier Tech "
            "Daily—your concise"
        )

        def report(observed: str) -> dict:
            return tts._orpheus_transcript_report(
                expected,
                [
                    {
                        "text": word,
                        "start": index * 0.2,
                        "end": index * 0.2 + 0.1,
                    }
                    for index, word in enumerate(observed.split())
                ],
            )

        live_whisper_text = (
            "It's Thursday August 20 2026 and this is Frontier Tech Daily "
            "You're concise"
        )
        accepted = report(live_whisper_text)

        self.assertTrue(accepted["verified"])
        self.assertEqual(accepted["expected_words"], 13)
        self.assertEqual(accepted["transcript_words"], 13)
        self.assertEqual(accepted["exact_asr_word_coverage"], 1.0)
        for rejected_text in (
            "It's Thursday August 20 2026 and this is Frontier Tech Daily concise",
            "It's Thursday August 20 2026 and this is Frontier Tech Daily our concise",
            "It's Thursday August 20 2026 and this is Frontier Tech Daily you concise",
            "It's Thursday August 20 2026 and this is Frontier Tech Daily your your concise",
        ):
            with self.subTest(rejected_text=rejected_text):
                self.assertFalse(report(rejected_text)["verified"])

    def test_orpheus_transcript_normalizes_break_through_compound_spelling(self):
        expected = "break through isolationist resistance in Congress."
        observed = "Breakthrough isolationist resistance in Congress".split()
        words = [
            {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
            for index, word in enumerate(observed)
        ]

        report = tts._orpheus_transcript_report(expected, words)

        self.assertTrue(report["verified"])
        self.assertEqual(report["exact_asr_word_coverage"], 1.0)

    def test_orpheus_transcript_normalizes_deeptech_publication_name(self):
        expected = (
            "DeepTech China reports that researchers at MIT are exploring "
            "whether living bacteria"
        )
        observed = (
            "Deep Tech China reports that researchers at MIT are exploring "
            "whether living bacteria"
        ).split()
        words = [
            {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
            for index, word in enumerate(observed)
        ]

        report = tts._orpheus_transcript_report(expected, words)

        self.assertTrue(report["verified"])
        self.assertEqual(report["expected_words"], 12)
        self.assertEqual(report["transcript_words"], 12)
        self.assertEqual(report["exact_asr_word_coverage"], 1.0)

    def test_orpheus_transcript_normalizes_qbitai_and_semiannual(self):
        expected = (
            "QbitAI reports that Perfect World's 2026 semiannual report, "
            "published August 19, shows"
        )
        observed = (
            "Qubit AI reports that Perfect World's 2026 semi-annual report, "
            "published August 19, shows"
        ).split()
        words = [
            {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
            for index, word in enumerate(observed)
        ]

        report = tts._orpheus_transcript_report(expected, words)

        self.assertTrue(report["verified"])
        self.assertEqual(report["expected_words"], 12)
        self.assertEqual(report["transcript_words"], 12)
        self.assertEqual(report["exact_asr_word_coverage"], 1.0)

    def test_orpheus_transcript_normalizes_world_possessive_spelling(self):
        expected = (
            "QbitAI reports that Perfect World's 2026 semiannual report, "
            "published August 19, shows"
        )
        observed = (
            "Qubit AI reports that Perfect Worlds' 2026 semiannual report, "
            "published August 19, shows"
        ).split()
        words = [
            {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
            for index, word in enumerate(observed)
        ]

        report = tts._orpheus_transcript_report(expected, words)

        self.assertTrue(report["verified"])
        self.assertEqual(report["expected_words"], 12)
        self.assertEqual(report["transcript_words"], 12)
        self.assertEqual(report["exact_asr_word_coverage"], 1.0)

    def test_orpheus_transcript_normalizes_brem_silent_h_spelling_only(self):
        expected = (
            "The article, authored by Alexander Brem, editor in chief of "
            "IEEE Engineering"
        )

        def report(name: str) -> dict:
            observed = (
                f"The article authored by Alexander {name} editor in chief of "
                "IEEE Engineering"
            ).split()
            words = [
                {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
                for index, word in enumerate(observed)
            ]
            return tts._orpheus_transcript_report(expected, words)

        accepted = report("Brehm")

        self.assertTrue(accepted["verified"])
        self.assertEqual(accepted["exact_asr_word_coverage"], 1.0)
        self.assertFalse(report("Bream")["verified"])
        self.assertFalse(report("")["verified"])

    def test_orpheus_transcript_normalizes_exact_compound_spellings(self):
        cases = (
            ("Brem's research pays off.", "Brehm's research pays off."),
            (
                "including bootlegging and skunkworks projects",
                "including bootlegging and Skunk Works projects",
            ),
            ("Google and 3M allocate work time", "Google and 3 M allocate work time"),
            ("multimodal content generation", "multi modal content generation"),
        )
        for expected, observed in cases:
            with self.subTest(expected=expected, observed=observed):
                words = [
                    {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
                    for index, word in enumerate(observed.split())
                ]
                report = tts._orpheus_transcript_report(expected, words)
                self.assertTrue(report["verified"])
                self.assertEqual(report["exact_asr_word_coverage"], 1.0)

        wrong_words = [
            {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
            for index, word in enumerate("including bootlegging and skunk projects".split())
        ]
        self.assertFalse(
            tts._orpheus_transcript_report(
                "including bootlegging and skunkworks projects",
                wrong_words,
            )["verified"]
        )

    def test_orpheus_transcript_normalizes_role_roll_homophone_only(self):
        expected = "role in the relevant product line."

        def report(opening: str) -> dict:
            observed = f"{opening} in the relevant product line".split()
            words = [
                {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
                for index, word in enumerate(observed)
            ]
            return tts._orpheus_transcript_report(expected, words)

        accepted = report("Roll")

        self.assertTrue(accepted["verified"])
        self.assertEqual(accepted["exact_asr_word_coverage"], 1.0)
        self.assertFalse(report("roil")["verified"])
        self.assertFalse(report("")["verified"])

    def test_orpheus_transcript_normalizes_whisper_eunuch_bias(self):
        expected = "The eunuch system, not interested."
        observed = "The Unix system not interested".split()
        words = [
            {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
            for index, word in enumerate(observed)
        ]

        report = tts._orpheus_transcript_report(expected, words)

        self.assertTrue(report["verified"])
        self.assertEqual(report["exact_asr_word_coverage"], 1.0)

    def test_orpheus_transcript_normalizes_whisper_scorsese_spelling(self):
        expected = "He directly shaped Bergman, Scorsese, Tarantino, George Lucas."
        observed = "He directly shaped Bergman Suarcese Tarantino George Lucas".split()
        words = [
            {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
            for index, word in enumerate(observed)
        ]

        report = tts._orpheus_transcript_report(expected, words)

        self.assertTrue(report["verified"])
        self.assertEqual(report["exact_asr_word_coverage"], 1.0)

        observed = "He directly shaped Bergman Sorsese Tarantino George Lucas".split()
        words = [
            {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
            for index, word in enumerate(observed)
        ]
        self.assertTrue(tts._orpheus_transcript_report(expected, words)["verified"])

    def test_orpheus_transcript_normalizes_spoken_and_comma_number(self):
        expected = "Feet has killed a hundred thousand people."
        observed = "Feet has killed 100 000 people".split()
        words = [
            {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
            for index, word in enumerate(observed)
        ]

        report = tts._orpheus_transcript_report(expected, words)

        self.assertTrue(report["verified"])
        self.assertEqual(report["exact_asr_word_coverage"], 1.0)

    def test_orpheus_transcript_normalizes_scale_across_whisper_words(self):
        expected = "Loss attributable to shareholders of 118 million yuan."
        observed = "Loss attributable to shareholders of 118 million yuan".split()
        words = [
            {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
            for index, word in enumerate(observed)
        ]

        report = tts._orpheus_transcript_report(expected, words)

        self.assertTrue(report["verified"])
        self.assertEqual(report["expected_words"], 7)
        self.assertEqual(report["transcript_words"], 7)
        self.assertEqual(report["exact_asr_word_coverage"], 1.0)

        missing_scale = [word for word in words if word["text"] != "million"]
        wrong_number = [
            {**word, "text": "119" if word["text"] == "118" else word["text"]}
            for word in words
        ]
        repeated_scale = words[:6] + [{**words[6], "text": "million"}] + words[6:]

        self.assertFalse(
            tts._orpheus_transcript_report(expected, missing_scale)["verified"]
        )
        self.assertFalse(
            tts._orpheus_transcript_report(expected, wrong_number)["verified"]
        )
        self.assertFalse(
            tts._orpheus_transcript_report(expected, repeated_scale)["verified"]
        )

    def test_orpheus_transcript_normalizes_spelled_thousands(self):
        expected = (
            "three thousand nautical miles across open ocean without anyone noticing. "
            "It worked."
        )
        observed = (
            "3 000 nautical miles across open ocean without anyone noticing it worked"
        ).split()
        words = [
            {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
            for index, word in enumerate(observed)
        ]

        report = tts._orpheus_transcript_report(expected, words)

        self.assertTrue(report["verified"])
        self.assertEqual(report["expected_words"], 11)
        self.assertEqual(report["transcript_words"], 11)
        self.assertEqual(report["exact_asr_word_coverage"], 1.0)

    async def test_orpheus_integrity_failure_retries_generation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            script = root / "script.txt"
            script.write_text("A retryable utterance.")
            generate = AsyncMock(
                side_effect=[tts.TtsIntegrityError("repeated speech"), "/verified.wav"]
            )
            with (
                patch.object(config, "ORPHEUS_TTS_API_KEY", "test-secret"),
                patch.object(tts, "_generate_orpheus", generate),
            ):
                result = await tts.generate_tts(
                    str(script), str(root / "audio"), ["tara"], "orpheus-en"
                )

            self.assertEqual(result, "/verified.wav")
            self.assertEqual(generate.await_count, 2)

    async def test_orpheus_integrity_retry_budget_is_independent_per_part(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            script = root / "script.txt"
            script.write_text("Two independently retryable utterances.")
            generate = AsyncMock(
                side_effect=[
                    tts.TtsIntegrityError("bad first sample", part_key="part-001"),
                    tts.TtsIntegrityError("bad first retry", part_key="part-001"),
                    tts.TtsIntegrityError("bad second sample", part_key="part-002"),
                    tts.TtsIntegrityError("bad second retry", part_key="part-002"),
                    "/verified.wav",
                ]
            )
            with (
                patch.object(config, "ORPHEUS_TTS_API_KEY", "test-secret"),
                patch.object(tts, "_generate_orpheus", generate),
            ):
                result = await tts.generate_tts(
                    str(script), str(root / "audio"), ["tara"], "orpheus-en"
                )

            self.assertEqual(result, "/verified.wav")
            self.assertEqual(generate.await_count, 5)

    def test_orpheus_transcript_report_rejects_repeated_utterance(self):
        expected = "The complete phrase is spoken once."
        repeated = (expected.rstrip(".") + " " + expected.rstrip(".")).split()
        words = [
            {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
            for index, word in enumerate(repeated)
        ]

        report = tts._orpheus_transcript_report(expected, words)

        self.assertFalse(report["verified"])
        self.assertIn("repeated", " ".join(report["failure_reasons"]))

    def test_orpheus_transcript_report_rejects_one_extra_spoken_word(self):
        expected = "Every requested word is spoken once."
        observed = (expected.rstrip(".") + " extra").split()
        words = [
            {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
            for index, word in enumerate(observed)
        ]

        report = tts._orpheus_transcript_report(expected, words)

        self.assertFalse(report["verified"])
        self.assertIn("likely repeated", " ".join(report["failure_reasons"]))

    def test_orpheus_transcript_report_finds_truncated_second_utterance(self):
        expected = "The complete phrase is spoken exactly once."
        observed = (expected.rstrip(".") + " The complete phrase").split()
        words = [
            {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
            for index, word in enumerate(observed)
        ]

        report = tts._orpheus_transcript_report(expected, words)

        self.assertEqual(report["repeat_start_seconds"], 1.4)

    def test_orpheus_short_utterance_uses_reduced_token_floor(self):
        self.assertEqual(
            tts._orpheus_request_token_budget("one two three", 16_384),
            512,
        )

    def test_orpheus_cache_is_invalidated_when_speed_changes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "part.wav"
            write_wav(path)
            text = "A fully verified utterance."
            with patch.object(config, "ORPHEUS_TTS_SPEED_PERCENT", 100):
                tts._write_orpheus_part_metadata(
                    path,
                    text,
                    job_id="job-1",
                    request_token_budget=512,
                    integrity=self.verified_report(None, text, None),
                )
                self.assertIsNotNone(tts._load_cached_orpheus_part(path, text))
            with patch.object(config, "ORPHEUS_TTS_SPEED_PERCENT", 140):
                self.assertIsNone(tts._load_cached_orpheus_part(path, text))

    async def test_orpheus_recovers_downloaded_wav_without_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / "part.wav"
            text = "A complete recovered utterance."
            write_wav(path, frames=24_000)
            integrity = self.verified_report(None, text, None)
            messages = []
            with patch.object(
                tts,
                "_verify_orpheus_part",
                AsyncMock(return_value=integrity),
            ) as verify:
                metadata = await tts._recover_orpheus_part(
                    path,
                    text,
                    root / "verification",
                    request_token_budget=512,
                    emit=messages.append,
                )

            self.assertIsNotNone(metadata)
            self.assertEqual(metadata["job_id"], "recovered-local-output")
            self.assertIsNotNone(tts._load_cached_orpheus_part(path, text))
            verify.assert_awaited_once()
            self.assertTrue(any("accepted existing WAV" in item for item in messages))

    def test_rejects_orpheus_audio_that_reaches_token_ceiling(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "part.wav"
            token_seconds = 2_048 / tts.ORPHEUS_AUDIO_TOKENS_PER_SECOND
            write_wav(path, frames=int(token_seconds * 24_000))

            with self.assertRaisesRegex(tts.TtsIntegrityError, "max-token ceiling"):
                tts._validate_wav_part(
                    path,
                    "short input",
                    token_limit_seconds=token_seconds,
                )

    async def test_lossless_join_rejects_format_changes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "first.wav"
            second = root / "second.wav"
            write_wav(first, frames=10, sample_rate=24_000)
            write_wav(second, frames=10, sample_rate=16_000)

            with self.assertRaisesRegex(tts.TtsIntegrityError, "format changed"):
                await tts._concat_wav_parts([first, second], root, log=None)

    async def test_orpheus_requires_api_key_before_network(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            script = root / "script.txt"
            script.write_text("Hello")
            with patch.object(config, "ORPHEUS_TTS_API_KEY", ""):
                with self.assertRaisesRegex(RuntimeError, "API key is not configured"):
                    await tts.generate_tts(
                        str(script), str(root / "audio"), ["tara"], "orpheus-en"
                    )


def test_vibevoice_pronunciation_expansion_preserves_canonical_names():
    spoken, applied = tts._expand_vibevoice_pronunciations(
        "IEEE reports on Qwen and QbitAI. IEEE-style remains unchanged."
    )

    assert spoken == (
        "I triple E reports on cue-when and Q-bit A-I. "
        "IEEE-style remains unchanged."
    )
    assert applied == [
        {"canonical": "IEEE", "pronunciation": "I triple E", "occurrences": "1"},
        {"canonical": "QbitAI", "pronunciation": "Q-bit A-I", "occurrences": "1"},
        {"canonical": "Qwen", "pronunciation": "cue-when", "occurrences": "1"},
    ]


if __name__ == "__main__":
    unittest.main()

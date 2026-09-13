import io
import json
import struct
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


def pocket_streaming_wav_bytes(*, frames: int = 48_000) -> bytes:
    payload = bytearray(wav_bytes(frames=frames))
    # Match Pocket TTS' non-seekable HTTP writer: the true PCM follows a header
    # that advertises one billion placeholder frames.
    struct.pack_into("<I", payload, 4, 2_000_000_036)
    struct.pack_into("<I", payload, 40, 2_000_000_000)
    return bytes(payload)


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

    def test_pocket_split_preserves_physical_story_lines(self):
        text = (
            "Opening sentence. A second opening sentence.\n"
            "Story one stays together. It keeps its natural sentence boundary.\n"
            "Closing line."
        )

        chunks = tts._split_pocket_tts_text(text, max_words=240)

        self.assertEqual(chunks, text.splitlines())
        self.assertNotIn("Opening sentence.", chunks)

    def test_pocket_split_only_divides_long_lines_between_sentences(self):
        text = "One two three four. Five six seven eight. Nine ten eleven twelve."

        chunks = tts._split_pocket_tts_text(text, max_words=8)

        self.assertEqual(
            chunks,
            ["One two three four. Five six seven eight.", "Nine ten eleven twelve."],
        )
        self.assertEqual(" ".join(chunks), text)

    def test_pocket_split_keeps_abbreviations_inside_complete_sentences(self):
        sentences = [
            "Inspectors identified E. coli in the food.",
            "Dr. Earley spoke to U.S. manufacturers.",
            "The final sentence remains separate.",
        ]
        text = " ".join(sentences)

        chunks = tts._split_pocket_tts_text(text, max_words=5)

        self.assertEqual(chunks, sentences)
        self.assertEqual(" ".join(chunks), text)

    def test_normalizes_pocket_streaming_wav_placeholder_header(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "stream.wav"
            path.write_bytes(pocket_streaming_wav_bytes(frames=48_000))
            with wave.open(str(path), "rb") as source:
                self.assertEqual(source.getnframes(), 1_000_000_000)

            info = tts._normalize_pocket_streaming_wav(path)

            self.assertEqual(info.frame_count, 48_000)
            self.assertEqual(info.duration_seconds, 2.0)
            self.assertEqual(tts._read_pcm_wav(path).frame_count, 48_000)

    def test_measures_pocket_wav_edge_silence_in_fixed_windows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "edges.wav"
            with wave.open(str(path), "wb") as destination:
                destination.setnchannels(1)
                destination.setsampwidth(2)
                destination.setframerate(24_000)
                destination.writeframes(
                    struct.pack("<h", 0) * 2_400
                    + struct.pack("<h", 10_000) * 4_800
                    + struct.pack("<h", 0) * 7_200
                )

            edges = tts._wav_edge_silence_seconds(path)

        self.assertEqual(edges, {"leading_seconds": 0.1, "trailing_seconds": 0.3})

    def test_rejects_long_silence_inside_a_pocket_story(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "stutter.wav"
            with wave.open(str(path), "wb") as destination:
                destination.setnchannels(1)
                destination.setsampwidth(2)
                destination.setframerate(24_000)
                destination.writeframes(
                    struct.pack("<h", 10_000) * 4_800
                    + struct.pack("<h", 0) * 21_600
                    + struct.pack("<h", 10_000) * 4_800
                )

            report = tts._pocket_internal_silence_report(path)
            with self.assertRaisesRegex(tts.TtsIntegrityError, "internal pause"):
                tts._validate_pocket_internal_silence(path)

        self.assertEqual(report["count"], 1)
        self.assertEqual(report["max_seconds"], 0.9)
        self.assertFalse(report["passed"])

    def test_allows_narrow_natural_pause_at_transcribed_sentence_boundary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sentence-boundary.wav"
            with wave.open(str(path), "wb") as destination:
                destination.setnchannels(1)
                destination.setsampwidth(2)
                destination.setframerate(24_000)
                destination.writeframes(
                    struct.pack("<h", 10_000) * 4_800
                    + struct.pack("<h", 0) * 26_400
                    + struct.pack("<h", 10_000) * 4_800
                )
            words = [
                {"text": "Done.", "start": 0.0, "end": 0.2},
                {"text": "Next", "start": 1.3, "end": 1.5},
            ]

            report = tts._validate_pocket_internal_silence(path, words)

        self.assertTrue(report["passed"])
        self.assertEqual(report["max_seconds"], 1.1)
        self.assertTrue(report["runs"][0]["sentence_boundary"])
        self.assertEqual(report["runs"][0]["maximum_allowed_seconds"], 1.2)

    def test_rejects_same_pause_without_sentence_boundary_evidence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "mid-sentence.wav"
            with wave.open(str(path), "wb") as destination:
                destination.setnchannels(1)
                destination.setsampwidth(2)
                destination.setframerate(24_000)
                destination.writeframes(
                    struct.pack("<h", 10_000) * 4_800
                    + struct.pack("<h", 0) * 21_600
                    + struct.pack("<h", 10_000) * 4_800
                )
            words = [
                {"text": "Still,", "start": 0.0, "end": 0.2},
                {"text": "speaking", "start": 1.1, "end": 1.3},
            ]

            with self.assertRaisesRegex(tts.TtsIntegrityError, "internal pause"):
                tts._validate_pocket_internal_silence(path, words)

    def test_rejects_excessive_pause_even_at_sentence_boundary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "long-sentence-boundary.wav"
            with wave.open(str(path), "wb") as destination:
                destination.setnchannels(1)
                destination.setsampwidth(2)
                destination.setframerate(24_000)
                destination.writeframes(
                    struct.pack("<h", 10_000) * 4_800
                    + struct.pack("<h", 0) * 31_200
                    + struct.pack("<h", 10_000) * 4_800
                )
            words = [
                {"text": "Done.", "start": 0.0, "end": 0.2},
                {"text": "Next", "start": 1.5, "end": 1.7},
            ]

            with self.assertRaisesRegex(
                tts.TtsIntegrityError,
                "sentence-boundary pause",
            ):
                tts._validate_pocket_internal_silence(path, words)

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

    def test_split_tts_text_moves_which_to_its_relative_clause(self):
        text = (
            "The design pairs a nuclear thermal engine with electric thrusters, "
            "which the engineers say could reduce transit time down to 335 days or less."
        )

        chunks = tts._split_tts_text(text, max_words=12)

        self.assertEqual(
            chunks,
            [
                "The design pairs a nuclear thermal engine with electric thrusters,",
                "which the engineers say could reduce transit time down to 335 days or less.",
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

    def test_split_tts_text_keeps_video_with_opening_prompts_inflection(self):
        text = (
            "Chinese tech outlet QbitAI reports that S1 uses in-context learning "
            "with video prompts, extending the task length to as long as 10 minutes."
        )

        chunks = tts._split_tts_text(text, max_words=12)

        self.assertEqual(
            chunks,
            [
                "Chinese tech outlet QbitAI reports that S1 uses in-context learning",
                "with video prompts, extending the task length to as long as 10 minutes.",
            ],
        )
        self.assertEqual(" ".join(" ".join(chunks).split()), text)

    def test_split_tts_text_keeps_positioning_object_with_verb(self):
        text = (
            "The company says the model rivals Opus 4.6 and V4-Flash, and positions "
            "the release as a lower-priced platform aimed at driving adoption of "
            "its marquee AI offering globally."
        )

        chunks = tts._split_tts_text(text, max_words=12)

        self.assertEqual(
            chunks,
            [
                "The company says the model rivals Opus 4.6 and V4-Flash,",
                "and positions the release as a lower-priced platform",
                "aimed at driving adoption of its marquee AI offering globally.",
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

    def test_orpheus_prompt_articulates_qwen_as_two_part_name(self):
        text = "Alibaba's Qwen Office, known in Chinese as Qianwen"

        self.assertEqual(
            tts._orpheus_prompt_text(text),
            "Alibaba's cue-when Office, known in Chinese as Chien-Wen.",
        )
        self.assertEqual(
            tts._orpheus_prompt_text("Qwenish pre-Qwen qianwen"),
            "Qwenish pre-Qwen qianwen.",
        )
        self.assertEqual(
            tts._orpheus_prompt_text(
                "Alibaba released Qwen3.8-Flash and Qwen 4"
            ),
            "Alibaba released cue-when 3.8-Flash and cue-when four.",
        )

    def test_orpheus_prompt_articulates_v4_flash_model_prefix(self):
        self.assertEqual(
            tts._orpheus_prompt_text("The model rivals V4-Flash"),
            "The model rivals V four Flash.",
        )
        self.assertEqual(
            tts._orpheus_prompt_text("AV4-Flash remains unchanged"),
            "AV4-Flash remains unchanged.",
        )

    def test_orpheus_prompt_uses_ascii_jalapeno_provider_spelling(self):
        self.assertEqual(
            tts._orpheus_prompt_text("the chip is named Jalapeño"),
            "the chip is named Jalapeno.",
        )
        self.assertEqual(
            tts._orpheus_prompt_text("Jalapeñorama stays unchanged"),
            "Jalapeñorama stays unchanged.",
        )

    def test_orpheus_prompt_articulates_zhu_yi_and_separates_prana_labs(self):
        text = (
            "DeepTech China published a conversation with Zhu Yi, "
            "co-founder of Prana Labs,"
        )

        self.assertEqual(
            tts._orpheus_prompt_text(text),
            "DeepTech China published a conversation with Joo Yee. "
            "Co-founder of Prana. Labs.",
        )

    def test_orpheus_transcript_accepts_only_contextual_zhu_yi_phonetics(self):
        expected = (
            "DeepTech China published a conversation with Zhu Yi, "
            "co-founder of Prana Labs,"
        )

        def verified(observed: str) -> bool:
            words = [
                {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
                for index, word in enumerate(observed.split())
            ]
            return bool(tts._orpheus_transcript_report(expected, words)["verified"])

        self.assertTrue(
            verified(
                "Deep Tech China published a conversation with Joo Yee "
                "co-founder of Prana Labs"
            )
        )
        self.assertTrue(
            verified(
                "Deep Tech China published a conversation with Jew Yee "
                "co-founder of Prana Labs"
            )
        )
        self.assertFalse(
            verified(
                "Deep Tech China published a conversation with Su Yi "
                "co-founder of Prana Labs"
            )
        )
        self.assertFalse(
            verified(
                "Deep Tech China published a conversation with Joo Lee "
                "co-founder of Prana Labs"
            )
        )

    def test_orpheus_prompt_articulates_techmeme_as_two_words(self):
        self.assertEqual(
            tts._orpheus_prompt_text("According to Techmeme's summary"),
            "According to Tech Meme's summary.",
        )

    def test_orpheus_transcript_accepts_tech_meme_but_rejects_tech_mean(self):
        expected = "According to Techmeme's summary"

        def verified(observed: str) -> bool:
            words = [
                {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
                for index, word in enumerate(observed.split())
            ]
            return bool(tts._orpheus_transcript_report(expected, words)["verified"])

        self.assertTrue(verified("According to Tech Meme's summary"))
        self.assertTrue(verified("According to TechMemes' summary"))
        self.assertFalse(verified("According to Tech Mean's summary"))

    def test_orpheus_prompt_articulates_earendil_one(self):
        self.assertEqual(
            tts._orpheus_prompt_text("a satellite named Earendil-1 later"),
            "a satellite named Ear-en-dill one later.",
        )

    def test_orpheus_transcript_accepts_earendil_syllables_not_arendelle(self):
        expected = "a satellite named Earendil-1 later"

        def verified(observed: str) -> bool:
            words = [
                {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
                for index, word in enumerate(observed.split())
            ]
            return bool(tts._orpheus_transcript_report(expected, words)["verified"])

        self.assertTrue(verified("a satellite named Ear en dill one later"))
        self.assertTrue(verified("a satellite named Irindil one later"))
        self.assertTrue(verified("a satellite named Ear Endil won later"))
        self.assertFalse(verified("a satellite named Arendelle one later"))

    def test_orpheus_prompt_articulates_brem_possessive_vowel(self):
        self.assertEqual(
            tts._orpheus_prompt_text("Brem's research indicates a result"),
            "Brehm's research indicates a result.",
        )
        self.assertEqual(
            tts._orpheus_prompt_text("Alexander Brem reported a result"),
            "Alexander Brem reported a result.",
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
            self.assertEqual(submitted["speed"], 1.0)
            self.assertEqual(requests[0].headers["X-API-Key"], "test-secret")
            manifest = json.loads((root / "audio" / "tts_manifest.json").read_text())
            self.assertEqual(
                manifest["pacing_policy"],
                tts.NARRATION_PACING_POLICY,
            )
            self.assertEqual(manifest["synthesis_speed_ratio"], 1.0)
            self.assertEqual(manifest["parts"][0]["synthesis_speed_ratio"], 1.0)
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

    async def test_orpheus_reuses_verified_audio_after_chunk_renumbering(self):
        requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(500)

        original_client = httpx.AsyncClient

        def client_factory(**kwargs):
            return original_client(transport=httpx.MockTransport(handler), **kwargs)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            script = root / "script.txt"
            script.write_text("One two three. Four five six.")
            output_dir = root / "audio"
            output_dir.mkdir()
            cached = (
                ("One two three.", 800, 8),
                ("Four five six.", 900, 9),
            )
            for text, frames, old_index in cached:
                path = output_dir / f"tts_input_part_{old_index:03d}_generated.wav"
                write_wav(path, frames=frames)
                tts._write_orpheus_part_metadata(
                    path,
                    text,
                    job_id=f"old-job-{old_index}",
                    request_token_budget=512,
                    integrity=self.verified_report(None, text, None),
                )
            messages = []

            with (
                patch.object(config, "ORPHEUS_TTS_API_KEY", "test-secret"),
                patch.object(config, "ORPHEUS_TTS_CHUNK_WORDS", 3),
                patch.object(tts.httpx, "AsyncClient", client_factory),
            ):
                result = await tts.generate_tts(
                    str(script),
                    str(output_dir),
                    ["tara"],
                    "orpheus-en",
                    log=messages.append,
                )

            current_parts = sorted(output_dir.glob("tts_input_part_00[12]_generated.wav"))
            result_frame_count = tts._read_pcm_wav(Path(result)).frame_count

        self.assertEqual(requests, [])
        self.assertEqual(len(current_parts), 2)
        self.assertEqual(result_frame_count, 1_700)
        self.assertTrue(any("after chunk renumbering" in item for item in messages))

    async def test_orpheus_revalidates_stale_audio_after_chunk_renumbering(self):
        requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(500)

        original_client = httpx.AsyncClient

        def client_factory(**kwargs):
            return original_client(transport=httpx.MockTransport(handler), **kwargs)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            script = root / "script.txt"
            text = "Revalidate this exact chunk."
            script.write_text(text)
            output_dir = root / "audio"
            output_dir.mkdir()
            old_path = output_dir / "tts_input_part_009_generated.wav"
            write_wav(old_path, frames=900)
            stale = tts._write_orpheus_part_metadata(
                old_path,
                text,
                job_id="old-job",
                request_token_budget=512,
                integrity=self.verified_report(None, text, None),
            )
            stale["integrity_verifier_version"] -= 1
            tts._part_metadata_path(old_path).write_text(json.dumps(stale))
            verifier = AsyncMock(side_effect=self.verified_report)
            messages = []

            with (
                patch.object(config, "ORPHEUS_TTS_API_KEY", "test-secret"),
                patch.object(tts.httpx, "AsyncClient", client_factory),
                patch.object(tts, "_verify_orpheus_part", verifier),
            ):
                result = await tts.generate_tts(
                    str(script),
                    str(output_dir),
                    ["tara"],
                    "orpheus-en",
                    log=messages.append,
                )
                refreshed = json.loads(
                    tts._part_metadata_path(Path(result)).read_text()
                )

        self.assertEqual(requests, [])
        verifier.assert_awaited_once()
        self.assertEqual(
            refreshed["integrity_verifier_version"],
            tts.ORPHEUS_INTEGRITY_VERIFIER_VERSION,
        )
        self.assertEqual(refreshed["job_id"], "recovered-local-output")
        self.assertTrue(any("revalidating exact-text" in item for item in messages))

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

    async def test_orpheus_verifier_rechecks_qwen_on_same_waveform_at_slower_speed(self):
        expected = "Alibaba's Qwen Office, known in Chinese as Qianwen."

        def words(text: str) -> list[dict]:
            return [
                {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
                for index, word in enumerate(text.split())
            ]

        original = words("Alibaba's Qwin Office known in Chinese as Qianwen")
        slower = words("Alibaba's Qwen Office known in Chinese as Qianwen")
        messages: list[str] = []
        with tempfile.TemporaryDirectory() as temp_dir:
            transcriber = AsyncMock(
                side_effect=[
                    (original, {"passed": True}),
                    (slower, {"passed": True}),
                ]
            )
            process = AsyncMock(return_value=(0, ""))
            with (
                patch(
                    "backend.pipeline.av_sync.ensure_word_transcript",
                    transcriber,
                ),
                patch.object(tts, "stream_subprocess", process),
            ):
                report = await tts._verify_orpheus_part(
                    Path(temp_dir) / "qwen.wav",
                    expected,
                    Path(temp_dir) / "verification",
                    emit=messages.append,
                )

        self.assertTrue(report["verified"])
        self.assertEqual(report["verification_playback_speed"], 0.8)
        self.assertEqual(report["speech_end_seconds"], 1.2)
        self.assertIn("exact name transcript recovered", " ".join(messages))
        self.assertEqual(transcriber.await_count, 2)
        self.assertEqual(process.await_count, 1)
        self.assertIn("atempo=0.8", process.await_args.kwargs["command"])

    async def test_orpheus_verifier_rechecks_qbitai_hubit_split_at_slower_speed(self):
        expected = (
            "QbitAI reports that the company behind PhanthyMotus has launched "
            "a new plan."
        )

        def words(text: str) -> list[dict]:
            return [
                {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
                for index, word in enumerate(text.split())
            ]

        original = words(
            "Hubit AI reports that the company behind FancyModus has launched "
            "a new plan"
        )
        slower = words(
            "QBit AI reports that the company behind Fantymodus has launched "
            "a new plan"
        )
        messages: list[str] = []
        with tempfile.TemporaryDirectory() as temp_dir:
            transcriber = AsyncMock(
                side_effect=[
                    (original, {"passed": True}),
                    (slower, {"passed": True}),
                ]
            )
            process = AsyncMock(return_value=(0, ""))
            with (
                patch(
                    "backend.pipeline.av_sync.ensure_word_transcript",
                    transcriber,
                ),
                patch.object(tts, "stream_subprocess", process),
            ):
                report = await tts._verify_orpheus_part(
                    Path(temp_dir) / "qbitai.wav",
                    expected,
                    Path(temp_dir) / "verification",
                    emit=messages.append,
                )

        self.assertTrue(report["verified"])
        self.assertEqual(report["verification_playback_speed"], 0.8)
        self.assertEqual(transcriber.await_count, 2)
        self.assertEqual(process.await_count, 1)
        self.assertIn("exact name transcript recovered", " ".join(messages))

    def test_orpheus_name_recheck_rejects_unsupported_qbitai_split(self):
        expected = "QbitAI reports the result."
        observed = [
            {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
            for index, word in enumerate("Unit AI reports the result".split())
        ]

        self.assertFalse(tts._has_only_name_transcript_mismatches(expected, observed))

    async def test_orpheus_name_recheck_rejects_unsupported_single_token(self):
        expected = "Alibaba's Qwen Office."

        def words(text: str) -> list[dict]:
            return [
                {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
                for index, word in enumerate(text.split())
            ]

        wrong = words("Alibaba's Khan Office")
        with tempfile.TemporaryDirectory() as temp_dir:
            transcriber = AsyncMock(
                return_value=(wrong, {"passed": True})
            )
            process = AsyncMock(return_value=(0, ""))
            with (
                patch(
                    "backend.pipeline.av_sync.ensure_word_transcript",
                    transcriber,
                ),
                patch.object(tts, "stream_subprocess", process),
            ):
                with self.assertRaises(tts.TtsIntegrityError):
                    await tts._verify_orpheus_part(
                        Path(temp_dir) / "qwen.wav",
                        expected,
                        Path(temp_dir) / "verification",
                        emit=lambda _message: None,
                    )

        self.assertEqual(transcriber.await_count, 1)
        process.assert_not_awaited()

    async def test_orpheus_name_recheck_tries_second_slow_speed(self):
        expected = "Alibaba's Qwen Office."

        def words(text: str) -> list[dict]:
            return [
                {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
                for index, word in enumerate(text.split())
            ]

        wrong = words("Alibaba's Qwin Office")
        correct = words("Alibaba's Qwen Office")
        with tempfile.TemporaryDirectory() as temp_dir:
            transcriber = AsyncMock(
                side_effect=[
                    (wrong, {"passed": True}),
                    (wrong, {"passed": True}),
                    (correct, {"passed": True}),
                ]
            )
            process = AsyncMock(return_value=(0, ""))
            with (
                patch(
                    "backend.pipeline.av_sync.ensure_word_transcript",
                    transcriber,
                ),
                patch.object(tts, "stream_subprocess", process),
            ):
                report = await tts._verify_orpheus_part(
                    Path(temp_dir) / "qwen.wav",
                    expected,
                    Path(temp_dir) / "verification",
                    emit=lambda _message: None,
                )

        self.assertTrue(report["verified"])
        self.assertEqual(report["verification_playback_speed"], 0.7)
        self.assertEqual(transcriber.await_count, 3)
        self.assertEqual(process.await_count, 2)

    async def test_orpheus_name_recheck_does_not_override_extra_words(self):
        expected = "Qwen Office ranked first."

        def words(text: str) -> list[dict]:
            return [
                {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
                for index, word in enumerate(text.split())
            ]

        original = words("Qwen Office really ranked first")
        slower = words("Qwen Office ranked first")
        transcriber = AsyncMock(
            side_effect=[
                (original, {"passed": True}),
                (slower, {"passed": True}),
            ]
        )
        process = AsyncMock(return_value=(0, ""))
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch(
                    "backend.pipeline.av_sync.ensure_word_transcript",
                    transcriber,
                ),
                patch.object(tts, "stream_subprocess", process),
            ):
                with self.assertRaises(tts.TtsIntegrityError):
                    await tts._verify_orpheus_part(
                        Path(temp_dir) / "qwen.wav",
                        expected,
                        Path(temp_dir) / "verification",
                        emit=lambda _message: None,
                    )

        self.assertEqual(transcriber.await_count, 1)
        process.assert_not_awaited()

    async def test_orpheus_name_recheck_handles_split_name_spelling(self):
        expected = "Alibaba's Qwen Office, known in Chinese as Qianwen."

        def words(text: str) -> list[dict]:
            return [
                {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
                for index, word in enumerate(text.split())
            ]

        original = words("Alibaba's Q Win Office known in Chinese as Can Wen")
        slower = words("Alibaba's Qwen Office known in Chinese as Qianwen")
        with tempfile.TemporaryDirectory() as temp_dir:
            transcriber = AsyncMock(
                side_effect=[
                    (original, {"passed": True}),
                    (slower, {"passed": True}),
                ]
            )
            process = AsyncMock(return_value=(0, ""))
            with (
                patch(
                    "backend.pipeline.av_sync.ensure_word_transcript",
                    transcriber,
                ),
                patch.object(tts, "stream_subprocess", process),
            ):
                report = await tts._verify_orpheus_part(
                    Path(temp_dir) / "qwen.wav",
                    expected,
                    Path(temp_dir) / "verification",
                    emit=lambda _message: None,
                )

        self.assertTrue(report["verified"])
        self.assertEqual(report["verification_playback_speed"], 0.8)
        self.assertEqual(transcriber.await_count, 2)
        self.assertEqual(process.await_count, 1)

    def test_orpheus_name_recheck_rejects_non_name_replacement(self):
        expected = "Qwen Office ranked first."
        observed = [
            {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
            for index, word in enumerate("Qwin really ranked first".split())
        ]

        self.assertFalse(tts._has_only_name_transcript_mismatches(expected, observed))

    def test_orpheus_name_recheck_accepts_observed_q_when_and_qian_wen_splits(self):
        expected = "Alibaba's Qwen Office, known in Chinese as Qianwen."
        observed = [
            {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
            for index, word in enumerate(
                "Alibaba's Q when Office known in Chinese as Qian Wen".split()
            )
        ]

        self.assertTrue(tts._has_only_name_transcript_mismatches(expected, observed))

    def test_orpheus_name_recheck_accepts_evidenced_qwin_spelling(self):
        def words(text: str) -> list[dict]:
            return [
                {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
                for index, word in enumerate(text.split())
            ]

        self.assertTrue(
            tts._has_only_name_transcript_mismatches(
                "Qwen Office ranked first", words("Qwin Office ranked first")
            )
        )
        self.assertTrue(
            tts._has_only_name_transcript_mismatches(
                "Qwen's Office ranked first", words("Qwin's Office ranked first")
            )
        )
        self.assertTrue(
            tts._has_only_name_transcript_mismatches(
                "Qwen's Office ranked first", words("Q Win's Office ranked first")
            )
        )

    def test_orpheus_name_recheck_rejects_unsupported_single_token_spellings(self):
        def words(text: str) -> list[dict]:
            return [
                {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
                for index, word in enumerate(text.split())
            ]

        cases = (
            ("Qwen Office ranked first", "Khan Office ranked first"),
            ("Qwen Office ranked first", "Banana Office ranked first"),
            ("Qwen Office ranked first", "Tianwen Office ranked first"),
            ("Qianwen Office ranked first", "Chanmen Office ranked first"),
            ("Qwen's Office ranked first", "Khan's Office ranked first"),
            ("Qwen's Office ranked first", "Banana's Office ranked first"),
            ("Qwen Office ranked first", "Qwin's Office ranked first"),
            ("Qwen's Office ranked first", "Qwin Office ranked first"),
            ("Qwen's Office ranked first", "Q Win Office ranked first"),
        )
        for expected, observed in cases:
            with self.subTest(expected=expected, observed=observed):
                self.assertFalse(
                    tts._has_only_name_transcript_mismatches(
                        expected, words(observed)
                    )
                )

    def test_orpheus_transcript_accepts_expected_name_pronunciation_splits(self):
        expected = "with results showing that Alibaba's Qwen Office, known in Chinese as Qianwen"
        observed = [
            {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
            for index, word in enumerate(
                "with results showing that Alibaba's Q when Office known in Chinese as Qian Wen".split()
            )
        ]

        report = tts._orpheus_transcript_report(expected, observed)

        self.assertTrue(report["verified"])
        self.assertEqual(report["matched_exact_words"], 12)
        self.assertEqual(report["transcript_words"], 12)
        self.assertEqual(report["transcript_word_ratio"], 1.0)

    def test_orpheus_name_split_requires_matching_possessive(self):
        def report(expected: str, observed: str) -> dict:
            words = [
                {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
                for index, word in enumerate(observed.split())
            ]
            return tts._orpheus_transcript_report(expected, words)

        self.assertTrue(
            report(
                "Qwen's report ranked first",
                "Q when's report ranked first",
            )["verified"]
        )
        self.assertTrue(
            report(
                "Qianwen's report ranked first",
                "Qian Wen's report ranked first",
            )["verified"]
        )
        self.assertFalse(
            report(
                "Qwen's report ranked first",
                "Q when report ranked first",
            )["verified"]
        )
        self.assertFalse(
            report(
                "Qwen report ranked first",
                "Q when's report ranked first",
            )["verified"]
        )

    def test_orpheus_name_split_alignment_rejects_extra_or_wrong_name_syllables(self):
        expected = "Alibaba's Qwen Office known in Chinese as Qianwen"

        def report(text: str) -> dict:
            words = [
                {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
                for index, word in enumerate(text.split())
            ]
            return tts._orpheus_transcript_report(expected, words)

        self.assertFalse(
            report("Alibaba's Q when really Office known in Chinese as Qian Wen")[
                "verified"
            ]
        )
        self.assertFalse(
            report("Alibaba's Q when Office known in Chinese as Qian Wong")[
                "verified"
            ]
        )

    def test_orpheus_name_split_alignment_is_not_a_global_phrase_alias(self):
        expected = "The variable Q when tested works"
        observed = [
            {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
            for index, word in enumerate("The variable Qwen tested works".split())
        ]

        self.assertFalse(tts._orpheus_transcript_report(expected, observed)["verified"])

    def test_orpheus_transcript_accepts_contextual_jefferies_spelling(self):
        expected = "The Jefferies report, as described by QbitAI, broke agent capability into model"
        observed = [
            {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
            for index, word in enumerate(
                "The Jeffreys report as described by Qubit AI broke agent capability into model".split()
            )
        ]

        report = tts._orpheus_transcript_report(expected, observed)

        self.assertTrue(report["verified"])
        self.assertEqual(report["matched_exact_words"], report["expected_words"])
        self.assertEqual(report["transcript_word_ratio"], 1.0)

    def test_orpheus_jefferies_report_equivalence_remains_context_limited(self):
        def report(expected: str, observed: str) -> dict:
            words = [
                {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
                for index, word in enumerate(observed.split())
            ]
            return tts._orpheus_transcript_report(expected, words)

        self.assertFalse(report("A Jefferies note", "A Jeffreys note")["verified"])
        self.assertFalse(
            report("The Jefferies report arrived", "The Jeffers report arrived")[
                "verified"
            ]
        )
        self.assertFalse(
            report(
                "The Jefferies report arrived",
                "The Jeffreys annual report arrived",
            )["verified"]
        )
        self.assertFalse(report("Q bit AI works", "QbitAI works")["verified"])

    def test_orpheus_transcript_preserves_percent_as_a_completeness_token(self):
        def verified(expected: str, observed: str) -> bool:
            words = [
                {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
                for index, word in enumerate(observed.split())
            ]
            return bool(tts._orpheus_transcript_report(expected, words)["verified"])

        self.assertTrue(verified("Growth reached 90 percent", "Growth reached 90%"))
        self.assertTrue(verified("Growth reached 90%", "Growth reached 90 percent"))
        self.assertFalse(verified("Growth reached 90", "Growth reached 90 percent"))
        self.assertFalse(verified("Growth reached 90 percent", "Growth reached 90"))

    def test_orpheus_name_collapses_do_not_hide_an_extra_percent(self):
        expected = "The Jefferies report described QbitAI"

        def verified(observed: str) -> bool:
            words = [
                {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
                for index, word in enumerate(observed.split())
            ]
            return bool(tts._orpheus_transcript_report(expected, words)["verified"])

        self.assertFalse(verified("The percent Jeffreys report described Qubit AI"))
        self.assertFalse(verified("The Jeffreys report described Qubit percent AI"))

    def test_orpheus_name_collapses_do_not_hide_spoken_symbols(self):
        expected = "The Jefferies report described QbitAI"

        def verified(raw_words: list[str]) -> bool:
            words = [
                {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
                for index, word in enumerate(raw_words)
            ]
            return bool(tts._orpheus_transcript_report(expected, words)["verified"])

        self.assertFalse(
            verified(["The", "&", "Jeffreys", "report", "described", "Qubit", "AI"])
        )
        self.assertFalse(
            verified(["The", "Jeffreys", "report", "described", "Qubit", "&AI"])
        )
        self.assertFalse(
            verified(["The", "Jeffreys", "report", "described", "Q-bit-AI"])
        )

    def test_orpheus_transcript_preserves_spoken_currency_units(self):
        def verified(expected: str, raw_words: list[str]) -> bool:
            words = [
                {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
                for index, word in enumerate(raw_words)
            ]
            return bool(tts._orpheus_transcript_report(expected, words)["verified"])

        self.assertTrue(verified("It cost $5", ["It", "cost", "$5"]))
        self.assertTrue(verified("It cost 5 dollars", ["It", "cost", "$5"]))
        self.assertFalse(verified("It cost 5", ["It", "cost", "$5"]))

    def test_orpheus_transcript_normalizes_currency_adjective_shorthand(self):
        expected = (
            "The company emerged from stealth in 2024 with a "
            "300-million-dollar Series"
        )

        def report(raw_words: list[str], *, source: str = expected) -> dict:
            words = [
                {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
                for index, word in enumerate(raw_words)
            ]
            return tts._orpheus_transcript_report(source, words)

        live_whisper_words = (
            "The company emerged from stealth in 2024 with a $300 million Series"
        ).split()
        accepted = report(live_whisper_words)

        self.assertTrue(accepted["verified"])
        self.assertEqual(accepted["exact_asr_word_coverage"], 1.0)
        self.assertFalse(report([
            "The", "company", "emerged", "from", "stealth", "in", "2024",
            "with", "a", "$301", "million", "Series",
        ])["verified"])
        self.assertFalse(report([
            "The", "company", "emerged", "from", "stealth", "in", "2024",
            "with", "a", "$300", "billion", "Series",
        ])["verified"])
        self.assertFalse(report([
            "The", "company", "emerged", "from", "stealth", "in", "2024",
            "with", "a", "300", "million", "dollars", "Series",
        ])["verified"])
        self.assertFalse(report(
            live_whisper_words,
            source=expected.replace("300-million-dollar", "300 million dollar"),
        )["verified"])

        decimal_source = "A at a 1.5-billion-dollar valuation"
        self.assertTrue(report(
            "A at a $1.5 billion valuation".split(),
            source=decimal_source,
        )["verified"])
        self.assertTrue(report(
            ["A", "at", "a", "$1", ".5", "billion", "valuation"],
            source=decimal_source,
        )["verified"])
        self.assertFalse(report(
            "A at a $1.6 billion valuation".split(),
            source=decimal_source,
        )["verified"])

        word_source = (
            "According to Bloomberg, the ceremony was held Thursday for the "
            "four-billion-dollar plant"
        )
        live_word_amount = (
            "According to Bloomberg the ceremony was held Thursday for the "
            "$4 billion plant"
        ).split()
        accepted_word_amount = report(live_word_amount, source=word_source)
        self.assertTrue(accepted_word_amount["verified"])
        self.assertEqual(accepted_word_amount["exact_asr_word_coverage"], 1.0)
        self.assertFalse(report(
            [*live_word_amount[:10], "$5", "billion", "plant"],
            source=word_source,
        )["verified"])
        self.assertFalse(report(
            [*live_word_amount[:10], "$4", "million", "plant"],
            source=word_source,
        )["verified"])
        self.assertFalse(report(
            [*live_word_amount[:10], "4", "billion", "plant"],
            source=word_source,
        )["verified"])

    def test_orpheus_transcript_normalizes_currency_decimal_split_by_whisper(self):
        expected = (
            "bond sale, equivalent to roughly six point three billion dollars, "
            "marking"
        )

        def report(raw_words: list[str]) -> dict:
            words = [
                {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
                for index, word in enumerate(raw_words)
            ]
            return tts._orpheus_transcript_report(expected, words)

        whisper_split = report(
            [
                "bond", "sale,", "equivalent", "to", "roughly", "$6", ".3",
                "billion,", "marking.",
            ]
        )

        self.assertTrue(whisper_split["verified"])
        self.assertEqual(whisper_split["exact_asr_word_coverage"], 1.0)
        self.assertTrue(
            report(
                [
                    "bond", "sale,", "equivalent", "to", "roughly", "$6.3",
                    "billion,", "marking.",
                ]
            )["verified"]
        )
        self.assertFalse(
            report(
                [
                    "bond", "sale,", "equivalent", "to", "roughly", "6", ".3",
                    "billion,", "marking.",
                ]
            )["verified"]
        )
        self.assertFalse(
            report(
                [
                    "bond", "sale,", "equivalent", "to", "roughly", "$6", ".4",
                    "billion,", "marking.",
                ]
            )["verified"]
        )

    def test_orpheus_transcript_normalizes_split_thousands_without_losing_unit(self):
        expected = "A prize of two hundred fifty thousand yuan."

        def report(raw_words: list[str]) -> dict:
            words = [
                {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
                for index, word in enumerate(raw_words)
            ]
            return tts._orpheus_transcript_report(expected, words)

        whisper_split = report(["A", "prize", "of", "250", ",000", "yuan."])

        self.assertTrue(whisper_split["verified"])
        self.assertEqual(whisper_split["exact_asr_word_coverage"], 1.0)
        self.assertFalse(
            report(["A", "prize", "of", "251", ",000", "yuan."])["verified"]
        )
        self.assertFalse(report(["A", "prize", "of", "250", ",000"])["verified"])
        self.assertFalse(
            report(["A", "prize", "of", "250", ",000", "ren."])["verified"]
        )

    def test_transcript_number_indexes_preserve_later_name_provenance(self):
        raw_words = "twenty twenty six Q bit AI works".split()
        words = [
            {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
            for index, word in enumerate(raw_words)
        ]

        tokens, indexes = tts._transcript_tokens(words)

        self.assertEqual(tokens, ["2026", "q", "bit", "ai", "works"])
        self.assertEqual(indexes, [0, 3, 4, 5, 6])
        self.assertFalse(
            tts._orpheus_transcript_report("2026 QbitAI works", words)["verified"]
        )

    def test_orpheus_name_recheck_rejects_adjacent_extra_or_missing_name(self):
        expected = "Qwen Office ranked first."

        def words(text: str) -> list[dict]:
            return [
                {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
                for index, word in enumerate(text.split())
            ]

        self.assertFalse(
            tts._has_only_name_transcript_mismatches(
                expected, words("Qwin really Office ranked first")
            )
        )
        self.assertFalse(
            tts._has_only_name_transcript_mismatches(
                expected, words("really Qwin Office ranked first")
            )
        )
        self.assertFalse(
            tts._has_only_name_transcript_mismatches(
                expected, words("Office ranked first")
            )
        )

    def test_orpheus_name_recheck_recognizes_possessive_name_token(self):
        self.assertTrue(tts._needs_name_playback_recheck("Qwen's launch"))
        self.assertTrue(tts._needs_name_playback_recheck("Qianwen's launch"))
        self.assertFalse(tts._needs_name_playback_recheck("Qwin's launch"))

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

    def test_orpheus_transcript_normalizes_hyphenated_numeric_range(self):
        expected = "The window is roughly two to three seconds."

        for spoken in (
            "The window is roughly 2 -3 seconds".split(),
            "The window is roughly 2-3 seconds".split(),
        ):
            words = [
                {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
                for index, word in enumerate(spoken)
            ]
            report = tts._orpheus_transcript_report(expected, words)

            self.assertTrue(report["verified"])
            self.assertEqual(report["exact_asr_word_coverage"], 1.0)

    def test_orpheus_transcript_does_not_invent_numeric_range_separator(self):
        expected = "The window is roughly two to three seconds."

        def report(spoken: str) -> dict:
            words = [
                {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
                for index, word in enumerate(spoken.split())
            ]
            return tts._orpheus_transcript_report(expected, words)

        self.assertFalse(report("The window is roughly 2 3 seconds")["verified"])
        self.assertFalse(report("The window is roughly 2 -4 seconds")["verified"])

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

    def test_orpheus_transcript_normalizes_compound_numeric_ordinal(self):
        expected = (
            "has been officially announced for November twentieth and "
            "twenty-first, co-hosted"
        )

        def report(second_date: str) -> dict:
            observed = (
                "has been officially announced for November 20th and "
                f"{second_date} co-hosted"
            ).split()
            words = [
                {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
                for index, word in enumerate(observed)
            ]
            return tts._orpheus_transcript_report(expected, words)

        accepted = report("21st")
        self.assertTrue(accepted["verified"])
        self.assertEqual(accepted["exact_asr_word_coverage"], 1.0)
        self.assertFalse(report("22nd")["verified"])

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

    def test_orpheus_transcript_accepts_night_knight_homophone_only(self):
        expected = "The headline question of whether night."

        def report(final_word: str) -> dict:
            observed = f"The headline question of whether {final_word}".split()
            words = [
                {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
                for index, word in enumerate(observed)
            ]
            return tts._orpheus_transcript_report(expected, words)

        accepted = report("Knight")

        self.assertTrue(accepted["verified"])
        self.assertEqual(accepted["exact_asr_word_coverage"], 1.0)
        self.assertFalse(report("light")["verified"])
        self.assertFalse(report("")["verified"])

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

    def test_orpheus_transcript_normalizes_fivefold_whisper_split(self):
        expected = (
            "45 percent since a more than fivefold surge on its August 19 "
            "debut in Shanghai."
        )
        observed = (
            "45 % since a more than five -fold surge on its August 19 debut "
            "in Shanghai"
        ).split()
        words = [
            {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
            for index, word in enumerate(observed)
        ]

        report = tts._orpheus_transcript_report(expected, words)

        self.assertTrue(report["verified"])
        self.assertEqual(report["expected_words"], 15)
        self.assertEqual(report["transcript_words"], 15)
        self.assertEqual(report["matched_exact_words"], 15)
        self.assertEqual(report["exact_asr_word_coverage"], 1.0)

    def test_orpheus_fivefold_normalization_rejects_wrong_multiplier(self):
        expected = "A more than fivefold surge followed."
        observed = "A more than four -fold surge followed".split()
        words = [
            {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
            for index, word in enumerate(observed)
        ]

        report = tts._orpheus_transcript_report(expected, words)

        self.assertFalse(report["verified"])
        self.assertLess(report["exact_asr_word_coverage"], 1.0)

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

    def test_orpheus_transcript_normalizes_xspark_company_name(self):
        expected = (
            "And in robotics, DeepTech China sits down with Ding Wenbo of Xspark"
        )
        observed = (
            "And in robotics Deep Tech China sits down with Ding Wenbo of X Spark"
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

    def test_orpheus_transcript_normalizes_postdoc_compound_split(self):
        expected = (
            "Back in 2016, he faced postdoc offers from MIT and Georgia Tech,"
        )
        observed = (
            "Back in 2016 he faced post doc offers from MIT and Georgia Tech"
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

    def test_orpheus_transcript_normalizes_phanthymotus_asr_spellings(self):
        expected = "The company behind PhanthyMotus has launched a new plan."

        def report(product_name: str) -> dict:
            observed = (
                f"The company behind {product_name} has launched a new plan"
            ).split()
            words = [
                {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
                for index, word in enumerate(observed)
            ]
            return tts._orpheus_transcript_report(expected, words)

        self.assertTrue(report("FancyModus")["verified"])
        self.assertTrue(report("Fantymodus")["verified"])
        self.assertFalse(report("FancyModels")["verified"])

    def test_orpheus_transcript_normalizes_skild_company_name(self):
        expected = (
            "North American robotics startup Skild AI has released a new robot "
            "foundation."
        )

        def report(company_name: str) -> dict:
            observed = (
                "North American robotics startup "
                f"{company_name} AI has released a new robot foundation"
            ).split()
            words = [
                {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
                for index, word in enumerate(observed)
            ]
            return tts._orpheus_transcript_report(expected, words)

        accepted = report("Skilled")
        self.assertTrue(accepted["verified"])
        self.assertEqual(accepted["exact_asr_word_coverage"], 0.9167)
        self.assertEqual(accepted["acoustic_asr_word_coverage"], 1.0)
        self.assertEqual(
            accepted["verification_mode"],
            "aligned_phonetic_substitution",
        )
        self.assertEqual(
            accepted["phonetic_substitutions"][0]["observed"],
            "skilled",
        )
        self.assertFalse(report("Skill")["verified"])
        self.assertFalse(report("Skillet")["verified"])

    def test_orpheus_transcript_accepts_only_evidenced_shein_spelling(self):
        expected = (
            "Reuters says Shein pivoted to a Hong Kong IPO after bids failed "
            "to secure Beijing's approval."
        )

        def report(brand_name: str) -> dict:
            observed = (
                f"Reuters says {brand_name} pivoted to a Hong Kong IPO after bids "
                "failed to secure Beijing's approval"
            ).split()
            words = [
                {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
                for index, word in enumerate(observed)
            ]
            return tts._orpheus_transcript_report(expected, words)

        accepted = report("Shane")
        self.assertTrue(accepted["verified"])
        self.assertEqual(accepted["exact_asr_word_coverage"], 0.9375)
        self.assertEqual(accepted["acoustic_asr_word_coverage"], 1.0)
        self.assertEqual(
            accepted["verification_mode"],
            "aligned_phonetic_substitution",
        )
        self.assertEqual(
            accepted["phonetic_substitutions"][0]["phonetic_key"],
            "evidenced:shein-shane",
        )
        self.assertFalse(report("Shawn")["verified"])
        self.assertFalse(report("Shine")["verified"])

    def test_orpheus_transcript_preserves_polish_l_stroke(self):
        expected = (
            "QbitAI says Transformer co-author Łukasz Kaiser will keynote "
            "November's Singularity Intelligence Conference in Beijing."
        )
        observed = (
            "Qubit AI says Transformer co-author Lukas Kaiser will keynote "
            "November's Singularity Intelligence Conference in Beijing"
        ).split()
        words = [
            {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
            for index, word in enumerate(observed)
        ]

        report = tts._orpheus_transcript_report(expected, words)

        self.assertTrue(report["verified"])
        self.assertEqual(report["exact_asr_word_coverage"], 0.9333)
        self.assertEqual(report["acoustic_asr_word_coverage"], 1.0)
        self.assertEqual(
            report["verification_mode"],
            "aligned_phonetic_substitution",
        )
        self.assertEqual(report["expected_words"], 15)

    def test_orpheus_transcript_accepts_rights_writes_homophone(self):
        expected = (
            "DeepTech China reports Hugging Face open-sourced a robot duck that "
            "walks, slides, and self-rights."
        )

        def report(final_word: str) -> dict:
            observed = (
                "Deep Tech China reports Hugging Face open sourced a robot duck "
                f"that walks slides and self {final_word}"
            ).split()
            words = [
                {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
                for index, word in enumerate(observed)
            ]
            return tts._orpheus_transcript_report(expected, words)

        accepted = report("writes")
        self.assertTrue(accepted["verified"])
        self.assertEqual(accepted["exact_asr_word_coverage"], 1.0)
        self.assertTrue(accepted["trailing_anchor"])
        self.assertFalse(report("rises")["verified"])

    def test_orpheus_transcript_collapses_complete_ox_alpha_and_z_ai_names(self):
        expected = (
            "The Rundown AI confirms Ox Alpha is Z AI's GLM-5.3-Flash, priced "
            "near a tenth of rivals."
        )
        observed = (
            "The Rundown AI confirms OxAlpha is ZAI's GLM 5.3 Flash priced near "
            "a tenth of rivals"
        ).split()
        words = [
            {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
            for index, word in enumerate(observed)
        ]

        report = tts._orpheus_transcript_report(expected, words)

        self.assertTrue(report["verified"])
        self.assertEqual(report["exact_asr_word_coverage"], 1.0)
        self.assertEqual(report["expected_words"], 16)
        self.assertEqual(report["transcript_words"], 16)

    def test_orpheus_transcript_normalizes_jalapeno_chip_name(self):
        expected = (
            "OpenAI has published the first performance results for its in-house "
            "AI inference chip, named Jalapeño."
        )

        def report(chip_name: str) -> dict:
            observed = (
                "OpenAI has published the first performance results for its "
                f"in-house AI inference chip named {chip_name}"
            ).split()
            words = [
                {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
                for index, word in enumerate(observed)
            ]
            return tts._orpheus_transcript_report(expected, words)

        self.assertTrue(report("Jalapeno")["verified"])
        self.assertFalse(report("Jalapena")["verified"])
        self.assertFalse(report("Jalape")["verified"])

    def test_orpheus_phonetic_fallback_rejects_compensated_omission(self):
        expected = (
            "North American robotics startup Skild AI has released a new robot "
            "foundation."
        )
        observed = (
            "North American startup modern Skild AI has released a new robot "
            "foundation"
        ).split()
        words = [
            {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
            for index, word in enumerate(observed)
        ]

        report = tts._orpheus_transcript_report(expected, words)

        self.assertFalse(report["verified"])
        self.assertEqual(report["exact_asr_word_coverage"], 0.9167)
        self.assertEqual(report["transcript_word_ratio"], 1.0)
        self.assertIn("not one aligned", " ".join(report["failure_reasons"]))

    async def test_orpheus_verifier_corroborates_unlisted_phonetic_spelling(self):
        expected = (
            "North American robotics startup Skild AI has released a new robot "
            "foundation."
        )

        def words(text: str) -> list[dict]:
            return [
                {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
                for index, word in enumerate(text.split())
            ]

        transcript = words(
            "North American robotics startup Skilled AI has released a new robot "
            "foundation"
        )
        messages: list[str] = []
        with tempfile.TemporaryDirectory() as temp_dir:
            transcriber = AsyncMock(
                side_effect=[
                    (transcript, {"passed": True}),
                    (transcript, {"passed": True}),
                ]
            )
            process = AsyncMock(return_value=(0, ""))
            with (
                patch(
                    "backend.pipeline.av_sync.ensure_word_transcript",
                    transcriber,
                ),
                patch.object(tts, "stream_subprocess", process),
            ):
                report = await tts._verify_orpheus_part(
                    Path(temp_dir) / "skild.wav",
                    expected,
                    Path(temp_dir) / "verification",
                    emit=messages.append,
                )

        self.assertTrue(report["verified"])
        self.assertEqual(
            report["verification_mode"],
            "corroborated_phonetic_substitution",
        )
        self.assertEqual(report["exact_asr_word_coverage"], 0.9167)
        self.assertEqual(report["acoustic_asr_word_coverage"], 1.0)
        self.assertEqual(transcriber.await_count, 2)
        self.assertEqual(process.await_count, 1)
        self.assertIn("corroborated", " ".join(messages))

    async def test_orpheus_verifier_corroborates_short_lukasz_spelling(self):
        expected = (
            "Headlining is Łukasz Kaiser, the OpenAI senior research scientist"
        )

        def words(text: str) -> list[dict]:
            return [
                {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
                for index, word in enumerate(text.split())
            ]

        transcript = words(
            "Headlining is Lukas Kaiser the OpenAI senior research scientist"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            transcriber = AsyncMock(
                side_effect=[
                    (transcript, {"passed": True}),
                    (transcript, {"passed": True}),
                ]
            )
            process = AsyncMock(return_value=(0, ""))
            with (
                patch(
                    "backend.pipeline.av_sync.ensure_word_transcript",
                    transcriber,
                ),
                patch.object(tts, "stream_subprocess", process),
            ):
                report = await tts._verify_orpheus_part(
                    Path(temp_dir) / "lukasz.wav",
                    expected,
                    Path(temp_dir) / "verification",
                    emit=lambda _message: None,
                )

        self.assertTrue(report["verified"])
        self.assertEqual(report["exact_asr_word_coverage"], 0.8889)
        self.assertEqual(report["acoustic_asr_word_coverage"], 1.0)
        self.assertEqual(
            report["verification_mode"],
            "corroborated_phonetic_substitution",
        )
        self.assertEqual(transcriber.await_count, 2)
        self.assertEqual(process.await_count, 1)

    async def test_orpheus_verifier_corroborates_two_evidenced_name_spellings(self):
        expected = (
            "From Hollywood, IEEE Spectrum profiles Jernej Barbič, the University "
            "of Southern California"
        )

        def words(text: str) -> list[dict]:
            return [
                {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
                for index, word in enumerate(text.split())
            ]

        transcript = words(
            "From Hollywood IEEE Spectrum profiles Jernesh Barbish the University "
            "of Southern California"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            transcriber = AsyncMock(
                side_effect=[
                    (transcript, {"passed": True}),
                    (transcript, {"passed": True}),
                ]
            )
            process = AsyncMock(return_value=(0, ""))
            with (
                patch(
                    "backend.pipeline.av_sync.ensure_word_transcript",
                    transcriber,
                ),
                patch.object(tts, "stream_subprocess", process),
            ):
                report = await tts._verify_orpheus_part(
                    Path(temp_dir) / "jernej-barbic.wav",
                    expected,
                    Path(temp_dir) / "verification",
                    emit=lambda _message: None,
                )

        self.assertTrue(report["verified"])
        self.assertEqual(report["exact_asr_word_coverage"], 0.8333)
        self.assertEqual(report["acoustic_asr_word_coverage"], 1.0)
        self.assertEqual(
            report["verification_mode"],
            "corroborated_phonetic_substitution",
        )
        self.assertEqual(
            {item["expected"] for item in report["phonetic_substitutions"]},
            {"jernej", "barbi"},
        )
        self.assertEqual(transcriber.await_count, 2)
        self.assertEqual(process.await_count, 1)

    async def test_orpheus_verifier_corroborates_evidenced_douyin_spelling(self):
        expected = (
            "one hundred animated dramas on Douyin in May were AI productions."
        )

        def words(text: str) -> list[dict]:
            return [
                {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
                for index, word in enumerate(text.split())
            ]

        transcript = words(
            "100 animated dramas on Duwayan in May were AI productions"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            transcriber = AsyncMock(
                side_effect=[
                    (transcript, {"passed": True}),
                    (transcript, {"passed": True}),
                ]
            )
            process = AsyncMock(return_value=(0, ""))
            with (
                patch(
                    "backend.pipeline.av_sync.ensure_word_transcript",
                    transcriber,
                ),
                patch.object(tts, "stream_subprocess", process),
            ):
                report = await tts._verify_orpheus_part(
                    Path(temp_dir) / "douyin.wav",
                    expected,
                    Path(temp_dir) / "verification",
                    emit=lambda _message: None,
                )

        self.assertTrue(report["verified"])
        self.assertEqual(report["exact_asr_word_coverage"], 0.9)
        self.assertEqual(report["acoustic_asr_word_coverage"], 1.0)
        self.assertEqual(
            report["phonetic_substitutions"][0]["phonetic_key"],
            "evidenced:douyin-duwayan",
        )
        self.assertEqual(
            report["verification_mode"],
            "corroborated_phonetic_substitution",
        )
        self.assertEqual(transcriber.await_count, 2)
        self.assertEqual(process.await_count, 1)

    def test_orpheus_transcript_does_not_generalize_duwayan_spelling(self):
        expected = "The Dorian platform remained available today."
        observed = "The Duwayan platform remained available today".split()
        words = [
            {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
            for index, word in enumerate(observed)
        ]

        report = tts._orpheus_transcript_report(expected, words)

        self.assertFalse(report["verified"])
        self.assertEqual(report["phonetic_substitutions"], [])

    def test_orpheus_transcript_rejects_three_evidenced_name_spellings(self):
        expected = "Shein profiles Jernej Barbič today."
        observed = "Shane profiles Jernesh Barbish today".split()
        words = [
            {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
            for index, word in enumerate(observed)
        ]

        report = tts._orpheus_transcript_report(expected, words)

        self.assertFalse(report["verified"])
        self.assertEqual(report["phonetic_substitutions"], [])
        self.assertIn("below 90.0%", " ".join(report["failure_reasons"]))

    def test_orpheus_transcript_rejects_evidenced_name_plus_unlisted_mismatch(self):
        expected = "IEEE Spectrum profiles Jernej Barbič today."
        observed = "IEEE Spectrum profiles Jernesh Smith today".split()
        words = [
            {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
            for index, word in enumerate(observed)
        ]

        report = tts._orpheus_transcript_report(expected, words)

        self.assertFalse(report["verified"])
        self.assertEqual(report["phonetic_substitutions"], [])

    async def test_orpheus_verifier_corroborates_phonetic_spelling_at_opening_edge(self):
        expected = (
            "Skild was founded in 2023 by Carnegie Mellon robotics researchers "
            "Deepak Pathak and Abhinav Gupta."
        )

        def words(text: str) -> list[dict]:
            return [
                {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
                for index, word in enumerate(text.split())
            ]

        transcript = words(
            "Skilled was founded in 2023 by Carnegie Mellon robotics researchers "
            "Deepak Pathak and Abhinav Gupta"
        )
        messages: list[str] = []
        with tempfile.TemporaryDirectory() as temp_dir:
            transcriber = AsyncMock(
                side_effect=[
                    (transcript, {"passed": True}),
                    (transcript, {"passed": True}),
                ]
            )
            process = AsyncMock(return_value=(0, ""))
            with (
                patch(
                    "backend.pipeline.av_sync.ensure_word_transcript",
                    transcriber,
                ),
                patch.object(tts, "stream_subprocess", process),
            ):
                report = await tts._verify_orpheus_part(
                    Path(temp_dir) / "skild-opening.wav",
                    expected,
                    Path(temp_dir) / "verification",
                    emit=messages.append,
                )

        self.assertTrue(report["verified"])
        self.assertTrue(report["leading_anchor"])
        self.assertFalse(report["exact_leading_anchor"])
        self.assertTrue(report["exact_trailing_anchor"])
        self.assertEqual(
            report["verification_mode"],
            "corroborated_phonetic_substitution",
        )
        self.assertEqual(transcriber.await_count, 2)
        self.assertEqual(process.await_count, 1)
        self.assertIn("corroborated", " ".join(messages))

    async def test_orpheus_verifier_rejects_uncorroborated_phonetic_spelling(self):
        expected = (
            "North American robotics startup Skild AI has released a new robot "
            "foundation."
        )

        def words(company_name: str) -> list[dict]:
            observed = (
                "North American robotics startup "
                f"{company_name} AI has released a new robot foundation"
            ).split()
            return [
                {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
                for index, word in enumerate(observed)
            ]

        with tempfile.TemporaryDirectory() as temp_dir:
            transcriber = AsyncMock(
                side_effect=[
                    (words("Skilled"), {"passed": True}),
                    (words("Skill"), {"passed": True}),
                    (words("Skillet"), {"passed": True}),
                ]
            )
            process = AsyncMock(return_value=(0, ""))
            with (
                patch(
                    "backend.pipeline.av_sync.ensure_word_transcript",
                    transcriber,
                ),
                patch.object(tts, "stream_subprocess", process),
            ):
                with self.assertRaisesRegex(
                    tts.TtsIntegrityError,
                    "not corroborated",
                ):
                    await tts._verify_orpheus_part(
                        Path(temp_dir) / "skild.wav",
                        expected,
                        Path(temp_dir) / "verification",
                        emit=lambda _message: None,
                    )

        self.assertEqual(transcriber.await_count, 3)
        self.assertEqual(process.await_count, 2)

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

    def test_orpheus_transcript_normalizes_office_possessive_homophone(self):
        expected = "Qwen Office's implied Harness score ranked highest."

        def report(product_word: str) -> dict:
            observed = f"Qwen {product_word} implied Harness score ranked highest".split()
            words = [
                {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
                for index, word in enumerate(observed)
            ]
            return tts._orpheus_transcript_report(expected, words)

        accepted = report("offices")

        self.assertTrue(accepted["verified"])
        self.assertEqual(accepted["exact_asr_word_coverage"], 1.0)
        self.assertFalse(report("office")["verified"])
        self.assertFalse(report("officers")["verified"])

    def test_orpheus_transcript_normalizes_qubit_possessive_homophone_only(self):
        expected = "Finding the qubits' transition frequencies completed the measurement."

        def report(qubit_word: str) -> dict:
            observed = f"Finding the {qubit_word} transition frequencies completed the measurement"
            words = [
                {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
                for index, word in enumerate(observed.split())
            ]
            return tts._orpheus_transcript_report(expected, words)

        accepted = report("qubit's")
        self.assertTrue(accepted["verified"])
        self.assertEqual(accepted["exact_asr_word_coverage"], 1.0)
        self.assertFalse(report("qubit")["verified"])
        self.assertFalse(report("cubits")["verified"])
        self.assertFalse(report("")["verified"])
        self.assertFalse(report("qubits qubits")["verified"])

    def test_orpheus_transcript_normalizes_nikkei_spelling_in_publication_name(self):
        expected = "Nikkei Asia reports that China is restricting exports"

        def report(observed: str) -> dict:
            words = [
                {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
                for index, word in enumerate(observed.split())
            ]
            return tts._orpheus_transcript_report(expected, words)

        self.assertTrue(
            report("Nikkei Asia reports that China is restricting exports")["verified"]
        )
        self.assertTrue(
            report("Nikke Asia reports that China is restricting exports")["verified"]
        )
        self.assertTrue(
            report("Nikkei Aja reports that China is restricting exports")["verified"]
        )
        self.assertFalse(
            report("Nike Asia reports that China is restricting exports")["verified"]
        )
        self.assertFalse(
            report("Nikke Aja reports that China is restricting exports")["verified"]
        )
        self.assertFalse(
            report("Nikkei Ajax reports that China is restricting exports")["verified"]
        )
        self.assertFalse(
            report("Nikkei Asha reports that China is restricting exports")["verified"]
        )
        self.assertFalse(
            report("Nikke Europe reports that China is restricting exports")["verified"]
        )
        self.assertFalse(
            report("Nikke reports that China is restricting exports")["verified"]
        )

    def test_orpheus_transcript_normalizes_spoken_decimal_and_api_initialism(self):
        expected = "Qwen 3.8 Max API pricing reported"

        def report(observed: str) -> dict:
            words = [
                {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
                for index, word in enumerate(observed.split())
            ]
            return tts._orpheus_transcript_report(expected, words)

        spoken = report("Qwen three point eight Max A P I pricing reported")
        numeric = report("Qwen 3.8 Max API pricing reported")

        self.assertTrue(spoken["verified"])
        self.assertTrue(numeric["verified"])
        self.assertEqual(spoken["exact_asr_word_coverage"], 1.0)
        self.assertFalse(report("Qwen three eight Max A P I pricing reported")["verified"])
        self.assertFalse(report("Qwen three point eight Max A I pricing reported")["verified"])

    def test_orpheus_transcript_normalizes_qwen_four_prompt_homophones(self):
        expected = "billion parameters built on its next-generation Qwen 4 architecture."

        def report(observed: str) -> dict:
            words = [
                {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
                for index, word in enumerate(observed.split())
            ]
            return tts._orpheus_transcript_report(expected, words)

        queue_when = report(
            "billion parameters built on its next generation queue when four architecture"
        )
        q_went = report(
            "billion parameters built on its next generation Q went for architecture"
        )
        joined = report(
            "billion parameters built on its next generation queue Wen4 architecture"
        )

        self.assertTrue(queue_when["verified"])
        self.assertTrue(q_went["verified"])
        self.assertTrue(joined["verified"])
        self.assertEqual(queue_when["matched_exact_words"], 10)
        self.assertEqual(q_went["matched_exact_words"], 10)
        self.assertFalse(
            report(
                "billion parameters built on its next generation Q went five architecture"
            )["verified"]
        )
        self.assertFalse(
            report(
                "billion parameters built on its next generation Q went for architects"
            )["verified"]
        )
        self.assertFalse(
            report(
                "billion parameters built on its next generation queue Wen5 architecture"
            )["verified"]
        )

    def test_orpheus_transcript_normalizes_spoken_v4_model_prefix(self):
        expected = "The company says the model rivals V4-Flash."

        def report(observed: str) -> dict:
            words = [
                {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
                for index, word in enumerate(observed.split())
            ]
            return tts._orpheus_transcript_report(expected, words)

        spoken = report("The company says the model rivals V four Flash")

        self.assertTrue(spoken["verified"])
        self.assertEqual(spoken["matched_exact_words"], 8)
        self.assertFalse(
            report("The company says the model rivals V five Flash")["verified"]
        )
        self.assertFalse(
            report("The company says the model rivals the Flash")["verified"]
        )

    def test_orpheus_transcript_normalizes_each_spoken_decimal_digit(self):
        expected = "revenue of 2.751 billion yuan"
        observed = "revenue of two point seven five one billion yuan".split()
        words = [
            {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
            for index, word in enumerate(observed)
        ]

        report = tts._orpheus_transcript_report(expected, words)

        self.assertTrue(report["verified"])
        self.assertEqual(report["exact_asr_word_coverage"], 1.0)

    def test_orpheus_transcript_normalizes_decimal_split_across_whisper_words(self):
        expected = "revenue of 2.751 billion yuan"
        observed = ["revenue", "of", "2", ".751", "billion", "yuan"]
        words = [
            {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
            for index, word in enumerate(observed)
        ]

        report = tts._orpheus_transcript_report(expected, words)

        self.assertTrue(report["verified"])
        self.assertEqual(report["exact_asr_word_coverage"], 1.0)

    def test_orpheus_transcript_normalizes_split_decimal_before_terminal_period(self):
        expected = "value 2.751."

        def report(observed: list[str]) -> dict:
            words = [
                {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
                for index, word in enumerate(observed)
            ]
            return tts._orpheus_transcript_report(expected, words)

        self.assertTrue(report(["value", "2", ".751."])["verified"])
        self.assertFalse(report(["value", "2", "751"])["verified"])

    def test_orpheus_transcript_does_not_merge_following_number_into_decimal(self):
        expected = "Values 3.8, 5 reported"
        observed = "Values 3.85 reported".split()
        words = [
            {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
            for index, word in enumerate(observed)
        ]

        report = tts._orpheus_transcript_report(expected, words)

        self.assertFalse(report["verified"])
        self.assertLess(report["exact_asr_word_coverage"], 1.0)

    def test_orpheus_decimal_repetition_retains_true_second_onset(self):
        expected = "Value 3.85 now."
        observed = (
            "Value three point eight five now "
            "Value three point eight five now"
        ).split()
        words = [
            {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
            for index, word in enumerate(observed)
        ]

        report = tts._orpheus_transcript_report(expected, words)

        self.assertEqual(report["repeat_start_seconds"], 1.2)

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

    def test_orpheus_cache_rejects_legacy_retimed_audio(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "part.wav"
            write_wav(path)
            text = "A fully verified utterance."
            metadata = tts._write_orpheus_part_metadata(
                path,
                text,
                job_id="job-1",
                request_token_budget=512,
                integrity=self.verified_report(None, text, None),
            )
            self.assertIsNotNone(tts._load_cached_orpheus_part(path, text))

            metadata["speed_percent"] = 140
            tts._part_metadata_path(path).write_text(
                json.dumps(metadata), encoding="utf-8"
            )

            self.assertIsNone(tts._load_cached_orpheus_part(path, text))

    def test_orpheus_cache_requires_current_integrity_verifier_version(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "part.wav"
            write_wav(path)
            text = "A fully verified utterance."
            metadata = tts._write_orpheus_part_metadata(
                path,
                text,
                job_id="job-1",
                request_token_budget=512,
                integrity=self.verified_report(None, text, None),
            )
            metadata_path = tts._part_metadata_path(path)

            self.assertEqual(
                metadata["integrity_verifier_version"],
                tts.ORPHEUS_INTEGRITY_VERIFIER_VERSION,
            )
            for stale_version in (
                None,
                tts.ORPHEUS_INTEGRITY_VERIFIER_VERSION - 1,
                True,
            ):
                with self.subTest(stale_version=stale_version):
                    stale = dict(metadata)
                    if stale_version is None:
                        stale.pop("integrity_verifier_version")
                    else:
                        stale["integrity_verifier_version"] = stale_version
                    metadata_path.write_text(json.dumps(stale), encoding="utf-8")
                    self.assertIsNone(tts._load_cached_orpheus_part(path, text))

    def test_orpheus_cache_rejects_duration_only_preview_sidecar(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "preview.wav"
            write_wav(path)
            text = "A duration-only voice preview."
            metadata = tts._write_orpheus_part_metadata(
                path,
                text,
                job_id="preview-job",
                request_token_budget=512,
                integrity={
                    "verified": True,
                    "method": "duration_only_preview",
                    "expected_words": 4,
                },
            )

            self.assertEqual(
                metadata["integrity_verifier_version"],
                tts.ORPHEUS_INTEGRITY_VERIFIER_VERSION,
            )
            self.assertIsNone(tts._load_cached_orpheus_part(path, text))

    def test_orpheus_cache_rejects_non_object_sidecar(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "part.wav"
            write_wav(path)
            text = "A malformed cache sidecar."
            tts._part_metadata_path(path).write_text("[]", encoding="utf-8")

            self.assertIsNone(tts._load_cached_orpheus_part(path, text))

    async def test_orpheus_old_verifier_sidecar_revalidates_wav_without_post(self):
        requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(500)

        original_client = httpx.AsyncClient

        def client_factory(**kwargs):
            return original_client(transport=httpx.MockTransport(handler), **kwargs)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            script = root / "script.txt"
            script.write_text("Speaker 1: Revalidate this existing waveform.")
            output_dir = root / "audio"
            output_dir.mkdir()
            text = "Revalidate this existing waveform."
            path = output_dir / "tts_input_generated.wav"
            write_wav(path)
            metadata = tts._write_orpheus_part_metadata(
                path,
                text,
                job_id="old-job",
                request_token_budget=512,
                integrity=self.verified_report(None, text, None),
            )
            metadata.pop("integrity_verifier_version")
            tts._part_metadata_path(path).write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            verifier = AsyncMock(side_effect=self.verified_report)

            with (
                patch.object(config, "ORPHEUS_TTS_API_KEY", "test-secret"),
                patch.object(tts.httpx, "AsyncClient", client_factory),
                patch.object(tts, "_verify_orpheus_part", verifier),
            ):
                result = await tts.generate_tts(
                    str(script), str(output_dir), ["tara"], "orpheus-en"
                )

            refreshed = json.loads(tts._part_metadata_path(path).read_text())

        self.assertEqual(Path(result), path.resolve())
        self.assertEqual(requests, [])
        verifier.assert_awaited_once_with(
            path.resolve(),
            text,
            output_dir.resolve() / "verification" / "tts_input",
            emit=unittest.mock.ANY,
            adjudicate_asr=True,
        )
        self.assertEqual(refreshed["job_id"], "recovered-local-output")
        self.assertEqual(
            refreshed["integrity_verifier_version"],
            tts.ORPHEUS_INTEGRITY_VERIFIER_VERSION,
        )

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

    async def test_pocket_http_generates_verified_story_parts_and_manifest(self):
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(
                200,
                headers={"Content-Type": "audio/wav"},
                content=pocket_streaming_wav_bytes(),
            )

        original_client = httpx.AsyncClient

        def client_factory(**kwargs):
            return original_client(transport=httpx.MockTransport(handler), **kwargs)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            script = root / "script.txt"
            script.write_text(
                "Speaker 1: Opening words stay together for natural delivery.\n"
                "Speaker 1: This story reports 44 billion yuan in funding."
            )
            verifier = AsyncMock(side_effect=self.verified_report)
            with (
                patch.object(config, "POCKET_TTS_URL", "http://pocket.test"),
                patch.object(config, "POCKET_TTS_CHUNK_WORDS", 240),
                patch.object(tts.httpx, "AsyncClient", client_factory),
                patch.object(tts, "_verify_orpheus_part", verifier),
            ):
                result = await tts.generate_tts(
                    str(script), str(root / "audio"), ["alba"], "pocket-tts-en"
                )

            output = Path(result)
            manifest = json.loads((output.parent / "tts_manifest.json").read_text())
            output_info = tts._read_pcm_wav(output)
            self.assertIn("44 billion", script.read_text())
            self.assertIn(
                "forty-four billion",
                (output.parent / "synthesis" / "tts_input_part_002.txt").read_text(),
            )

        self.assertEqual(len(requests), 2)
        self.assertTrue(all(request.url.path == "/tts" for request in requests))
        self.assertTrue(all(b"voice_url=alba" in request.content for request in requests))
        self.assertIn(b"forty-four+billion", requests[1].content)
        self.assertIn("44 billion", verifier.await_args_list[1].args[1])
        self.assertEqual(output_info.frame_count, 96_000)
        self.assertEqual(manifest["model"], "pocket-tts-en")
        self.assertEqual(manifest["chunk_count"], 2)
        self.assertEqual(manifest["integrity"]["verified_source_coverage"], 1.0)
        self.assertEqual(
            manifest["continuity"]["strategy"],
            "physical_script_lines_then_complete_sentences",
        )
        self.assertEqual(manifest["continuity"]["arbitrary_mid_sentence_splits"], 0)
        self.assertEqual(manifest["continuity"]["physical_script_line_count"], 2)
        self.assertEqual(manifest["continuity"]["intra_line_application_join_count"], 0)
        self.assertEqual(
            manifest["continuity"]["silence_measurement"]["join_seconds"],
            [4.0],
        )
        self.assertEqual(
            manifest["continuity"]["silence_measurement"]["internal_long_silence_count"],
            0,
        )
        self.assertEqual(verifier.await_count, 2)

    async def test_pocket_integrity_failure_retries_only_failed_part_budget(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            script = root / "script.txt"
            script.write_text("A retryable Pocket paragraph.")
            generate = AsyncMock(
                side_effect=[
                    tts.TtsIntegrityError("missing word", part_key="part-001"),
                    "/verified.wav",
                ]
            )
            with patch.object(tts, "_generate_pocket_tts", generate):
                result = await tts.generate_tts(
                    str(script), str(root / "audio"), ["alba"], "pocket-tts-en"
                )

        self.assertEqual(result, "/verified.wav")
        self.assertEqual(generate.await_count, 2)

    def test_pocket_cache_is_bound_to_voice_and_deployment_revision(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "part.wav"
            write_wav(path)
            text = "A fully verified Pocket paragraph."
            metadata = tts._write_pocket_part_metadata(
                path,
                text,
                "alba",
                integrity=self.verified_report(None, text, None),
                generation_seconds=1.25,
            )
            self.assertIsNotNone(tts._load_cached_pocket_part(path, text, "alba"))
            self.assertIsNone(tts._load_cached_pocket_part(path, text, "marius"))

            with patch.object(config, "POCKET_TTS_MODEL_REVISION", "new-revision"):
                self.assertIsNone(tts._load_cached_pocket_part(path, text, "alba"))
            self.assertEqual(metadata["audio_sha256"], tts._file_sha256(path))

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

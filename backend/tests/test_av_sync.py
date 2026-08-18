import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

from backend import config
from backend.pipeline import av_sync


class TranscriptRetryTests(unittest.IsolatedAsyncioTestCase):
    def test_short_complete_transcript_passes_preflight(self):
        words = [
            {"text": text, "start": index * 0.3, "end": index * 0.3 + 0.2}
            for index, text in enumerate(("short", "clips", "still", "align"))
        ]

        self.assertEqual(av_sync._transcript_quality(words), (True, ""))

    async def test_exhausted_mlx_retries_return_warning_instead_of_raising(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "vibevoice.wav"
            audio.write_bytes(b"wave")
            transcribe = AsyncMock(return_value=(1, "temporary failure"))

            with (
                patch.object(av_sync, "_cache_is_current", return_value=False),
                patch.object(av_sync, "_mlx_command", return_value=["mlx-whisper"]),
                patch.object(av_sync, "stream_subprocess", transcribe),
                patch.object(config, "AV_SYNC_TRANSCRIBE_MAX_RETRIES", 2),
            ):
                words, report = await av_sync.ensure_word_transcript(audio, root)

            self.assertEqual(words, [])
            self.assertFalse(report["passed"])
            self.assertEqual(report["backend"], "unavailable")
            self.assertEqual(report["attempts"], 3)
            self.assertEqual(transcribe.await_count, 3)
            self.assertEqual(len(report["failure_reasons"]), 3)

    async def test_mlx_success_after_retry_is_recorded(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "vibevoice.wav"
            audio.write_bytes(b"wave")
            words = [
                {"text": f"word-{index}", "start": index, "end": index + 0.5}
                for index in range(12)
            ]
            transcribe = AsyncMock(side_effect=[(1, "busy"), (0, "ok")])

            with (
                patch.object(av_sync, "_cache_is_current", return_value=False),
                patch.object(av_sync, "_mlx_command", return_value=["mlx-whisper"]),
                patch.object(av_sync, "_load_words", return_value=words),
                patch.object(av_sync, "stream_subprocess", transcribe),
                patch.object(config, "AV_SYNC_TRANSCRIBE_MAX_RETRIES", 2),
            ):
                actual, report = await av_sync.ensure_word_transcript(audio, root)

            self.assertEqual(actual, words)
            self.assertTrue(report["passed"])
            self.assertEqual(report["backend"], "mlx-whisper")
            self.assertEqual(report["attempts"], 2)
            self.assertEqual(transcribe.await_count, 2)

    def test_hyperframes_is_not_a_transcription_backend(self):
        self.assertFalse(hasattr(av_sync, "_hyperframes_command"))

    def test_tail_window_replaces_overlap_and_keeps_monotonic_words(self):
        primary = [
            {"text": "before", "start": 8.0, "end": 8.4},
            {"text": "stale", "start": 12.0, "end": 12.4},
        ]
        tail = [
            {"text": "thanks", "start": 12.1, "end": 12.5},
            {"text": "tomorrow", "start": 13.0, "end": 13.5},
        ]

        merged = av_sync._replace_tail_window(primary, tail, 10.0)

        self.assertEqual(
            [word["text"] for word in merged],
            ["before", "thanks", "tomorrow"],
        )

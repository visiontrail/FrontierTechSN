import asyncio
import logging
import os
import signal
import sys
import unittest
from unittest.mock import patch

from backend.pipeline import process_logging
from backend.pipeline.process_logging import stream_subprocess

logger = logging.getLogger("test")


def python_command(source: str) -> list[str]:
    return [sys.executable, "-u", "-c", source]


class StreamSubprocessTests(unittest.IsolatedAsyncioTestCase):
    async def test_throttles_carriage_return_progress_redraws(self):
        lines: list[str] = []
        # 200 tqdm-style redraws on one line, then a real newline-terminated line.
        source = (
            "import sys\n"
            "for i in range(200):\n"
            "    sys.stdout.write('step %d\\r' % i)\n"
            "sys.stdout.write('done\\n')\n"
        )

        returncode, _ = await stream_subprocess(
            name="P",
            command=python_command(source),
            logger=logger,
            log=lines.append,
            progress_interval=30.0,
        )

        self.assertEqual(returncode, 0)
        progress = [line for line in lines if line.startswith("P| step ")]
        # One sample for the whole burst, not 200 lines. This ratio is the fix:
        # a 12-minute TTS run used to emit ~4500 lines and outrun the log reader.
        self.assertLessEqual(len(progress), 3)
        self.assertIn("P| done", lines)

    async def test_final_progress_frame_is_logged_when_output_ends_mid_bar(self):
        """A run that stops on a redraw (killed, or a bar that never closes)
        must still show where it got to."""
        lines: list[str] = []
        source = (
            "import sys\n"
            "for i in range(200):\n"
            "    sys.stdout.write('step %d\\r' % i)\n"
        )

        await stream_subprocess(
            name="P",
            command=python_command(source),
            logger=logger,
            log=lines.append,
            progress_interval=30.0,
        )

        self.assertIn("P| step 199", lines)

    async def test_keeps_every_newline_terminated_line(self):
        lines: list[str] = []
        source = "import sys\nfor i in range(50):\n    sys.stdout.write('line %d\\n' % i)\n"

        await stream_subprocess(
            name="P",
            command=python_command(source),
            logger=logger,
            log=lines.append,
            progress_interval=30.0,
        )

        emitted = [line for line in lines if line.startswith("P| line ")]
        self.assertEqual(len(emitted), 50)

    async def test_stall_timeout_fires_while_child_is_silent(self):
        lines: list[str] = []
        # Prints once, then goes quiet forever.
        source = "import sys, time\nsys.stdout.write('alive\\n')\ntime.sleep(600)\n"

        with patch.object(process_logging, "WATCHDOG_INTERVAL", 0.05):
            with self.assertRaisesRegex(TimeoutError, "no output for"):
                await stream_subprocess(
                    name="P",
                    command=python_command(source),
                    logger=logger,
                    log=lines.append,
                    stall_timeout=0.3,
                    timeout=600,
                )

        self.assertIn("P| alive", lines)

    async def test_live_output_survives_a_short_stall_timeout(self):
        """A slow-but-talking process must not be killed by the stall watchdog."""
        lines: list[str] = []
        source = (
            "import sys, time\n"
            "for i in range(10):\n"
            "    sys.stdout.write('tick %d\\n' % i)\n"
            "    time.sleep(0.1)\n"
        )

        with patch.object(process_logging, "WATCHDOG_INTERVAL", 0.05):
            returncode, _ = await stream_subprocess(
                name="P",
                command=python_command(source),
                logger=logger,
                log=lines.append,
                stall_timeout=0.5,
            )

        self.assertEqual(returncode, 0)
        self.assertIn("P| tick 9", lines)

    async def test_overall_timeout_still_applies_to_a_chatty_process(self):
        lines: list[str] = []
        source = (
            "import sys, time\n"
            "while True:\n"
            "    sys.stdout.write('busy\\n')\n"
            "    time.sleep(0.05)\n"
        )

        with patch.object(process_logging, "WATCHDOG_INTERVAL", 0.05):
            with self.assertRaisesRegex(TimeoutError, "still running after"):
                await stream_subprocess(
                    name="P",
                    command=python_command(source),
                    logger=logger,
                    log=lines.append,
                    stall_timeout=60,
                    timeout=0.3,
                )

    async def test_timeout_kills_the_whole_process_tree(self):
        """The shell wrapper is not the process doing the work; killing only it
        would leave the real child (here: a sleeping grandchild) running."""
        marker = asyncio.Queue()
        source = (
            "import os, sys, time\n"
            "pid = os.fork()\n"
            "if pid:\n"
            "    sys.stdout.write('child %d\\n' % pid)\n"
            "    sys.stdout.flush()\n"
            "time.sleep(600)\n"
        )
        lines: list[str] = []

        def capture(line: str):
            lines.append(line)
            if line.startswith("P| child "):
                marker.put_nowait(int(line.rsplit(" ", 1)[1]))

        with patch.object(process_logging, "WATCHDOG_INTERVAL", 0.05):
            with self.assertRaises(TimeoutError):
                await stream_subprocess(
                    name="P",
                    command=python_command(source),
                    logger=logger,
                    log=capture,
                    stall_timeout=0.3,
                )

        grandchild = marker.get_nowait()
        await asyncio.sleep(0.2)
        with self.assertRaises(ProcessLookupError):
            os.kill(grandchild, signal.SIGTERM)


if __name__ == "__main__":
    unittest.main()

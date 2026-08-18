import logging
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from backend import logging_setup
from backend.logging_setup import DroppingQueueHandler


class BlockingHandler(logging.Handler):
    """Stands in for a log sink that has stopped draining — a full stdout pipe,
    or a terminal the OS has throttled under memory pressure."""

    # NB: not `self.release` / `self.acquire` — those are logging.Handler's own
    # lock methods, and shadowing them wedges logging.shutdown() at exit.
    def __init__(self, gate: threading.Event):
        super().__init__()
        self.gate = gate
        self.emitted = threading.Event()

    def emit(self, record):
        self.emitted.set()
        self.gate.wait(30)


class LoggingSetupTestCase(unittest.TestCase):
    """Snapshots and restores the process-wide logging state each test."""

    def setUp(self):
        root = logging.getLogger()
        self._root_handlers = root.handlers[:]
        self._root_level = root.level
        self._uvicorn = {
            name: (logging.getLogger(name).handlers[:], logging.getLogger(name).propagate)
            for name in logging_setup._UVICORN_LOGGERS
        }
        logging_setup._listeners[:] = []
        logging_setup._configured = False

        self._tmp = tempfile.TemporaryDirectory()
        self.log_dir = Path(self._tmp.name)
        patcher = patch.object(logging_setup, "log_dir", lambda: self.log_dir)
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        logging_setup.shutdown_logging(timeout=1)
        root = logging.getLogger()
        root.handlers[:] = self._root_handlers
        root.setLevel(self._root_level)
        for name, (handlers, propagate) in self._uvicorn.items():
            logger = logging.getLogger(name)
            logger.handlers[:] = handlers
            logger.propagate = propagate
        self._tmp.cleanup()


class ConfigureLoggingTests(LoggingSetupTestCase):
    def test_every_sink_sits_behind_its_own_queue(self):
        logging_setup.configure_logging()
        handlers = logging.getLogger().handlers
        self.assertEqual(len(handlers), 2, "expected a file sink and a console sink")
        for handler in handlers:
            self.assertIsInstance(handler, DroppingQueueHandler)
        # Independent queues, so one blocked sink cannot starve the other.
        self.assertIsNot(handlers[0].queue, handlers[1].queue)

    def test_uvicorn_loggers_are_funnelled_through_the_same_queues(self):
        logging_setup.configure_logging()
        for name in logging_setup._UVICORN_LOGGERS:
            logger = logging.getLogger(name)
            self.assertEqual(logger.handlers, [], name)
            self.assertTrue(logger.propagate, name)

    def test_writes_to_its_own_file_not_only_the_terminal(self):
        """The record of a run must survive the terminal that started it."""
        logging_setup.configure_logging()
        logging.getLogger("test.file").info("synthesis progress 4460/8192")
        logging_setup.shutdown_logging(timeout=5)

        log_file = self.log_dir / logging_setup.LOG_FILE_NAME
        self.assertTrue(log_file.exists(), "backend.log was not created")
        self.assertIn("synthesis progress 4460/8192", log_file.read_text(encoding="utf-8"))


class BlockedSinkTests(LoggingSetupTestCase):
    def _wedge(self, maxsize: int) -> BlockingHandler:
        gate = threading.Event()
        blocking = BlockingHandler(gate)
        self.addCleanup(gate.set)

        handler = logging_setup._async_sink(blocking, maxsize)
        root = logging.getLogger()
        root.handlers[:] = [handler]
        root.setLevel(logging.INFO)

        logging.getLogger("test.blocked").info("first record wedges the sink")
        self.assertTrue(blocking.emitted.wait(5), "handler never ran")
        return blocking

    def test_a_blocked_sink_does_not_block_the_caller(self):
        """The regression this module exists for: a log sink that stops draining
        must never stall the thread doing the logging — that thread is the
        asyncio event loop streaming subprocess output."""
        self._wedge(maxsize=10_000)
        logger = logging.getLogger("test.blocked")

        started = time.monotonic()
        for i in range(500):
            logger.info("record %d", i)
        elapsed = time.monotonic() - started

        self.assertLess(elapsed, 2.0, f"logging blocked for {elapsed:.1f}s behind a stalled sink")

    def test_a_blocked_sink_drops_instead_of_growing_without_bound(self):
        """Memory pressure is what stalls the sink in the first place, so the
        backlog must be capped rather than allowed to feed the problem."""
        self._wedge(maxsize=50)
        logger = logging.getLogger("test.blocked")
        handler = logging.getLogger().handlers[0]

        for i in range(5_000):
            logger.info("record %d", i)

        self.assertLessEqual(handler.queue.qsize(), 50)
        self.assertGreater(handler.dropped, 0, "nothing was dropped despite a wedged sink")

    def test_recovered_sink_reports_the_gap(self):
        blocking = self._wedge(maxsize=10)
        logger = logging.getLogger("test.blocked")
        handler = logging.getLogger().handlers[0]

        for i in range(500):
            logger.info("record %d", i)
        self.assertGreater(handler.dropped, 0)

        blocking.gate.set()  # sink recovers
        deadline = time.monotonic() + 5
        while handler.dropped and time.monotonic() < deadline:
            logger.info("post-recovery record")
            time.sleep(0.01)

        self.assertEqual(handler.dropped, 0, "drop counter never cleared after recovery")

    def test_shutdown_does_not_hang_on_a_blocked_sink(self):
        self._wedge(maxsize=10_000)

        started = time.monotonic()
        logging_setup.shutdown_logging(timeout=0.5)
        elapsed = time.monotonic() - started

        self.assertLess(elapsed, 3.0, f"shutdown blocked for {elapsed:.1f}s")


if __name__ == "__main__":
    unittest.main()

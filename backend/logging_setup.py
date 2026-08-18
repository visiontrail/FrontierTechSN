import atexit
import contextlib
import logging
import os
import queue
import threading
import time
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from pathlib import Path

LOG_FORMAT = "%(asctime)s [%(name)s] %(levelname)s %(message)s"

# How long to wait for buffered records to drain at exit before giving up.
SHUTDOWN_TIMEOUT = 5.0

# Per-sink backlog ceilings. These bound how much memory a stalled sink can cost
# before records start being dropped instead. Deliberately modest: the trigger
# for a stall here is memory pressure from TTS, so an unbounded backlog would
# feed the very problem it is trying to survive.
FILE_QUEUE_MAXSIZE = 50_000
CONSOLE_QUEUE_MAXSIZE = 5_000

# The backend's own log file. Rotating rather than per-run so `tail -f` always
# has one stable path, and so a runaway subprocess cannot fill the disk.
LOG_FILE_NAME = "backend.log"
LOG_FILE_MAX_BYTES = 20 * 1024 * 1024
LOG_FILE_BACKUPS = 5

# uvicorn installs its own stderr handlers on these and turns off propagation,
# so they bypass the root handler unless we funnel them back through it.
_UVICORN_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access")

_listeners: list[QueueListener] = []
_configured = False


class DroppingQueueHandler(QueueHandler):
    """A QueueHandler that discards records rather than ever blocking.

    ``QueueHandler`` normally calls ``queue.put_nowait``, which raises on a
    bounded full queue; the base class then routes that through
    ``handleError``. Dropping is the deliberate choice here: the thread calling
    this is the asyncio event loop, and losing progress lines is always
    preferable to stalling video synthesis. Drops are counted and reported once
    the sink recovers, so a gap in the log is never silent.
    """

    def __init__(self, log_queue: queue.Queue):
        super().__init__(log_queue)
        self.dropped = 0

    def enqueue(self, record: logging.LogRecord):
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            self.dropped += 1
            return
        if self.dropped:
            dropped, self.dropped = self.dropped, 0
            notice = logging.LogRecord(
                name=__name__,
                level=logging.WARNING,
                pathname=__file__,
                lineno=0,
                msg="log sink stalled; dropped %d record(s) to keep the pipeline running",
                args=(dropped,),
                exc_info=None,
            )
            with contextlib.suppress(queue.Full):
                self.queue.put_nowait(notice)


def _async_sink(handler: logging.Handler, maxsize: int) -> DroppingQueueHandler:
    """Put `handler` behind its own queue and its own listener thread.

    One thread per sink on purpose: a blocked console must not also starve the
    file log. They fail independently.
    """
    log_queue: queue.Queue = queue.Queue(maxsize=maxsize)
    listener = QueueListener(log_queue, handler, respect_handler_level=True)
    listener.start()
    _listeners.append(listener)
    return DroppingQueueHandler(log_queue)


def log_dir() -> Path:
    """Where backend.log lives. LOG_DIR is what scripts/start.sh already sets."""
    configured = os.getenv("LOG_DIR", "").strip()
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parent.parent / "logs"


def configure_logging(level: int = logging.INFO) -> list[QueueListener]:
    """Make logging independent of the main flow, in both directions.

    Two separate problems are being solved here.

    *Nothing may block the event loop.* The backend streams long-running
    subprocess output (TTS synthesis, video render) into `logging` from inside
    the asyncio event loop. A plain StreamHandler makes each of those calls a
    blocking `write()` to whatever sits on the other end of stdout. When that
    reader cannot keep up — and under TTS memory pressure the terminal's log
    reader is exactly what degrades first — the 64 KiB pipe buffer fills and the
    write blocks *on the event loop*, which then stops draining the child's
    stdout, stops firing timers, and stops serving HTTP. Every sink therefore
    sits behind its own thread, reached through a bounded queue that drops
    rather than waits.

    *Logs must not depend on the terminal.* The rotating file handler writes
    straight to `logs/backend.log`, so the record of a run survives a terminal
    that is throttled, suspended, or closed. The console sink is best-effort on
    top of that.
    """
    global _configured
    if _configured:
        return _listeners

    formatter = logging.Formatter(LOG_FORMAT)
    handlers: list[logging.Handler] = []

    directory = log_dir()
    try:
        directory.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            directory / LOG_FILE_NAME,
            maxBytes=LOG_FILE_MAX_BYTES,
            backupCount=LOG_FILE_BACKUPS,
            encoding="utf-8",
            delay=True,
        )
        file_handler.setFormatter(formatter)
        handlers.append(_async_sink(file_handler, FILE_QUEUE_MAXSIZE))
    except OSError:
        # An unwritable log directory must not stop the app from starting; the
        # console sink below still carries everything.
        pass

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    handlers.append(_async_sink(console, CONSOLE_QUEUE_MAXSIZE))

    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    for handler in handlers:
        root.addHandler(handler)
    root.setLevel(level)

    for name in _UVICORN_LOGGERS:
        uvicorn_logger = logging.getLogger(name)
        for handler in uvicorn_logger.handlers[:]:
            uvicorn_logger.removeHandler(handler)
        # Filters attached to these loggers (e.g. the access-log status filter)
        # still run first, in Logger.handle, before the queue swallows the args.
        uvicorn_logger.propagate = True

    _configured = True
    atexit.register(shutdown_logging)
    return _listeners


def _stop_listener(listener: QueueListener):
    """Stop one listener, tolerating a sink that is still wedged.

    ``QueueListener.stop()`` signals its thread by enqueuing a sentinel with
    ``put_nowait``, which *raises* on a bounded queue that a stalled sink has
    filled. Drop the backlog to make room and retry: we are shutting down, so
    those records were never going to reach the sink anyway.
    """
    try:
        listener.stop()
    except queue.Full:
        with contextlib.suppress(queue.Empty):
            while True:
                listener.queue.get_nowait()
        with contextlib.suppress(Exception):
            listener.stop()


def shutdown_logging(timeout: float = SHUTDOWN_TIMEOUT):
    """Flush and stop the listener threads. Safe to call more than once.

    ``QueueListener.stop()`` joins its thread, so if the sink it is writing to
    is itself blocked (the very case this module exists to survive) a plain stop
    would hang process exit instead. Bound the wait and give up: listener
    threads are daemons, so at worst a few trailing records are lost — far
    better than a shutdown that never returns.
    """
    global _configured
    listeners, _listeners[:] = _listeners[:], []
    _configured = False
    if not listeners:
        return
    stoppers = [
        (listener, threading.Thread(target=_stop_listener, args=(listener,), daemon=True))
        for listener in listeners
    ]
    for _, stopper in stoppers:
        stopper.start()
    # One shared budget, not `timeout` per sink.
    deadline = time.monotonic() + timeout
    for listener, stopper in stoppers:
        stopper.join(max(0.0, deadline - time.monotonic()))
        if stopper.is_alive():
            continue  # still wedged; leave the daemon thread and its handler be
        for handler in listener.handlers:
            with contextlib.suppress(Exception):
                handler.close()

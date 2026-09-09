"""Process-lifetime crash sink: never queued, rotated, or tied to a terminal."""

import faulthandler
import os
import sys
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path

_sink = None


def configure_crash_logging() -> Path:
    global _sink
    if _sink is not None:
        return Path(_sink.name)
    directory = Path(os.environ.get("LOG_DIR") or Path(__file__).resolve().parent.parent / "logs")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "backend-crash.log"
    # Keep this descriptor open for the entire process lifetime. Rotation or
    # closing it during logging shutdown would invalidate faulthandler's fd.
    sink = path.open("ab", buffering=0)
    sink.write(
        f"\n{datetime.now(timezone.utc).isoformat()} crash capture enabled pid={os.getpid()}\n".encode()
    )
    faulthandler.enable(file=sink, all_threads=True)
    _sink = sink

    def record(kind, exc_type, exc_value, exc_tb):
        try:
            header = f"\n{datetime.now(timezone.utc).isoformat()} pid={os.getpid()} {kind}\n"
            body = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
            sink.write((header + body).encode("utf-8", errors="backslashreplace"))
            os.fsync(sink.fileno())
        except Exception:
            # Preserve the original exception and the standard reporting hook.
            pass

    previous_sys = sys.excepthook
    previous_thread = threading.excepthook
    previous_unraisable = sys.unraisablehook

    def uncaught(exc_type, exc_value, exc_tb):
        record("uncaught main-thread exception", exc_type, exc_value, exc_tb)
        previous_sys(exc_type, exc_value, exc_tb)

    def thread_uncaught(args):
        record("uncaught thread exception", args.exc_type, args.exc_value, args.exc_traceback)
        previous_thread(args)

    def unraisable(args):
        record("unraisable exception", args.exc_type, args.exc_value, args.exc_traceback)
        previous_unraisable(args)

    sys.excepthook = uncaught
    threading.excepthook = thread_uncaught
    sys.unraisablehook = unraisable
    return path

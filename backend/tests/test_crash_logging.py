import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
PREAMBLE = """
import resource
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
from backend.crash_logging import configure_crash_logging
configure_crash_logging()
"""


def run_child(tmp_path, code):
    return subprocess.run(
        [sys.executable, "-c", PREAMBLE + code], cwd=ROOT,
        env={**os.environ, "LOG_DIR": str(tmp_path)},
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15,
    )


@pytest.mark.parametrize("fatal_signal", [signal.SIGABRT, signal.SIGSEGV, signal.SIGBUS])
def test_fatal_signal_saved_without_terminal(tmp_path, fatal_signal):
    child = run_child(tmp_path, f"""
import os, signal, threading
ready = threading.Event()
def background_crash_witness():
    ready.set()
    threading.Event().wait()
threading.Thread(target=background_crash_witness, daemon=True).start()
ready.wait()
def fatal_crash_witness():
    os.kill(os.getpid(), {int(fatal_signal)})
fatal_crash_witness()
""")
    assert child.returncode == -fatal_signal
    content = (tmp_path / "backend-crash.log").read_text()
    assert "Fatal Python error" in content
    assert "fatal_crash_witness" in content
    assert "background_crash_witness" in content


@pytest.mark.parametrize("mode", ["main", "thread", "unraisable"])
def test_exception_saved_without_terminal(tmp_path, mode):
    code = {
        "main": "raise RuntimeError('main crash witness')",
        "thread": """
import threading
def fail():
    raise RuntimeError('thread crash witness')
t = threading.Thread(target=fail)
t.start()
t.join()
""",
        "unraisable": """
class Broken:
    def __del__(self):
        raise RuntimeError('unraisable crash witness')
obj = Broken()
del obj
""",
    }[mode]
    child = run_child(tmp_path, code)
    assert child.returncode == (1 if mode == "main" else 0)
    content = (tmp_path / "backend-crash.log").read_text()
    assert "Traceback (most recent call last)" in content
    assert f"{mode} crash witness" in content


def test_crash_sink_survives_regular_logging_shutdown(tmp_path):
    child = run_child(tmp_path, """
from backend.logging_setup import configure_logging, shutdown_logging
configure_logging()
shutdown_logging()
raise RuntimeError('after shutdown witness')
""")
    assert child.returncode == 1
    assert "after shutdown witness" in (tmp_path / "backend-crash.log").read_text()

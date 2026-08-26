"""Cross-process pacing for Gemini and ChatGPT generation requests."""

from __future__ import annotations

import fcntl
import os
import sys
import time
from pathlib import Path
from typing import Callable


MIN_INTERVAL_SECONDS = 3 * 60.0
MAX_INTERVAL_SECONDS = 10 * 60.0
DEFAULT_INTERVAL_SECONDS = MIN_INTERVAL_SECONDS
RATE_LIMITED_SITES = frozenset({"chatgpt", "gemini"})
RATE_LIMITED_ACTIONS = frozenset({"ask", "image"})


def is_rate_limited_command(args: list[str] | tuple[str, ...]) -> bool:
    """Return whether an OpenCLI command starts a rate-sensitive generation.

    Page reads, conversation-detail recovery, status checks, and model selection
    do not submit a prompt. Charging those commands a generation slot turns a
    bounded recovery poll into several minutes of artificial delay.
    """
    return (
        len(args) >= 2
        and str(args[0]).strip().lower() in RATE_LIMITED_SITES
        and str(args[1]).strip().lower() in RATE_LIMITED_ACTIONS
    )


def normalize_interval(value: object | None) -> float:
    """Coerce an interval and fail safe inside the supported 3–10 minute range."""
    if value is None:
        value = os.getenv(
            "OPENCLI_WEB_REQUEST_INTERVAL_SECONDS",
            str(DEFAULT_INTERVAL_SECONDS),
        )
    try:
        interval = float(value)
    except (TypeError, ValueError):
        interval = DEFAULT_INTERVAL_SECONDS
    return min(MAX_INTERVAL_SECONDS, max(MIN_INTERVAL_SECONDS, interval))


def default_state_path() -> Path:
    override = os.getenv("OPENCLI_WEB_RATE_LIMIT_STATE_PATH", "").strip()
    if override:
        return Path(override).expanduser()
    project_root = Path(__file__).resolve().parents[2]
    return project_root / ".run" / "opencli-web-rate-limit"


def wait_for_opencli_web_slot(
    site: str,
    *,
    interval: object | None = None,
    state_path: Path | None = None,
    clock: Callable[[], float] = time.time,
    sleeper: Callable[[float], None] = time.sleep,
    reporter: Callable[[str], None] | None = None,
) -> float:
    """Wait for and reserve the next global OpenCLI web-request start slot.

    The exclusive file lock stays held through any wait, so concurrent backend
    tasks and direct wrapper invocations cannot reserve the same start time.
    The timestamp survives process restarts; a crashed waiter releases the OS
    lock automatically.
    """
    selected_interval = normalize_interval(interval)
    path = state_path or default_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    report = reporter or (lambda message: print(message, file=sys.stderr, flush=True))

    with path.open("a+", encoding="utf-8") as state:
        fcntl.flock(state.fileno(), fcntl.LOCK_EX)
        try:
            state.seek(0)
            try:
                last_started_at = float(state.read().strip() or "0")
            except ValueError:
                last_started_at = 0.0

            now = clock()
            # Cap the wait at one interval so a wall-clock rollback cannot
            # strand the pipeline behind a stale timestamp from the future.
            delay = min(
                selected_interval,
                max(0.0, last_started_at + selected_interval - now),
            )
            if delay > 0:
                report(
                    f"OpenCLI pacing: waiting {delay:.1f}s before the next "
                    f"{site} web request (minimum {selected_interval:.1f}s)."
                )
                sleeper(delay)

            started_at = clock()
            state.seek(0)
            state.truncate()
            state.write(f"{started_at:.6f}\n")
            state.flush()
            os.fsync(state.fileno())
            return delay
        finally:
            fcntl.flock(state.fileno(), fcntl.LOCK_UN)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not is_rate_limited_command(args):
        return 0
    wait_for_opencli_web_slot(args[0].strip().lower())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

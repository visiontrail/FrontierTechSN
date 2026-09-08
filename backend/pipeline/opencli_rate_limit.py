"""Cross-process generation pacing and provider-wide access cooldowns."""

from __future__ import annotations

import fcntl
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable


MIN_INTERVAL_SECONDS = 10 * 60.0
MAX_INTERVAL_SECONDS = 30 * 60.0
DEFAULT_INTERVAL_SECONDS = MIN_INTERVAL_SECONDS
RATE_LIMITED_SITES = frozenset({"chatgpt", "gemini"})
RATE_LIMITED_ACTIONS = frozenset({"ask", "image"})
GENERATION_QUIET_PERIOD_ACTIONS = frozenset({("chatgpt", "model")})
PROVIDER_COOLDOWN_SECONDS = 30 * 60.0
MAX_PROVIDER_COOLDOWN_SECONDS = 2 * 60 * 60.0
PROVIDER_COOLDOWN_ESCALATION_WINDOW_SECONDS = 6 * 60 * 60.0


def _cooldown_path(site: str, state_path: Path | None = None) -> Path:
    if site not in RATE_LIMITED_SITES:
        raise ValueError(f"Unsupported rate-limited site: {site}")
    return Path(f"{state_path or default_state_path()}.{site}.cooldown")


def record_opencli_rate_limit(
    site: str, *, state_path: Path | None = None, clock=time.time,
) -> float:
    """Persist a provider-wide circuit breaker, including reads and refreshes."""
    path = _cooldown_path(site, state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as state:
        fcntl.flock(state.fileno(), fcntl.LOCK_EX)
        try:
            state.seek(0)
            try:
                prior = json.load(state)
            except (ValueError, TypeError):
                prior = {}
            now = clock()
            recent = (
                now - float(prior.get("recorded_at", 0))
                < PROVIDER_COOLDOWN_ESCALATION_WINDOW_SECONDS
            )
            previous_delay = float(prior.get("delay", 0)) if recent else 0
            delay = min(MAX_PROVIDER_COOLDOWN_SECONDS,
                        max(PROVIDER_COOLDOWN_SECONDS, previous_delay * 2))
            until = max(now + delay, float(prior.get("until", 0)))
            state.seek(0)
            state.truncate()
            json.dump({"recorded_at": now, "until": until, "delay": delay}, state)
            state.flush()
            os.fsync(state.fileno())
            return until
        finally:
            fcntl.flock(state.fileno(), fcntl.LOCK_UN)


def opencli_cooldown_remaining(
    site: str, *, state_path: Path | None = None, clock=time.time,
) -> float:
    if site not in RATE_LIMITED_SITES:
        return 0.0
    path = _cooldown_path(site, state_path)
    try:
        with path.open(encoding="utf-8") as state:
            fcntl.flock(state.fileno(), fcntl.LOCK_SH)
            data = json.load(state)
            return min(MAX_PROVIDER_COOLDOWN_SECONDS,
                       max(0.0, float(data["until"]) - clock()))
    except FileNotFoundError:
        return 0.0


def wait_for_opencli_cooldown(site: str) -> None:
    """Direct-wrapper gate; no lock is held while sleeping."""
    while remaining := opencli_cooldown_remaining(site):
        print(f"OpenCLI {site} access cooldown: {remaining:.0f}s remaining",
              file=sys.stderr, flush=True)
        time.sleep(min(30.0, remaining))


def is_rate_limited_command(args: list[str] | tuple[str, ...]) -> bool:
    """Return whether an OpenCLI command starts a rate-sensitive generation.

    Reads and model selection skip generation slots, but still honor the
    separate provider access cooldown when the site explicitly limits access.
    """
    return (
        len(args) >= 2
        and str(args[0]).strip().lower() in RATE_LIMITED_SITES
        and str(args[1]).strip().lower() in RATE_LIMITED_ACTIONS
    )


def needs_generation_quiet_period(args: list[str] | tuple[str, ...]) -> bool:
    """Return whether a browser read must wait behind the last generation.

    ChatGPT's model picker is a conversation-level browser operation and has
    triggered the same access-limit dialog when run immediately after an audit
    response. Recovery reads remain outside this proactive gate so an owned
    partial response can still be captured without waiting for another full
    generation interval.
    """
    return (
        len(args) >= 2
        and (
            str(args[0]).strip().lower(),
            str(args[1]).strip().lower(),
        ) in GENERATION_QUIET_PERIOD_ACTIONS
    )


def normalize_interval(value: object | None) -> float:
    """Coerce an interval and fail safe inside the supported 10–30 minute range."""
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


def wait_for_opencli_generation_quiet_period(
    site: str,
    *,
    interval: object | None = None,
    state_path: Path | None = None,
    clock: Callable[[], float] = time.time,
    sleeper: Callable[[float], None] = time.sleep,
    reporter: Callable[[str], None] | None = None,
) -> float:
    """Wait until the last generation slot is old enough without reserving one.

    The same lock used by generation reservations prevents a concurrent prompt
    from starting while a model-policy check is waiting. The check does not
    update the timestamp, so the following prompt may start immediately after
    the quiet period instead of paying the interval twice.
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
            delay = min(
                selected_interval,
                max(0.0, last_started_at + selected_interval - clock()),
            )
            if delay > 0:
                report(
                    f"OpenCLI pacing: waiting {delay:.1f}s before the next "
                    f"{site} model check (generation quiet period "
                    f"{selected_interval:.1f}s)."
                )
                sleeper(delay)
            return delay
        finally:
            fcntl.flock(state.fileno(), fcntl.LOCK_UN)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    site = args[0].lower() if args else ""
    if site in RATE_LIMITED_SITES:
        wait_for_opencli_cooldown(site)
    if is_rate_limited_command(args):
        wait_for_opencli_web_slot(site)
        # A concurrent process may have opened the provider breaker while this
        # direct invocation waited for its global generation slot.
        wait_for_opencli_cooldown(site)
    elif needs_generation_quiet_period(args):
        wait_for_opencli_generation_quiet_period(site)
        wait_for_opencli_cooldown(site)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import asyncio
import contextlib
import logging
import os
import re
import shlex
import signal
import subprocess
import time
from collections.abc import Callable, Sequence
from pathlib import Path


ProcessLogCallback = Callable[[str], None]

# tqdm and friends redraw their progress bar with a carriage return many times
# per second. Logging every redraw is what made a 12-minute TTS run emit ~4500
# log lines: enough to outrun the log reader, fill the stdout pipe, and wedge
# the process. Sample them instead — the bar is a live status, not a record.
PROGRESS_LOG_INTERVAL = 5.0

# How often the watchdog wakes to compare elapsed/idle time against its limits.
WATCHDOG_INTERVAL = 5.0

# Grace period between SIGTERM and SIGKILL when tearing down a timed-out child.
TERMINATE_GRACE = 10.0


def command_to_string(command: str | Sequence[object]) -> str:
    if isinstance(command, str):
        return command.strip()
    return shlex.join(str(part) for part in command)


def log_process_result(
    *,
    name: str,
    command: str | Sequence[object],
    returncode: int | None,
    stdout: str | bytes | None,
    stderr: str | bytes | None,
    logger: logging.Logger,
    log: ProcessLogCallback | None = None,
    cwd: str | Path | None = None,
    stdout_log_limit: int | None = None,
    stderr_log_limit: int | None = None,
):
    def decode(value: str | bytes | None) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode(errors="replace")
        return value

    stdout_text = decode(stdout).rstrip()
    stderr_text = decode(stderr).rstrip()

    def truncate(value: str, limit: int | None) -> str:
        if limit is None or len(value) <= limit:
            return value
        omitted = len(value) - limit
        return f"{value[:limit]}\n... <truncated {omitted} chars>"

    stdout_text = truncate(stdout_text, stdout_log_limit)
    stderr_text = truncate(stderr_text, stderr_log_limit)
    lines = [
        f"{name} command: {command_to_string(command)}",
    ]
    if cwd:
        lines.append(f"{name} cwd: {cwd}")
    if returncode is not None:
        lines.append(f"{name} exit code: {returncode}")
    if stdout_text:
        lines.append(f"{name} stdout:")
        lines.extend(stdout_text.splitlines())
    if stderr_text:
        lines.append(f"{name} stderr:")
        lines.extend(stderr_text.splitlines())

    message = "\n".join(lines)
    if returncode == 0:
        logger.info(message)
    else:
        logger.error(message)

    if log:
        for line in lines:
            log(line)


def run_capture_logged(
    *,
    name: str,
    command: Sequence[object],
    logger: logging.Logger,
    log: ProcessLogCallback | None = None,
    cwd: str | Path | None = None,
    timeout: float | None = None,
    text: bool = True,
    stdout_log_limit: int | None = None,
    stderr_log_limit: int | None = None,
) -> subprocess.CompletedProcess:
    """Run a short subprocess whose output must be captured for parsing, while
    still logging the result. Callers can cap logged stdout/stderr for noisy
    commands without changing the returned ``CompletedProcess``.

    Unlike a bare ``subprocess.run``, a timeout does not swallow the output: the
    partial stdout/stderr captured before the process was killed is logged
    (which is exactly what you need to diagnose a hang) before the
    ``TimeoutExpired`` is re-raised.
    """
    cmd = [str(part) for part in command]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=text,
            timeout=timeout,
            cwd=str(cwd) if cwd else None,
        )
    except subprocess.TimeoutExpired as exc:
        log_process_result(
            name=f"{name} (TIMEOUT after {timeout}s)",
            command=command,
            returncode=None,
            stdout=exc.stdout,
            stderr=exc.stderr,
            logger=logger,
            log=log,
            cwd=cwd,
            stdout_log_limit=stdout_log_limit,
            stderr_log_limit=stderr_log_limit,
        )
        raise
    log_process_result(
        name=name,
        command=command,
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        logger=logger,
        log=log,
        cwd=cwd,
        stdout_log_limit=stdout_log_limit,
        stderr_log_limit=stderr_log_limit,
    )
    return result


async def _terminate_process_tree(proc: asyncio.subprocess.Process):
    """Kill the child *and everything it spawned*.

    These commands are ``bash -c 'source env; cd dir; python infer.py'``, so the
    process actually doing the work is a grandchild. Signalling only ``proc``
    reaps the shell and leaves the inference process alive, still holding the
    GPU. ``start_new_session=True`` puts the whole tree in its own process
    group, which lets us signal all of it at once.
    """
    if proc.returncode is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, PermissionError):
        pgid = None

    def signal_tree(sig: int) -> bool:
        try:
            if pgid is not None:
                os.killpg(pgid, sig)
            else:
                proc.send_signal(sig)
            return True
        except (ProcessLookupError, PermissionError):
            return False

    if signal_tree(signal.SIGTERM):
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(proc.wait(), timeout=TERMINATE_GRACE)
    if proc.returncode is None:
        signal_tree(signal.SIGKILL)
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(proc.wait(), timeout=TERMINATE_GRACE)


async def stream_subprocess(
    *,
    name: str,
    command: Sequence[object],
    logger: logging.Logger,
    log: ProcessLogCallback | None = None,
    cwd: str | Path | None = None,
    timeout: float | None = None,
    env: dict | None = None,
    stall_timeout: float | None = None,
    progress_interval: float = PROGRESS_LOG_INTERVAL,
) -> tuple[int | None, str]:
    """Run a long subprocess, streaming its merged stdout/stderr line-by-line to
    the logs *as it is produced* instead of dumping everything after it exits.

    This is what lets you watch TTS synthesis and video rendering progress live
    (in both the start.sh log and the in-app LogPanel) rather than staring at a
    silent terminal for minutes. It is ``async`` so the event loop — and the SSE
    log stream feeding the UI — stays responsive while the child runs.

    Carriage-return progress redraws (tqdm and friends) are recognised and
    sampled at most once per ``progress_interval`` seconds; newline-terminated
    lines are always logged. Without that throttle a single long render emits
    thousands of near-identical lines per minute, which is enough to outrun the
    log reader and block the event loop on a full pipe.

    Two independent ceilings guard the run. ``stall_timeout`` is the useful one:
    it fires when the child has produced *no output at all* for that long, so a
    wedged process is caught in minutes while a slow-but-live one is left alone.
    ``timeout`` is a coarse backstop on total runtime. Either way the whole
    process tree is killed and ``TimeoutError`` is raised, with everything the
    child printed already logged. Returns ``(returncode, captured_text)``.
    """

    def emit(message: str):
        # Prefer the task callback when present: it already mirrors the line to
        # the root logger (start.sh log) with the task id, the per-task
        # pipeline.log, and the SSE stream. Falling through to the module logger
        # would double-log every line to start.sh.
        if log is not None:
            log(message)
        else:
            logger.info(message)

    cmd = [str(part) for part in command]
    emit(f"{name} command: {command_to_string(command)}")
    if cwd:
        emit(f"{name} cwd: {cwd}")

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(cwd) if cwd else None,
        env=env,
        start_new_session=True,
    )

    captured: list[str] = []
    started = time.monotonic()
    last_output = started

    async def pump():
        nonlocal last_output
        assert proc.stdout is not None
        buffer = b""
        # Backdated so the first progress frame is always logged, rather than
        # depending on how large time.monotonic()'s epoch happens to be.
        last_progress_at = started - progress_interval
        pending_progress: str | None = None

        def record(line: str):
            captured.append(line)
            emit(f"{name}| {line}")

        while True:
            chunk = await proc.stdout.read(4096)
            if not chunk:
                break
            now = time.monotonic()
            # Any byte counts as liveness, even a progress frame we drop.
            last_output = now
            buffer += chunk
            # Keep the delimiter so a \r redraw can be told apart from a real
            # \n line: only the former is a transient status worth sampling.
            parts = re.split(rb"([\r\n])", buffer)
            buffer = parts.pop()
            for text, delimiter in zip(parts[::2], parts[1::2]):
                line = text.decode(errors="replace").rstrip()
                if not line:
                    continue
                if delimiter == b"\r":
                    pending_progress = line
                    if now - last_progress_at >= progress_interval:
                        last_progress_at = now
                        pending_progress = None
                        record(line)
                else:
                    pending_progress = None
                    record(line)

        tail = buffer.decode(errors="replace").rstrip()
        if tail:
            pending_progress = None
            record(tail)
        # Always show where the progress bar actually stopped.
        if pending_progress:
            record(pending_progress)

    runner = asyncio.ensure_future(asyncio.gather(pump(), proc.wait()))
    try:
        while True:
            done, _ = await asyncio.wait({runner}, timeout=WATCHDOG_INTERVAL)
            if done:
                await runner
                break

            now = time.monotonic()
            idle = now - last_output
            elapsed = now - started
            if stall_timeout is not None and idle >= stall_timeout:
                reason = (
                    f"no output for {int(idle)}s (stall limit {int(stall_timeout)}s) "
                    f"after {int(elapsed)}s of runtime"
                )
            elif timeout is not None and elapsed >= timeout:
                reason = f"still running after {int(elapsed)}s (limit {int(timeout)}s)"
            else:
                continue

            emit(f"{name} TIMEOUT — {reason}; killing process tree (partial output above)")
            raise TimeoutError(f"{name} timed out: {reason}")
    except BaseException:
        # Timeout, a cancelled pipeline, or a failure in the pump itself — none
        # of them may leave an inference process behind still holding the GPU.
        runner.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await runner
        await _terminate_process_tree(proc)
        raise

    emit(f"{name} exit code: {proc.returncode}")
    return proc.returncode, "\n".join(captured)

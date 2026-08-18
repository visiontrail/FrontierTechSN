#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# Single-port production service: the backend serves the API and the built
# frontend on one port. Override with PORT/HOST if needed.
PORT="${PORT:-8101}"
HOST="${HOST:-0.0.0.0}"

# The ASGI lifespan runs before uvicorn binds the listening socket. Without a
# launcher-level lock, a second invocation can therefore start background
# workers and open write transactions before it eventually discovers that the
# port belongs to the first instance. Keep the lock inside the ignored project
# runtime directory so separate checkouts do not block each other.
RUN_DIR="$PROJECT_ROOT/.run"
LOCK_DIR="$RUN_DIR/start.lock"
mkdir -p "$RUN_DIR"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    lock_pid="$(sed -n '1p' "$LOCK_DIR/pid" 2>/dev/null || true)"
    if [[ "$lock_pid" =~ ^[0-9]+$ ]] && kill -0 "$lock_pid" 2>/dev/null; then
        echo "FrontierTechSN is already running (launcher PID $lock_pid)."
        echo "Stop that instance before starting another one."
        exit 1
    fi
    rm -f "$LOCK_DIR/pid"
    rmdir "$LOCK_DIR" 2>/dev/null || true
    if ! mkdir "$LOCK_DIR" 2>/dev/null; then
        echo "Could not acquire startup lock: $LOCK_DIR" >&2
        exit 1
    fi
fi
printf '%s\n' "$$" > "$LOCK_DIR/pid"

LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs}"
RUN_ID="$(date '+%Y%m%d-%H%M%S')"
LOG_FILE="$LOG_DIR/start-$RUN_ID.log"
LATEST_LOG="$LOG_DIR/start-latest.log"
mkdir -p "$LOG_DIR"
ln -sf "$LOG_FILE" "$LATEST_LOG"

# Prefix every line with a timestamp WITHOUT forking a process per line.
# The previous `while read; do ... $(date) ...; done` loop spawned /bin/date
# once per line, which cannot keep up with streamed subprocess output (TTS and
# render emit progress several times a second). The 64 KiB stdout pipe then
# fills, and the backend's next log write blocks — inside its asyncio event
# loop — until something drains the pipe. That deadlocked a live TTS run for
# ten hours and surfaced as a bogus "TTS timed out" once the pipe drained.
if command -v python3 >/dev/null 2>&1; then
    exec > >(python3 -u -c '
import sys, time
for line in sys.stdin:
    sys.stdout.write(time.strftime("%Y-%m-%d %H:%M:%S ") + line)
' | tee -a "$LOG_FILE") 2>&1
else
    exec > >(tee -a "$LOG_FILE") 2>&1
fi

# Load nvm so that node/npm/npx are available
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && source "$NVM_DIR/nvm.sh"

if [ -z "${YTDLP_JS_RUNTIME:-}" ]; then
    if command -v node >/dev/null 2>&1; then
        export YTDLP_JS_RUNTIME="node:$(command -v node)"
    elif [ -x "$HOME/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node" ]; then
        export YTDLP_JS_RUNTIME="node:$HOME/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node"
    fi
fi

if [ -z "${YTDLP_COOKIES:-}" ] && [ -z "${YTDLP_COOKIES_FROM_BROWSER:-}" ]; then
    export YTDLP_COOKIES_FROM_BROWSER="chrome"
fi

pids=()

cleanup() {
    status=$?
    trap - EXIT INT TERM
    for pid in "${pids[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    # Do not wait forever for a Python process whose non-daemon dependency
    # thread failed to terminate. Escalate only the exact children we started.
    for _ in {1..50}; do
        alive=0
        for pid in "${pids[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                alive=1
            fi
        done
        [ "$alive" = "0" ] && break
        sleep 0.1
    done
    for pid in "${pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -KILL "$pid" 2>/dev/null || true
        fi
        wait "$pid" 2>/dev/null || true
    done
    if [ "$(sed -n '1p' "$LOCK_DIR/pid" 2>/dev/null || true)" = "$$" ]; then
        rm -f "$LOCK_DIR/pid"
        rmdir "$LOCK_DIR" 2>/dev/null || true
    fi
    exit "$status"
}

trap cleanup EXIT
trap 'exit 130' INT TERM

echo "=== Starting FrontierTechSN ==="
echo "Run log: $LOG_FILE"
echo "Latest log: $LATEST_LOG"
if [ -n "${YTDLP_JS_RUNTIME:-}" ]; then
    echo "yt-dlp JS runtime: ${YTDLP_JS_RUNTIME%%:*}"
fi
if [ -n "${YTDLP_COOKIES:-}" ]; then
    echo "yt-dlp cookies: file"
elif [ -n "${YTDLP_COOKIES_FROM_BROWSER:-}" ]; then
    echo "yt-dlp cookies: browser (${YTDLP_COOKIES_FROM_BROWSER%%:*})"
fi

# Build the frontend so the backend serves the current UI. Set
# SKIP_FRONTEND_BUILD=1 to reuse the existing frontend/dist (faster restarts).
if [ "${SKIP_FRONTEND_BUILD:-0}" = "1" ]; then
    echo "Skipping frontend build (SKIP_FRONTEND_BUILD=1); reusing frontend/dist"
else
    echo "Building frontend (npm run build) ..."
    (
        cd frontend
        if [ ! -d node_modules ]; then
            echo "Installing frontend dependencies (npm install) ..."
            npm install
        fi
        npm run build
    )
    echo "Frontend build complete -> frontend/dist"
fi

# Single process on purpose: the app runs an in-process background worker and
# keeps in-memory log subscriptions, so it must NOT be scaled with --workers,
# and --reload is a dev-only feature we don't want in a production service.
echo "Starting server on http://localhost:$PORT (API + frontend) ..."
(
    source .venv/bin/activate
    exec uvicorn backend.main:app --host "$HOST" --port "$PORT"
) &
pids+=("$!")

wait "${pids[@]}"

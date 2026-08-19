#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OPENCLI_BIN="$PROJECT_ROOT/tools/opencli/node_modules/.bin/opencli"
RATE_LIMITER="$PROJECT_ROOT/backend/pipeline/opencli_rate_limit.py"

if [ ! -x "$OPENCLI_BIN" ]; then
  echo "Project-local OpenCLI is not installed." >&2
  echo "Run: npm install --prefix \"$PROJECT_ROOT/tools/opencli\"" >&2
  exit 127
fi

# Gemini and ChatGPT are browser-backed and enforce burst limits. Gate every
# project-wrapper invocation, including calls made outside the backend. Backend
# calls reserve their slot before their command timeout starts and mark the
# child so this wrapper does not reserve the same request twice.
case "${1:-}" in
  chatgpt|gemini)
    if [ "${OPENCLI_WEB_REQUEST_SLOT_RESERVED:-0}" != "1" ]; then
      if [ -x "$PROJECT_ROOT/.venv/bin/python" ]; then
        "$PROJECT_ROOT/.venv/bin/python" "$RATE_LIMITER" "$1"
      elif command -v python3 >/dev/null 2>&1; then
        python3 "$RATE_LIMITER" "$1"
      else
        echo "Python 3 is required for OpenCLI Gemini/ChatGPT request pacing." >&2
        exit 127
      fi
    fi
    ;;
esac

exec "$OPENCLI_BIN" "$@"

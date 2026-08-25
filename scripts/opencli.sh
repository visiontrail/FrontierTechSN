#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OPENCLI_BIN="$PROJECT_ROOT/tools/opencli/node_modules/.bin/opencli"
RATE_LIMITER="$PROJECT_ROOT/backend/pipeline/opencli_rate_limit.py"

# OpenCLI otherwise gives every checkout the same persistent `site:<provider>`
# browser lease. Namespace this project's tabs so another local media pipeline
# cannot navigate an in-flight FrontierTechSN conversation away.
export OPENCLI_SITE_SESSION_NAMESPACE="${OPENCLI_SITE_SESSION_NAMESPACE:-frontiertechsn}"

if [ ! -x "$OPENCLI_BIN" ]; then
  echo "Project-local OpenCLI is not installed." >&2
  echo "Run: npm install --prefix \"$PROJECT_ROOT/tools/opencli\"" >&2
  exit 127
fi

# Backend calls export the live Admin selection; agent/direct wrapper calls can
# opt in through the same environment variable. The default bridge path never
# starts or probes another browser. Isolated mode must return an exact dedicated
# profile or this command fails before OpenCLI can auto-select operator Chrome.
if [ "${OPENCLI_BROWSER_RUNTIME:-bridge}" = "isolated-headless" ] \
  && [ "${OPENCLI_ISOLATED_RUNTIME_READY:-0}" != "1" ]; then
  if [ -x "$PROJECT_ROOT/.venv/bin/python" ]; then
    RUNTIME_PYTHON="$PROJECT_ROOT/.venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    RUNTIME_PYTHON="$(command -v python3)"
  else
    echo "Python 3 is required for the isolated OpenCLI browser." >&2
    exit 127
  fi
  ISOLATED_PROFILE="$(
    cd "$PROJECT_ROOT"
    "$RUNTIME_PYTHON" -m backend.pipeline.opencli_browser_runtime prepare
  )"
  if [ -z "$ISOLATED_PROFILE" ]; then
    echo "Isolated OpenCLI profile is empty; refusing browser auto-selection." >&2
    exit 78
  fi
  export OPENCLI_PROFILE="$ISOLATED_PROFILE"
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

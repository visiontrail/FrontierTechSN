#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [ -x "$PROJECT_ROOT/.venv/bin/python" ]; then
  PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
else
  echo "Python 3 is required to manage the isolated OpenCLI browser." >&2
  exit 127
fi

cd "$PROJECT_ROOT"
exec "$PYTHON_BIN" -m backend.pipeline.opencli_browser_runtime "$@"

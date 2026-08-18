#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OPENCLI_BIN="$PROJECT_ROOT/tools/opencli/node_modules/.bin/opencli"

if [ ! -x "$OPENCLI_BIN" ]; then
  echo "Project-local OpenCLI is not installed." >&2
  echo "Run: npm install --prefix \"$PROJECT_ROOT/tools/opencli\"" >&2
  exit 127
fi

exec "$OPENCLI_BIN" "$@"

#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONTENT_PROMPT="${ACCOUNT_OPS_CONTENT_PROMPT:-}"

if [ -z "$CONTENT_PROMPT" ]; then
  echo "ACCOUNT_OPS_CONTENT_PROMPT is required" >&2
  exit 64
fi

exec "$PROJECT_ROOT/scripts/opencli.sh" chatgpt ask "$CONTENT_PROMPT" \
  --new \
  --timeout 120 \
  --window background \
  --site-session persistent \
  --keep-tab false \
  -f json

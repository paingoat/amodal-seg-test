#!/usr/bin/env bash
# Stop whatever process is listening on the LISA Gradio port (default 7860).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/paths.env"

PORT="${LISA_SERVER_PORT:-7860}"
echo "[stop_lisa] Looking for listeners on port $PORT ..."

PIDS=""
if command -v lsof >/dev/null 2>&1; then
  PIDS="$(lsof -t -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)"
elif command -v fuser >/dev/null 2>&1; then
  PIDS="$(fuser "${PORT}/tcp" 2>/dev/null || true)"
elif command -v ss >/dev/null 2>&1; then
  PIDS="$(ss -lptn "sport = :$PORT" 2>/dev/null | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' | sort -u || true)"
fi

if [ -z "${PIDS// /}" ]; then
  echo "[stop_lisa] No process found on port $PORT."
  exit 0
fi

echo "[stop_lisa] Killing PIDs: $PIDS"
# shellcheck disable=SC2086
kill $PIDS 2>/dev/null || true
sleep 2
# shellcheck disable=SC2086
kill -9 $PIDS 2>/dev/null || true
echo "[stop_lisa] Done. VRAM from LISA should now be free."

#!/usr/bin/env bash
# Starts both services needed for the live demo:
#   1. apps/web  -> Next.js dashboard + /live demo page (http://localhost:3000)
#   2. apps/agent -> the LiveKit worker (registers with LiveKit Cloud, waits for a room)
#
# Usage:  ./start_demo.sh
# Stop:   ./stop_demo.sh   (or Ctrl-C won't work since these are backgrounded — use stop_demo.sh)

set -euo pipefail
cd "$(dirname "$0")"

echo "== Starting web app (Next.js) on :3000 =="
(cd apps/web && nohup npm run dev > /tmp/demo_web.log 2>&1 &)

echo "== Starting agent worker =="
(cd apps/agent && nohup .venv/bin/python3 worker.py dev > /tmp/demo_worker.log 2>&1 &)

echo ""
echo "Waiting for both to come up..."
sleep 5

# Don't assume :3000 — if a stale next-server from a previous run (or another
# project) is still holding it, Next.js silently falls back to :3001+, and a
# hardcoded port here would report a false failure while the app is actually
# fine on a different port. Read the port Next.js itself reports it bound to.
WEB_PORT=$(grep -oE "Local:\s+http://localhost:[0-9]+" /tmp/demo_web.log | grep -oE "[0-9]+$" | tail -1)
WEB_PORT=${WEB_PORT:-3000}
WEB_URL="http://localhost:${WEB_PORT}/live"
WEB_OK=$(curl -s -o /dev/null -w "%{http_code}" "$WEB_URL" || echo "000")
WORKER_OK=$(grep -c "registered worker" /tmp/demo_worker.log 2>/dev/null || echo "0")

echo ""
if [ "$WEB_PORT" != "3000" ]; then
  echo "NOTE: port 3000 was already in use (stale process from a previous run?) — Next.js used :$WEB_PORT instead."
  echo "      Run ./stop_demo.sh first if you want a clean :3000, then start_demo.sh again."
fi
echo "Web app:    $WEB_URL   (HTTP $WEB_OK)"
if [ "$WORKER_OK" -ge 1 ]; then
  echo "Worker:     registered with LiveKit Cloud (see /tmp/demo_worker.log)"
else
  echo "Worker:     NOT confirmed yet — check /tmp/demo_worker.log"
fi
echo ""
echo "Logs: tail -f /tmp/demo_web.log   /tmp/demo_worker.log"
echo "Open $WEB_URL , pick 'Backchannel ON', connect, and talk."

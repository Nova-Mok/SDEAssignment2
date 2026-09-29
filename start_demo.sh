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

WEB_OK=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:3000/live || echo "000")
WORKER_OK=$(grep -c "registered worker" /tmp/demo_worker.log 2>/dev/null || echo "0")

echo ""
echo "Web app:    http://localhost:3000/live   (HTTP $WEB_OK)"
if [ "$WORKER_OK" -ge 1 ]; then
  echo "Worker:     registered with LiveKit Cloud (see /tmp/demo_worker.log)"
else
  echo "Worker:     NOT confirmed yet — check /tmp/demo_worker.log"
fi
echo ""
echo "Logs: tail -f /tmp/demo_web.log   /tmp/demo_worker.log"
echo "Open http://localhost:3000/live , pick 'Backchannel ON', connect, and talk."

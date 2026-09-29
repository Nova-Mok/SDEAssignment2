#!/usr/bin/env bash
# Stops both demo processes started by start_demo.sh.
echo "Stopping web app and worker..."
pkill -f "next dev" 2>/dev/null && echo "  stopped web app" || echo "  web app was not running"
pkill -f "worker.py dev" 2>/dev/null && echo "  stopped worker" || echo "  worker was not running"

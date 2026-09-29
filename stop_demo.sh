#!/usr/bin/env bash
# Stops both demo processes started by start_demo.sh.
#
# Next.js 15's dev server spawns a SEPARATE "next-server" child process
# distinct from the "next dev" wrapper — killing only "next dev" leaves
# "next-server" holding the port, so the next start_demo.sh run silently
# falls back to :3001 instead of :3000. Both patterns are killed here.
echo "Stopping web app and worker..."
pkill -f "next dev" 2>/dev/null && echo "  stopped web app (next dev)" || echo "  next dev was not running"
pkill -f "next-server" 2>/dev/null && echo "  stopped web app (next-server)" || echo "  next-server was not running"
pkill -f "worker.py dev" 2>/dev/null && echo "  stopped worker" || echo "  worker was not running"

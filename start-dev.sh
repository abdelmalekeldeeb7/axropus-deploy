#!/bin/bash
set -euo pipefail

# Start Axropus development environment

echo "Starting Axropus backend..."
cd /home/korith/axropus-cloud
source .venv/bin/activate
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

echo "Starting Axropus frontend..."
cd /home/korith/axropus-cloud/frontend
npm run dev &
FRONTEND_PID=$!

echo ""
echo "═══════════════════════════════════"
echo "  AXROPUS DEV ENVIRONMENT RUNNING"
echo "  Frontend: http://localhost:5173"
echo "  Backend:  http://localhost:8000"
echo "  API Docs: http://localhost:8000/docs"
echo "═══════════════════════════════════"
echo ""

cleanup() {
  kill "$BACKEND_PID" "$FRONTEND_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

wait "$BACKEND_PID" "$FRONTEND_PID"

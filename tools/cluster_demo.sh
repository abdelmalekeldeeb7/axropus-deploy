#!/usr/bin/env bash
set -euo pipefail

export KORITH_QUEUE_BACKEND=sqlite
export KORITH_QUEUE_DB=./platform_data/queue.sqlite
export KORITH_PLATFORM_DB=./platform_data/ledger.sqlite
export KORITH_PLATFORM_ARTIFACTS=./platform_data/artifacts

python -m platform.runtime.router_service --host 127.0.0.1 --port 8000 &
ROUTER_PID=$!
python -m platform.runtime.worker_service --worker-id worker-0 --gpu-id 0 --host 127.0.0.1 --port 9000 &
WORKER_PID=$!

sleep 1

J1=$(python platform/korith_platform.py submit demo/jobs/ticket_triage.json --url http://127.0.0.1:8000)
J2=$(python platform/korith_platform.py submit demo/jobs/ticket_triage.json --url http://127.0.0.1:8000)
J3=$(python platform/korith_platform.py submit demo/jobs/ticket_triage.json --url http://127.0.0.1:8000)

sleep 3

python tools/cluster_benchmark.py --job-ids "$J1,$J2,$J3" --artifacts-dir ./platform_data/artifacts

kill $WORKER_PID $ROUTER_PID || true

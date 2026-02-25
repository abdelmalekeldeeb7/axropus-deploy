#!/usr/bin/env bash
set -euo pipefail

URL="${KORITH_URL:-http://127.0.0.1:8000}"
JOBSPEC="${1:-demo/jobs/ticket_triage.json}"
RUNS="${2:-3}"
TIMEOUT_S="${3:-60}"
API_KEY="${KORITH_API_KEY:-}"

if [[ -z "${API_KEY}" ]]; then
  echo "KORITH_API_KEY is required"
  exit 1
fi

echo "== Phase6 Golden Benchmark =="
echo "jobspec=${JOBSPEC} runs=${RUNS} timeout=${TIMEOUT_S}"

echo "-- cold/warm replay baseline --"
KORITH_URL="${URL}" KORITH_API_KEY="${API_KEY}" ./scripts/amf_jobs_check.sh "${JOBSPEC}" "${RUNS}" "${TIMEOUT_S}"

echo "-- warm replay + speculative lane --"
KORITH_URL="${URL}" KORITH_API_KEY="${API_KEY}" ./scripts/spec_jobs_check.sh "${JOBSPEC}" "${RUNS}" "${TIMEOUT_S}" 6

echo "-- kernel status --"
python3 platform/korith_platform.py kernels status --url "${URL}" --api-key "${API_KEY}"


#!/usr/bin/env bash
set -euo pipefail

# Thin wrapper around scripts/soak_chaos.py with production-like defaults.
# Override any value by exporting env vars before invoking this script.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

DURATION_S="${DURATION_S:-86400}"
TARGET_REQUESTS="${TARGET_REQUESTS:-0}"
RPS="${RPS:-0.25}"
CONCURRENCY="${CONCURRENCY:-2}"
INITIAL_WORKERS="${INITIAL_WORKERS:-1}"
MAX_WORKERS="${MAX_WORKERS:-2}"
CHAOS_MIN_INTERVAL_S="${CHAOS_MIN_INTERVAL_S:-7200}"
CHAOS_MAX_INTERVAL_S="${CHAOS_MAX_INTERVAL_S:-14400}"
SYSTEM_SAMPLE_S="${SYSTEM_SAMPLE_S:-10}"
REPORT_BUCKET_S="${REPORT_BUCKET_S:-900}"

BACKEND_ID="${BACKEND_ID:-${KORITH_SOAK_BACKEND_ID:-korith_local}}"
MODEL_ID="${MODEL_ID:-${KORITH_SOAK_MODEL_ID:-soak-model}}"
MODEL_PATH="${MODEL_PATH:-${KORITH_SOAK_MODEL_PATH:-}}"
MODEL_ENDPOINT="${MODEL_ENDPOINT:-${KORITH_SOAK_MODEL_ENDPOINT:-}}"
N_CTX="${N_CTX:-8192}"
N_BATCH="${N_BATCH:-512}"
MAX_TOKENS="${MAX_TOKENS:-128}"
PROMPT_TOKENS_MAX="${PROMPT_TOKENS_MAX:-4096}"

ORGS="${ORGS:-tenant_a,tenant_b,tenant_c}"
USERS_PER_ORG="${USERS_PER_ORG:-3}"
DOCUMENTS="${DOCUMENTS:-8}"
SHORT_CONTEXT_RATE="${SHORT_CONTEXT_RATE:-0.30}"
LONG_CONTEXT_RATE="${LONG_CONTEXT_RATE:-0.20}"
MUTATION_RATE="${MUTATION_RATE:-0.25}"
PARTIAL_OVERLAP_RATE="${PARTIAL_OVERLAP_RATE:-0.55}"

START_STACK="${START_STACK:-1}"
CHAOS_ENABLED="${CHAOS_ENABLED:-1}"
AUTOSCALE_CHAOS="${AUTOSCALE_CHAOS:-1}"
DISK_PRESSURE_BYTES="${DISK_PRESSURE_BYTES:-0}"
ALLOW_SPEC="${ALLOW_SPEC:-0}"

python3 ./scripts/soak_chaos.py \
  --duration-s "${DURATION_S}" \
  --target-requests "${TARGET_REQUESTS}" \
  --start-stack "${START_STACK}" \
  --backend-id "${BACKEND_ID}" \
  --model-id "${MODEL_ID}" \
  --model-path "${MODEL_PATH}" \
  --model-endpoint "${MODEL_ENDPOINT}" \
  --n-ctx "${N_CTX}" \
  --n-batch "${N_BATCH}" \
  --max-tokens "${MAX_TOKENS}" \
  --allow-spec "${ALLOW_SPEC}" \
  --rps "${RPS}" \
  --concurrency "${CONCURRENCY}" \
  --initial-workers "${INITIAL_WORKERS}" \
  --max-workers "${MAX_WORKERS}" \
  --chaos-enabled "${CHAOS_ENABLED}" \
  --autoscale-chaos "${AUTOSCALE_CHAOS}" \
  --chaos-min-interval-s "${CHAOS_MIN_INTERVAL_S}" \
  --chaos-max-interval-s "${CHAOS_MAX_INTERVAL_S}" \
  --disk-pressure-bytes "${DISK_PRESSURE_BYTES}" \
  --system-sample-s "${SYSTEM_SAMPLE_S}" \
  --report-bucket-s "${REPORT_BUCKET_S}" \
  --orgs "${ORGS}" \
  --users-per-org "${USERS_PER_ORG}" \
  --documents "${DOCUMENTS}" \
  --prompt-tokens-max "${PROMPT_TOKENS_MAX}" \
  --short-context-rate "${SHORT_CONTEXT_RATE}" \
  --long-context-rate "${LONG_CONTEXT_RATE}" \
  --mutation-rate "${MUTATION_RATE}" \
  --partial-overlap-rate "${PARTIAL_OVERLAP_RATE}" \
  "$@"

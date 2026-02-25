#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

URL="${KORITH_URL:-http://127.0.0.1:8000}"
JOBSPEC_TEMPLATE="${1:-demo/jobs/ticket_triage.json}"
RUNS="${2:-20}"
TIMEOUT_S="${3:-120}"
TOKENS_CSV="${4:-64,256,512}"
SALT="${KORITH_API_KEY_SALT:-korith_local_dev_salt_2026}"
ENGINE_LIB="${KORITH_ENGINE_LIB_PATH:-${ROOT_DIR}/build/engine-cuda/libkorith_engine.so}"

if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required"
  exit 1
fi

if [[ ! -f "${JOBSPEC_TEMPLATE}" ]]; then
  echo "jobspec template not found: ${JOBSPEC_TEMPLATE}"
  exit 1
fi

IFS=',' read -r -a TOKENS <<< "${TOKENS_CSV}"

reset_runtime_state() {
  python3 - <<'PY'
import sqlite3, time, os
if os.path.exists("platform_data/ledger.sqlite"):
    db = sqlite3.connect("platform_data/ledger.sqlite")
    db.execute("update spec_governance set spec_disabled=0, reason=NULL, cooldown_until=0, bad_accept_streak=0")
    db.commit()
    db.close()
if os.path.exists("platform_data/queue.sqlite"):
    q = sqlite3.connect("platform_data/queue.sqlite")
    q.execute("update queue set status='QUEUED', worker_id=NULL, available_at=? where status='INFLIGHT'", (time.time(),))
    q.commit()
    q.close()
PY
}

stop_stack() {
  pkill -f "platform/korith_platform.py worker" || true
  pkill -f "platform/korith_platform.py router" || true
}

start_stack() {
  local name="$1"
  local accel="$2"
  local kernels="$3"

  stop_stack
  mkdir -p platform_data/logs

  nohup env \
    KORITH_API_KEY_SALT="${SALT}" \
    KORITH_SPEC_ENABLED="0" \
    KORITH_SPEC_CACHE_ONLY="0" \
    KORITH_ACCEL_ENABLED="${accel}" \
    KORITH_KERNELS="${kernels}" \
    KORITH_KERNEL_BACKEND="cuda" \
    KORITH_ENGINE_LIB_PATH="${ENGINE_LIB}" \
    python3 platform/korith_platform.py worker --worker-id worker-gpu0 --gpu-id 0 --host 127.0.0.1 --port 9000 \
    > "platform_data/logs/worker.${name}.log" 2>&1 &

  nohup env \
    KORITH_API_KEY_SALT="${SALT}" \
    KORITH_SPEC_ENABLED="0" \
    KORITH_SPEC_CACHE_ONLY="0" \
    KORITH_ACCEL_ENABLED="${accel}" \
    KORITH_KERNELS="${kernels}" \
    KORITH_KERNEL_BACKEND="cuda" \
    KORITH_ENGINE_LIB_PATH="${ENGINE_LIB}" \
    python3 platform/korith_platform.py router --host 127.0.0.1 --port 8000 \
    > "platform_data/logs/router.${name}.log" 2>&1 &

  sleep 1
  curl -fsS "${URL}/health" >/dev/null
}

create_api_key() {
  KORITH_API_KEY_SALT="${SALT}" python3 platform/korith_platform.py keys create --org pilot_a > /tmp/korith_ab_key.json
  if [[ ! -s /tmp/korith_ab_key.json ]]; then
    echo "failed to create API key"
    return 1
  fi
  python3 - <<'PY'
import json
print(json.load(open("/tmp/korith_ab_key.json"))["api_key"])
PY
}

run_case() {
  local name="$1"
  local accel="$2"
  local kernels="$3"
  local max_tokens="$4"

  start_stack "${name}" "${accel}" "${kernels}"
  local api_key
  api_key="$(create_api_key)"

  local jobspec="/tmp/korith_ab_${max_tokens}.json"
  jq ".deterministic_cfg.max_tokens=${max_tokens} | .policy.allow_spec=false" "${JOBSPEC_TEMPLATE}" > "${jobspec}"

  local raw="/tmp/korith_ab_raw_${name}_${max_tokens}.log"
  local json="/tmp/korith_ab_${name}_${max_tokens}.json"
  python3 platform/korith_platform.py bench spec \
    --jobspec "${jobspec}" \
    --runs "${RUNS}" \
    --timeout-s "${TIMEOUT_S}" \
    --url "${URL}" \
    --api-key "${api_key}" > "${raw}"
  sed -n '/^{/,$p' "${raw}" > "${json}"
  if [[ ! -s "${json}" ]]; then
    echo "bench output was not valid JSON for ${name} max_tokens=${max_tokens}"
    tail -n 80 "${raw}" || true
    return 1
  fi

  python3 - <<PY
import json, math
j=json.load(open("${json}"))
rows=j["rows"]
total=[float(r["metrics"]["perf"]["total_ms"]) for r in rows]
decode=[float(r["metrics"]["perf"]["decode_ms"]) for r in rows]
tps=[float(r["metrics"]["perf"]["avg_tps"]) for r in rows]
total_s=sorted(total)
decode_s=sorted(decode)
def pct(vals,p):
    i=max(0,min(len(vals)-1,math.ceil(len(vals)*p)-1))
    return vals[i]
out={
  "runs": len(rows),
  "total_avg": sum(total)/len(total),
  "total_p50": total_s[len(total_s)//2],
  "total_p95": pct(total_s,0.95),
  "decode_avg": sum(decode)/len(decode),
  "decode_p50": decode_s[len(decode_s)//2],
  "decode_p95": pct(decode_s,0.95),
  "tps_avg": sum(tps)/len(tps),
}
print(json.dumps(out))
PY
}

reset_runtime_state

echo "== Phase6 A/B Matrix =="
echo "jobspec=${JOBSPEC_TEMPLATE} runs=${RUNS} timeout_s=${TIMEOUT_S} tokens=${TOKENS_CSV}"

for tok in "${TOKENS[@]}"; do
  echo
  echo "-- max_tokens=${tok} --"

  base_json="$(run_case replay_baseline 0 0 "${tok}")"
  kern_json="$(run_case replay_kernel 1 1 "${tok}")"

  python3 - <<PY
import json
b=json.loads('''${base_json}''')
k=json.loads('''${kern_json}''')
def pct_delta(base,new):
    if base == 0:
        return 0.0
    return (base-new)/base*100.0
print("baseline:", json.dumps(b))
print("kernel  :", json.dumps(k))
print("delta   :", json.dumps({
  "total_avg_pct": round(pct_delta(b["total_avg"], k["total_avg"]), 3),
  "total_p50_pct": round(pct_delta(b["total_p50"], k["total_p50"]), 3),
  "total_p95_pct": round(pct_delta(b["total_p95"], k["total_p95"]), 3),
  "decode_avg_pct": round(pct_delta(b["decode_avg"], k["decode_avg"]), 3),
  "decode_p50_pct": round(pct_delta(b["decode_p50"], k["decode_p50"]), 3),
  "decode_p95_pct": round(pct_delta(b["decode_p95"], k["decode_p95"]), 3),
  "tps_avg_pct": round(((k["tps_avg"]-b["tps_avg"])/b["tps_avg"]*100.0) if b["tps_avg"] else 0.0, 3),
}))
PY
done

stop_stack
echo
echo "done"

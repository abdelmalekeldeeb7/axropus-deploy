#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

TARGET_REPO="${TARGET_REPO:-meta-llama/Llama-3.1-8B-Instruct}"
DRAFT_REPO="${DRAFT_REPO:-meta-llama/Llama-3.2-1B-Instruct}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${TARGET_REPO}}"

MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-${ROOT_DIR}/models/safetensors}"
TARGET_MODEL_DIR="${TARGET_MODEL_DIR:-${MODEL_CACHE_DIR}/Meta-Llama-3.1-8B-Instruct}"
DRAFT_MODEL_DIR="${DRAFT_MODEL_DIR:-${MODEL_CACHE_DIR}/Llama-3.2-1B-Instruct}"

VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-18001}"
ROUTER_HOST="${ROUTER_HOST:-127.0.0.1}"
ROUTER_PORT="${ROUTER_PORT:-18000}"
WORKER_HOST="${WORKER_HOST:-127.0.0.1}"
WORKER_PORT="${WORKER_PORT:-19000}"
GPU_ID="${GPU_ID:-0}"

RUNS_PER_SCENARIO="${RUNS_PER_SCENARIO:-3}"
WARMUP_RUNS="${WARMUP_RUNS:-1}"
BENCH_TIMEOUT_S="${BENCH_TIMEOUT_S:-1800}"

PROMPT_TOKENS="${PROMPT_TOKENS:-4000}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-128}"
N_CTX="${N_CTX:-16384}"
N_BATCH="${N_BATCH:-512}"
SEED="${SEED:-1234}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
SPEC_K="${SPEC_K:-5}"

API_SALT="${KORITH_API_KEY_SALT:-korith_lambda_benchmark_salt_2026}"
BENCH_ORG="${BENCH_ORG:-lambda_h100_bench}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RESULTS_DIR="${RESULTS_DIR:-${ROOT_DIR}/platform_data/lambda_bench/${RUN_ID}}"
LOG_DIR="${RESULTS_DIR}/logs"
RUNTIME_DIR="${RESULTS_DIR}/runtime"
PROMPT_PATH="${RESULTS_DIR}/prompt_4000.txt"
RESULTS_JSON="${RESULTS_DIR}/benchmark_results.json"

ROUTER_URL="http://${ROUTER_HOST}:${ROUTER_PORT}"
VLLM_URL="http://${VLLM_HOST}:${VLLM_PORT}"

VLLM_PID=""
ROUTER_PID=""
WORKER_PID=""
API_KEY=""
declare -a STACK_ENV=()
declare -a SUMMARY_FILES=()

log() {
  echo "[$(date +%H:%M:%S)] $*"
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

model_ready() {
  local model_dir="$1"
  [[ -d "${model_dir}" ]] || return 1
  [[ -f "${model_dir}/config.json" ]] || return 1
  find "${model_dir}" -maxdepth 1 -type f -name "*.safetensors" | grep -q .
}

ensure_hf_hub() {
  if python3 - <<'PY' >/dev/null 2>&1
import importlib.util
import sys
sys.exit(0 if importlib.util.find_spec("huggingface_hub") else 1)
PY
  then
    return
  fi
  log "installing huggingface_hub (user install)"
  python3 -m pip install --user --upgrade huggingface_hub >/dev/null
}

download_repo_snapshot() {
  local repo_id="$1"
  local out_dir="$2"

  mkdir -p "${out_dir}"
  python3 - "${repo_id}" "${out_dir}" <<'PY'
import os
import sys
from huggingface_hub import snapshot_download

repo_id = sys.argv[1]
out_dir = sys.argv[2]
token = os.environ.get("HF_TOKEN", "").strip()
if not token:
    raise SystemExit("HF_TOKEN is required to download models from Hugging Face")

allow_patterns = [
    "*.safetensors",
    "*.safetensors.index.json",
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
]

snapshot_download(
    repo_id=repo_id,
    local_dir=out_dir,
    token=token,
    resume_download=True,
    allow_patterns=allow_patterns,
)
PY
}

ensure_models() {
  mkdir -p "${MODEL_CACHE_DIR}"
  local need_download=0
  if ! model_ready "${TARGET_MODEL_DIR}"; then
    need_download=1
  fi
  if ! model_ready "${DRAFT_MODEL_DIR}"; then
    need_download=1
  fi
  if [[ "${need_download}" -eq 0 ]]; then
    log "safetensors models already present"
    return
  fi

  [[ -n "${HF_TOKEN:-}" ]] || die "HF_TOKEN is required when models are not already downloaded"
  ensure_hf_hub

  if ! model_ready "${TARGET_MODEL_DIR}"; then
    log "downloading target model ${TARGET_REPO} -> ${TARGET_MODEL_DIR}"
    download_repo_snapshot "${TARGET_REPO}" "${TARGET_MODEL_DIR}"
  fi
  if ! model_ready "${DRAFT_MODEL_DIR}"; then
    log "downloading draft model ${DRAFT_REPO} -> ${DRAFT_MODEL_DIR}"
    download_repo_snapshot "${DRAFT_REPO}" "${DRAFT_MODEL_DIR}"
  fi
}

build_prompt_file() {
  mkdir -p "${RESULTS_DIR}"
  python3 - "${PROMPT_PATH}" "${PROMPT_TOKENS}" <<'PY'
import sys
from pathlib import Path

out_path = Path(sys.argv[1])
count = int(sys.argv[2])
tokens = ["a"] * count
prompt = " ".join(tokens)
out_path.write_text(prompt, encoding="utf-8")
if len(prompt.split()) != count:
    raise SystemExit("prompt token count mismatch")
PY
}

wait_http_ok() {
  local url="$1"
  local timeout_s="$2"
  local label="$3"
  local start_ts
  start_ts="$(date +%s)"
  while true; do
    if curl -fsS "${url}" >/dev/null 2>&1; then
      return 0
    fi
    if (( $(date +%s) - start_ts > timeout_s )); then
      die "timeout waiting for ${label} at ${url}"
    fi
    sleep 1
  done
}

stop_pid_if_running() {
  local pid="$1"
  if [[ -z "${pid}" ]]; then
    return
  fi
  if kill -0 "${pid}" >/dev/null 2>&1; then
    kill "${pid}" >/dev/null 2>&1 || true
    wait "${pid}" >/dev/null 2>&1 || true
  fi
}

stop_stack() {
  stop_pid_if_running "${ROUTER_PID}"
  stop_pid_if_running "${WORKER_PID}"
  stop_pid_if_running "${VLLM_PID}"
  ROUTER_PID=""
  WORKER_PID=""
  VLLM_PID=""
}

cleanup() {
  stop_stack
}

trap cleanup EXIT

start_vllm() {
  local scenario_id="$1"
  local prefix_enabled="$2"
  local fp8_enabled="$3"
  local native_spec_enabled="$4"
  local full_budget_on_hit="$5"

  local log_path="${LOG_DIR}/vllm.${scenario_id}.log"
  local prefix_flag="--no-enable-prefix-caching"
  local kv_dtype="auto"
  local spec_json=""

  if [[ "${prefix_enabled}" == "1" ]]; then
    prefix_flag="--enable-prefix-caching"
  fi
  if [[ "${fp8_enabled}" == "1" ]]; then
    kv_dtype="fp8"
  fi
  if [[ "${native_spec_enabled}" == "1" ]]; then
    spec_json="$(python3 - "${DRAFT_MODEL_DIR}" "${SPEC_K}" <<'PY'
import json
import sys

draft_path = sys.argv[1]
spec_k = int(sys.argv[2])
print(json.dumps({
    "method": "draft_model",
    "model": draft_path,
    "num_speculative_tokens": spec_k,
    "draft_tensor_parallel_size": 1,
}, separators=(",", ":")))
PY
)"
  fi

  local -a cmd=(
    vllm serve "${TARGET_MODEL_DIR}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --host "${VLLM_HOST}"
    --port "${VLLM_PORT}"
    --dtype bfloat16
    "${prefix_flag}"
    --enable-prompt-tokens-details
    --kv-cache-dtype "${kv_dtype}"
    --max-model-len "${MAX_MODEL_LEN}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --scheduling-policy priority
    --scheduler-cls korith_vllm_ext.decode_scheduler.KorithDecodeScheduler
  )
  if [[ -n "${spec_json}" ]]; then
    cmd+=(--speculative-config "${spec_json}")
  fi

  log "starting vLLM scenario=${scenario_id} prefix=${prefix_enabled} fp8=${fp8_enabled} native_spec=${native_spec_enabled} full_budget_on_hit=${full_budget_on_hit}"
  env \
    PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}" \
    KORITH_VLLM_DECODE_FULL_BUDGET_ON_HIT="${full_budget_on_hit}" \
    KORITH_VLLM_DECODE_FIRST="${KORITH_VLLM_DECODE_FIRST:-1}" \
    KORITH_VLLM_ADAPTIVE_BUDGET="${KORITH_VLLM_ADAPTIVE_BUDGET:-1}" \
    KORITH_VLLM_PRIORITY_SCHED="${KORITH_VLLM_PRIORITY_SCHED:-1}" \
    "${cmd[@]}" >"${log_path}" 2>&1 &
  VLLM_PID="$!"

  wait_http_ok "${VLLM_URL}/health" 1200 "vLLM"
}

start_platform() {
  local scenario_id="$1"
  local mf_enabled="$2"

  local scenario_dir="${RUNTIME_DIR}/${scenario_id}"
  mkdir -p "${scenario_dir}"

  STACK_ENV=(
    "KORITH_API_KEY_SALT=${API_SALT}"
    "KORITH_PLATFORM_DB=${scenario_dir}/ledger.sqlite"
    "KORITH_QUEUE_DB=${scenario_dir}/queue.sqlite"
    "KORITH_REGISTRY_DB=${scenario_dir}/registry.sqlite"
    "KORITH_RESTORE_DB=${scenario_dir}/restore.sqlite"
    "KORITH_NODE_REGISTRY_DB=${scenario_dir}/node_registry.sqlite"
    "KORITH_RATE_LIMIT_DB=${scenario_dir}/rate_limit.sqlite"
    "KORITH_PLATFORM_ARTIFACTS=${scenario_dir}/artifacts"
    "KORITH_QUEUE_BACKEND=sqlite"
    "KORITH_VLLM_ENDPOINT=${VLLM_URL}"
    "KORITH_VLLM_MODEL_ID=${SERVED_MODEL_NAME}"
    "KORITH_VLLM_MF_ENABLED=${mf_enabled}"
    "KORITH_VLLM_ENABLE_SPEC_DECODE=0"
    "KORITH_VLLM_RUNTIME_CONTRACT_ENABLED=1"
    "KORITH_VLLM_PRIORITY_SCHED=1"
  )

  local worker_log="${LOG_DIR}/worker.${scenario_id}.log"
  local router_log="${LOG_DIR}/router.${scenario_id}.log"

  log "starting worker scenario=${scenario_id}"
  env "${STACK_ENV[@]}" \
    python3 platform/korith_platform.py worker \
      --worker-id "worker-${scenario_id}" \
      --gpu-id "${GPU_ID}" \
      --host "${WORKER_HOST}" \
      --port "${WORKER_PORT}" >"${worker_log}" 2>&1 &
  WORKER_PID="$!"

  log "starting router scenario=${scenario_id}"
  env "${STACK_ENV[@]}" \
    python3 platform/korith_platform.py router \
      --host "${ROUTER_HOST}" \
      --port "${ROUTER_PORT}" >"${router_log}" 2>&1 &
  ROUTER_PID="$!"

  wait_http_ok "${ROUTER_URL}/health" 120 "router health"
  wait_http_ok "${ROUTER_URL}/ready" 120 "router ready"
  wait_http_ok "http://${WORKER_HOST}:${WORKER_PORT}/health" 120 "worker health"
}

create_api_key() {
  local scenario_id="$1"
  local key_json="${RESULTS_DIR}/api_key.${scenario_id}.json"

  env "${STACK_ENV[@]}" \
    python3 platform/korith_platform.py keys create --org "${BENCH_ORG}" >"${key_json}"
  API_KEY="$(python3 - "${key_json}" <<'PY'
import json
import sys

data = json.load(open(sys.argv[1], encoding="utf-8"))
print(data["api_key"])
PY
)"
  [[ -n "${API_KEY}" ]] || die "failed to create API key"
}

write_jobspec() {
  local jobspec_path="$1"
  local allow_amf_reuse="$2"
  local allow_spec="$3"

  python3 - "${PROMPT_PATH}" "${jobspec_path}" "${allow_amf_reuse}" "${allow_spec}" "${OUTPUT_TOKENS}" "${N_CTX}" "${N_BATCH}" "${SEED}" "${SERVED_MODEL_NAME}" "${VLLM_URL}" <<'PY'
import json
import sys
from pathlib import Path

prompt_path = Path(sys.argv[1])
jobspec_path = Path(sys.argv[2])
allow_amf_reuse = bool(int(sys.argv[3]))
allow_spec = bool(int(sys.argv[4]))
max_tokens = int(sys.argv[5])
n_ctx = int(sys.argv[6])
n_batch = int(sys.argv[7])
seed = int(sys.argv[8])
model_id = sys.argv[9]
endpoint = sys.argv[10]

prompt = prompt_path.read_text(encoding="utf-8")
jobspec = {
    "schema_version": "korith.jobspec.v1",
    "backend_id": "vllm",
    "model": {
        "model_id": model_id,
        "endpoint": endpoint,
    },
    "prompt": prompt,
    "deterministic_cfg": {
        "seed": seed,
        "n_ctx": n_ctx,
        "n_batch": n_batch,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
    },
    "policy": {
        "allow_amf_reuse": allow_amf_reuse,
        "allow_spec": allow_spec,
    },
}
jobspec_path.write_text(json.dumps(jobspec, ensure_ascii=False), encoding="utf-8")
PY
}

run_bench_spec() {
  local scenario_id="$1"
  local jobspec_path="$2"
  local runs="$3"
  local tag="$4"

  local raw_path="${RESULTS_DIR}/raw.${scenario_id}.${tag}.log"
  local json_path="${RESULTS_DIR}/bench.${scenario_id}.${tag}.json"

  env "${STACK_ENV[@]}" \
    python3 platform/korith_platform.py bench spec \
      --jobspec "${jobspec_path}" \
      --runs "${runs}" \
      --timeout-s "${BENCH_TIMEOUT_S}" \
      --url "${ROUTER_URL}" \
      --api-key "${API_KEY}" >"${raw_path}"

  sed -n '/^{/,$p' "${raw_path}" >"${json_path}"
  python3 - "${json_path}" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as f:
    data = json.load(f)
if not isinstance(data, dict) or "rows" not in data:
    raise SystemExit(f"invalid benchmark json: {path}")
PY
  echo "${json_path}"
}

summarize_scenario() {
  local scenario_id="$1"
  local scenario_label="$2"
  local measured_json="$3"
  local summary_json="${RESULTS_DIR}/summary.${scenario_id}.json"

  python3 - "${measured_json}" "${summary_json}" "${scenario_id}" "${scenario_label}" <<'PY'
import json
import statistics
import sys
from collections import Counter

in_path = sys.argv[1]
out_path = sys.argv[2]
scenario_id = sys.argv[3]
scenario_label = sys.argv[4]

with open(in_path, encoding="utf-8") as f:
    payload = json.load(f)

rows = payload.get("rows", [])
if not rows:
    raise SystemExit(f"no rows in measured benchmark: {in_path}")

statuses = [str(r.get("status", "UNKNOWN")) for r in rows]
totals = [float(r.get("metrics", {}).get("perf", {}).get("total_ms", 0.0) or 0.0) for r in rows]
decodes = [float(r.get("metrics", {}).get("perf", {}).get("decode_ms", 0.0) or 0.0) for r in rows]
tps_vals = [float(r.get("metrics", {}).get("perf", {}).get("avg_tps", 0.0) or 0.0) for r in rows]
skip_vals = [float(r.get("metrics", {}).get("amf", {}).get("skip_ratio", 0.0) or 0.0) for r in rows]
amf_decisions = [str(r.get("metrics", {}).get("amf", {}).get("decision", "unknown")) for r in rows]
decision_counts = Counter(amf_decisions)
decision_rollup = ",".join(f"{k}:{decision_counts[k]}" for k in sorted(decision_counts.keys()))

summary = {
    "scenario_id": scenario_id,
    "scenario_label": scenario_label,
    "runs": len(rows),
    "all_succeeded": all(s == "SUCCEEDED" for s in statuses),
    "statuses": statuses,
    "total_ms_avg": float(statistics.fmean(totals)),
    "total_ms_p50": float(statistics.median(totals)),
    "decode_ms_avg": float(statistics.fmean(decodes)),
    "tps_avg": float(statistics.fmean(tps_vals)),
    "skip_ratio_avg": float(statistics.fmean(skip_vals)),
    "amf_decision_rollup": decision_rollup,
}

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
PY

  SUMMARY_FILES+=("${summary_json}")
}

run_scenario() {
  local scenario_id="$1"
  local scenario_label="$2"
  local prefix_enabled="$3"
  local fp8_enabled="$4"
  local native_spec_enabled="$5"
  local allow_amf_reuse="$6"
  local mf_enabled="$7"
  local allow_spec="$8"
  local full_budget_on_hit="$9"

  stop_stack
  mkdir -p "${RESULTS_DIR}" "${LOG_DIR}" "${RUNTIME_DIR}"

  start_vllm "${scenario_id}" "${prefix_enabled}" "${fp8_enabled}" "${native_spec_enabled}" "${full_budget_on_hit}"
  start_platform "${scenario_id}" "${mf_enabled}"
  create_api_key "${scenario_id}"

  local jobspec_path="${RESULTS_DIR}/jobspec.${scenario_id}.json"
  write_jobspec "${jobspec_path}" "${allow_amf_reuse}" "${allow_spec}"

  if [[ "${prefix_enabled}" == "1" && "${WARMUP_RUNS}" -gt 0 ]]; then
    log "warmup scenario=${scenario_id} runs=${WARMUP_RUNS}"
    run_bench_spec "${scenario_id}" "${jobspec_path}" "${WARMUP_RUNS}" "warmup" >/dev/null
  fi

  log "measured benchmark scenario=${scenario_id} runs=${RUNS_PER_SCENARIO}"
  local measured_json
  measured_json="$(run_bench_spec "${scenario_id}" "${jobspec_path}" "${RUNS_PER_SCENARIO}" "measured")"
  summarize_scenario "${scenario_id}" "${scenario_label}" "${measured_json}"
}

render_final_report() {
  python3 - "${RESULTS_JSON}" "${SUMMARY_FILES[@]}" <<'PY'
import json
import sys
from datetime import datetime, timezone

out_path = sys.argv[1]
summary_paths = sys.argv[2:]
if not summary_paths:
    raise SystemExit("no scenario summaries were generated")

rows = []
for path in summary_paths:
    with open(path, encoding="utf-8") as f:
        rows.append(json.load(f))

baseline = rows[0]
baseline_total = float(baseline.get("total_ms_avg", 0.0) or 0.0)
for row in rows:
    total = float(row.get("total_ms_avg", 0.0) or 0.0)
    if baseline_total > 0.0:
        row["savings_pct"] = ((baseline_total - total) / baseline_total) * 100.0
    else:
        row["savings_pct"] = 0.0

best = max(rows, key=lambda r: float(r.get("savings_pct", 0.0) or 0.0))
payload = {
    "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "baseline_total_ms": baseline_total,
    "total_savings_pct": float(best.get("savings_pct", 0.0) or 0.0),
    "best_scenario_id": best.get("scenario_id", ""),
    "best_scenario_label": best.get("scenario_label", ""),
    "scenarios": rows,
}
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)

headers = (
    "Scenario",
    "Total(ms)",
    "Decode(ms)",
    "TPS",
    "AMF decision",
    "Skip ratio",
    "Savings(%)",
)
print("")
print("{:<34} {:>11} {:>11} {:>9} {:>16} {:>11} {:>11}".format(*headers))
print("-" * 112)
for row in rows:
    print(
        "{:<34} {:>11.2f} {:>11.2f} {:>9.2f} {:>16} {:>11.4f} {:>11.2f}".format(
            str(row.get("scenario_label", "")),
            float(row.get("total_ms_avg", 0.0) or 0.0),
            float(row.get("decode_ms_avg", 0.0) or 0.0),
            float(row.get("tps_avg", 0.0) or 0.0),
            str(row.get("amf_decision_rollup", ""))[:16],
            float(row.get("skip_ratio_avg", 0.0) or 0.0),
            float(row.get("savings_pct", 0.0) or 0.0),
        )
    )
print("-" * 112)
print(
    "total_savings_pct={:.2f} best_scenario={} ({})".format(
        float(payload["total_savings_pct"]),
        str(payload["best_scenario_id"]),
        str(payload["best_scenario_label"]),
    )
)
print(f"results_json={out_path}")
print("")
PY
}

main() {
  need_cmd python3
  need_cmd curl
  need_cmd vllm

  mkdir -p "${RESULTS_DIR}" "${LOG_DIR}" "${RUNTIME_DIR}"

  if command -v nvidia-smi >/dev/null 2>&1; then
    log "GPU inventory:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
  fi

  ensure_models
  build_prompt_file

  # id|label|prefix|fp8|native_spec|allow_amf_reuse|mf_enabled|allow_spec|full_budget_on_hit
  local -a scenarios=(
    "baseline|Baseline (No AMF/Spec/FP8)|0|0|0|0|0|0|0"
    "amf_only|AMF Only|1|0|0|1|1|0|0"
    "amf_fp8|AMF + FP8 KV|1|1|0|1|1|0|0"
    "amf_fp8_native_spec|AMF + FP8 KV + Native Spec|1|1|1|1|1|1|0"
    "amf_fp8_native_spec_sched|AMF + FP8 KV + Native Spec + Scheduler|1|1|1|1|1|1|1"
  )

  # Optional: restrict execution to a subset of scenario IDs.
  # Example:
  #   SCENARIO_IDS_CSV=amf_only
  #   SCENARIO_IDS_CSV=baseline,amf_only,amf_fp8
  if [[ -n "${SCENARIO_IDS_CSV:-}" ]]; then
    local -a selected=()
    local -a wanted=()
    local wanted_raw
    IFS=',' read -r -a wanted_raw <<<"${SCENARIO_IDS_CSV}"
    local w
    for w in "${wanted_raw[@]}"; do
      w="${w//[[:space:]]/}"
      if [[ -n "${w}" ]]; then
        wanted+=("${w}")
      fi
    done

    local row sid
    for row in "${scenarios[@]}"; do
      IFS='|' read -r sid _ <<<"${row}"
      for w in "${wanted[@]}"; do
        if [[ "${sid}" == "${w}" ]]; then
          selected+=("${row}")
          break
        fi
      done
    done

    if [[ "${#selected[@]}" -eq 0 ]]; then
      die "SCENARIO_IDS_CSV did not match any known scenario ids"
    fi
    scenarios=("${selected[@]}")
    log "scenario filter active ids=${SCENARIO_IDS_CSV} count=${#scenarios[@]}"
  fi

  local row
  for row in "${scenarios[@]}"; do
    IFS='|' read -r scenario_id scenario_label prefix_enabled fp8_enabled native_spec_enabled allow_amf_reuse mf_enabled allow_spec full_budget_on_hit <<<"${row}"
    log "running scenario=${scenario_id}"
    run_scenario \
      "${scenario_id}" \
      "${scenario_label}" \
      "${prefix_enabled}" \
      "${fp8_enabled}" \
      "${native_spec_enabled}" \
      "${allow_amf_reuse}" \
      "${mf_enabled}" \
      "${allow_spec}" \
      "${full_budget_on_hit}"
  done

  stop_stack
  render_final_report
}

main "$@"

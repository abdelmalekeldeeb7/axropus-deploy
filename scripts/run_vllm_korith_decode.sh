#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

MODEL_PATH="${1:-models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf}"
DRAFT_MODEL_PATH="${2:-${KORITH_VLLM_SPECULATIVE_MODEL:-}}"
HOST="${VLLM_HOST:-127.0.0.1}"
PORT="${VLLM_PORT:-8001}"
MAX_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
MAX_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-4096}"
MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-64}"
VLLM_DTYPE="${VLLM_DTYPE:-}"
VLLM_ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING:-1}"
VLLM_ENABLE_PROMPT_TOKENS_DETAILS="${VLLM_ENABLE_PROMPT_TOKENS_DETAILS:-1}"
VLLM_KV_CACHE_DTYPE="${VLLM_KV_CACHE_DTYPE:-auto}"
KORITH_VLLM_USE_FP8_KV="${KORITH_VLLM_USE_FP8_KV:-0}"
KORITH_VLLM_FP8_FALLBACK_DTYPE="${KORITH_VLLM_FP8_FALLBACK_DTYPE:-auto}"
KORITH_VLLM_NATIVE_SPEC="${KORITH_VLLM_NATIVE_SPEC:-0}"
KORITH_VLLM_NATIVE_SPEC_K="${KORITH_VLLM_NATIVE_SPEC_K:-5}"
KORITH_VLLM_NATIVE_SPEC_DRAFT_TP="${KORITH_VLLM_NATIVE_SPEC_DRAFT_TP:-1}"
VLLM_CUDAGRAPH_CAPTURE_SIZES="${VLLM_CUDAGRAPH_CAPTURE_SIZES:-}"
VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE="${VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE:-}"
VLLM_PROFILER_CONFIG="${VLLM_PROFILER_CONFIG:-}"

truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

if [[ -z "${VLLM_DTYPE}" ]]; then
  if [[ "${MODEL_PATH}" == *.gguf ]]; then
    # vLLM GGUF path rejects bf16; keep this deterministic.
    VLLM_DTYPE="float16"
  else
    VLLM_DTYPE="auto"
  fi
fi

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "model not found: ${MODEL_PATH}"
  exit 1
fi

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export KORITH_VLLM_DECODE_FIRST="${KORITH_VLLM_DECODE_FIRST:-1}"
export KORITH_VLLM_ADAPTIVE_BUDGET="${KORITH_VLLM_ADAPTIVE_BUDGET:-1}"
export KORITH_VLLM_BUDGET_SCALE_DECODE="${KORITH_VLLM_BUDGET_SCALE_DECODE:-0.75}"
export KORITH_VLLM_DECODE_RATIO_THRESHOLD="${KORITH_VLLM_DECODE_RATIO_THRESHOLD:-0.6}"
export KORITH_VLLM_MIN_SCHEDULED_TOKENS="${KORITH_VLLM_MIN_SCHEDULED_TOKENS:-128}"
export KORITH_VLLM_SHAPE_BOOST="${KORITH_VLLM_SHAPE_BOOST:-0.15}"
export KORITH_VLLM_STARVATION_S="${KORITH_VLLM_STARVATION_S:-0.75}"
export KORITH_VLLM_LANE_BIAS_SPEC_HIT="${KORITH_VLLM_LANE_BIAS_SPEC_HIT:-1.25}"
export KORITH_VLLM_LANE_BIAS_HIT="${KORITH_VLLM_LANE_BIAS_HIT:-1.10}"
export KORITH_VLLM_LANE_BIAS_SPEC_MISS="${KORITH_VLLM_LANE_BIAS_SPEC_MISS:-0.95}"
export KORITH_VLLM_LANE_BIAS_MISS="${KORITH_VLLM_LANE_BIAS_MISS:-1.00}"
export KORITH_VLLM_WAITING_PRESSURE_THRESHOLD="${KORITH_VLLM_WAITING_PRESSURE_THRESHOLD:-8}"
export KORITH_VLLM_WAITING_RETUNE="${KORITH_VLLM_WAITING_RETUNE:-1}"
export KORITH_VLLM_WAITING_RETUNE_MAX="${KORITH_VLLM_WAITING_RETUNE_MAX:-128}"
export KORITH_VLLM_PREFILL_PENALTY="${KORITH_VLLM_PREFILL_PENALTY:-0.20}"
export KORITH_VLLM_SHORT_DECODE_BONUS="${KORITH_VLLM_SHORT_DECODE_BONUS:-0.15}"
export KORITH_VLLM_SPEC_SCORE_PER_K="${KORITH_VLLM_SPEC_SCORE_PER_K:-0.02}"
export KORITH_VLLM_SPEC_SCORE_CAP="${KORITH_VLLM_SPEC_SCORE_CAP:-0.15}"
export KORITH_VLLM_RUNTIME_CONTRACT_ENABLED="${KORITH_VLLM_RUNTIME_CONTRACT_ENABLED:-1}"
export KORITH_VLLM_PRIORITY_SCHED="${KORITH_VLLM_PRIORITY_SCHED:-1}"
export KORITH_VLLM_ENABLE_SPEC_DECODE="${KORITH_VLLM_ENABLE_SPEC_DECODE:-1}"
export KORITH_VLLM_SPECULATIVE_NUM_TOKENS="${KORITH_VLLM_SPECULATIVE_NUM_TOKENS:-6}"
if [[ -n "${DRAFT_MODEL_PATH}" ]]; then
  export KORITH_VLLM_SPECULATIVE_MODEL="${DRAFT_MODEL_PATH}"
fi

echo "starting vLLM with KorithDecodeScheduler on ${HOST}:${PORT}"
echo "model=${MODEL_PATH}"
if [[ -n "${KORITH_VLLM_SPECULATIVE_MODEL:-}" ]]; then
  echo "spec_draft=${KORITH_VLLM_SPECULATIVE_MODEL}"
fi
echo "dtype=${VLLM_DTYPE}"
echo "prefix_caching=${VLLM_ENABLE_PREFIX_CACHING}"
echo "prompt_tokens_details=${VLLM_ENABLE_PROMPT_TOKENS_DETAILS}"
echo "kv_cache_dtype=${VLLM_KV_CACHE_DTYPE}"

PREFIX_CACHE_FLAG="--no-enable-prefix-caching"
if truthy "${VLLM_ENABLE_PREFIX_CACHING}"; then
  PREFIX_CACHE_FLAG="--enable-prefix-caching"
fi

PROMPT_DETAILS_FLAG="--no-enable-prompt-tokens-details"
if truthy "${VLLM_ENABLE_PROMPT_TOKENS_DETAILS}"; then
  PROMPT_DETAILS_FLAG="--enable-prompt-tokens-details"
fi

if truthy "${KORITH_VLLM_USE_FP8_KV}"; then
  VLLM_KV_CACHE_DTYPE="fp8"
fi

if truthy "${KORITH_VLLM_NATIVE_SPEC}" && [[ -n "${DRAFT_MODEL_PATH}" ]]; then
  if [[ -z "${KORITH_VLLM_ENABLE_SPEC_DECODE:-}" ]]; then
    # Avoid double-spec logic: native vLLM speculative decode should be the only path.
    export KORITH_VLLM_ENABLE_SPEC_DECODE=0
  fi
  NATIVE_SPEC_JSON="{\"method\":\"draft_model\",\"model\":\"${DRAFT_MODEL_PATH}\",\"num_speculative_tokens\":${KORITH_VLLM_NATIVE_SPEC_K},\"draft_tensor_parallel_size\":${KORITH_VLLM_NATIVE_SPEC_DRAFT_TP}}"
  echo "native_spec=1 k=${KORITH_VLLM_NATIVE_SPEC_K} draft_tp=${KORITH_VLLM_NATIVE_SPEC_DRAFT_TP}"
else
  NATIVE_SPEC_JSON=""
fi

declare -a CMD
CMD=(vllm serve "${MODEL_PATH}"
  --host "${HOST}"
  --port "${PORT}"
  --dtype "${VLLM_DTYPE}"
  "${PREFIX_CACHE_FLAG}"
  "${PROMPT_DETAILS_FLAG}"
  --kv-cache-dtype "${VLLM_KV_CACHE_DTYPE}"
  --max-model-len "${MAX_LEN}"
  --max-num-batched-tokens "${MAX_BATCHED_TOKENS}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --scheduling-policy priority
  --scheduler-cls korith_vllm_ext.decode_scheduler.KorithDecodeScheduler
)

if [[ -n "${NATIVE_SPEC_JSON}" ]]; then
  CMD+=(--speculative-config "${NATIVE_SPEC_JSON}")
fi
if [[ -n "${VLLM_CUDAGRAPH_CAPTURE_SIZES}" ]]; then
  # shellcheck disable=SC2206
  CG_SIZES=( ${VLLM_CUDAGRAPH_CAPTURE_SIZES} )
  CMD+=(--cudagraph-capture-sizes "${CG_SIZES[@]}")
fi
if [[ -n "${VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE}" ]]; then
  CMD+=(--max-cudagraph-capture-size "${VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE}")
fi
if [[ -n "${VLLM_PROFILER_CONFIG}" ]]; then
  CMD+=(--profiler-config "${VLLM_PROFILER_CONFIG}")
fi

if [[ "${VLLM_KV_CACHE_DTYPE}" == "fp8" ]]; then
  "${CMD[@]}" || {
    echo "fp8 kv-cache failed; retrying with kv-cache-dtype=${KORITH_VLLM_FP8_FALLBACK_DTYPE}"
    CMD_FP8_FALLBACK=( "${CMD[@]}" )
    for i in "${!CMD_FP8_FALLBACK[@]}"; do
      if [[ "${CMD_FP8_FALLBACK[$i]}" == "fp8" ]]; then
        CMD_FP8_FALLBACK[$i]="${KORITH_VLLM_FP8_FALLBACK_DTYPE}"
      fi
    done
    exec "${CMD_FP8_FALLBACK[@]}"
  }
else
  exec "${CMD[@]}"
fi

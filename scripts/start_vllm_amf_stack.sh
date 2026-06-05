#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/amf_stack.env}"

if [[ ! -f "$CONFIG" ]]; then
  echo "[ERROR] config not found: $CONFIG" >&2
  echo "Copy configs/amf_stack.env.example to configs/amf_stack.env and edit it." >&2
  exit 2
fi

set -a
# shellcheck disable=SC1090
source "$CONFIG"
set +a

cd "${AMF_REPO_DIR:-$PWD}"

if [[ -d ".venv" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

required=(
  AMF_MODEL
  KORITH_ENABLE_AMF
  KORITH_AMF_PATH
)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "[ERROR] required config value missing: $name" >&2
    exit 2
  fi
done

mkdir -p "$KORITH_AMF_PATH"

echo "[AMF_STACK] repo=$PWD"
echo "[AMF_STACK] model=$AMF_MODEL"
echo "[AMF_STACK] host=${AMF_HOST:-0.0.0.0} port=${AMF_PORT:-8000}"
echo "[AMF_STACK] vllm_kv_gb=${AMF_KV_CACHE_MEMORY_GB:-auto} amf_raw_gb=${KORITH_VRAM_CACHE_GB:-0} amf_pool_gb=${KORITH_VRAM_POOL_GB:-0} quant=${KORITH_VRAM_POOL_QUANT:-none}"
echo "[AMF_STACK] lmcache=${AXROPUS_ENABLE_LMCACHE:-0} lmcache_cpu_gb=${LMCACHE_MAX_LOCAL_CPU_SIZE:-0}"

args=(
  -m korith_vllm_ext.korith_vllm_server
  --serve
  --model "$AMF_MODEL"
  --host "${AMF_HOST:-0.0.0.0}"
  --port "${AMF_PORT:-8000}"
  --kv-cache-dtype "${AMF_KV_CACHE_DTYPE:-auto}"
)

if [[ "${AMF_ENFORCE_EAGER:-0}" =~ ^(1|true|yes|on)$ ]]; then
  args+=(--enforce-eager)
fi
if [[ -n "${AMF_MAX_MODEL_LEN:-}" && "${AMF_MAX_MODEL_LEN:-0}" != "0" ]]; then
  args+=(--max-model-len "$AMF_MAX_MODEL_LEN")
fi
if [[ -n "${AMF_GPU_MEMORY_UTILIZATION:-}" && "${AMF_GPU_MEMORY_UTILIZATION:-0}" != "0" ]]; then
  args+=(--gpu-memory-utilization "$AMF_GPU_MEMORY_UTILIZATION")
fi
if [[ -n "${AMF_KV_CACHE_MEMORY_GB:-}" && "${AMF_KV_CACHE_MEMORY_GB:-0}" != "0" ]]; then
  args+=(--kv-cache-memory-gb "$AMF_KV_CACHE_MEMORY_GB")
fi
if [[ -n "${AMF_TENSOR_PARALLEL_SIZE:-}" && "${AMF_TENSOR_PARALLEL_SIZE:-0}" != "0" ]]; then
  args+=(--tensor-parallel-size "$AMF_TENSOR_PARALLEL_SIZE")
fi
if [[ -n "${AMF_MAX_NUM_SEQS:-}" && "${AMF_MAX_NUM_SEQS:-0}" != "0" ]]; then
  args+=(--max-num-seqs "$AMF_MAX_NUM_SEQS")
fi
if [[ -n "${AMF_CPU_OFFLOAD_GB:-}" && "${AMF_CPU_OFFLOAD_GB:-0}" != "0" ]]; then
  args+=(--cpu-offload-gb "$AMF_CPU_OFFLOAD_GB")
fi

exec "${PYTHON_BIN:-python3}" "${args[@]}"

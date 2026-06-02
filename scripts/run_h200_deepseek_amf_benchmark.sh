#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
ARM="${2:-all}"

cd "${AMF_REPO_DIR:-$HOME/amf}"

if [[ -d ".venv" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

export PYTHONPATH="$PWD"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

case "$MODE" in
  smoke)
    CONFIG="configs/h200_deepseek_v4_flash_smoke_32req.yaml"
    ;;
  proof)
    CONFIG="configs/h200_deepseek_v4_flash_amf_stack_256req.yaml"
    ;;
  *)
    echo "usage: $0 [smoke|proof] [all|apc|lmcache|amf|amf_lmcache]" >&2
    exit 2
    ;;
esac

mkdir -p benchmark_results/logs
LOG="benchmark_results/logs/h200_deepseek_${MODE}_${ARM}_$(date +%Y%m%d_%H%M%S).log"

echo "[RUN] mode=$MODE arm=$ARM config=$CONFIG log=$LOG"

if [[ "$ARM" == "all" ]]; then
  python scripts/production_realistic_apc_vs_amf.py \
    --config "$CONFIG" \
    2>&1 | tee "$LOG"
else
  python scripts/production_realistic_apc_vs_amf.py \
    --config "$CONFIG" \
    --only-arm "$ARM" \
    2>&1 | tee "$LOG"
fi

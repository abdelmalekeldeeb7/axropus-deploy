#!/usr/bin/env bash
# h100_setup.sh — One-shot setup for AMF benchmark on a fresh H100 node
# Run this once after SSH'ing into the rented H100 (Lambda, RunPod, Nebius, etc.)
#
# Usage:
#   bash scripts/h100_setup.sh
#
# Environment:
#   HF_TOKEN       — Hugging Face token (required for model download)
#   MODEL_SIZE     — "70b" (default) or "8b" (faster, smaller)
#   SKIP_MODEL     — set to 1 to skip model download (if already present)
#   SKIP_BUILD     — set to 1 to skip C++ build (if binary already exists)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

MODEL_SIZE="${MODEL_SIZE:-70b}"
SKIP_MODEL="${SKIP_MODEL:-0}"
SKIP_BUILD="${SKIP_BUILD:-0}"
N_GPUS="${N_GPUS:-1}"   # Set to 4 for 120b, 8 for 405b NVL8

log() { echo "[$(date +%H:%M:%S)] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

# ── 1. System check ──────────────────────────────────────────────────────────
log "=== System check ==="
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found — not an H100 node?"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo unknown)"
log "GPU: ${GPU_NAME}"

command -v python3 >/dev/null 2>&1 || die "python3 not found"
python3 --version

# ── 2. Python deps ───────────────────────────────────────────────────────────
log "=== Installing Python dependencies ==="
pip install --quiet --upgrade huggingface_hub hf_transfer 2>/dev/null || \
  python3 -m pip install --user --quiet --upgrade huggingface_hub hf_transfer

# ── 3. Build korith_dynamic ──────────────────────────────────────────────────
if [[ "${SKIP_BUILD}" == "1" ]] && [[ -x "${ROOT_DIR}/build/korith_dynamic" ]]; then
    log "Skipping build (binary exists)"
else
    log "=== Building korith_dynamic ==="

    # Check for llama.cpp dependency
    if [[ -z "${LLAMA_ROOT:-}" ]]; then
        if [[ -d "/opt/llama.cpp" ]]; then
            export LLAMA_ROOT="/opt/llama.cpp"
        elif [[ -d "${ROOT_DIR}/../llama.cpp" ]]; then
            export LLAMA_ROOT="$(cd "${ROOT_DIR}/../llama.cpp" && pwd)"
        else
            log "llama.cpp not found — cloning..."
            cd /tmp
            if [[ ! -d llama.cpp ]]; then
                git clone --depth=1 https://github.com/ggml-org/llama.cpp.git llama.cpp
                cd llama.cpp && mkdir -p build && cd build
                cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_CUDA_F16=ON
                ninja -j "$(nproc)"
            fi
            export LLAMA_ROOT="/tmp/llama.cpp"
            cd "${ROOT_DIR}"
        fi
    fi
    log "LLAMA_ROOT=${LLAMA_ROOT}"

    command -v cmake >/dev/null 2>&1 || die "cmake not found — install with: apt install cmake"
    command -v ninja >/dev/null 2>&1 || sudo apt-get install -y ninja-build 2>/dev/null || true

    mkdir -p "${ROOT_DIR}/build"
    cmake -S "${ROOT_DIR}/korith_c_api" -B "${ROOT_DIR}/build" \
        -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_ROOT="${LLAMA_ROOT}" \
        -DGGML_CUDA=ON \
        2>&1 | tail -5

    cmake --build "${ROOT_DIR}/build" -j "$(nproc)" 2>&1 | tail -10

    [[ -x "${ROOT_DIR}/build/korith_dynamic" ]] || die "Build failed — korith_dynamic not found"
    log "Build OK: ${ROOT_DIR}/build/korith_dynamic"
fi

# ── 4. Download model ────────────────────────────────────────────────────────
mkdir -p "${ROOT_DIR}/models"

if [[ "${MODEL_SIZE}" == "405b" ]]; then
    # 8x H100 NVL8 — ~230GB Q4_K_M, needs ~500GB free disk for model + 1M KV cache
    GGUF_FILENAME="Meta-Llama-3.1-405B-Instruct-Q4_K_M.gguf"
    HF_REPO="bartowski/Meta-Llama-3.1-405B-Instruct-GGUF"
    DRAFT_FILENAME="Meta-Llama-3.2-1B-Instruct-Q4_K_M.gguf"
    DRAFT_REPO="bartowski/Llama-3.2-1B-Instruct-GGUF"
    BENCH_PROFILE="405b_1m"
elif [[ "${MODEL_SIZE}" == "120b" ]]; then
    # 4x H100 SXM — Llama-3.1-Nemotron-70B is closest open 120B-class GGUF
    # Nebius Token Factory "120B" = nvidia/Llama-3_1-Nemotron-Ultra-253B-v1 lite variant
    # Use bartowski's Nemotron-70B GGUF as the local proxy (same architecture family)
    GGUF_FILENAME="Llama-3.1-Nemotron-70B-Instruct-HF-Q4_K_M.gguf"
    HF_REPO="bartowski/Llama-3.1-Nemotron-70B-Instruct-HF-GGUF"
    DRAFT_FILENAME="Meta-Llama-3.2-1B-Instruct-Q4_K_M.gguf"
    DRAFT_REPO="bartowski/Llama-3.2-1B-Instruct-GGUF"
    BENCH_PROFILE="120b_1m"
    N_GPUS=2
elif [[ "${MODEL_SIZE}" == "70b" ]]; then
    GGUF_FILENAME="Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf"
    HF_REPO="bartowski/Meta-Llama-3.1-70B-Instruct-GGUF"
    DRAFT_FILENAME="Meta-Llama-3.2-1B-Instruct-Q4_K_M.gguf"
    DRAFT_REPO="bartowski/Llama-3.2-1B-Instruct-GGUF"
    BENCH_PROFILE="70b_1m"
else
    # 8B fallback — faster download, good for verifying the pipeline
    GGUF_FILENAME="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    HF_REPO="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
    DRAFT_FILENAME="Meta-Llama-3.2-1B-Instruct-Q4_K_M.gguf"
    DRAFT_REPO="bartowski/Llama-3.2-1B-Instruct-GGUF"
    BENCH_PROFILE=""
fi

MODEL_PATH="${ROOT_DIR}/models/${GGUF_FILENAME}"
DRAFT_PATH="${ROOT_DIR}/models/${DRAFT_FILENAME}"

if [[ "${SKIP_MODEL}" == "1" ]] && [[ -f "${MODEL_PATH}" ]]; then
    log "Skipping model download (file exists)"
else
    [[ -n "${HF_TOKEN:-}" ]] || die "HF_TOKEN is required — export HF_TOKEN=hf_..."
    log "=== Downloading ${GGUF_FILENAME} ==="
    HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download "${HF_REPO}" \
        "${GGUF_FILENAME}" \
        --local-dir "${ROOT_DIR}/models/" \
        --token "${HF_TOKEN}"
    log "Model downloaded: ${MODEL_PATH} ($(du -sh "${MODEL_PATH}" | cut -f1))"
fi

if [[ ! -f "${DRAFT_PATH}" ]]; then
    log "=== Downloading draft model ${DRAFT_FILENAME} ==="
    HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download "${DRAFT_REPO}" \
        "${DRAFT_FILENAME}" \
        --local-dir "${ROOT_DIR}/models/" \
        --token "${HF_TOKEN:-}" || log "Draft model download failed (optional, skipping)"
fi

# ── 5. Storage check ─────────────────────────────────────────────────────────
log "=== Storage check ==="
df -h "${ROOT_DIR}" | tail -1
FREE_GB="$(df --output=avail -BG "${ROOT_DIR}" 2>/dev/null | tail -1 | tr -d 'G' | tr -d ' ' || echo 0)"
if [[ "${MODEL_SIZE}" == "405b" ]]; then
    DISK_WARN_GB=500
    DISK_MSG="405B model (~230GB) + 1M FP8 KV cache (~252GB) needs ~500GB disk"
else
    DISK_WARN_GB=400
    DISK_MSG="1M context KV cache for 70B needs ~320GB disk"
fi
if [[ "${FREE_GB}" -lt "${DISK_WARN_GB}" ]]; then
    log "WARNING: Only ${FREE_GB}GB free — ${DISK_MSG}"
    log "  Consider using --context 131072 (128K) which needs ~40GB"
fi

# ── 6. Print run command ─────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  SETUP COMPLETE — Ready to benchmark"
echo "============================================================"
echo ""
echo "  GPU:    ${GPU_NAME}"
echo "  Model:  ${MODEL_PATH}"
echo "  Draft:  ${DRAFT_PATH}"
echo ""
echo "  Run the benchmark:"
echo ""
if [[ "${MODEL_SIZE}" == "405b" ]]; then
    echo "    # Step 1 — Sanity check: 128K context, 5 runs (~5 min)"
    echo "    python3 scripts/h100_1m_benchmark.py \\"
    echo "        --model '${MODEL_PATH}' \\"
    echo "        --draft-model '${DRAFT_PATH}' \\"
    echo "        --profile 405b_128k --runs 5"
    echo ""
    echo "    # Step 2 — AMF baseline: 1M context, 100 runs, no reasoning (~2hr)"
    echo "    python3 scripts/h100_1m_benchmark.py \\"
    echo "        --model '${MODEL_PATH}' \\"
    echo "        --draft-model '${DRAFT_PATH}' \\"
    echo "        --profile 405b_1m --runs 100"
    echo ""
    echo "    # Step 3 — AMF + AI reasoning: 1M context, 100 runs (~2hr)"
    echo "    python3 scripts/h100_1m_benchmark.py \\"
    echo "        --model '${MODEL_PATH}' \\"
    echo "        --draft-model '${DRAFT_PATH}' \\"
    echo "        --profile 405b_1m --runs 100 --reasoning"
else
    echo "    # Full 1M context benchmark (50 runs, cold + warm)"
    echo "    python3 scripts/h100_1m_benchmark.py \\"
    echo "        --model '${MODEL_PATH}' \\"
    echo "        --draft-model '${DRAFT_PATH}' \\"
    echo "        --context 1000000 \\"
    echo "        --runs 50"
    echo ""
    echo "    # Quick 128K test to verify pipeline (5 runs)"
    echo "    python3 scripts/h100_1m_benchmark.py \\"
    echo "        --model '${MODEL_PATH}' \\"
    echo "        --context 131072 \\"
    echo "        --runs 5"
fi
echo ""
echo "  Results land in: platform_data/h100_bench/"
echo "============================================================"

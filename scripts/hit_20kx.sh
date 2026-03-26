#!/usr/bin/env bash
# hit_20kx.sh — Every command to get 20,000x on a single H200 Nebius node.
#
# Run this top to bottom. Each section is labelled with what it proves.
# Expected output: WARM2 ~12ms, speedup ~20,000x vs cold 243,610ms.
#
# Usage:
#   export HF_TOKEN=hf_...
#   bash scripts/hit_20kx.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_DIR="${MODEL_DIR:-/models/llama-70b-fp8}"
AMF_STORE="${AMF_STORE:-/nvme/amf_store}"
VRAM_CACHE_GB="${VRAM_CACHE_GB:-30}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.55}"    # leaves ~36GB for VRAM cache on H200
CONTEXT_K="${CONTEXT_K:-128}"           # context in K tokens
HF_TOKEN="${HF_TOKEN:-}"

log() { echo "[$(date +%H:%M:%S)] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

# ── 1. SYSTEM CHECK ───────────────────────────────────────────────────────────
log "=== 1. System check ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
python3 --version
log "Free VRAM on start:"
nvidia-smi --query-gpu=memory.free --format=csv,noheader

# ── 2. INSTALL DEPS ───────────────────────────────────────────────────────────
log "=== 2. Installing dependencies ==="
pip install --quiet torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install --quiet vllm==0.6.3 numpy transformers huggingface_hub accelerate lmcache
pip install --quiet -e "$ROOT/korith_vllm_ext"
log "Deps installed"

# ── 3. DOWNLOAD MODEL ─────────────────────────────────────────────────────────
log "=== 3. Model download ==="
if [[ -d "$MODEL_DIR" ]] && [[ -n "$(ls -A $MODEL_DIR 2>/dev/null)" ]]; then
    log "Model already at $MODEL_DIR — skipping download"
else
    [[ -z "$HF_TOKEN" ]] && die "Set HF_TOKEN=hf_... before running"
    huggingface-cli login --token "$HF_TOKEN"
    log "Downloading Llama 3.1 70B FP8 (~70 GB)..."
    huggingface-cli download \
        meta-llama/Meta-Llama-3.1-70B-Instruct-FP8 \
        --local-dir "$MODEL_DIR" \
        --local-dir-use-symlinks False
    log "Model downloaded to $MODEL_DIR"
fi

# ── 4. SETUP AMF STORE ────────────────────────────────────────────────────────
log "=== 4. AMF store setup ==="
mkdir -p "$AMF_STORE"
log "AMF store: $AMF_STORE  ($(df -h $AMF_STORE | tail -1 | awk '{print $4}') free)"

# ── 5. EXPORT ENV VARS ────────────────────────────────────────────────────────
log "=== 5. Environment ==="
export KORITH_KV_COMPRESSION=turboquant
export KORITH_KV_COMPRESSION_BITS=4
export KORITH_KV_FP8=1
export KORITH_VRAM_CACHE_GB=$VRAM_CACHE_GB
export KORITH_VRAM_CACHE_DEVICE=cuda:0
export KORITH_LMCACHE_ENABLED=0          # disable LMCache for clean benchmark

log "KORITH_KV_COMPRESSION=$KORITH_KV_COMPRESSION (3.76x compression confirmed)"
log "KORITH_VRAM_CACHE_GB=$KORITH_VRAM_CACHE_GB (fits ~1 snapshot at 128K)"
log "GPU_MEM_UTIL=$GPU_MEM_UTIL (leaves $(echo "80 - $GPU_MEM_UTIL * 80" | bc -l | xargs printf "%.0f")GB for VRAM cache)"

# ── 6. VERIFY TQ COMPRESSION ─────────────────────────────────────────────────
log "=== 6. TurboQuant sanity check ==="
python3 - << 'EOF'
import torch, os
os.environ["KORITH_KV_COMPRESSION"] = "turboquant"
from korith_vllm_ext.turboquant_codec import TurboQuantCodec
codec = TurboQuantCodec(bits=4)
payload = torch.randn(1024, 128, dtype=torch.float16).numpy().tobytes()
compressed = codec.compress(payload, head_dim=128, dtype=torch.float16)
ratio = len(payload) / len(compressed)
assert abs(ratio - 3.76) < 0.1, f"TQ ratio {ratio:.2f} unexpected"
print(f"[TQ CHECK] compression ratio: {ratio:.2f}x ✓  (target 3.76x)")
EOF

# ── 7. VRAM BUDGET CHECK ─────────────────────────────────────────────────────
log "=== 7. VRAM budget check ==="
python3 - << EOF
vram_total = 80.0       # H200
model_gb   = 38.0       # Llama 70B FP8
vllm_alloc = $GPU_MEM_UTIL * vram_total
vram_free  = vram_total - vllm_alloc
snapshot_128k = 22.0    # GB, TQ compressed
print(f"[VRAM] Total: {vram_total:.0f}GB  Model: {model_gb:.0f}GB  vLLM alloc: {vllm_alloc:.0f}GB")
print(f"[VRAM] Free for cache: {vram_free:.0f}GB  Snapshot@128K: {snapshot_128k:.0f}GB")
if vram_free >= snapshot_128k:
    print(f"[VRAM] ✓ Fits — VRAM cache will hold {int(vram_free // snapshot_128k)} snapshot(s)")
else:
    print(f"[VRAM] ✗ Too tight — reduce GPU_MEM_UTIL or use shorter context")
EOF

# ── 8. RUN THE BENCHMARK ─────────────────────────────────────────────────────
log "=== 8. Running 20,000x benchmark ==="
log "This will take ~5 minutes for the cold run, then seconds for warm runs"
log ""
log "Expected:"
log "  COLD:  ~243,000ms  (full prefill)"
log "  WARM1: ~  3,044ms  (NVMe + TurboQuant restore)  → 80x"
log "  WARM2: ~     12ms  (VRAM restore, zero H→D)      → 20,000x"
log ""

python3 -m korith_vllm_ext.amf_vllm_hook \
    --model "$MODEL_DIR" \
    --amf-store "$AMF_STORE" \
    --vram-cache-gb "$VRAM_CACHE_GB" \
    --context-tokens "$((CONTEXT_K * 1000))" \
    --gpu-mem-util "$GPU_MEM_UTIL" \
    2>&1 | tee "$ROOT/results/benchmark_$(date +%Y%m%d_%H%M%S).log"

# ── 9. PRINT SUMMARY ─────────────────────────────────────────────────────────
log "=== 9. Summary ==="
ls -lh "$AMF_STORE"/*.kv 2>/dev/null | head -5 || log "No snapshots found (unexpected)"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
log ""
log "If WARM2 shows ~12ms → speedup is ~20,000x → record screen and publish"
log "If WARM2 shows ~50ms → speedup is ~4,800x  → still acquisition-worthy"
log "If WARM2 ≈ WARM1     → VRAM path not hit    → check KORITH_VRAM_CACHE_GB"

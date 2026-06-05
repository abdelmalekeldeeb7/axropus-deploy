#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/amf_stack.env}"

if [[ ! -f "$CONFIG" ]]; then
  echo "[ERROR] config not found: $CONFIG" >&2
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

echo "[VERIFY] repo=$PWD"
echo "[VERIFY] python=$("${PYTHON_BIN:-python3}" -c 'import sys; print(sys.executable)')"

"${PYTHON_BIN:-python3}" - <<'PY'
import importlib
import os

import korith_vllm_ext  # noqa: F401 - patches vLLM EngineCore.

checks = []

try:
    import vllm
    checks.append(("vllm", True, getattr(vllm, "__version__", "unknown")))
except Exception as exc:
    checks.append(("vllm", False, repr(exc)))

try:
    from vllm.v1.engine.core import EngineCore
    checks.append(("amf_register_prefix", hasattr(EngineCore, "amf_register_prefix"), "EngineCore patch"))
    checks.append(("amf_get_cached_block_ids", hasattr(EngineCore, "amf_get_cached_block_ids"), "EngineCore patch"))
except Exception as exc:
    checks.append(("engine_core_patch", False, repr(exc)))

try:
    mod = importlib.import_module("korith_vllm_ext.amf_worker_ext")
    checks.append(("amf_worker_ext", hasattr(mod, "AmfWorkerExtension"), "import ok"))
except Exception as exc:
    checks.append(("amf_worker_ext", False, repr(exc)))

try:
    import lmcache  # type: ignore
    checks.append(("lmcache", True, getattr(lmcache, "__version__", "installed")))
except Exception as exc:
    checks.append(("lmcache", False, repr(exc)))

failed = False
for name, ok, detail in checks:
    print(f"[VERIFY] {name} ok={ok} detail={detail}")
    if name != "lmcache" and not ok:
        failed = True

print(f"[VERIFY] KORITH_ENABLE_AMF={os.environ.get('KORITH_ENABLE_AMF')}")
print(f"[VERIFY] KORITH_AMF_PATH={os.environ.get('KORITH_AMF_PATH')}")
print(f"[VERIFY] KORITH_VRAM_CACHE_GB={os.environ.get('KORITH_VRAM_CACHE_GB')}")
print(f"[VERIFY] KORITH_VRAM_POOL_GB={os.environ.get('KORITH_VRAM_POOL_GB')}")
print(f"[VERIFY] AXROPUS_ENABLE_LMCACHE={os.environ.get('AXROPUS_ENABLE_LMCACHE')}")

raise SystemExit(1 if failed else 0)
PY

if [[ "${RUN_AMF_LMCACHE_SMOKE:-0}" =~ ^(1|true|yes|on)$ ]]; then
  echo "[VERIFY] running AMF+LMCache smoke"
  "${PYTHON_BIN:-python3}" scripts/amf_lmcache_vllm_smoke.py \
    --model "$AMF_MODEL" \
    --max-model-len "${SMOKE_MAX_MODEL_LEN:-1024}" \
    --kv-cache-mb "${SMOKE_KV_CACHE_MB:-384}" \
    --lmcache-cpu-gb "${SMOKE_LMCACHE_CPU_GB:-1}" \
    --amf-path "${SMOKE_AMF_PATH:-/tmp/axropus-amf-stack-smoke}" \
    --prefix-repeat "${SMOKE_PREFIX_REPEAT:-80}" \
    --max-tokens "${SMOKE_MAX_TOKENS:-1}"
fi

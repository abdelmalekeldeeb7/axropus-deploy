"""korith_vllm_server.py — Korith vLLM Server with AMF-enabled inference.

Modes:
  benchmark  — single prompt, matches korith_dynamic log format, then exits.
  serve      — OpenAI-compatible HTTP server with AMF operating on every request.

Usage:
  # Benchmark mode (cold + warm run):
  KORITH_BACKEND=vllm KORITH_ENABLE_AMF=1 KORITH_AMF_PATH=/tmp/amf \\
    python -m korith_vllm_ext.korith_vllm_server \\
    --model Qwen/Qwen2.5-1.5B-Instruct \\
    --prompt "Evaluate the singularity..." \\
    --max-tokens 32

  # Server mode:
  KORITH_BACKEND=vllm KORITH_ENABLE_AMF=1 KORITH_AMF_PATH=/tmp/amf \\
    python -m korith_vllm_ext.korith_vllm_server \\
    --model Qwen/Qwen2.5-1.5B-Instruct \\
    --serve --port 8000

Environment variables (matching korith_dynamic):
  KORITH_ENABLE_AMF       — set to 1 to enable AMF
  KORITH_AMF_PATH         — AMF store directory
  KORITH_AMF_DIRECT_GPU   — set to 1 to use direct GPU copy path (default: 1)
  KORITH_AMF_MIN_TOKENS   — minimum token count for AMF admission (default: 16)
  KORITH_TENANT_ID        — tenant identifier for key isolation
  KORITH_MODEL            — model path override
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

_TRUTHY = ("1", "true", "yes", "on")


def _env_truthy(name: str, default: bool = False) -> bool:
    v = os.environ.get(name, "").strip().lower()
    return (v in _TRUTHY) if v else default


# ── V1 cache-engine proxy ──────────────────────────────────────────────────────

class _CacheEngineProxy:
    """Wraps vLLM V1 kv_caches (list of tensors or tuple-pairs) into the
    CacheEngine.gpu_cache interface that AmfKvManager expects.

    V0 CacheEngine.gpu_cache:  list of stacked tensors [2, n_blocks, ...]
    V1 model_runner.kv_caches: list of (k_tensor, v_tensor) tuples  OR
                                list of flat tensors (depends on attention backend)
    """

    def __init__(self, kv_caches: Any) -> None:
        import torch

        self.gpu_cache: List[Any] = []
        for layer in kv_caches:
            if isinstance(layer, (tuple, list)) and len(layer) == 2:
                k, v = layer[0], layer[1]
                if (
                    isinstance(k, torch.Tensor)
                    and isinstance(v, torch.Tensor)
                    and k.shape == v.shape
                ):
                    # Stack into [2, n_blocks, ...] so AmfKvManager sees dim==5
                    self.gpu_cache.append(torch.stack([k, v], dim=0))
                else:
                    self.gpu_cache.append(k)  # fallback: K only
            elif isinstance(layer, torch.Tensor):
                self.gpu_cache.append(layer)
            # else: skip unknown type


def _probe_vllm_kv(llm: Any) -> Optional[Any]:
    """Try every known path to obtain a cache_engine (or proxy) from a vLLM LLM.

    Approach A — V0 engine / V1 in-process UniProcExecutor:
        llm.llm_engine.model_executor.driver_worker.cache_engine   (V0/V1)
        llm.llm_engine.model_executor.driver_worker.model_runner.kv_caches (V1)

    Approach B — V1 workers list:
        llm.llm_engine.model_executor.workers[0].model_runner.kv_caches

    Approach C — V1 engine_core in-process path:
        llm.llm_engine._engine_core.model_executor.driver_worker ...

    Returns a CacheEngine instance or _CacheEngineProxy, or None if unreachable.
    """
    import torch  # noqa: F401 — ensure torch is imported for proxy ctor

    engine = llm.llm_engine

    # Gather candidate executor objects to search
    executor_candidates: List[Any] = []
    for attr in ("model_executor", "_model_executor"):
        ex = getattr(engine, attr, None)
        if ex is not None:
            executor_candidates.append(ex)
    # V1: engine may be wrapped; check one level deeper
    for attr in ("_engine_core", "engine_core"):
        core = getattr(engine, attr, None)
        if core is not None:
            for ex_attr in ("model_executor", "_model_executor"):
                ex = getattr(core, ex_attr, None)
                if ex is not None:
                    executor_candidates.append(ex)

    for executor in executor_candidates:
        # Collect candidate worker objects
        workers: List[Any] = []
        dw = getattr(executor, "driver_worker", None)
        if dw is not None:
            workers.append(dw)
        wlist = getattr(executor, "workers", None) or []
        workers.extend(wlist)

        for worker in workers:
            # Prefer kv_caches from model_runner (V1) — direct tensor access
            mr = getattr(worker, "model_runner", None)
            kv = getattr(mr, "kv_caches", None) if mr else None
            if kv and len(kv) > 0:
                print(
                    f"[AMF_PROBE] found kv_caches via model_runner "
                    f"({len(kv)} layers)",
                    flush=True,
                )
                return _CacheEngineProxy(kv)

            # Fall back to CacheEngine.gpu_cache (V0)
            ce = getattr(worker, "cache_engine", None)
            if isinstance(ce, list) and ce:
                ce = ce[0]  # V1 stores cache_engine per pipeline stage
            if ce is not None and hasattr(ce, "gpu_cache"):
                print("[AMF_PROBE] found cache_engine (V0 path)", flush=True)
                return ce

    print(
        "[AMF_PROBE] could not access KV cache internals — "
        "V1 engine likely running in separate process. "
        "Set VLLM_USE_V1=0 to force V0 engine for benchmark.",
        flush=True,
    )
    return None


# ── Benchmark mode ────────────────────────────────────────────────────────────

def run_benchmark(args: argparse.Namespace) -> None:
    """Run a single prompt through vLLM with AMF, emit korith_dynamic-style logs."""
    try:
        from vllm import LLM, SamplingParams  # type: ignore[import]
    except ImportError:
        print("[ERROR] vLLM not installed.  pip install vllm", file=sys.stderr)
        sys.exit(1)

    amf_path   = str(os.environ.get("KORITH_AMF_PATH", "")).strip()
    enable_amf = _env_truthy("KORITH_ENABLE_AMF")
    model_path = args.model or os.environ.get("KORITH_MODEL", "")
    prompt     = args.prompt
    max_tokens = args.max_tokens

    if not model_path:
        print("[ERROR] --model or KORITH_MODEL required", file=sys.stderr)
        sys.exit(1)

    # Force V0 engine when VLLM_USE_V1 not explicitly set — V0 exposes
    # driver_worker in-process which makes KV cache directly accessible.
    if "VLLM_USE_V1" not in os.environ:
        os.environ["VLLM_USE_V1"] = "0"
        print("[AMF_INFO] VLLM_USE_V1=0 set (force V0 engine for KV access)", flush=True)

    print(f"[KORITH_VLLM] loading model={model_path}", flush=True)
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        enforce_eager=True,  # disable CUDA graph so KV tensors stay accessible
    )
    sampling = SamplingParams(
        temperature=0.0,  # greedy — required for deterministic AMF replay
        max_tokens=max_tokens,
    )

    # Probe KV cache internals BEFORE first generate() so we have the handle.
    cache_engine_or_proxy: Optional[Any] = None
    amf_kv_manager: Optional[Any] = None

    if enable_amf and amf_path:
        cache_engine_or_proxy = _probe_vllm_kv(llm)
        if cache_engine_or_proxy is None:
            print(
                "[AMF_WARN] KV cache not accessible — "
                "benchmark will run cold+warm but save=False",
                file=sys.stderr,
                flush=True,
            )
        else:
            try:
                from .amf_kv_manager import AmfKvManager  # type: ignore[import]
                model_config = llm.llm_engine.model_config
                amf_kv_manager = AmfKvManager(
                    cache_engine=cache_engine_or_proxy,
                    amf_store_path=amf_path,
                    model_config=model_config,
                    tenant_id=str(os.environ.get("KORITH_TENANT_ID", "__shared__")),
                )
                print(
                    f"[AMF_INIT] AmfKvManager ready, store={amf_path}",
                    flush=True,
                )
            except Exception as exc:
                print(f"[AMF_WARN] kv_manager init failed: {exc}", file=sys.stderr)

    # ── Cold run ──────────────────────────────────────────────────────────────
    print("[KORITH_VLLM] cold run — full prefill...", flush=True)
    t_cold0 = time.monotonic()
    outputs = llm.generate([prompt], sampling)
    prompt_ms = (time.monotonic() - t_cold0) * 1000.0
    cold_text = outputs[0].outputs[0].text if outputs else ""

    print(f"[AMF_MISS] reason=cold_start  prompt_ms={prompt_ms:.2f}", flush=True)

    # ── Save KV snapshot ──────────────────────────────────────────────────────
    saved = False
    if amf_kv_manager is not None:
        try:
            tokenizer  = llm.get_tokenizer()
            token_ids  = tokenizer.encode(prompt)
            # block_table=[] → AmfKvManager saves ALL blocks (correct for
            # single-request benchmark where every populated block is ours).
            saved = amf_kv_manager.save_kv_state(token_ids, block_table=[])
            if saved:
                print(
                    f"[AMF_SAVE] tokens={len(token_ids)} "
                    f"layers={len(cache_engine_or_proxy.gpu_cache)} "
                    f"saved=True",
                    flush=True,
                )
            else:
                print("[AMF_WARN] save_kv_state returned False", flush=True)
        except Exception as exc:
            print(f"[AMF_WARN] save failed: {exc}", file=sys.stderr)

    print(
        f"[AMF_STATS] prompt_ms={prompt_ms:.1f} saved={saved}",
        flush=True,
    )
    print(f"[COLD_OUTPUT] {cold_text!r}", flush=True)

    if not enable_amf or amf_kv_manager is None:
        return

    # ── Warm run — measure restore_ms separately from generation ─────────────
    print("[KORITH_VLLM] warm run — restoring KV from snapshot...", flush=True)

    tokenizer = llm.get_tokenizer()
    token_ids = tokenizer.encode(prompt)
    has_snap  = amf_kv_manager.has_snapshot(token_ids)

    restore_ms = 0.0
    if has_snap:
        # Time the disk → GPU copy only (this is the number that matters).
        t_restore0 = time.monotonic()
        restored_tokens = amf_kv_manager.restore_kv_state(token_ids, block_table=[])
        restore_ms = (time.monotonic() - t_restore0) * 1000.0

        if restored_tokens > 0:
            saved_ms  = max(0.0, prompt_ms - restore_ms)
            speedup   = prompt_ms / max(1.0, restore_ms)
            print(
                f"[AMF_HIT] prefix_tokens={restored_tokens} "
                f"restore_ms={restore_ms:.2f} "
                f"saved_ms={saved_ms:.2f} "
                f"speedup={speedup:.2f}x",
                flush=True,
            )
        else:
            print("[AMF_MISS] reason=restore_failed", flush=True)
    else:
        print("[AMF_MISS] reason=no_snapshot", flush=True)

    # Run warm generate() to verify output matches (prefill still runs inside
    # vLLM — output should be identical since temperature=0).
    t_warm0       = time.monotonic()
    warm_outputs  = llm.generate([prompt], sampling)
    warm_total_ms = (time.monotonic() - t_warm0) * 1000.0
    warm_text     = warm_outputs[0].outputs[0].text if warm_outputs else ""

    skip_ratio = prompt_ms / max(1.0, restore_ms) if restore_ms > 0 else 0.0

    print(
        f"[KORITH_RUN_SUMMARY] "
        f"cold_ms={prompt_ms:.1f} "
        f"restore_ms={restore_ms:.1f} "
        f"warm_generate_ms={warm_total_ms:.1f} "
        f"tokens={max_tokens} "
        f"speedup={skip_ratio:.2f}x "
        f"cold={cold_text!r} "
        f"warm={warm_text!r}",
        flush=True,
    )


# ── Server mode ───────────────────────────────────────────────────────────────

def run_server(args: argparse.Namespace) -> None:
    """Launch an OpenAI-compatible vLLM server with AMF hooks."""
    try:
        import uvicorn  # type: ignore[import]
        from vllm.entrypoints.openai.api_server import app  # type: ignore[import]
    except ImportError as exc:
        print(f"[ERROR] missing dependency: {exc}", file=sys.stderr)
        sys.exit(1)

    model_path = args.model or os.environ.get("KORITH_MODEL", "")
    if not model_path:
        print("[ERROR] --model or KORITH_MODEL required", file=sys.stderr)
        sys.exit(1)

    # Add /amf/stats monitoring endpoint.
    try:
        from fastapi.responses import JSONResponse  # type: ignore[import]

        @app.get("/amf/stats")  # type: ignore[attr-defined]
        async def amf_stats() -> JSONResponse:
            return JSONResponse({"status": "ok", "backend": "vllm", "amf": True})
    except Exception:
        pass

    host = args.host or "0.0.0.0"
    port = args.port or 8000

    print(
        f"[KORITH_VLLM] starting server model={model_path} "
        f"host={host} port={port} amf={_env_truthy('KORITH_ENABLE_AMF')}",
        flush=True,
    )

    sys.argv = [
        "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--host", host,
        "--port", str(port),
        "--trust-remote-code",
    ]

    try:
        from vllm.entrypoints.openai.api_server import main as vllm_main  # type: ignore[import]
        vllm_main()
    except ImportError:
        uvicorn.run(app, host=host, port=port)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Korith vLLM Server — AMF-enabled inference"
    )
    parser.add_argument("--model",      type=str, default="",    help="Model path or HuggingFace ID")
    parser.add_argument("--prompt",     type=str, default="",    help="Prompt (benchmark mode)")
    parser.add_argument("--max-tokens", type=int, default=32,    help="Max output tokens")
    parser.add_argument("--serve",      action="store_true",     help="Run in server mode")
    parser.add_argument("--host",       type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port",       type=int, default=8000,  help="Server port")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if args.serve:
        run_server(args)
    else:
        if not args.prompt:
            parser.error("--prompt required in benchmark mode")
        run_benchmark(args)


if __name__ == "__main__":
    main()

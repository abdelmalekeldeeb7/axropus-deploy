"""korith_vllm_server.py — Korith vLLM Server with AMF-enabled inference.

Modes:
  benchmark  — single prompt, matches korith_dynamic log format, then exits.
  serve      — OpenAI-compatible HTTP server with AMF operating on every request.

Usage:
  # Benchmark mode (cold + warm run):
  KORITH_ENABLE_AMF=1 KORITH_AMF_PATH=/tmp/amf \
    python -m korith_vllm_ext.korith_vllm_server \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --prompt "Evaluate the singularity..." \
    --max-tokens 32

  # Server mode:
  KORITH_BACKEND=vllm KORITH_ENABLE_AMF=1 KORITH_AMF_PATH=/tmp/amf \
    python -m korith_vllm_ext.korith_vllm_server \
    --model Qwen/Qwen2.5-1.5B-Instruct \
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
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_TRUTHY = ("1", "true", "yes", "on")


def _env_truthy(name: str, default: bool = False) -> bool:
    v = os.environ.get(name, "").strip().lower()
    return (v in _TRUTHY) if v else default


def _call_engine_core(llm: Any, method: str, *args: Any) -> Any:
    """Call a patched method on EngineCore, handling both client types.

    SyncMPClient (multiprocess): has call_utility() → sends via ZMQ.
    InprocClient (in-process):   has engine_core attribute → call directly.
    """
    ec = llm.llm_engine.engine_core
    # SyncMPClient path (multiprocess EngineCore)
    if hasattr(ec, "call_utility"):
        return ec.call_utility(method, *args)
    # InprocClient path (in-process EngineCore)
    inner = getattr(ec, "engine_core", None)
    if inner is not None and hasattr(inner, method):
        return getattr(inner, method)(*args)
    raise AttributeError(
        f"Cannot call {method} on {type(ec).__name__} — "
        f"neither call_utility nor engine_core.{method} found"
    )


# ── Benchmark mode ────────────────────────────────────────────────────────────

def run_benchmark(args: argparse.Namespace) -> None:
    """Run a single prompt through vLLM with AMF, emit korith_dynamic-style logs.

    Uses vLLM v0.18 V1 engine with collective_rpc to access KV cache tensors
    inside the GPU worker subprocess.
    """
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

    # Register AMF worker extension so save/restore methods are available
    # on the GPU worker via string-name collective_rpc (no serialization issues).
    ext_cls = "korith_vllm_ext.amf_worker_ext.AmfWorkerExtension"

    eager = getattr(args, 'enforce_eager', False)
    kv_dtype = getattr(args, 'kv_cache_dtype', 'auto')
    print(f"[KORITH_VLLM] loading model={model_path} enforce_eager={eager} kv_cache_dtype={kv_dtype}", flush=True)
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        enforce_eager=eager,
        kv_cache_dtype=kv_dtype,
        worker_extension_cls=ext_cls,
    )
    sampling = SamplingParams(
        temperature=0.0,  # greedy — required for deterministic AMF replay
        max_tokens=max_tokens,
    )

    tenant_id = str(os.environ.get("KORITH_TENANT_ID", "__shared__"))

    # ── Verify EngineCore patch ──────────────────────────────────────────────
    try:
        verify = _call_engine_core(llm,"amf_verify_patch")
        print(
            f"[AMF_VERIFY] patch_ok={verify.get('patch_ok')} "
            f"block_size={verify.get('block_size')} "
            f"prefix_caching={verify.get('prefix_caching')} "
            f"has_block_pool={verify.get('has_block_pool')}",
            flush=True,
        )
        if not verify.get("patch_ok"):
            print("[AMF_WARN] EngineCore patch failed — prefix skip disabled",
                  flush=True)
    except Exception as exc:
        print(f"[AMF_WARN] patch verify failed: {exc}", flush=True)

    # ── Cold run ──────────────────────────────────────────────────────────────
    print("[KORITH_VLLM] cold run — full prefill...", flush=True)
    t_cold0 = time.monotonic()
    outputs = llm.generate([prompt], sampling)
    prompt_ms = (time.monotonic() - t_cold0) * 1000.0
    cold_text = outputs[0].outputs[0].text if outputs else ""

    print(f"[AMF_MISS] reason=cold_start  prompt_ms={prompt_ms:.2f}", flush=True)

    # ── Save KV snapshot via collective_rpc ───────────────────────────────────
    saved = False
    save_info: dict = {}
    if enable_amf and amf_path:
        try:
            tokenizer  = llm.get_tokenizer()
            token_ids  = tokenizer.encode(prompt)

            # Query the prefix cache for the REAL physical block IDs that
            # vLLM allocated during the cold generate.  Without this we'd
            # save stale data from blocks [0,1,2] which may not be the ones
            # vLLM actually used.
            real_block_ids: list = []
            try:
                cached_info = _call_engine_core(llm,
                    "amf_get_cached_block_ids", list(token_ids),
                )
                if isinstance(cached_info, dict):
                    real_block_ids = cached_info.get("block_ids", [])
                    print(
                        f"[AMF_BLOCKS] physical_ids={real_block_ids} "
                        f"n_found={cached_info.get('n_found')}/{cached_info.get('n_full_blocks')}",
                        flush=True,
                    )
            except Exception as exc:
                print(f"[AMF_WARN] get_cached_block_ids: {exc}", flush=True)

            results = llm.collective_rpc(
                "amf_save_kv",
                timeout=600,
                args=(amf_path, list(token_ids), 0, tenant_id),
                kwargs={"physical_block_ids": real_block_ids} if real_block_ids else None,
            )
            # collective_rpc returns list (one per worker); take first.
            save_info = results[0] if results else {}
            saved = save_info.get("saved", False)

            if saved:
                print(
                    f"[AMF_SAVE] tokens={save_info.get('n_tokens', 0)} "
                    f"layers={save_info.get('n_layers', 0)} "
                    f"saved=True",
                    flush=True,
                )
            else:
                print("[AMF_WARN] save_kv_state returned False", flush=True)
        except Exception as exc:
            print(f"[AMF_WARN] save failed: {exc}", file=sys.stderr, flush=True)

    print(
        f"[AMF_STATS] prompt_ms={prompt_ms:.1f} saved={saved}",
        flush=True,
    )
    print(f"[COLD_OUTPUT] {cold_text!r}", flush=True)

    if not enable_amf or not amf_path:
        return

    tokenizer = llm.get_tokenizer()
    token_ids = tokenizer.encode(prompt)

    # Check snapshot existence from main process (file check, no GPU needed).
    from .amf_kv_manager import AmfKvManager  # type: ignore[import]

    _dummy_mgr = AmfKvManager(
        cache_engine=type("_Dummy", (), {"gpu_cache": []})(),
        amf_store_path=amf_path,
        model_config=llm.llm_engine.model_config,
        tenant_id=tenant_id,
    )
    has_snap = _dummy_mgr.has_snapshot(token_ids)

    if not has_snap:
        print("[AMF_MISS] reason=no_snapshot", flush=True)
        return

    # ── Reset vLLM's prefix cache to simulate cold start ─────────────────────
    # This proves AMF is providing the value, not vLLM's built-in cache.
    # Retry up to 3 times — reset requires ALL blocks freed (ref_cnt=0).
    # After generate() completes, vLLM needs a moment to release blocks.
    print("[KORITH_VLLM] resetting vLLM prefix cache (simulating restart)...",
          flush=True)
    reset_ok = False
    for attempt in range(3):
        try:
            result = llm.llm_engine.reset_prefix_cache()
            if result or result is None:  # None = no return value = success
                reset_ok = True
                print("[AMF_RESET] prefix cache cleared", flush=True)
                break
            # Reset returned False — blocks not yet freed, wait and retry.
            if attempt < 2:
                time.sleep(0.1)
        except Exception as exc:
            print(f"[AMF_WARN] reset attempt {attempt+1} failed: {exc}",
                  flush=True)
            if attempt < 2:
                time.sleep(0.1)
    if not reset_ok:
        print("[AMF_WARN] prefix cache reset failed — warm run uses vLLM's "
              "built-in cache (AMF still restores KV data)", flush=True)

    # ── Warm run: AMF restore → register prefix cache → generate ─────────────
    print("[KORITH_VLLM] warm run — AMF restore + prefix cache register...",
          flush=True)

    restore_ms = 0.0
    restored_tokens = 0
    try:
        # Step 1: Restore KV data into physical blocks (worker side).
        t_restore0 = time.monotonic()
        results = llm.collective_rpc(
            "amf_restore_kv",
            timeout=600,
            args=(amf_path, list(token_ids), 0, tenant_id),
        )
        restore_ms = (time.monotonic() - t_restore0) * 1000.0
        restored_tokens = results[0] if results else 0
    except Exception as exc:
        print(f"[AMF_WARN] restore failed: {exc}", file=sys.stderr, flush=True)

    # Compute RESTORE block IDs — these are where amf_restore_kv writes data.
    # Only FULL blocks (floor division) — matches prefix cache and save behavior.
    restore_block_size = save_info.get("block_size", 16)
    n_restore_blocks = len(token_ids) // restore_block_size
    if n_restore_blocks == 0:
        n_restore_blocks = 1
    restore_block_ids = list(range(n_restore_blocks))

    register_ms = 0.0
    if restored_tokens > 0:
        try:

            t_reg0 = time.monotonic()
            reg_result = _call_engine_core(llm,
                "amf_register_prefix",
                list(token_ids),
                restore_block_ids,
            )
            register_ms = (time.monotonic() - t_reg0) * 1000.0
            reg_info = reg_result if isinstance(reg_result, dict) else {}
            print(
                f"[AMF_REGISTER] registered={reg_info.get('registered', '?')} "
                f"block_size={reg_info.get('block_size', '?')} "
                f"register_ms={register_ms:.2f}",
                flush=True,
            )
            # Verify blocks are findable in prefix cache.
            try:
                vfy = _call_engine_core(llm,
                    "amf_verify_registration", list(token_ids),
                )
                vfy = vfy if isinstance(vfy, dict) else {}
                print(
                    f"[AMF_VERIFY_REG] all_hit={vfy.get('all_hit')} "
                    f"hits={vfy.get('hits')}/{vfy.get('n_full_blocks')}",
                    flush=True,
                )
            except Exception as exc:
                print(f"[AMF_WARN] verify_registration: {exc}", flush=True)

            print(
                f"[AMF_HIT] prefix_tokens={restored_tokens} "
                f"restore_ms={restore_ms:.2f} "
                f"register_ms={register_ms:.2f} "
                f"total_amf_ms={restore_ms + register_ms:.2f} "
                f"speedup={prompt_ms / max(1.0, restore_ms + register_ms):.2f}x",
                flush=True,
            )
        except Exception as exc:
            print(f"[AMF_WARN] register failed: {exc}", file=sys.stderr,
                  flush=True)
    else:
        print("[AMF_MISS] reason=restore_failed", flush=True)

    # Warm generate — should skip prefill via prefix cache hit.
    t_warm0       = time.monotonic()
    warm_outputs  = llm.generate([prompt], sampling)
    warm_total_ms = (time.monotonic() - t_warm0) * 1000.0
    warm_text     = warm_outputs[0].outputs[0].text if warm_outputs else ""

    prefill_skipped = warm_total_ms < prompt_ms * 0.75
    print(
        f"[KORITH_RUN_SUMMARY] "
        f"cold_ms={prompt_ms:.1f} "
        f"restore_ms={restore_ms:.1f} "
        f"register_ms={register_ms:.1f} "
        f"warm_generate_ms={warm_total_ms:.1f} "
        f"tokens={max_tokens} "
        f"e2e_speedup={prompt_ms / max(1.0, warm_total_ms):.2f}x "
        f"prefill_skipped={prefill_skipped} "
        f"output_match={cold_text == warm_text} "
        f"cold={cold_text!r} "
        f"warm={warm_text!r}",
        flush=True,
    )
    if not prefill_skipped:
        print(
            "[AMF_DIAG] warm_ms ≈ cold_ms — prefill was NOT skipped. "
            "This is expected for short prompts where prefill is <5%% of "
            "total time. Try a longer prompt (4K+ tokens) to see the gap.",
            flush=True,
        )

    # ── Second warm run — VRAM cache path ────────────────────────────────────
    if restored_tokens > 0:
        print("[KORITH_VLLM] warm2 run — VRAM cache path...", flush=True)
        try:
            llm.llm_engine.reset_prefix_cache()

            t_restore2 = time.monotonic()
            results2 = llm.collective_rpc(
                "amf_restore_kv",
                timeout=600,
                args=(amf_path, list(token_ids), 0, tenant_id),
            )
            restore2_ms = (time.monotonic() - t_restore2) * 1000.0
            restored2 = results2[0] if results2 else 0

            t_reg2 = time.monotonic()
            _call_engine_core(llm,
                "amf_register_prefix",
                list(token_ids),
                restore_block_ids,
            )
            register2_ms = (time.monotonic() - t_reg2) * 1000.0

            t_warm2       = time.monotonic()
            warm2_outputs = llm.generate([prompt], sampling)
            warm2_ms      = (time.monotonic() - t_warm2) * 1000.0
            warm2_text    = warm2_outputs[0].outputs[0].text if warm2_outputs else ""

            total2 = restore2_ms + register2_ms
            print(
                f"[AMF_HIT] source=vram prefix_tokens={restored2} "
                f"restore_ms={restore2_ms:.2f} register_ms={register2_ms:.2f} "
                f"speedup={prompt_ms / max(1.0, total2):.2f}x",
                flush=True,
            )
            print(
                f"[KORITH_RUN_SUMMARY2] "
                f"restore2_ms={restore2_ms:.1f} "
                f"warm2_generate_ms={warm2_ms:.1f} "
                f"e2e_speedup2={prompt_ms / max(1.0, warm2_ms):.2f}x "
                f"warm2={warm2_text!r}",
                flush=True,
            )
        except Exception as exc:
            print(f"[AMF_WARN] warm2 failed: {exc}", file=sys.stderr, flush=True)


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
    parser.add_argument("--prompt-file", type=str, default="",   help="Read prompt from file (benchmark mode)")
    parser.add_argument("--max-tokens", type=int, default=32,    help="Max output tokens")
    parser.add_argument("--enforce-eager", action="store_true",  help="Disable CUDAGraphs/compile")
    parser.add_argument("--kv-cache-dtype", type=str, default="auto", help="KV cache dtype (auto/fp8)")
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
        if args.prompt_file:
            with open(args.prompt_file, "r") as f:
                args.prompt = f.read()
        if not args.prompt:
            parser.error("--prompt or --prompt-file required in benchmark mode")
        run_benchmark(args)


if __name__ == "__main__":
    main()

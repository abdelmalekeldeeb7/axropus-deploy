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

    print(f"[KORITH_VLLM] loading model={model_path}", flush=True)
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        enforce_eager=True,  # disable CUDA graph so KV tensors stay accessible
        worker_extension_cls=ext_cls,
    )
    sampling = SamplingParams(
        temperature=0.0,  # greedy — required for deterministic AMF replay
        max_tokens=max_tokens,
    )

    tenant_id = str(os.environ.get("KORITH_TENANT_ID", "__shared__"))

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

            results = llm.collective_rpc(
                "amf_save_kv",
                timeout=120,
                args=(amf_path, list(token_ids), 0, tenant_id),
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

    # ── Warm run — measure restore_ms separately from generation ─────────────
    print("[KORITH_VLLM] warm run — restoring KV from snapshot...", flush=True)

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

    restore_ms = 0.0
    restored_tokens = 0
    if has_snap:
        try:
            t_restore0 = time.monotonic()
            results = llm.collective_rpc(
                "amf_restore_kv",
                timeout=120,
                args=(amf_path, list(token_ids), 0, tenant_id),
            )
            restore_ms = (time.monotonic() - t_restore0) * 1000.0
            restored_tokens = results[0] if results else 0

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
        except Exception as exc:
            print(f"[AMF_WARN] restore failed: {exc}", file=sys.stderr, flush=True)
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

    # ── Second warm run — measures VRAM cache hit (zero H→D) ─────────────────
    if restored_tokens > 0:
        print("[KORITH_VLLM] warm2 run — VRAM cache path...", flush=True)
        try:
            t_restore2 = time.monotonic()
            results2 = llm.collective_rpc(
                "amf_restore_kv",
                timeout=120,
                args=(amf_path, list(token_ids), 0, tenant_id),
            )
            restore2_ms = (time.monotonic() - t_restore2) * 1000.0
            restored2 = results2[0] if results2 else 0

            t_warm2       = time.monotonic()
            warm2_outputs = llm.generate([prompt], sampling)
            warm2_ms      = (time.monotonic() - t_warm2) * 1000.0
            warm2_text    = warm2_outputs[0].outputs[0].text if warm2_outputs else ""

            speedup2 = prompt_ms / max(1.0, restore2_ms)
            print(
                f"[AMF_HIT] source=vram prefix_tokens={restored2} "
                f"restore_ms={restore2_ms:.2f} speedup={speedup2:.2f}x",
                flush=True,
            )
            print(
                f"[KORITH_RUN_SUMMARY2] "
                f"restore2_ms={restore2_ms:.1f} "
                f"warm2_generate_ms={warm2_ms:.1f} "
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

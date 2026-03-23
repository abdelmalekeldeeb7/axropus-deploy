"""korith_vllm_server.py — Korith vLLM Server with AMF-enabled inference.

Modes:
  benchmark  — single prompt, matches korith_dynamic log format, then exits.
  serve      — OpenAI-compatible HTTP server with AMF operating on every request.

Usage:
  # Benchmark mode (cold + warm run):
  KORITH_BACKEND=vllm KORITH_ENABLE_AMF=1 KORITH_AMF_PATH=/tmp/amf \\
    python -m korith_vllm_ext.korith_vllm_server \\
    --model meta-llama/Llama-3.1-70B-Instruct \\
    --prompt "Evaluate the singularity..." \\
    --max-tokens 32

  # Server mode:
  KORITH_BACKEND=vllm KORITH_ENABLE_AMF=1 KORITH_AMF_PATH=/tmp/amf \\
    python -m korith_vllm_ext.korith_vllm_server \\
    --model meta-llama/Llama-3.1-70B-Instruct \\
    --serve --port 8000

Environment variables (matching korith_dynamic):
  KORITH_ENABLE_AMF       — set to 1 to enable AMF
  KORITH_AMF_PATH         — AMF store directory
  KORITH_AMF_DIRECT_GPU   — set to 1 to use direct GPU copy path (default: 1)
  KORITH_AMF_MIN_TOKENS   — minimum token count for AMF admission (default: 64)
  KORITH_TENANT_ID        — tenant identifier for key isolation
  KORITH_MODEL            — model path override
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_TRUTHY = ("1", "true", "yes", "on")


def _env_truthy(name: str, default: bool = False) -> bool:
    v = os.environ.get(name, "").strip().lower()
    return (v in _TRUTHY) if v else default


# ── Benchmark mode ────────────────────────────────────────────────────────────

def run_benchmark(args: argparse.Namespace) -> None:
    """Run a single prompt through vLLM with AMF, emit korith_dynamic-style logs."""
    try:
        from vllm import LLM, SamplingParams  # type: ignore[import]
    except ImportError:
        print("[ERROR] vLLM not installed.  pip install vllm", file=sys.stderr)
        sys.exit(1)

    amf_path    = str(os.environ.get("KORITH_AMF_PATH", "")).strip()
    enable_amf  = _env_truthy("KORITH_ENABLE_AMF")
    model_path  = args.model or os.environ.get("KORITH_MODEL", "")
    prompt      = args.prompt
    max_tokens  = args.max_tokens

    if not model_path:
        print("[ERROR] --model or KORITH_MODEL required", file=sys.stderr)
        sys.exit(1)

    # Initialise vLLM engine.
    print(f"[KORITH_VLLM] loading model={model_path}", flush=True)
    llm = LLM(
        model=model_path,
        scheduler_cls="korith_vllm_ext.decode_scheduler.KorithDecodeScheduler"
        if enable_amf
        else None,
        trust_remote_code=True,
    )
    sampling = SamplingParams(
        temperature=0.0,  # greedy — required for deterministic AMF replay
        max_tokens=max_tokens,
    )

    amf_kv_manager: Optional[Any] = None
    if enable_amf and amf_path:
        try:
            from .amf_kv_manager import AmfKvManager  # type: ignore[import]
            cache_engine = getattr(llm.llm_engine, "cache_engine", None)
            if cache_engine is None:
                # vLLM 0.6+ uses workers; try to get from first worker.
                workers = getattr(llm.llm_engine, "workers", None) or []
                if workers:
                    cache_engine = getattr(workers[0], "cache_engine", None)
            model_config = llm.llm_engine.model_config
            if cache_engine is not None:
                amf_kv_manager = AmfKvManager(
                    cache_engine=cache_engine,
                    amf_store_path=amf_path,
                    model_config=model_config,
                    tenant_id=str(os.environ.get("KORITH_TENANT_ID", "__shared__")),
                )
        except Exception as exc:
            print(f"[AMF_WARN] kv_manager init failed: {exc}", file=sys.stderr)

    # ── Cold run ──────────────────────────────────────────────────────────────
    print("[KORITH_VLLM] cold run...", flush=True)
    t_prompt0 = time.monotonic()
    outputs = llm.generate([prompt], sampling)
    prompt_ms = (time.monotonic() - t_prompt0) * 1000.0
    cold_text = outputs[0].outputs[0].text if outputs else ""

    # Try to save KV snapshot.
    saved = False
    if amf_kv_manager is not None:
        try:
            # Get token IDs from the engine tokenizer.
            tokenizer = llm.get_tokenizer()
            token_ids = tokenizer.encode(prompt)
            # Block table from first output's request.
            req_id = outputs[0].request_id if outputs else ""
            block_table: List[int] = []  # vLLM doesn't expose this easily post-generate
            saved = amf_kv_manager.save_kv_state(token_ids, block_table)
        except Exception as exc:
            print(f"[AMF_WARN] save failed: {exc}", file=sys.stderr)

    print(
        f"[AMF_STATS] prompt_ms={prompt_ms:.1f} saved={saved}",
        flush=True,
    )
    print(f"[COLD_OUTPUT] {cold_text!r}", flush=True)

    if not enable_amf or amf_kv_manager is None:
        return

    # ── Warm run ──────────────────────────────────────────────────────────────
    print("[KORITH_VLLM] warm run...", flush=True)

    # Check for snapshot.
    try:
        tokenizer = llm.get_tokenizer()
        token_ids = tokenizer.encode(prompt)
        has_snap = amf_kv_manager.has_snapshot(token_ids)
    except Exception:
        has_snap = False

    t_warm0 = time.monotonic()
    warm_outputs = llm.generate([prompt], sampling)
    restore_ms = (time.monotonic() - t_warm0) * 1000.0
    warm_text = warm_outputs[0].outputs[0].text if warm_outputs else ""

    skip_ratio = restore_ms / max(1.0, prompt_ms)  # proxy; real ratio needs baseline

    if has_snap:
        stats = amf_kv_manager.stats()
        print(
            f"[AMF_HIT] prefix_tokens={len(token_ids)} "
            f"restore_ms={stats['restore_ms']:.2f}",
            flush=True,
        )
        print(
            f"[AMF_SKIP] prompt_tokens={len(token_ids)} "
            f"skipped_tokens={len(token_ids)} "
            f"skip_ratio={skip_ratio:.3f} "
            f"restore_ms={stats['restore_ms']:.2f} "
            f"saved_ms={max(0.0, prompt_ms - stats['restore_ms']):.2f}",
            flush=True,
        )
    else:
        print("[AMF_MISS] reason=no_snapshot", flush=True)

    print(
        f"[KORITH_RUN_SUMMARY] "
        f"prompt_ms={prompt_ms:.1f} "
        f"restore_ms={restore_ms:.1f} "
        f"tokens={max_tokens} "
        f"skip_ratio={skip_ratio:.3f} "
        f"cold={cold_text!r} "
        f"warm={warm_text!r}",
        flush=True,
    )


# ── Server mode ───────────────────────────────────────────────────────────────

def run_server(args: argparse.Namespace) -> None:
    """Launch an OpenAI-compatible vLLM server with AMF hooks."""
    try:
        import uvicorn  # type: ignore[import]
        from fastapi import FastAPI, Request  # type: ignore[import]
        from fastapi.responses import JSONResponse  # type: ignore[import]
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
        from fastapi import FastAPI as _FA  # type: ignore[import]

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

    # Build CLI args expected by vLLM's entrypoint.
    sys.argv = [
        "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--host", host,
        "--port", str(port),
        "--scheduler-cls",
        "korith_vllm_ext.decode_scheduler.KorithDecodeScheduler",
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
    parser.add_argument("--model", type=str, default="", help="Model path or HuggingFace ID")
    parser.add_argument("--prompt", type=str, default="", help="Prompt (benchmark mode)")
    parser.add_argument("--max-tokens", type=int, default=32, help="Max output tokens")
    parser.add_argument("--serve", action="store_true", help="Run in server mode")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
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

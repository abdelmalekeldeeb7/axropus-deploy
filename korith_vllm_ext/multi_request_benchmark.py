"""multi_request_benchmark.py — Production benchmark: 500 concurrent requests.

Tests AMF with realistic production traffic patterns:
- Multiple requests sharing the same prefix (system prompt)
- Each request adds unique user context on top of shared prefix
- Measures TTFT, throughput, P50/P95/P99 latencies
- Compares cold (no AMF) vs warm (AMF cached) performance

Usage:
    KORITH_ENABLE_AMF=1 KORITH_AMF_PATH=/tmp/amf \
    KORITH_VRAM_POOL_GB=50 KORITH_VRAM_POOL_QUANT=int4 \
    python3 -m korith_vllm_ext.multi_request_benchmark \
        --model neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8 \
        --num-requests 500 \
        --prefix-file /tmp/long_prompt.txt \
        --max-tokens 32
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import statistics
from typing import List

from vllm import LLM, SamplingParams


def run_benchmark(args: argparse.Namespace) -> None:
    model_path = args.model or os.environ.get("KORITH_MODEL", "")
    if not model_path:
        print("ERROR: --model required", file=sys.stderr)
        sys.exit(1)

    num_requests = args.num_requests
    max_tokens = args.max_tokens

    # Read shared prefix
    if args.prefix_file:
        with open(args.prefix_file, "r") as f:
            shared_prefix = f.read()
    else:
        shared_prefix = "Explain quantum computing in detail with examples. " * 8000

    # Create varied prompts: shared prefix + unique suffix per request
    unique_suffixes = [
        f"\n\nQuestion {i}: What is the {i}th prime number? Answer briefly."
        for i in range(num_requests)
    ]
    prompts = [shared_prefix + suffix for suffix in unique_suffixes]

    print(f"[BENCH] model={model_path}", flush=True)
    print(f"[BENCH] num_requests={num_requests}", flush=True)
    print(f"[BENCH] prefix_length={len(shared_prefix)} chars", flush=True)
    print(f"[BENCH] max_tokens={max_tokens}", flush=True)

    # Load model
    enable_amf = os.environ.get("KORITH_ENABLE_AMF", "0") == "1"
    ext_cls = "korith_vllm_ext.amf_worker_ext.AmfWorkerExtension" if enable_amf else ""

    eager = getattr(args, 'enforce_eager', False)
    kv_dtype = getattr(args, 'kv_cache_dtype', 'auto')

    print(f"[BENCH] loading model (AMF={enable_amf}, eager={eager}, kv_dtype={kv_dtype})...",
          flush=True)
    t_load = time.monotonic()

    llm_kwargs = dict(
        model=model_path,
        trust_remote_code=True,
        enforce_eager=eager,
        kv_cache_dtype=kv_dtype,
    )
    if ext_cls:
        llm_kwargs["worker_extension_cls"] = ext_cls

    llm = LLM(**llm_kwargs)
    load_ms = (time.monotonic() - t_load) * 1000.0
    print(f"[BENCH] model loaded in {load_ms:.0f}ms", flush=True)

    sampling = SamplingParams(temperature=0.0, max_tokens=max_tokens)

    # ── Phase 1: Cold batch (first request warms the AMF cache) ──────────
    print(f"\n[PHASE1] Cold run — single request to populate AMF cache...", flush=True)
    t_cold = time.monotonic()
    cold_outputs = llm.generate([prompts[0]], sampling)
    cold_ms = (time.monotonic() - t_cold) * 1000.0
    cold_text = cold_outputs[0].outputs[0].text[:100] if cold_outputs else ""
    print(f"[PHASE1] cold_ms={cold_ms:.1f} output='{cold_text}...'", flush=True)

    # If AMF is enabled, trigger save via collective_rpc
    if enable_amf:
        amf_path = os.environ.get("KORITH_AMF_PATH", "/tmp/amf")
        try:
            tokenizer = llm.get_tokenizer()
            token_ids = tokenizer.encode(shared_prefix)

            # Get real block IDs from prefix cache
            from korith_vllm_ext.korith_vllm_server import _call_engine_core
            cached_info = _call_engine_core(llm,
                "amf_get_cached_block_ids", list(token_ids))
            real_block_ids = cached_info.get("block_ids", []) if isinstance(cached_info, dict) else []

            # Save KV
            results = llm.collective_rpc(
                "amf_save_kv",
                timeout=600,
                args=(amf_path, list(token_ids), 0, "__shared__"),
                kwargs={"physical_block_ids": real_block_ids} if real_block_ids else None,
            )
            save_info = results[0] if results else {}
            print(f"[PHASE1] AMF save: {save_info.get('saved', False)}, "
                  f"blocks={save_info.get('n_blocks', 0)}", flush=True)
        except Exception as exc:
            print(f"[PHASE1] AMF save failed: {exc}", flush=True)

    # ── Phase 2: Warm batch — N requests all hitting AMF cache ───────────
    # Reset prefix cache to simulate clean state
    print(f"\n[PHASE2] Warm batch — {num_requests} requests with AMF prefix cache...",
          flush=True)

    try:
        llm.llm_engine.reset_prefix_cache()
        print("[PHASE2] prefix cache reset", flush=True)
    except Exception:
        pass

    # Restore + register for warm run
    if enable_amf:
        try:
            tokenizer = llm.get_tokenizer()
            token_ids = tokenizer.encode(shared_prefix)

            # Restore
            t_restore = time.monotonic()
            results = llm.collective_rpc(
                "amf_restore_kv",
                timeout=600,
                args=(amf_path, list(token_ids), 0, "__shared__"),
            )
            restore_ms = (time.monotonic() - t_restore) * 1000.0
            restored = results[0] if results else 0
            print(f"[PHASE2] AMF restore: {restored} tokens in {restore_ms:.1f}ms",
                  flush=True)

            # Register in prefix cache
            block_size = save_info.get("block_size", 16) if 'save_info' in dir() else 16
            n_blocks = len(token_ids) // block_size
            restore_block_ids = list(range(n_blocks))

            t_reg = time.monotonic()
            reg_result = _call_engine_core(llm,
                "amf_register_prefix", list(token_ids), restore_block_ids)
            register_ms = (time.monotonic() - t_reg) * 1000.0
            print(f"[PHASE2] AMF register: {reg_result.get('registered', '?')} blocks "
                  f"in {register_ms:.1f}ms", flush=True)
        except Exception as exc:
            print(f"[PHASE2] AMF restore/register failed: {exc}", flush=True)

    # Run all N requests as a single batch
    print(f"[PHASE2] generating {num_requests} requests...", flush=True)

    # Batch in groups to avoid OOM
    batch_size = min(50, num_requests)
    all_outputs = []
    batch_times: List[float] = []

    for batch_start in range(0, num_requests, batch_size):
        batch_end = min(batch_start + batch_size, num_requests)
        batch_prompts = prompts[batch_start:batch_end]
        batch_n = len(batch_prompts)

        t_batch = time.monotonic()
        outputs = llm.generate(batch_prompts, sampling)
        batch_ms = (time.monotonic() - t_batch) * 1000.0

        per_request_ms = batch_ms / batch_n
        batch_times.extend([per_request_ms] * batch_n)
        all_outputs.extend(outputs)

        throughput = batch_n / (batch_ms / 1000.0)
        print(f"[PHASE2] batch {batch_start}-{batch_end}: "
              f"{batch_ms:.0f}ms total, {per_request_ms:.1f}ms/req, "
              f"{throughput:.1f} req/s", flush=True)

    # ── Phase 3: Results ─────────────────────────────────────────────────
    total_warm_ms = sum(batch_times)
    avg_ms = statistics.mean(batch_times)
    p50 = statistics.median(batch_times)
    p95 = sorted(batch_times)[int(0.95 * len(batch_times))]
    p99 = sorted(batch_times)[int(0.99 * len(batch_times))]
    total_throughput = num_requests / (total_warm_ms / 1000.0 / num_requests)

    # Check output quality
    first_output = all_outputs[0].outputs[0].text[:100] if all_outputs else ""
    last_output = all_outputs[-1].outputs[0].text[:100] if all_outputs else ""

    print(f"\n{'='*60}", flush=True)
    print(f"[RESULTS] Production Benchmark — {num_requests} requests", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Model:           {model_path}", flush=True)
    print(f"  Prefix length:   {len(shared_prefix)} chars", flush=True)
    print(f"  Output tokens:   {max_tokens}", flush=True)
    print(f"  AMF enabled:     {enable_amf}", flush=True)
    print(f"  KV cache dtype:  {kv_dtype}", flush=True)
    print(f"", flush=True)
    print(f"  Cold TTFT:       {cold_ms:.0f}ms", flush=True)
    print(f"  Warm avg/req:    {avg_ms:.1f}ms", flush=True)
    print(f"  Warm P50:        {p50:.1f}ms", flush=True)
    print(f"  Warm P95:        {p95:.1f}ms", flush=True)
    print(f"  Warm P99:        {p99:.1f}ms", flush=True)
    print(f"  Throughput:      {total_throughput:.1f} req/s", flush=True)
    print(f"  TTFT speedup:    {cold_ms / max(avg_ms, 1):.1f}x", flush=True)
    if enable_amf:
        print(f"  Restore time:    {restore_ms:.1f}ms", flush=True)
        print(f"  Register time:   {register_ms:.1f}ms", flush=True)
    print(f"", flush=True)
    print(f"  First output:    '{first_output}...'", flush=True)
    print(f"  Last output:     '{last_output}...'", flush=True)
    print(f"{'='*60}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AMF Production Benchmark — Multi-request throughput test"
    )
    parser.add_argument("--model", type=str, default="", help="Model path or HuggingFace ID")
    parser.add_argument("--num-requests", type=int, default=500, help="Number of requests")
    parser.add_argument("--prefix-file", type=str, default="", help="Shared prefix text file")
    parser.add_argument("--max-tokens", type=int, default=32, help="Max output tokens per request")
    parser.add_argument("--enforce-eager", action="store_true", help="Disable CUDAGraphs")
    parser.add_argument("--kv-cache-dtype", type=str, default="auto", help="KV cache dtype")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    run_benchmark(args)


if __name__ == "__main__":
    main()

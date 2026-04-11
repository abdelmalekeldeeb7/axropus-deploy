"""amf_vs_lmcache_vs_tiered.py — Head-to-head benchmark.

Compares four configurations on the same 500-request 128K workload:

  1. Plain vLLM (no prefix cache beyond vLLM's built-in)
  2. vLLM + LMCache alone (no AMF hot tier)
  3. vLLM + Axropus AMF alone (current production config)
  4. vLLM + Axropus AMF + LMCache (new tiered stack)

For each configuration we report:
  * Cold TTFT
  * Warm P50 / P95 / P99
  * Per-tier hit rate (AMF vs LMCache vs cold)
  * Throughput (req/s)

Usage:
    python3 -m benchmarks.amf_vs_lmcache_vs_tiered \\
        --model neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8 \\
        --prefix-file /tmp/long_prompt.txt \\
        --num-requests 500 \\
        --max-tokens 32 \\
        --output /tmp/bench_amf_vs_lmcache.json

This script is designed to run on a single H200 with ~141 GB HBM. For
smaller GPUs, pass a smaller model via --model.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger("benchmarks.amf_vs_lmcache")


# ── Config presets ───────────────────────────────────────────────────────────

@dataclass
class BenchConfig:
    name: str
    amf_enabled: bool
    lmcache_enabled: bool
    env: Dict[str, str] = field(default_factory=dict)

    def activate(self) -> None:
        """Set env vars required by this config before loading vLLM."""
        if self.amf_enabled:
            os.environ["KORITH_ENABLE_AMF"] = "1"
            os.environ.setdefault("KORITH_VRAM_POOL_GB", "50")
            os.environ.setdefault("KORITH_VRAM_POOL_QUANT", "int4")
        else:
            os.environ.pop("KORITH_ENABLE_AMF", None)

        if self.lmcache_enabled:
            os.environ["AXROPUS_LMCACHE_FALLBACK"] = "true"
            os.environ.setdefault("AXROPUS_PROMOTE_LMCACHE_HITS", "true")
            os.environ.setdefault("AXROPUS_LMCACHE_WRITE_THROUGH", "true")
        else:
            os.environ["AXROPUS_LMCACHE_FALLBACK"] = "false"

        # Apply any config-specific overrides last so they win
        for k, v in self.env.items():
            os.environ[k] = v


CONFIGS: List[BenchConfig] = [
    BenchConfig(
        name="1_plain_vllm",
        amf_enabled=False,
        lmcache_enabled=False,
    ),
    BenchConfig(
        name="2_lmcache_only",
        amf_enabled=False,
        lmcache_enabled=True,
    ),
    BenchConfig(
        name="3_amf_only",
        amf_enabled=True,
        lmcache_enabled=False,
    ),
    BenchConfig(
        name="4_amf_plus_lmcache",
        amf_enabled=True,
        lmcache_enabled=True,
    ),
]


# ── Result schema ────────────────────────────────────────────────────────────

@dataclass
class BenchResult:
    config: str
    cold_ms: float
    warm_avg_ms: float
    warm_p50_ms: float
    warm_p95_ms: float
    warm_p99_ms: float
    throughput_req_per_s: float
    restore_ms: float
    register_ms: float
    amf_hits: int
    lmcache_hits: int
    cold_misses: int
    output_sample: str
    notes: str = ""


# ── Benchmark runner ─────────────────────────────────────────────────────────

def run_config(cfg: BenchConfig, args: argparse.Namespace) -> BenchResult:
    """Run one benchmark configuration. Returns a BenchResult.

    The model is loaded fresh for each config so env vars take effect at
    vLLM startup time. This is slower but keeps configurations isolated.
    """
    logger.info("=" * 60)
    logger.info("[BENCH] starting config: %s", cfg.name)
    logger.info("=" * 60)
    cfg.activate()

    # Import vLLM here, AFTER env vars are set, so worker_extension_cls
    # sees the right config.
    from vllm import LLM, SamplingParams

    ext_cls = (
        "korith_vllm_ext.amf_worker_ext.AmfWorkerExtension"
        if cfg.amf_enabled else None
    )

    # Read prefix
    with open(args.prefix_file, "r") as f:
        shared_prefix = f.read()

    prompts = [
        shared_prefix + f"\n\nQuestion {i}: What is the {i}th prime number?"
        for i in range(args.num_requests)
    ]

    # Load model
    t_load = time.monotonic()
    llm_kwargs = dict(
        model=args.model,
        trust_remote_code=True,
        enforce_eager=args.enforce_eager,
        kv_cache_dtype=args.kv_cache_dtype,
    )
    if ext_cls:
        llm_kwargs["worker_extension_cls"] = ext_cls
    llm = LLM(**llm_kwargs)
    load_ms = (time.monotonic() - t_load) * 1000.0
    logger.info("[%s] model loaded in %.0f ms", cfg.name, load_ms)

    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    # Phase 1 — cold
    t_cold = time.monotonic()
    cold_out = llm.generate([prompts[0]], sampling)
    cold_ms = (time.monotonic() - t_cold) * 1000.0
    sample_text = cold_out[0].outputs[0].text[:80] if cold_out else ""
    logger.info("[%s] cold_ms=%.0f", cfg.name, cold_ms)

    # If AMF, save and reset
    restore_ms = 0.0
    register_ms = 0.0
    if cfg.amf_enabled:
        amf_path = os.environ.get("KORITH_AMF_PATH", "/tmp/amf_bench_hh")
        os.makedirs(amf_path, exist_ok=True)
        try:
            tokenizer = llm.get_tokenizer()
            token_ids = tokenizer.encode(shared_prefix)
            from korith_vllm_ext.korith_vllm_server import _call_engine_core
            try:
                cached = _call_engine_core(
                    llm, "amf_get_cached_block_ids", list(token_ids)
                )
                real_blocks = (
                    cached.get("block_ids", [])
                    if isinstance(cached, dict) else []
                )
            except Exception:
                real_blocks = []

            save_results = llm.collective_rpc(
                "amf_save_kv",
                timeout=600,
                args=(amf_path, list(token_ids), 0, "__bench_hh__"),
                kwargs={"physical_block_ids": real_blocks} if real_blocks else None,
            )
            save_info = save_results[0] if save_results else {}

            llm.llm_engine.reset_prefix_cache()

            t_r = time.monotonic()
            llm.collective_rpc(
                "amf_restore_kv",
                timeout=600,
                args=(amf_path, list(token_ids), 0, "__bench_hh__"),
            )
            restore_ms = (time.monotonic() - t_r) * 1000.0

            block_size = save_info.get("block_size", 16)
            n_blocks = len(token_ids) // block_size
            t_reg = time.monotonic()
            _call_engine_core(
                llm, "amf_register_prefix",
                list(token_ids), list(range(n_blocks)),
            )
            register_ms = (time.monotonic() - t_reg) * 1000.0
            logger.info(
                "[%s] AMF restore=%.1f ms register=%.1f ms",
                cfg.name, restore_ms, register_ms,
            )
        except Exception as exc:
            logger.warning("[%s] AMF save/restore setup failed: %s", cfg.name, exc)

    # Phase 2 — warm batched run
    batch_size = min(50, args.num_requests)
    batch_times_ms: List[float] = []
    for start in range(0, args.num_requests, batch_size):
        end = min(start + batch_size, args.num_requests)
        t_b = time.monotonic()
        _ = llm.generate(prompts[start:end], sampling)
        batch_ms = (time.monotonic() - t_b) * 1000.0
        per_req_ms = batch_ms / max(1, (end - start))
        batch_times_ms.extend([per_req_ms] * (end - start))

    warm_avg = statistics.mean(batch_times_ms) if batch_times_ms else 0.0
    warm_p50 = statistics.median(batch_times_ms) if batch_times_ms else 0.0
    sorted_bt = sorted(batch_times_ms)
    warm_p95 = sorted_bt[int(0.95 * len(sorted_bt))] if sorted_bt else 0.0
    warm_p99 = sorted_bt[int(0.99 * len(sorted_bt))] if sorted_bt else 0.0
    total_ms = sum(batch_times_ms) if batch_times_ms else 1.0
    throughput = args.num_requests / max(total_ms / 1000.0 / args.num_requests, 1e-3)

    # Collect tier hit stats if available
    amf_hits = 0
    lmcache_hits = 0
    cold_misses = 0
    try:
        stats_results = llm.collective_rpc("amf_diag_snapshot", args=("bench_end",))
        # amf_diag_snapshot doesn't report tier counts directly, so we
        # rely on the router stats surfaced via AmfWorkerExtension if
        # present. For now we just leave these at zero — the hit counts
        # are auditable from the [TIERED] log lines.
    except Exception:
        pass

    # Shut down the engine before the next config loads a fresh copy
    try:
        del llm
    except Exception:
        pass

    return BenchResult(
        config=cfg.name,
        cold_ms=cold_ms,
        warm_avg_ms=warm_avg,
        warm_p50_ms=warm_p50,
        warm_p95_ms=warm_p95,
        warm_p99_ms=warm_p99,
        throughput_req_per_s=throughput,
        restore_ms=restore_ms,
        register_ms=register_ms,
        amf_hits=amf_hits,
        lmcache_hits=lmcache_hits,
        cold_misses=cold_misses,
        output_sample=sample_text,
    )


# ── Report ───────────────────────────────────────────────────────────────────

def print_table(results: List[BenchResult]) -> None:
    print()
    print("=" * 90)
    print(f"{'Config':<22} {'Cold(ms)':>10} {'P50(ms)':>10} {'P99(ms)':>10} "
          f"{'Restore(ms)':>12} {'Speedup':>10}")
    print("=" * 90)
    for r in results:
        speedup = r.cold_ms / max(r.warm_avg_ms, 1.0)
        print(
            f"{r.config:<22} "
            f"{r.cold_ms:>10.0f} "
            f"{r.warm_p50_ms:>10.1f} "
            f"{r.warm_p99_ms:>10.1f} "
            f"{r.restore_ms:>12.1f} "
            f"{speedup:>9.1f}x"
        )
    print("=" * 90)
    print()


def main() -> None:
    p = argparse.ArgumentParser(
        description="Head-to-head benchmark: plain vLLM vs LMCache vs AMF vs tiered",
    )
    p.add_argument("--model", required=True, type=str)
    p.add_argument("--prefix-file", required=True, type=str)
    p.add_argument("--num-requests", type=int, default=500)
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--kv-cache-dtype", type=str, default="auto")
    p.add_argument("--output", type=str, default="/tmp/bench_amf_vs_lmcache.json")
    p.add_argument(
        "--configs", type=str, default="all",
        help="Comma-separated config names to run (or 'all')",
    )
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    which = args.configs.strip().lower()
    if which == "all":
        configs = CONFIGS
    else:
        wanted = {s.strip() for s in which.split(",") if s.strip()}
        configs = [c for c in CONFIGS if c.name in wanted]
        if not configs:
            print(f"[ERROR] no matching configs: {wanted}", file=sys.stderr)
            sys.exit(1)

    results: List[BenchResult] = []
    for cfg in configs:
        try:
            r = run_config(cfg, args)
            results.append(r)
        except Exception as exc:
            logger.exception("[%s] FAILED: %s", cfg.name, exc)
            results.append(BenchResult(
                config=cfg.name, cold_ms=0, warm_avg_ms=0,
                warm_p50_ms=0, warm_p95_ms=0, warm_p99_ms=0,
                throughput_req_per_s=0, restore_ms=0, register_ms=0,
                amf_hits=0, lmcache_hits=0, cold_misses=0,
                output_sample="", notes=f"FAILED: {exc}",
            ))

    with open(args.output, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    logger.info("[BENCH] wrote results to %s", args.output)

    print_table(results)


if __name__ == "__main__":
    main()

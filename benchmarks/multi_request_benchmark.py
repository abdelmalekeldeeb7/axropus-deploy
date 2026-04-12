"""multi_request_benchmark.py — End-to-end AMF-vs-cold benchmark.

Simulates N requests hitting an Axropus-fronted vLLM. Measures the TTFT
and total latency with AMF enabled vs disabled, then reports the
aggregate savings.

This benchmark does not require a real vLLM or GPU — it uses the AMF
hook in stub mode to isolate the cache-layer performance. For the
production benchmark on H200 see ``axropus_vs_lmcache_h200.md``.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List

import torch

from korith_vllm_ext.compressed_vram_pool import CompressedVRAMPool
from korith_vllm_ext.tiered_router import (
    CacheTier,
    TieredCacheRouter,
    WritePolicy,
    compute_prefix_hash,
)
from korith_vllm_ext.lmcache_adapter import LMCacheAdapter


@dataclass
class RequestSpec:
    prompt_tokens: List[int]
    shared_prefix_len: int


def _generate_workload(
    n_requests: int = 500,
    shared_prefixes: int = 10,
    tokens_per_prefix: int = 2048,
    suffix_len: int = 64,
    seed: int = 7,
) -> List[RequestSpec]:
    """Generate a workload with prefix reuse typical of agent chat."""
    random.seed(seed)
    prefixes = [
        [random.randint(1, 32000) for _ in range(tokens_per_prefix)]
        for _ in range(shared_prefixes)
    ]
    out: List[RequestSpec] = []
    for _ in range(n_requests):
        p = random.choice(prefixes)
        suffix = [random.randint(1, 32000) for _ in range(suffix_len)]
        out.append(RequestSpec(prompt_tokens=p + suffix, shared_prefix_len=len(p)))
    return out


def _simulate_prefill_cost(tokens: int) -> float:
    """Fake prefill cost model: 0.05 ms per token on a 70B model."""
    return tokens * 0.05


def run_benchmark(n_requests: int = 500) -> Dict[str, Any]:
    workload = _generate_workload(n_requests=n_requests)

    # Cold baseline: no caching.
    cold_total_ms = sum(_simulate_prefill_cost(len(r.prompt_tokens)) for r in workload)

    # AMF-enabled run. We drive the router directly so we can hash at
    # the shared-prefix boundary (the benchmark knows it; the hook in
    # production uses a block-level matcher not shown here).
    pool = CompressedVRAMPool(
        num_layers=4,
        bytes_per_layer=1 << 22,  # 4 MB per layer for the simulation
        block_bytes=1 << 15,
        default_format="int4_sym_block",
        device="cpu",
    )
    lmc = LMCacheAdapter(enabled=True, backend="cpu")
    router = TieredCacheRouter(
        pool,
        lmc,
        min_prefix_tokens=16,
        write_policy=WritePolicy.ALWAYS,
    )

    amf_total_ms = 0.0
    hits = 0
    misses = 0
    tokens_skipped = 0

    for req in workload:
        shared = req.prompt_tokens[: req.shared_prefix_len]
        suffix = req.prompt_tokens[req.shared_prefix_len :]
        prefix_hash = compute_prefix_hash(shared)

        result = router.lookup(prefix_hash, shared)
        if result.tier == CacheTier.G1_AMF or result.tier == CacheTier.G3_LMCACHE:
            hits += 1
            tokens_skipped += req.shared_prefix_len
            amf_total_ms += _simulate_prefill_cost(len(suffix)) + result.latency_ms
        else:
            misses += 1
            cost = _simulate_prefill_cost(len(req.prompt_tokens))
            amf_total_ms += cost
            fake_kv = torch.randn(4, 2, len(shared), 4, 64)
            router.store_after_prefill(prefix_hash, shared, fake_kv, savings_ms=cost)

    stats = {
        "hits":              hits,
        "misses":            misses,
        "total_skip_tokens": tokens_skipped,
        "hit_rate":          hits / max(1, hits + misses),
    }
    return {
        "n_requests":    n_requests,
        "cold_total_ms": round(cold_total_ms, 1),
        "amf_total_ms":  round(amf_total_ms, 1),
        "savings_ms":    round(cold_total_ms - amf_total_ms, 1),
        "savings_pct":   round(100.0 * (cold_total_ms - amf_total_ms) / max(1.0, cold_total_ms), 2),
        "hits":          stats["hits"],
        "misses":        stats["misses"],
        "hit_rate":      round(stats["hit_rate"], 3),
        "tokens_skipped": stats["total_skip_tokens"],
    }


if __name__ == "__main__":
    res = run_benchmark()
    print(json.dumps(res, indent=2))

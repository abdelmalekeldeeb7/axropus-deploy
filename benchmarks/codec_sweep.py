"""codec_sweep.py — Compression / accuracy / speed sweep across all codecs.

Measures:

    * Memory ratio (compressed bytes / FP16 bytes)
    * Reconstruction error (relative L2 norm)
    * Compress / decompress throughput on CPU

The sweep runs on synthetic data by default (fast, deterministic). If
an environment variable ``AXROPUS_BENCH_KV_PATH`` points at a saved
KV tensor on disk the benchmark uses that instead, so customers can
drop in a real trace from a production run.

Invoke from the CLI:

    axropus bench

or directly:

    python -m benchmarks.codec_sweep
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List

import torch

from korith_vllm_ext.codecs import get_codec, list_codecs


def _load_kv(quick: bool) -> torch.Tensor:
    path = os.environ.get("AXROPUS_BENCH_KV_PATH")
    if path and os.path.exists(path):
        t = torch.load(path, map_location="cpu")
        if isinstance(t, torch.Tensor):
            return t.float()
    torch.manual_seed(0)
    tokens = 128 if quick else 1024
    return torch.randn(2, tokens, 8, 128)


def _time_ms(fn, iters: int) -> float:
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    end = time.perf_counter()
    return (end - start) * 1000.0 / iters


def run_codec_sweep(quick: bool = False) -> List[Dict[str, Any]]:
    kv = _load_kv(quick=quick)
    base_bytes = kv.numel() * 2  # FP16 baseline
    results: List[Dict[str, Any]] = []

    iters = 3 if quick else 10
    for name in list_codecs():
        codec = get_codec(name)
        blob = codec.compress(kv)
        recon = codec.decompress_to(blob, target_dtype=torch.float32)
        rel_err = (recon - kv).norm().item() / kv.norm().item()

        comp_ms   = _time_ms(lambda: codec.compress(kv), iters)
        decomp_ms = _time_ms(lambda: codec.decompress_to(blob, torch.float32), iters)

        results.append({
            "codec":      name,
            "memory_ratio": round(blob.nbytes() / base_bytes, 4),
            "rel_err":    round(rel_err, 4),
            "compress_ms":   round(comp_ms, 2),
            "decompress_ms": round(decomp_ms, 2),
        })

    # Pretty-print for humans.
    print("codec             ratio   rel_err  comp_ms  decomp_ms")
    print("─" * 56)
    for r in results:
        print(
            f"{r['codec']:15s}  {r['memory_ratio']:6.3f}  {r['rel_err']:7.4f}  "
            f"{r['compress_ms']:7.2f}  {r['decompress_ms']:9.2f}"
        )
    return results


if __name__ == "__main__":
    import sys
    quick = "--quick" in sys.argv
    res = run_codec_sweep(quick=quick)
    print()
    print(json.dumps(res, indent=2))

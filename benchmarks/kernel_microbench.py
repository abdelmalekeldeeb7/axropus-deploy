"""kernel_microbench.py — Per-kernel decode latency microbenchmark.

On a real GPU this exercises each dispatchable kernel and measures
microsecond-level latency. On CI / CPU-only hosts it runs against the
pure-Python fallback so we still get regression signal.

Invocation:

    python -m benchmarks.kernel_microbench [--quick]
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

from korith_vllm_ext.codecs import FMT_FP8_E4M3, FMT_INT4_BLOCK, get_codec
from korith_vllm_ext.kernels import (
    dispatch_kernel,
    fallback_fp16_kernel,
    get_current_sm_version,
)


def _time_kernel(fn, iters: int) -> float:
    start = time.perf_counter()
    for _ in range(iters):
        out = fn()
        if isinstance(out, torch.Tensor) and out.is_cuda:
            torch.cuda.synchronize()
    end = time.perf_counter()
    return (end - start) / iters * 1e6  # microseconds


def run_kernel_microbench(quick: bool = False) -> List[Dict[str, Any]]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B = 2 if quick else 4
    H = 8
    D = 128
    T_values = [1024] if quick else [1024, 4096, 16384]
    iters = 20 if quick else 50

    results: List[Dict[str, Any]] = []

    sm = get_current_sm_version()

    for T in T_values:
        q = torch.randn(B, H, 1, D, device=device).half()
        k = torch.randn(B, H, T, D, device=device).half()
        v = torch.randn(B, H, T, D, device=device).half()

        # Reference: torch SDPA.
        ref_us = _time_kernel(lambda: F.scaled_dot_product_attention(q, k, v), iters)

        # AMF fallback (no kernel compiled).
        amf_us = _time_kernel(lambda: fallback_fp16_kernel(q, k, v), iters)

        # Dispatch path with INT4 storage + FP8 active.
        fn = dispatch_kernel(FMT_INT4_BLOCK, FMT_FP8_E4M3)
        int4_us = _time_kernel(lambda: fn(q, k, v), iters)

        results.append({
            "T":       T,
            "sm":      sm,
            "sdpa_us": round(ref_us, 2),
            "amf_fallback_us": round(amf_us, 2),
            "amf_int4_dispatch_us": round(int4_us, 2),
            "speedup_vs_sdpa": round(ref_us / max(1e-6, int4_us), 3),
        })

    print("T       sm   sdpa(us)   fallback(us)   int4_dispatch(us)   speedup")
    print("─" * 68)
    for r in results:
        print(
            f"{r['T']:6d}  {r['sm']:3d}  {r['sdpa_us']:9.1f}  "
            f"{r['amf_fallback_us']:12.1f}  {r['amf_int4_dispatch_us']:18.1f}  "
            f"{r['speedup_vs_sdpa']:7.2f}x"
        )

    return results


if __name__ == "__main__":
    import sys
    quick = "--quick" in sys.argv
    res = run_kernel_microbench(quick=quick)
    print()
    print(json.dumps(res, indent=2))

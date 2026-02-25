from __future__ import annotations

from typing import Dict


def _safe_float(value) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return 0.0


def extract_savings_components(metrics: Dict) -> Dict[str, float]:
    amf = metrics.get("amf", {})
    spec = metrics.get("spec", {})
    kernels = metrics.get("kernels", {})
    savings = metrics.get("savings", {})

    if isinstance(savings, dict):
        has_components = (
            "prefill_saved_ms" in savings
            and "spec_saved_ms" in savings
            and "kernels_saved_ms" in savings
        )
        if has_components:
            prefill_saved_ms = max(0.0, _safe_float(savings.get("prefill_saved_ms", 0.0)))
            spec_saved_ms = max(0.0, _safe_float(savings.get("spec_saved_ms", 0.0)))
            kernels_saved_ms = max(0.0, _safe_float(savings.get("kernels_saved_ms", 0.0)))
            total_saved_ms = prefill_saved_ms + spec_saved_ms + kernels_saved_ms
            return {
                "prefill_saved_ms": prefill_saved_ms,
                "spec_saved_ms": spec_saved_ms,
                "kernels_saved_ms": kernels_saved_ms,
                "total_saved_ms": total_saved_ms,
            }

    prefill_saved_ms = max(0.0, _safe_float(amf.get("saved_ms", 0.0)))
    raw_kernel_saved = max(0.0, _safe_float(kernels.get("ms_saved", 0.0)))
    kernels_applied = bool(kernels.get("kernels_applied", False))
    kernels_comparable_known = "comparable" in kernels
    kernels_comparable = bool(kernels.get("comparable", False))
    if not kernels_applied or (kernels_comparable_known and not kernels_comparable):
        raw_kernel_saved = 0.0
    spec_saved_raw = max(0.0, _safe_float(spec.get("saved_ms", 0.0)))
    spec_saved_ms = max(0.0, spec_saved_raw - raw_kernel_saved)

    return {
        "prefill_saved_ms": prefill_saved_ms,
        "spec_saved_ms": spec_saved_ms,
        "kernels_saved_ms": raw_kernel_saved,
        "total_saved_ms": prefill_saved_ms + spec_saved_ms + raw_kernel_saved,
    }


def estimate_savings(metrics: Dict, gpu_hourly_cost: float) -> Dict[str, float]:
    components = extract_savings_components(metrics)
    perf = metrics.get("perf", {})
    saved_ms = float(components.get("total_saved_ms", 0.0) or 0.0)
    total_ms = _safe_float(perf.get("total_ms", 0.0))
    gpu_cost_per_ms = gpu_hourly_cost / 3_600_000.0
    cost_saved = saved_ms * gpu_cost_per_ms
    cost_total = total_ms * gpu_cost_per_ms
    return {
        "prefill_saved_ms": float(components.get("prefill_saved_ms", 0.0) or 0.0),
        "spec_saved_ms": float(components.get("spec_saved_ms", 0.0) or 0.0),
        "kernels_saved_ms": float(components.get("kernels_saved_ms", 0.0) or 0.0),
        "saved_ms": saved_ms,
        "total_ms": total_ms,
        "gpu_cost_per_ms": gpu_cost_per_ms,
        "cost_saved_usd": cost_saved,
        "cost_total_usd": cost_total,
    }

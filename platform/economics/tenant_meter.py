from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict

from .model import extract_savings_components


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return 0.0


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class TenantMeter:
    tenant_id: str
    total_requests: int = 0
    amf_hits: int = 0
    amf_misses: int = 0
    prefill_saved_ms: float = 0.0
    decode_ms: float = 0.0
    total_saved_ms: float = 0.0
    total_compute_ms: float = 0.0
    cache_entries: int = 0
    cache_bytes: int = 0

    def accumulate(self, metrics: Dict[str, Any]) -> None:
        self.total_requests += 1
        amf = metrics.get("amf", {}) if isinstance(metrics.get("amf", {}), dict) else {}
        perf = metrics.get("perf", {}) if isinstance(metrics.get("perf", {}), dict) else {}

        decision = str(amf.get("decision", "") or "").strip().lower()
        if decision == "hit":
            self.amf_hits += 1
        elif decision == "miss":
            self.amf_misses += 1

        savings = extract_savings_components(metrics)
        prefill_saved = max(0.0, _safe_float(savings.get("prefill_saved_ms", 0.0)))
        total_saved = max(0.0, _safe_float(savings.get("total_saved_ms", 0.0)))
        total_ms = max(0.0, _safe_float(perf.get("total_ms", 0.0)))
        decode_ms = max(0.0, _safe_float(perf.get("decode_ms", 0.0)))

        self.prefill_saved_ms += prefill_saved
        self.total_saved_ms += total_saved
        self.total_compute_ms += total_ms
        self.decode_ms += decode_ms

        try:
            self.cache_entries = max(self.cache_entries, int(amf.get("cache_entries", 0) or 0))
        except Exception:
            pass
        try:
            self.cache_bytes = max(self.cache_bytes, int(amf.get("cache_bytes", 0) or 0))
        except Exception:
            pass

    def summary(
        self,
        *,
        gpu_hourly_cost: float = 2.5,
        period_start: str = "",
        period_end: str = "",
    ) -> Dict[str, Any]:
        gpu_cost_per_ms = max(0.0, float(gpu_hourly_cost)) / 3_600_000.0
        baseline_ms = max(0.0, self.total_compute_ms + self.total_saved_ms)
        hit_rate = (float(self.amf_hits) / float(max(1, self.amf_hits + self.amf_misses)))
        savings_pct = (self.total_saved_ms / baseline_ms * 100.0) if baseline_ms > 0.0 else 0.0
        now = _utc_now_iso()
        out = {
            "tenant_id": self.tenant_id,
            "period_start": period_start or now,
            "period_end": period_end or now,
            "total_requests": int(self.total_requests),
            "amf_hit_rate": float(hit_rate),
            "total_saved_ms": float(self.total_saved_ms),
            "total_compute_ms": float(self.total_compute_ms),
            "savings_pct": float(savings_pct),
            "cost_saved_usd": float(self.total_saved_ms * gpu_cost_per_ms),
            "cost_served_usd": float(self.total_compute_ms * gpu_cost_per_ms),
            "cache_entries": int(self.cache_entries),
            "cache_bytes": int(self.cache_bytes),
        }
        return out

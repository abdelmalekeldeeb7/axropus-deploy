"""axropus.metrics — Prometheus metric exposure for AMF.

All metrics follow the names from §8 of the design doc. When
``prometheus_client`` is not installed we stub the surface with no-op
counters so imports succeed and the server can run without telemetry.

Metrics exposed:

    axropus_cache_hits_total{tier}
    axropus_cache_latency_ms{tier,quantile}
    axropus_pool_prefixes_resident
    axropus_pool_bytes_used
    axropus_pool_evictions_total
    axropus_compression_ratio{codec}
    axropus_kernel_latency_us{kernel}
    axropus_tokens_saved_total
    axropus_compute_cost_saved_usd     (optional, requires $/GPU-hr env)
    axropus_miss_reason_total{reason}

Use ``MetricRegistry`` as the single entry point; the server and the
router push into it and the HTTP handler renders the Prometheus text
format.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

try:
    from prometheus_client import (  # type: ignore
        REGISTRY,
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )
    _HAS_PROM = True
except ImportError:  # pragma: no cover - optional dependency
    _HAS_PROM = False


# ── No-op stand-ins ─────────────────────────────────────────────────────────


class _NoopMetric:
    def labels(self, *args: Any, **kwargs: Any) -> "_NoopMetric":
        return self

    def inc(self, value: float = 1.0) -> None:
        return

    def observe(self, value: float) -> None:
        return

    def set(self, value: float) -> None:
        return


# ── Registry wrapper ────────────────────────────────────────────────────────


class MetricRegistry:
    """Owns all Axropus Prometheus metrics.

    Instantiate once at server startup and pass a reference to the
    router / hook so they can push values. ``render()`` returns the
    Prometheus text-format payload for ``/metrics``.
    """

    def __init__(self, *, gpu_hourly_usd: float = 0.0) -> None:
        self.gpu_hourly_usd = gpu_hourly_usd

        if _HAS_PROM:
            self._registry = CollectorRegistry(auto_describe=True)
            self.cache_hits = Counter(
                "axropus_cache_hits_total",
                "Cache hits per tier",
                ["tier"],
                registry=self._registry,
            )
            self.cache_latency = Histogram(
                "axropus_cache_latency_ms",
                "Cache lookup latency in milliseconds",
                ["tier"],
                buckets=(0.5, 1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000),
                registry=self._registry,
            )
            self.pool_prefixes = Gauge(
                "axropus_pool_prefixes_resident",
                "Number of prefixes currently resident in the AMF pool",
                registry=self._registry,
            )
            self.pool_bytes = Gauge(
                "axropus_pool_bytes_used",
                "Bytes used by the AMF compressed pool",
                registry=self._registry,
            )
            self.pool_evictions = Counter(
                "axropus_pool_evictions_total",
                "Total evictions from the AMF pool",
                registry=self._registry,
            )
            self.compression_ratio = Gauge(
                "axropus_compression_ratio",
                "Compressed bytes over FP16 bytes per codec",
                ["codec"],
                registry=self._registry,
            )
            self.kernel_latency = Histogram(
                "axropus_kernel_latency_us",
                "Decode kernel latency in microseconds",
                ["kernel"],
                buckets=(50, 100, 200, 500, 1000, 2500, 5000, 10000, 25000, 50000),
                registry=self._registry,
            )
            self.tokens_saved = Counter(
                "axropus_tokens_saved_total",
                "Total prefix tokens skipped by warm AMF hits",
                registry=self._registry,
            )
            self.cost_saved = Counter(
                "axropus_compute_cost_saved_usd",
                "Estimated USD saved from AMF hits",
                registry=self._registry,
            )
            self.miss_reason = Counter(
                "axropus_miss_reason_total",
                "AMF miss counts per diagnostic reason",
                ["reason"],
                registry=self._registry,
            )
        else:
            self._registry = None
            self.cache_hits        = _NoopMetric()
            self.cache_latency     = _NoopMetric()
            self.pool_prefixes     = _NoopMetric()
            self.pool_bytes        = _NoopMetric()
            self.pool_evictions    = _NoopMetric()
            self.compression_ratio = _NoopMetric()
            self.kernel_latency    = _NoopMetric()
            self.tokens_saved      = _NoopMetric()
            self.cost_saved        = _NoopMetric()
            self.miss_reason       = _NoopMetric()

    # ── Convenience updaters ───────────────────────────────────────────────

    def record_hit(self, tier: str, latency_ms: float) -> None:
        self.cache_hits.labels(tier=tier).inc()
        self.cache_latency.labels(tier=tier).observe(latency_ms)

    def record_miss(self, reason: str) -> None:
        self.miss_reason.labels(reason=reason).inc()

    def record_pool_snapshot(self, bytes_used: int, prefixes: int, evictions: int) -> None:
        self.pool_bytes.set(bytes_used)
        self.pool_prefixes.set(prefixes)
        # Evictions is monotonic; don't overwrite — only inc by delta.
        if hasattr(self, "_last_evictions"):
            delta = max(0, evictions - self._last_evictions)  # type: ignore[attr-defined]
            if delta:
                self.pool_evictions.inc(delta)
        else:
            if evictions:
                self.pool_evictions.inc(evictions)
        self._last_evictions = evictions

    def record_tokens_saved(self, n: int, *, compute_seconds: float = 0.0) -> None:
        self.tokens_saved.inc(n)
        if self.gpu_hourly_usd and compute_seconds > 0:
            self.cost_saved.inc(compute_seconds * self.gpu_hourly_usd / 3600.0)

    def record_compression_ratio(self, codec: str, ratio: float) -> None:
        self.compression_ratio.labels(codec=codec).set(ratio)

    def record_kernel_latency(self, kernel: str, micros: float) -> None:
        self.kernel_latency.labels(kernel=kernel).observe(micros)

    # ── Rendering ──────────────────────────────────────────────────────────

    def render(self) -> bytes:
        if _HAS_PROM and self._registry is not None:
            return generate_latest(self._registry)
        return b"# prometheus_client not installed\n"


__all__ = ["MetricRegistry"]

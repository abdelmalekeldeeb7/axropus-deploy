"""
Component 1: AMF Metrics Collector
===================================
Bridges AMF's C++ metrics (via the Python observability layer) and the AMF
Coordinator HTTP API into a fixed-size numpy tensor consumed by the reasoning
model every 100 ms.

Tensor layout — shape (64,) float32
------------------------------------
Slot  Name                    Range / units
0     hit_count_delta         hits since last window (raw)
1     miss_count_delta        misses since last window (raw)
2     hit_rate                lifetime hits / lifetime requests  [0, 1]
3     tokens_saved_delta      tokens saved since last window (raw)
4     savings_ms_delta        ms saved since last window (raw)
5     restore_ms_avg          rolling avg restore latency (ms)
6     eviction_rate           evictions per second
7     admission_rate          admissions per second
8     rejection_rate          admission rejections per second
9     failure_rate            restore failures per second
10    storage_utilization     bytes_used / budget  [0, 1]
11    eviction_pressure       from AMF eviction_pressure field  [0, 1]
12    entry_count_norm        total entries / 1000
13    hot_entry_ratio         hot_entries / total_entries  [0, 1]
14    cold_entry_ratio        cold_entries / total_entries  [0, 1]
15    instant_tps_norm        instant TPS / 1000
16    rolling_tps_norm        rolling TPS / 1000
17    node_count              active node count (raw)
18    avg_node_hit_rate       mean hit rate across nodes  [0, 1]
19    avg_node_warmth         mean warm_ratio across nodes  [0, 1]
20    total_cache_entries     total entries across all nodes / 1000
21    total_cache_gb          total cache bytes / 1e9
22    request_rate            requests per second (10-second window)
23    hit_streak_norm         consecutive hits / 100  [0, 1]
24    miss_streak_norm        consecutive misses / 20  [0, 1]
25    reuse_ratio             lifetime hits / lifetime requests  [0, 1]
26    unique_prefixes_norm    unique prefix keys / 1000
27    hit_queue_depth_norm    hit queue depth / 100  [0, 1]
28    miss_queue_depth_norm   miss queue depth / 100  [0, 1]
29    roi_ema_norm            EMA of per-prefix ROI scores / 10  [0, 1]
30    memory_pressure         memory usage ratio  [0, 1]
31    decode_latency_norm     recent decode latency / 10 000 ms  [0, 1]

Usage
-----
    collector = MetricsCollector(coordinator_url="http://localhost:8500")
    collector.start()

    tensor = collector.get_metrics_tensor()   # numpy float32 (64,)
    info   = collector.get_metrics_dict()     # human-readable snapshot

    # Called by Worker on every request outcome:
    collector.record_hit(prefix_hash="abc123", roi=2.4, restore_ms=8.1)
    collector.record_miss(prefix_hash="def456")
    collector.record_decode(decode_ms=11_400.0)

    collector.stop()
"""

from __future__ import annotations

import json
import math
import threading
import time
import urllib.request
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

# ── tensor constants ────────────────────────────────────────────────────────
TENSOR_SIZE = 80              # extended from 64: slots 54-69 hold Dynamo fleet metrics
POLL_INTERVAL_S = 0.10        # 100 ms
RATE_WINDOW_S   = 10.0        # rolling window for rate computations
ROI_EMA_ALPHA   = 0.05        # EMA smoothing for ROI signals
DECODE_EMA_ALPHA = 0.10       # EMA smoothing for decode latency

# Slot indices — use these names in downstream code instead of magic numbers
class Slot:
    HIT_DELTA          = 0
    MISS_DELTA         = 1
    HIT_RATE           = 2
    TOKENS_SAVED_DELTA = 3
    SAVINGS_MS_DELTA   = 4
    RESTORE_MS_AVG     = 5
    EVICTION_RATE      = 6
    ADMISSION_RATE     = 7
    REJECTION_RATE     = 8
    FAILURE_RATE       = 9
    STORAGE_UTIL       = 10
    EVICTION_PRESSURE  = 11
    ENTRY_COUNT_NORM   = 12
    HOT_ENTRY_RATIO    = 13
    COLD_ENTRY_RATIO   = 14
    INSTANT_TPS_NORM   = 15
    ROLLING_TPS_NORM   = 16
    NODE_COUNT         = 17
    AVG_NODE_HIT_RATE  = 18
    AVG_NODE_WARMTH    = 19
    TOTAL_CACHE_ENTRIES = 20
    TOTAL_CACHE_GB     = 21
    REQUEST_RATE       = 22
    HIT_STREAK_NORM    = 23
    MISS_STREAK_NORM   = 24
    REUSE_RATIO        = 25
    UNIQUE_PREFIXES_NORM = 26
    HIT_QUEUE_NORM     = 27
    MISS_QUEUE_NORM    = 28
    ROI_EMA_NORM       = 29
    MEMORY_PRESSURE    = 30
    DECODE_LATENCY_NORM = 31

    # ── decode slots (32-63) — updated by record_decode_step() ──────────────
    ACCEPTANCE_EMA          = 32
    ACCEPTANCE_VARIANCE     = 33
    CURRENT_SPEC_DEPTH      = 34
    ENTROPY_EMA             = 35
    TPS_CURRENT             = 36
    TPS_DELTA               = 37
    TOKENS_COMMITTED_NODECODE = 38
    TOKENS_DECODED_NORMALLY = 39
    SPEC_PROFIT_RATIO       = 40
    SPEC_BAD_STREAK         = 41
    SPEC_COOLDOWN_REMAINING = 42
    LANE_CURRENT            = 43
    LANE_HOLD_REMAINING     = 44
    STARVATION_COUNTER      = 45
    DRAFT_MODEL_TPS         = 46
    VERIFY_BATCH_SIZE       = 47
    KV_CACHE_UTILIZATION    = 48
    CUDA_GRAPH_ACTIVE       = 49
    DECODE_LATENCY_PER_TOKEN = 50
    V3_BLOCK_DEPTH          = 51
    HOLO_FUTURE_SAFE        = 52
    POWER_WATTS_NORM        = 53
    # slots 54-63: reserved for future decode signals

    # ── Dynamo fleet slots (54-69) — filled by DynamoMetricsSource ───────────
    DYNAMO_FLEET_HIT_RATE      = 54   # KV routing hit rate across fleet [0, 1]
    DYNAMO_FLEET_MISS_RATE     = 55   # routing miss rate [0, 1]
    DYNAMO_EVICTION_RATE       = 56   # blocks evicted / second across fleet
    DYNAMO_BLOCK_COUNT_NORM    = 57   # tracked blocks / 100 000
    DYNAMO_WORKER_COUNT        = 58   # active Dynamo workers
    DYNAMO_AVG_WORKER_LOAD     = 59   # mean worker load [0, 1]
    DYNAMO_MAX_WORKER_LOAD     = 60   # max worker load [0, 1]
    DYNAMO_PREFILL_QUEUE_DEPTH = 61   # pending prefill requests / 100
    DYNAMO_DECODE_QUEUE_DEPTH  = 62   # pending decode requests / 100
    DYNAMO_KV_OVERLAP_AVG      = 63   # mean KV overlap on routed requests [0, 1]
    DYNAMO_AMF_ROUTED_RATIO    = 64   # fraction routed via AMF vs Dynamo [0, 1]
    DYNAMO_RESTORE_RATIO       = 65   # AMF restores / cold recomputes [0, inf)
    DYNAMO_CROSS_NODE_XFERS    = 66   # cross-node KV transfers / second
    DYNAMO_TIER_G1_PCT         = 67   # fraction of blocks in GPU HBM [0, 1]
    DYNAMO_TIER_G2_PCT         = 68   # fraction in CPU RAM [0, 1]
    DYNAMO_TIER_G3G4_PCT       = 69   # fraction in NVMe/remote [0, 1]
    # slots 70-79: reserved


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except Exception:
        return default


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


class _RateWindow:
    """Tracks event counts in a sliding time window to produce rates."""

    def __init__(self, window_s: float = RATE_WINDOW_S) -> None:
        self._window_s = window_s
        self._events: Deque[Tuple[float, float]] = deque()  # (timestamp, count)
        self._lock = threading.Lock()

    def record(self, count: float = 1.0) -> None:
        now = time.monotonic()
        with self._lock:
            self._events.append((now, count))
            self._evict(now)

    def rate(self) -> float:
        """Events per second over the window."""
        now = time.monotonic()
        with self._lock:
            self._evict(now)
            if not self._events:
                return 0.0
            total = sum(c for _, c in self._events)
            span = max(now - self._events[0][0], 1e-6)
            return total / min(span, self._window_s)

    def _evict(self, now: float) -> None:
        cutoff = now - self._window_s
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()


class MetricsCollector:
    """
    Polls all AMF metrics sources and maintains a current tensor snapshot.

    Sources
    -------
    1. GLOBAL_METRICS (platform.observability.metrics) — Python-side counters /
       gauges updated by workers and routers.
    2. AMF Coordinator HTTP /health — aggregate key/node/row counts.
    3. AMF Coordinator HTTP /metrics — per-node hit rates, warmth, storage.
    4. Direct event hooks — record_hit / record_miss / record_decode called by
       the Worker on each request completion (zero extra latency, no polling).

    The collector is safe to call from multiple threads.
    """

    def __init__(
        self,
        coordinator_url: str = "",
        coordinator_timeout_s: float = 0.10,
        poll_interval_s: float = POLL_INTERVAL_S,
        worker_hit_queue: Optional[Any] = None,   # queue.Queue from Worker
        worker_miss_queue: Optional[Any] = None,  # queue.Queue from Worker
        dynamo_source: Optional["DynamoMetricsSource"] = None,
    ) -> None:
        self._coordinator_url = str(coordinator_url or "").rstrip("/")
        self._coordinator_timeout = max(0.02, coordinator_timeout_s)
        self._poll_interval = max(0.01, poll_interval_s)

        # Optional worker queue references for depth sampling
        self._q_hit  = worker_hit_queue
        self._q_miss = worker_miss_queue

        # ── cumulative counters from GLOBAL_METRICS (previous snapshot) ──
        self._prev_counters: Dict[str, float] = {}
        self._prev_gauges: Dict[str, float] = {}
        self._prev_ts: float = time.monotonic()

        # ── lifetime totals (accumulated from deltas) ──────────────────
        self._total_hits:    int = 0
        self._total_misses:  int = 0
        self._total_requests: int = 0

        # ── streak trackers ─────────────────────────────────────────────
        self._hit_streak:  int = 0
        self._miss_streak: int = 0

        # ── rate windows ────────────────────────────────────────────────
        self._rw_requests   = _RateWindow()
        self._rw_evictions  = _RateWindow()
        self._rw_admissions = _RateWindow()
        self._rw_rejections = _RateWindow()
        self._rw_failures   = _RateWindow()

        # ── EMAs ────────────────────────────────────────────────────────
        self._roi_ema:    float = 0.0
        self._decode_ema: float = 0.0   # ms

        # ── restore latency samples ─────────────────────────────────────
        self._restore_ms_samples: Deque[float] = deque(maxlen=200)

        # ── per-node snapshots from coordinator ─────────────────────────
        self._node_snapshots: List[Dict[str, float]] = []

        # ── current tensor ───────────────────────────────────────────────
        self._tensor_lock = threading.Lock()
        self._tensor: np.ndarray = np.zeros(TENSOR_SIZE, dtype=np.float32)

        # ── Dynamo fleet metrics source (optional) ───────────────────────
        self._dynamo_source = dynamo_source

        # ── background poll thread ───────────────────────────────────────
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._poll_loop,
            name="amf-metrics-collector",
            daemon=True,
        )

    # ── public API ──────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background polling thread."""
        self._thread.start()

    def stop(self) -> None:
        """Stop the background polling thread (waits up to 1 s)."""
        self._stop_event.set()
        self._thread.join(timeout=1.0)

    def get_metrics_tensor(self) -> np.ndarray:
        """Return a copy of the current metrics tensor (64-element float32)."""
        with self._tensor_lock:
            return self._tensor.copy()

    def get_metrics_dict(self) -> Dict[str, float]:
        """Return the current tensor as a human-readable dict."""
        with self._tensor_lock:
            t = self._tensor
            return {
                "hit_count_delta":       float(t[Slot.HIT_DELTA]),
                "miss_count_delta":      float(t[Slot.MISS_DELTA]),
                "hit_rate":              float(t[Slot.HIT_RATE]),
                "tokens_saved_delta":    float(t[Slot.TOKENS_SAVED_DELTA]),
                "savings_ms_delta":      float(t[Slot.SAVINGS_MS_DELTA]),
                "restore_ms_avg":        float(t[Slot.RESTORE_MS_AVG]),
                "eviction_rate":         float(t[Slot.EVICTION_RATE]),
                "admission_rate":        float(t[Slot.ADMISSION_RATE]),
                "rejection_rate":        float(t[Slot.REJECTION_RATE]),
                "failure_rate":          float(t[Slot.FAILURE_RATE]),
                "storage_utilization":   float(t[Slot.STORAGE_UTIL]),
                "eviction_pressure":     float(t[Slot.EVICTION_PRESSURE]),
                "entry_count_norm":      float(t[Slot.ENTRY_COUNT_NORM]),
                "hot_entry_ratio":       float(t[Slot.HOT_ENTRY_RATIO]),
                "cold_entry_ratio":      float(t[Slot.COLD_ENTRY_RATIO]),
                "instant_tps_norm":      float(t[Slot.INSTANT_TPS_NORM]),
                "rolling_tps_norm":      float(t[Slot.ROLLING_TPS_NORM]),
                "node_count":            float(t[Slot.NODE_COUNT]),
                "avg_node_hit_rate":     float(t[Slot.AVG_NODE_HIT_RATE]),
                "avg_node_warmth":       float(t[Slot.AVG_NODE_WARMTH]),
                "total_cache_entries":   float(t[Slot.TOTAL_CACHE_ENTRIES]),
                "total_cache_gb":        float(t[Slot.TOTAL_CACHE_GB]),
                "request_rate":          float(t[Slot.REQUEST_RATE]),
                "hit_streak_norm":       float(t[Slot.HIT_STREAK_NORM]),
                "miss_streak_norm":      float(t[Slot.MISS_STREAK_NORM]),
                "reuse_ratio":           float(t[Slot.REUSE_RATIO]),
                "unique_prefixes_norm":  float(t[Slot.UNIQUE_PREFIXES_NORM]),
                "hit_queue_depth_norm":  float(t[Slot.HIT_QUEUE_NORM]),
                "miss_queue_depth_norm": float(t[Slot.MISS_QUEUE_NORM]),
                "roi_ema_norm":          float(t[Slot.ROI_EMA_NORM]),
                "memory_pressure":           float(t[Slot.MEMORY_PRESSURE]),
                "decode_latency_norm":       float(t[Slot.DECODE_LATENCY_NORM]),
                # decode slots
                "acceptance_ema":            float(t[Slot.ACCEPTANCE_EMA]),
                "acceptance_variance":       float(t[Slot.ACCEPTANCE_VARIANCE]),
                "current_spec_depth":        float(t[Slot.CURRENT_SPEC_DEPTH]),
                "entropy_ema":               float(t[Slot.ENTROPY_EMA]),
                "tps_current":               float(t[Slot.TPS_CURRENT]),
                "tps_delta":                 float(t[Slot.TPS_DELTA]),
                "lane_current":              float(t[Slot.LANE_CURRENT]),
                "lane_hold_remaining":       float(t[Slot.LANE_HOLD_REMAINING]),
                "starvation_counter":        float(t[Slot.STARVATION_COUNTER]),
                "holo_future_safe":          float(t[Slot.HOLO_FUTURE_SAFE]),
                "power_watts_norm":          float(t[Slot.POWER_WATTS_NORM]),
            }

    # ── event hooks (call from Worker thread, zero-latency path) ───────────

    def record_hit(
        self,
        prefix_hash: str = "",
        roi: float = 0.0,
        restore_ms: float = 0.0,
    ) -> None:
        """Record a cache hit event. Call from Worker immediately after a HIT."""
        self._total_hits += 1
        self._total_requests += 1
        self._hit_streak += 1
        self._miss_streak = 0
        self._rw_requests.record()

        if restore_ms > 0:
            self._restore_ms_samples.append(restore_ms)

        roi_f = _safe_float(roi)
        if roi_f > 0:
            self._roi_ema = (
                ROI_EMA_ALPHA * roi_f + (1.0 - ROI_EMA_ALPHA) * self._roi_ema
            )

    def record_miss(self, prefix_hash: str = "") -> None:
        """Record a cache miss event. Call from Worker immediately after a MISS."""
        self._total_misses += 1
        self._total_requests += 1
        self._miss_streak += 1
        self._hit_streak = 0
        self._rw_requests.record()

    def record_decode(self, decode_ms: float) -> None:
        """Record a decode latency sample (ms). Call at the end of each request."""
        ms = _safe_float(decode_ms)
        if ms > 0:
            self._decode_ema = (
                DECODE_EMA_ALPHA * ms + (1.0 - DECODE_EMA_ALPHA) * self._decode_ema
            )

    def record_eviction(self, count: int = 1) -> None:
        """Call when AMF evicts a prefix from the store."""
        self._rw_evictions.record(count)

    def record_admission(self, accepted: bool = True) -> None:
        """Call when AMF makes an admission decision."""
        if accepted:
            self._rw_admissions.record()
        else:
            self._rw_rejections.record()

    def record_failure(self) -> None:
        """Call when a restore operation fails."""
        self._rw_failures.record()

    def record_decode_step(
        self,
        acceptance_ema: float = 0.0,
        acceptance_variance: float = 0.0,
        spec_depth: float = 0.0,
        entropy_ema: float = 0.0,
        tps_current: float = 0.0,
        tps_delta: float = 0.0,
        lane_current: float = 0.0,
        lane_hold_remaining: float = 0.0,
        starvation_counter: float = 0.0,
        holo_future_safe: float = 0.0,
        power_watts_norm: float = 0.0,
    ) -> None:
        """Update decode-domain slots directly from the Rust scheduler / engine.

        Called from Worker on each speculative decode step. Zero-latency — writes
        directly to the tensor under the tensor lock (no queue, same pattern as
        record_decode).
        """
        with self._tensor_lock:
            self._tensor[Slot.ACCEPTANCE_EMA]      = _safe_float(acceptance_ema)
            self._tensor[Slot.ACCEPTANCE_VARIANCE]  = _safe_float(acceptance_variance)
            self._tensor[Slot.CURRENT_SPEC_DEPTH]   = _safe_float(spec_depth)
            self._tensor[Slot.ENTROPY_EMA]          = _safe_float(entropy_ema)
            self._tensor[Slot.TPS_CURRENT]          = _safe_float(tps_current)
            self._tensor[Slot.TPS_DELTA]            = _safe_float(tps_delta)
            self._tensor[Slot.LANE_CURRENT]         = _safe_float(lane_current)
            self._tensor[Slot.LANE_HOLD_REMAINING]  = _safe_float(lane_hold_remaining)
            self._tensor[Slot.STARVATION_COUNTER]   = _safe_float(starvation_counter)
            self._tensor[Slot.HOLO_FUTURE_SAFE]     = _safe_float(holo_future_safe)
            self._tensor[Slot.POWER_WATTS_NORM]     = _safe_float(power_watts_norm)

    # ── background polling ──────────────────────────────────────────────────

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            t0 = time.monotonic()
            try:
                self._refresh()
            except Exception:
                pass  # never crash the poll thread
            elapsed = time.monotonic() - t0
            sleep_for = max(0.0, self._poll_interval - elapsed)
            self._stop_event.wait(timeout=sleep_for)

    def _refresh(self) -> None:
        global_snap  = self._read_global_metrics()
        coord_health = self._fetch_coordinator_health()
        coord_nodes  = self._fetch_coordinator_nodes()
        self._node_snapshots = coord_nodes

        now = time.monotonic()
        dt  = max(now - self._prev_ts, 1e-6)
        self._prev_ts = now

        # ── deltas from GLOBAL_METRICS ────────────────────────────────
        def cdelta(name: str) -> float:
            cur = _safe_float(global_snap["counters"].get(name, 0.0))
            prev = self._prev_counters.get(name, cur)
            delta = max(0.0, cur - prev)
            self._prev_counters[name] = cur
            return delta

        def gauge(name: str, default: float = 0.0) -> float:
            return _safe_float(global_snap["gauges"].get(name, default), default)

        hit_delta          = cdelta("korith_amf_hits_total")
        miss_delta         = cdelta("korith_amf_misses_total")
        tokens_saved_delta = cdelta("korith_amf_tokens_saved_total")
        savings_ms_delta   = cdelta("korith_amf_savings_ms_total")
        evictions_delta    = cdelta("korith_amf_evictions_total")
        admissions_delta   = cdelta("korith_amf_admissions_total")
        rejections_delta   = cdelta("korith_amf_admission_rejects_total")
        failures_delta     = cdelta("korith_amf_failures_total")

        # Update rate windows from GLOBAL_METRICS deltas (in addition to hooks)
        if evictions_delta > 0:
            self._rw_evictions.record(evictions_delta)
        if admissions_delta > 0:
            self._rw_admissions.record(admissions_delta)
        if rejections_delta > 0:
            self._rw_rejections.record(rejections_delta)
        if failures_delta > 0:
            self._rw_failures.record(failures_delta)

        # Storage stats from gauges
        storage_bytes  = gauge("korith_amf_storage_bytes_total")
        budget_bytes   = gauge("korith_amf_storage_budget_bytes", 1.0)
        entry_count    = gauge("korith_amf_entries_total")
        hot_entries    = gauge("korith_amf_hot_entries")
        warm_entries   = gauge("korith_amf_warm_entries")
        cold_entries   = gauge("korith_amf_cold_entries")
        eviction_press = gauge("korith_amf_eviction_pressure", 1.0)
        instant_tps    = gauge("korith_amf_instant_tps")
        rolling_tps    = gauge("korith_amf_rolling_tps")
        mem_pressure   = gauge("korith_amf_memory_pressure")

        # ── derived signals ───────────────────────────────────────────
        total_req   = max(1, self._total_hits + self._total_misses)
        hit_rate    = _safe_float(self._total_hits / total_req)
        reuse_ratio = hit_rate

        storage_util = _clamp01(
            _safe_float(storage_bytes / max(budget_bytes, 1.0))
        )

        entry_total = max(1.0, entry_count)
        hot_ratio   = _clamp01(hot_entries / entry_total)
        cold_ratio  = _clamp01(cold_entries / entry_total)

        restore_avg = (
            sum(self._restore_ms_samples) / len(self._restore_ms_samples)
            if self._restore_ms_samples else 0.0
        )

        # ── coordinator-derived signals ───────────────────────────────
        coord_keys  = _safe_float((coord_health or {}).get("keys", 0))
        coord_nodes_count = _safe_float((coord_health or {}).get("nodes", 0))

        node_hit_rates: List[float] = []
        node_warmths:   List[float] = []
        total_node_entries = 0.0
        total_node_bytes   = 0.0

        for ns in coord_nodes:
            node_hit_rates.append(_safe_float(ns.get("hit_rate", 0.0)))
            node_warmths.append(_safe_float(ns.get("warmth", 0.0)))
            total_node_entries += _safe_float(ns.get("entries", 0.0))
            total_node_bytes   += _safe_float(ns.get("storage_bytes", 0.0))

        avg_node_hit_rate = (
            sum(node_hit_rates) / len(node_hit_rates) if node_hit_rates else 0.0
        )
        avg_node_warmth = (
            sum(node_warmths) / len(node_warmths) if node_warmths else 0.0
        )

        # ── queue depths ──────────────────────────────────────────────
        q_hit_depth  = float(self._q_hit.qsize())  if self._q_hit  else 0.0
        q_miss_depth = float(self._q_miss.qsize()) if self._q_miss else 0.0

        # ── assemble tensor ───────────────────────────────────────────
        t = np.zeros(TENSOR_SIZE, dtype=np.float32)

        t[Slot.HIT_DELTA]          = float(hit_delta)
        t[Slot.MISS_DELTA]         = float(miss_delta)
        t[Slot.HIT_RATE]           = float(_clamp01(hit_rate))
        t[Slot.TOKENS_SAVED_DELTA] = float(tokens_saved_delta)
        t[Slot.SAVINGS_MS_DELTA]   = float(savings_ms_delta)
        t[Slot.RESTORE_MS_AVG]     = float(restore_avg)
        t[Slot.EVICTION_RATE]      = float(self._rw_evictions.rate())
        t[Slot.ADMISSION_RATE]     = float(self._rw_admissions.rate())
        t[Slot.REJECTION_RATE]     = float(self._rw_rejections.rate())
        t[Slot.FAILURE_RATE]       = float(self._rw_failures.rate())
        t[Slot.STORAGE_UTIL]       = float(storage_util)
        t[Slot.EVICTION_PRESSURE]  = float(_clamp01(eviction_press))
        t[Slot.ENTRY_COUNT_NORM]   = float(entry_count / 1_000.0)
        t[Slot.HOT_ENTRY_RATIO]    = float(hot_ratio)
        t[Slot.COLD_ENTRY_RATIO]   = float(cold_ratio)
        t[Slot.INSTANT_TPS_NORM]   = float(instant_tps / 1_000.0)
        t[Slot.ROLLING_TPS_NORM]   = float(rolling_tps / 1_000.0)
        t[Slot.NODE_COUNT]         = float(coord_nodes_count)
        t[Slot.AVG_NODE_HIT_RATE]  = float(_clamp01(avg_node_hit_rate))
        t[Slot.AVG_NODE_WARMTH]    = float(_clamp01(avg_node_warmth))
        t[Slot.TOTAL_CACHE_ENTRIES] = float(total_node_entries / 1_000.0)
        t[Slot.TOTAL_CACHE_GB]     = float(total_node_bytes / 1e9)
        t[Slot.REQUEST_RATE]       = float(self._rw_requests.rate())
        t[Slot.HIT_STREAK_NORM]    = float(_clamp01(self._hit_streak / 100.0))
        t[Slot.MISS_STREAK_NORM]   = float(_clamp01(self._miss_streak / 20.0))
        t[Slot.REUSE_RATIO]        = float(_clamp01(reuse_ratio))
        t[Slot.UNIQUE_PREFIXES_NORM] = float(coord_keys / 1_000.0)
        t[Slot.HIT_QUEUE_NORM]     = float(_clamp01(q_hit_depth / 100.0))
        t[Slot.MISS_QUEUE_NORM]    = float(_clamp01(q_miss_depth / 100.0))
        t[Slot.ROI_EMA_NORM]       = float(_clamp01(self._roi_ema / 10.0))
        t[Slot.MEMORY_PRESSURE]    = float(_clamp01(mem_pressure))
        t[Slot.DECODE_LATENCY_NORM] = float(_clamp01(self._decode_ema / 10_000.0))

        # ── Dynamo fleet slots (54-69) ───────────────────────────────────
        if self._dynamo_source is not None:
            try:
                dm = self._dynamo_source.get_dynamo_metrics()
                t[Slot.DYNAMO_FLEET_HIT_RATE]      = float(_clamp01(dm.get("fleet_hit_rate", 0.0)))
                t[Slot.DYNAMO_FLEET_MISS_RATE]      = float(_clamp01(dm.get("fleet_miss_rate", 0.0)))
                t[Slot.DYNAMO_EVICTION_RATE]        = float(_safe_float(dm.get("eviction_rate", 0.0)))
                t[Slot.DYNAMO_BLOCK_COUNT_NORM]     = float(_safe_float(dm.get("block_count", 0)) / 100_000.0)
                t[Slot.DYNAMO_WORKER_COUNT]         = float(_safe_float(dm.get("worker_count", 0)))
                t[Slot.DYNAMO_AVG_WORKER_LOAD]      = float(_clamp01(dm.get("avg_worker_load", 0.0)))
                t[Slot.DYNAMO_MAX_WORKER_LOAD]      = float(_clamp01(dm.get("max_worker_load", 0.0)))
                t[Slot.DYNAMO_PREFILL_QUEUE_DEPTH]  = float(_safe_float(dm.get("prefill_queue_depth", 0)) / 100.0)
                t[Slot.DYNAMO_DECODE_QUEUE_DEPTH]   = float(_safe_float(dm.get("decode_queue_depth", 0)) / 100.0)
                t[Slot.DYNAMO_KV_OVERLAP_AVG]       = float(_clamp01(dm.get("kv_overlap_avg", 0.0)))
                t[Slot.DYNAMO_AMF_ROUTED_RATIO]     = float(_clamp01(dm.get("amf_routed_ratio", 0.0)))
                t[Slot.DYNAMO_RESTORE_RATIO]        = float(_safe_float(dm.get("restore_ratio", 0.0)))
                t[Slot.DYNAMO_CROSS_NODE_XFERS]     = float(_safe_float(dm.get("cross_node_xfers_per_s", 0.0)))
                total_blocks = max(1, _safe_float(dm.get("block_count", 1)))
                t[Slot.DYNAMO_TIER_G1_PCT]          = float(_clamp01(dm.get("tier_g1_blocks", 0) / total_blocks))
                t[Slot.DYNAMO_TIER_G2_PCT]          = float(_clamp01(dm.get("tier_g2_blocks", 0) / total_blocks))
                t[Slot.DYNAMO_TIER_G3G4_PCT]        = float(_clamp01(dm.get("tier_g3g4_blocks", 0) / total_blocks))
            except Exception:
                pass  # Leave Dynamo slots as zero if source fails.

        with self._tensor_lock:
            self._tensor = t

    # ── metrics source readers ──────────────────────────────────────────────

    def _read_global_metrics(self) -> Dict[str, Dict[str, float]]:
        """Read from the Python-side GLOBAL_METRICS registry."""
        try:
            from ..observability.metrics import GLOBAL_METRICS
            return GLOBAL_METRICS.snapshot()
        except Exception:
            return {"counters": {}, "gauges": {}}

    def _fetch_coordinator_health(self) -> Optional[Dict[str, Any]]:
        """GET /health from the AMF Coordinator."""
        if not self._coordinator_url:
            return None
        return self._http_get(f"{self._coordinator_url}/health")

    def _fetch_coordinator_nodes(self) -> List[Dict[str, float]]:
        """
        GET /metrics from the AMF Coordinator (Prometheus text) and parse
        per-node hit_rate, warmth, entries, storage_bytes.
        """
        if not self._coordinator_url:
            return []
        raw = self._http_get_text(f"{self._coordinator_url}/metrics")
        if not raw:
            return []
        return self._parse_node_metrics(raw)

    def _http_get(self, url: str) -> Optional[Dict[str, Any]]:
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=self._coordinator_timeout) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            obj = json.loads(body)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    def _http_get_text(self, url: str) -> str:
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=self._coordinator_timeout) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception:
            return ""

    @staticmethod
    def _parse_node_metrics(prometheus_text: str) -> List[Dict[str, float]]:
        """
        Parse Prometheus text into a list of per-node dicts.
        Handles lines like:
            korith_amf_node_hit_rate{node="n1"} 0.998
            korith_amf_node_entries{node="n1"} 816
            korith_amf_node_storage_bytes{node="n1"} 8765432100
            korith_amf_node_warmth{node="n1"} 0.92
        """
        nodes: Dict[str, Dict[str, float]] = {}

        METRIC_MAP = {
            "korith_amf_node_hit_rate":       "hit_rate",
            "korith_amf_node_entries":        "entries",
            "korith_amf_node_storage_bytes":  "storage_bytes",
            "korith_amf_node_warmth":         "warmth",
        }

        for line in prometheus_text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            for metric_name, field in METRIC_MAP.items():
                if not line.startswith(metric_name):
                    continue
                # Extract node label: metric_name{node="..."} value
                try:
                    brace_start = line.index("{")
                    brace_end   = line.index("}")
                    labels_raw  = line[brace_start + 1 : brace_end]
                    value_str   = line[brace_end + 1 :].strip()
                    value       = _safe_float(value_str)

                    node_id = ""
                    for part in labels_raw.split(","):
                        k, _, v = part.partition("=")
                        if k.strip() == "node":
                            node_id = v.strip().strip('"')
                    if node_id:
                        nodes.setdefault(node_id, {})
                        nodes[node_id][field] = value
                except (ValueError, IndexError):
                    pass

        return list(nodes.values())


# ── Dynamo fleet metrics source ───────────────────────────────────────────────

class DynamoMetricsSource:
    """
    Aggregates Dynamo KVBM fleet-level statistics for the metrics tensor.

    Subscribes to the same NATS event stream used by DynamoEventSubscriber and
    maintains rolling counters for block lifecycle events.  Optionally queries
    the Dynamo router HTTP endpoint for per-worker load.

    Usage
    -----
        source = DynamoMetricsSource(
            nats_url="nats://localhost:4222",
            router_url="http://localhost:8000",
        )
        asyncio.create_task(source.start())          # in async context
        # or call source.start_sync() from a thread

        metrics = source.get_dynamo_metrics()        # always returns a dict
    """

    # How many seconds of events to keep for rate calculations.
    _RATE_WINDOW_S: float = 10.0

    def __init__(
        self,
        nats_url: str = "nats://localhost:4222",
        nats_subject_prefix: str = "dynamo.kv.block",
        router_url: str = "",
        router_timeout_s: float = 0.25,
    ) -> None:
        self._nats_url = nats_url
        self._subject_prefix = nats_subject_prefix
        self._router_url = router_url.rstrip("/")
        self._router_timeout = router_timeout_s

        # ── rolling event windows ─────────────────────────────────────
        self._rw_created  = _RateWindow(self._RATE_WINDOW_S)
        self._rw_evicted  = _RateWindow(self._RATE_WINDOW_S)
        self._rw_stored   = _RateWindow(self._RATE_WINDOW_S)
        self._rw_xfer     = _RateWindow(self._RATE_WINDOW_S)

        # ── counters maintained from events ───────────────────────────
        self._lock = threading.Lock()
        self._block_count:      int   = 0
        self._hit_count:        int   = 0
        self._miss_count:       int   = 0
        self._amf_routed:       int   = 0
        self._total_routed:     int   = 0
        self._restore_count:    int   = 0
        self._recompute_count:  int   = 0
        self._cross_node_xfers: int   = 0

        # Tier block counts — set by block metadata when available.
        self._tier_g1_blocks:   int   = 0   # GPU HBM
        self._tier_g2_blocks:   int   = 0   # CPU RAM
        self._tier_g3g4_blocks: int   = 0   # NVMe / remote

        # ── KV overlap EMA ────────────────────────────────────────────
        self._kv_overlap_ema: float = 0.0
        _KV_OVERLAP_ALPHA = 0.05

        # ── last router snapshot ──────────────────────────────────────
        self._last_router_snap: Dict[str, Any] = {}
        self._last_router_ts:   float = 0.0
        self._router_cache_ttl: float = 2.0   # re-query every 2 s

        # ── background threads / tasks ────────────────────────────────
        self._stop_event = threading.Event()
        self._nats_thread: Optional[threading.Thread] = None
        self._router_thread: Optional[threading.Thread] = None

        # Store KV overlap alpha as instance var for use in _handle_event
        self._kv_overlap_alpha = _KV_OVERLAP_ALPHA

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def start_sync(self) -> None:
        """Start background threads (sync entry point for non-async hosts)."""
        self._nats_thread = threading.Thread(
            target=self._nats_thread_main,
            name="dynamo-metrics-nats",
            daemon=True,
        )
        self._nats_thread.start()

        if self._router_url:
            self._router_thread = threading.Thread(
                target=self._router_poll_loop,
                name="dynamo-metrics-router",
                daemon=True,
            )
            self._router_thread.start()

    def stop(self) -> None:
        """Stop all background threads."""
        self._stop_event.set()
        if self._nats_thread:
            self._nats_thread.join(timeout=2.0)
        if self._router_thread:
            self._router_thread.join(timeout=2.0)

    # ── public API ────────────────────────────────────────────────────────────

    def get_dynamo_metrics(self) -> Dict[str, Any]:
        """
        Return a snapshot dict consumed by MetricsCollector._refresh().

        Keys match the slot-fill code in MetricsCollector exactly:
          fleet_hit_rate, fleet_miss_rate, eviction_rate, block_count,
          worker_count, avg_worker_load, max_worker_load,
          prefill_queue_depth, decode_queue_depth,
          kv_overlap_avg, amf_routed_ratio, restore_ratio,
          cross_node_xfers_per_s,
          tier_g1_blocks, tier_g2_blocks, tier_g3g4_blocks
        """
        with self._lock:
            total_req = max(1, self._hit_count + self._miss_count)
            fleet_hit  = _clamp01(self._hit_count  / total_req)
            fleet_miss = _clamp01(self._miss_count / total_req)

            total_routed = max(1, self._total_routed)
            amf_ratio = _clamp01(self._amf_routed / total_routed)

            total_decisions = max(1, self._restore_count + self._recompute_count)
            restore_ratio = self._restore_count / total_decisions

            block_count    = self._block_count
            tier_g1        = self._tier_g1_blocks
            tier_g2        = self._tier_g2_blocks
            tier_g3g4      = self._tier_g3g4_blocks
            kv_overlap_avg = self._kv_overlap_ema
            xfers          = self._cross_node_xfers

        eviction_rate = self._rw_evicted.rate()
        xfer_rate     = self._rw_xfer.rate()

        # Worker load from latest router snapshot
        router_snap    = self._last_router_snap
        worker_count   = _safe_float(router_snap.get("worker_count", 0))
        avg_load       = _safe_float(router_snap.get("avg_worker_load", 0.0))
        max_load       = _safe_float(router_snap.get("max_worker_load", 0.0))
        prefill_queue  = _safe_float(router_snap.get("prefill_queue_depth", 0))
        decode_queue   = _safe_float(router_snap.get("decode_queue_depth", 0))

        return {
            "fleet_hit_rate":       fleet_hit,
            "fleet_miss_rate":      fleet_miss,
            "eviction_rate":        eviction_rate,
            "block_count":          block_count,
            "worker_count":         worker_count,
            "avg_worker_load":      avg_load,
            "max_worker_load":      max_load,
            "prefill_queue_depth":  prefill_queue,
            "decode_queue_depth":   decode_queue,
            "kv_overlap_avg":       kv_overlap_avg,
            "amf_routed_ratio":     amf_ratio,
            "restore_ratio":        restore_ratio,
            "cross_node_xfers_per_s": xfer_rate,
            "tier_g1_blocks":       tier_g1,
            "tier_g2_blocks":       tier_g2,
            "tier_g3g4_blocks":     tier_g3g4,
        }

    def record_routing_decision(
        self,
        used_amf: bool,
        kv_overlap: float = 0.0,
        used_restore: bool = False,
    ) -> None:
        """Call from AmfRouterPlugin after scoring a request."""
        with self._lock:
            self._total_routed += 1
            if used_amf:
                self._amf_routed += 1
            if used_restore:
                self._restore_count += 1
            else:
                self._recompute_count += 1
            self._kv_overlap_ema = (
                self._kv_overlap_alpha * _clamp01(kv_overlap)
                + (1.0 - self._kv_overlap_alpha) * self._kv_overlap_ema
            )

    # ── NATS background thread ────────────────────────────────────────────────

    def _nats_thread_main(self) -> None:
        """Run the asyncio NATS event loop in a dedicated thread."""
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._nats_subscribe_loop())
        except Exception:
            pass
        finally:
            loop.close()

    async def _nats_subscribe_loop(self) -> None:
        try:
            import nats  # type: ignore[import]
        except ImportError:
            # nats-py not installed; silently skip NATS subscription.
            return

        import asyncio

        nc = await nats.connect(self._nats_url)
        try:
            subjects = [
                f"{self._subject_prefix}.created",
                f"{self._subject_prefix}.evicted",
                f"{self._subject_prefix}.stored",
                f"{self._subject_prefix}.moved",
            ]
            for subj in subjects:
                await nc.subscribe(subj, cb=self._make_nats_callback(subj))

            # Wait until stop requested
            while not self._stop_event.is_set():
                await asyncio.sleep(0.5)
        finally:
            await nc.drain()

    def _make_nats_callback(self, subject: str):
        import asyncio

        async def _cb(msg):
            try:
                payload = json.loads(msg.data.decode("utf-8", errors="replace"))
            except Exception:
                payload = {}
            self._handle_event(subject, payload)

        return _cb

    def _handle_event(self, subject: str, payload: Dict[str, Any]) -> None:
        """Update counters from a Dynamo KVBM lifecycle event."""
        tier = str(payload.get("storage_tier", "")).lower()
        num_tokens = int(_safe_float(payload.get("num_tokens", 0)))

        with self._lock:
            if subject.endswith(".created"):
                self._block_count += 1
                self._rw_created.record()
                self._miss_count  += 1
                # Update tier counters
                self._update_tier(tier, +1)

            elif subject.endswith(".evicted"):
                self._block_count = max(0, self._block_count - 1)
                self._rw_evicted.record()
                self._update_tier(tier, -1)

            elif subject.endswith(".stored"):
                self._hit_count += 1
                self._rw_stored.record()

            elif subject.endswith(".moved"):
                # Cross-node KV transfer
                self._cross_node_xfers += 1
                self._rw_xfer.record()
                # Update tier if src/dst differ
                src_tier = str(payload.get("src_tier", "")).lower()
                dst_tier = str(payload.get("dst_tier", tier)).lower()
                if src_tier != dst_tier:
                    self._update_tier(src_tier, -1)
                    self._update_tier(dst_tier, +1)

    def _update_tier(self, tier: str, delta: int) -> None:
        """Adjust per-tier block counters (must hold self._lock)."""
        if tier in ("vram", "hbm", "g1"):
            self._tier_g1_blocks   = max(0, self._tier_g1_blocks   + delta)
        elif tier in ("ram", "dram", "g2"):
            self._tier_g2_blocks   = max(0, self._tier_g2_blocks   + delta)
        elif tier in ("nvme", "remote", "g3", "g4", "g3g4"):
            self._tier_g3g4_blocks = max(0, self._tier_g3g4_blocks + delta)

    # ── router poll loop ──────────────────────────────────────────────────────

    def _router_poll_loop(self) -> None:
        while not self._stop_event.is_set():
            snap = self._fetch_router_workers()
            if snap:
                self._last_router_snap = snap
            self._stop_event.wait(timeout=self._router_cache_ttl)

    def _fetch_router_workers(self) -> Dict[str, Any]:
        """GET /v1/workers from the Dynamo router and compute load stats."""
        if not self._router_url:
            return {}
        try:
            req = urllib.request.Request(
                f"{self._router_url}/v1/workers", method="GET"
            )
            with urllib.request.urlopen(req, timeout=self._router_timeout) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            data = json.loads(body)
        except Exception:
            return {}

        workers = data if isinstance(data, list) else data.get("workers", [])
        if not workers:
            return {}

        loads = [_safe_float(w.get("load", w.get("utilization", 0.0))) for w in workers]
        prefill_q = sum(
            _safe_float(w.get("prefill_queue", w.get("pending_prefills", 0)))
            for w in workers
        )
        decode_q = sum(
            _safe_float(w.get("decode_queue", w.get("pending_decodes", 0)))
            for w in workers
        )

        return {
            "worker_count":       float(len(loads)),
            "avg_worker_load":    _clamp01(sum(loads) / max(1, len(loads))),
            "max_worker_load":    _clamp01(max(loads, default=0.0)),
            "prefill_queue_depth": prefill_q,
            "decode_queue_depth":  decode_q,
        }

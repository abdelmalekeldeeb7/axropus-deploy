"""Tests for platform/reasoning/metrics_collector.py (Component 1)."""

from __future__ import annotations

import math
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from platform.reasoning.metrics_collector import (
    TENSOR_SIZE,
    MetricsCollector,
    Slot,
    _RateWindow,
    _clamp01,
    _safe_float,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_collector(**kwargs) -> MetricsCollector:
    """Return a collector with no coordinator URL (offline mode)."""
    return MetricsCollector(coordinator_url="", **kwargs)


# ── _safe_float ───────────────────────────────────────────────────────────────

class TestSafeFloat:
    def test_normal(self):
        assert _safe_float(3.14) == pytest.approx(3.14)

    def test_string(self):
        assert _safe_float("2.5") == pytest.approx(2.5)

    def test_none(self):
        assert _safe_float(None) == 0.0

    def test_nan(self):
        assert _safe_float(float("nan")) == 0.0

    def test_inf(self):
        assert _safe_float(float("inf")) == 0.0

    def test_default(self):
        assert _safe_float("bad", default=7.0) == 7.0


# ── _clamp01 ─────────────────────────────────────────────────────────────────

class TestClamp01:
    def test_middle(self):
        assert _clamp01(0.5) == pytest.approx(0.5)

    def test_above_one(self):
        assert _clamp01(1.5) == pytest.approx(1.0)

    def test_below_zero(self):
        assert _clamp01(-0.5) == pytest.approx(0.0)


# ── _RateWindow ───────────────────────────────────────────────────────────────

class TestRateWindow:
    def test_empty_rate_is_zero(self):
        rw = _RateWindow(window_s=10.0)
        assert rw.rate() == 0.0

    def test_single_event(self):
        rw = _RateWindow(window_s=10.0)
        rw.record(1.0)
        # Rate should be > 0 immediately after recording
        assert rw.rate() > 0.0

    def test_high_volume(self):
        rw = _RateWindow(window_s=10.0)
        for _ in range(100):
            rw.record(1.0)
        # With 100 events in a very short span, rate should be very high
        assert rw.rate() > 10.0

    def test_window_eviction(self):
        rw = _RateWindow(window_s=0.05)  # 50ms window
        rw.record(1.0)
        time.sleep(0.1)
        # Events should have been evicted from the window
        assert rw.rate() == pytest.approx(0.0, abs=1.0)


# ── MetricsCollector — tensor shape and dtype ─────────────────────────────────

class TestTensorStructure:
    def test_shape(self):
        c = _make_collector()
        t = c.get_metrics_tensor()
        assert t.shape == (TENSOR_SIZE,)

    def test_dtype(self):
        c = _make_collector()
        t = c.get_metrics_tensor()
        assert t.dtype == np.float32

    def test_initial_zeros(self):
        c = _make_collector()
        t = c.get_metrics_tensor()
        assert np.all(t == 0.0)

    def test_tensor_is_copy(self):
        c = _make_collector()
        t1 = c.get_metrics_tensor()
        t1[0] = 999.0
        t2 = c.get_metrics_tensor()
        assert t2[0] == 0.0  # original unchanged


# ── record_hit / record_miss ──────────────────────────────────────────────────

class TestEventHooks:
    def test_hit_increments_total(self):
        c = _make_collector()
        c.record_hit(prefix_hash="abc", roi=2.0, restore_ms=10.0)
        assert c._total_hits == 1
        assert c._total_requests == 1

    def test_miss_increments_total(self):
        c = _make_collector()
        c.record_miss(prefix_hash="xyz")
        assert c._total_misses == 1
        assert c._total_requests == 1

    def test_hit_resets_miss_streak(self):
        c = _make_collector()
        c.record_miss()
        c.record_miss()
        c.record_hit()
        assert c._miss_streak == 0
        assert c._hit_streak == 1

    def test_miss_resets_hit_streak(self):
        c = _make_collector()
        c.record_hit()
        c.record_hit()
        c.record_miss()
        assert c._hit_streak == 0
        assert c._miss_streak == 1

    def test_roi_ema_updates(self):
        c = _make_collector()
        initial = c._roi_ema
        c.record_hit(roi=5.0)
        assert c._roi_ema > initial

    def test_restore_ms_stored(self):
        c = _make_collector()
        c.record_hit(restore_ms=42.0)
        assert 42.0 in c._restore_ms_samples

    def test_decode_ema_updates(self):
        c = _make_collector()
        c.record_decode(11_400.0)
        assert c._decode_ema > 0.0


# ── get_metrics_dict ──────────────────────────────────────────────────────────

class TestMetricsDict:
    def test_keys(self):
        c = _make_collector()
        d = c.get_metrics_dict()
        assert "hit_rate" in d
        assert "miss_count_delta" in d
        assert "roi_ema_norm" in d
        # prefill slots (32) + decode slots (11 named) = 43 total named entries
        assert "acceptance_ema" in d
        assert "entropy_ema" in d
        assert "holo_future_safe" in d

    def test_values_are_floats(self):
        c = _make_collector()
        d = c.get_metrics_dict()
        for k, v in d.items():
            assert isinstance(v, float), f"Expected float for {k}, got {type(v)}"


# ── _refresh with mocked GLOBAL_METRICS ───────────────────────────────────────

class TestRefresh:
    def _make_mock_metrics(self, hits=10, misses=2, savings_ms=5000.0, evictions=3):
        mock = MagicMock()
        mock.snapshot.return_value = {
            "counters": {
                "korith_amf_hits_total":           float(hits),
                "korith_amf_misses_total":         float(misses),
                "korith_amf_tokens_saved_total":   float(hits * 120_000),
                "korith_amf_savings_ms_total":     savings_ms,
                "korith_amf_evictions_total":      float(evictions),
                "korith_amf_admissions_total":     float(hits),
                "korith_amf_admission_rejects_total": 0.0,
                "korith_amf_failures_total":       0.0,
            },
            "gauges": {
                "korith_amf_storage_bytes_total":   8_000_000_000.0,
                "korith_amf_storage_budget_bytes":  20_000_000_000.0,
                "korith_amf_entries_total":         816.0,
                "korith_amf_hot_entries":           500.0,
                "korith_amf_warm_entries":          200.0,
                "korith_amf_cold_entries":          116.0,
                "korith_amf_eviction_pressure":     0.4,
                "korith_amf_instant_tps":           88.0,
                "korith_amf_rolling_tps":           85.0,
                "korith_amf_memory_pressure":       0.3,
            },
        }
        return mock

    def test_hit_rate_computed(self):
        c = _make_collector()
        # Simulate 10 hits, 0 misses via hooks
        for _ in range(10):
            c.record_hit()

        with patch(
            "platform.reasoning.metrics_collector.MetricsCollector._read_global_metrics",
            return_value={"counters": {}, "gauges": {}},
        ):
            c._refresh()

        t = c.get_metrics_tensor()
        assert t[Slot.HIT_RATE] == pytest.approx(1.0)

    def test_storage_utilization(self):
        c = _make_collector()
        mock = self._make_mock_metrics()

        with patch(
            "platform.reasoning.metrics_collector.MetricsCollector._read_global_metrics",
            return_value=mock.snapshot(),
        ):
            c._refresh()

        t = c.get_metrics_tensor()
        # 8 GB / 20 GB = 0.4
        assert t[Slot.STORAGE_UTIL] == pytest.approx(0.4, abs=0.01)

    def test_hot_entry_ratio(self):
        c = _make_collector()
        mock = self._make_mock_metrics()

        with patch(
            "platform.reasoning.metrics_collector.MetricsCollector._read_global_metrics",
            return_value=mock.snapshot(),
        ):
            c._refresh()

        t = c.get_metrics_tensor()
        # 500 hot / 816 total ≈ 0.613
        assert t[Slot.HOT_ENTRY_RATIO] == pytest.approx(500 / 816, abs=0.01)

    def test_eviction_pressure_clamped(self):
        c = _make_collector()

        with patch(
            "platform.reasoning.metrics_collector.MetricsCollector._read_global_metrics",
            return_value={
                "counters": {},
                "gauges": {"korith_amf_eviction_pressure": 99.0},
            },
        ):
            c._refresh()

        t = c.get_metrics_tensor()
        assert t[Slot.EVICTION_PRESSURE] == pytest.approx(1.0)

    def test_decode_latency_norm_range(self):
        c = _make_collector()
        c.record_decode(5_000.0)  # 5 000 ms → norm = 0.5

        with patch(
            "platform.reasoning.metrics_collector.MetricsCollector._read_global_metrics",
            return_value={"counters": {}, "gauges": {}},
        ):
            c._refresh()

        t = c.get_metrics_tensor()
        # After one EMA step: ema = 0.10 * 5000 = 500, norm = 500 / 10000 = 0.05
        assert 0.0 < t[Slot.DECODE_LATENCY_NORM] <= 1.0

    def test_no_nan_in_tensor(self):
        c = _make_collector()
        with patch(
            "platform.reasoning.metrics_collector.MetricsCollector._read_global_metrics",
            return_value={"counters": {}, "gauges": {}},
        ):
            c._refresh()

        t = c.get_metrics_tensor()
        assert not np.any(np.isnan(t)), "Tensor must never contain NaN"

    def test_no_inf_in_tensor(self):
        c = _make_collector()
        with patch(
            "platform.reasoning.metrics_collector.MetricsCollector._read_global_metrics",
            return_value={"counters": {}, "gauges": {}},
        ):
            c._refresh()

        t = c.get_metrics_tensor()
        assert not np.any(np.isinf(t)), "Tensor must never contain Inf"


# ── Prometheus parser ─────────────────────────────────────────────────────────

class TestParseNodeMetrics:
    SAMPLE = """\
# HELP korith_amf_node_hit_rate AMF hit rate per node
# TYPE korith_amf_node_hit_rate gauge
korith_amf_node_hit_rate{node="gpu-0"} 0.998
korith_amf_node_hit_rate{node="gpu-1"} 0.975
korith_amf_node_entries{node="gpu-0"} 816.0
korith_amf_node_entries{node="gpu-1"} 400.0
korith_amf_node_storage_bytes{node="gpu-0"} 8765432100.0
korith_amf_node_warmth{node="gpu-0"} 0.92
korith_amf_node_warmth{node="gpu-1"} 0.80
"""

    def test_parses_two_nodes(self):
        nodes = MetricsCollector._parse_node_metrics(self.SAMPLE)
        assert len(nodes) == 2

    def test_hit_rate_parsed(self):
        nodes = MetricsCollector._parse_node_metrics(self.SAMPLE)
        hit_rates = {n.get("hit_rate") for n in nodes}
        assert 0.998 in hit_rates
        assert 0.975 in hit_rates

    def test_entries_parsed(self):
        nodes = MetricsCollector._parse_node_metrics(self.SAMPLE)
        entries = {n.get("entries") for n in nodes}
        assert 816.0 in entries

    def test_storage_bytes_parsed(self):
        nodes = MetricsCollector._parse_node_metrics(self.SAMPLE)
        storage = [n.get("storage_bytes", 0) for n in nodes if "storage_bytes" in n]
        assert len(storage) == 1
        assert storage[0] == pytest.approx(8_765_432_100.0)

    def test_empty_input(self):
        nodes = MetricsCollector._parse_node_metrics("")
        assert nodes == []

    def test_comments_ignored(self):
        nodes = MetricsCollector._parse_node_metrics("# just a comment\n")
        assert nodes == []


# ── start/stop lifecycle ──────────────────────────────────────────────────────

class TestLifecycle:
    def test_start_stop(self):
        c = _make_collector(poll_interval_s=0.02)
        c.start()
        time.sleep(0.06)   # allow 2-3 poll cycles
        c.stop()
        assert not c._thread.is_alive()

    def test_tensor_updates_after_poll(self):
        """After polling with real GLOBAL_METRICS, tensor should have no NaN."""
        c = _make_collector(poll_interval_s=0.02)
        c.start()
        time.sleep(0.08)
        c.stop()
        t = c.get_metrics_tensor()
        assert not np.any(np.isnan(t))


# ── queue depth integration ───────────────────────────────────────────────────

class TestQueueDepth:
    def test_hit_queue_depth_reflected(self):
        import queue as q_mod
        q_hit  = q_mod.Queue()
        q_miss = q_mod.Queue()
        for _ in range(50):
            q_hit.put("job")

        c = MetricsCollector(
            coordinator_url="",
            worker_hit_queue=q_hit,
            worker_miss_queue=q_miss,
        )
        with patch(
            "platform.reasoning.metrics_collector.MetricsCollector._read_global_metrics",
            return_value={"counters": {}, "gauges": {}},
        ):
            c._refresh()

        t = c.get_metrics_tensor()
        # 50 jobs / 100 = 0.5
        assert t[Slot.HIT_QUEUE_NORM] == pytest.approx(0.5, abs=0.01)


# ── record_decode_step ────────────────────────────────────────────────────────

class TestRecordDecodeStep:
    def test_updates_acceptance_ema(self):
        c = _make_collector()
        c.record_decode_step(acceptance_ema=0.85)
        t = c.get_metrics_tensor()
        assert t[Slot.ACCEPTANCE_EMA] == pytest.approx(0.85)

    def test_updates_entropy_ema(self):
        c = _make_collector()
        c.record_decode_step(entropy_ema=0.62)
        t = c.get_metrics_tensor()
        assert t[Slot.ENTROPY_EMA] == pytest.approx(0.62)

    def test_updates_lane_hold_remaining(self):
        c = _make_collector()
        c.record_decode_step(lane_hold_remaining=20.0)
        t = c.get_metrics_tensor()
        assert t[Slot.LANE_HOLD_REMAINING] == pytest.approx(20.0)

    def test_updates_power_watts_norm(self):
        c = _make_collector()
        c.record_decode_step(power_watts_norm=0.73)
        t = c.get_metrics_tensor()
        assert t[Slot.POWER_WATTS_NORM] == pytest.approx(0.73)

    def test_tensor_size_is_80(self):
        c = _make_collector()
        t = c.get_metrics_tensor()
        assert t.shape == (80,)
        assert TENSOR_SIZE == 80

    def test_zero_args_leaves_decode_slots_zero(self):
        c = _make_collector()
        c.record_decode_step()
        t = c.get_metrics_tensor()
        assert t[Slot.ACCEPTANCE_EMA] == pytest.approx(0.0)
        assert t[Slot.ENTROPY_EMA] == pytest.approx(0.0)


# ── DynamoMetricsSource ───────────────────────────────────────────────────────

from platform.reasoning.metrics_collector import DynamoMetricsSource  # noqa: E402


def _make_dynamo_source(**kwargs) -> DynamoMetricsSource:
    return DynamoMetricsSource(
        nats_url="nats://localhost:4222",
        router_url="",
        **kwargs,
    )


class TestDynamoMetricsSourceDefaults:
    def test_get_dynamo_metrics_returns_dict(self):
        src = _make_dynamo_source()
        dm = src.get_dynamo_metrics()
        assert isinstance(dm, dict)

    def test_all_required_keys_present(self):
        src = _make_dynamo_source()
        dm = src.get_dynamo_metrics()
        required = {
            "fleet_hit_rate", "fleet_miss_rate", "eviction_rate", "block_count",
            "worker_count", "avg_worker_load", "max_worker_load",
            "prefill_queue_depth", "decode_queue_depth",
            "kv_overlap_avg", "amf_routed_ratio", "restore_ratio",
            "cross_node_xfers_per_s",
            "tier_g1_blocks", "tier_g2_blocks", "tier_g3g4_blocks",
        }
        assert required.issubset(dm.keys())

    def test_initial_values_are_zero_or_finite(self):
        src = _make_dynamo_source()
        dm = src.get_dynamo_metrics()
        for k, v in dm.items():
            assert math.isfinite(float(v)), f"Non-finite value for key {k}: {v}"


class TestDynamoSlotsPopulatedWhenSourceConnected:
    """Slots 54-69 must be non-zero when the DynamoMetricsSource has data."""

    def _make_collector_with_source(self) -> tuple:
        src = _make_dynamo_source()
        # Inject some synthetic data directly
        with src._lock:
            src._hit_count  = 80
            src._miss_count = 20
            src._block_count = 5000
            src._tier_g1_blocks = 3000
            src._tier_g2_blocks = 1500
            src._tier_g3g4_blocks = 500
            src._restore_count   = 60
            src._recompute_count = 10
        c = MetricsCollector(coordinator_url="", dynamo_source=src)
        return c, src

    def test_fleet_hit_rate_populated(self):
        c, _ = self._make_collector_with_source()
        with patch(
            "platform.reasoning.metrics_collector.MetricsCollector._read_global_metrics",
            return_value={"counters": {}, "gauges": {}},
        ):
            c._refresh()
        t = c.get_metrics_tensor()
        # 80 hits / 100 total = 0.8
        assert t[Slot.DYNAMO_FLEET_HIT_RATE] == pytest.approx(0.8, abs=0.01)

    def test_fleet_miss_rate_populated(self):
        c, _ = self._make_collector_with_source()
        with patch(
            "platform.reasoning.metrics_collector.MetricsCollector._read_global_metrics",
            return_value={"counters": {}, "gauges": {}},
        ):
            c._refresh()
        t = c.get_metrics_tensor()
        assert t[Slot.DYNAMO_FLEET_MISS_RATE] == pytest.approx(0.2, abs=0.01)

    def test_block_count_norm_populated(self):
        c, _ = self._make_collector_with_source()
        with patch(
            "platform.reasoning.metrics_collector.MetricsCollector._read_global_metrics",
            return_value={"counters": {}, "gauges": {}},
        ):
            c._refresh()
        t = c.get_metrics_tensor()
        # 5000 / 100 000 = 0.05
        assert t[Slot.DYNAMO_BLOCK_COUNT_NORM] == pytest.approx(0.05, abs=0.001)

    def test_tier_percentages_sum_to_one(self):
        c, src = self._make_collector_with_source()
        with patch(
            "platform.reasoning.metrics_collector.MetricsCollector._read_global_metrics",
            return_value={"counters": {}, "gauges": {}},
        ):
            c._refresh()
        t = c.get_metrics_tensor()
        total = t[Slot.DYNAMO_TIER_G1_PCT] + t[Slot.DYNAMO_TIER_G2_PCT] + t[Slot.DYNAMO_TIER_G3G4_PCT]
        assert total == pytest.approx(1.0, abs=0.01)

    def test_restore_ratio_populated(self):
        c, _ = self._make_collector_with_source()
        with patch(
            "platform.reasoning.metrics_collector.MetricsCollector._read_global_metrics",
            return_value={"counters": {}, "gauges": {}},
        ):
            c._refresh()
        t = c.get_metrics_tensor()
        # 60 restores / (60+10) = ~0.857
        assert t[Slot.DYNAMO_RESTORE_RATIO] == pytest.approx(60 / 70, abs=0.01)

    def test_no_nan_in_dynamo_slots(self):
        c, _ = self._make_collector_with_source()
        with patch(
            "platform.reasoning.metrics_collector.MetricsCollector._read_global_metrics",
            return_value={"counters": {}, "gauges": {}},
        ):
            c._refresh()
        t = c.get_metrics_tensor()
        dynamo_slots = t[Slot.DYNAMO_FLEET_HIT_RATE : Slot.DYNAMO_TIER_G3G4_PCT + 1]
        assert not np.any(np.isnan(dynamo_slots))


class TestDynamoSlotsZeroWhenNoSource:
    """Slots 54-69 must be zero when dynamo_source=None (default)."""

    def test_dynamo_slots_are_zero(self):
        c = _make_collector()   # no dynamo_source
        with patch(
            "platform.reasoning.metrics_collector.MetricsCollector._read_global_metrics",
            return_value={"counters": {}, "gauges": {}},
        ):
            c._refresh()
        t = c.get_metrics_tensor()
        for slot in range(Slot.DYNAMO_FLEET_HIT_RATE, Slot.DYNAMO_TIER_G3G4_PCT + 1):
            assert t[slot] == pytest.approx(0.0), f"Slot {slot} should be zero without source"


class TestDynamoHandleEvent:
    """Unit tests for DynamoMetricsSource._handle_event."""

    def test_created_event_increments_block_count(self):
        src = _make_dynamo_source()
        src._handle_event("dynamo.kv.block.created", {"storage_tier": "vram"})
        assert src._block_count == 1
        assert src._tier_g1_blocks == 1

    def test_evicted_event_decrements_block_count(self):
        src = _make_dynamo_source()
        with src._lock:
            src._block_count = 5
            src._tier_g2_blocks = 5
        src._handle_event("dynamo.kv.block.evicted", {"storage_tier": "ram"})
        assert src._block_count == 4
        assert src._tier_g2_blocks == 4

    def test_stored_event_increments_hit_count(self):
        src = _make_dynamo_source()
        src._handle_event("dynamo.kv.block.stored", {})
        assert src._hit_count == 1

    def test_moved_event_increments_xfer_and_updates_tier(self):
        src = _make_dynamo_source()
        with src._lock:
            src._tier_g1_blocks = 3
        src._handle_event("dynamo.kv.block.moved", {
            "src_tier": "vram",
            "dst_tier": "nvme",
        })
        assert src._cross_node_xfers == 1
        assert src._tier_g1_blocks == 2
        assert src._tier_g3g4_blocks == 1

    def test_block_count_never_negative(self):
        src = _make_dynamo_source()
        src._handle_event("dynamo.kv.block.evicted", {"storage_tier": "vram"})
        assert src._block_count == 0   # clamp at 0


class TestDynamoRecordRoutingDecision:
    def test_amf_routed_ratio_updates(self):
        src = _make_dynamo_source()
        src.record_routing_decision(used_amf=True, kv_overlap=0.9, used_restore=True)
        src.record_routing_decision(used_amf=False, kv_overlap=0.0, used_restore=False)
        dm = src.get_dynamo_metrics()
        # 1 of 2 routed via AMF
        assert dm["amf_routed_ratio"] == pytest.approx(0.5, abs=0.01)

    def test_restore_ratio_updates(self):
        src = _make_dynamo_source()
        for _ in range(3):
            src.record_routing_decision(used_amf=True, used_restore=True)
        src.record_routing_decision(used_amf=False, used_restore=False)
        dm = src.get_dynamo_metrics()
        # 3 restores / 4 total = 0.75
        assert dm["restore_ratio"] == pytest.approx(0.75, abs=0.01)

    def test_kv_overlap_ema_updates(self):
        src = _make_dynamo_source()
        initial_ema = src._kv_overlap_ema
        src.record_routing_decision(used_amf=True, kv_overlap=1.0)
        dm = src.get_dynamo_metrics()
        assert dm["kv_overlap_avg"] > initial_ema


class TestDynamoNormalizationRanges:
    """All [0,1]-bounded slots must stay within range under extreme inputs."""

    def test_fleet_hit_rate_clamped(self):
        src = _make_dynamo_source()
        # overflow hits
        with src._lock:
            src._hit_count  = 10_000_000
            src._miss_count = 0
        dm = src.get_dynamo_metrics()
        assert 0.0 <= dm["fleet_hit_rate"] <= 1.0

    def test_amf_routed_ratio_clamped(self):
        src = _make_dynamo_source()
        with src._lock:
            src._amf_routed   = 99_999
            src._total_routed = 100
        dm = src.get_dynamo_metrics()
        assert 0.0 <= dm["amf_routed_ratio"] <= 1.0

    def test_tier_pcts_clamped(self):
        src = _make_dynamo_source()
        with src._lock:
            src._block_count      = 10
            src._tier_g1_blocks   = 20   # intentionally > total
            src._tier_g2_blocks   = 0
            src._tier_g3g4_blocks = 0
        c = MetricsCollector(coordinator_url="", dynamo_source=src)
        with patch(
            "platform.reasoning.metrics_collector.MetricsCollector._read_global_metrics",
            return_value={"counters": {}, "gauges": {}},
        ):
            c._refresh()
        t = c.get_metrics_tensor()
        assert 0.0 <= t[Slot.DYNAMO_TIER_G1_PCT] <= 1.0

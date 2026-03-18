"""Tests for platform/reasoning/decision_executor.py (Component 3)."""

from __future__ import annotations

from unittest.mock import MagicMock, call

import numpy as np
import pytest

from platform.reasoning.decision_executor import (
    OVERRIDE_HOT_PROTECT,
    OVERRIDE_ROI_GATE,
    OVERRIDE_THROTTLE_CAP,
    DecisionExecutor,
    ExecutionContext,
    ExecutionResult,
    _slot,
)
from platform.reasoning.metrics_collector import Slot
from platform.reasoning.reasoning_model import DecisionVector, _safe_fallback


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_tensor(**overrides) -> np.ndarray:
    """Return a zeroed 32-element tensor with named slot overrides."""
    t = np.zeros(32, dtype=np.float32)
    for slot_name, value in overrides.items():
        slot_idx = getattr(Slot, slot_name.upper())
        t[slot_idx] = value
    return t


def _make_dv(
    cache_admission: bool = True,
    admission_priority: float = 0.5,
    eviction_target=None,
    pre_warm_predictions=None,
    route_decision=None,
    batch_group=None,
    spec_decode_enable: bool = True,
    spec_decode_k: int = 4,
    throttle_back_pressure: float = 0.0,
    inference_ms: float = 30.0,
    is_fallback: bool = False,
) -> DecisionVector:
    return DecisionVector(
        cache_admission=cache_admission,
        admission_priority=admission_priority,
        eviction_target=eviction_target,
        pre_warm_predictions=pre_warm_predictions or [],
        route_decision=route_decision,
        batch_group=batch_group,
        spec_decode_enable=spec_decode_enable,
        spec_decode_k=spec_decode_k,
        throttle_back_pressure=throttle_back_pressure,
        inference_ms=inference_ms,
        is_fallback=is_fallback,
    )


def _make_coordinator(evict_returns: bool = True) -> MagicMock:
    c = MagicMock()
    c.evict.return_value = evict_returns
    c.lookup.return_value = [{"node_id": "gpu-0"}]
    return c


def _make_context(**tensor_kwargs) -> ExecutionContext:
    return ExecutionContext(
        tenant_id="tenant-test",
        node_id="gpu-0",
        worker_id="w-1",
        tensor=_make_tensor(**tensor_kwargs),
    )


def _make_executor(**kwargs) -> DecisionExecutor:
    coordinator = kwargs.pop("coordinator", _make_coordinator())
    return DecisionExecutor(coordinator_client=coordinator, **kwargs)


# ── _slot helper ──────────────────────────────────────────────────────────────

class TestSlotHelper:
    def test_reads_value(self):
        t = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert _slot(t, 1) == pytest.approx(2.0)

    def test_out_of_bounds_returns_zero(self):
        t = np.array([1.0], dtype=np.float32)
        assert _slot(t, 99) == 0.0


# ── ExecutionResult structure ─────────────────────────────────────────────────

class TestExecutionResultStructure:
    def test_result_is_dataclass(self):
        ex = _make_executor()
        result = ex.execute(_make_dv(), _make_context())
        assert isinstance(result, ExecutionResult)

    def test_decision_dict_present(self):
        ex = _make_executor()
        dv = _make_dv(admission_priority=0.7)
        result = ex.execute(dv, _make_context())
        assert result.decision["admission_priority"] == pytest.approx(0.7)

    def test_elapsed_ms_positive(self):
        ex = _make_executor()
        result = ex.execute(_make_dv(), _make_context())
        assert result.elapsed_ms >= 0.0

    def test_overrides_is_list(self):
        ex = _make_executor()
        result = ex.execute(_make_dv(), _make_context())
        assert isinstance(result.overrides, list)


# ── ROI gate ──────────────────────────────────────────────────────────────────

class TestRoiGate:
    def test_blocks_admission_when_roi_below_threshold(self):
        ex = _make_executor(roi_threshold=0.3)
        # roi_ema_norm = 0.1 → below threshold
        ctx = _make_context(roi_ema_norm=0.1)
        result = ex.execute(_make_dv(cache_admission=True), ctx)
        assert result.admitted is False
        assert OVERRIDE_ROI_GATE in result.overrides

    def test_allows_admission_when_roi_above_threshold(self):
        ex = _make_executor(roi_threshold=0.3)
        ctx = _make_context(roi_ema_norm=0.5)
        result = ex.execute(_make_dv(cache_admission=True), ctx)
        assert result.admitted is True
        assert OVERRIDE_ROI_GATE not in result.overrides

    def test_no_override_when_model_says_no_admission(self):
        """If model says don't cache, ROI gate doesn't add an override."""
        ex = _make_executor(roi_threshold=0.3)
        ctx = _make_context(roi_ema_norm=0.1)
        result = ex.execute(_make_dv(cache_admission=False), ctx)
        assert result.admitted is False
        assert OVERRIDE_ROI_GATE not in result.overrides

    def test_roi_at_threshold_not_blocked(self):
        """Exactly at threshold — should be allowed (strictly less than blocks)."""
        ex = _make_executor(roi_threshold=0.3)
        ctx = _make_context(roi_ema_norm=0.3)
        result = ex.execute(_make_dv(cache_admission=True), ctx)
        assert result.admitted is True

    def test_zero_roi_always_blocked(self):
        ex = _make_executor(roi_threshold=0.3)
        ctx = _make_context(roi_ema_norm=0.0)
        result = ex.execute(_make_dv(cache_admission=True), ctx)
        assert result.admitted is False


# ── Hot prefix protection ─────────────────────────────────────────────────────

class TestHotProtection:
    def test_blocks_eviction_when_hot_and_low_pressure(self):
        ex = _make_executor(hot_ratio_threshold=0.6, pressure_threshold=0.5)
        ctx = _make_context(hot_entry_ratio=0.8, eviction_pressure=0.2)
        dv = _make_dv(eviction_target="prefix-abc")
        result = ex.execute(dv, ctx)
        assert result.eviction_blocked is True
        assert result.eviction_sent is False
        assert OVERRIDE_HOT_PROTECT in result.overrides

    def test_allows_eviction_when_not_hot(self):
        coordinator = _make_coordinator(evict_returns=True)
        ex = _make_executor(
            coordinator=coordinator,
            hot_ratio_threshold=0.6,
            pressure_threshold=0.5,
        )
        ctx = _make_context(hot_entry_ratio=0.3, eviction_pressure=0.2)
        dv = _make_dv(eviction_target="prefix-abc")
        result = ex.execute(dv, ctx)
        assert result.eviction_blocked is False
        assert result.eviction_sent is True
        coordinator.evict.assert_called_once_with(
            prefix_hash="prefix-abc", tenant_id="tenant-test"
        )

    def test_allows_eviction_under_high_pressure_even_if_hot(self):
        """High eviction pressure overrides hot protection — AMF needs to free space."""
        coordinator = _make_coordinator(evict_returns=True)
        ex = _make_executor(
            coordinator=coordinator,
            hot_ratio_threshold=0.6,
            pressure_threshold=0.5,
        )
        ctx = _make_context(hot_entry_ratio=0.9, eviction_pressure=0.8)
        dv = _make_dv(eviction_target="prefix-xyz")
        result = ex.execute(dv, ctx)
        assert result.eviction_blocked is False
        assert result.eviction_sent is True

    def test_no_eviction_when_target_is_none(self):
        coordinator = _make_coordinator()
        ex = _make_executor(coordinator=coordinator)
        result = ex.execute(_make_dv(eviction_target=None), _make_context())
        assert result.eviction_blocked is False
        assert result.eviction_sent is False
        coordinator.evict.assert_not_called()

    def test_eviction_false_when_coordinator_returns_false(self):
        coordinator = _make_coordinator(evict_returns=False)
        ex = _make_executor(
            coordinator=coordinator,
            hot_ratio_threshold=0.6,
            pressure_threshold=0.5,
        )
        ctx = _make_context(hot_entry_ratio=0.1, eviction_pressure=0.1)
        result = ex.execute(_make_dv(eviction_target="prefix-abc"), ctx)
        assert result.eviction_sent is False
        assert result.eviction_blocked is False


# ── Throttle cap ──────────────────────────────────────────────────────────────

class TestThrottleCap:
    def test_caps_throttle_on_healthy_cache(self):
        ex = _make_executor(
            healthy_hit_rate=0.95,
            throttle_max_when_healthy=0.10,
        )
        ctx = _make_context(hit_rate=0.98)
        dv = _make_dv(throttle_back_pressure=0.8)
        result = ex.execute(dv, ctx)
        assert result.throttle_applied == pytest.approx(0.10)
        assert OVERRIDE_THROTTLE_CAP in result.overrides

    def test_passes_throttle_through_on_unhealthy_cache(self):
        ex = _make_executor(healthy_hit_rate=0.95)
        ctx = _make_context(hit_rate=0.70)
        dv = _make_dv(throttle_back_pressure=0.8)
        result = ex.execute(dv, ctx)
        assert result.throttle_applied == pytest.approx(0.8)
        assert OVERRIDE_THROTTLE_CAP not in result.overrides

    def test_no_cap_when_throttle_already_below_max(self):
        ex = _make_executor(
            healthy_hit_rate=0.95,
            throttle_max_when_healthy=0.10,
        )
        ctx = _make_context(hit_rate=0.98)
        dv = _make_dv(throttle_back_pressure=0.05)
        result = ex.execute(dv, ctx)
        assert result.throttle_applied == pytest.approx(0.05)
        assert OVERRIDE_THROTTLE_CAP not in result.overrides

    def test_zero_throttle_never_capped(self):
        ex = _make_executor(healthy_hit_rate=0.95)
        ctx = _make_context(hit_rate=1.0)
        result = ex.execute(_make_dv(throttle_back_pressure=0.0), ctx)
        assert result.throttle_applied == pytest.approx(0.0)
        assert OVERRIDE_THROTTLE_CAP not in result.overrides


# ── Speculative decode ────────────────────────────────────────────────────────

class TestSpecDecode:
    def _make_spec_controller(self, enabled=False, k=4):
        spec = MagicMock()
        spec.enabled = enabled
        spec.current_k = k
        spec.MIN_K = 2
        spec.MAX_K = 12
        return spec

    def test_spec_k_applied_when_changed(self):
        spec = self._make_spec_controller(enabled=False, k=4)
        ex = _make_executor(spec_controller=spec)
        dv = _make_dv(spec_decode_enable=True, spec_decode_k=6)
        result = ex.execute(dv, _make_context())
        assert result.spec_k_applied == 6
        assert spec.enabled is True
        assert spec.current_k == 6

    def test_spec_k_none_when_unchanged(self):
        spec = self._make_spec_controller(enabled=True, k=4)
        ex = _make_executor(spec_controller=spec)
        dv = _make_dv(spec_decode_enable=True, spec_decode_k=4)
        result = ex.execute(dv, _make_context())
        assert result.spec_k_applied is None

    def test_spec_k_clamped_to_max(self):
        spec = self._make_spec_controller(enabled=False, k=4)
        ex = _make_executor(spec_controller=spec)
        dv = _make_dv(spec_decode_enable=True, spec_decode_k=99)
        result = ex.execute(dv, _make_context())
        assert result.spec_k_applied == 12

    def test_spec_k_clamped_to_min(self):
        spec = self._make_spec_controller(enabled=False, k=4)
        ex = _make_executor(spec_controller=spec)
        dv = _make_dv(spec_decode_enable=True, spec_decode_k=0)
        result = ex.execute(dv, _make_context())
        assert result.spec_k_applied == 2

    def test_no_spec_controller_gives_none(self):
        ex = _make_executor(spec_controller=None)
        result = ex.execute(_make_dv(), _make_context())
        assert result.spec_k_applied is None


# ── Pre-warm lookups ──────────────────────────────────────────────────────────

class TestPreWarm:
    def test_lookups_dispatched_for_each_prediction(self):
        coordinator = _make_coordinator()
        ex = _make_executor(coordinator=coordinator)
        dv = _make_dv(pre_warm_predictions=["h1", "h2", "h3"])
        result = ex.execute(dv, _make_context())
        assert result.pre_warm_queued == 3
        coordinator.lookup.assert_any_call(prefix_hash="h1", tenant_id="tenant-test")
        coordinator.lookup.assert_any_call(prefix_hash="h2", tenant_id="tenant-test")
        coordinator.lookup.assert_any_call(prefix_hash="h3", tenant_id="tenant-test")

    def test_no_lookups_for_empty_predictions(self):
        coordinator = _make_coordinator()
        ex = _make_executor(coordinator=coordinator)
        result = ex.execute(_make_dv(pre_warm_predictions=[]), _make_context())
        assert result.pre_warm_queued == 0
        coordinator.lookup.assert_not_called()

    def test_lookup_exception_does_not_raise(self):
        coordinator = _make_coordinator()
        coordinator.lookup.side_effect = RuntimeError("network error")
        ex = _make_executor(coordinator=coordinator)
        dv = _make_dv(pre_warm_predictions=["h1"])
        result = ex.execute(dv, _make_context())
        # Exception is swallowed; pre_warm_queued stays 0
        assert result.pre_warm_queued == 0


# ── Multiple overrides in one cycle ───────────────────────────────────────────

class TestMultipleOverrides:
    def test_roi_and_throttle_both_trigger(self):
        ex = _make_executor(
            roi_threshold=0.3,
            healthy_hit_rate=0.95,
            throttle_max_when_healthy=0.10,
        )
        ctx = _make_context(roi_ema_norm=0.1, hit_rate=0.98)
        dv = _make_dv(cache_admission=True, throttle_back_pressure=0.9)
        result = ex.execute(dv, ctx)
        assert result.admitted is False
        assert result.throttle_applied == pytest.approx(0.10)
        assert OVERRIDE_ROI_GATE in result.overrides
        assert OVERRIDE_THROTTLE_CAP in result.overrides

    def test_all_three_overrides_can_trigger(self):
        coordinator = _make_coordinator()
        ex = _make_executor(
            coordinator=coordinator,
            roi_threshold=0.3,
            hot_ratio_threshold=0.6,
            pressure_threshold=0.5,
            healthy_hit_rate=0.95,
            throttle_max_when_healthy=0.10,
        )
        ctx = _make_context(
            roi_ema_norm=0.05,       # triggers ROI gate
            hot_entry_ratio=0.9,     # triggers hot protect
            eviction_pressure=0.1,   # confirms hot protect (low pressure)
            hit_rate=0.99,           # triggers throttle cap
        )
        dv = _make_dv(
            cache_admission=True,
            eviction_target="hot-prefix",
            throttle_back_pressure=0.8,
        )
        result = ex.execute(dv, ctx)
        assert result.admitted is False
        assert result.eviction_blocked is True
        assert result.throttle_applied == pytest.approx(0.10)
        assert len(result.overrides) == 3


# ── Safe fallback passes through cleanly ─────────────────────────────────────

class TestFallbackDecision:
    def test_neutral_fallback_produces_no_overrides(self):
        """
        The safe fallback from reasoning_model should produce no overrides when
        the cache is healthy (high roi, high hit_rate, low hot_ratio).
        """
        ex = _make_executor(
            roi_threshold=0.3,
            healthy_hit_rate=0.95,
        )
        ctx = _make_context(
            roi_ema_norm=0.8,    # high ROI → no gate
            hit_rate=0.99,       # healthy, but throttle=0.0 → no cap
            hot_entry_ratio=0.5, # below threshold
        )
        dv = _safe_fallback(inference_ms=0.0)
        result = ex.execute(dv, ctx)
        assert result.admitted is True
        assert result.overrides == []
        assert result.eviction_blocked is False
        assert result.eviction_sent is False


# ── Decision dict echoes original DV ─────────────────────────────────────────

class TestDecisionDict:
    def test_decision_contains_all_dv_fields(self):
        ex = _make_executor()
        dv = _make_dv(
            spec_decode_k=6,
            admission_priority=0.8,
            throttle_back_pressure=0.3,
        )
        result = ex.execute(dv, _make_context())
        d = result.decision
        assert d["spec_decode_k"] == 6
        assert d["admission_priority"] == pytest.approx(0.8)
        assert d["throttle_back_pressure"] == pytest.approx(0.3)
        assert "is_fallback" in d
        assert "inference_ms" in d


# ── decode safety rules ────────────────────────────────────────────────────────

from platform.reasoning.decision_executor import (
    OVERRIDE_DEPTH_CAP,
    OVERRIDE_ENTROPY_GUARD,
    OVERRIDE_KV_COMPRESS_GUARD,
    OVERRIDE_RUST_VETO,
)
from platform.reasoning.reasoning_model import UnifiedDecisionVector


def _make_unified_dv(**kwargs) -> UnifiedDecisionVector:
    defaults = dict(
        cache_admission=True,
        admission_priority=0.5,
        eviction_target=None,
        pre_warm_predictions=[],
        route_decision=None,
        batch_group=None,
        throttle_back_pressure=0.0,
        inference_ms=30.0,
        is_fallback=False,
        domain="unified",
        spec_decode_enable=True,
        spec_decode_depth=8,
        spec_mode="v2_clean",
        spec_v3_block_size=8,
        draft_temperature=0.8,
        kv_compression_enable=False,
        kv_compression_ratio=1.0,
        decode_batch_priority=2,
        early_stop_confidence=0.0,
        cuda_graph_hint=True,
        adaptive_depth_override=True,   # AI wants to control depth
    )
    defaults.update(kwargs)
    return UnifiedDecisionVector(**defaults)


def _make_unified_tensor(**overrides) -> np.ndarray:
    t = np.zeros(64, dtype=np.float32)
    for slot_name, value in overrides.items():
        slot_idx = getattr(Slot, slot_name.upper())
        t[slot_idx] = value
    return t


class TestDecodeDepthCap:
    def test_caps_depth_when_acceptance_low(self):
        coord = MagicMock()
        coord.evict.return_value = False
        coord.lookup.return_value = []
        executor = DecisionExecutor(coordinator_client=coord)
        dv = _make_unified_dv(spec_decode_depth=12, adaptive_depth_override=True)
        tensor = _make_unified_tensor(acceptance_ema=0.40)  # < 0.50
        ctx = ExecutionContext(tenant_id="t", node_id="n", worker_id="w", tensor=tensor)
        result = executor.execute(dv, ctx)
        assert OVERRIDE_DEPTH_CAP in result.overrides
        # spec_k_applied should reflect capped depth (≤ 3)
        if result.spec_k_applied is not None:
            assert result.spec_k_applied <= 3

    def test_no_cap_when_acceptance_above_threshold(self):
        coord = MagicMock()
        coord.evict.return_value = False
        coord.lookup.return_value = []
        executor = DecisionExecutor(coordinator_client=coord)
        dv = _make_unified_dv(spec_decode_depth=8, adaptive_depth_override=True)
        tensor = _make_unified_tensor(acceptance_ema=0.75)  # >= 0.50
        ctx = ExecutionContext(tenant_id="t", node_id="n", worker_id="w", tensor=tensor)
        result = executor.execute(dv, ctx)
        assert OVERRIDE_DEPTH_CAP not in result.overrides

    def test_no_cap_when_not_overriding_rust(self):
        """depth_cap only triggers when AI is requesting adaptive_depth_override."""
        coord = MagicMock()
        coord.evict.return_value = False
        coord.lookup.return_value = []
        executor = DecisionExecutor(coordinator_client=coord)
        dv = _make_unified_dv(spec_decode_depth=12, adaptive_depth_override=False)
        tensor = _make_unified_tensor(acceptance_ema=0.20)  # very low, but no override
        ctx = ExecutionContext(tenant_id="t", node_id="n", worker_id="w", tensor=tensor)
        result = executor.execute(dv, ctx)
        assert OVERRIDE_DEPTH_CAP not in result.overrides


class TestEntropyGuard:
    def test_disables_spec_when_entropy_high(self):
        coord = MagicMock()
        coord.evict.return_value = False
        coord.lookup.return_value = []
        executor = DecisionExecutor(coordinator_client=coord)
        dv = _make_unified_dv(spec_decode_enable=True)
        tensor = _make_unified_tensor(entropy_ema=0.80)  # > 0.70
        ctx = ExecutionContext(tenant_id="t", node_id="n", worker_id="w", tensor=tensor)
        result = executor.execute(dv, ctx)
        assert OVERRIDE_ENTROPY_GUARD in result.overrides

    def test_no_guard_when_entropy_normal(self):
        coord = MagicMock()
        coord.evict.return_value = False
        coord.lookup.return_value = []
        executor = DecisionExecutor(coordinator_client=coord)
        dv = _make_unified_dv(spec_decode_enable=True)
        tensor = _make_unified_tensor(entropy_ema=0.50)  # <= 0.70
        ctx = ExecutionContext(tenant_id="t", node_id="n", worker_id="w", tensor=tensor)
        result = executor.execute(dv, ctx)
        assert OVERRIDE_ENTROPY_GUARD not in result.overrides


class TestKvCompressionGuard:
    def test_blocks_compression_when_util_low(self):
        coord = MagicMock()
        coord.evict.return_value = False
        coord.lookup.return_value = []
        executor = DecisionExecutor(coordinator_client=coord)
        dv = _make_unified_dv(kv_compression_enable=True)
        tensor = _make_unified_tensor(kv_cache_utilization=0.60)  # < 0.80
        ctx = ExecutionContext(tenant_id="t", node_id="n", worker_id="w", tensor=tensor)
        result = executor.execute(dv, ctx)
        assert OVERRIDE_KV_COMPRESS_GUARD in result.overrides

    def test_allows_compression_when_util_high(self):
        coord = MagicMock()
        coord.evict.return_value = False
        coord.lookup.return_value = []
        executor = DecisionExecutor(coordinator_client=coord)
        dv = _make_unified_dv(kv_compression_enable=True)
        tensor = _make_unified_tensor(kv_cache_utilization=0.90)  # >= 0.80
        ctx = ExecutionContext(tenant_id="t", node_id="n", worker_id="w", tensor=tensor)
        result = executor.execute(dv, ctx)
        assert OVERRIDE_KV_COMPRESS_GUARD not in result.overrides

    def test_no_guard_when_compression_disabled(self):
        coord = MagicMock()
        coord.evict.return_value = False
        coord.lookup.return_value = []
        executor = DecisionExecutor(coordinator_client=coord)
        dv = _make_unified_dv(kv_compression_enable=False)
        tensor = _make_unified_tensor(kv_cache_utilization=0.50)
        ctx = ExecutionContext(tenant_id="t", node_id="n", worker_id="w", tensor=tensor)
        result = executor.execute(dv, ctx)
        assert OVERRIDE_KV_COMPRESS_GUARD not in result.overrides


class TestRustSchedulerVeto:
    def test_vetos_when_lane_held_long(self):
        coord = MagicMock()
        coord.evict.return_value = False
        coord.lookup.return_value = []
        executor = DecisionExecutor(coordinator_client=coord)
        dv = _make_unified_dv(adaptive_depth_override=True)
        tensor = _make_unified_tensor(lane_hold_remaining=20.0)  # > 16
        ctx = ExecutionContext(tenant_id="t", node_id="n", worker_id="w", tensor=tensor)
        result = executor.execute(dv, ctx)
        assert OVERRIDE_RUST_VETO in result.overrides

    def test_no_veto_when_lane_hold_short(self):
        coord = MagicMock()
        coord.evict.return_value = False
        coord.lookup.return_value = []
        executor = DecisionExecutor(coordinator_client=coord)
        dv = _make_unified_dv(adaptive_depth_override=True)
        tensor = _make_unified_tensor(lane_hold_remaining=8.0)  # <= 16
        ctx = ExecutionContext(tenant_id="t", node_id="n", worker_id="w", tensor=tensor)
        result = executor.execute(dv, ctx)
        assert OVERRIDE_RUST_VETO not in result.overrides

    def test_no_veto_when_not_overriding(self):
        coord = MagicMock()
        coord.evict.return_value = False
        coord.lookup.return_value = []
        executor = DecisionExecutor(coordinator_client=coord)
        dv = _make_unified_dv(adaptive_depth_override=False)
        tensor = _make_unified_tensor(lane_hold_remaining=30.0)
        ctx = ExecutionContext(tenant_id="t", node_id="n", worker_id="w", tensor=tensor)
        result = executor.execute(dv, ctx)
        assert OVERRIDE_RUST_VETO not in result.overrides

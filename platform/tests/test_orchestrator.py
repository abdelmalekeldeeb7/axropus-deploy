"""Tests for platform/reasoning/orchestrator.py (Component 6)."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

from platform.reasoning.orchestrator import (
    Orchestrator,
    OrchestratorStats,
    RequestOutcome,
)
from platform.reasoning.reward_calculator import Outcome


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_dv(
    admitted: bool = True,
    eviction_target: str | None = None,
    pre_warm_predictions: list | None = None,
    route_decision: str | None = None,
    batch_group: str | None = None,
    throttle_back_pressure: float = 0.0,
    spec_decode_k: int = 4,
    is_fallback: bool = False,
) -> MagicMock:
    dv = MagicMock()
    dv.cache_admission = admitted
    dv.eviction_target = eviction_target
    dv.pre_warm_predictions = pre_warm_predictions or []
    dv.route_decision = route_decision
    dv.batch_group = batch_group
    dv.throttle_back_pressure = throttle_back_pressure
    dv.spec_decode_k = spec_decode_k
    dv.is_fallback = is_fallback
    return dv


def _make_result(
    admitted: bool = True,
    eviction_sent: bool = False,
    throttle_applied: float = 0.0,
    spec_k_applied: int | None = None,
    overrides: list | None = None,
) -> MagicMock:
    r = MagicMock()
    r.admitted = admitted
    r.eviction_sent = eviction_sent
    r.throttle_applied = throttle_applied
    r.spec_k_applied = spec_k_applied
    r.overrides = overrides or []
    r.decision = {"inference_ms": 30.0, "is_fallback": False}
    return r


def _make_orchestrator(
    enabled: bool = True,
    ab_testing: bool = False,
    ab_ratio: float = 0.5,
    admission_window: int = 100,
    pre_warm_window: int = 100,
    eviction_window: int = 1000,
    dv: MagicMock | None = None,
    result: MagicMock | None = None,
) -> tuple[Orchestrator, dict[str, MagicMock]]:
    collector = MagicMock()
    collector.get_metrics_tensor.return_value = np.zeros(32, dtype=np.float32)

    _dv     = dv     or _make_dv()
    _result = result or _make_result()

    model    = MagicMock()
    model.decide.return_value = _dv

    executor = MagicMock()
    executor.execute.return_value = _result

    calc = MagicMock()
    calc.get_stats.return_value = {}
    calc.record_outcome.return_value = 1.0

    loop = MagicMock()
    loop._cycle_count = 3

    orch = Orchestrator(
        collector=collector,
        model=model,
        executor=executor,
        reward_calculator=calc,
        learning_loop=loop,
        enabled=enabled,
        ab_testing=ab_testing,
        ab_ratio=ab_ratio,
        admission_window=admission_window,
        pre_warm_window=pre_warm_window,
        eviction_window=eviction_window,
    )
    mocks = dict(collector=collector, model=model, executor=executor, calc=calc, loop=loop)
    return orch, mocks


def _request(orch: Orchestrator, prefix_hash: str = "px-1") -> RequestOutcome:
    return orch.on_request(
        prefix_hash=prefix_hash,
        tenant_id="t-1",
        node_id="gpu-0",
        worker_id="w-1",
    )


# ── RequestOutcome structure ──────────────────────────────────────────────────

class TestRequestOutcomeStructure:
    def test_augmented_result_has_decision_id(self):
        orch, _ = _make_orchestrator()
        outcome = _request(orch)
        assert outcome.ai_augmented is True
        assert outcome.decision_id is not None
        assert len(outcome.decision_id) == 36  # UUID

    def test_not_augmented_when_disabled(self):
        orch, _ = _make_orchestrator(enabled=False)
        outcome = _request(orch)
        assert outcome.ai_augmented is False
        assert outcome.decision_id is None

    def test_admitted_from_executor_result(self):
        result = _make_result(admitted=False)
        orch, _ = _make_orchestrator(result=result)
        outcome = _request(orch)
        assert outcome.admitted is False

    def test_throttle_from_executor_result(self):
        result = _make_result(throttle_applied=0.4)
        orch, _ = _make_orchestrator(result=result)
        outcome = _request(orch)
        assert outcome.throttle == pytest.approx(0.4)

    def test_route_decision_from_dv(self):
        dv = _make_dv(route_decision="gpu-3")
        orch, _ = _make_orchestrator(dv=dv)
        outcome = _request(orch)
        assert outcome.route_decision == "gpu-3"

    def test_batch_group_from_dv(self):
        dv = _make_dv(batch_group="bg-A")
        orch, _ = _make_orchestrator(dv=dv)
        outcome = _request(orch)
        assert outcome.batch_group == "bg-A"

    def test_spec_k_from_executor_result(self):
        result = _make_result(spec_k_applied=6)
        orch, _ = _make_orchestrator(result=result)
        outcome = _request(orch)
        assert outcome.spec_k == 6


# ── enable / disable ──────────────────────────────────────────────────────────

class TestEnableDisable:
    def test_disabled_skips_all_components(self):
        orch, mocks = _make_orchestrator(enabled=False)
        _request(orch)
        mocks["model"].decide.assert_not_called()
        mocks["executor"].execute.assert_not_called()
        mocks["calc"].record_decision.assert_not_called()

    def test_enabled_setter(self):
        orch, _ = _make_orchestrator(enabled=True)
        orch.enabled = False
        assert orch.enabled is False

    def test_pipeline_error_returns_not_augmented(self):
        orch, mocks = _make_orchestrator()
        mocks["model"].decide.side_effect = RuntimeError("GPU OOM")
        outcome = _request(orch)
        assert outcome.ai_augmented is False


# ── A/B testing ───────────────────────────────────────────────────────────────

class TestAbTesting:
    def test_disabled_ab_always_augments(self):
        orch, _ = _make_orchestrator(ab_testing=False)
        outcomes = [_request(orch) for _ in range(20)]
        assert all(o.ai_augmented for o in outcomes)

    def test_ab_ratio_zero_never_augments(self):
        orch, _ = _make_orchestrator(ab_testing=True, ab_ratio=0.0)
        outcomes = [_request(orch) for _ in range(20)]
        assert all(not o.ai_augmented for o in outcomes)

    def test_ab_ratio_one_always_augments(self):
        orch, _ = _make_orchestrator(ab_testing=True, ab_ratio=1.0)
        outcomes = [_request(orch) for _ in range(20)]
        assert all(o.ai_augmented for o in outcomes)

    def test_ab_ratio_half_approximately_respected(self):
        orch, _ = _make_orchestrator(ab_testing=True, ab_ratio=0.5)
        outcomes = [_request(orch) for _ in range(200)]
        augmented = sum(1 for o in outcomes if o.ai_augmented)
        # With 200 requests, expect ~100 ± 30 augmented
        assert 40 <= augmented <= 160

    def test_control_count_increments(self):
        orch, _ = _make_orchestrator(ab_testing=True, ab_ratio=0.0)
        for _ in range(5):
            _request(orch)
        assert orch.get_stats().control_requests == 5


# ── request counting ──────────────────────────────────────────────────────────

class TestRequestCounting:
    def test_total_requests_increments(self):
        orch, _ = _make_orchestrator()
        for _ in range(7):
            _request(orch)
        assert orch.get_stats().total_requests == 7

    def test_ai_augmented_count_increments(self):
        orch, _ = _make_orchestrator()
        for _ in range(4):
            _request(orch)
        assert orch.get_stats().ai_augmented_requests == 4

    def test_disabled_does_not_increment_ai_count(self):
        orch, _ = _make_orchestrator(enabled=False)
        for _ in range(3):
            _request(orch)
        assert orch.get_stats().ai_augmented_requests == 0


# ── decision recording ────────────────────────────────────────────────────────

class TestDecisionRecording:
    def test_record_decision_called(self):
        orch, mocks = _make_orchestrator()
        _request(orch)
        mocks["calc"].record_decision.assert_called_once()

    def test_decision_id_passed_to_record(self):
        orch, mocks = _make_orchestrator()
        outcome = _request(orch)
        call_args = mocks["calc"].record_decision.call_args
        assert call_args[0][0] == outcome.decision_id


# ── on_hit ────────────────────────────────────────────────────────────────────

class TestOnHit:
    def test_records_hit_admitted_for_admitted_prefix(self):
        orch, mocks = _make_orchestrator(result=_make_result(admitted=True))
        outcome = _request(orch, "px-hit")
        orch.on_hit("px-hit")
        mocks["calc"].record_outcome.assert_any_call(
            outcome.decision_id, Outcome.HIT_ADMITTED, "px-hit"
        )

    def test_does_not_record_hit_for_not_admitted_prefix(self):
        orch, mocks = _make_orchestrator(result=_make_result(admitted=False))
        _request(orch, "px-noadmit")
        orch.on_hit("px-noadmit")
        for c in mocks["calc"].record_outcome.call_args_list:
            assert c[0][1] != Outcome.HIT_ADMITTED

    def test_does_nothing_for_untracked_prefix(self):
        orch, mocks = _make_orchestrator()
        orch.on_hit("unknown-prefix")
        mocks["calc"].record_outcome.assert_not_called()

    def test_records_pre_warm_hit(self):
        dv = _make_dv(pre_warm_predictions=["pw-hash"])
        orch, mocks = _make_orchestrator(dv=dv)
        outcome = _request(orch)
        orch.on_hit("pw-hash")
        mocks["calc"].record_outcome.assert_any_call(
            outcome.decision_id, Outcome.PRE_WARM_HIT, "pw-hash"
        )

    def test_hit_removes_prefix_from_admitted_window(self):
        orch, mocks = _make_orchestrator(result=_make_result(admitted=True))
        _request(orch, "px-once")
        orch.on_hit("px-once")
        # Reset mock to verify second hit produces no outcome
        mocks["calc"].record_outcome.reset_mock()
        orch.on_hit("px-once")
        mocks["calc"].record_outcome.assert_not_called()


# ── on_miss ───────────────────────────────────────────────────────────────────

class TestOnMiss:
    def test_records_eviction_miss(self):
        dv     = _make_dv(eviction_target="evict-hash")
        result = _make_result(eviction_sent=True)
        orch, mocks = _make_orchestrator(dv=dv, result=result)
        outcome = _request(orch)
        orch.on_miss("evict-hash")
        mocks["calc"].record_outcome.assert_any_call(
            outcome.decision_id, Outcome.EVICTION_MISS, "evict-hash"
        )

    def test_does_nothing_for_untracked_prefix(self):
        orch, mocks = _make_orchestrator()
        orch.on_miss("nobody-evicted-this")
        # Only the record_decision call from on_request should exist, not record_outcome
        mocks["calc"].record_outcome.assert_not_called()

    def test_eviction_tracked_only_when_eviction_sent(self):
        # eviction_target set but eviction_sent=False (blocked by hot protect)
        dv     = _make_dv(eviction_target="blocked-hash")
        result = _make_result(eviction_sent=False)
        orch, mocks = _make_orchestrator(dv=dv, result=result)
        _request(orch)
        orch.on_miss("blocked-hash")
        mocks["calc"].record_outcome.assert_not_called()


# ── on_prefix_expired ─────────────────────────────────────────────────────────

class TestOnPrefixExpired:
    def test_records_miss_never_reused(self):
        orch, mocks = _make_orchestrator(result=_make_result(admitted=True))
        outcome = _request(orch, "px-exp")
        orch.on_prefix_expired("px-exp")
        mocks["calc"].record_outcome.assert_any_call(
            outcome.decision_id, Outcome.MISS_NEVER_REUSED, "px-exp"
        )

    def test_does_nothing_for_untracked_prefix(self):
        orch, mocks = _make_orchestrator()
        orch.on_prefix_expired("untracked")
        mocks["calc"].record_outcome.assert_not_called()

    def test_not_recorded_if_already_hit(self):
        """If prefix was hit, expiry should not double-record MISS_NEVER_REUSED."""
        orch, mocks = _make_orchestrator(result=_make_result(admitted=True))
        _request(orch, "px-hit-then-exp")
        orch.on_hit("px-hit-then-exp")
        mocks["calc"].record_outcome.reset_mock()
        orch.on_prefix_expired("px-hit-then-exp")
        for c in mocks["calc"].record_outcome.call_args_list:
            assert c[0][1] != Outcome.MISS_NEVER_REUSED


# ── attribution window expiry ─────────────────────────────────────────────────

class TestAttributionWindowExpiry:
    def test_admitted_prefix_scores_miss_never_reused_after_window(self):
        orch, mocks = _make_orchestrator(
            result=_make_result(admitted=True),
            admission_window=3,
        )
        outcome = _request(orch, "px-expire")
        # Drive 4 more requests to push past the window of 3
        for i in range(4):
            _request(orch, f"other-{i}")
        # The expiry is triggered inside _expire_windows on the next on_request
        _request(orch, "trigger")
        outcome_types = [c[0][1] for c in mocks["calc"].record_outcome.call_args_list]
        assert Outcome.MISS_NEVER_REUSED in outcome_types

    def test_pre_warm_scores_miss_after_window(self):
        # First request predicts "pw-expire"; all subsequent ones predict nothing
        dv_first  = _make_dv(pre_warm_predictions=["pw-expire"])
        dv_filler = _make_dv(pre_warm_predictions=[])
        orch, mocks = _make_orchestrator(pre_warm_window=2)
        mocks["model"].decide.side_effect = [dv_first] + [dv_filler] * 10
        _request(orch)
        for i in range(4):
            _request(orch, f"other-{i}")
        outcome_types = [c[0][1] for c in mocks["calc"].record_outcome.call_args_list]
        assert Outcome.PRE_WARM_MISS in outcome_types

    def test_hit_before_window_prevents_miss_never_reused(self):
        # Only the first request admits; fillers don't admit anything
        result_admit    = _make_result(admitted=True)
        result_no_admit = _make_result(admitted=False)
        orch, mocks = _make_orchestrator(admission_window=10)
        mocks["executor"].execute.side_effect = [result_admit] + [result_no_admit] * 20
        _request(orch, "px-save")
        orch.on_hit("px-save")  # resolved before window expires
        for i in range(15):
            _request(orch, f"filler-{i}")
        outcome_types = [c[0][1] for c in mocks["calc"].record_outcome.call_args_list]
        assert Outcome.MISS_NEVER_REUSED not in outcome_types


# ── notify_route_latency ──────────────────────────────────────────────────────

class TestNotifyRouteLatency:
    def test_positive_delta_records_improvement(self):
        orch, mocks = _make_orchestrator()
        orch.notify_route_latency("d-1", delta_ms=50.0)
        mocks["calc"].record_outcome.assert_called_once_with(
            "d-1", Outcome.ROUTE_IMPROVEMENT, value=50.0
        )

    def test_negative_delta_records_regression(self):
        orch, mocks = _make_orchestrator()
        orch.notify_route_latency("d-1", delta_ms=-30.0)
        mocks["calc"].record_outcome.assert_called_once_with(
            "d-1", Outcome.ROUTE_REGRESSION, value=30.0
        )


# ── lifecycle ─────────────────────────────────────────────────────────────────

class TestLifecycle:
    def test_start_calls_collector_and_loop(self):
        orch, mocks = _make_orchestrator()
        orch.start()
        mocks["collector"].start.assert_called_once()
        mocks["loop"].start.assert_called_once()

    def test_stop_calls_loop_and_collector(self):
        orch, mocks = _make_orchestrator()
        orch.stop()
        mocks["loop"].stop.assert_called_once()
        mocks["collector"].stop.assert_called_once()


# ── get_stats ─────────────────────────────────────────────────────────────────

class TestGetStats:
    def test_returns_orchestrator_stats(self):
        orch, _ = _make_orchestrator()
        stats = orch.get_stats()
        assert isinstance(stats, OrchestratorStats)

    def test_stats_reflect_config(self):
        orch, _ = _make_orchestrator(enabled=True, ab_testing=True, ab_ratio=0.3)
        stats = orch.get_stats()
        assert stats.enabled is True
        assert stats.ab_testing is True
        assert stats.ab_ratio == pytest.approx(0.3)

    def test_loop_cycles_from_loop(self):
        orch, mocks = _make_orchestrator()
        mocks["loop"]._cycle_count = 7
        assert orch.get_stats().loop_cycles == 7

    def test_admitted_tracked_count(self):
        orch, _ = _make_orchestrator(result=_make_result(admitted=True))
        _request(orch, "px-1")
        _request(orch, "px-2")
        stats = orch.get_stats()
        assert stats.admitted_tracked == 2

    def test_pre_warm_tracked_count(self):
        dv = _make_dv(pre_warm_predictions=["h1", "h2", "h3"])
        orch, _ = _make_orchestrator(dv=dv)
        _request(orch)
        assert orch.get_stats().pre_warm_tracked == 3


# ── on_decode_step_complete ───────────────────────────────────────────────────

class TestOnDecodeStepComplete:
    def test_records_high_acceptance(self):
        orch, _ = _make_orchestrator()
        outcome = _request(orch)
        calc_mock = MagicMock()
        orch._calc = calc_mock
        orch.on_decode_step_complete(
            decision_id=outcome.decision_id or "d-1",
            acceptance_ema=0.85,
            spec_depth_used=4,
            lane_held_steps=5,
        )
        calls = [str(c) for c in calc_mock.record_outcome.call_args_list]
        assert any("spec_high_acceptance" in s for s in calls)

    def test_records_low_acceptance(self):
        orch, _ = _make_orchestrator()
        outcome = _request(orch)
        calc_mock = MagicMock()
        orch._calc = calc_mock
        orch.on_decode_step_complete(
            decision_id=outcome.decision_id or "d-1",
            acceptance_ema=0.30,
            spec_depth_used=4,
            lane_held_steps=5,
        )
        calls = [str(c) for c in calc_mock.record_outcome.call_args_list]
        assert any("spec_low_acceptance" in s for s in calls)

    def test_records_depth_wasteful(self):
        orch, _ = _make_orchestrator()
        outcome = _request(orch)
        calc_mock = MagicMock()
        orch._calc = calc_mock
        orch.on_decode_step_complete(
            decision_id=outcome.decision_id or "d-1",
            acceptance_ema=0.50,
            spec_depth_used=10,  # > 8 with low acceptance
            lane_held_steps=5,
        )
        calls = [str(c) for c in calc_mock.record_outcome.call_args_list]
        assert any("spec_depth_wasteful" in s for s in calls)

    def test_no_crash_on_calc_error(self):
        orch, _ = _make_orchestrator()
        orch._calc = MagicMock(record_outcome=MagicMock(side_effect=RuntimeError("db fail")))
        # Should not raise
        orch.on_decode_step_complete("d-1", 0.85, 4, 5)


# ── on_decode_complete ────────────────────────────────────────────────────────

class TestOnDecodeComplete:
    def test_records_latency_improved(self):
        orch, _ = _make_orchestrator()
        calc_mock = MagicMock()
        orch._calc = calc_mock
        orch.on_decode_complete(
            decision_id="d-1",
            mode_used="v2_clean",
            baseline_ms_per_token=10.0,
            actual_ms_per_token=8.0,   # improved
            cuda_graph_active=False,
            early_stop_fired=False,
            output_complete=True,
        )
        calls = [str(c) for c in calc_mock.record_outcome.call_args_list]
        assert any("decode_latency_improved" in s for s in calls)

    def test_records_latency_worsened(self):
        orch, _ = _make_orchestrator()
        calc_mock = MagicMock()
        orch._calc = calc_mock
        orch.on_decode_complete(
            decision_id="d-1",
            mode_used="v2_clean",
            baseline_ms_per_token=10.0,
            actual_ms_per_token=12.0,  # worsened
            cuda_graph_active=False,
            early_stop_fired=False,
            output_complete=True,
        )
        calls = [str(c) for c in calc_mock.record_outcome.call_args_list]
        assert any("decode_latency_worsened" in s for s in calls)

    def test_records_early_stop_correct(self):
        orch, _ = _make_orchestrator()
        calc_mock = MagicMock()
        orch._calc = calc_mock
        orch.on_decode_complete(
            decision_id="d-1",
            mode_used="v2_clean",
            baseline_ms_per_token=10.0,
            actual_ms_per_token=10.0,
            cuda_graph_active=False,
            early_stop_fired=True,
            output_complete=True,   # correct stop
        )
        calls = [str(c) for c in calc_mock.record_outcome.call_args_list]
        assert any("early_stop_correct" in s for s in calls)

    def test_records_early_stop_premature(self):
        orch, _ = _make_orchestrator()
        calc_mock = MagicMock()
        orch._calc = calc_mock
        orch.on_decode_complete(
            decision_id="d-1",
            mode_used="v2_clean",
            baseline_ms_per_token=10.0,
            actual_ms_per_token=10.0,
            cuda_graph_active=False,
            early_stop_fired=True,
            output_complete=False,  # premature
        )
        calls = [str(c) for c in calc_mock.record_outcome.call_args_list]
        assert any("early_stop_premature" in s for s in calls)

    def test_no_crash_on_calc_error(self):
        orch, _ = _make_orchestrator()
        orch._calc = MagicMock(record_outcome=MagicMock(side_effect=RuntimeError("db fail")))
        orch.on_decode_complete("d-1", "v2_clean", 10.0, 9.0, False, False, True)

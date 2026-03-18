"""Tests for platform/reasoning/reward_calculator.py (Component 4)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import numpy as np
import pytest

from platform.reasoning.reward_calculator import (
    REWARD_WEIGHTS,
    Outcome,
    RewardCalculator,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _calc() -> RewardCalculator:
    """In-memory calculator for test isolation."""
    return RewardCalculator(db_path=":memory:")


def _make_context(
    tenant_id: str = "t-1",
    node_id: str = "gpu-0",
    tensor: np.ndarray | None = None,
) -> MagicMock:
    ctx = MagicMock()
    ctx.tenant_id = tenant_id
    ctx.node_id = node_id
    ctx.tensor = tensor if tensor is not None else np.zeros(32, dtype=np.float32)
    return ctx


def _make_result(
    admitted: bool = True,
    eviction_target: str | None = None,
    pre_warm_predictions: list | None = None,
    overrides: list | None = None,
    throttle_applied: float = 0.0,
    spec_k_applied: int | None = None,
    inference_ms: float = 30.0,
    is_fallback: bool = False,
) -> MagicMock:
    r = MagicMock()
    r.admitted = admitted
    r.throttle_applied = throttle_applied
    r.spec_k_applied = spec_k_applied
    r.overrides = overrides or []
    r.decision = {
        "cache_admission": admitted,
        "admission_priority": 0.5,
        "eviction_target": eviction_target,
        "pre_warm_predictions": pre_warm_predictions or [],
        "route_decision": None,
        "batch_group": None,
        "spec_decode_enable": True,
        "spec_decode_k": spec_k_applied or 4,
        "throttle_back_pressure": throttle_applied,
        "inference_ms": inference_ms,
        "is_fallback": is_fallback,
    }
    return r


def _record(calc: RewardCalculator, decision_id: str = "d-1", **kwargs) -> None:
    calc.record_decision(decision_id, _make_result(**kwargs), _make_context())


# ── Outcome constants ─────────────────────────────────────────────────────────

class TestOutcomeConstants:
    def test_all_outcomes_in_reward_weights(self):
        all_outcomes = [
            Outcome.HIT_ADMITTED,
            Outcome.MISS_NEVER_REUSED,
            Outcome.EVICTION_MISS,
            Outcome.PRE_WARM_HIT,
            Outcome.PRE_WARM_MISS,
            Outcome.ROUTE_IMPROVEMENT,
            Outcome.ROUTE_REGRESSION,
        ]
        for o in all_outcomes:
            assert o in REWARD_WEIGHTS, f"{o} missing from REWARD_WEIGHTS"

    def test_positive_rewards(self):
        assert REWARD_WEIGHTS[Outcome.HIT_ADMITTED] > 0
        assert REWARD_WEIGHTS[Outcome.PRE_WARM_HIT] > 0
        assert REWARD_WEIGHTS[Outcome.ROUTE_IMPROVEMENT] > 0

    def test_negative_rewards(self):
        assert REWARD_WEIGHTS[Outcome.MISS_NEVER_REUSED] < 0
        assert REWARD_WEIGHTS[Outcome.EVICTION_MISS] < 0
        assert REWARD_WEIGHTS[Outcome.PRE_WARM_MISS] < 0
        assert REWARD_WEIGHTS[Outcome.ROUTE_REGRESSION] < 0

    def test_eviction_miss_worst_penalty(self):
        """Evicting a hot prefix that then misses is the worst outcome."""
        assert REWARD_WEIGHTS[Outcome.EVICTION_MISS] == min(REWARD_WEIGHTS.values())


# ── record_decision ───────────────────────────────────────────────────────────

class TestRecordDecision:
    def test_decision_stored(self):
        c = _calc()
        _record(c, "d-1")
        stats = c.get_stats()
        assert stats["total_decisions"] == 1

    def test_multiple_decisions_stored(self):
        c = _calc()
        for i in range(5):
            _record(c, f"d-{i}")
        assert c.get_stats()["total_decisions"] == 5

    def test_tensor_stored_and_recovered(self):
        c = _calc()
        tensor = np.arange(32, dtype=np.float32)
        ctx = _make_context(tensor=tensor)
        c.record_decision("d-tensor", _make_result(), ctx)

        # recover via training batch (need an outcome first)
        c.record_outcome("d-tensor", Outcome.HIT_ADMITTED)
        batch = c.get_training_batch()
        assert len(batch) == 1
        np.testing.assert_array_almost_equal(batch[0]["tensor"], tensor)

    def test_none_tensor_handled(self):
        c = _calc()
        ctx = _make_context(tensor=None)
        ctx.tensor = None
        c.record_decision("d-none", _make_result(), ctx)
        c.record_outcome("d-none", Outcome.HIT_ADMITTED)
        batch = c.get_training_batch()
        assert batch[0]["tensor"] is None

    def test_decision_json_roundtrip(self):
        c = _calc()
        result = _make_result(admitted=True, spec_k_applied=6)
        c.record_decision("d-json", result, _make_context())
        c.record_outcome("d-json", Outcome.HIT_ADMITTED)
        batch = c.get_training_batch()
        assert batch[0]["decision"]["spec_decode_k"] == 6

    def test_overrides_stored(self):
        c = _calc()
        result = _make_result(overrides=["roi_gate:admission_blocked"])
        c.record_decision("d-ov", result, _make_context())
        c.record_outcome("d-ov", Outcome.HIT_ADMITTED)
        batch = c.get_training_batch()
        assert "roi_gate:admission_blocked" in batch[0]["overrides"]

    def test_idempotent_upsert(self):
        """Inserting the same decision_id twice does not duplicate the row."""
        c = _calc()
        _record(c, "d-dup")
        _record(c, "d-dup")
        assert c.get_stats()["total_decisions"] == 1


# ── record_outcome ────────────────────────────────────────────────────────────

class TestRecordOutcome:
    def test_returns_correct_reward(self):
        c = _calc()
        _record(c)
        reward = c.record_outcome("d-1", Outcome.HIT_ADMITTED)
        assert reward == pytest.approx(REWARD_WEIGHTS[Outcome.HIT_ADMITTED])

    def test_unknown_outcome_type_returns_zero(self):
        c = _calc()
        _record(c)
        reward = c.record_outcome("d-1", "unknown_type")
        assert reward == pytest.approx(0.0)

    def test_outcome_stored(self):
        c = _calc()
        _record(c)
        c.record_outcome("d-1", Outcome.EVICTION_MISS, prefix_hash="abc")
        stats = c.get_stats()
        assert stats["total_outcomes"] == 1

    def test_multiple_outcomes_for_same_decision(self):
        c = _calc()
        _record(c)
        c.record_outcome("d-1", Outcome.PRE_WARM_HIT, "h1")
        c.record_outcome("d-1", Outcome.PRE_WARM_HIT, "h2")
        c.record_outcome("d-1", Outcome.PRE_WARM_MISS, "h3")
        assert c.get_stats()["total_outcomes"] == 3

    def test_prefix_hash_stored(self):
        c = _calc()
        _record(c)
        c.record_outcome("d-1", Outcome.PRE_WARM_HIT, prefix_hash="deadbeef")
        # Verify via training batch (prefix_hash not directly exposed, but reward is)
        assert c.score_decision("d-1") == pytest.approx(REWARD_WEIGHTS[Outcome.PRE_WARM_HIT])


# ── score_decision ────────────────────────────────────────────────────────────

class TestScoreDecision:
    def test_zero_with_no_outcomes(self):
        c = _calc()
        _record(c)
        assert c.score_decision("d-1") == pytest.approx(0.0)

    def test_single_positive_outcome(self):
        c = _calc()
        _record(c)
        c.record_outcome("d-1", Outcome.HIT_ADMITTED)
        assert c.score_decision("d-1") == pytest.approx(1.0)

    def test_single_negative_outcome(self):
        c = _calc()
        _record(c)
        c.record_outcome("d-1", Outcome.EVICTION_MISS)
        assert c.score_decision("d-1") == pytest.approx(-1.0)

    def test_mixed_outcomes_sum_correctly(self):
        c = _calc()
        _record(c)
        c.record_outcome("d-1", Outcome.HIT_ADMITTED)          # +1.0
        c.record_outcome("d-1", Outcome.PRE_WARM_HIT)          # +1.0
        c.record_outcome("d-1", Outcome.PRE_WARM_MISS)         # -0.3
        c.record_outcome("d-1", Outcome.ROUTE_IMPROVEMENT)     # +0.5
        expected = 1.0 + 1.0 - 0.3 + 0.5
        assert c.score_decision("d-1") == pytest.approx(expected)

    def test_worst_case_score(self):
        c = _calc()
        _record(c)
        c.record_outcome("d-1", Outcome.EVICTION_MISS)         # -1.0
        c.record_outcome("d-1", Outcome.MISS_NEVER_REUSED)     # -0.5
        c.record_outcome("d-1", Outcome.ROUTE_REGRESSION)      # -0.5
        assert c.score_decision("d-1") == pytest.approx(-2.0)

    def test_unknown_decision_returns_zero(self):
        c = _calc()
        assert c.score_decision("nonexistent") == pytest.approx(0.0)

    def test_decisions_isolated(self):
        """Outcomes for d-2 don't affect score of d-1."""
        c = _calc()
        _record(c, "d-1")
        _record(c, "d-2")
        c.record_outcome("d-1", Outcome.HIT_ADMITTED)
        c.record_outcome("d-2", Outcome.EVICTION_MISS)
        assert c.score_decision("d-1") == pytest.approx(1.0)
        assert c.score_decision("d-2") == pytest.approx(-1.0)


# ── get_training_batch ────────────────────────────────────────────────────────

class TestGetTrainingBatch:
    def test_empty_when_no_decisions(self):
        assert _calc().get_training_batch() == []

    def test_empty_when_no_outcomes(self):
        c = _calc()
        _record(c)
        assert c.get_training_batch() == []

    def test_returns_decision_with_outcome(self):
        c = _calc()
        _record(c, "d-1")
        c.record_outcome("d-1", Outcome.HIT_ADMITTED)
        batch = c.get_training_batch()
        assert len(batch) == 1
        assert batch[0]["decision_id"] == "d-1"

    def test_total_reward_correct(self):
        c = _calc()
        _record(c, "d-1")
        c.record_outcome("d-1", Outcome.HIT_ADMITTED)       # +1.0
        c.record_outcome("d-1", Outcome.PRE_WARM_MISS)      # -0.3
        batch = c.get_training_batch()
        assert batch[0]["total_reward"] == pytest.approx(0.7)

    def test_outcome_count_correct(self):
        c = _calc()
        _record(c, "d-1")
        c.record_outcome("d-1", Outcome.HIT_ADMITTED)
        c.record_outcome("d-1", Outcome.PRE_WARM_HIT)
        batch = c.get_training_batch()
        assert batch[0]["outcome_count"] == 2

    def test_limit_respected(self):
        c = _calc()
        for i in range(10):
            _record(c, f"d-{i}")
            c.record_outcome(f"d-{i}", Outcome.HIT_ADMITTED)
        batch = c.get_training_batch(limit=5)
        assert len(batch) == 5

    def test_consumed_excluded(self):
        c = _calc()
        _record(c, "d-1")
        c.record_outcome("d-1", Outcome.HIT_ADMITTED)
        c.mark_batch_consumed(["d-1"])
        assert c.get_training_batch() == []

    def test_min_outcomes_filter(self):
        c = _calc()
        _record(c, "d-1")
        c.record_outcome("d-1", Outcome.HIT_ADMITTED)
        # d-1 has 1 outcome; min_outcomes=2 should exclude it
        assert c.get_training_batch(min_outcomes=2) == []
        c.record_outcome("d-1", Outcome.PRE_WARM_HIT)
        assert len(c.get_training_batch(min_outcomes=2)) == 1

    def test_ordered_by_timestamp_ascending(self):
        c = _calc()
        for i in range(3):
            _record(c, f"d-{i}")
            c.record_outcome(f"d-{i}", Outcome.HIT_ADMITTED)
        batch = c.get_training_batch()
        ids = [b["decision_id"] for b in batch]
        assert ids == sorted(ids)  # ascending by insertion order


# ── mark_batch_consumed ───────────────────────────────────────────────────────

class TestMarkBatchConsumed:
    def test_marks_decisions_consumed(self):
        c = _calc()
        for i in range(3):
            _record(c, f"d-{i}")
        count = c.mark_batch_consumed(["d-0", "d-1"])
        assert count == 2
        assert c.get_stats()["consumed_decisions"] == 2

    def test_unconsumed_still_available(self):
        c = _calc()
        _record(c, "d-0")
        _record(c, "d-1")
        c.record_outcome("d-0", Outcome.HIT_ADMITTED)
        c.record_outcome("d-1", Outcome.HIT_ADMITTED)
        c.mark_batch_consumed(["d-0"])
        batch = c.get_training_batch()
        assert len(batch) == 1
        assert batch[0]["decision_id"] == "d-1"

    def test_empty_list_returns_zero(self):
        c = _calc()
        assert c.mark_batch_consumed([]) == 0


# ── get_stats ─────────────────────────────────────────────────────────────────

class TestGetStats:
    def test_initial_stats_all_zero(self):
        stats = _calc().get_stats()
        assert stats["total_decisions"] == 0
        assert stats["total_outcomes"] == 0
        assert stats["mean_reward"] == pytest.approx(0.0)
        assert stats["outcome_breakdown"] == {}

    def test_outcome_breakdown(self):
        c = _calc()
        _record(c)
        c.record_outcome("d-1", Outcome.HIT_ADMITTED)
        c.record_outcome("d-1", Outcome.HIT_ADMITTED)
        c.record_outcome("d-1", Outcome.PRE_WARM_MISS)
        breakdown = c.get_stats()["outcome_breakdown"]
        assert breakdown[Outcome.HIT_ADMITTED] == 2
        assert breakdown[Outcome.PRE_WARM_MISS] == 1

    def test_mean_reward(self):
        c = _calc()
        _record(c)
        c.record_outcome("d-1", Outcome.HIT_ADMITTED)    # +1.0
        c.record_outcome("d-1", Outcome.EVICTION_MISS)   # -1.0
        stats = c.get_stats()
        assert stats["mean_reward"] == pytest.approx(0.0)

    def test_consumed_count(self):
        c = _calc()
        _record(c, "d-1")
        _record(c, "d-2")
        c.mark_batch_consumed(["d-1"])
        stats = c.get_stats()
        assert stats["consumed_decisions"] == 1
        assert stats["total_decisions"] == 2

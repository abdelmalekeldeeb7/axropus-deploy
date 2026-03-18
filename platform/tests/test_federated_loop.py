"""
Tests for Level 4: FederatedLearningLoop, CoopOutcome, _dict_l2_norm.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from platform.reasoning.federated_loop import (
    COOP_REWARD_WEIGHTS,
    CoopOutcome,
    FederatedLearningLoop,
    FederationResult,
    _NoOpCoordinator,
    _dict_l2_norm,
)


# ── CoopOutcome / reward weight tests ─────────────────────────────────────────

class TestCoopOutcome:

    def test_all_constants_exist(self):
        assert hasattr(CoopOutcome, "COOPERATIVE_TRANSFER_HIT")
        assert hasattr(CoopOutcome, "COOPERATIVE_TRANSFER_MISS")
        assert hasattr(CoopOutcome, "LOAD_SHED_SUCCESS")
        assert hasattr(CoopOutcome, "LOAD_SHED_FAILURE")
        assert hasattr(CoopOutcome, "SPECIALISATION_BONUS")

    def test_transfer_hit_positive(self):
        assert COOP_REWARD_WEIGHTS[CoopOutcome.COOPERATIVE_TRANSFER_HIT] > 0

    def test_transfer_miss_negative(self):
        assert COOP_REWARD_WEIGHTS[CoopOutcome.COOPERATIVE_TRANSFER_MISS] < 0

    def test_load_shed_success_positive(self):
        assert COOP_REWARD_WEIGHTS[CoopOutcome.LOAD_SHED_SUCCESS] > 0

    def test_load_shed_failure_negative(self):
        assert COOP_REWARD_WEIGHTS[CoopOutcome.LOAD_SHED_FAILURE] < 0

    def test_specialisation_bonus_positive(self):
        assert COOP_REWARD_WEIGHTS[CoopOutcome.SPECIALISATION_BONUS] > 0


# ── _dict_l2_norm tests ───────────────────────────────────────────────────────

class TestDictL2Norm:

    def test_empty_dict(self):
        assert _dict_l2_norm({}) == 0.0

    def test_single_value(self):
        norm = _dict_l2_norm({"a": [3.0, 4.0]})
        assert abs(norm - 5.0) < 1e-5

    def test_ignores_non_list(self):
        norm = _dict_l2_norm({"_n_nodes": 3, "w": [1.0]})
        assert abs(norm - 1.0) < 1e-5


# ── _NoOpCoordinator tests ────────────────────────────────────────────────────

class TestNoOpCoordinator:

    def test_submit_does_not_raise(self):
        c = _NoOpCoordinator()
        c.submit_adapter_delta("node-0", {"w": [1.0, 2.0]})

    def test_get_returns_none(self):
        c = _NoOpCoordinator()
        assert c.get_averaged_delta("node-0") is None


# ── FederatedLearningLoop tests ───────────────────────────────────────────────

def _make_loop_and_fed(sync_every=2):
    """Build a FederatedLearningLoop with a mocked local LearningLoop."""
    from platform.reasoning.learning_loop import TrainingResult

    mock_model = MagicMock()
    mock_model.loaded = False
    mock_model._model = MagicMock()
    mock_model._model.named_parameters.return_value = []

    mock_calc = MagicMock()

    local_result = TrainingResult(
        examples_used=4,
        mean_reward=1.0,
        loss=0.5,
        adapter_path=None,
        elapsed_s=0.1,
        skipped=False,
        skip_reason="",
    )

    local_loop = MagicMock()
    local_loop.run_once.return_value = local_result
    local_loop._interval_s = 0.01
    local_loop._model = mock_model
    local_loop._swap_lock = __import__("threading").Lock()

    fed = FederatedLearningLoop(
        local_loop=local_loop,
        coordinator=_NoOpCoordinator(),
        node_id="test-node",
        sync_every=sync_every,
    )
    return local_loop, fed


class TestFederatedLearningLoop:

    def test_node_id_stored(self):
        _, fed = _make_loop_and_fed()
        assert fed._node_id == "test-node"

    def test_sync_every_stored(self):
        _, fed = _make_loop_and_fed(sync_every=3)
        assert fed._sync_every == 3

    def test_run_once_returns_result(self):
        _, fed = _make_loop_and_fed()
        result = fed.run_once()
        assert isinstance(result, FederationResult)
        assert result.node_id == "test-node"

    def test_local_cycles_increment(self):
        _, fed = _make_loop_and_fed(sync_every=10)
        fed.run_once()
        assert fed._local_cycles == 1
        fed.run_once()
        assert fed._local_cycles == 2

    def test_no_sync_before_sync_every(self):
        _, fed = _make_loop_and_fed(sync_every=5)
        result = fed.run_once()
        assert result.synced is False

    def test_sync_fires_at_sync_every(self):
        _, fed = _make_loop_and_fed(sync_every=2)
        fed.run_once()          # cycle 1 — no sync
        result = fed.run_once() # cycle 2 — sync fires
        assert result.synced is True

    def test_sync_count_increments_on_sync(self):
        _, fed = _make_loop_and_fed(sync_every=2)
        fed.run_once()
        fed.run_once()
        assert fed._sync_count == 1

    def test_running_false_before_start(self):
        _, fed = _make_loop_and_fed()
        assert fed.running is False

    def test_start_stop(self):
        _, fed = _make_loop_and_fed()
        fed.start()
        assert fed.running is True
        fed.stop(timeout=1.0)
        # After stop, running may be False
        time.sleep(0.05)

    def test_extract_adapter_delta_empty_when_not_loaded(self):
        _, fed = _make_loop_and_fed()
        delta = fed._extract_adapter_delta()
        assert delta == {}

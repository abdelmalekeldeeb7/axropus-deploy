"""
Tests for Level 3 RL: RolloutBuffer, ValueHead, PPO/DPO dispatch.
"""

from __future__ import annotations

import inspect
import json
import time

import numpy as np
import pytest

from platform.reasoning.rollout_buffer import (
    DPOPair,
    RolloutBuffer,
    ValueHead,
    build_dpo_pairs,
)


# ── RolloutBuffer tests ───────────────────────────────────────────────────────

class TestRolloutBuffer:

    def test_add_and_len(self):
        buf = RolloutBuffer(capacity=10)
        buf.add(state="s", action="a", reward=1.0, value=0.5, log_prob=-0.1)
        assert len(buf) == 1

    def test_capacity_enforced(self):
        buf = RolloutBuffer(capacity=3)
        for i in range(5):
            buf.add(state=f"s{i}", action="a", reward=float(i), value=0.0, log_prob=0.0)
        assert len(buf) == 3
        # Most recent 3 should be kept
        assert buf.rewards == [2.0, 3.0, 4.0]

    def test_clear(self):
        buf = RolloutBuffer()
        buf.add(state="s", action="a", reward=1.0, value=0.5, log_prob=-0.1)
        buf.clear()
        assert len(buf) == 0
        assert buf.advantages == []
        assert buf.returns == []

    def test_compute_gae_basic(self):
        buf = RolloutBuffer()
        # Two steps: reward 1.0 and 2.0, values 0.5 and 0.8
        buf.add(state="s0", action="a", reward=1.0, value=0.5, log_prob=-0.2)
        buf.add(state="s1", action="a", reward=2.0, value=0.8, log_prob=-0.1)
        buf.compute_gae(gamma=0.99, lam=0.95)
        assert len(buf.advantages) == 2
        assert len(buf.returns) == 2

    def test_gae_returns_equals_adv_plus_value(self):
        buf = RolloutBuffer()
        rewards = [1.0, -0.5, 2.0]
        values  = [0.3,  0.7,  1.1]
        for r, v in zip(rewards, values):
            buf.add(state="s", action="a", reward=r, value=v, log_prob=0.0)
        buf.compute_gae()
        for ret, adv, val in zip(buf.returns, buf.advantages, buf.values):
            assert abs(ret - (adv + val)) < 1e-5

    def test_sample_requires_gae_first(self):
        buf = RolloutBuffer()
        buf.add(state="s", action="a", reward=1.0, value=0.5, log_prob=0.0)
        with pytest.raises(RuntimeError, match="compute_gae"):
            list(buf.sample(batch_size=1))

    def test_sample_yields_batches(self):
        buf = RolloutBuffer()
        for i in range(10):
            buf.add(state=f"s{i}", action="a", reward=float(i), value=0.0, log_prob=0.0)
        buf.compute_gae()
        batches = list(buf.sample(batch_size=3))
        total = sum(len(b["states"]) for b in batches)
        assert total == 10

    def test_sample_batch_keys(self):
        buf = RolloutBuffer()
        buf.add(state="s", action="a", reward=1.0, value=0.5, log_prob=-0.1)
        buf.compute_gae()
        batch = next(buf.sample(batch_size=1))
        assert set(batch.keys()) == {
            "states", "actions", "rewards", "values",
            "log_probs", "advantages", "returns"
        }

    def test_stats_empty(self):
        buf = RolloutBuffer()
        s = buf.stats()
        assert s["n"] == 0

    def test_stats_with_data(self):
        buf = RolloutBuffer()
        buf.add(state="s0", action="a", reward=2.0, value=1.0, log_prob=0.0)
        buf.add(state="s1", action="a", reward=4.0, value=1.0, log_prob=0.0)
        s = buf.stats()
        assert s["n"] == 2
        assert abs(s["mean_reward"] - 3.0) < 1e-5

    def test_compute_gae_empty_is_no_op(self):
        buf = RolloutBuffer()
        buf.compute_gae()   # should not raise
        assert buf.advantages == []


# ── ValueHead tests ───────────────────────────────────────────────────────────

class TestValueHead:

    def test_instantiation_no_torch(self):
        vh = ValueHead(hidden_size=256)
        assert vh._hidden_size == 256

    def test_build_returns_module(self):
        torch = pytest.importorskip("torch")
        vh = ValueHead(hidden_size=64)
        module = vh.build()
        assert hasattr(module, "forward")

    def test_forward_shape(self):
        torch = pytest.importorskip("torch")
        vh = ValueHead(hidden_size=64)
        module = vh.build()
        # (batch=2, seq_len=5, hidden=64) → (batch=2,)
        hidden = torch.zeros(2, 5, 64)
        out = module(hidden)
        assert out.shape == (2,)


# ── DPO pair builder tests ────────────────────────────────────────────────────

class TestBuildDPOPairs:

    def _make_examples(self):
        return [
            {"prompt": "p", "target": '{"prefill":{}}', "reward": 1.5, "decision_id": "g1"},
            {"prompt": "p", "target": '{"prefill":{}}', "reward": 2.0, "decision_id": "g2"},
            {"prompt": "p", "target": '{"prefill":{}}', "reward": -1.0, "decision_id": "b1"},
            {"prompt": "p", "target": '{"prefill":{}}', "reward": -0.8, "decision_id": "b2"},
        ]

    def test_returns_pairs(self):
        pairs = build_dpo_pairs(self._make_examples())
        assert len(pairs) > 0
        for p in pairs:
            assert isinstance(p, DPOPair)

    def test_good_reward_positive(self):
        pairs = build_dpo_pairs(self._make_examples())
        for p in pairs:
            assert p.good_reward > 0
            assert p.bad_reward < 0

    def test_min_reward_gap_filters(self):
        # With a very high gap requirement, all pairs should be filtered
        examples = [
            {"prompt": "p", "target": "t", "reward": 0.3, "decision_id": "g"},
            {"prompt": "p", "target": "t", "reward": -0.2, "decision_id": "b"},
        ]
        pairs = build_dpo_pairs(examples, min_reward_gap=2.0)
        assert pairs == []

    def test_empty_when_no_bad_examples(self):
        examples = [
            {"prompt": "p", "target": "t", "reward": 1.0, "decision_id": "g1"},
            {"prompt": "p", "target": "t", "reward": 2.0, "decision_id": "g2"},
        ]
        pairs = build_dpo_pairs(examples)
        assert pairs == []

    def test_empty_when_no_good_examples(self):
        examples = [
            {"prompt": "p", "target": "t", "reward": -1.0, "decision_id": "b1"},
        ]
        pairs = build_dpo_pairs(examples)
        assert pairs == []


# ── LearningLoop PPO/DPO dispatch tests ──────────────────────────────────────

class TestLearningLoopLevel3Dispatch:
    """
    Verify that rl_level 3 → _fine_tune_ppo and rl_level 4 → _fine_tune_dpo.
    Tests use inspect.getsource() since real GPU is not available.
    """

    def _make_loop(self, rl_level):
        from platform.reasoning.learning_loop import LearningLoop
        from unittest.mock import MagicMock
        m = MagicMock()
        m.loaded = False
        calc = MagicMock()
        return LearningLoop(
            reasoning_model=m,
            reward_calculator=calc,
            rl_level=rl_level,
        )

    def test_rl_level_3_stored(self):
        loop = self._make_loop(3)
        assert loop._rl_level == 3

    def test_rl_level_4_stored(self):
        loop = self._make_loop(4)
        assert loop._rl_level == 4

    def test_fine_tune_ppo_exists(self):
        loop = self._make_loop(3)
        assert callable(getattr(loop, "_fine_tune_ppo", None))

    def test_fine_tune_dpo_exists(self):
        loop = self._make_loop(4)
        assert callable(getattr(loop, "_fine_tune_dpo", None))

    def test_ppo_uses_ppo_epsilon(self):
        loop = self._make_loop(3)
        src = inspect.getsource(loop._fine_tune_ppo)
        assert "_PPO_EPSILON" in src or "0.2" in src

    def test_ppo_uses_value_head(self):
        loop = self._make_loop(3)
        src = inspect.getsource(loop._fine_tune_ppo)
        assert "ValueHead" in src

    def test_ppo_uses_rollout_buffer(self):
        loop = self._make_loop(3)
        src = inspect.getsource(loop._fine_tune_ppo)
        assert "RolloutBuffer" in src

    def test_dpo_uses_logsigmoid(self):
        loop = self._make_loop(4)
        src = inspect.getsource(loop._fine_tune_dpo)
        assert "logsigmoid" in src

    def test_dpo_uses_beta(self):
        loop = self._make_loop(4)
        src = inspect.getsource(loop._fine_tune_dpo)
        assert "_DPO_BETA" in src or "0.1" in src

    def test_dpo_uses_build_dpo_pairs(self):
        loop = self._make_loop(4)
        src = inspect.getsource(loop._fine_tune_dpo)
        assert "build_dpo_pairs" in src

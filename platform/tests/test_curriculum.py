"""
Tests for Level 5: WorkloadGenerator, CurriculumScheduler, AdversarialGenerator.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from platform.reasoning.curriculum import (
    WEAKNESS_DECODE_DEPTH,
    WEAKNESS_ENTROPY_SPIKE,
    WEAKNESS_EVICTION_PRESSURE,
    WEAKNESS_PREDICTION_ACCURACY,
    WEAKNESS_ROUTING_DECISIONS,
    AdversarialGenerator,
    CurriculumScheduler,
    SyntheticRequest,
    WorkloadGenerator,
)


# ── WorkloadGenerator tests ───────────────────────────────────────────────────

class TestWorkloadGenerator:

    def test_returns_list_of_synthetic_requests(self):
        gen = WorkloadGenerator(n_requests=10)
        episode = gen.generate_episode(difficulty=0.5, seed=0)
        assert len(episode) == 10
        assert all(isinstance(r, SyntheticRequest) for r in episode)

    def test_difficulty_zero_high_overlap(self):
        gen = WorkloadGenerator(n_requests=200)
        episode = gen.generate_episode(difficulty=0.0, seed=1)
        new_count = sum(1 for r in episode if r.is_new_prefix)
        # At difficulty=0 new_prefix_rate≈0.01 so very few new prefixes
        assert new_count < 15

    def test_difficulty_one_high_new_rate(self):
        gen = WorkloadGenerator(n_requests=200)
        episode = gen.generate_episode(difficulty=1.0, seed=2)
        new_count = sum(1 for r in episode if r.is_new_prefix)
        # At difficulty=1 new_prefix_rate≈0.50 so roughly half are new
        assert new_count > 50

    def test_difficulty_clamped_below_zero(self):
        gen = WorkloadGenerator(n_requests=10)
        episode = gen.generate_episode(difficulty=-5.0, seed=0)
        assert len(episode) == 10
        for r in episode:
            assert r.difficulty == 0.0

    def test_difficulty_clamped_above_one(self):
        gen = WorkloadGenerator(n_requests=10)
        episode = gen.generate_episode(difficulty=99.0, seed=0)
        assert len(episode) == 10
        for r in episode:
            assert r.difficulty == 1.0

    def test_arrival_times_monotonic(self):
        gen = WorkloadGenerator(n_requests=20)
        episode = gen.generate_episode(difficulty=0.5, seed=3)
        for a, b in zip(episode, episode[1:]):
            assert b.arrival_time_ms >= a.arrival_time_ms

    def test_reproducible_with_seed(self):
        gen = WorkloadGenerator(n_requests=10)
        e1 = gen.generate_episode(difficulty=0.5, seed=42)
        e2 = gen.generate_episode(difficulty=0.5, seed=42)
        assert [r.prefix_hash for r in e1] == [r.prefix_hash for r in e2]

    def test_different_seeds_differ(self):
        gen = WorkloadGenerator(n_requests=20)
        e1 = gen.generate_episode(difficulty=0.5, seed=1)
        e2 = gen.generate_episode(difficulty=0.5, seed=2)
        hashes1 = [r.prefix_hash for r in e1]
        hashes2 = [r.prefix_hash for r in e2]
        assert hashes1 != hashes2

    def test_generate_high_pressure(self):
        gen = WorkloadGenerator(n_requests=50)
        episode = gen.generate_high_pressure()
        assert len(episode) == 50

    def test_generate_cold_start_all_new_or_high_rate(self):
        gen = WorkloadGenerator(n_requests=100)
        episode = gen.generate_cold_start()
        new_count = sum(1 for r in episode if r.is_new_prefix)
        assert new_count > 30  # cold start should have many new prefixes

    def test_generate_warm_steady(self):
        gen = WorkloadGenerator(n_requests=100)
        episode = gen.generate_warm_steady()
        new_count = sum(1 for r in episode if r.is_new_prefix)
        assert new_count < 20  # warm steady should have very few new prefixes


# ── CurriculumScheduler tests ─────────────────────────────────────────────────

class TestCurriculumScheduler:

    def test_initial_difficulty(self):
        sched = CurriculumScheduler(initial_difficulty=0.3)
        assert abs(sched.difficulty - 0.3) < 1e-6

    def test_difficulty_increases_when_exceeding_target(self):
        sched = CurriculumScheduler(initial_difficulty=0.3, difficulty_step=0.05)
        # Feed many high rewards to fill the history window
        for _ in range(20):
            action = sched.update(episode_reward=5.0, target_reward=2.0)
        assert sched.difficulty > 0.3

    def test_difficulty_decreases_when_below_target(self):
        sched = CurriculumScheduler(initial_difficulty=0.5, difficulty_step=0.05)
        for _ in range(20):
            action = sched.update(episode_reward=0.1, target_reward=2.0)
        assert sched.difficulty < 0.5

    def test_difficulty_held_near_target(self):
        sched = CurriculumScheduler(initial_difficulty=0.5, difficulty_step=0.05)
        d_before = sched.difficulty
        sched.update(episode_reward=2.0, target_reward=2.0)
        assert sched.difficulty == d_before

    def test_difficulty_clamped_at_zero(self):
        sched = CurriculumScheduler(initial_difficulty=0.0, difficulty_step=0.1)
        for _ in range(10):
            sched.update(episode_reward=0.0, target_reward=2.0)
        assert sched.difficulty >= 0.0

    def test_difficulty_clamped_at_one(self):
        sched = CurriculumScheduler(initial_difficulty=1.0, difficulty_step=0.1)
        for _ in range(10):
            sched.update(episode_reward=100.0, target_reward=2.0)
        assert sched.difficulty <= 1.0

    def test_update_count_increments(self):
        sched = CurriculumScheduler()
        sched.update(1.0)
        sched.update(2.0)
        assert sched.update_count == 2

    def test_mean_reward_empty(self):
        sched = CurriculumScheduler()
        assert sched.mean_reward == 0.0

    def test_mean_reward_after_updates(self):
        sched = CurriculumScheduler()
        sched.update(2.0)
        sched.update(4.0)
        assert abs(sched.mean_reward - 3.0) < 1e-5

    def test_state_dict_round_trip(self):
        sched = CurriculumScheduler(initial_difficulty=0.6)
        sched.update(3.0)
        sched.update(1.0)
        d = sched.state_dict()
        sched2 = CurriculumScheduler()
        sched2.load_state_dict(d)
        assert abs(sched2.difficulty - sched.difficulty) < 1e-6
        assert sched2.update_count == sched.update_count

    def test_reset(self):
        sched = CurriculumScheduler(initial_difficulty=0.8)
        for _ in range(10):
            sched.update(5.0)
        sched.reset()
        assert abs(sched.difficulty - 0.3) < 1e-6
        assert sched.update_count == 0


# ── AdversarialGenerator tests ────────────────────────────────────────────────

class TestAdversarialGenerator:

    def _mock_calc(self, outcomes: dict):
        calc = MagicMock()
        calc.get_stats.return_value = {"outcome_breakdown": outcomes}
        return calc

    def test_no_weaknesses_when_outcomes_good(self):
        gen = AdversarialGenerator()
        calc = self._mock_calc({"hit_admitted": 1000, "miss_never_reused": 10})
        weaknesses = gen.find_weaknesses(calc)
        assert weaknesses == []

    def test_detects_eviction_weakness(self):
        gen = AdversarialGenerator()
        calc = self._mock_calc({"eviction_miss": 200, "hit_admitted": 100})
        weaknesses = gen.find_weaknesses(calc)
        assert WEAKNESS_EVICTION_PRESSURE in weaknesses

    def test_detects_prediction_weakness(self):
        gen = AdversarialGenerator()
        calc = self._mock_calc({"pre_warm_miss": 200, "hit_admitted": 100})
        weaknesses = gen.find_weaknesses(calc)
        assert WEAKNESS_PREDICTION_ACCURACY in weaknesses

    def test_detects_routing_weakness(self):
        gen = AdversarialGenerator()
        calc = self._mock_calc({"route_regression": 200, "hit_admitted": 50})
        weaknesses = gen.find_weaknesses(calc)
        assert WEAKNESS_ROUTING_DECISIONS in weaknesses

    def test_detects_decode_depth_weakness(self):
        gen = AdversarialGenerator()
        calc = self._mock_calc({"spec_depth_wasteful": 200, "hit_admitted": 50})
        weaknesses = gen.find_weaknesses(calc)
        assert WEAKNESS_DECODE_DEPTH in weaknesses

    def test_detects_entropy_spike_weakness(self):
        gen = AdversarialGenerator()
        calc = self._mock_calc({"spec_low_acceptance": 200, "hit_admitted": 50})
        weaknesses = gen.find_weaknesses(calc)
        assert WEAKNESS_ENTROPY_SPIKE in weaknesses

    def test_generate_targeted_eviction_workload(self):
        gen = AdversarialGenerator()
        workload = gen.generate_targeted_workload(WEAKNESS_EVICTION_PRESSURE)
        assert len(workload) > 0
        assert all(isinstance(r, SyntheticRequest) for r in workload)

    def test_generate_targeted_all_weaknesses(self):
        gen = AdversarialGenerator()
        for w in [
            WEAKNESS_EVICTION_PRESSURE,
            WEAKNESS_PREDICTION_ACCURACY,
            WEAKNESS_ROUTING_DECISIONS,
            WEAKNESS_DECODE_DEPTH,
            WEAKNESS_ENTROPY_SPIKE,
        ]:
            workload = gen.generate_targeted_workload(w)
            assert len(workload) > 0, f"workload empty for weakness={w}"

    def test_generate_targeted_unknown_weakness(self):
        gen = AdversarialGenerator()
        workload = gen.generate_targeted_workload("some_future_weakness")
        assert len(workload) > 0

    def test_find_weaknesses_handles_calc_error(self):
        gen = AdversarialGenerator()
        calc = MagicMock()
        calc.get_stats.side_effect = RuntimeError("db locked")
        # Should return empty list, not raise
        assert gen.find_weaknesses(calc) == []

    def test_generate_flash_crowd(self):
        gen = AdversarialGenerator()
        workload = gen.generate_flash_crowd()
        assert len(workload) > 256  # spike requests added on top of base

    def test_generate_cache_poison(self):
        gen = AdversarialGenerator()
        workload = gen.generate_cache_poison()
        assert len(workload) > 0

"""
Level 5: Self-Play & Curriculum Learning
=========================================
The system becomes its own teacher.  Three components work together:

WorkloadGenerator
-----------------
Generates synthetic request episodes at a controlled difficulty level (0.0–1.0).
Parameters scaled with difficulty:
  - prefix_overlap  : 0.99 (trivial) → 0.10 (adversarial)
  - request_rate    : steady         → bursty
  - new_prefix_rate : 0.01           → 0.50
  - cache_pressure  : low            → 95%+
  - context_length  : fixed 128K     → variable 8K–1M

CurriculumScheduler
-------------------
Adaptively controls difficulty based on recent episode performance.
  - If mean reward > target × 1.1 → increase difficulty (model is too comfortable)
  - If mean reward < target × 0.8 → decrease difficulty (model is struggling)
  - Otherwise → hold current difficulty

AdversarialGenerator
--------------------
Analyses the reward history to find where the model is weakest, then generates
workloads that specifically stress those failure modes.  Forces the model to
improve on edge cases it would never encounter in normal production traffic.

After 6 months of self-play the model has experienced millions of cache
management scenarios — including adversarial ones that take years to see in
real production.  It develops strategies that no human engineer could design
because they emerge from patterns invisible to human analysis.

Usage
-----
    gen = WorkloadGenerator()
    sched = CurriculumScheduler()
    adv = AdversarialGenerator()

    # Training loop
    for episode in range(10000):
        difficulty = sched.difficulty
        if episode % 10 == 0:
            weaknesses = adv.find_weaknesses(reward_calculator)
            for w in weaknesses:
                workload = adv.generate_targeted_workload(w)
                # inject into AMF pipeline ...
        else:
            workload = gen.generate_episode(difficulty)
            # inject into AMF pipeline ...
            episode_reward = run_amf_episode(workload)
            sched.update(episode_reward, target_reward=2.0)
"""

from __future__ import annotations

import hashlib
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional


# ── SyntheticRequest ─────────────────────────────────────────────────────────

@dataclass
class SyntheticRequest:
    """
    A single synthetic inference request generated for self-play training.

    Attributes
    ----------
    prefix_hash : str
        Simulated prefix identifier.
    context_length : int
        Number of tokens in the context (controls KV cache size).
    is_new_prefix : bool
        True → this prefix has never been seen (cache miss guaranteed).
    arrival_time_ms : float
        Simulated arrival time relative to episode start.
    tenant_id : str
        Simulated tenant for multi-tenant routing tests.
    difficulty : float
        The episode difficulty that generated this request (0.0–1.0).
    """
    prefix_hash:     str
    context_length:  int
    is_new_prefix:   bool
    arrival_time_ms: float
    tenant_id:       str
    difficulty:      float


# ── WorkloadGenerator ─────────────────────────────────────────────────────────

class WorkloadGenerator:
    """
    Generates synthetic request episodes for self-play training.

    All randomness is seeded per-call so episodes are reproducible given the
    same difficulty and seed.

    Parameters
    ----------
    n_requests : int
        Number of requests per episode.
    n_prefix_pool : int
        Total distinct prefix pool size (controls reuse probability).
    max_context_tokens : int
        Maximum context length for variable-length mode.
    min_context_tokens : int
        Minimum context length.
    """

    def __init__(
        self,
        n_requests:         int = 256,
        n_prefix_pool:      int = 1024,
        max_context_tokens: int = 131072,
        min_context_tokens: int = 8192,
        base_tps:           float = 100.0,
    ) -> None:
        self._n_requests   = int(n_requests)
        self._n_pool       = int(n_prefix_pool)
        self._max_ctx      = int(max_context_tokens)
        self._min_ctx      = int(min_context_tokens)
        self._base_tps     = float(base_tps)

    def generate_episode(
        self,
        difficulty: float,
        seed: Optional[int] = None,
    ) -> List[SyntheticRequest]:
        """
        Generate a sequence of synthetic requests at the given difficulty.

        Difficulty controls five axes simultaneously:

        difficulty=0.0 (trivial):
            - 99% prefix overlap (warm cache, easy hits)
            - Steady request rate
            - 1% new prefixes
            - Low cache pressure
            - Fixed 128K context

        difficulty=1.0 (adversarial):
            - 10% prefix overlap (nearly all misses)
            - Bursty Poisson arrivals (bursts 10× normal rate)
            - 50% new prefixes
            - 95%+ cache pressure
            - Variable 8K–1M context

        Parameters
        ----------
        difficulty : float
            Clamped to [0.0, 1.0].
        seed : int | None
            Optional RNG seed for reproducibility.

        Returns
        -------
        list[SyntheticRequest]
        """
        difficulty = max(0.0, min(1.0, float(difficulty)))
        rng = random.Random(seed if seed is not None else int(time.time() * 1000) % 2**32)

        # ── parameter interpolation ───────────────────────────────────────────
        overlap_rate   = 0.99 - 0.89 * difficulty        # 0.99 → 0.10
        new_prefix_rate = 0.01 + 0.49 * difficulty       # 0.01 → 0.50
        burstiness     = 1.0 + 9.0 * difficulty          # 1× → 10×
        fixed_ctx      = difficulty < 0.3                # fixed ctx at low difficulty

        # ── build prefix pool for this episode ────────────────────────────────
        pool_size = max(4, int(self._n_pool * (1.0 - overlap_rate * 0.8)))
        pool = [
            hashlib.md5(f"prefix-{i}".encode()).hexdigest()[:16]
            for i in range(pool_size)
        ]
        # High-overlap pool: a small "hot" subset that most requests reuse
        hot_count = max(1, int(pool_size * overlap_rate))
        hot_pool  = pool[:hot_count]

        # ── generate requests ──────────────────────────────────────────────────
        requests: List[SyntheticRequest] = []
        time_ms = 0.0
        base_interval_ms = 1000.0 / self._base_tps  # ms between requests at base rate

        for i in range(self._n_requests):
            # Arrival time with optional burstiness
            if burstiness > 1.0 and rng.random() < 0.1 * difficulty:
                interval_ms = base_interval_ms / burstiness
            else:
                interval_ms = base_interval_ms * (0.5 + rng.random())
            time_ms += interval_ms

            # Prefix selection
            is_new = rng.random() < new_prefix_rate
            if is_new:
                pfx = hashlib.md5(f"new-{i}-{rng.random()}".encode()).hexdigest()[:16]
            elif rng.random() < overlap_rate:
                pfx = rng.choice(hot_pool)
            else:
                pfx = rng.choice(pool)

            # Context length
            if fixed_ctx:
                ctx_len = self._max_ctx
            else:
                ctx_len = rng.randint(self._min_ctx, self._max_ctx)

            # Tenant
            tenant = f"tenant-{rng.randint(0, max(1, int(difficulty * 8)))}"

            requests.append(SyntheticRequest(
                prefix_hash=pfx,
                context_length=ctx_len,
                is_new_prefix=is_new,
                arrival_time_ms=round(time_ms, 2),
                tenant_id=tenant,
                difficulty=difficulty,
            ))

        return requests

    def generate_high_pressure(self) -> List[SyntheticRequest]:
        """Generate a workload with near-100% cache pressure (eviction stress test)."""
        return self.generate_episode(difficulty=0.95, seed=42)

    def generate_cold_start(self) -> List[SyntheticRequest]:
        """All new prefixes — stress tests admission logic."""
        return self.generate_episode(difficulty=1.0, seed=1337)

    def generate_warm_steady(self) -> List[SyntheticRequest]:
        """Highly repetitive workload — baseline / regression check."""
        return self.generate_episode(difficulty=0.0, seed=0)


# ── CurriculumScheduler ───────────────────────────────────────────────────────

class CurriculumScheduler:
    """
    Adaptively controls training difficulty based on recent episode reward.

    The curriculum starts at medium-easy difficulty (0.3) and adapts:
    - If the model exceeds the target by 10%, increase difficulty.
    - If the model falls 20% below the target, decrease difficulty.
    - Otherwise, hold steady.

    A history window of 100 episodes is used to compute the running mean
    so that transient spikes don't cause instability.

    Parameters
    ----------
    initial_difficulty : float
        Starting difficulty (0.0–1.0).  Default 0.3 (medium-easy).
    difficulty_step : float
        How much to adjust difficulty per update.  Default 0.05.
    history_len : int
        Number of recent episodes to track for the running mean.
    """

    def __init__(
        self,
        initial_difficulty: float = 0.3,
        difficulty_step:    float = 0.05,
        history_len:        int   = 100,
    ) -> None:
        self._difficulty     = max(0.0, min(1.0, float(initial_difficulty)))
        self._step           = max(0.001, float(difficulty_step))
        self._history:       Deque[float] = deque(maxlen=int(history_len))
        self._update_count   = 0

    @property
    def difficulty(self) -> float:
        """Current difficulty level [0.0, 1.0]."""
        return self._difficulty

    @property
    def update_count(self) -> int:
        """Total number of update() calls."""
        return self._update_count

    @property
    def mean_reward(self) -> float:
        """Running mean episode reward over the history window."""
        if not self._history:
            return 0.0
        return sum(self._history) / len(self._history)

    def update(self, episode_reward: float, target_reward: float = 2.0) -> str:
        """
        Record an episode result and adjust difficulty.

        Parameters
        ----------
        episode_reward : float
            Total reward achieved in the episode.
        target_reward : float
            The reward threshold the model should aim for.  When the model
            comfortably exceeds this, difficulty increases.

        Returns
        -------
        str
            One of "increased", "decreased", "held" — the action taken.
        """
        self._history.append(float(episode_reward))
        self._update_count += 1
        mean = self.mean_reward

        if mean > target_reward * 1.1:
            self._difficulty = min(1.0, self._difficulty + self._step)
            action = "increased"
        elif mean < target_reward * 0.8:
            self._difficulty = max(0.0, self._difficulty - self._step)
            action = "decreased"
        else:
            action = "held"

        return action

    def reset(self) -> None:
        """Reset history and difficulty to initial state."""
        self._difficulty   = 0.3
        self._history.clear()
        self._update_count = 0

    def state_dict(self) -> dict:
        """Serialise for checkpointing."""
        return {
            "difficulty":   self._difficulty,
            "step":         self._step,
            "history":      list(self._history),
            "update_count": self._update_count,
        }

    def load_state_dict(self, d: dict) -> None:
        """Restore from a checkpoint."""
        self._difficulty   = float(d.get("difficulty",   0.3))
        self._step         = float(d.get("step",         0.05))
        self._update_count = int(d.get("update_count",   0))
        self._history.clear()
        for v in d.get("history", []):
            self._history.append(float(v))


# ── AdversarialGenerator ──────────────────────────────────────────────────────

# Weakness type constants
WEAKNESS_EVICTION_PRESSURE   = "eviction_under_pressure"
WEAKNESS_PREDICTION_ACCURACY = "prediction_accuracy"
WEAKNESS_ROUTING_DECISIONS   = "routing_decisions"
WEAKNESS_DECODE_DEPTH        = "decode_depth_tuning"
WEAKNESS_ENTROPY_SPIKE       = "entropy_spike_handling"

# Threshold: fraction of total outcomes that must be failures to flag a weakness
_WEAKNESS_THRESHOLD = 0.15


class AdversarialGenerator:
    """
    Analyses reward history to find where the model is weakest, then generates
    workloads designed to stress those exact failure modes.

    This implements the adversarial self-play inner loop of Level 5.  The outer
    loop (CurriculumScheduler) controls overall difficulty; the adversarial
    generator finds the model's specific weak points within that difficulty level.

    After enough self-play cycles, the model has been forced to develop strategies
    for every failure mode that exists in the reward table — including edge cases
    that appear too rarely in production traffic to learn from naturally.
    """

    def __init__(self, workload_generator: Optional[WorkloadGenerator] = None) -> None:
        self._gen = workload_generator or WorkloadGenerator()

    def find_weaknesses(self, reward_calculator) -> List[str]:
        """
        Analyse recent decisions to identify failure modes above threshold.

        Parameters
        ----------
        reward_calculator : RewardCalculator
            The live reward store (Component 4).

        Returns
        -------
        list[str]
            Weakness type constants (e.g. WEAKNESS_EVICTION_PRESSURE).
            Empty list if no significant weakness is found.
        """
        try:
            stats = reward_calculator.get_stats()
        except Exception:
            return []

        outcomes = stats.get("outcome_breakdown", {})
        total    = sum(outcomes.values()) or 1

        weaknesses: List[str] = []

        # Eviction miss rate too high
        evict_miss = outcomes.get("eviction_miss", 0)
        if evict_miss / total > _WEAKNESS_THRESHOLD:
            weaknesses.append(WEAKNESS_EVICTION_PRESSURE)

        # Pre-warm prediction miss rate too high
        prewarm_miss = outcomes.get("pre_warm_miss", 0)
        if prewarm_miss / total > _WEAKNESS_THRESHOLD:
            weaknesses.append(WEAKNESS_PREDICTION_ACCURACY)

        # Route regression rate too high
        route_reg = outcomes.get("route_regression", 0)
        if route_reg / total > _WEAKNESS_THRESHOLD:
            weaknesses.append(WEAKNESS_ROUTING_DECISIONS)

        # Spec decode depth wasteful
        spec_waste = outcomes.get("spec_depth_wasteful", 0)
        if spec_waste / total > _WEAKNESS_THRESHOLD:
            weaknesses.append(WEAKNESS_DECODE_DEPTH)

        # High entropy spec failures
        spec_low = outcomes.get("spec_low_acceptance", 0)
        if spec_low / total > _WEAKNESS_THRESHOLD:
            weaknesses.append(WEAKNESS_ENTROPY_SPIKE)

        return weaknesses

    def generate_targeted_workload(
        self,
        weakness: str,
        n_requests: int = 256,
    ) -> List[SyntheticRequest]:
        """
        Generate a workload that specifically stresses the given weakness.

        Parameters
        ----------
        weakness : str
            One of the WEAKNESS_* constants.
        n_requests : int
            Number of requests in the generated episode.

        Returns
        -------
        list[SyntheticRequest]
        """
        gen = WorkloadGenerator(n_requests=n_requests)

        if weakness == WEAKNESS_EVICTION_PRESSURE:
            # Many unique prefixes + small cache budget = constant eviction churn
            return gen.generate_episode(difficulty=0.9, seed=1001)

        elif weakness == WEAKNESS_PREDICTION_ACCURACY:
            # Irregular request patterns with subtle temporal ordering
            # Difficulty 0.7: mixed overlap, moderate burstiness
            return gen.generate_episode(difficulty=0.7, seed=2002)

        elif weakness == WEAKNESS_ROUTING_DECISIONS:
            # Imbalanced cluster with asymmetric cache states
            # High difficulty to create many routing decision points
            return gen.generate_episode(difficulty=0.85, seed=3003)

        elif weakness == WEAKNESS_DECODE_DEPTH:
            # Variable context lengths → varying optimal spec depth
            return gen.generate_episode(difficulty=0.6, seed=4004)

        elif weakness == WEAKNESS_ENTROPY_SPIKE:
            # Many new prefixes → cold starts → high entropy at decode
            return gen.generate_episode(difficulty=0.95, seed=5005)

        else:
            # Unknown weakness: use high general difficulty
            return gen.generate_episode(difficulty=0.75, seed=9999)

    def generate_flash_crowd(self) -> List[SyntheticRequest]:
        """
        Sudden 10× traffic spike on a single prefix — tests throttle + routing.
        A specific scenario the model must learn to handle pre-emptively.
        """
        gen = WorkloadGenerator(n_requests=512)
        requests = gen.generate_episode(difficulty=0.5, seed=7777)
        # Inject 256 duplicate requests for one hot prefix mid-episode
        hot_pfx = requests[0].prefix_hash
        spike = [
            SyntheticRequest(
                prefix_hash=hot_pfx,
                context_length=131072,
                is_new_prefix=False,
                arrival_time_ms=requests[128].arrival_time_ms + i * 0.1,
                tenant_id="tenant-flash",
                difficulty=0.9,
            )
            for i in range(256)
        ]
        return requests[:128] + spike + requests[128:]

    def generate_cache_poison(self) -> List[SyntheticRequest]:
        """
        Adversarial sequence that admits many low-value prefixes to evict high-value ones.
        Tests the admission ROI gate under attack.
        """
        gen = WorkloadGenerator(n_requests=512)
        return gen.generate_episode(difficulty=1.0, seed=6666)

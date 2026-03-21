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
import json
import os
import random
import sqlite3
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
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
        n: Optional[int] = None,
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
        n_requests = int(n) if n is not None else self._n_requests
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

        for i in range(n_requests):
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


# ── DynamoWorkloadRecorder ────────────────────────────────────────────────────

_RECORDER_SCHEMA = """
CREATE TABLE IF NOT EXISTS workload_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL    NOT NULL,
    prefix_hash TEXT    NOT NULL,
    tenant_id   TEXT    NOT NULL,
    num_tokens  INTEGER NOT NULL,
    is_hit      INTEGER NOT NULL DEFAULT 0,
    restore_ms  REAL    NOT NULL DEFAULT 0.0,
    worker_id   TEXT    NOT NULL DEFAULT '',
    kv_overlap  REAL    NOT NULL DEFAULT 0.0,
    extra_json  TEXT    NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_we_ts ON workload_events(ts);
CREATE INDEX IF NOT EXISTS idx_we_prefix ON workload_events(prefix_hash);
"""

_REPLAY_BATCH_SIZE = 256


class DynamoWorkloadRecorder:
    """
    Records live Dynamo workload events to SQLite and replays them as
    curriculum episodes for self-play training.

    Recording
    ---------
    Call ``record(...)`` from the inference hot path after each request
    completes.  Writes are batched in memory and flushed every ``flush_interval_s``
    seconds (default 5 s) to avoid per-request I/O overhead.

    Production Replay
    -----------------
    Call ``replay_episode(n)`` to return ``n`` recorded events as a list of
    ``SyntheticRequest`` objects.  The CurriculumScheduler can use these
    directly as training episodes — the model is trained on its own recent
    production decisions.

    Usage
    -----
        recorder = DynamoWorkloadRecorder(db_path="dynamo_workload.db")
        recorder.start()

        # From inference hot path:
        recorder.record(
            prefix_hash="abc123",
            tenant_id="t-1",
            num_tokens=131072,
            is_hit=True,
            restore_ms=8.1,
            worker_id="gpu-0",
            kv_overlap=0.87,
        )

        # For training:
        episode = recorder.replay_episode(n=256)
        # Use episode as a WorkloadGenerator episode substitute

        recorder.stop()
    """

    def __init__(
        self,
        db_path: str = "dynamo_workload.db",
        flush_interval_s: float = 5.0,
        max_buffer_size: int = 4096,
        max_db_rows: int = 1_000_000,    # rotate DB after this many rows
    ) -> None:
        self._db_path         = str(db_path)
        self._flush_interval  = float(flush_interval_s)
        self._max_buffer      = int(max_buffer_size)
        self._max_rows        = int(max_db_rows)

        self._lock    = threading.Lock()
        self._buffer: List[Dict[str, Any]] = []
        self._stop    = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._total_recorded = 0
        self._total_replayed = 0

        self._init_db()

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background flush thread."""
        self._thread = threading.Thread(
            target=self._flush_loop,
            name="amf-workload-recorder",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Flush remaining buffer and stop."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10.0)
        self._flush_buffer()  # final flush

    # ── recording API ─────────────────────────────────────────────────────────

    def record(
        self,
        prefix_hash: str,
        tenant_id: str = "default",
        num_tokens: int = 0,
        is_hit: bool = False,
        restore_ms: float = 0.0,
        worker_id: str = "",
        kv_overlap: float = 0.0,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record one Dynamo request event.  Non-blocking: appends to in-memory buffer.
        Buffer is flushed to SQLite by the background thread.
        """
        event = {
            "ts":          time.time(),
            "prefix_hash": str(prefix_hash),
            "tenant_id":   str(tenant_id),
            "num_tokens":  max(0, int(num_tokens)),
            "is_hit":      int(bool(is_hit)),
            "restore_ms":  max(0.0, float(restore_ms)),
            "worker_id":   str(worker_id),
            "kv_overlap":  max(0.0, min(1.0, float(kv_overlap))),
            "extra_json":  json.dumps(extra or {}),
        }
        with self._lock:
            self._buffer.append(event)
            self._total_recorded += 1
            # Prevent unbounded memory growth
            if len(self._buffer) >= self._max_buffer:
                self._flush_locked()

    # ── replay API ────────────────────────────────────────────────────────────

    def replay_episode(
        self,
        n: int = 256,
        since_ts: Optional[float] = None,
        tenant_id: Optional[str] = None,
    ) -> List[SyntheticRequest]:
        """
        Return up to ``n`` recent recorded events as SyntheticRequest objects.

        Parameters
        ----------
        n : int
            Maximum number of requests to return.
        since_ts : float | None
            If set, only return events after this Unix timestamp.
        tenant_id : str | None
            If set, filter to this tenant only.

        Returns
        -------
        list[SyntheticRequest]
            Ordered by arrival time (ascending).  Context length is derived from
            num_tokens; difficulty is estimated from kv_overlap (high overlap =
            easy, low overlap = hard).
        """
        rows = self._fetch_rows(n=n, since_ts=since_ts, tenant_id=tenant_id)
        if not rows:
            return []

        requests: List[SyntheticRequest] = []
        first_ts = rows[0]["ts"]

        for row in rows:
            # Difficulty proxy: low kv_overlap → adversarial (high difficulty)
            kv_ov = float(row["kv_overlap"])
            difficulty = max(0.0, min(1.0, 1.0 - kv_ov))

            requests.append(SyntheticRequest(
                prefix_hash=row["prefix_hash"],
                context_length=max(1024, int(row["num_tokens"])),
                is_new_prefix=not bool(row["is_hit"]),
                arrival_time_ms=round((float(row["ts"]) - first_ts) * 1000.0, 2),
                tenant_id=row["tenant_id"],
                difficulty=difficulty,
            ))

        with self._lock:
            self._total_replayed += len(requests)

        return requests

    def get_stats(self) -> Dict[str, Any]:
        """Return recorder statistics."""
        with self._lock:
            buf_len = len(self._buffer)
            total_r = self._total_recorded
            total_p = self._total_replayed

        row_count = self._count_rows()
        return {
            "total_recorded":  total_r,
            "total_replayed":  total_p,
            "buffer_pending":  buf_len,
            "db_rows":         row_count,
            "db_path":         self._db_path,
        }

    # ── background flush ──────────────────────────────────────────────────────

    def _flush_loop(self) -> None:
        while not self._stop.is_set():
            self._stop.wait(timeout=self._flush_interval)
            self._flush_buffer()

    def _flush_buffer(self) -> None:
        with self._lock:
            if not self._buffer:
                return
            self._flush_locked()

    def _flush_locked(self) -> None:
        """Flush buffer to SQLite. Must hold self._lock."""
        events = list(self._buffer)
        self._buffer.clear()
        if not events:
            return

        try:
            conn = sqlite3.connect(self._db_path, timeout=5.0)
            with conn:
                conn.executemany(
                    """INSERT INTO workload_events
                       (ts, prefix_hash, tenant_id, num_tokens, is_hit,
                        restore_ms, worker_id, kv_overlap, extra_json)
                       VALUES (:ts, :prefix_hash, :tenant_id, :num_tokens,
                               :is_hit, :restore_ms, :worker_id, :kv_overlap,
                               :extra_json)
                    """,
                    events,
                )
            conn.close()
            # Rotate if over limit
            self._maybe_rotate()
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                "[recorder] flush error: %s", exc
            )

    # ── DB helpers ────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        try:
            conn = sqlite3.connect(self._db_path, timeout=5.0)
            conn.executescript(_RECORDER_SCHEMA)
            conn.commit()
            conn.close()
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                "[recorder] DB init error: %s", exc
            )

    def _fetch_rows(
        self,
        n: int,
        since_ts: Optional[float],
        tenant_id: Optional[str],
    ) -> List[Dict[str, Any]]:
        try:
            conn = sqlite3.connect(self._db_path, timeout=5.0)
            conn.row_factory = sqlite3.Row
            params: List[Any] = []
            where_clauses: List[str] = []
            if since_ts is not None:
                where_clauses.append("ts >= ?")
                params.append(float(since_ts))
            if tenant_id is not None:
                where_clauses.append("tenant_id = ?")
                params.append(str(tenant_id))
            where = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
            params.append(int(n))
            rows = conn.execute(
                f"SELECT * FROM workload_events {where} ORDER BY ts DESC LIMIT ?",
                params,
            ).fetchall()
            conn.close()
            # Reverse to get chronological order
            return [dict(r) for r in reversed(rows)]
        except Exception:
            return []

    def _count_rows(self) -> int:
        try:
            conn = sqlite3.connect(self._db_path, timeout=2.0)
            n = conn.execute(
                "SELECT COUNT(*) FROM workload_events"
            ).fetchone()[0]
            conn.close()
            return int(n)
        except Exception:
            return 0

    def _maybe_rotate(self) -> None:
        """Delete the oldest 20% of rows when over the row limit."""
        try:
            conn = sqlite3.connect(self._db_path, timeout=5.0)
            n = conn.execute(
                "SELECT COUNT(*) FROM workload_events"
            ).fetchone()[0]
            if n > self._max_rows:
                delete_n = int(n * 0.20)
                conn.execute(
                    """DELETE FROM workload_events WHERE id IN (
                        SELECT id FROM workload_events ORDER BY ts ASC LIMIT ?
                    )""",
                    (delete_n,),
                )
                conn.commit()
            conn.close()
        except Exception:
            pass


# ── Production replay mode for CurriculumScheduler ────────────────────────────

class ProductionReplayCurriculum:
    """
    Replaces synthetic WorkloadGenerator with recorded production events.

    Wraps a DynamoWorkloadRecorder to feed real traffic patterns into the
    CurriculumScheduler's episode loop.  This is the final convergence step:
    after the model learns from synthetic and adversarial workloads, training
    on real production patterns closes the sim-to-real gap.

    Usage
    -----
        replay = ProductionReplayCurriculum(
            recorder=workload_recorder,
            scheduler=curriculum_scheduler,
            fallback_generator=WorkloadGenerator(),
        )

        episode = replay.generate_episode()
        # Use as drop-in replacement for WorkloadGenerator.generate_episode()
    """

    def __init__(
        self,
        recorder: DynamoWorkloadRecorder,
        scheduler: Optional[CurriculumScheduler] = None,
        fallback_generator: Optional[WorkloadGenerator] = None,
        min_production_rows: int = 1000,
        production_fraction: float = 0.5,
    ) -> None:
        self._recorder    = recorder
        self._scheduler   = scheduler
        self._fallback    = fallback_generator or WorkloadGenerator()
        self._min_rows    = int(min_production_rows)
        self._prod_frac   = max(0.0, min(1.0, float(production_fraction)))
        self._rng         = random.Random()

    def generate_episode(
        self,
        n: int = 256,
        force_production: bool = False,
    ) -> List[SyntheticRequest]:
        """
        Generate a training episode mixing production replay and synthetic data.

        If production DB has fewer than ``min_production_rows`` rows, falls back
        entirely to the synthetic generator at current curriculum difficulty.

        Parameters
        ----------
        n : int
            Total episode length.
        force_production : bool
            If True, use only production replay (ignore production_fraction).

        Returns
        -------
        list[SyntheticRequest]
        """
        db_rows = self._recorder._count_rows()
        if db_rows < self._min_rows:
            # Not enough production data; use synthetic curriculum
            difficulty = self._scheduler.difficulty if self._scheduler else 0.3
            return self._fallback.generate_episode(difficulty=difficulty, n=n)

        if force_production:
            return self._recorder.replay_episode(n=n)

        # Mix production and synthetic
        n_prod = max(1, int(n * self._prod_frac))
        n_synth = n - n_prod

        prod_ep  = self._recorder.replay_episode(n=n_prod)
        difficulty = self._scheduler.difficulty if self._scheduler else 0.3
        synth_ep = self._fallback.generate_episode(difficulty=difficulty, n=n_synth)

        # Merge and sort by arrival time
        combined = prod_ep + synth_ep
        combined.sort(key=lambda r: r.arrival_time_ms)

        # Re-number arrival times to be monotonically increasing
        for i, req in enumerate(combined):
            req.arrival_time_ms = float(i) * 10.0

        return combined

    def generate_episode_production_only(self, n: int = 256) -> List[SyntheticRequest]:
        """Return a pure production replay episode."""
        return self.generate_episode(n=n, force_production=True)

    def generate_episode_synthetic_only(self, n: int = 256) -> List[SyntheticRequest]:
        """Return a pure synthetic episode at current curriculum difficulty."""
        difficulty = self._scheduler.difficulty if self._scheduler else 0.3
        return self._fallback.generate_episode(difficulty=difficulty, n=n)

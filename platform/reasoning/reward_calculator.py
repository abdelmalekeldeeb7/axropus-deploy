"""
Component 4: Reward Calculator
==============================
Scores every decision produced by Component 3 (DecisionExecutor) based on observed
AMF outcomes, and persists the scored decisions to SQLite for Component 5 (LearningLoop).

Reward table
------------
Outcome                              Signal
──────────────────────────────────────────
cache hit within window              +1.0
admitted but never reused            −0.5
eviction caused miss                 −1.0
pre-warm prediction correct          +1.0
pre-warm prediction wrong            −0.3
route reduced latency                +0.5
route increased latency              −0.5

Attribution model
-----------------
The Orchestrator (Component 6) is responsible for linking outcomes back to the
decision that caused them.  The RewardCalculator is a pure store: decisions are
inserted immediately; outcomes are appended asynchronously as AMF reports results.

Component 5 (LearningLoop) calls get_training_batch() to pull
(tensor, decision, total_reward) triples for LoRA fine-tuning.

Thread safety
-------------
All writes are serialised through a threading.Lock.  Reads (score_decision,
get_training_batch, get_stats) acquire the same lock so they see a consistent view.

Usage
-----
    calc = RewardCalculator(db_path="amf_rewards.db")

    # immediately after executor.execute():
    calc.record_decision(decision_id, result, context)

    # when AMF reports a hit for a prefix that was admitted in decision D:
    calc.record_outcome(decision_id, Outcome.HIT_ADMITTED, prefix_hash="abc123")

    # when a pre-warm prediction was never requested (timeout):
    calc.record_outcome(decision_id, Outcome.PRE_WARM_MISS, prefix_hash="def456")

    # Component 5 training pull:
    batch = calc.get_training_batch(limit=64)
    calc.mark_batch_consumed([b["decision_id"] for b in batch])
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)

# Prefill / decode keys mirrored from learning_loop to avoid circular import.
_RC_PREFILL_KEYS = [
    "cache_admission", "admission_priority", "eviction_target",
    "pre_warm_predictions", "route_decision", "batch_group", "throttle_back_pressure",
    "prefer_local_restore", "cross_node_restore_target", "persist_priority",
    "replicate_to_nodes", "evict_hint_nodes",
]
_RC_DECODE_KEYS = [
    "spec_decode_enable", "spec_decode_depth", "spec_mode", "spec_v3_block_size",
    "draft_temperature", "kv_compression_enable", "kv_compression_ratio",
    "decode_batch_priority", "early_stop_confidence", "cuda_graph_hint",
    "adaptive_depth_override",
]


def _to_two_section(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure a stored decision dict is in two-section ``{"prefill": {}, "decode": {}}``
    format.  Handles both flat and already-structured inputs transparently.
    Old flat-format records are lifted into the two-section schema without loss.
    """
    if not isinstance(raw, dict):
        return {"prefill": {}, "decode": {}}
    if "prefill" in raw and "decode" in raw:
        return raw  # Already two-section.
    # Flat format — split into sections.
    return {
        "prefill": {k: raw[k] for k in _RC_PREFILL_KEYS if k in raw},
        "decode":  {k: raw[k] for k in _RC_DECODE_KEYS  if k in raw},
    }


# ── outcome constants ─────────────────────────────────────────────────────────

class Outcome:
    """Outcome type strings used as foreign keys in the outcomes table."""
    # prefill / cache outcomes
    HIT_ADMITTED      = "hit_admitted"      # admitted prefix was hit within window
    MISS_NEVER_REUSED = "miss_never_reused" # admitted prefix was never requested again
    EVICTION_MISS     = "eviction_miss"     # evicted prefix was then requested → miss
    PRE_WARM_HIT      = "pre_warm_hit"      # pre-warm prediction was correct
    PRE_WARM_MISS     = "pre_warm_miss"     # pre-warm prediction was never requested
    ROUTE_IMPROVEMENT = "route_improvement" # routing reduced latency vs baseline
    ROUTE_REGRESSION  = "route_regression"  # routing increased latency vs baseline
    # decode / speculative outcomes
    SPEC_HIGH_ACCEPTANCE   = "spec_high_acceptance"    # acceptance_ema >= 0.80
    SPEC_LOW_ACCEPTANCE    = "spec_low_acceptance"     # acceptance_ema < 0.40
    SPEC_DEPTH_OPTIMAL     = "spec_depth_optimal"      # depth matched acceptance tier
    SPEC_DEPTH_WASTEFUL    = "spec_depth_wasteful"     # depth > 8 with acceptance < 0.60
    SPEC_MODE_MATCH        = "spec_mode_match"         # recommended mode matches Rust lane
    SPEC_MODE_MISMATCH     = "spec_mode_mismatch"      # recommended mode conflicts with Rust
    KV_COMPRESS_SAVED_OOM  = "kv_compress_saved_oom"   # compression prevented OOM
    KV_COMPRESS_UNNECESSARY= "kv_compress_unnecessary" # compression with util < 0.60
    DECODE_LATENCY_IMPROVED= "decode_latency_improved" # decode ms/token improved
    DECODE_LATENCY_WORSENED= "decode_latency_worsened" # decode ms/token worsened
    CUDA_GRAPH_SPEEDUP     = "cuda_graph_speedup"      # cuda_graph_hint=True + speedup
    EARLY_STOP_CORRECT     = "early_stop_correct"      # early stop fired, output complete
    EARLY_STOP_PREMATURE   = "early_stop_premature"    # early stop fired, output truncated
    # Dynamo fleet outcomes
    REPLICATE_HIT          = "replicate_hit"           # replicated prefix was restored on that node
    REPLICATE_WASTE        = "replicate_waste"         # replicated prefix was never reused
    EVICT_HINT_CORRECT     = "evict_hint_correct"      # evict-hinted prefix was not needed
    EVICT_HINT_WRONG       = "evict_hint_wrong"        # evict-hinted prefix was requested → miss


# Reward weights — single source of truth consumed by record_outcome()
REWARD_WEIGHTS: Dict[str, float] = {
    # prefill / cache
    Outcome.HIT_ADMITTED:        +1.0,
    Outcome.MISS_NEVER_REUSED:   -0.5,
    Outcome.EVICTION_MISS:       -1.0,
    Outcome.PRE_WARM_HIT:        +1.0,
    Outcome.PRE_WARM_MISS:       -0.3,
    Outcome.ROUTE_IMPROVEMENT:   +0.5,
    Outcome.ROUTE_REGRESSION:    -0.5,
    # decode / speculative
    Outcome.SPEC_HIGH_ACCEPTANCE:    +1.0,
    Outcome.SPEC_LOW_ACCEPTANCE:     -0.8,
    Outcome.SPEC_DEPTH_OPTIMAL:      +0.5,
    Outcome.SPEC_DEPTH_WASTEFUL:     -0.5,
    Outcome.SPEC_MODE_MATCH:         +0.3,
    Outcome.SPEC_MODE_MISMATCH:      -0.3,
    Outcome.KV_COMPRESS_SAVED_OOM:   +1.5,
    Outcome.KV_COMPRESS_UNNECESSARY: -0.2,
    Outcome.DECODE_LATENCY_IMPROVED: +0.5,
    Outcome.DECODE_LATENCY_WORSENED: -0.5,
    Outcome.CUDA_GRAPH_SPEEDUP:      +0.3,
    Outcome.EARLY_STOP_CORRECT:      +0.8,
    Outcome.EARLY_STOP_PREMATURE:    -0.6,
    # Dynamo fleet outcomes
    Outcome.REPLICATE_HIT:           +0.7,
    Outcome.REPLICATE_WASTE:         -0.3,
    Outcome.EVICT_HINT_CORRECT:      +0.4,
    Outcome.EVICT_HINT_WRONG:        -0.8,
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    decision_id   TEXT    PRIMARY KEY,
    timestamp     REAL    NOT NULL,
    tenant_id     TEXT    NOT NULL DEFAULT '',
    node_id       TEXT    NOT NULL DEFAULT '',
    admitted      INTEGER NOT NULL DEFAULT 1,
    eviction_target TEXT,
    pre_warm_json TEXT    NOT NULL DEFAULT '[]',
    overrides_json TEXT   NOT NULL DEFAULT '[]',
    throttle_applied REAL NOT NULL DEFAULT 0.0,
    spec_k_applied   INTEGER,
    inference_ms  REAL    NOT NULL DEFAULT 0.0,
    is_fallback   INTEGER NOT NULL DEFAULT 0,
    tensor_blob   BLOB,
    decision_json TEXT    NOT NULL DEFAULT '{}',
    consumed      INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS outcomes (
    outcome_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_id   TEXT    NOT NULL,
    timestamp     REAL    NOT NULL,
    outcome_type  TEXT    NOT NULL,
    prefix_hash   TEXT,
    value         REAL    NOT NULL DEFAULT 0.0,
    reward        REAL    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_outcomes_decision ON outcomes (decision_id);
CREATE INDEX IF NOT EXISTS idx_decisions_consumed ON decisions (consumed, timestamp);
"""


# ── RewardCalculator ──────────────────────────────────────────────────────────

class RewardCalculator:
    """
    Persists AMF reasoning decisions and their observed outcomes to SQLite,
    and exposes training batches for the LearningLoop.

    Parameters
    ----------
    db_path : str | Path
        Path to the SQLite database file.  Pass ":memory:" for tests.
    """

    def __init__(self, db_path: str | Path = "amf_rewards.db") -> None:
        self._db_path = str(db_path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            self._db_path,
            check_same_thread=False,
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        log.info("[reward] opened db at %s", self._db_path)

    # ── write path ───────────────────────────────────────────────────────────

    def record_decision(
        self,
        decision_id: str,
        result,          # ExecutionResult from decision_executor
        context,         # ExecutionContext from decision_executor
    ) -> None:
        """
        Persist a decision immediately after executor.execute().

        The metrics tensor and full decision dict are stored so the LearningLoop
        can reconstruct training examples without re-running the model.

        Parameters
        ----------
        decision_id : str
            Unique ID for this decision cycle (generated by the Orchestrator).
        result : ExecutionResult
            Output from DecisionExecutor.execute().
        context : ExecutionContext
            Runtime context that was passed to the executor.
        """
        tensor_blob: Optional[bytes] = None
        if context.tensor is not None:
            arr = np.asarray(context.tensor, dtype=np.float32)
            tensor_blob = arr.tobytes()

        pre_warm = result.decision.get("pre_warm_predictions", [])
        overrides = result.overrides if isinstance(result.overrides, list) else []

        row = (
            decision_id,
            time.time(),
            context.tenant_id,
            context.node_id,
            int(result.admitted),
            result.decision.get("eviction_target"),
            json.dumps(pre_warm),
            json.dumps(overrides),
            float(result.throttle_applied),
            result.spec_k_applied,
            float(result.decision.get("inference_ms", 0.0)),
            int(result.decision.get("is_fallback", False)),
            tensor_blob,
            json.dumps(result.decision),
        )

        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO decisions
                (decision_id, timestamp, tenant_id, node_id, admitted,
                 eviction_target, pre_warm_json, overrides_json,
                 throttle_applied, spec_k_applied, inference_ms, is_fallback,
                 tensor_blob, decision_json)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                row,
            )
            self._conn.commit()

        log.debug(
            "[reward] recorded decision %s admitted=%s overrides=%s",
            decision_id,
            result.admitted,
            overrides,
        )

    def record_outcome(
        self,
        decision_id: str,
        outcome_type: str,
        prefix_hash: Optional[str] = None,
        value: float = 0.0,
    ) -> float:
        """
        Record an observed outcome and return the reward applied.

        The reward is determined solely by outcome_type (see REWARD_WEIGHTS).
        The value field is stored for reference (e.g., latency delta in ms)
        but does not affect the reward magnitude.

        Parameters
        ----------
        decision_id : str
            The decision this outcome is attributed to.
        outcome_type : str
            One of the Outcome.* constants.
        prefix_hash : str | None
            The specific prefix hash this outcome relates to, if any.
        value : float
            Optional auxiliary value (e.g. latency delta). Stored but not
            used to scale the reward — keeps reward accounting simple.

        Returns
        -------
        float
            The reward that was applied (0.0 if unknown outcome_type).
        """
        reward = REWARD_WEIGHTS.get(outcome_type, 0.0)

        with self._lock:
            self._conn.execute(
                """
                INSERT INTO outcomes
                (decision_id, timestamp, outcome_type, prefix_hash, value, reward)
                VALUES (?,?,?,?,?,?)
                """,
                (decision_id, time.time(), outcome_type, prefix_hash, float(value), reward),
            )
            self._conn.commit()

        log.debug(
            "[reward] outcome %s → %s prefix=%s reward=%.2f",
            decision_id,
            outcome_type,
            prefix_hash,
            reward,
        )
        return reward

    def mark_batch_consumed(self, decision_ids: List[str]) -> int:
        """
        Mark decisions as consumed so the LearningLoop doesn't re-train on them.

        Returns
        -------
        int
            Number of rows updated.
        """
        if not decision_ids:
            return 0
        placeholders = ",".join("?" * len(decision_ids))
        with self._lock:
            cur = self._conn.execute(
                f"UPDATE decisions SET consumed=1 WHERE decision_id IN ({placeholders})",
                decision_ids,
            )
            self._conn.commit()
        count = cur.rowcount
        log.debug("[reward] marked %d decisions consumed", count)
        return count

    # ── read path ────────────────────────────────────────────────────────────

    def score_decision(self, decision_id: str) -> float:
        """
        Compute the total reward for a decision from all its recorded outcomes.

        Returns 0.0 if no outcomes exist.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(SUM(reward), 0.0) FROM outcomes WHERE decision_id=?",
                (decision_id,),
            ).fetchone()
        return float(row[0]) if row else 0.0

    def get_training_batch(
        self,
        limit: int = 64,
        min_outcomes: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Pull unconsumed, scored decisions for Component 5 (LearningLoop).

        Only decisions with at least `min_outcomes` recorded outcomes are returned,
        so the LearningLoop always has a real signal to learn from.

        Each entry contains:
            decision_id   : str
            tensor        : np.ndarray float32 (32,) or None if not stored
            decision      : dict   (full ExecutionResult.decision)
            total_reward  : float
            outcome_count : int
            overrides     : list[str]
            timestamp     : float

        Returns
        -------
        List[dict]
        """
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT
                    d.decision_id,
                    d.tensor_blob,
                    d.decision_json,
                    d.overrides_json,
                    d.timestamp,
                    COALESCE(SUM(o.reward), 0.0)  AS total_reward,
                    COUNT(o.outcome_id)            AS outcome_count
                FROM decisions d
                LEFT JOIN outcomes o ON o.decision_id = d.decision_id
                WHERE d.consumed = 0
                GROUP BY d.decision_id
                HAVING outcome_count >= ?
                ORDER BY d.timestamp ASC
                LIMIT ?
                """,
                (min_outcomes, limit),
            ).fetchall()

        batch: List[Dict[str, Any]] = []
        for row in rows:
            decision_id, tensor_blob, decision_json, overrides_json, ts, total_reward, outcome_count = row

            tensor: Optional[np.ndarray] = None
            if tensor_blob:
                try:
                    tensor = np.frombuffer(tensor_blob, dtype=np.float32).copy()
                except Exception:
                    tensor = None

            try:
                raw_decision = json.loads(decision_json) if decision_json else {}
            except Exception:
                raw_decision = {}

            # Normalise to two-section format for the LearningLoop.
            # Old flat-format records (stored before the two-section refactor) are
            # transparently lifted into {"prefill": {...}, "decode": {...}} so
            # _build_training_examples() always receives a consistent structure.
            decision = _to_two_section(raw_decision)

            try:
                overrides = json.loads(overrides_json) if overrides_json else []
            except Exception:
                overrides = []

            batch.append(
                {
                    "decision_id":  decision_id,
                    "tensor":       tensor,
                    "decision":     decision,
                    "total_reward": float(total_reward),
                    "outcome_count": int(outcome_count),
                    "overrides":    overrides,
                    "timestamp":    float(ts),
                }
            )

        log.debug("[reward] training batch: %d items (limit=%d)", len(batch), limit)
        return batch

    def get_stats(self) -> Dict[str, Any]:
        """
        Return aggregate statistics for monitoring.

        Keys:
            total_decisions   : int
            consumed_decisions: int
            total_outcomes    : int
            mean_reward       : float
            outcome_breakdown : dict[outcome_type → count]
        """
        with self._lock:
            total_decisions = self._conn.execute(
                "SELECT COUNT(*) FROM decisions"
            ).fetchone()[0]

            consumed = self._conn.execute(
                "SELECT COUNT(*) FROM decisions WHERE consumed=1"
            ).fetchone()[0]

            total_outcomes = self._conn.execute(
                "SELECT COUNT(*) FROM outcomes"
            ).fetchone()[0]

            mean_reward_row = self._conn.execute(
                "SELECT COALESCE(AVG(reward), 0.0) FROM outcomes"
            ).fetchone()
            mean_reward = float(mean_reward_row[0]) if mean_reward_row else 0.0

            breakdown_rows = self._conn.execute(
                "SELECT outcome_type, COUNT(*) FROM outcomes GROUP BY outcome_type"
            ).fetchall()

        return {
            "total_decisions":    int(total_decisions),
            "consumed_decisions": int(consumed),
            "total_outcomes":     int(total_outcomes),
            "mean_reward":        round(mean_reward, 4),
            "outcome_breakdown":  {r[0]: r[1] for r in breakdown_rows},
        }

    def close(self) -> None:
        """Close the SQLite connection."""
        with self._lock:
            self._conn.close()
        log.info("[reward] db closed")

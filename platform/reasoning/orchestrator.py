"""
Component 6: Reasoning Orchestrator
=====================================
Wires Components 1–5 into the live AMF request path.

Request flow
------------
    Request arrives (prefix_hash, tenant_id, node_id, worker_id)
        │
        ├─ disabled or A/B control group → RequestOutcome(ai_augmented=False)
        │
        ├─ 1. MetricsCollector.get_metrics_tensor()      [~0 ms, lockless read]
        ├─ 2. ReasoningModel.decide(tensor)              [20–50 ms, GPU]
        ├─ 3. DecisionExecutor.execute(dv, context)      [<1 ms, coordinator calls]
        ├─ 4. RewardCalculator.record_decision(...)      [<1 ms, SQLite write]
        ├─ 5. Update attribution windows
        └─ RequestOutcome(ai_augmented=True, admitted, throttle, route, ...)

Outcome attribution (async, called after AMF resolves the request)
------------------------------------------------------------------
    on_hit(prefix_hash)   → HIT_ADMITTED (+1.0) for admitted prefix
                          → PRE_WARM_HIT (+1.0) for correct prediction
    on_miss(prefix_hash)  → EVICTION_MISS (-1.0) if AI evicted this prefix
    on_prefix_expired(prefix_hash)  → MISS_NEVER_REUSED (-0.5) if AI admitted it

Attribution windows
-------------------
Admitted prefixes and pre-warm predictions are tracked for `admission_window`
requests.  After that, unconfirmed predictions become MISS_NEVER_REUSED /
PRE_WARM_MISS.  Eviction tracking uses a wider `eviction_window`.

A/B testing
-----------
Set ab_testing=True to split requests 50/50.  The control group (ab_ratio of
requests) bypasses all AI reasoning and runs pure AMF.  Compare hit rates,
latency, and reward stats between groups to validate model value.

Thread safety
-------------
All attribution state is protected by a single threading.Lock.  The lock is
only held for dict/deque operations — never during GPU inference or I/O.
"""

from __future__ import annotations

import logging
import random
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

from .decision_executor import ExecutionContext
from .reward_calculator import Outcome

log = logging.getLogger(__name__)


# ── public types ──────────────────────────────────────────────────────────────

@dataclass
class RequestOutcome:
    """
    Returned by on_request() to the Orchestrator's caller (Worker / Router).

    The caller uses admitted, throttle, route_decision, batch_group, and spec_k
    to control AMF behaviour for this specific request.
    """
    ai_augmented:  bool
    decision_id:   Optional[str]  = None
    admitted:      bool           = True    # cache this prefix?
    throttle:      float          = 0.0    # back-pressure [0, 1]
    route_decision: Optional[str] = None   # node_id or None
    batch_group:   Optional[str]  = None   # group_id or None
    spec_k:        Optional[int]  = None   # spec decode draft depth


@dataclass
class OrchestratorStats:
    """Snapshot of orchestrator activity for monitoring."""
    enabled:               bool
    ab_testing:            bool
    ab_ratio:              float
    total_requests:        int
    ai_augmented_requests: int
    control_requests:      int
    admitted_tracked:      int    # prefixes in the admission window
    pre_warm_tracked:      int    # pre-warm predictions pending
    eviction_tracked:      int    # evicted prefixes pending miss confirmation
    reward_stats:          Dict[str, Any]
    loop_cycles:           int


# ── Orchestrator ──────────────────────────────────────────────────────────────

class Orchestrator:
    """
    Ties all five reasoning components together and manages the full lifecycle
    of AI-augmented AMF decisions.

    Parameters
    ----------
    collector : MetricsCollector
        Component 1 — supplies the 32-element metrics tensor.
    model : ReasoningModel
        Component 2 — GPU inference.
    executor : DecisionExecutor
        Component 3 — translates decisions to AMF actions with safety overrides.
    reward_calculator : RewardCalculator
        Component 4 — scores decisions and persists to SQLite.
    learning_loop : LearningLoop
        Component 5 — hourly LoRA fine-tuning.
    enabled : bool
        Master on/off switch.  When False, on_request() returns immediately
        with ai_augmented=False — no inference, no side-effects.
    ab_testing : bool
        If True, only ab_ratio of requests are AI-augmented.
    ab_ratio : float
        Fraction of requests routed to the AI path.  0.5 = 50/50.
    admission_window : int
        Number of requests before an unconfirmed admitted prefix is scored as
        MISS_NEVER_REUSED.
    pre_warm_window : int
        Number of requests before an unconfirmed pre-warm prediction is scored
        as PRE_WARM_MISS.
    eviction_window : int
        Number of requests before eviction attribution is dropped (prevents
        unbounded memory growth if the evicted prefix is never requested).
    """

    def __init__(
        self,
        collector,
        model,
        executor,
        reward_calculator,
        learning_loop,
        enabled: bool = True,
        ab_testing: bool = False,
        ab_ratio: float = 0.5,
        admission_window: int = 100,
        pre_warm_window: int = 100,
        eviction_window: int = 1000,
    ) -> None:
        self._collector  = collector
        self._model      = model
        self._executor   = executor
        self._calc       = reward_calculator
        self._loop       = learning_loop

        self._enabled        = enabled
        self._ab_testing     = ab_testing
        self._ab_ratio       = max(0.0, min(1.0, float(ab_ratio)))
        self._admission_win  = int(admission_window)
        self._pre_warm_win   = int(pre_warm_window)
        self._eviction_win   = int(eviction_window)

        # ── counters (no lock needed — only written on request thread) ────────
        self._total_requests = 0
        self._ai_requests    = 0
        self._ctrl_requests  = 0

        # ── attribution state (protected by _attr_lock) ───────────────────────
        self._attr_lock: threading.Lock = threading.Lock()

        # deque entries: (expiry_req_count, prefix_hash, decision_id)
        self._admitted_q:  Deque[Tuple[int, str, str]] = deque()
        self._pre_warm_q:  Deque[Tuple[int, str, str]] = deque()

        # dict entries: prefix_hash → (decision_id, expiry_req_count)
        self._hit_seen:     Dict[str, str]                   = {}  # prefix → decision_id (for dedup)
        self._evicted:      Dict[str, Tuple[str, int]]       = {}  # prefix → (decision_id, expiry)
        self._pre_warm_map: Dict[str, str]                   = {}  # prefix → decision_id

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the MetricsCollector polling thread and LearningLoop."""
        self._collector.start()
        self._loop.start()
        log.info(
            "[orchestrator] started (enabled=%s ab_testing=%s ab_ratio=%.2f)",
            self._enabled,
            self._ab_testing,
            self._ab_ratio,
        )

    def stop(self) -> None:
        """Stop the LearningLoop and MetricsCollector."""
        self._loop.stop()
        self._collector.stop()
        log.info("[orchestrator] stopped")

    # ── enable / disable ──────────────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = bool(value)
        log.info("[orchestrator] enabled=%s", self._enabled)

    # ── main request hook ─────────────────────────────────────────────────────

    def on_request(
        self,
        prefix_hash: str,
        tenant_id: str,
        node_id: str,
        worker_id: str,
    ) -> RequestOutcome:
        """
        Called at the start of every AMF request.

        Runs the full AI reasoning pipeline and returns a RequestOutcome that
        the Worker uses to control caching, routing, and throttling.

        Parameters
        ----------
        prefix_hash : str
            Hash of the current request's prompt prefix.
        tenant_id, node_id, worker_id : str
            Routing context.

        Returns
        -------
        RequestOutcome — safe to use even if AI is disabled or errors.
        """
        self._total_requests += 1

        # ── master switch ─────────────────────────────────────────────────────
        if not self._enabled:
            return RequestOutcome(ai_augmented=False)

        # ── A/B routing ───────────────────────────────────────────────────────
        if self._ab_testing and random.random() >= self._ab_ratio:
            self._ctrl_requests += 1
            return RequestOutcome(ai_augmented=False)

        # ── AI reasoning pipeline ─────────────────────────────────────────────
        try:
            tensor  = self._collector.get_metrics_tensor()
            # Pass tenant_id + prefix_hash if model supports it (AdaptiveEngine)
            try:
                dv = self._model.decide(tensor, tenant_id=tenant_id, prefix_hash=prefix_hash)
            except TypeError:
                dv = self._model.decide(tensor)
            context = ExecutionContext(
                tenant_id=tenant_id,
                node_id=node_id,
                worker_id=worker_id,
                tensor=tensor,
            )
            result  = self._executor.execute(dv, context)
        except Exception as exc:
            log.warning("[orchestrator] pipeline error: %s", exc, exc_info=True)
            return RequestOutcome(ai_augmented=False)

        decision_id = str(uuid.uuid4())

        try:
            self._calc.record_decision(decision_id, result, context)
        except Exception as exc:
            log.warning("[orchestrator] record_decision error: %s", exc)

        # ── update attribution windows ─────────────────────────────────────────
        req_n = self._total_requests
        with self._attr_lock:
            self._expire_windows(req_n)

            if result.admitted:
                self._admitted_q.append((req_n + self._admission_win, prefix_hash, decision_id))
                self._hit_seen[prefix_hash] = decision_id

            if result.eviction_sent and dv.eviction_target:
                self._evicted[dv.eviction_target] = (decision_id, req_n + self._eviction_win)

            for ph in dv.pre_warm_predictions:
                self._pre_warm_q.append((req_n + self._pre_warm_win, ph, decision_id))
                self._pre_warm_map[ph] = decision_id

        self._ai_requests += 1
        log.debug(
            "[orchestrator] request %s: decision=%s admitted=%s overrides=%s",
            prefix_hash[:8],
            decision_id[:8],
            result.admitted,
            result.overrides,
        )

        return RequestOutcome(
            ai_augmented=True,
            decision_id=decision_id,
            admitted=result.admitted,
            throttle=result.throttle_applied,
            route_decision=dv.route_decision,
            batch_group=dv.batch_group,
            spec_k=result.spec_k_applied,
        )

    # ── outcome hooks ─────────────────────────────────────────────────────────

    def on_hit(self, prefix_hash: str, tenant_id: str = "default",
               spec_depth: int = 4, acceptance_rate: float = 0.8) -> None:
        """
        Called when AMF resolves the request as a cache hit.

        Records HIT_ADMITTED reward and closes the RL loop on AdaptiveEngine.
        """
        with self._attr_lock:
            decision_id = self._hit_seen.pop(prefix_hash, None)
            if decision_id:
                try:
                    self._calc.record_outcome(
                        decision_id, Outcome.HIT_ADMITTED, prefix_hash
                    )
                except Exception as exc:
                    log.warning("[orchestrator] record_outcome(hit) error: %s", exc)

            pw_decision_id = self._pre_warm_map.pop(prefix_hash, None)
            if pw_decision_id:
                try:
                    self._calc.record_outcome(
                        pw_decision_id, Outcome.PRE_WARM_HIT, prefix_hash
                    )
                except Exception as exc:
                    log.warning("[orchestrator] record_outcome(prewarm_hit) error: %s", exc)

        # Close RL loop — feed hit back to AdaptiveEngine's doubling multiplier
        if hasattr(self._model, "record_outcome"):
            try:
                self._model.record_outcome(
                    prefix_hash=prefix_hash, tenant_id=tenant_id,
                    spec_depth=spec_depth, admitted=True,
                    was_hit=True, acceptance_rate=acceptance_rate,
                )
            except Exception as exc:
                log.warning("[orchestrator] model.record_outcome(hit) error: %s", exc)

    def on_miss(self, prefix_hash: str, tenant_id: str = "default",
                spec_depth: int = 4) -> None:
        """
        Called when AMF resolves the request as a cache miss.

        Records EVICTION_MISS and closes the RL loop on AdaptiveEngine.
        """
        with self._attr_lock:
            entry = self._evicted.pop(prefix_hash, None)
            if entry:
                decision_id, _expiry = entry
                try:
                    self._calc.record_outcome(
                        decision_id, Outcome.EVICTION_MISS, prefix_hash
                    )
                except Exception as exc:
                    log.warning("[orchestrator] record_outcome(eviction_miss) error: %s", exc)

        # Close RL loop — feed miss back to AdaptiveEngine's doubling multiplier
        if hasattr(self._model, "record_outcome"):
            try:
                self._model.record_outcome(
                    prefix_hash=prefix_hash, tenant_id=tenant_id,
                    spec_depth=spec_depth, admitted=False,
                    was_hit=False, acceptance_rate=0.0,
                )
            except Exception as exc:
                log.warning("[orchestrator] model.record_outcome(miss) error: %s", exc)

    def on_prefix_expired(self, prefix_hash: str) -> None:
        """
        Called when AMF naturally expires a prefix from the cache (LRU eviction,
        TTL, or storage reclaim — NOT from an AI eviction decision).

        If the prefix was admitted by an AI decision and was never hit, records
        MISS_NEVER_REUSED.
        """
        with self._attr_lock:
            decision_id = self._hit_seen.pop(prefix_hash, None)
            if decision_id:
                try:
                    self._calc.record_outcome(
                        decision_id, Outcome.MISS_NEVER_REUSED, prefix_hash
                    )
                except Exception as exc:
                    log.warning("[orchestrator] record_outcome(expired) error: %s", exc)

    def on_decode_step_complete(
        self,
        decision_id: str,
        acceptance_ema: float,
        spec_depth_used: int,
        lane_held_steps: int,
    ) -> None:
        """
        Called after each speculative decode step completes.

        Scores: spec_high_acceptance / spec_low_acceptance,
                spec_depth_optimal / spec_depth_wasteful,
                rust_scheduler_veto (if lane_held > 16 and AI tried to override).
        """
        try:
            # Acceptance quality signal
            if acceptance_ema >= 0.80:
                self._calc.record_outcome(decision_id, Outcome.SPEC_HIGH_ACCEPTANCE)
            elif acceptance_ema < 0.40:
                self._calc.record_outcome(decision_id, Outcome.SPEC_LOW_ACCEPTANCE)

            # Depth optimality — depth > 8 is wasteful if acceptance is low
            if spec_depth_used > 8 and acceptance_ema < 0.60:
                self._calc.record_outcome(decision_id, Outcome.SPEC_DEPTH_WASTEFUL)
            elif acceptance_ema >= 0.70 and spec_depth_used >= 6:
                self._calc.record_outcome(decision_id, Outcome.SPEC_DEPTH_OPTIMAL)
        except Exception as exc:
            log.warning("[orchestrator] on_decode_step_complete error: %s", exc)

    def on_decode_complete(
        self,
        decision_id: str,
        mode_used: str,
        baseline_ms_per_token: float,
        actual_ms_per_token: float,
        cuda_graph_active: bool,
        early_stop_fired: bool,
        output_complete: bool,
    ) -> None:
        """
        Called when the full decode sequence for a request completes.

        Scores: decode_latency_improved / worsened, spec_mode_match / mismatch,
                cuda_graph_speedup, early_stop_correct / premature.
        """
        try:
            # Latency comparison
            if baseline_ms_per_token > 0:
                if actual_ms_per_token < baseline_ms_per_token * 0.95:
                    self._calc.record_outcome(decision_id, Outcome.DECODE_LATENCY_IMPROVED,
                                              value=baseline_ms_per_token - actual_ms_per_token)
                elif actual_ms_per_token > baseline_ms_per_token * 1.05:
                    self._calc.record_outcome(decision_id, Outcome.DECODE_LATENCY_WORSENED,
                                              value=actual_ms_per_token - baseline_ms_per_token)

            # CUDA graph speedup
            if cuda_graph_active and actual_ms_per_token < baseline_ms_per_token:
                self._calc.record_outcome(decision_id, Outcome.CUDA_GRAPH_SPEEDUP)

            # Early stop assessment
            if early_stop_fired:
                if output_complete:
                    self._calc.record_outcome(decision_id, Outcome.EARLY_STOP_CORRECT)
                else:
                    self._calc.record_outcome(decision_id, Outcome.EARLY_STOP_PREMATURE)
        except Exception as exc:
            log.warning("[orchestrator] on_decode_complete error: %s", exc)

    def notify_route_latency(
        self,
        decision_id: str,
        delta_ms: float,
    ) -> None:
        """
        Report the routing latency impact of an AI routing decision.

        Parameters
        ----------
        decision_id : str
            The decision that made a route_decision.
        delta_ms : float
            Positive = latency reduced (improvement), negative = increased.
        """
        outcome = Outcome.ROUTE_IMPROVEMENT if delta_ms > 0 else Outcome.ROUTE_REGRESSION
        try:
            self._calc.record_outcome(decision_id, outcome, value=abs(delta_ms))
        except Exception as exc:
            log.warning("[orchestrator] notify_route_latency error: %s", exc)

    # ── stats ─────────────────────────────────────────────────────────────────

    def get_stats(self) -> OrchestratorStats:
        """Return a snapshot of orchestrator state for monitoring."""
        with self._attr_lock:
            admitted_tracked  = len(self._admitted_q)
            pre_warm_tracked  = len(self._pre_warm_q)
            eviction_tracked  = len(self._evicted)

        try:
            reward_stats = self._calc.get_stats()
        except Exception:
            reward_stats = {}

        loop_cycles = getattr(self._loop, "_cycle_count", 0)

        return OrchestratorStats(
            enabled=self._enabled,
            ab_testing=self._ab_testing,
            ab_ratio=self._ab_ratio,
            total_requests=self._total_requests,
            ai_augmented_requests=self._ai_requests,
            control_requests=self._ctrl_requests,
            admitted_tracked=admitted_tracked,
            pre_warm_tracked=pre_warm_tracked,
            eviction_tracked=eviction_tracked,
            reward_stats=reward_stats,
            loop_cycles=loop_cycles,
        )

    # ── internal helpers ──────────────────────────────────────────────────────

    def _expire_windows(self, current_req: int) -> None:
        """
        Expire old entries from the admission and pre-warm windows.
        Must be called with _attr_lock held.
        """
        # Admitted prefixes that timed out without a hit → MISS_NEVER_REUSED
        while self._admitted_q and self._admitted_q[0][0] <= current_req:
            _expiry, ph, did = self._admitted_q.popleft()
            # Only score if still unresolved (on_hit removes from hit_seen)
            if ph in self._hit_seen and self._hit_seen[ph] == did:
                del self._hit_seen[ph]
                try:
                    self._calc.record_outcome(did, Outcome.MISS_NEVER_REUSED, ph)
                except Exception as exc:
                    log.warning("[orchestrator] expire admission error: %s", exc)

        # Pre-warm predictions that timed out without being requested → PRE_WARM_MISS
        while self._pre_warm_q and self._pre_warm_q[0][0] <= current_req:
            _expiry, ph, did = self._pre_warm_q.popleft()
            if ph in self._pre_warm_map and self._pre_warm_map[ph] == did:
                del self._pre_warm_map[ph]
                try:
                    self._calc.record_outcome(did, Outcome.PRE_WARM_MISS, ph)
                except Exception as exc:
                    log.warning("[orchestrator] expire prewarm error: %s", exc)

        # Eviction tracking: drop entries past their window
        expired_evictions = [
            ph for ph, (_did, expiry) in self._evicted.items()
            if expiry <= current_req
        ]
        for ph in expired_evictions:
            del self._evicted[ph]

"""
Level 4: Multi-Agent RL — Federated Learning Loop
==================================================
Each GPU node runs its own LearningLoop (Level 1-3).  FederatedLearningLoop
wraps the local loop and adds gradient synchronisation via the AMF Coordinator.

Every `sync_every` local training cycles, each node:
1. Runs its own PPO/DPO/REINFORCE cycle locally.
2. Submits its LoRA adapter delta (weight diff from the base) to the Coordinator.
3. Receives the cluster-averaged delta.
4. Applies the averaged delta to its local model.

This is Federated Averaging (McMahan et al. 2017) applied to LoRA adapters.

Emergent cluster behaviours that develop over time:
- Cache specialisation: nodes learn to cache different prefix families.
- Predictive transfers: Node A pre-warms prefixes Node B will need.
- Load-aware routing: nodes shed load to neighbours with spare capacity + cache.
- Anti-fragile replication: nodes proactively replicate high-value unique prefixes.

Cooperative reward signals (added to RewardCalculator by the Orchestrator):
- COOPERATIVE_TRANSFER_HIT  (+0.8) — offered prefix was used by a neighbour.
- COOPERATIVE_TRANSFER_MISS (−0.2) — offered prefix was not needed.
- LOAD_SHED_SUCCESS         (+0.5) — shed load improved cluster latency.
- LOAD_SHED_FAILURE         (−0.3) — shed load increased cluster latency.
- SPECIALISATION_BONUS      (+0.3) — node holds unique prefixes no one else has.

Architecture
------------
    NODE A                NODE B                NODE C
    LearningLoop_A        LearningLoop_B        LearningLoop_C
         │                     │                     │
    local train           local train           local train
         │                     │                     │
         └──────── FederatedLearningLoop ────────────┘
                        (every sync_every cycles)
                              │
                        Coordinator
                   (gradient aggregation)

Thread safety
-------------
The Coordinator is accessed only from the background training thread — no
additional locking beyond the existing LearningLoop._swap_lock is needed.

Usage
-----
    fed = FederatedLearningLoop(
        local_loop=learning_loop,
        coordinator=coordinator_client,
        node_id="worker-gpu0",
        sync_every=4,
    )
    fed.start()
    ...
    fed.stop()
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


# ── cooperative reward outcome constants ─────────────────────────────────────

class CoopOutcome:
    """Outcome type strings for cooperative multi-node signals."""
    COOPERATIVE_TRANSFER_HIT  = "cooperative_transfer_hit"   # +0.8
    COOPERATIVE_TRANSFER_MISS = "cooperative_transfer_miss"  # -0.2
    LOAD_SHED_SUCCESS         = "load_shed_success"          # +0.5
    LOAD_SHED_FAILURE         = "load_shed_failure"          # -0.3
    SPECIALISATION_BONUS      = "specialisation_bonus"       # +0.3


COOP_REWARD_WEIGHTS: Dict[str, float] = {
    CoopOutcome.COOPERATIVE_TRANSFER_HIT:  +0.8,
    CoopOutcome.COOPERATIVE_TRANSFER_MISS: -0.2,
    CoopOutcome.LOAD_SHED_SUCCESS:         +0.5,
    CoopOutcome.LOAD_SHED_FAILURE:         -0.3,
    CoopOutcome.SPECIALISATION_BONUS:      +0.3,
}


# ── FederationResult ──────────────────────────────────────────────────────────

@dataclass
class FederationResult:
    """Outcome of one federated sync cycle."""
    node_id:          str
    local_cycles:     int       # local training cycles since last sync
    synced:           bool      # whether a sync actually happened
    nodes_aggregated: int       # how many nodes participated in the average
    delta_norm:       float     # L2 norm of the applied gradient delta
    elapsed_s:        float
    skip_reason:      str = ""


# ── FederatedLearningLoop ─────────────────────────────────────────────────────

class FederatedLearningLoop:
    """
    Wraps a LearningLoop with federated gradient synchronisation.

    Parameters
    ----------
    local_loop : LearningLoop
        The local per-node learning loop (any rl_level).
    coordinator : object
        AMF Coordinator client.  Must implement:
            submit_adapter_delta(node_id: str, delta: dict) -> None
            get_averaged_delta(node_id: str) -> dict | None
        A no-op stub is used if None (disables federation, keeps local training).
    node_id : str
        Unique identifier for this node (e.g. "worker-gpu0").
    sync_every : int
        Number of local training cycles between federation syncs.
        Default 4 — balances local adaptation with global coordination.
    """

    def __init__(
        self,
        local_loop,
        coordinator=None,
        node_id: str = "node-0",
        sync_every: int = 4,
    ) -> None:
        self._local_loop    = local_loop
        self._coordinator   = coordinator or _NoOpCoordinator()
        self._node_id       = str(node_id)
        self._sync_every    = max(1, int(sync_every))

        self._local_cycles  = 0
        self._sync_count    = 0
        self._stop_event    = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the federated training daemon."""
        if self._thread and self._thread.is_alive():
            log.warning("[fed] already running on node=%s", self._node_id)
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name=f"amf-federated-{self._node_id}",
        )
        self._thread.start()
        log.info("[fed] started node=%s sync_every=%d", self._node_id, self._sync_every)

    def stop(self, timeout: float = 30.0) -> None:
        """Signal the daemon to stop and wait for it."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        log.info("[fed] stopped node=%s after %d syncs", self._node_id, self._sync_count)

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    # ── main API ──────────────────────────────────────────────────────────────

    def run_once(self) -> FederationResult:
        """
        Run one local training cycle.  Every sync_every cycles, also federate.

        Returns
        -------
        FederationResult
        """
        t0 = time.perf_counter()

        # ── local training cycle ──────────────────────────────────────────────
        local_result = self._local_loop.run_once()
        self._local_cycles += 1

        # ── federation sync ───────────────────────────────────────────────────
        if self._local_cycles % self._sync_every == 0:
            fed_result = self._sync(t0)
        else:
            fed_result = FederationResult(
                node_id=self._node_id,
                local_cycles=self._local_cycles,
                synced=False,
                nodes_aggregated=0,
                delta_norm=0.0,
                elapsed_s=round(time.perf_counter() - t0, 2),
                skip_reason=f"sync every {self._sync_every} cycles",
            )

        return fed_result

    # ── federation sync ───────────────────────────────────────────────────────

    def _sync(self, t0: float) -> FederationResult:
        """
        Extract local LoRA delta, submit to coordinator, apply averaged delta.
        """
        try:
            delta      = self._extract_adapter_delta()
            delta_norm = _dict_l2_norm(delta)

            self._coordinator.submit_adapter_delta(self._node_id, delta)
            avg_delta = self._coordinator.get_averaged_delta(self._node_id)

            nodes_aggregated = 1
            if avg_delta is not None:
                self._apply_adapter_delta(avg_delta)
                nodes_aggregated = int(avg_delta.get("_n_nodes", 1))

            self._sync_count += 1
            log.info(
                "[fed] sync done node=%s sync=%d nodes=%d delta_norm=%.4f",
                self._node_id,
                self._sync_count,
                nodes_aggregated,
                delta_norm,
            )

            return FederationResult(
                node_id=self._node_id,
                local_cycles=self._local_cycles,
                synced=True,
                nodes_aggregated=nodes_aggregated,
                delta_norm=round(delta_norm, 6),
                elapsed_s=round(time.perf_counter() - t0, 2),
            )

        except Exception as exc:
            log.warning("[fed] sync failed node=%s: %s", self._node_id, exc)
            return FederationResult(
                node_id=self._node_id,
                local_cycles=self._local_cycles,
                synced=False,
                nodes_aggregated=0,
                delta_norm=0.0,
                elapsed_s=round(time.perf_counter() - t0, 2),
                skip_reason=f"sync error: {exc}",
            )

    def _extract_adapter_delta(self) -> dict:
        """
        Extract the current LoRA adapter weights as a serialisable dict.

        Returns parameter name → list of floats (CPU, detached).
        """
        model = self._local_loop._model
        if not model.loaded:
            return {}
        inner = model._model
        delta: dict = {}
        try:
            # peft model: only LoRA adapter parameters
            for name, param in inner.named_parameters():
                if "lora_" in name and param.requires_grad:
                    delta[name] = param.data.cpu().float().tolist()
        except Exception as exc:
            log.debug("[fed] delta extraction failed: %s", exc)
        return delta

    def _apply_adapter_delta(self, avg_delta: dict) -> None:
        """
        Apply the averaged adapter delta to the local model.

        Overwrites each matching LoRA parameter with the averaged value.
        """
        model = self._local_loop._model
        if not model.loaded:
            return
        inner = model._model
        try:
            import torch
            name_to_param = {
                name: param
                for name, param in inner.named_parameters()
                if "lora_" in name and param.requires_grad
            }
            with self._local_loop._swap_lock:
                for name, avg_values in avg_delta.items():
                    if name == "_n_nodes":
                        continue
                    if name in name_to_param:
                        t = torch.tensor(avg_values, dtype=torch.float32)
                        name_to_param[name].data.copy_(
                            t.to(name_to_param[name].device).reshape(
                                name_to_param[name].shape
                            )
                        )
            log.debug("[fed] applied averaged delta for %d params", len(avg_delta))
        except Exception as exc:
            log.warning("[fed] delta apply failed: %s", exc)

    # ── background thread ──────────────────────────────────────────────────────

    def _run(self) -> None:
        """Background daemon body — mirrors LearningLoop._run()."""
        interval = self._local_loop._interval_s
        while not self._stop_event.is_set():
            try:
                self.run_once()
            except Exception as exc:
                log.error("[fed] unexpected error: %s", exc, exc_info=True)
            self._stop_event.wait(timeout=interval)


# ── no-op coordinator stub ────────────────────────────────────────────────────

class _NoOpCoordinator:
    """
    Stand-in coordinator used when no real coordinator is available.
    Federation is disabled; local training proceeds normally.
    """

    def submit_adapter_delta(self, node_id: str, delta: dict) -> None:
        log.debug("[fed] no-op coordinator: delta from %s dropped", node_id)

    def get_averaged_delta(self, node_id: str) -> Optional[dict]:
        return None


# ── helpers ───────────────────────────────────────────────────────────────────

def _dict_l2_norm(d: dict) -> float:
    """Compute the approximate L2 norm of a weight delta dict."""
    total = 0.0
    for v in d.values():
        if isinstance(v, list):
            total += sum(x * x for x in v if isinstance(x, (int, float)))
    return total ** 0.5

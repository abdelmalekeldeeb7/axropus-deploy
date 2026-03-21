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

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# ── NATS transport auto-detection ─────────────────────────────────────────────
# If DYNAMO_NATS_URL is set and nats-py is installed, FederatedLearningLoop
# will automatically use NatsFederatedTransport instead of the HTTP coordinator.
_NATS_URL_ENV = "DYNAMO_NATS_URL"
_NATS_SUBJECT_GRADIENT = "amf.federated.gradient"
_NATS_SUBJECT_COOP_REWARD = "amf.federated.coop_reward"


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
        nats_transport: Optional["NatsFederatedTransport"] = None,
        auto_detect_nats: bool = True,
    ) -> None:
        self._local_loop    = local_loop
        self._node_id       = str(node_id)
        self._sync_every    = max(1, int(sync_every))

        # Transport priority: explicit nats_transport > auto-detect > HTTP coordinator
        if nats_transport is not None:
            self._coordinator = nats_transport
            log.info("[fed] using explicit NATS transport for node=%s", self._node_id)
        elif auto_detect_nats:
            nats = _build_nats_transport_if_available(self._node_id)
            self._coordinator = nats if nats is not None else (coordinator or _NoOpCoordinator())
        else:
            self._coordinator = coordinator or _NoOpCoordinator()

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


def _build_nats_transport_if_available(node_id: str) -> Optional["NatsFederatedTransport"]:
    """
    Auto-detect NATS and return a NatsFederatedTransport if available.
    Returns None if DYNAMO_NATS_URL is not set or nats-py is not installed.
    """
    nats_url = os.environ.get(_NATS_URL_ENV, "")
    if not nats_url:
        return None
    try:
        import nats  # noqa: F401 — just checking import
        transport = NatsFederatedTransport(nats_url=nats_url, node_id=node_id)
        transport.start_sync()
        log.info("[fed] NATS transport auto-detected: url=%s", nats_url)
        return transport
    except ImportError:
        log.debug("[fed] nats-py not installed; using HTTP coordinator")
        return None
    except Exception as exc:
        log.warning("[fed] NATS transport init failed: %s", exc)
        return None


# ── NatsFederatedTransport ────────────────────────────────────────────────────

class NatsFederatedTransport:
    """
    NATS-based federated transport replacing the HTTP AMF Coordinator.

    Uses two subjects:
      amf.federated.gradient      — serialised LoRA adapter deltas
      amf.federated.coop_reward   — cooperative reward signals

    Each message includes a ``node_id`` field so the GradientAggregator
    (running in an aggregator service or embedded in the coordinator) can
    identify the sender and build the cluster average.

    The transport is symmetric — every node both publishes its own delta AND
    subscribes to receive the averaged result published by the aggregator.

    Parameters
    ----------
    nats_url : str
        NATS server URL (e.g. "nats://localhost:4222").
    node_id : str
        Unique identifier for this node.
    gradient_subject : str
        Subject to publish deltas on (default: amf.federated.gradient).
    coop_reward_subject : str
        Subject to publish cooperative rewards on.
    aggregator_reply_subject : str
        Subject the aggregator publishes averaged results to.
        Default: amf.federated.gradient.avg
    """

    def __init__(
        self,
        nats_url: str = "nats://localhost:4222",
        node_id: str = "node-0",
        gradient_subject: str = _NATS_SUBJECT_GRADIENT,
        coop_reward_subject: str = _NATS_SUBJECT_COOP_REWARD,
        aggregator_reply_subject: str = "amf.federated.gradient.avg",
    ) -> None:
        self._nats_url        = nats_url
        self._node_id         = node_id
        self._grad_subject    = gradient_subject
        self._coop_subject    = coop_reward_subject
        self._reply_subject   = aggregator_reply_subject

        self._stop_event = threading.Event()
        self._nats_thread: Optional[threading.Thread] = None

        # inbox for averaged deltas published by aggregator
        self._avg_delta_inbox: Optional[Dict[str, Any]] = None
        self._avg_delta_event = threading.Event()
        self._lock = threading.Lock()

        # asyncio loop running in the NATS thread
        self._loop = None

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def start_sync(self) -> None:
        """Start the NATS event loop in a background thread."""
        self._nats_thread = threading.Thread(
            target=self._nats_thread_main,
            name=f"amf-nats-fed-{self._node_id}",
            daemon=True,
        )
        self._nats_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._nats_thread:
            self._nats_thread.join(timeout=3.0)

    # ── coordinator interface ─────────────────────────────────────────────────
    # These match the interface expected by FederatedLearningLoop._coordinator.

    def submit_adapter_delta(self, node_id: str, delta: Dict[str, Any]) -> None:
        """Publish the LoRA adapter delta to the NATS gradient subject."""
        payload = {"node_id": node_id, "delta": delta}
        self._publish_sync(self._grad_subject, payload)
        # Reset inbox and wait for averaged response
        with self._lock:
            self._avg_delta_inbox = None
        self._avg_delta_event.clear()

    def get_averaged_delta(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        Wait up to 10 s for the aggregator to publish an averaged delta.
        Returns the averaged delta dict, or None on timeout.
        """
        received = self._avg_delta_event.wait(timeout=10.0)
        if not received:
            log.warning("[nats-fed] timeout waiting for averaged delta node=%s", node_id)
            return None
        with self._lock:
            return self._avg_delta_inbox

    def emit_coop_reward(
        self,
        outcome: str,
        from_node: str,
        to_node: str,
        prefix_hash: str = "",
        reward: float = 0.0,
    ) -> None:
        """
        Publish a cooperative reward signal so neighbours can score their decisions.

        Parameters
        ----------
        outcome : str
            CoopOutcome constant (e.g. COOPERATIVE_TRANSFER_HIT).
        from_node : str
            Node that initiated the cooperative action.
        to_node : str
            Node that benefited (or didn't).
        prefix_hash : str
            Relevant prefix hash, if applicable.
        reward : float
            Reward magnitude (positive = good, negative = bad).
        """
        payload = {
            "outcome":     outcome,
            "from_node":   from_node,
            "to_node":     to_node,
            "prefix_hash": prefix_hash,
            "reward":      reward,
            "ts":          time.time(),
        }
        self._publish_sync(self._coop_subject, payload)
        log.debug(
            "[nats-fed] coop_reward outcome=%s from=%s to=%s reward=%.2f",
            outcome, from_node, to_node, reward,
        )

    # ── internal NATS loop ────────────────────────────────────────────────────

    def _nats_thread_main(self) -> None:
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            loop.run_until_complete(self._nats_main())
        except Exception as exc:
            log.warning("[nats-fed] loop error: %s", exc)
        finally:
            loop.close()

    async def _nats_main(self) -> None:
        import asyncio
        try:
            import nats  # type: ignore[import]
        except ImportError:
            log.error("[nats-fed] nats-py not installed")
            return

        nc = await nats.connect(self._nats_url)
        try:
            # Subscribe to averaged delta replies from the aggregator
            await nc.subscribe(
                self._reply_subject,
                cb=self._on_avg_delta_message,
            )
            self._nc = nc  # store for _publish_sync

            while not self._stop_event.is_set():
                await asyncio.sleep(0.5)
        finally:
            await nc.drain()

    async def _on_avg_delta_message(self, msg) -> None:
        """Callback for averaged delta published by the aggregator."""
        try:
            payload = json.loads(msg.data.decode("utf-8", errors="replace"))
            with self._lock:
                self._avg_delta_inbox = payload
            self._avg_delta_event.set()
            log.debug("[nats-fed] received averaged delta n_nodes=%s",
                      payload.get("_n_nodes", "?"))
        except Exception as exc:
            log.debug("[nats-fed] avg_delta parse error: %s", exc)

    def _publish_sync(self, subject: str, payload: Dict[str, Any]) -> None:
        """Thread-safe publish: schedule a coroutine on the NATS asyncio loop."""
        import asyncio
        if self._loop is None or self._loop.is_closed():
            log.debug("[nats-fed] loop not ready, dropping publish to %s", subject)
            return
        nc = getattr(self, "_nc", None)
        if nc is None:
            return

        async def _pub():
            try:
                body = json.dumps(payload).encode()
                await nc.publish(subject, body)
            except Exception as exc:
                log.debug("[nats-fed] publish error subject=%s: %s", subject, exc)

        asyncio.run_coroutine_threadsafe(_pub(), self._loop)

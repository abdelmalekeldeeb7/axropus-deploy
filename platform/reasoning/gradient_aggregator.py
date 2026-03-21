"""
Gradient Aggregator
===================
Subscribes to the NATS ``amf.federated.gradient`` subject, collects LoRA adapter
deltas from all active nodes, computes the Federated Averaging (FedAvg) result,
and publishes it back on ``amf.federated.gradient.avg``.

This service can run:
  a) Embedded inside the AMF Coordinator process.
  b) As a standalone Docker container (``python -m platform.reasoning.gradient_aggregator``).
  c) Directly from the FederatedLearningLoop on the "aggregator" node.

FedAvg algorithm
----------------
  avg_delta[param] = sum(delta_i[param] for i in 1..N) / N

All nodes are given equal weight (uniform averaging).  The ``_n_nodes`` metadata
field tells each node how many peers participated.

Cooperative reward relay
------------------------
The aggregator also subscribes to ``amf.federated.coop_reward`` and relays each
cooperative reward signal to the target node's personal subject
``amf.federated.coop_reward.<node_id>``.  Each FederatedLearningLoop subscribes
to its own subject and feeds rewards into its local RewardCalculator.

Usage
-----
    agg = GradientAggregator(
        nats_url="nats://localhost:4222",
        window_s=30.0,      # collect deltas for 30 s then publish average
        min_nodes=2,        # wait for at least 2 nodes before averaging
    )
    asyncio.run(agg.run())

    # Or from a thread:
    agg.start_sync()
    ...
    agg.stop()
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

_DEFAULT_NATS_URL        = os.environ.get("DYNAMO_NATS_URL", "nats://localhost:4222")
_GRADIENT_SUBJECT        = "amf.federated.gradient"
_GRADIENT_AVG_SUBJECT    = "amf.federated.gradient.avg"
_COOP_REWARD_SUBJECT     = "amf.federated.coop_reward"
_COOP_REWARD_NODE_PREFIX = "amf.federated.coop_reward."


# ── FedAvg helpers ────────────────────────────────────────────────────────────

def _fedavg(deltas: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute uniform FedAvg over a list of parameter dicts.

    Each dict maps parameter_name → list[float].  Parameters absent in some
    nodes are skipped (they stay at the local node's value).

    Returns the averaged dict with an extra ``_n_nodes`` metadata field.
    """
    if not deltas:
        return {"_n_nodes": 0}

    n = len(deltas)
    avg: Dict[str, Any] = {}

    # Collect all parameter names present in at least one delta
    all_names = set()
    for d in deltas:
        all_names.update(k for k in d if k != "_n_nodes")

    for name in all_names:
        # Only average if ALL nodes have this parameter (avoid partial averages)
        values = [d[name] for d in deltas if name in d]
        if len(values) != n:
            continue
        if not values or not isinstance(values[0], list):
            continue
        vec_len = len(values[0])
        # Check all vectors are same length
        if any(len(v) != vec_len for v in values):
            continue

        avg_vec = [sum(v[i] for v in values) / n for i in range(vec_len)]
        avg[name] = avg_vec

    avg["_n_nodes"] = n
    return avg


# ── GradientAggregator ────────────────────────────────────────────────────────

class GradientAggregator:
    """
    NATS-based gradient aggregator for FederatedLearningLoop.

    Parameters
    ----------
    nats_url : str
        NATS server URL.
    window_s : float
        Aggregation window in seconds.  Deltas arriving within the window are
        averaged together; the average is published at window close.
    min_nodes : int
        Minimum number of unique nodes required before publishing an average.
        If fewer nodes submitted in the window, the window is extended.
    max_window_s : float
        Maximum window duration even if min_nodes is not reached.
    """

    def __init__(
        self,
        nats_url: str = _DEFAULT_NATS_URL,
        window_s: float = 30.0,
        min_nodes: int = 2,
        max_window_s: float = 120.0,
    ) -> None:
        self._nats_url    = nats_url
        self._window_s    = float(window_s)
        self._min_nodes   = max(1, int(min_nodes))
        self._max_window  = float(max_window_s)

        # Current aggregation window
        self._lock          = threading.Lock()
        self._window_deltas: Dict[str, Dict[str, Any]] = {}   # node_id → delta
        self._window_start  = time.monotonic()

        # Stats
        self._cycles_published = 0
        self._total_nodes_seen = 0

        # Lifecycle
        self._stop_event  = threading.Event()
        self._bg_thread:   Optional[threading.Thread] = None
        self._loop:        Optional[asyncio.AbstractEventLoop] = None
        self._nc = None    # nats connection (set in async context)

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def start_sync(self) -> None:
        """Start the aggregator in a background thread."""
        self._bg_thread = threading.Thread(
            target=self._thread_main,
            name="amf-gradient-aggregator",
            daemon=True,
        )
        self._bg_thread.start()
        log.info("[aggregator] started  nats=%s window=%.0fs min_nodes=%d",
                 self._nats_url, self._window_s, self._min_nodes)

    def stop(self) -> None:
        self._stop_event.set()
        if self._bg_thread:
            self._bg_thread.join(timeout=5.0)

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "cycles_published": self._cycles_published,
                "total_nodes_seen": self._total_nodes_seen,
                "current_window_nodes": len(self._window_deltas),
            }

    # ── async main ────────────────────────────────────────────────────────────

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            loop.run_until_complete(self.run())
        except Exception as exc:
            log.error("[aggregator] event loop error: %s", exc)
        finally:
            loop.close()

    async def run(self) -> None:
        """Main async entry point — connect to NATS, subscribe, aggregate."""
        try:
            import nats  # type: ignore[import]
        except ImportError:
            log.error("[aggregator] nats-py not installed")
            return

        nc = await nats.connect(self._nats_url)
        self._nc = nc

        try:
            await nc.subscribe(_GRADIENT_SUBJECT, cb=self._on_gradient)
            await nc.subscribe(_COOP_REWARD_SUBJECT, cb=self._on_coop_reward)

            log.info(
                "[aggregator] subscribed to %s and %s",
                _GRADIENT_SUBJECT, _COOP_REWARD_SUBJECT,
            )

            # Aggregation loop
            while not self._stop_event.is_set():
                await asyncio.sleep(1.0)
                await self._maybe_publish()

        finally:
            await nc.drain()

    # ── NATS callbacks ────────────────────────────────────────────────────────

    async def _on_gradient(self, msg) -> None:
        """Receive a gradient delta from one node."""
        try:
            payload = json.loads(msg.data.decode("utf-8", errors="replace"))
        except Exception as exc:
            log.debug("[aggregator] gradient parse error: %s", exc)
            return

        node_id = str(payload.get("node_id", "unknown"))
        delta   = payload.get("delta", {})

        if not isinstance(delta, dict):
            return

        with self._lock:
            self._window_deltas[node_id] = delta
            self._total_nodes_seen += 1

        log.debug("[aggregator] received delta from node=%s params=%d",
                  node_id, len(delta))

    async def _on_coop_reward(self, msg) -> None:
        """Relay cooperative reward to the target node's personal subject."""
        try:
            payload = json.loads(msg.data.decode("utf-8", errors="replace"))
        except Exception:
            return

        to_node = str(payload.get("to_node", ""))
        if not to_node:
            return

        target_subject = f"{_COOP_REWARD_NODE_PREFIX}{to_node}"
        try:
            body = msg.data  # relay as-is
            await self._nc.publish(target_subject, body)
            log.debug("[aggregator] relayed coop_reward to %s", target_subject)
        except Exception as exc:
            log.debug("[aggregator] relay error: %s", exc)

    # ── aggregation window ────────────────────────────────────────────────────

    async def _maybe_publish(self) -> None:
        """Publish averaged delta if window conditions are met."""
        now = time.monotonic()
        with self._lock:
            elapsed     = now - self._window_start
            n_nodes     = len(self._window_deltas)
            window_copy = dict(self._window_deltas)

        window_full    = elapsed >= self._window_s and n_nodes >= self._min_nodes
        window_timeout = elapsed >= self._max_window

        if not (window_full or window_timeout) or n_nodes == 0:
            return

        # Compute FedAvg
        deltas = list(window_copy.values())
        avg    = _fedavg(deltas)

        # Publish
        try:
            body = json.dumps(avg).encode()
            await self._nc.publish(_GRADIENT_AVG_SUBJECT, body)
            with self._lock:
                self._cycles_published += 1
                self._window_deltas.clear()
                self._window_start = time.monotonic()
            log.info(
                "[aggregator] published avg cycle=%d nodes=%d params=%d",
                self._cycles_published, n_nodes, len(avg) - 1,
            )
        except Exception as exc:
            log.warning("[aggregator] publish error: %s", exc)


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    """CLI entry point: ``python -m platform.reasoning.gradient_aggregator``."""
    import argparse

    parser = argparse.ArgumentParser(description="AMF Gradient Aggregator")
    parser.add_argument("--nats-url",   default=_DEFAULT_NATS_URL)
    parser.add_argument("--window-s",   type=float, default=30.0)
    parser.add_argument("--min-nodes",  type=int,   default=2)
    parser.add_argument("--max-window", type=float, default=120.0)
    parser.add_argument("--log-level",  default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    agg = GradientAggregator(
        nats_url   = args.nats_url,
        window_s   = args.window_s,
        min_nodes  = args.min_nodes,
        max_window_s = args.max_window,
    )
    asyncio.run(agg.run())


if __name__ == "__main__":
    main()

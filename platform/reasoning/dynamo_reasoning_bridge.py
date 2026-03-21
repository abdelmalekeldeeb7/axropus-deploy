"""
Dynamo Reasoning Bridge
========================
Connects the AMF AdaptiveEngine / ReasoningModel to the NVIDIA Dynamo router,
translating UnifiedDecisionVector fleet decisions into Dynamo API calls and
feeding Dynamo worker metrics back into MetricsCollector.

Architecture
------------
                  ┌──────────────────────────────────┐
                  │      DynamoReasoningBridge        │
                  │                                  │
  tensor ─────────►  AdaptiveEngine.decide()         │
                  │        │                         │
                  │        ▼                         │
                  │  UnifiedDecisionVector            │
                  │   ├─ prefer_local_restore         │
                  │   ├─ cross_node_restore_target    │
                  │   ├─ persist_priority             │
                  │   ├─ replicate_to_nodes           │
                  │   └─ evict_hint_nodes             │
                  │        │                         │
                  │        ▼                         │
                  │  _translate_to_dynamo()           │──► Dynamo Router HTTP
                  │                                  │
                  │  _pre_warm_queue thread           │──► AMF Coordinator
                  │                                  │
                  │  /reasoning/status HTTP           │◄── operator / Grafana
                  └──────────────────────────────────┘

Key responsibilities
--------------------
1. Call AdaptiveEngine.decide() on every incoming request.
2. Translate fleet fields from UnifiedDecisionVector into Dynamo API calls
   (router scoring hints, cross-node restore triggers, replication requests).
3. Maintain a bounded pre-warm prediction queue: predicted prefix hashes are
   sent to the AMF coordinator before the request that needs them arrives.
4. Expose a /reasoning/status HTTP endpoint for operator dashboards.
5. Feed Dynamo router worker loads back to MetricsCollector via
   DynamoMetricsSource.record_routing_decision().

Usage
-----
    bridge = DynamoReasoningBridge(
        engine=AdaptiveEngine(ai_model=ReasoningModel()),
        dynamo_router_url="http://dynamo-router:8000",
        amf_coordinator_url="http://amf-coordinator:8500",
        metrics_source=dynamo_source,   # DynamoMetricsSource instance
        status_port=9100,
    )
    bridge.start()

    # On each request:
    result = bridge.route_request(
        tensor=collector.get_metrics_tensor(),
        tenant_id="t-1",
        prefix_hash="abc123",
        num_tokens=131072,
    )
    # result.worker_id is the selected Dynamo worker
    # result.amf_decision is the UnifiedDecisionVector

    bridge.stop()
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional

import numpy as np

from .adaptive_engine import AdaptiveEngine
from .metrics_collector import DynamoMetricsSource, Slot

log = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────
_PRE_WARM_QUEUE_MAX   = 256     # bounded pre-warm queue depth
_PRE_WARM_WORKER_CNT  = 2       # threads draining the pre-warm queue
_ROUTER_TIMEOUT_S     = 0.10    # Dynamo router HTTP call timeout (100 ms)
_COORD_TIMEOUT_S      = 0.08    # AMF coordinator HTTP call timeout (80 ms)
_STATUS_REFRESH_S     = 5.0     # background status snapshot interval


# ── data types ────────────────────────────────────────────────────────────────

@dataclass
class RouteResult:
    """Output of DynamoReasoningBridge.route_request()."""
    worker_id: str                       # selected Dynamo worker node
    amf_decision: Any                    # UnifiedDecisionVector
    kv_overlap: float                    # reported KV overlap for the chosen worker
    used_cross_node_restore: bool        # True if cross-node restore was triggered
    pre_warm_queued: int                 # number of predictions sent to pre-warm queue
    routing_ms: float                    # wall-clock time for the routing decision


@dataclass
class _PreWarmJob:
    prefix_hash: str
    tenant_id: str
    priority: float


# ── bridge ────────────────────────────────────────────────────────────────────

class DynamoReasoningBridge:
    """
    Connects AdaptiveEngine decisions to the Dynamo router.

    Parameters
    ----------
    engine : AdaptiveEngine
        The local decision engine (may have ai_model attached).
    dynamo_router_url : str
        Base URL of the Dynamo router (e.g. "http://dynamo-router:8000").
    amf_coordinator_url : str
        Base URL of the AMF coordinator (e.g. "http://amf-coordinator:8500").
    metrics_source : DynamoMetricsSource | None
        If provided, routing decisions are fed back for fleet-level metrics.
    status_port : int | None
        If set, start /reasoning/status HTTP server on this port.
    """

    def __init__(
        self,
        engine: AdaptiveEngine,
        dynamo_router_url: str = "",
        amf_coordinator_url: str = "",
        metrics_source: Optional[DynamoMetricsSource] = None,
        status_port: Optional[int] = None,
    ) -> None:
        self._engine          = engine
        self._router_url      = dynamo_router_url.rstrip("/")
        self._coordinator_url = amf_coordinator_url.rstrip("/")
        self._metrics_source  = metrics_source
        self._status_port     = status_port

        # ── pre-warm queue ────────────────────────────────────────────────
        self._prewarm_q: queue.Queue = queue.Queue(maxsize=_PRE_WARM_QUEUE_MAX)
        self._prewarm_workers: List[threading.Thread] = []

        # ── stats ─────────────────────────────────────────────────────────
        self._lock = threading.Lock()
        self._stats: Dict[str, Any] = {
            "total_requests":      0,
            "ai_routed":           0,
            "cross_node_restores": 0,
            "replications_sent":   0,
            "evict_hints_sent":    0,
            "pre_warm_dispatched": 0,
            "pre_warm_dropped":    0,
            "router_errors":       0,
            "last_update":         time.time(),
        }

        # ── status snapshot (updated periodically) ────────────────────────
        self._status_snap: Dict[str, Any] = {}

        # ── lifecycle ─────────────────────────────────────────────────────
        self._stop_event = threading.Event()
        self._status_thread: Optional[threading.Thread] = None
        self._http_thread:   Optional[threading.Thread] = None

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start background threads."""
        for i in range(_PRE_WARM_WORKER_CNT):
            t = threading.Thread(
                target=self._prewarm_worker_loop,
                name=f"amf-prewarm-{i}",
                daemon=True,
            )
            t.start()
            self._prewarm_workers.append(t)

        self._status_thread = threading.Thread(
            target=self._status_refresh_loop,
            name="amf-bridge-status",
            daemon=True,
        )
        self._status_thread.start()

        if self._status_port:
            self._http_thread = threading.Thread(
                target=self._run_status_server,
                name="amf-bridge-http",
                daemon=True,
            )
            self._http_thread.start()
            log.info(
                "[bridge] /reasoning/status listening on port %d", self._status_port
            )

        log.info(
            "[bridge] started  router=%s  coordinator=%s",
            self._router_url or "(none)",
            self._coordinator_url or "(none)",
        )

    def stop(self) -> None:
        """Signal all background threads to stop."""
        self._stop_event.set()
        # Unblock pre-warm workers
        for _ in self._prewarm_workers:
            try:
                self._prewarm_q.put_nowait(None)
            except queue.Full:
                pass
        for t in self._prewarm_workers:
            t.join(timeout=2.0)
        if self._status_thread:
            self._status_thread.join(timeout=2.0)
        log.info("[bridge] stopped")

    # ── main API ──────────────────────────────────────────────────────────────

    def route_request(
        self,
        tensor: np.ndarray,
        tenant_id: str = "default",
        prefix_hash: str = "",
        num_tokens: int = 0,
    ) -> RouteResult:
        """
        Select a Dynamo worker for the incoming request using AMF fleet reasoning.

        Steps
        -----
        1. AdaptiveEngine.decide() → UnifiedDecisionVector
        2. Fetch Dynamo worker list (cached if <2 s old)
        3. Apply AMF scoring on top of Dynamo KV-overlap scores
        4. Dispatch fleet actions (cross-node restore, replication, evict hints)
        5. Enqueue pre-warm predictions
        6. Return RouteResult
        """
        t0 = time.perf_counter()

        # Step 1: get decision
        dv = self._engine.decide(tensor, tenant_id=tenant_id, prefix_hash=prefix_hash)

        # Step 2: query Dynamo router for worker candidates
        workers = self._fetch_dynamo_workers()

        # Step 3: select best worker
        worker_id, kv_overlap = self._select_worker(dv, workers, tensor)

        # Step 4: fleet actions
        used_cross_node = self._dispatch_fleet_actions(dv, tenant_id, prefix_hash)

        # Step 5: pre-warm queue
        pw_queued = self._enqueue_pre_warm(dv, tenant_id)

        routing_ms = (time.perf_counter() - t0) * 1000.0

        # Feed back to DynamoMetricsSource
        if self._metrics_source is not None:
            self._metrics_source.record_routing_decision(
                used_amf=not dv.is_fallback,
                kv_overlap=kv_overlap,
                used_restore=used_cross_node or dv.prefer_local_restore,
            )

        with self._lock:
            self._stats["total_requests"] += 1
            if not dv.is_fallback:
                self._stats["ai_routed"] += 1
            if used_cross_node:
                self._stats["cross_node_restores"] += 1
            self._stats["last_update"] = time.time()

        log.debug(
            "[bridge] routed tenant=%s worker=%s kv_overlap=%.2f "
            "cross_node=%s prewarm=%d ms=%.1f",
            tenant_id, worker_id, kv_overlap, used_cross_node, pw_queued, routing_ms,
        )

        return RouteResult(
            worker_id=worker_id,
            amf_decision=dv,
            kv_overlap=kv_overlap,
            used_cross_node_restore=used_cross_node,
            pre_warm_queued=pw_queued,
            routing_ms=routing_ms,
        )

    # ── Dynamo worker selection ────────────────────────────────────────────────

    def _fetch_dynamo_workers(self) -> List[Dict[str, Any]]:
        """GET /v1/workers from the Dynamo router."""
        if not self._router_url:
            return []
        try:
            req = urllib.request.Request(
                f"{self._router_url}/v1/workers", method="GET"
            )
            with urllib.request.urlopen(req, timeout=_ROUTER_TIMEOUT_S) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            data = json.loads(body)
            workers = data if isinstance(data, list) else data.get("workers", [])
            return workers if isinstance(workers, list) else []
        except Exception as exc:
            with self._lock:
                self._stats["router_errors"] += 1
            log.debug("[bridge] fetch_workers error: %s", exc)
            return []

    def _select_worker(
        self,
        dv: Any,
        workers: List[Dict[str, Any]],
        tensor: np.ndarray,
    ) -> tuple:
        """
        Select the best worker combining Dynamo KV overlap with AMF fleet signals.

        Scoring:
          score = kv_overlap * kv_w - load * load_w + amf_bonus

        amf_bonus:
          +0.2 if prefer_local_restore and worker is local
          +0.1 * persist_priority if worker has available AMF restore capacity

        Returns (worker_id, kv_overlap).
        """
        if not workers:
            return ("dynamo-default", 0.0)

        prefer_local  = getattr(dv, "prefer_local_restore", True)
        cross_target  = getattr(dv, "cross_node_restore_target", None)
        persist_pri   = float(getattr(dv, "persist_priority", 0.5))

        # Read weights from tensor (Dynamo fleet slots)
        amf_load_w     = 0.3
        amf_kv_w       = 0.5

        best_id    = ""
        best_score = float("-inf")
        best_kv    = 0.0

        for w in workers:
            wid     = str(w.get("node_id", w.get("id", "unknown")))
            kv_ov   = float(w.get("kv_overlap", w.get("kv_cache_hit_rate", 0.0)))
            load    = float(w.get("load", w.get("utilization", 0.0)))

            score = kv_ov * amf_kv_w - load * amf_load_w

            # AMF bonus: prefer local restore
            if prefer_local and cross_target is None:
                score += 0.2  # same weight for all since we don't know local node here

            # Boost the explicitly requested cross-node target
            if cross_target and wid == cross_target:
                score += 0.3

            # Persist priority bonus: workers with more free capacity handle persist
            free_capacity = 1.0 - load
            score += persist_pri * free_capacity * 0.1

            if score > best_score:
                best_score = score
                best_id    = wid
                best_kv    = kv_ov

        return (best_id or "dynamo-default", best_kv)

    # ── fleet action dispatch ──────────────────────────────────────────────────

    def _dispatch_fleet_actions(
        self, dv: Any, tenant_id: str, prefix_hash: str
    ) -> bool:
        """
        Dispatch cross-node restore, replication, and eviction hints.
        Returns True if a cross-node restore was triggered.
        """
        used_cross_node = False

        cross_target = getattr(dv, "cross_node_restore_target", None)
        if cross_target and not getattr(dv, "prefer_local_restore", True):
            ok = self._post_coordinator(
                "/fleet/restore",
                {
                    "node_id":     cross_target,
                    "prefix_hash": prefix_hash,
                    "tenant_id":   tenant_id,
                },
            )
            if ok:
                used_cross_node = True
                with self._lock:
                    self._stats["cross_node_restores"] += 1

        replicate_to = getattr(dv, "replicate_to_nodes", [])
        for node_id in replicate_to:
            ok = self._post_coordinator(
                "/fleet/replicate",
                {
                    "node_id":     node_id,
                    "prefix_hash": prefix_hash,
                    "tenant_id":   tenant_id,
                    "priority":    float(getattr(dv, "persist_priority", 0.5)),
                },
            )
            if ok:
                with self._lock:
                    self._stats["replications_sent"] += 1

        evict_hints = getattr(dv, "evict_hint_nodes", [])
        for node_id in evict_hints:
            ok = self._post_coordinator(
                "/fleet/evict_hint",
                {
                    "node_id":   node_id,
                    "tenant_id": tenant_id,
                },
            )
            if ok:
                with self._lock:
                    self._stats["evict_hints_sent"] += 1

        return used_cross_node

    def _post_coordinator(self, path: str, payload: Dict[str, Any]) -> bool:
        """POST to the AMF coordinator. Returns True on 2xx response."""
        if not self._coordinator_url:
            return False
        try:
            body = json.dumps(payload).encode()
            req = urllib.request.Request(
                f"{self._coordinator_url}{path}",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=_COORD_TIMEOUT_S) as resp:
                return 200 <= resp.status < 300
        except Exception as exc:
            log.debug("[bridge] coordinator %s error: %s", path, exc)
            return False

    # ── pre-warm queue ─────────────────────────────────────────────────────────

    def _enqueue_pre_warm(self, dv: Any, tenant_id: str) -> int:
        """Enqueue pre-warm predictions. Returns number successfully enqueued."""
        predictions = getattr(dv, "pre_warm_predictions", [])
        priority    = float(getattr(dv, "persist_priority", 0.5))
        queued = 0
        for ph in predictions:
            job = _PreWarmJob(prefix_hash=ph, tenant_id=tenant_id, priority=priority)
            try:
                self._prewarm_q.put_nowait(job)
                queued += 1
            except queue.Full:
                with self._lock:
                    self._stats["pre_warm_dropped"] += 1
                break
        with self._lock:
            self._stats["pre_warm_dispatched"] += queued
        return queued

    def _prewarm_worker_loop(self) -> None:
        """Drain the pre-warm queue, calling AMF coordinator /lookup."""
        while not self._stop_event.is_set():
            try:
                job = self._prewarm_q.get(timeout=1.0)
            except queue.Empty:
                continue
            if job is None:
                break
            try:
                self._post_coordinator(
                    "/lookup",
                    {
                        "prefix_hash": job.prefix_hash,
                        "tenant_id":   job.tenant_id,
                    },
                )
            except Exception as exc:
                log.debug("[bridge] pre_warm error: %s", exc)
            finally:
                self._prewarm_q.task_done()

    # ── status endpoint ────────────────────────────────────────────────────────

    def _status_refresh_loop(self) -> None:
        while not self._stop_event.is_set():
            with self._lock:
                snap = dict(self._stats)
            # Augment with engine stats if available
            try:
                eng_stats = self._engine.get_stats()  # type: ignore[attr-defined]
                snap["engine"] = eng_stats
            except Exception:
                pass
            self._status_snap = snap
            self._stop_event.wait(timeout=_STATUS_REFRESH_S)

    def get_status(self) -> Dict[str, Any]:
        """Return the current status snapshot (safe for any thread)."""
        return dict(self._status_snap) if self._status_snap else dict(self._stats)

    def _run_status_server(self) -> None:
        """Run a minimal HTTP server serving /reasoning/status as JSON."""
        bridge = self

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/reasoning/status":
                    body = json.dumps(bridge.get_status(), indent=2).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, fmt, *args):
                pass  # suppress default HTTP logging

        try:
            server = HTTPServer(("0.0.0.0", self._status_port), _Handler)
            server.timeout = 1.0
            while not self._stop_event.is_set():
                server.handle_request()
        except Exception as exc:
            log.warning("[bridge] status server error: %s", exc)

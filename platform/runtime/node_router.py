from __future__ import annotations

import hashlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse

from ..cluster.node_registry import NodeRegistry
from ..cluster.snapshot_index import SnapshotIndex
from ..observability.metrics import GLOBAL_METRICS
from ..observability.platform_logging import emit_log
from ..runtime.registry import WorkerRegistry
from ..queue.base import QueueBase


class NodeRouter:
    def __init__(
        self,
        node_id: str,
        host: str,
        port: int,
        queue: QueueBase,
        worker_registry: WorkerRegistry,
        node_registry: NodeRegistry,
        snapshot_index: SnapshotIndex,
    ) -> None:
        self.node_id = node_id
        self.host = host
        self.port = int(port)
        self.queue = queue
        self.worker_registry = worker_registry
        self.node_registry = node_registry
        self.snapshot_index = snapshot_index
        self._stop = threading.Event()
        self._hb_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)

    def start(self) -> None:
        self._hb_thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._hb_thread.join(timeout=2)

    def _local_workers(self) -> list[Dict[str, Any]]:
        rows = self.worker_registry.list_workers()
        out = []
        for row in rows:
            caps = row.get("capabilities", {})
            if caps.get("node_id") == self.node_id:
                out.append(row)
        return out

    def _heartbeat_loop(self) -> None:
        while not self._stop.is_set():
            workers = self._local_workers()
            inflight = sum(int(w.get("inflight", 0) or 0) for w in workers)
            stats = self.queue.stats()
            queue_depth_hit = int(stats.get("queue_depth_hit", 0) or 0) if isinstance(stats, dict) else 0
            queue_depth_miss = int(stats.get("queue_depth_miss", 0) or 0) if isinstance(stats, dict) else 0
            self.node_registry.register_or_heartbeat(
                node_id=self.node_id,
                host=self.host,
                router_port=self.port,
                gpu_count=max(1, len(workers)),
                inflight=inflight,
                queue_depth_hit=queue_depth_hit,
                queue_depth_miss=queue_depth_miss,
                capabilities={"workers": [w.get("worker_id", "") for w in workers]},
            )
            emit_log(
                "node_router",
                "NODE_HEARTBEAT",
                {
                    "job_id": "",
                    "run_id": "",
                    "worker_id": "",
                    "session_id": "",
                    "org_id": "",
                    "request_id": "",
                    "latency_ms": 0.0,
                    "node_id": self.node_id,
                    "inflight": inflight,
                },
            )
            GLOBAL_METRICS.set_gauge("node_inflight", float(inflight), labels={"node_id": self.node_id})
            time.sleep(2.0)

    def status(self) -> Dict[str, Any]:
        workers = self._local_workers()
        stats = self.queue.stats()
        return {
            "node_id": self.node_id,
            "host": self.host,
            "router_port": self.port,
            "workers": workers,
            "queue": stats,
            "inflight": sum(int(w.get("inflight", 0) or 0) for w in workers),
        }

    def fetch_snapshot(self, fingerprint_hash: str, org_id: str) -> Optional[Dict[str, Any]]:
        locations = self.snapshot_index.get_locations(fingerprint_hash=fingerprint_hash, org_id=org_id)
        for row in locations:
            if row.get("node_id") != self.node_id:
                continue
            path = Path(str(row.get("snapshot_path", "")))
            if not path.exists():
                continue
            blob = path.read_bytes()
            checksum = hashlib.sha256(blob).hexdigest()
            row = dict(row)
            row["checksum_sha256"] = checksum
            return {"bytes": blob, "meta": row}
        return None


class NodeRouterHandler(BaseHTTPRequestHandler):
    node_router: Optional[NodeRouter] = None

    def _send_json(self, code: int, payload: Dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        nr = NodeRouterHandler.node_router
        if nr is None:
            self._send_json(503, {"error": "node router unavailable"})
            return
        parsed = urlparse(self.path)
        if parsed.path in ("/health", "/v1/node/health"):
            self._send_json(200, {"status": "ok", "node_id": nr.node_id})
            return
        if parsed.path in ("/ready", "/v1/node/ready"):
            workers = nr.status().get("workers", [])
            self._send_json(200 if workers else 503, {"ready": bool(workers), "node_id": nr.node_id})
            return
        if parsed.path in ("/v1/node/status", "/v1/node/runtime_status"):
            self._send_json(200, nr.status())
            return
        if parsed.path == "/v1/snapshots/fetch":
            q = parse_qs(parsed.query)
            fingerprint = (q.get("fingerprint") or [""])[0]
            org_id = (q.get("org_id") or ["default"])[0]
            if not fingerprint:
                self._send_json(400, {"error": "fingerprint required"})
                return
            fetched = nr.fetch_snapshot(fingerprint_hash=fingerprint, org_id=org_id)
            if not fetched:
                self._send_json(404, {"error": "snapshot_not_found"})
                return
            blob = fetched["bytes"]
            meta = fetched["meta"]
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(blob)))
            self.send_header("X-Korith-Snapshot-Meta", json.dumps(meta, ensure_ascii=False))
            self.end_headers()
            self.wfile.write(blob)
            return
        self._send_json(404, {"error": "not found"})


def serve_node_router(node_router: NodeRouter, host: str, port: int) -> None:
    NodeRouterHandler.node_router = node_router
    node_router.start()
    httpd = ThreadingHTTPServer((host, port), NodeRouterHandler)
    httpd.serve_forever()

from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .cluster_worker import ClusterWorker
from .config import (
    apply_config_to_env,
    build_artifacts,
    build_ledger,
    build_node_registry,
    build_queue,
    build_registry,
    build_restore_store,
    build_snapshot_index,
    load_platform_config,
)
from ..adapters.registry import AdapterRegistry
from ..observability.metrics import GLOBAL_METRICS


class WorkerHandler(BaseHTTPRequestHandler):
    worker: ClusterWorker | None = None

    def _send(self, code: int, payload: dict) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send(200, {"status": "ok"})
            return
        if self.path == "/ready":
            worker = WorkerHandler.worker
            ready = bool(worker and worker._thread.is_alive())
            self._send(200 if ready else 503, {"ready": ready})
            return
        if self.path == "/runtime_status":
            worker = WorkerHandler.worker
            self._send(200, {"worker_id": worker.worker_id if worker else "", "gpu_id": worker.gpu_id if worker else -1, "node_id": worker.node_id if worker else ""})
            return
        if self.path == "/gpu_status":
            worker = WorkerHandler.worker
            self._send(200, {"gpu_id": worker.gpu_id if worker else -1, "inflight": getattr(worker, "_inflight", 0)})
            return
        if self.path == "/metrics":
            data = GLOBAL_METRICS.render_prometheus().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self._send(404, {"error": "not found"})


def _check_gpu_available(gpu_id: int) -> bool:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            return False
        values = {int(x.strip()) for x in proc.stdout.splitlines() if x.strip().isdigit()}
        return gpu_id in values
    except Exception:
        return False


def run_worker(worker_id: str, gpu_id: int, host: str, port: int, config_path: str | None = None, node_id: str = "") -> None:
    apply_config_to_env(load_platform_config(config_path))
    if os.environ.get("KORITH_REQUIRE_GPU", "0") == "1" and not _check_gpu_available(gpu_id):
        raise RuntimeError(f"GPU {gpu_id} unavailable")
    ledger = build_ledger()
    ledger.init()
    artifacts = build_artifacts()
    queue = build_queue()
    registry = build_registry()
    node_registry = build_node_registry()
    restore_store = build_restore_store()
    snapshot_index = build_snapshot_index(ledger)
    adapters = AdapterRegistry()

    resolved_node_id = node_id or os.environ.get("KORITH_NODE_ID", "")

    worker = ClusterWorker(
        worker_id=worker_id,
        gpu_id=gpu_id,
        ledger=ledger,
        artifacts=artifacts,
        queue_backend=queue,
        registry=registry,
        restore_store=restore_store,
        adapter_registry=adapters,
        snapshot_index=snapshot_index,
        node_registry=node_registry,
        node_id=resolved_node_id,
        host=host,
    )
    worker.start()

    WorkerHandler.worker = worker
    httpd = ThreadingHTTPServer((host, port), WorkerHandler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    print(f"[KORITH_WORKER] {worker_id} serving on {host}:{port}")

    try:
        t.join()
    except KeyboardInterrupt:
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", default=os.environ.get("KORITH_WORKER_ID", "worker-0"))
    parser.add_argument("--gpu-id", type=int, default=int(os.environ.get("KORITH_GPU_ID", "0")))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--node-id", default=os.environ.get("KORITH_NODE_ID", ""))
    parser.add_argument("--config", default=os.environ.get("KORITH_PLATFORM_CONFIG", ""))
    args = parser.parse_args()
    run_worker(args.worker_id, args.gpu_id, args.host, args.port, args.config or None, node_id=args.node_id)


if __name__ == "__main__":
    main()

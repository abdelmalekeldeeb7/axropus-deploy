from __future__ import annotations

import argparse
import os

from .cluster_router import ClusterRouter
from .amf_coordinator_client import AmfCoordinatorClient
from .node_router import NodeRouter, serve_node_router
from ..gateway.auth import require_api_key_salt_configured
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
from ..gateway.http_server import serve


def run_router(host: str, port: int, config_path: str | None = None) -> None:
    apply_config_to_env(load_platform_config(config_path))
    require_api_key_salt_configured()
    ledger = build_ledger()
    ledger.init()
    artifacts = build_artifacts()
    queue = build_queue()
    registry = build_registry()
    node_registry = build_node_registry()
    restore_store = build_restore_store()
    snapshot_index = build_snapshot_index(ledger)
    adapters = AdapterRegistry()

    router = ClusterRouter(
        ledger=ledger,
        artifacts=artifacts,
        queue=queue,
        registry=registry,
        restore_store=restore_store,
        adapter_registry=adapters,
        node_registry=node_registry,
        snapshot_index=snapshot_index,
        amf_coordinator_client=(
            AmfCoordinatorClient(
                os.environ.get("KORITH_AMF_COORDINATOR_URL", "").strip(),
                timeout_s=float(os.environ.get("KORITH_AMF_COORDINATOR_TIMEOUT_S", "0.25") or 0.25),
            )
            if str(os.environ.get("KORITH_AMF_DISTRIBUTED", "0")).strip().lower() in ("1", "true", "yes", "on")
            else None
        ),
        transfer_bandwidth_mbps=float(os.environ.get("KORITH_SNAPSHOT_BW_MBPS", "200")),
        transfer_rtt_ms=float(os.environ.get("KORITH_SNAPSHOT_RTT_MS", "5")),
        transfer_threshold=float(os.environ.get("KORITH_TRANSFER_VS_RECOMPUTE_THRESHOLD", "0.8")),
    )
    print(f"[KORITH_ROUTER] serving on {host}:{port}")
    serve(router, host=host, port=port)


def run_node_router(host: str, port: int, config_path: str | None = None, node_id: str = "") -> None:
    apply_config_to_env(load_platform_config(config_path))
    ledger = build_ledger()
    ledger.init()
    queue = build_queue()
    worker_registry = build_registry()
    node_registry = build_node_registry()
    snapshot_index = build_snapshot_index(ledger)
    resolved_node_id = node_id or os.environ.get("KORITH_NODE_ID", "node-0")
    resolved_host = os.environ.get("KORITH_NODE_HOST", host)
    resolved_port = int(os.environ.get("KORITH_NODE_ROUTER_PORT", str(port)))
    node_router = NodeRouter(
        node_id=resolved_node_id,
        host=resolved_host,
        port=resolved_port,
        queue=queue,
        worker_registry=worker_registry,
        node_registry=node_registry,
        snapshot_index=snapshot_index,
    )
    print(f"[KORITH_NODE_ROUTER] {resolved_node_id} serving on {host}:{port}")
    serve_node_router(node_router, host=host, port=port)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--mode", choices=["cluster", "node"], default=os.environ.get("KORITH_ROUTER_MODE", "cluster"))
    parser.add_argument("--node-id", default=os.environ.get("KORITH_NODE_ID", ""))
    parser.add_argument("--config", default=os.environ.get("KORITH_PLATFORM_CONFIG", ""))
    args = parser.parse_args()
    if args.mode == "node":
        run_node_router(args.host, args.port, args.config or None, node_id=args.node_id)
    else:
        run_router(args.host, args.port, args.config or None)


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
import os
import uuid
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from platform.adapters.base import Capabilities
from platform.artifacts.adapter import LocalArtifactStoreAdapter
from platform.cluster.node_registry import NodeRegistry
from platform.cluster.snapshot_index import SnapshotIndex
from platform.ledger.store import SQLiteLedgerStore
from platform.queue.inproc_queue import InProcQueue
from platform.runtime.cluster_router import ClusterRouter, RouterRequestError
from platform.runtime.registry import WorkerRegistry
from platform.runtime.restore_store import RestoreStore


class _FakeAdapter:
    backend_version = "test"

    def get_fingerprint(self):
        return {"model_hash": "m", "tokenizer_hash": "t", "backend_version": "test"}

    def get_capabilities(self):
        return Capabilities(
            kv_replay=True,
            deterministic_seeding=True,
            verify_tokens=True,
            draft_supported=True,
        )

    def tokenize(self, prompt: str) -> int:
        return max(1, len(prompt.split()))


class _AdapterRegistry:
    def get_adapter(self, jobspec):
        return _FakeAdapter()

    def list_capabilities(self):
        return {"korith_local": _FakeAdapter().get_capabilities().as_dict()}


class _CoordinatorStub:
    def __init__(self, rows=None):
        self.enabled = True
        self._rows = rows or []

    def lookup(self, prefix_hash: str, tenant_id: str):
        return list(self._rows)


class _CoordinatorFailStub:
    enabled = True

    def lookup(self, prefix_hash: str, tenant_id: str):
        raise RuntimeError("coordinator unavailable")


class Phase5ClusterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.ledger = SQLiteLedgerStore(base / "ledger.sqlite")
        self.ledger.init()
        self.artifacts = LocalArtifactStoreAdapter(base / "artifacts")
        self.queue = InProcQueue()
        self.workers = WorkerRegistry(base / "workers.sqlite")
        self.restore = RestoreStore(base / "restore.sqlite")
        self.nodes = NodeRegistry(sqlite_path=base / "nodes.sqlite")
        self.snapshots = SnapshotIndex(self.ledger)
        self.router = ClusterRouter(
            ledger=self.ledger,
            artifacts=self.artifacts,
            queue=self.queue,
            registry=self.workers,
            restore_store=self.restore,
            adapter_registry=_AdapterRegistry(),
            node_registry=self.nodes,
            snapshot_index=self.snapshots,
            transfer_bandwidth_mbps=200.0,
            transfer_rtt_ms=5.0,
            transfer_threshold=0.8,
        )

    def tearDown(self):
        self.tmp.cleanup()

    def _register_nodes(self):
        self.nodes.register_or_heartbeat(
            node_id="node-a",
            host="127.0.0.1",
            router_port=8100,
            gpu_count=1,
            inflight=0,
            queue_depth_hit=0,
            queue_depth_miss=0,
            capabilities={},
        )
        self.nodes.register_or_heartbeat(
            node_id="node-b",
            host="127.0.0.1",
            router_port=8101,
            gpu_count=1,
            inflight=1,
            queue_depth_hit=0,
            queue_depth_miss=1,
            capabilities={},
        )

    def test_routing_preference_worker_affinity_then_locality(self):
        self._register_nodes()
        self.router._session_affinity["sess-1"] = {"node_id": "node-a", "worker_id": "worker-a"}
        route = self.router._select_route(
            jobspec={"session_id": "sess-1"},
            fingerprint_hash="fp1",
            org_id="org1",
            predicted_lane="HIT",
        )
        self.assertEqual(route["reason"], "worker_affinity")
        self.assertEqual(route["chosen_node_id"], "node-a")
        self.assertEqual(route["chosen_worker_id"], "worker-a")

        self.router._session_affinity.clear()
        self.snapshots.upsert_location(
            fingerprint_hash="fp1",
            snapshot_id="snap1",
            org_id="org1",
            node_id="node-b",
            worker_id="worker-b",
            snapshot_path="/tmp/snap1.bin",
            size_bytes=1024,
        )
        route2 = self.router._select_route(
            jobspec={},
            fingerprint_hash="fp1",
            org_id="org1",
            predicted_lane="HIT",
        )
        self.assertEqual(route2["reason"], "node_locality")
        self.assertEqual(route2["chosen_node_id"], "node-b")

    def test_single_node_hit_lane_uses_fingerprint_affinity(self):
        self.workers.register("worker-a", "127.0.0.1", 0, capabilities={})
        self.workers.register("worker-b", "127.0.0.1", 1, capabilities={})
        self.workers.heartbeat("worker-a", inflight=0)
        self.workers.heartbeat("worker-b", inflight=3)

        first = self.router._select_route(
            jobspec={},
            fingerprint_hash="fp-affinity",
            org_id="org1",
            predicted_lane="HIT",
        )
        self.assertEqual(first["reason"], "least_loaded")
        self.assertEqual(first["chosen_worker_id"], "worker-a")

        # Force least-loaded to shift, then verify the hit lane still follows
        # prior fingerprint affinity to preserve replay locality.
        self.workers.heartbeat("worker-a", inflight=5)
        self.workers.heartbeat("worker-b", inflight=0)
        second = self.router._select_route(
            jobspec={},
            fingerprint_hash="fp-affinity",
            org_id="org1",
            predicted_lane="HIT",
        )
        self.assertEqual(second["reason"], "fingerprint_affinity")
        self.assertEqual(second["chosen_worker_id"], "worker-a")

    def test_single_node_miss_lane_uses_shape_affinity(self):
        self.workers.register("worker-a", "127.0.0.1", 0, capabilities={"lane_role": "miss"})
        self.workers.register("worker-b", "127.0.0.1", 1, capabilities={"lane_role": "miss"})
        self.workers.heartbeat("worker-a", inflight=0)
        self.workers.heartbeat("worker-b", inflight=3)

        first = self.router._select_route(
            jobspec={},
            fingerprint_hash="fp-shape",
            org_id="org1",
            predicted_lane="MISS",
            shape_key="shape-a",
        )
        self.assertEqual(first["reason"], "least_loaded")
        self.assertEqual(first["chosen_worker_id"], "worker-a")

        # Keep worker-a close to least-loaded so shape affinity should hold.
        self.workers.heartbeat("worker-a", inflight=1)
        self.workers.heartbeat("worker-b", inflight=0)
        second = self.router._select_route(
            jobspec={},
            fingerprint_hash="fp-shape",
            org_id="org1",
            predicted_lane="MISS",
            shape_key="shape-a",
        )
        self.assertEqual(second["reason"], "shape_affinity")
        self.assertEqual(second["chosen_worker_id"], "worker-a")

    def test_single_node_miss_lane_shape_affinity_yields_when_overloaded(self):
        old = os.environ.get("KORITH_ROUTER_SHAPE_AFFINITY_MAX_INFLIGHT_DELTA")
        try:
            os.environ["KORITH_ROUTER_SHAPE_AFFINITY_MAX_INFLIGHT_DELTA"] = "0"
            self.workers.register("worker-a", "127.0.0.1", 0, capabilities={"lane_role": "miss"})
            self.workers.register("worker-b", "127.0.0.1", 1, capabilities={"lane_role": "miss"})
            self.workers.heartbeat("worker-a", inflight=0)
            self.workers.heartbeat("worker-b", inflight=3)

            _ = self.router._select_route(
                jobspec={},
                fingerprint_hash="fp-shape-overload",
                org_id="org1",
                predicted_lane="MISS",
                shape_key="shape-overload",
            )

            self.workers.heartbeat("worker-a", inflight=2)
            self.workers.heartbeat("worker-b", inflight=0)
            second = self.router._select_route(
                jobspec={},
                fingerprint_hash="fp-shape-overload",
                org_id="org1",
                predicted_lane="MISS",
                shape_key="shape-overload",
            )
            self.assertEqual(second["reason"], "least_loaded")
            self.assertEqual(second["chosen_worker_id"], "worker-b")
        finally:
            if old is None:
                os.environ.pop("KORITH_ROUTER_SHAPE_AFFINITY_MAX_INFLIGHT_DELTA", None)
            else:
                os.environ["KORITH_ROUTER_SHAPE_AFFINITY_MAX_INFLIGHT_DELTA"] = old

    def test_single_node_lane_role_filters_worker_selection(self):
        self.workers.register("worker-hit", "127.0.0.1", 0, capabilities={"lane_role": "hit"})
        self.workers.register("worker-miss", "127.0.0.1", 1, capabilities={"lane_role": "miss"})
        self.workers.heartbeat("worker-hit", inflight=5)
        self.workers.heartbeat("worker-miss", inflight=0)

        hit_route = self.router._select_route(
            jobspec={},
            fingerprint_hash="fp-lane-role-hit",
            org_id="org1",
            predicted_lane="HIT",
        )
        miss_route = self.router._select_route(
            jobspec={},
            fingerprint_hash="fp-lane-role-miss",
            org_id="org1",
            predicted_lane="MISS",
        )
        self.assertEqual(hit_route["chosen_worker_id"], "worker-hit")
        self.assertEqual(miss_route["chosen_worker_id"], "worker-miss")

    def test_hit_lane_prefers_amf_ready_workers_during_warmup(self):
        self.workers.register("worker-cold", "127.0.0.1", 0, capabilities={"amf_ready": False})
        self.workers.register("worker-warm", "127.0.0.1", 1, capabilities={"amf_ready": True})
        self.workers.heartbeat("worker-cold", inflight=0, capabilities={"amf_ready": False})
        self.workers.heartbeat("worker-warm", inflight=5, capabilities={"amf_ready": True})

        hit_route = self.router._select_route(
            jobspec={},
            fingerprint_hash="fp-amf-warm",
            org_id="org1",
            predicted_lane="HIT",
        )
        miss_route = self.router._select_route(
            jobspec={},
            fingerprint_hash="fp-amf-cold",
            org_id="org1",
            predicted_lane="MISS",
        )
        self.assertEqual(hit_route["chosen_worker_id"], "worker-warm")
        self.assertTrue(bool(hit_route.get("amf_ready_preferred", False)))
        self.assertEqual(miss_route["chosen_worker_id"], "worker-cold")

    def test_distributed_coordinator_lookup_prefers_cached_node(self):
        self._register_nodes()
        self.workers.register("worker-a", "127.0.0.1", 0, capabilities={}, node_id="node-a")
        self.workers.register("worker-b", "127.0.0.1", 1, capabilities={}, node_id="node-b")
        self.workers.heartbeat("worker-a", inflight=3, capabilities={})
        self.workers.heartbeat("worker-b", inflight=0, capabilities={})
        self.router.amf_coordinator_client = _CoordinatorStub(
            rows=[{"node_id": "node-a", "worker_id": "worker-a"}]
        )
        route = self.router._select_route(
            jobspec={},
            fingerprint_hash="fp-coord",
            org_id="org1",
            tenant_id="tenant-a",
            amf_lookup_hash="abc123",
            predicted_lane="HIT",
        )
        self.assertEqual(route["reason"], "amf_coordinator_cache")
        self.assertEqual(route["chosen_node_id"], "node-a")
        self.assertEqual(route["chosen_worker_id"], "worker-a")

    def test_distributed_coordinator_fail_open_falls_back_to_default_routing(self):
        self.workers.register("worker-a", "127.0.0.1", 0, capabilities={})
        self.workers.register("worker-b", "127.0.0.1", 1, capabilities={})
        self.workers.heartbeat("worker-a", inflight=3, capabilities={})
        self.workers.heartbeat("worker-b", inflight=0, capabilities={})
        self.router.amf_coordinator_client = _CoordinatorFailStub()
        route = self.router._select_route(
            jobspec={},
            fingerprint_hash="fp-coord-fail",
            org_id="org1",
            tenant_id="tenant-a",
            amf_lookup_hash="abc123",
            predicted_lane="HIT",
        )
        self.assertEqual(route["reason"], "least_loaded")
        self.assertEqual(route["chosen_worker_id"], "worker-b")

    def test_single_node_strict_lane_filter_raises_when_no_compatible_worker(self):
        old = os.environ.get("KORITH_ROUTER_STRICT_WORKER_LANES")
        try:
            os.environ["KORITH_ROUTER_STRICT_WORKER_LANES"] = "1"
            self.workers.register("worker-miss", "127.0.0.1", 0, capabilities={"lane_role": "miss"})
            with self.assertRaises(RuntimeError):
                self.router._select_route(
                    jobspec={},
                    fingerprint_hash="fp-no-hit-worker",
                    org_id="org1",
                    predicted_lane="HIT",
                )
        finally:
            if old is None:
                os.environ.pop("KORITH_ROUTER_STRICT_WORKER_LANES", None)
            else:
                os.environ["KORITH_ROUTER_STRICT_WORKER_LANES"] = old

    def test_fingerprint_affinity_ttl_can_disable_stale_affinity(self):
        old_ttl = os.environ.get("KORITH_ROUTER_AFFINITY_TTL_S")
        try:
            self.workers.register("worker-a", "127.0.0.1", 0, capabilities={})
            self.workers.register("worker-b", "127.0.0.1", 1, capabilities={})
            self.workers.heartbeat("worker-a", inflight=0)
            self.workers.heartbeat("worker-b", inflight=3)

            first = self.router._select_route(
                jobspec={},
                fingerprint_hash="fp-expire",
                org_id="org1",
                predicted_lane="HIT",
            )
            self.assertEqual(first["chosen_worker_id"], "worker-a")

            self.workers.heartbeat("worker-a", inflight=5)
            self.workers.heartbeat("worker-b", inflight=0)
            os.environ["KORITH_ROUTER_AFFINITY_TTL_S"] = "0"

            second = self.router._select_route(
                jobspec={},
                fingerprint_hash="fp-expire",
                org_id="org1",
                predicted_lane="HIT",
            )
            self.assertEqual(second["reason"], "least_loaded")
            self.assertEqual(second["chosen_worker_id"], "worker-b")
        finally:
            if old_ttl is None:
                os.environ.pop("KORITH_ROUTER_AFFINITY_TTL_S", None)
            else:
                os.environ["KORITH_ROUTER_AFFINITY_TTL_S"] = old_ttl

    def test_node_locality_prefers_live_worker_when_snapshot_worker_is_stale(self):
        self._register_nodes()
        self.workers.register("worker-live", "127.0.0.1", 0, capabilities={}, node_id="node-b")
        self.snapshots.upsert_location(
            fingerprint_hash="fp-live",
            snapshot_id="snap-live",
            org_id="org1",
            node_id="node-b",
            worker_id="worker-stale",
            snapshot_path="/tmp/snap-live.bin",
            size_bytes=1024,
        )
        route = self.router._select_route(
            jobspec={},
            fingerprint_hash="fp-live",
            org_id="org1",
            predicted_lane="HIT",
        )
        self.assertEqual(route["reason"], "node_locality")
        self.assertEqual(route["chosen_node_id"], "node-b")
        self.assertEqual(route["chosen_worker_id"], "worker-live")

    def test_transfer_vs_recompute_decision(self):
        self._register_nodes()
        # Force worker-affinity to node-a while snapshot exists on node-b.
        self.router._session_affinity["sess-x"] = {"node_id": "node-a", "worker_id": "worker-a"}
        self.snapshots.upsert_location(
            fingerprint_hash="fp2",
            snapshot_id="snap2",
            org_id="org1",
            node_id="node-b",
            worker_id="worker-b",
            snapshot_path="/tmp/snap2.bin",
            size_bytes=1024 * 1024,
        )
        self.router._estimate_baseline_prefix_ms = lambda fingerprint_hash, org_id: 50.0  # type: ignore
        route = self.router._select_route(
            jobspec={"session_id": "sess-x"},
            fingerprint_hash="fp2",
            org_id="org1",
            predicted_lane="HIT",
        )
        self.assertEqual(route["reason"], "worker_affinity")
        self.assertTrue(route.get("transfer_requested"))
        self.assertEqual(route["final_lane"], "HIT")

        # Very large snapshot should force recompute.
        self.snapshots.upsert_location(
            fingerprint_hash="fp2",
            snapshot_id="snap2",
            org_id="org1",
            node_id="node-b",
            worker_id="worker-b",
            snapshot_path="/tmp/snap2.bin",
            size_bytes=300 * 1024 * 1024,
        )
        route2 = self.router._select_route(
            jobspec={"session_id": "sess-x"},
            fingerprint_hash="fp2",
            org_id="org1",
            predicted_lane="HIT",
        )
        self.assertEqual(route2["reason"], "transfer_too_expensive")
        self.assertTrue(route2.get("force_miss"))
        self.assertEqual(route2["final_lane"], "MISS")

    def test_hit_lane_can_prefer_replay_local_over_session_affinity(self):
        old = os.environ.get("KORITH_ROUTER_PREFER_REPLAY_LOCAL_FOR_HIT")
        try:
            os.environ["KORITH_ROUTER_PREFER_REPLAY_LOCAL_FOR_HIT"] = "1"
            self._register_nodes()
            self.router._session_affinity["sess-local"] = {"node_id": "node-a", "worker_id": "worker-a"}
            self.snapshots.upsert_location(
                fingerprint_hash="fp-local",
                snapshot_id="snap-local",
                org_id="org1",
                node_id="node-b",
                worker_id="worker-b",
                snapshot_path="/tmp/snap-local.bin",
                size_bytes=1024,
            )
            route = self.router._select_route(
                jobspec={"session_id": "sess-local"},
                fingerprint_hash="fp-local",
                org_id="org1",
                predicted_lane="HIT",
            )
            self.assertEqual(route["reason"], "node_locality")
            self.assertEqual(route["chosen_node_id"], "node-b")
            self.assertEqual(route["final_lane"], "HIT")
            self.assertFalse(bool(route.get("transfer_requested", False)))
        finally:
            if old is None:
                os.environ.pop("KORITH_ROUTER_PREFER_REPLAY_LOCAL_FOR_HIT", None)
            else:
                os.environ["KORITH_ROUTER_PREFER_REPLAY_LOCAL_FOR_HIT"] = old

    def test_transfer_candidate_uses_tier_adjusted_cost(self):
        keys = (
            "KORITH_ROUTER_PREFER_REPLAY_LOCAL_FOR_HIT",
            "KORITH_ROUTER_TRANSFER_TIER_FACTOR_VRAM",
            "KORITH_ROUTER_TRANSFER_TIER_FACTOR_NVME",
        )
        old = {k: os.environ.get(k) for k in keys}
        try:
            os.environ["KORITH_ROUTER_PREFER_REPLAY_LOCAL_FOR_HIT"] = "0"
            os.environ["KORITH_ROUTER_TRANSFER_TIER_FACTOR_VRAM"] = "1.0"
            os.environ["KORITH_ROUTER_TRANSFER_TIER_FACTOR_NVME"] = "5.0"
            self._register_nodes()
            self.nodes.register_or_heartbeat(
                node_id="node-c",
                host="127.0.0.1",
                router_port=8102,
                gpu_count=1,
                inflight=1,
                queue_depth_hit=0,
                queue_depth_miss=1,
                capabilities={},
            )
            self.router._session_affinity["sess-tier"] = {"node_id": "node-a", "worker_id": "worker-a"}
            self.snapshots.upsert_location(
                fingerprint_hash="fp-tier-choice",
                snapshot_id="snap-cold",
                org_id="org1",
                node_id="node-b",
                worker_id="worker-b",
                snapshot_path="/tmp/korith_router/nvme/snap.bin",
                size_bytes=1 * 1024 * 1024,
            )
            self.snapshots.upsert_location(
                fingerprint_hash="fp-tier-choice",
                snapshot_id="snap-hot",
                org_id="org1",
                node_id="node-c",
                worker_id="worker-c",
                snapshot_path="/tmp/korith_router/vram/snap.bin",
                size_bytes=2 * 1024 * 1024,
            )
            self.router._estimate_baseline_prefix_ms = lambda fingerprint_hash, org_id: 500.0  # type: ignore
            route = self.router._select_route(
                jobspec={"session_id": "sess-tier"},
                fingerprint_hash="fp-tier-choice",
                org_id="org1",
                predicted_lane="HIT",
            )
            self.assertTrue(bool(route.get("transfer_requested", False)))
            self.assertEqual(str(route.get("transfer_from_node_id", "")), "node-c")
            self.assertEqual(str(route.get("transfer_from_tier", "")), "vram")
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_transfer_candidate_unavailable_forces_miss(self):
        keys = (
            "KORITH_ROUTER_PREFER_REPLAY_LOCAL_FOR_HIT",
            "KORITH_ROUTER_TRANSFER_ALLOWED_TIERS",
        )
        old = {k: os.environ.get(k) for k in keys}
        try:
            os.environ["KORITH_ROUTER_PREFER_REPLAY_LOCAL_FOR_HIT"] = "0"
            os.environ["KORITH_ROUTER_TRANSFER_ALLOWED_TIERS"] = "vram"
            self._register_nodes()
            self.router._session_affinity["sess-disallow"] = {"node_id": "node-a", "worker_id": "worker-a"}
            self.snapshots.upsert_location(
                fingerprint_hash="fp-disallow",
                snapshot_id="snap-nvme",
                org_id="org1",
                node_id="node-b",
                worker_id="worker-b",
                snapshot_path="/tmp/korith_router/nvme/snap-disallow.bin",
                size_bytes=1024 * 1024,
            )
            route = self.router._select_route(
                jobspec={"session_id": "sess-disallow"},
                fingerprint_hash="fp-disallow",
                org_id="org1",
                predicted_lane="HIT",
            )
            self.assertTrue(bool(route.get("force_miss", False)))
            self.assertEqual(str(route.get("final_lane", "")), "MISS")
            self.assertEqual(str(route.get("reason", "")), "transfer_candidate_unavailable")
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_snapshot_index_org_isolation(self):
        self.snapshots.upsert_location(
            fingerprint_hash="fp3",
            snapshot_id="a1",
            org_id="org-a",
            node_id="node-a",
            worker_id="w-a",
            snapshot_path="/tmp/a1.bin",
            size_bytes=10,
        )
        self.snapshots.upsert_location(
            fingerprint_hash="fp3",
            snapshot_id="b1",
            org_id="org-b",
            node_id="node-b",
            worker_id="w-b",
            snapshot_path="/tmp/b1.bin",
            size_bytes=20,
        )
        org_a = self.snapshots.get_locations("fp3", "org-a")
        org_b = self.snapshots.get_locations("fp3", "org-b")
        self.assertEqual(len(org_a), 1)
        self.assertEqual(len(org_b), 1)
        self.assertEqual(org_a[0]["org_id"], "org-a")
        self.assertEqual(org_b[0]["org_id"], "org-b")

    def test_snapshot_index_prefers_hot_storage_tiers(self):
        self._register_nodes()
        keys = ("KORITH_SNAPSHOT_VRAM_DIR", "KORITH_SNAPSHOT_RAM_DIR", "KORITH_SNAPSHOT_NVME_DIR")
        old = {k: os.environ.get(k) for k in keys}
        try:
            os.environ["KORITH_SNAPSHOT_VRAM_DIR"] = "/tmp/korith_test_tier/vram"
            os.environ["KORITH_SNAPSHOT_RAM_DIR"] = "/tmp/korith_test_tier/ram"
            os.environ["KORITH_SNAPSHOT_NVME_DIR"] = "/tmp/korith_test_tier/nvme"

            self.snapshots.upsert_location(
                fingerprint_hash="fp-tier",
                snapshot_id="cold",
                org_id="org1",
                node_id="node-a",
                worker_id="worker-a",
                snapshot_path="/tmp/korith_test_tier/nvme/fp-tier.bin",
                size_bytes=2048,
            )
            self.snapshots.upsert_location(
                fingerprint_hash="fp-tier",
                snapshot_id="hot",
                org_id="org1",
                node_id="node-b",
                worker_id="worker-b",
                snapshot_path="/tmp/korith_test_tier/vram/fp-tier.bin",
                size_bytes=2048,
            )

            locations = self.snapshots.get_locations("fp-tier", "org1")
            self.assertGreaterEqual(len(locations), 2)
            self.assertEqual(str(locations[0].get("storage_tier", "")), "vram")

            route = self.router._select_route(
                jobspec={},
                fingerprint_hash="fp-tier",
                org_id="org1",
                predicted_lane="HIT",
            )
            self.assertEqual(route["reason"], "node_locality")
            self.assertEqual(route["chosen_node_id"], "node-b")
            self.assertEqual(str(route.get("snapshot_tier", "")), "vram")
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_node_registry_heartbeat_and_selection(self):
        self.nodes.register_or_heartbeat(
            node_id="node-a",
            host="127.0.0.1",
            router_port=8100,
            gpu_count=1,
            inflight=3,
            queue_depth_hit=1,
            queue_depth_miss=3,
            capabilities={},
        )
        self.nodes.register_or_heartbeat(
            node_id="node-b",
            host="127.0.0.1",
            router_port=8101,
            gpu_count=1,
            inflight=1,
            queue_depth_hit=0,
            queue_depth_miss=1,
            capabilities={},
        )
        rows = self.nodes.list_nodes(healthy_only=True, max_stale_s=60)
        self.assertGreaterEqual(len(rows), 2)
        self.assertEqual(rows[0]["node_id"], "node-b")

        # Force node-a stale in sqlite fallback path.
        with sqlite3.connect(self.nodes.sqlite_path) as conn:
            conn.execute("UPDATE nodes SET last_heartbeat=? WHERE node_id='node-a'", (0.0,))
        healthy = self.nodes.list_nodes(healthy_only=True, max_stale_s=1)
        ids = {row["node_id"] for row in healthy}
        self.assertIn("node-b", ids)
        self.assertNotIn("node-a", ids)

    def test_select_lane_disables_spec_for_short_outputs(self):
        old = {k: os.environ.get(k) for k in ("KORITH_SPEC_ENABLED", "KORITH_SPEC_MIN_OUTPUT_TOKENS")}
        try:
            os.environ["KORITH_SPEC_ENABLED"] = "1"
            os.environ["KORITH_SPEC_MIN_OUTPUT_TOKENS"] = "128"
            jobspec = {
                "policy": {"allow_amf_reuse": True, "allow_spec": True},
                "deterministic_cfg": {"max_tokens": 64},
            }
            lane = self.router._select_lane(jobspec, _FakeAdapter(), "fp-short", "org1")
            self.assertEqual(lane, "MISS")
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_select_lane_requires_valid_draft_config(self):
        keys = (
            "KORITH_SPEC_ENABLED",
            "KORITH_SPEC_MIN_OUTPUT_TOKENS",
            "KORITH_SPEC_REQUIRE_DISTINCT_DRAFT",
            "KORITH_SPEC_CACHE_ONLY",
            "KORITH_SPEC_AUTO_CACHE_ONLY",
            "KORITH_SPEC_MAX_DRAFT_SIZE_RATIO",
            "KORITH_DRAFT_MODEL_PATH",
        )
        old = {k: os.environ.get(k) for k in keys}
        try:
            os.environ["KORITH_SPEC_ENABLED"] = "1"
            os.environ["KORITH_SPEC_MIN_OUTPUT_TOKENS"] = "1"
            os.environ["KORITH_SPEC_REQUIRE_DISTINCT_DRAFT"] = "1"
            os.environ["KORITH_SPEC_CACHE_ONLY"] = "0"
            os.environ["KORITH_SPEC_AUTO_CACHE_ONLY"] = "0"
            os.environ["KORITH_SPEC_MAX_DRAFT_SIZE_RATIO"] = "0.75"

            jobspec = {
                "policy": {"allow_amf_reuse": True, "allow_spec": True},
                "deterministic_cfg": {"max_tokens": 256},
                "model": {"model_path": ""},
            }

            # Missing draft model should disable spec lanes.
            lane_missing = self.router._select_lane(jobspec, _FakeAdapter(), "fp-draft-missing", "org1")
            self.assertEqual(lane_missing, "MISS")

            with tempfile.TemporaryDirectory() as td2:
                tdp = Path(td2)
                verify = tdp / "verify.gguf"
                draft_large = tdp / "draft_large.gguf"
                draft_small = tdp / "draft_small.gguf"
                verify.write_bytes(b"v" * 1000)
                draft_large.write_bytes(b"d" * 900)
                draft_small.write_bytes(b"d" * 500)
                jobspec["model"]["model_path"] = str(verify)

                os.environ["KORITH_DRAFT_MODEL_PATH"] = str(draft_large)
                lane_large = self.router._select_lane(jobspec, _FakeAdapter(), "fp-draft-large", "org1")
                self.assertEqual(lane_large, "MISS")

                os.environ["KORITH_DRAFT_MODEL_PATH"] = str(draft_small)
                lane_small = self.router._select_lane(jobspec, _FakeAdapter(), "fp-draft-small", "org1")
                self.assertEqual(lane_small, "SPEC_MISS")
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_select_lane_requires_ready_snapshot_when_enabled(self):
        old = os.environ.get("KORITH_ROUTER_REQUIRE_READY_SNAPSHOT_FOR_HIT")
        try:
            os.environ["KORITH_ROUTER_REQUIRE_READY_SNAPSHOT_FOR_HIT"] = "1"
            fingerprint_hash = "fp-ready-required"
            self.ledger.insert_job(
                job_id=str(uuid.uuid4()),
                created_at="2026-02-16T00:00:00Z",
                jobspec={},
                fingerprint={"model_hash": "m", "tokenizer_hash": "t"},
                prompt_hash="ph-ready-required",
                fingerprint_hash=fingerprint_hash,
                idempotency_key=None,
                status="SUCCEEDED",
                org_id="org1",
            )
            jobspec = {
                "policy": {"allow_amf_reuse": True, "allow_spec": False},
                "deterministic_cfg": {"max_tokens": 128},
            }
            lane, reason = self.router._select_lane_with_reason(jobspec, _FakeAdapter(), fingerprint_hash, "org1")
            self.assertEqual(lane, "MISS")
            self.assertEqual(reason, "replay_snapshot_unavailable")
        finally:
            if old is None:
                os.environ.pop("KORITH_ROUTER_REQUIRE_READY_SNAPSHOT_FOR_HIT", None)
            else:
                os.environ["KORITH_ROUTER_REQUIRE_READY_SNAPSHOT_FOR_HIT"] = old

    def test_select_lane_allows_hit_when_ready_snapshot_exists(self):
        old = os.environ.get("KORITH_ROUTER_REQUIRE_READY_SNAPSHOT_FOR_HIT")
        try:
            os.environ["KORITH_ROUTER_REQUIRE_READY_SNAPSHOT_FOR_HIT"] = "1"
            fingerprint_hash = "fp-ready-present"
            self.ledger.insert_job(
                job_id=str(uuid.uuid4()),
                created_at="2026-02-16T00:00:00Z",
                jobspec={},
                fingerprint={"model_hash": "m", "tokenizer_hash": "t"},
                prompt_hash="ph-ready-present",
                fingerprint_hash=fingerprint_hash,
                idempotency_key=None,
                status="SUCCEEDED",
                org_id="org1",
            )
            with tempfile.TemporaryDirectory() as td:
                snap_path = Path(td) / "snap.bin"
                snap_path.write_bytes(b"snapshot")
                self.snapshots.upsert_location(
                    fingerprint_hash=fingerprint_hash,
                    snapshot_id="snap-ready-present",
                    org_id="org1",
                    node_id="node-a",
                    worker_id="worker-a",
                    snapshot_path=str(snap_path),
                    size_bytes=int(snap_path.stat().st_size),
                )
                jobspec = {
                    "policy": {"allow_amf_reuse": True, "allow_spec": False},
                    "deterministic_cfg": {"max_tokens": 128},
                }
                lane, _ = self.router._select_lane_with_reason(jobspec, _FakeAdapter(), fingerprint_hash, "org1")
                self.assertEqual(lane, "HIT")
        finally:
            if old is None:
                os.environ.pop("KORITH_ROUTER_REQUIRE_READY_SNAPSHOT_FOR_HIT", None)
            else:
                os.environ["KORITH_ROUTER_REQUIRE_READY_SNAPSHOT_FOR_HIT"] = old

    def test_render_prompt_can_canonicalize_template_json_inputs(self):
        old = os.environ.get("KORITH_AMF_CANONICALIZE_TEMPLATE_JSON")
        try:
            os.environ["KORITH_AMF_CANONICALIZE_TEMPLATE_JSON"] = "1"
            jobspec = {
                "prompt_template": "CTX={ctx}",
                "input": {"ctx": {"b": 1, "a": 2}},
            }
            rendered = self.router._render_prompt(jobspec)
            self.assertEqual(rendered, "CTX={\"a\":2,\"b\":1}")
        finally:
            if old is None:
                os.environ.pop("KORITH_AMF_CANONICALIZE_TEMPLATE_JSON", None)
            else:
                os.environ["KORITH_AMF_CANONICALIZE_TEMPLATE_JSON"] = old

    def test_submit_propagates_tenant_id_and_sanitizes_owner_tenant(self):
        self.workers.register("worker-0", "127.0.0.1", 0, capabilities={})
        jobspec = {
            "schema_version": "korith.jobspec.v1",
            "backend_id": "korith_local",
            "model": {"model_id": "local-model", "model_path": "/tmp/fake.gguf"},
            "prompt": "hello world",
            "deterministic_cfg": {"seed": 1, "n_ctx": 1024, "n_batch": 64, "max_tokens": 16},
            "policy": {"allow_amf_reuse": True, "allow_spec": False},
            "owner": {"tenant_id": "tenant/alpha"},
        }
        self.router.submit(jobspec, org_id="org1")
        item = self.queue.dequeue("worker-0", timeout_s=0.1)
        self.assertIsNotNone(item)
        payload = item.payload
        self.assertEqual(str(payload["tenant_id"]), "tenant_alpha")
        self.assertEqual(str(payload["job"]["tenant_id"]), "tenant_alpha")
        self.assertEqual(str(payload["job"]["routing_decision"]["tenant_id"]), "tenant_alpha")

    def test_submit_routes_miss_execution_to_vllm_when_enabled(self):
        keys = (
            "KORITH_VLLM_MISS_BACKEND_ENABLED",
            "KORITH_VLLM_ENDPOINT",
            "KORITH_VLLM_MODEL_ID",
            "KORITH_VLLM_MISS_SOURCE_BACKENDS",
            "KORITH_VLLM_MISS_MIN_MAX_TOKENS",
            "KORITH_DECODE_OPT_PROFILE",
        )
        old = {k: os.environ.get(k) for k in keys}
        try:
            os.environ["KORITH_VLLM_MISS_BACKEND_ENABLED"] = "1"
            os.environ["KORITH_VLLM_ENDPOINT"] = "http://127.0.0.1:8000"
            os.environ["KORITH_VLLM_MODEL_ID"] = "vllm-model"
            os.environ["KORITH_VLLM_MISS_SOURCE_BACKENDS"] = "korith_local"
            os.environ["KORITH_VLLM_MISS_MIN_MAX_TOKENS"] = "0"
            os.environ["KORITH_DECODE_OPT_PROFILE"] = "0"
            self.workers.register("worker-0", "127.0.0.1", 0, capabilities={})
            jobspec = {
                "schema_version": "korith.jobspec.v1",
                "backend_id": "korith_local",
                "model": {"model_id": "local-model", "model_path": "/tmp/fake.gguf"},
                "prompt": "hello world",
                "deterministic_cfg": {"seed": 1, "n_ctx": 1024, "n_batch": 64, "max_tokens": 16},
                "policy": {"allow_amf_reuse": True, "allow_spec": False},
            }
            self.router.submit(jobspec, org_id="org1")
            item = self.queue.dequeue("worker-0", timeout_s=0.1)
            self.assertIsNotNone(item)
            payload = item.payload
            self.assertEqual(payload["lane"], "MISS")
            self.assertEqual(payload["job"]["execution_backend_id"], "vllm")
            self.assertEqual(payload["job"]["execution_model"]["model_id"], "vllm-model")
            self.assertEqual(payload["job"]["execution_model"]["endpoint"], "http://127.0.0.1:8000")
            self.assertEqual(payload["job"]["routing_decision"]["execution_reason"], "miss_backend_override")
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_submit_keeps_local_backend_when_vllm_threshold_not_met(self):
        keys = (
            "KORITH_VLLM_MISS_BACKEND_ENABLED",
            "KORITH_VLLM_ENDPOINT",
            "KORITH_VLLM_MODEL_ID",
            "KORITH_VLLM_MISS_SOURCE_BACKENDS",
            "KORITH_VLLM_MISS_MIN_MAX_TOKENS",
            "KORITH_DECODE_OPT_PROFILE",
        )
        old = {k: os.environ.get(k) for k in keys}
        try:
            os.environ["KORITH_VLLM_MISS_BACKEND_ENABLED"] = "1"
            os.environ["KORITH_VLLM_ENDPOINT"] = "http://127.0.0.1:8000"
            os.environ["KORITH_VLLM_MODEL_ID"] = "vllm-model"
            os.environ["KORITH_VLLM_MISS_SOURCE_BACKENDS"] = "korith_local"
            os.environ["KORITH_VLLM_MISS_MIN_MAX_TOKENS"] = "128"
            os.environ["KORITH_DECODE_OPT_PROFILE"] = "0"
            self.workers.register("worker-0", "127.0.0.1", 0, capabilities={})
            jobspec = {
                "schema_version": "korith.jobspec.v1",
                "backend_id": "korith_local",
                "model": {"model_id": "local-model", "model_path": "/tmp/fake.gguf"},
                "prompt": "hello world",
                "deterministic_cfg": {"seed": 1, "n_ctx": 1024, "n_batch": 64, "max_tokens": 16},
                "policy": {"allow_amf_reuse": True, "allow_spec": False},
            }
            self.router.submit(jobspec, org_id="org1")
            item = self.queue.dequeue("worker-0", timeout_s=0.1)
            self.assertIsNotNone(item)
            payload = item.payload
            self.assertEqual(payload["lane"], "MISS")
            self.assertEqual(payload["job"]["execution_backend_id"], "korith_local")
            self.assertEqual(payload["job"]["routing_decision"]["execution_reason"], "requested_backend")
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_submit_routes_miss_execution_to_vllm_with_decode_profile(self):
        keys = (
            "KORITH_VLLM_MISS_BACKEND_ENABLED",
            "KORITH_VLLM_ENDPOINT",
            "KORITH_VLLM_MODEL_ID",
            "KORITH_VLLM_MISS_SOURCE_BACKENDS",
            "KORITH_VLLM_MISS_MIN_MAX_TOKENS",
            "KORITH_DECODE_OPT_PROFILE",
        )
        old = {k: os.environ.get(k) for k in keys}
        try:
            os.environ["KORITH_VLLM_MISS_BACKEND_ENABLED"] = "0"
            os.environ["KORITH_VLLM_ENDPOINT"] = "http://127.0.0.1:8000"
            os.environ["KORITH_VLLM_MODEL_ID"] = "vllm-model"
            os.environ["KORITH_VLLM_MISS_SOURCE_BACKENDS"] = "korith_local"
            os.environ["KORITH_VLLM_MISS_MIN_MAX_TOKENS"] = "128"
            os.environ["KORITH_DECODE_OPT_PROFILE"] = "1"
            self.workers.register("worker-0", "127.0.0.1", 0, capabilities={})
            jobspec = {
                "schema_version": "korith.jobspec.v1",
                "backend_id": "korith_local",
                "model": {"model_id": "local-model", "model_path": "/tmp/fake.gguf"},
                "prompt": "hello world",
                "deterministic_cfg": {"seed": 1, "n_ctx": 1024, "n_batch": 64, "max_tokens": 256},
                "policy": {"allow_amf_reuse": True, "allow_spec": False},
            }
            self.router.submit(jobspec, org_id="org1")
            item = self.queue.dequeue("worker-0", timeout_s=0.1)
            self.assertIsNotNone(item)
            payload = item.payload
            self.assertEqual(payload["lane"], "MISS")
            self.assertEqual(payload["job"]["execution_backend_id"], "vllm")
            self.assertEqual(payload["job"]["routing_decision"]["execution_reason"], "miss_backend_override")
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_submit_keeps_hit_execution_on_requested_backend(self):
        keys = (
            "KORITH_VLLM_MISS_BACKEND_ENABLED",
            "KORITH_VLLM_ALL_LANES_BACKEND_ENABLED",
            "KORITH_VLLM_ENDPOINT",
            "KORITH_VLLM_MODEL_ID",
            "KORITH_VLLM_MISS_SOURCE_BACKENDS",
            "KORITH_VLLM_MISS_MIN_MAX_TOKENS",
            "KORITH_DECODE_OPT_PROFILE",
        )
        old = {k: os.environ.get(k) for k in keys}
        try:
            os.environ["KORITH_VLLM_MISS_BACKEND_ENABLED"] = "1"
            os.environ["KORITH_VLLM_ALL_LANES_BACKEND_ENABLED"] = "0"
            os.environ["KORITH_VLLM_ENDPOINT"] = "http://127.0.0.1:8000"
            os.environ["KORITH_VLLM_MODEL_ID"] = "vllm-model"
            os.environ["KORITH_VLLM_MISS_SOURCE_BACKENDS"] = "korith_local"
            os.environ["KORITH_VLLM_MISS_MIN_MAX_TOKENS"] = "0"
            os.environ["KORITH_DECODE_OPT_PROFILE"] = "0"
            self.workers.register("worker-0", "127.0.0.1", 0, capabilities={})
            jobspec = {
                "schema_version": "korith.jobspec.v1",
                "backend_id": "korith_local",
                "model": {"model_id": "local-model", "model_path": "/tmp/fake.gguf"},
                "prompt": "hello world",
                "deterministic_cfg": {"seed": 1, "n_ctx": 1024, "n_batch": 64, "max_tokens": 16},
                "policy": {"allow_amf_reuse": True, "allow_spec": False},
            }
            self.router.submit(jobspec, org_id="org1")
            first = self.queue.dequeue("worker-0", timeout_s=0.1)
            self.assertIsNotNone(first)
            self.assertEqual(first.payload["lane"], "MISS")
            self.router.submit(jobspec, org_id="org1")
            second = self.queue.dequeue("worker-0", timeout_s=0.1)
            self.assertIsNotNone(second)
            self.assertEqual(second.payload["lane"], "HIT")
            self.assertEqual(second.payload["job"]["execution_backend_id"], "korith_local")
            self.assertEqual(second.payload["job"]["routing_decision"]["execution_reason"], "requested_backend")
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_submit_routes_hit_execution_to_vllm_when_all_lanes_enabled(self):
        keys = (
            "KORITH_VLLM_MISS_BACKEND_ENABLED",
            "KORITH_VLLM_HIT_BACKEND_ENABLED",
            "KORITH_VLLM_ALL_LANES_BACKEND_ENABLED",
            "KORITH_VLLM_ENDPOINT",
            "KORITH_VLLM_MODEL_ID",
            "KORITH_VLLM_MISS_SOURCE_BACKENDS",
            "KORITH_VLLM_MISS_MIN_MAX_TOKENS",
            "KORITH_DECODE_OPT_PROFILE",
        )
        old = {k: os.environ.get(k) for k in keys}
        try:
            os.environ["KORITH_VLLM_MISS_BACKEND_ENABLED"] = "1"
            os.environ["KORITH_VLLM_HIT_BACKEND_ENABLED"] = "0"
            os.environ["KORITH_VLLM_ALL_LANES_BACKEND_ENABLED"] = "1"
            os.environ["KORITH_VLLM_ENDPOINT"] = "http://127.0.0.1:8000"
            os.environ["KORITH_VLLM_MODEL_ID"] = "vllm-model"
            os.environ["KORITH_VLLM_MISS_SOURCE_BACKENDS"] = "korith_local"
            os.environ["KORITH_VLLM_MISS_MIN_MAX_TOKENS"] = "0"
            os.environ["KORITH_DECODE_OPT_PROFILE"] = "0"
            self.workers.register("worker-0", "127.0.0.1", 0, capabilities={})
            jobspec = {
                "schema_version": "korith.jobspec.v1",
                "backend_id": "korith_local",
                "model": {"model_id": "local-model", "model_path": "/tmp/fake.gguf"},
                "prompt": "hello world",
                "deterministic_cfg": {"seed": 1, "n_ctx": 1024, "n_batch": 64, "max_tokens": 16},
                "policy": {"allow_amf_reuse": True, "allow_spec": False},
            }
            self.router.submit(jobspec, org_id="org1")
            first = self.queue.dequeue("worker-0", timeout_s=0.1)
            self.assertIsNotNone(first)
            self.assertEqual(first.payload["lane"], "MISS")
            self.assertEqual(first.payload["job"]["execution_backend_id"], "vllm")
            self.router.submit(jobspec, org_id="org1")
            second = self.queue.dequeue("worker-0", timeout_s=0.1)
            self.assertIsNotNone(second)
            self.assertEqual(second.payload["lane"], "HIT")
            self.assertEqual(second.payload["job"]["execution_backend_id"], "vllm")
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_submit_routes_hit_execution_to_vllm_when_hit_backend_enabled(self):
        keys = (
            "KORITH_VLLM_MISS_BACKEND_ENABLED",
            "KORITH_VLLM_HIT_BACKEND_ENABLED",
            "KORITH_VLLM_ALL_LANES_BACKEND_ENABLED",
            "KORITH_VLLM_ENDPOINT",
            "KORITH_VLLM_MODEL_ID",
            "KORITH_VLLM_MISS_SOURCE_BACKENDS",
            "KORITH_VLLM_MISS_MIN_MAX_TOKENS",
            "KORITH_VLLM_HIT_MIN_MAX_TOKENS",
            "KORITH_DECODE_OPT_PROFILE",
        )
        old = {k: os.environ.get(k) for k in keys}
        try:
            os.environ["KORITH_VLLM_MISS_BACKEND_ENABLED"] = "0"
            os.environ["KORITH_VLLM_HIT_BACKEND_ENABLED"] = "1"
            os.environ["KORITH_VLLM_ALL_LANES_BACKEND_ENABLED"] = "0"
            os.environ["KORITH_VLLM_ENDPOINT"] = "http://127.0.0.1:8000"
            os.environ["KORITH_VLLM_MODEL_ID"] = "vllm-model"
            os.environ["KORITH_VLLM_MISS_SOURCE_BACKENDS"] = "korith_local"
            os.environ["KORITH_VLLM_MISS_MIN_MAX_TOKENS"] = "0"
            os.environ["KORITH_VLLM_HIT_MIN_MAX_TOKENS"] = "0"
            os.environ["KORITH_DECODE_OPT_PROFILE"] = "0"
            self.workers.register("worker-0", "127.0.0.1", 0, capabilities={})
            jobspec = {
                "schema_version": "korith.jobspec.v1",
                "backend_id": "korith_local",
                "model": {"model_id": "local-model", "model_path": "/tmp/fake.gguf"},
                "prompt": "hello world",
                "deterministic_cfg": {"seed": 1, "n_ctx": 1024, "n_batch": 64, "max_tokens": 16},
                "policy": {"allow_amf_reuse": True, "allow_spec": False},
            }
            self.router.submit(jobspec, org_id="org1")
            first = self.queue.dequeue("worker-0", timeout_s=0.1)
            self.assertIsNotNone(first)
            self.assertEqual(first.payload["lane"], "MISS")
            self.assertEqual(first.payload["job"]["execution_backend_id"], "korith_local")
            self.assertEqual(first.payload["job"]["routing_decision"]["execution_reason"], "requested_backend")

            self.router.submit(jobspec, org_id="org1")
            second = self.queue.dequeue("worker-0", timeout_s=0.1)
            self.assertIsNotNone(second)
            self.assertEqual(second.payload["lane"], "HIT")
            self.assertEqual(second.payload["job"]["execution_backend_id"], "vllm")
            self.assertEqual(second.payload["job"]["routing_decision"]["execution_reason"], "hit_backend_override")
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_submit_uses_lane_specific_vllm_endpoint_and_model(self):
        keys = (
            "KORITH_VLLM_MISS_BACKEND_ENABLED",
            "KORITH_VLLM_HIT_BACKEND_ENABLED",
            "KORITH_VLLM_ALL_LANES_BACKEND_ENABLED",
            "KORITH_VLLM_ENDPOINT",
            "KORITH_VLLM_MODEL_ID",
            "KORITH_VLLM_HIT_ENDPOINT",
            "KORITH_VLLM_HIT_MODEL_ID",
            "KORITH_VLLM_MISS_SOURCE_BACKENDS",
            "KORITH_DECODE_OPT_PROFILE",
        )
        old = {k: os.environ.get(k) for k in keys}
        try:
            os.environ["KORITH_VLLM_MISS_BACKEND_ENABLED"] = "0"
            os.environ["KORITH_VLLM_HIT_BACKEND_ENABLED"] = "1"
            os.environ["KORITH_VLLM_ALL_LANES_BACKEND_ENABLED"] = "0"
            os.environ["KORITH_VLLM_ENDPOINT"] = "http://127.0.0.1:8000"
            os.environ["KORITH_VLLM_MODEL_ID"] = "vllm-default"
            os.environ["KORITH_VLLM_HIT_ENDPOINT"] = "http://127.0.0.1:9000"
            os.environ["KORITH_VLLM_HIT_MODEL_ID"] = "vllm-hit"
            os.environ["KORITH_VLLM_MISS_SOURCE_BACKENDS"] = "korith_local"
            os.environ["KORITH_DECODE_OPT_PROFILE"] = "0"
            self.workers.register("worker-0", "127.0.0.1", 0, capabilities={})
            jobspec = {
                "schema_version": "korith.jobspec.v1",
                "backend_id": "korith_local",
                "model": {"model_id": "local-model", "model_path": "/tmp/fake.gguf"},
                "prompt": "hello world",
                "deterministic_cfg": {"seed": 1, "n_ctx": 1024, "n_batch": 64, "max_tokens": 16},
                "policy": {"allow_amf_reuse": True, "allow_spec": False},
            }
            self.router.submit(jobspec, org_id="org1")
            _ = self.queue.dequeue("worker-0", timeout_s=0.1)
            self.router.submit(jobspec, org_id="org1")
            second = self.queue.dequeue("worker-0", timeout_s=0.1)
            self.assertIsNotNone(second)
            model = second.payload["job"]["execution_model"]
            self.assertEqual(str(second.payload["job"]["execution_backend_id"]), "vllm")
            self.assertEqual(str(model.get("endpoint", "")), "http://127.0.0.1:9000")
            self.assertEqual(str(model.get("model_id", "")), "vllm-hit")
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_submit_rejects_context_overflow_preflight(self):
        keys = (
            "KORITH_CONTEXT_RESERVE_TOKENS",
            "KORITH_CONTEXT_PROMPT_MULTIPLIER",
            "KORITH_CONTEXT_CHARS_PER_TOKEN_FLOOR",
        )
        old = {k: os.environ.get(k) for k in keys}
        try:
            os.environ["KORITH_CONTEXT_RESERVE_TOKENS"] = "8"
            os.environ["KORITH_CONTEXT_PROMPT_MULTIPLIER"] = "1.0"
            # De-emphasize character-based estimate in this test so prompt token count dominates.
            os.environ["KORITH_CONTEXT_CHARS_PER_TOKEN_FLOOR"] = "100.0"

            self.workers.register("worker-0", "127.0.0.1", 0, capabilities={})
            jobspec = {
                "schema_version": "korith.jobspec.v1",
                "backend_id": "korith_local",
                "model": {"model_id": "fake-model", "model_path": "/tmp/fake.gguf"},
                "prompt": "one two three four five six seven eight nine ten",
                "deterministic_cfg": {"seed": 1, "n_ctx": 32, "n_batch": 64, "max_tokens": 16},
                "policy": {"allow_amf_reuse": True, "allow_spec": False},
            }

            with self.assertRaises(RouterRequestError) as err:
                self.router.submit(jobspec, org_id="org1")
            self.assertEqual(err.exception.error_code, "CONTEXT_OVERFLOW")
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_kpi_summary_aggregates_savings_and_decode_targets(self):
        base = Path(self.tmp.name)
        m1 = base / "metrics_1.json"
        m2 = base / "metrics_2.json"
        m1.write_text(
            json.dumps(
                {
                    "amf": {"supported": True, "decision": "miss", "skip_ratio": 0.0},
                    "spec": {"enabled": False},
                    "kernels": {"kernels_applied": False},
                    "health": {"decode_cache": {"enabled": True, "hit": False, "saved_decode_ms_est": 0.0}},
                    "perf": {"tokens_out": 100, "total_ms": 100.0, "prefill_ms": 60.0, "decode_ms": 40.0},
                    "savings": {
                        "prefill_saved_ms": 0.0,
                        "spec_saved_ms": 0.0,
                        "kernels_saved_ms": 0.0,
                        "total_saved_ms": 0.0,
                    },
                }
            ),
            encoding="utf-8",
        )
        m2.write_text(
            json.dumps(
                {
                    "amf": {"supported": True, "decision": "hit", "skip_ratio": 1.0},
                    "spec": {"enabled": False},
                    "kernels": {"kernels_applied": False},
                    "health": {"decode_cache": {"enabled": True, "hit": True, "saved_decode_ms_est": 25.0}},
                    "perf": {"tokens_out": 100, "total_ms": 50.0, "prefill_ms": 0.0, "decode_ms": 50.0},
                    "savings": {
                        "prefill_saved_ms": 60.0,
                        "spec_saved_ms": 0.0,
                        "kernels_saved_ms": 0.0,
                        "total_saved_ms": 60.0,
                    },
                }
            ),
            encoding="utf-8",
        )

        self.ledger.insert_job(
            job_id="job-1",
            created_at="2026-01-01T00:00:00Z",
            jobspec={"schema_version": "korith.jobspec.v1"},
            fingerprint={},
            prompt_hash="p1",
            fingerprint_hash="fp1",
            idempotency_key=None,
            status="SUCCEEDED",
            org_id="org1",
        )
        self.ledger.insert_run(
            run_id="run-1",
            job_id="job-1",
            started_at="2026-01-01T00:00:00Z",
            finished_at="2026-01-01T00:01:00Z",
            exit_code=0,
            metrics_path=str(m1),
            events_path=str(base / "events_1.jsonl"),
            output_path=str(base / "output_1.txt"),
            log_path=str(base / "log_1.txt"),
            org_id="org1",
        )

        self.ledger.insert_job(
            job_id="job-2",
            created_at="2026-01-01T00:02:00Z",
            jobspec={"schema_version": "korith.jobspec.v1"},
            fingerprint={},
            prompt_hash="p2",
            fingerprint_hash="fp2",
            idempotency_key=None,
            status="SUCCEEDED",
            org_id="org1",
        )
        self.ledger.insert_run(
            run_id="run-2",
            job_id="job-2",
            started_at="2026-01-01T00:02:00Z",
            finished_at="2026-01-01T00:03:00Z",
            exit_code=0,
            metrics_path=str(m2),
            events_path=str(base / "events_2.jsonl"),
            output_path=str(base / "output_2.txt"),
            log_path=str(base / "log_2.txt"),
            org_id="org1",
        )

        out = self.router.kpi_summary(org_id="org1", limit=10, gpu_hourly_cost=3.6)
        kpi = out["kpi"]

        self.assertEqual(int(out["jobs_analyzed"]), 2)
        self.assertEqual(int(kpi["amf"]["hit_rows"]), 1)
        self.assertAlmostEqual(float(kpi["amf"]["hit_rate_pct"]), 50.0, places=3)
        self.assertAlmostEqual(float(kpi["amf"]["avg_skip_ratio"]), 0.5, places=3)
        self.assertEqual(int(kpi["decode_cache"]["checked_rows"]), 2)
        self.assertEqual(int(kpi["decode_cache"]["hit_rows"]), 1)
        self.assertAlmostEqual(float(kpi["decode_cache"]["hit_rate_pct"]), 50.0, places=3)
        self.assertAlmostEqual(float(kpi["decode_cache"]["saved_decode_ms_est"]), 25.0, places=3)
        self.assertEqual(int(kpi["decode_cache"]["avoided_decode_calls"]), 1)
        self.assertEqual(int(kpi["decode_cache"]["tokens_served_from_cache"]), 100)

        self.assertAlmostEqual(float(kpi["savings"]["total_saved_ms"]), 60.0, places=3)
        self.assertAlmostEqual(float(kpi["savings"]["baseline_total_ms"]), 210.0, places=3)
        self.assertAlmostEqual(float(kpi["savings"]["blended_savings_pct"]), (60.0 / 210.0) * 100.0, places=3)
        self.assertAlmostEqual(float(kpi["savings"]["decode_cut_pct"]), 0.0, places=3)

        targets = kpi["savings"]["decode_targets"]
        by_target = {int(row["target_savings_pct"]): row for row in targets}
        self.assertIn(70, by_target)
        self.assertIn(80, by_target)
        self.assertGreater(float(by_target[70]["required_decode_cut_pct"]), 0.0)

    def test_tenant_metrics_and_billing_report_are_isolated(self):
        base = Path(self.tmp.name)
        m_a = base / "metrics_tenant_a.json"
        m_b = base / "metrics_tenant_b.json"
        m_a.write_text(
            json.dumps(
                {
                    "amf": {"supported": True, "decision": "hit", "saved_ms": 40.0},
                    "perf": {"total_ms": 10.0, "decode_ms": 5.0},
                    "savings": {
                        "prefill_saved_ms": 40.0,
                        "spec_saved_ms": 0.0,
                        "kernels_saved_ms": 0.0,
                        "total_saved_ms": 40.0,
                    },
                }
            ),
            encoding="utf-8",
        )
        m_b.write_text(
            json.dumps(
                {
                    "amf": {"supported": True, "decision": "miss", "saved_ms": 0.0},
                    "perf": {"total_ms": 20.0, "decode_ms": 8.0},
                    "savings": {
                        "prefill_saved_ms": 0.0,
                        "spec_saved_ms": 0.0,
                        "kernels_saved_ms": 0.0,
                        "total_saved_ms": 0.0,
                    },
                }
            ),
            encoding="utf-8",
        )

        self.ledger.insert_job(
            job_id="job-tenant-a",
            created_at="2026-02-01T00:00:00Z",
            jobspec={"schema_version": "korith.jobspec.v1", "owner": {"tenant_id": "tenant-a"}},
            fingerprint={},
            prompt_hash="pta",
            fingerprint_hash="fpta",
            idempotency_key=None,
            status="SUCCEEDED",
            org_id="org1",
        )
        self.ledger.insert_run(
            run_id="run-tenant-a",
            job_id="job-tenant-a",
            started_at="2026-02-01T00:00:00Z",
            finished_at="2026-02-01T00:01:00Z",
            exit_code=0,
            metrics_path=str(m_a),
            events_path=str(base / "events_tenant_a.jsonl"),
            output_path=str(base / "output_tenant_a.txt"),
            log_path=str(base / "log_tenant_a.txt"),
            org_id="org1",
        )

        self.ledger.insert_job(
            job_id="job-tenant-b",
            created_at="2026-02-01T00:02:00Z",
            jobspec={"schema_version": "korith.jobspec.v1", "owner": {"tenant_id": "tenant-b"}},
            fingerprint={},
            prompt_hash="ptb",
            fingerprint_hash="fptb",
            idempotency_key=None,
            status="SUCCEEDED",
            org_id="org1",
        )
        self.ledger.insert_run(
            run_id="run-tenant-b",
            job_id="job-tenant-b",
            started_at="2026-02-01T00:02:00Z",
            finished_at="2026-02-01T00:03:00Z",
            exit_code=0,
            metrics_path=str(m_b),
            events_path=str(base / "events_tenant_b.jsonl"),
            output_path=str(base / "output_tenant_b.txt"),
            log_path=str(base / "log_tenant_b.txt"),
            org_id="org1",
        )

        tenant_a = self.router.tenant_metrics("tenant-a", org_id="org1", limit=10, gpu_hourly_cost=3.6)
        self.assertEqual(str(tenant_a["tenant_id"]), "tenant-a")
        self.assertEqual(int(tenant_a["total_requests"]), 1)
        self.assertAlmostEqual(float(tenant_a["amf_hit_rate"]), 1.0, places=6)
        self.assertAlmostEqual(float(tenant_a["total_saved_ms"]), 40.0, places=6)

        tenant_b = self.router.tenant_metrics("tenant-b", org_id="org1", limit=10, gpu_hourly_cost=3.6)
        self.assertEqual(str(tenant_b["tenant_id"]), "tenant-b")
        self.assertEqual(int(tenant_b["total_requests"]), 1)
        self.assertAlmostEqual(float(tenant_b["amf_hit_rate"]), 0.0, places=6)
        self.assertAlmostEqual(float(tenant_b["total_saved_ms"]), 0.0, places=6)

        billing = self.router.billing_report(org_id="org1", limit=10, gpu_hourly_cost=3.6)
        self.assertEqual(str(billing["org_id"]), "org1")
        reports = billing.get("reports", [])
        self.assertEqual(len(reports), 2)


if __name__ == "__main__":
    unittest.main()

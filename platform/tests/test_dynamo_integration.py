"""Integration tests for AMF × Dynamo integration modules.

Tests are isolated from external services — NATS, coordinator HTTP calls, and
filesystem snapshot writes are all mocked.  Each test is independent and fast.

Modules under test:
  - platform.runtime.dynamo_event_subscriber
  - platform.runtime.dynamo_nixl_backend
  - platform.runtime.dynamo_amf_router
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_nats_msg(subject: str, payload: Dict[str, Any]) -> MagicMock:
    """Create a minimal mock NATS message."""
    msg = MagicMock()
    msg.subject = subject
    msg.data = json.dumps(payload).encode("utf-8")
    return msg


def _block_payload(
    worker_id: str = "w1",
    block_id: str = "blk-001",
    token_hash: str = "abc123",
    num_tokens: int = 128,
    model_id: str = "llama3-8b",
    storage_tier: str = "G3",
    tenant_id: str = "tenant-a",
) -> Dict[str, Any]:
    return {
        "worker_id": worker_id,
        "block_id": block_id,
        "token_hash": token_hash,
        "num_tokens": num_tokens,
        "layer_start": 0,
        "layer_end": 32,
        "storage_tier": storage_tier,
        "model_id": model_id,
        "timestamp_ms": 1_700_000_000_000,
        "metadata": {"tenant_id": tenant_id},
    }


# ===========================================================================
# 1. Event Subscriber tests
# ===========================================================================


class TestKvBlockEventParsing(unittest.TestCase):
    def setUp(self) -> None:
        from platform.runtime.dynamo_event_subscriber import KvBlockEvent
        self.KvBlockEvent = KvBlockEvent

    def test_parse_valid_created(self) -> None:
        payload = _block_payload(num_tokens=256, storage_tier="G3")
        event = self.KvBlockEvent.from_nats_payload(
            "dynamo.kv.block.created", json.dumps(payload).encode()
        )
        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(event.event_type, "created")
        self.assertEqual(event.num_tokens, 256)
        self.assertEqual(event.storage_tier, "G3")
        self.assertEqual(event.storage_tier_label, "nvme")

    def test_parse_valid_evicted(self) -> None:
        payload = _block_payload(num_tokens=64)
        event = self.KvBlockEvent.from_nats_payload(
            "dynamo.kv.block.evicted", json.dumps(payload).encode()
        )
        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(event.event_type, "evicted")

    def test_parse_invalid_json(self) -> None:
        event = self.KvBlockEvent.from_nats_payload(
            "dynamo.kv.block.created", b"not-json"
        )
        self.assertIsNone(event)

    def test_parse_empty_payload(self) -> None:
        event = self.KvBlockEvent.from_nats_payload(
            "dynamo.kv.block.created", b"{}"
        )
        # Empty payload should still parse — fields default to sensible values.
        self.assertIsNotNone(event)

    def test_parse_invalid_subject(self) -> None:
        payload = _block_payload()
        event = self.KvBlockEvent.from_nats_payload(
            "bad_subject", json.dumps(payload).encode()
        )
        self.assertIsNone(event)


class TestPrefixAggregation(unittest.TestCase):
    def _make_config(self, min_tokens: int = 64) -> Any:
        from platform.runtime.dynamo_event_subscriber import DynamoEventConfig
        cfg = DynamoEventConfig()
        cfg.min_prefix_tokens = min_tokens
        return cfg

    def test_prefix_aggregation_from_blocks(self) -> None:
        from platform.runtime.dynamo_event_subscriber import KvBlockEvent, PrefixAggregator

        cfg = self._make_config(min_tokens=64)
        agg = PrefixAggregator(cfg)

        # Feed three created events with the same token_hash/worker_id.
        for i in range(3):
            payload = _block_payload(block_id=f"blk-{i}", num_tokens=128)
            ev = KvBlockEvent.from_nats_payload("dynamo.kv.block.created", json.dumps(payload).encode())
            self.assertIsNotNone(ev)
            result = agg.handle_event(ev)
            self.assertIsNone(result)  # No eviction yet.

        self.assertEqual(agg.active_prefix_count, 1)
        # Verify the accumulated token count.
        state_map = agg._prefixes
        self.assertEqual(len(state_map), 1)
        state = next(iter(state_map.values()))
        self.assertEqual(state.total_tokens, 384)

    def test_eviction_returns_prefix_state(self) -> None:
        from platform.runtime.dynamo_event_subscriber import KvBlockEvent, PrefixAggregator

        cfg = self._make_config(min_tokens=64)
        agg = PrefixAggregator(cfg)

        # First create a block.
        created = KvBlockEvent.from_nats_payload(
            "dynamo.kv.block.created",
            json.dumps(_block_payload(num_tokens=256)).encode(),
        )
        agg.handle_event(created)

        # Now evict.
        evicted = KvBlockEvent.from_nats_payload(
            "dynamo.kv.block.evicted",
            json.dumps(_block_payload(num_tokens=256)).encode(),
        )
        state = agg.handle_event(evicted)
        self.assertIsNotNone(state)
        assert state is not None
        self.assertEqual(state.total_tokens, 256)
        self.assertEqual(state.eviction_count, 1)
        # Prefix should be removed from active tracking.
        self.assertEqual(agg.active_prefix_count, 0)

    def test_small_prefix_not_returned_on_eviction(self) -> None:
        from platform.runtime.dynamo_event_subscriber import KvBlockEvent, PrefixAggregator

        cfg = self._make_config(min_tokens=512)
        agg = PrefixAggregator(cfg)

        created = KvBlockEvent.from_nats_payload(
            "dynamo.kv.block.created",
            json.dumps(_block_payload(num_tokens=32)).encode(),
        )
        agg.handle_event(created)

        evicted = KvBlockEvent.from_nats_payload(
            "dynamo.kv.block.evicted",
            json.dumps(_block_payload(num_tokens=32)).encode(),
        )
        state = agg.handle_event(evicted)
        # 32 tokens < 512 threshold → should be None.
        self.assertIsNone(state)


class TestEvictionTriggersPersist(unittest.TestCase):
    def test_eviction_triggers_persist_bridge(self) -> None:
        from platform.runtime.dynamo_event_subscriber import (
            AmfPersistBridge,
            DynamoEventConfig,
            KvBlockEvent,
            PrefixAggregator,
        )

        cfg = DynamoEventConfig()
        cfg.min_prefix_tokens = 64
        cfg.min_saved_ms_to_persist = 100.0

        mock_client = MagicMock()
        mock_client.register.return_value = True

        bridge = AmfPersistBridge(cfg, mock_client)
        agg = PrefixAggregator(cfg)

        # 512 tokens → prefill ~268K ms well above 100 ms threshold.
        created = KvBlockEvent.from_nats_payload(
            "dynamo.kv.block.created",
            json.dumps(_block_payload(num_tokens=512)).encode(),
        )
        agg.handle_event(created)

        evicted = KvBlockEvent.from_nats_payload(
            "dynamo.kv.block.evicted",
            json.dumps(_block_payload(num_tokens=512)).encode(),
        )
        state = agg.handle_event(evicted)
        self.assertIsNotNone(state)
        assert state is not None

        ok = bridge.maybe_persist(state)
        self.assertTrue(ok)
        mock_client.register.assert_called_once()

    def test_roi_gate_rejects_low_value(self) -> None:
        from platform.runtime.dynamo_event_subscriber import (
            AmfPersistBridge,
            DynamoEventConfig,
            KvBlockEvent,
            PrefixAggregator,
        )

        cfg = DynamoEventConfig()
        cfg.min_prefix_tokens = 1
        cfg.min_saved_ms_to_persist = 1_000_000.0  # Very high threshold.

        mock_client = MagicMock()
        bridge = AmfPersistBridge(cfg, mock_client)
        agg = PrefixAggregator(cfg)

        created = KvBlockEvent.from_nats_payload(
            "dynamo.kv.block.created",
            json.dumps(_block_payload(num_tokens=8)).encode(),
        )
        agg.handle_event(created)

        evicted = KvBlockEvent.from_nats_payload(
            "dynamo.kv.block.evicted",
            json.dumps(_block_payload(num_tokens=8)).encode(),
        )
        state = agg.handle_event(evicted)
        self.assertIsNotNone(state)
        assert state is not None

        ok = bridge.maybe_persist(state)
        self.assertFalse(ok)
        mock_client.register.assert_not_called()


# ===========================================================================
# 2. NIXL Backend tests
# ===========================================================================


class TestNixlBackendRoundtrip(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def _make_backend(self, min_roi: float = 0.0) -> Any:
        from platform.runtime.dynamo_nixl_backend import AmfNixlBackend, NixlBackendConfig
        cfg = NixlBackendConfig()
        cfg.snapshot_dir = self.tmpdir
        cfg.min_admit_roi = min_roi
        cfg.agg_window_ms = 0.0   # Flush immediately.
        cfg.max_storage_bytes = 10 * 1024 * 1024 * 1024
        cfg.coordinator_url = ""  # Disable real coordinator calls.
        return AmfNixlBackend(cfg)

    def test_put_and_get_roundtrip(self) -> None:
        backend = self._make_backend(min_roi=0.0)
        backend.register_volume({})

        meta = {
            "token_hash": "hash-abc",
            "worker_id": "w1",
            "model_id": "model-v1",
            "tenant_id": "tenant-a",
            "num_tokens": 10000,
            "storage_tier": "G3",
            "model_hash": "mh-1",
            "kv_version": "v1",
        }
        data = b"fake-kv-data-block"
        backend.put("blk-1", data, meta)

        # Flush aggregator manually (window=0 means it already flushed).
        backend._aggregator.flush_all()

        # Now try to get — must match model_hash and kv_version.
        request_meta = {"model_hash": "mh-1", "kv_version": "v1"}
        result = backend.get("blk-1", request_meta)
        self.assertIsNotNone(result)

    def test_correctness_validation_pass(self) -> None:
        from platform.runtime.dynamo_nixl_backend import CorrectnessValidator
        v = CorrectnessValidator()
        ok, reason = v.validate(
            stored_meta={"model_hash": "abc", "kv_version": "v2"},
            request_meta={"model_hash": "abc", "kv_version": "v2"},
        )
        self.assertTrue(ok)
        self.assertEqual(reason, "ok")

    def test_correctness_validation_fail_model(self) -> None:
        from platform.runtime.dynamo_nixl_backend import CorrectnessValidator
        v = CorrectnessValidator()
        ok, reason = v.validate(
            stored_meta={"model_hash": "abc", "kv_version": "v1"},
            request_meta={"model_hash": "WRONG", "kv_version": "v1"},
        )
        self.assertFalse(ok)
        self.assertIn("model_hash", reason)

    def test_correctness_validation_fail_kv_version(self) -> None:
        from platform.runtime.dynamo_nixl_backend import CorrectnessValidator
        v = CorrectnessValidator()
        ok, reason = v.validate(
            stored_meta={"model_hash": "abc", "kv_version": "v1"},
            request_meta={"model_hash": "abc", "kv_version": "v99"},
        )
        self.assertFalse(ok)
        self.assertIn("kv_version", reason)

    def test_admission_gate_rejects_low_roi(self) -> None:
        from platform.runtime.dynamo_nixl_backend import AdmissionGate
        gate = AdmissionGate(min_admit_roi=100.0)  # Absurdly high threshold.
        record = gate.evaluate("pfx-1", num_tokens=10, storage_tier="G3")
        self.assertFalse(record.admitted)
        self.assertEqual(gate.rejection_count, 1)

    def test_admission_gate_accepts_high_roi(self) -> None:
        from platform.runtime.dynamo_nixl_backend import AdmissionGate
        gate = AdmissionGate(min_admit_roi=1.5)
        # 500K tokens → estimated_prefill_ms >> restore_ms → ROI >> 1.5.
        record = gate.evaluate("pfx-2", num_tokens=500_000, storage_tier="G3")
        self.assertTrue(record.admitted)
        self.assertGreater(record.roi, 1.5)
        self.assertEqual(gate.admission_count, 1)

    def test_eviction_by_value(self) -> None:
        """When storage budget exceeded, lowest value_per_byte entries evicted first."""
        from platform.runtime.dynamo_nixl_backend import (
            AmfNixlBackend,
            NixlBackendConfig,
            _SnapshotEntry,
        )
        import time as _time

        cfg = NixlBackendConfig()
        cfg.snapshot_dir = self.tmpdir
        cfg.min_admit_roi = 0.0
        cfg.max_storage_bytes = 1000        # tiny budget
        cfg.eviction_watermark = 0.5
        cfg.coordinator_url = ""
        backend = AmfNixlBackend(cfg)

        # Manually inject two entries with different value_per_byte.
        low_val_entry = _SnapshotEntry(
            prefix_id="low",
            snapshot_path=os.path.join(self.tmpdir, "low.amf"),
            num_tokens=100,
            size_bytes=800,
            model_hash="m",
            kv_version="v1",
            metadata={"storage_tier": "G3"},
            stored_at=_time.time(),
        )
        low_val_entry.value_per_byte = 0.001

        high_val_entry = _SnapshotEntry(
            prefix_id="high",
            snapshot_path=os.path.join(self.tmpdir, "high.amf"),
            num_tokens=100,
            size_bytes=100,
            model_hash="m",
            kv_version="v1",
            metadata={"storage_tier": "G3"},
            stored_at=_time.time(),
        )
        high_val_entry.value_per_byte = 10.0

        with backend._mu:
            backend._entries["low"] = low_val_entry
            backend._entries["high"] = high_val_entry
            backend._total_bytes = 900  # below max but triggers watermark check

        # Evict to make room for 200 more bytes.
        backend._maybe_evict(200)

        with backend._mu:
            remaining = set(backend._entries.keys())

        # The low-value entry should be evicted first.
        self.assertNotIn("low", remaining)
        self.assertIn("high", remaining)


# ===========================================================================
# 3. Router Plugin tests
# ===========================================================================


def _make_worker(
    worker_id: str = "w1",
    node_id: str = "node-1",
    kv_overlap: float = 0.5,
    load: float = 0.2,
) -> Dict[str, Any]:
    return {
        "worker_id": worker_id,
        "node_id": node_id,
        "kv_overlap": kv_overlap,
        "load": load,
    }


class TestRouterPlugin(unittest.TestCase):
    def _make_router(self, coordinator_url: str = "") -> Any:
        from platform.runtime.dynamo_amf_router import AmfRouterPlugin
        with patch.dict(os.environ, {"AMF_ROUTER_ENABLED": "true", "AMF_COORDINATOR_URL": coordinator_url}):
            return AmfRouterPlugin(coordinator_url=coordinator_url)

    def test_short_context_defers_to_dynamo(self) -> None:
        router = self._make_router()
        workers = [
            _make_worker("w1", "node-1", kv_overlap=0.9, load=0.1),
            _make_worker("w2", "node-2", kv_overlap=0.1, load=0.9),
        ]
        scores = router.score_workers(
            request_tokens=100,   # < 4096 threshold
            prefix_hash="ph1",
            tenant_id="t1",
            dynamo_worker_scores=workers,
        )
        self.assertEqual(len(scores), 2)
        # Highest kv_overlap worker should win when tokens < threshold.
        self.assertEqual(scores[0].worker_id, "w1")
        self.assertEqual(scores[0].routed_by, "dynamo")

    def test_long_context_prefers_amf_snapshot(self) -> None:
        from platform.runtime.dynamo_amf_router import AmfRouterPlugin

        mock_client = MagicMock()
        # node-2 has an AMF snapshot.
        mock_client.lookup.return_value = [{"node_id": "node-2", "metadata": {"storage_tier": "G3"}}]

        with patch("platform.runtime.dynamo_amf_router.AmfCoordinatorClient", return_value=mock_client):
            with patch.dict(os.environ, {
                "AMF_ROUTER_ENABLED": "true",
                "AMF_COORDINATOR_URL": "http://fake:8500",
                "AMF_MIN_PREFIX_TOKENS_FOR_AMF": "4096",
            }):
                router = AmfRouterPlugin(coordinator_url="http://fake:8500")
                router._coordinator = mock_client

        workers = [
            _make_worker("w1", "node-1", kv_overlap=0.95, load=0.05),  # High Dynamo overlap.
            _make_worker("w2", "node-2", kv_overlap=0.10, load=0.10),  # Has AMF snapshot.
        ]
        scores = router.score_workers(
            request_tokens=130_000,   # Long context.
            prefix_hash="ph-long",
            tenant_id="t1",
            dynamo_worker_scores=workers,
        )
        # node-2 with AMF snapshot should beat node-1's higher kv_overlap.
        self.assertEqual(scores[0].worker_id, "w2")
        self.assertEqual(scores[0].routed_by, "amf")

    def test_no_snapshot_falls_through_to_dynamo(self) -> None:
        from platform.runtime.dynamo_amf_router import AmfRouterPlugin

        mock_client = MagicMock()
        mock_client.lookup.return_value = []  # No AMF snapshots anywhere.

        router = AmfRouterPlugin.__new__(AmfRouterPlugin)
        router._enabled = True
        router._coordinator = mock_client
        router._restore_weight = 0.8
        router._kv_overlap_weight = 0.5
        router._load_weight = 0.3
        router._min_prefix_tokens = 4096
        router._total_queries = 0
        router._amf_routed = 0
        router._dynamo_routed = 0
        router._total_savings_ms = 0.0

        workers = [
            _make_worker("w1", "node-1", kv_overlap=0.7, load=0.1),
            _make_worker("w2", "node-2", kv_overlap=0.3, load=0.1),
        ]
        scores = router.score_workers(130_000, "ph-x", "t1", workers)
        # No AMF → highest Dynamo overlap wins.
        self.assertEqual(scores[0].worker_id, "w1")

    def test_recompute_vs_restore_estimate(self) -> None:
        from platform.runtime.dynamo_amf_router import estimate_recompute_cost_ms, estimate_restore_cost_ms

        # Verify rough benchmark ranges.
        recompute_120k = estimate_recompute_cost_ms(120_000)
        self.assertGreater(recompute_120k, 50_000)    # At least 50 seconds.
        self.assertLess(recompute_120k, 500_000)      # Less than 500 seconds.

        restore_nvme_120k = estimate_restore_cost_ms(120_000, "nvme")
        self.assertGreater(restore_nvme_120k, 1_000)  # At least 1 second.
        # Recompute should be more expensive than restore for 120K tokens.
        self.assertGreater(recompute_120k, restore_nvme_120k)

        # vram should be faster than nvme.
        restore_vram = estimate_restore_cost_ms(120_000, "vram")
        restore_nvme = estimate_restore_cost_ms(120_000, "nvme")
        self.assertLess(restore_vram, restore_nvme)

        # remote should be slowest.
        restore_remote = estimate_restore_cost_ms(120_000, "remote")
        self.assertGreater(restore_remote, restore_nvme)

    def test_combined_scoring_formula(self) -> None:
        from platform.runtime.dynamo_amf_router import AmfRouterPlugin

        mock_client = MagicMock()
        mock_client.lookup.return_value = [{"node_id": "node-1", "metadata": {"storage_tier": "G3"}}]

        router = AmfRouterPlugin.__new__(AmfRouterPlugin)
        router._enabled = True
        router._coordinator = mock_client
        router._restore_weight = 0.8
        router._kv_overlap_weight = 0.5
        router._load_weight = 0.3
        router._min_prefix_tokens = 1
        router._total_queries = 0
        router._amf_routed = 0
        router._dynamo_routed = 0
        router._total_savings_ms = 0.0

        workers = [
            _make_worker("w1", "node-1", kv_overlap=0.5, load=0.2),
        ]
        scores = router.score_workers(200_000, "ph-formula", "t1", workers)
        self.assertEqual(len(scores), 1)
        s = scores[0]
        # Score should reflect weighted formula.
        expected_floor = 0.5 * 0.5 - 0.2 * 0.3  # kv_weight + amf_contribution - load_weight
        self.assertGreater(s.combined_score, expected_floor)

    def test_canonicalization_applied(self) -> None:
        from platform.runtime.dynamo_amf_router import AmfRouterPlugin

        router = AmfRouterPlugin.__new__(AmfRouterPlugin)
        router._enabled = False
        router._coordinator = None
        router._restore_weight = 0.8
        router._kv_overlap_weight = 0.5
        router._load_weight = 0.3
        router._min_prefix_tokens = 4096
        router._total_queries = 0
        router._amf_routed = 0
        router._dynamo_routed = 0
        router._total_savings_ms = 0.0

        # Two prompts that differ only in trailing whitespace should hash identically.
        h1 = router.hash_prefix("Hello world  \n\n")
        h2 = router.hash_prefix("Hello world")
        self.assertEqual(h1, h2)


# ===========================================================================
# 4. End-to-end integration test
# ===========================================================================


class TestEvictionToRestoreCycle(unittest.TestCase):
    """Full cycle: blocks created → evicted → AMF persists → router finds snapshot."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def test_eviction_to_restore_cycle(self) -> None:
        from platform.runtime.dynamo_event_subscriber import (
            AmfPersistBridge,
            DynamoEventConfig,
            KvBlockEvent,
            PrefixAggregator,
        )
        from platform.runtime.dynamo_amf_router import AmfRouterPlugin

        # --- Step 1: Simulate block creation ---
        cfg = DynamoEventConfig()
        cfg.min_prefix_tokens = 64
        cfg.min_saved_ms_to_persist = 100.0
        cfg.snapshot_dir = self.tmpdir
        cfg.node_id = "node-1"

        registered_hash: Dict[str, Any] = {}

        mock_client = MagicMock()

        def _fake_register(*, prefix_hash, tenant_id, node_id, worker_id, metadata):
            registered_hash["hash"] = prefix_hash
            registered_hash["node_id"] = node_id
            return True

        mock_client.register.side_effect = _fake_register

        bridge = AmfPersistBridge(cfg, mock_client)
        agg = PrefixAggregator(cfg)

        payload = _block_payload(worker_id="w1", num_tokens=1024, model_id="llama3", tenant_id="t1")

        created = KvBlockEvent.from_nats_payload("dynamo.kv.block.created", json.dumps(payload).encode())
        agg.handle_event(created)

        # --- Step 2: Block gets evicted ---
        evicted = KvBlockEvent.from_nats_payload("dynamo.kv.block.evicted", json.dumps(payload).encode())
        state = agg.handle_event(evicted)
        self.assertIsNotNone(state)
        assert state is not None

        # --- Step 3: AMF persists the prefix ---
        ok = bridge.maybe_persist(state)
        self.assertTrue(ok)
        self.assertIn("hash", registered_hash)
        persisted_hash = registered_hash["hash"]
        persisted_node = registered_hash["node_id"]

        # --- Step 4: New request arrives; router looks up AMF ---
        router_client = MagicMock()
        router_client.lookup.return_value = [
            {"node_id": persisted_node, "metadata": {"storage_tier": "G3"}}
        ]

        router = AmfRouterPlugin.__new__(AmfRouterPlugin)
        router._enabled = True
        router._coordinator = router_client
        router._restore_weight = 0.8
        router._kv_overlap_weight = 0.5
        router._load_weight = 0.3
        router._min_prefix_tokens = 64
        router._total_queries = 0
        router._amf_routed = 0
        router._dynamo_routed = 0
        router._total_savings_ms = 0.0

        workers = [
            _make_worker("w1", persisted_node, kv_overlap=0.0, load=0.1),
            _make_worker("w2", "node-2", kv_overlap=0.8, load=0.1),
        ]

        best = router.best_worker(
            request_tokens=1024,
            prefix_hash=persisted_hash,
            tenant_id="t1",
            dynamo_worker_scores=workers,
        )

        # Router should prefer the node with the AMF snapshot even though
        # the other node has higher Dynamo kv_overlap.
        self.assertIsNotNone(best)
        assert best is not None
        self.assertEqual(best.node_id, persisted_node)
        self.assertEqual(best.routed_by, "amf")


if __name__ == "__main__":
    unittest.main()

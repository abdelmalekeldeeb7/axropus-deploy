"""Unit tests for TieredCacheRouter (§5 of the design doc)."""

from __future__ import annotations

import pytest
import torch

from korith_vllm_ext.compressed_vram_pool import CompressedVRAMPool
from korith_vllm_ext.lmcache_adapter import LMCacheAdapter
from korith_vllm_ext.tiered_router import (
    CacheTier,
    MissReason,
    TieredCacheRouter,
    WritePolicy,
    compute_prefix_hash,
)


@pytest.fixture
def tiny_router():
    pool = CompressedVRAMPool(
        num_layers=2,
        bytes_per_layer=1 << 18,
        block_bytes=1 << 13,
        default_format="int4_sym_block",
        device="cpu",
    )
    lmc = LMCacheAdapter(enabled=True, backend="cpu")
    return TieredCacheRouter(
        pool=pool,
        lmcache=lmc,
        min_prefix_tokens=4,
        write_policy=WritePolicy.ALWAYS,
    )


def _kv(layers=2, tokens=64, heads=4, head_dim=64):
    return torch.randn(layers, 2, tokens, heads, head_dim)


def test_first_request_misses(tiny_router):
    tokens = list(range(1, 65))
    h = compute_prefix_hash(tokens)
    result = tiny_router.lookup(h, tokens)
    assert result.tier == CacheTier.COLD
    assert result.miss_reason == MissReason.FIRST_REQUEST


def test_cold_then_warm_path(tiny_router):
    tokens = list(range(1, 65))
    h = compute_prefix_hash(tokens)
    kv = _kv()
    tiny_router.store_after_prefill(h, tokens, kv, savings_ms=100)
    r = tiny_router.lookup(h, tokens)
    assert r.tier == CacheTier.G1_AMF
    assert r.payload is not None


def test_lmcache_fallback_on_pool_eviction(tiny_router):
    tokens = list(range(1, 65))
    h = compute_prefix_hash(tokens)
    kv = _kv()
    tiny_router.store_after_prefill(h, tokens, kv)
    tiny_router.pool.clear()
    r = tiny_router.lookup(h, tokens)
    assert r.tier == CacheTier.G3_LMCACHE


def test_short_prefix_rejected(tiny_router):
    tokens = [1, 2]
    h = compute_prefix_hash(tokens)
    r = tiny_router.lookup(h, tokens)
    assert r.tier == CacheTier.COLD
    assert r.miss_reason == MissReason.PREFIX_TOO_SHORT


def test_write_policy_large_only():
    pool = CompressedVRAMPool(num_layers=2, bytes_per_layer=1<<18, device="cpu")
    lmc = LMCacheAdapter(enabled=True, backend="cpu")
    router = TieredCacheRouter(
        pool=pool,
        lmcache=lmc,
        min_prefix_tokens=4,
        write_policy=WritePolicy.LARGE_ONLY,
        large_write_threshold=1000,
    )
    small_tokens = list(range(100))
    h = compute_prefix_hash(small_tokens)
    kv = _kv(tokens=100)
    router.store_after_prefill(h, small_tokens, kv)
    # Small prefill should not have been mirrored to lmcache.
    assert lmc.stats()["stores"] == 0


def test_miss_reason_tracks_evicted_state(tiny_router):
    tokens = list(range(1, 65))
    h = compute_prefix_hash(tokens)
    kv = _kv()
    tiny_router.store_after_prefill(h, tokens, kv)
    # Warm hit first.
    tiny_router.lookup(h, tokens)
    # Manually evict from both tiers.
    tiny_router.pool.clear()
    tiny_router.lmcache._cpu_store.clear()
    r = tiny_router.lookup(h, tokens)
    assert r.tier == CacheTier.COLD
    assert r.miss_reason == MissReason.EVICTED_FROM_AMF


def test_stats_json_serializable(tiny_router):
    tokens = list(range(1, 65))
    h = compute_prefix_hash(tokens)
    tiny_router.store_after_prefill(h, tokens, _kv())
    tiny_router.lookup(h, tokens)
    import json
    s = tiny_router.stats()
    s_json = json.dumps(s, default=str)
    assert "tier_hits" in s_json
    assert "miss_reasons" in s_json


def test_prefetch_resolver_populates_pool():
    pool = CompressedVRAMPool(num_layers=2, bytes_per_layer=1<<18, device="cpu")
    lmc = LMCacheAdapter(enabled=False)
    router = TieredCacheRouter(pool, lmc, min_prefix_tokens=4)

    kv = _kv()
    def resolver(h):
        return kv

    router.set_prefetch_resolver(resolver)
    router.prefetch("warmme")
    # Give the worker thread a moment.
    import time
    for _ in range(50):
        if "warmme" in pool:
            break
        time.sleep(0.02)
    router.stop_prefetch()
    assert "warmme" in pool

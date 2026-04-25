"""Unit tests for CompressedVRAMPool (§3 of the design doc).

Covered behaviours:

    * Insertion, retrieval, deletion
    * Eviction under memory pressure
    * Reuse-score policy protecting high-value prefixes
    * vLLM block-table mapping (via a fake table)
    * Restore round-trip producing tensors of the original shape
"""

from __future__ import annotations

import time

import pytest
import torch

from korith_vllm_ext.compressed_vram_pool import (
    CompressedVRAMPool,
    _ExternalBlockRef,
)


@pytest.fixture
def small_pool():
    return CompressedVRAMPool(
        num_layers=4,
        bytes_per_layer=1 << 18,   # 256 KB / layer
        block_bytes=1 << 13,       # 8 KB blocks → 32 per layer
        default_format="int4_sym_block",
        device="cpu",
    )


def _fake_kv(layers: int, tokens: int, heads: int = 4, head_dim: int = 64) -> torch.Tensor:
    return torch.randn(layers, 2, tokens, heads, head_dim)


def test_pool_insert_and_get(small_pool):
    kv = _fake_kv(small_pool.num_layers, 64)
    assert small_pool.put_from_raw("abc", kv) is True
    assert "abc" in small_pool
    entry = small_pool.get("abc")
    assert entry is not None
    assert entry.num_tokens == 64
    assert entry.format == "int4_sym_block"


def test_pool_delete(small_pool):
    kv = _fake_kv(small_pool.num_layers, 32)
    small_pool.put_from_raw("x", kv)
    assert small_pool.delete("x") is True
    assert small_pool.get("x") is None


def test_pool_eviction_under_pressure(small_pool):
    # Each prefix uses ~16KB across 4 layers; pool has 128KB total.
    # Inserting many prefixes must trigger eviction.
    for i in range(32):
        kv = _fake_kv(small_pool.num_layers, 64)
        small_pool.put_from_raw(f"p{i:02d}", kv)
        time.sleep(0.0005)

    stats = small_pool.stats()
    assert stats["evictions"] > 0
    assert small_pool.used_bytes() <= small_pool.capacity_bytes()


def test_pool_reuse_score_protects_valuable_entries(small_pool):
    # Insert a very valuable entry, then hammer it with hits.
    kv_hot  = _fake_kv(small_pool.num_layers, 32)
    kv_cold = _fake_kv(small_pool.num_layers, 32)
    small_pool.put_from_raw("hot", kv_hot, savings_ms=500.0)
    for _ in range(10):
        small_pool.get("hot")

    # Flood the pool with fresh low-value inserts.
    for i in range(20):
        small_pool.put_from_raw(f"cold{i}", kv_cold, savings_ms=0.0)

    # The hot entry should survive thanks to hit_count + savings weighting.
    assert "hot" in small_pool


def test_pool_restore_to_tensor_roundtrip(small_pool):
    kv = _fake_kv(small_pool.num_layers, 64)
    small_pool.put_from_raw("a", kv)
    restored = small_pool.restore_to_tensor("a", target_dtype=torch.float32)
    assert restored is not None
    assert restored.shape == kv.shape
    rel_err = (restored - kv).norm().item() / kv.norm().item()
    # int4_sym_block baseline error ~0.20 on unit-normal data.
    assert rel_err < 0.30


def test_pool_stats_reflect_operations(small_pool):
    kv = _fake_kv(small_pool.num_layers, 32)
    small_pool.put_from_raw("a", kv)
    small_pool.get("a")
    small_pool.get("not_here")
    stats = small_pool.stats()
    assert stats["hits"] >= 1
    assert stats["misses"] >= 1
    assert stats["inserts"] == 1


def test_pool_map_into_fake_vllm_table(small_pool):
    kv = _fake_kv(small_pool.num_layers, 32)
    small_pool.put_from_raw("seq", kv)
    fake_table = {42: [None] * small_pool.num_layers}
    ok = small_pool.map_into_vllm("seq", 42, fake_table)
    assert ok
    for ref in fake_table[42]:
        assert isinstance(ref, _ExternalBlockRef)
        assert ref.is_external is True


def test_pool_put_wrong_shape_raises(small_pool):
    with pytest.raises(ValueError):
        small_pool.put_from_raw("bad", torch.randn(2, 3))  # not 5-D


def test_pool_overflow_returns_false():
    # Pool deliberately sized smaller than a single insert so allocation fails.
    tiny = CompressedVRAMPool(
        num_layers=1,
        bytes_per_layer=1024,
        block_bytes=1024,
        default_format="int4_sym_block",
        device="cpu",
    )
    kv = torch.randn(1, 2, 1024, 8, 128)  # way too big
    assert tiny.put_from_raw("huge", kv) is False

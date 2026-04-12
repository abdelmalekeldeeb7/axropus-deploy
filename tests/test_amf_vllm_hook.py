"""End-to-end hook tests covering the request lifecycle (§6 of the design)."""

from __future__ import annotations

import pytest
import torch

from korith_vllm_ext.amf_vllm_hook import AMFvLLMHook, HookAction
from korith_vllm_ext.compressed_vram_pool import CompressedVRAMPool
from korith_vllm_ext.lmcache_adapter import LMCacheAdapter
from korith_vllm_ext.tiered_router import (
    TieredCacheRouter,
    WritePolicy,
    compute_prefix_hash,
)


@pytest.fixture
def hook():
    pool = CompressedVRAMPool(
        num_layers=4,
        bytes_per_layer=1 << 20,
        block_bytes=1 << 13,
        default_format="int4_sym_block",
        device="cpu",
    )
    lmc = LMCacheAdapter(enabled=True, backend="cpu")
    router = TieredCacheRouter(
        pool=pool,
        lmcache=lmc,
        min_prefix_tokens=4,
        write_policy=WritePolicy.ALWAYS,
    )
    return AMFvLLMHook(
        pool=pool,
        router=router,
        num_layers=4,
        num_kv_heads=4,
        head_dim=64,
        min_prefix=4,
    )


def _kv(layers=4, tokens=128, heads=4, head_dim=64):
    return torch.randn(layers, 2, tokens, heads, head_dim)


def test_cold_then_warm_lifecycle(hook):
    tokens = list(range(1, 129))

    # Cold.
    d1 = hook.on_request_arrival(1, tokens)
    assert d1.action == HookAction.COLD_PREFILL

    hook.on_prefill_complete(1, _kv(), saved_ms=100.0)
    hook.on_request_complete(1)

    # Warm.
    d2 = hook.on_request_arrival(2, tokens)
    assert d2.action == HookAction.SKIP_PREFILL_TO_DECODE
    assert d2.pool_entry is not None
    hook.on_request_complete(2)

    stats = hook.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["saves"] == 1


def test_short_prefix_bypasses_amf(hook):
    d = hook.on_request_arrival(3, [1, 2])
    assert d.action == HookAction.COLD_PREFILL
    assert d.prefix_hash == ""


def test_hit_prompts_block_table_injection(hook):
    tokens = list(range(1, 129))
    hook.on_request_arrival(1, tokens)
    hook.on_prefill_complete(1, _kv())
    hook.on_request_complete(1)

    # Warm hit again; inject into a fake block table.
    d = hook.on_request_arrival(2, tokens)
    fake_table = {2: [None] * 4}
    ok = hook.inject_blocks(2, fake_table)
    assert ok is True
    # Every slot should now point at a CompressedBlock ref.
    for ref in fake_table[2]:
        assert ref is not None


def test_model_runner_skip_flag(hook):
    tokens = list(range(1, 129))
    hook.on_request_arrival(1, tokens)
    hook.on_prefill_complete(1, _kv())
    hook.on_request_complete(1)

    hook.on_request_arrival(2, tokens)
    assert hook.should_skip_prefill_tensors(2) is True
    hook.on_request_complete(2)
    assert hook.should_skip_prefill_tensors(2) is False


def test_disabled_hook_is_passthrough():
    pool = CompressedVRAMPool(num_layers=2, bytes_per_layer=1 << 18, device="cpu")
    hook = AMFvLLMHook(pool, num_layers=2, num_kv_heads=4, head_dim=64, enabled=False)
    tokens = list(range(64))
    d = hook.on_request_arrival(1, tokens)
    assert d.action == HookAction.COLD_PREFILL
    assert hook.stats()["hits"] == 0

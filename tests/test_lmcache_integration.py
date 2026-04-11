"""Tests for the LMCache fallback tier integration.

Test plan (from the integration design):

Unit tests (5):
  1. test_adapter_no_lmcache_installed — import-guarded no-op path
  2. test_adapter_store_and_retrieve — roundtrip via a stubbed engine
  3. test_router_amf_hit_skips_lmcache
  4. test_router_amf_miss_tries_lmcache
  5. test_router_promotion — LMCache hit promotes into AMF pool

Integration tests (3):
  6. test_lmcache_fallback_end_to_end — full e2e (skipped if torch+cuda unavailable)
  7. test_no_regression_without_lmcache — zero overhead with flag off
  8. test_writethrough — cold prefill stores to both tiers

The integration tests are marked with ``@pytest.mark.integration`` and
skipped automatically when torch or CUDA is not available so this file
is safe to run in any CI environment.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, List, Optional
from unittest.mock import MagicMock

import pytest

# Ensure the project root is importable.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ── Test 1: Adapter no-op when LMCache is not installed ─────────────────────

def test_adapter_no_lmcache_installed(monkeypatch):
    """If LMCache isn't installed, LMCacheAdapter must be a silent no-op.

    Every public method should be safe to call and return a falsy result
    rather than raising ImportError or AttributeError.
    """
    # Force LMCACHE_AVAILABLE = False even if the lib happens to be installed
    # in the test environment.
    from korith_vllm_ext import lmcache_adapter
    monkeypatch.setattr(lmcache_adapter, "LMCACHE_AVAILABLE", False)
    monkeypatch.setattr(lmcache_adapter, "LMCacheEngine", None)
    monkeypatch.setattr(lmcache_adapter, "LMCacheEngineConfig", None)
    monkeypatch.setattr(lmcache_adapter, "CacheEngineKey", None)

    adapter = lmcache_adapter.LMCacheAdapter(config_path=None)

    # Health check
    assert adapter.available is False

    # Lookup must not raise and must return None
    result = adapter.lookup("deadbeef", [1, 2, 3])
    assert result is None

    # Store must not raise and must return False
    ok = adapter.store("deadbeef", [1, 2, 3], kv_tensor=None)
    assert ok is False

    # Stats must be a dict
    stats = adapter.stats()
    assert isinstance(stats, dict)
    assert stats["available"] is False
    assert stats["hits"] == 0


# ── Test 2: Adapter store/retrieve round-trip with a stubbed engine ─────────

def test_adapter_store_and_retrieve(monkeypatch):
    """Stub out the LMCache engine and verify store/retrieve round-trips."""
    from korith_vllm_ext import lmcache_adapter

    # Fake CacheEngineKey that the stub engine can use as a dict key.
    class FakeKey:
        def __init__(self, h):
            self.h = h
        def __hash__(self): return hash(self.h)
        def __eq__(self, o): return isinstance(o, FakeKey) and self.h == o.h

    class FakeEngine:
        def __init__(self, cfg=None):
            self._data: dict = {}
        def store(self, key, kv):
            self._data[key] = kv
        def retrieve(self, key):
            return self._data.get(key)

    class FakeConfig:
        @classmethod
        def from_file(cls, p): return cls()
        @classmethod
        def from_env(cls):     return cls()

    monkeypatch.setattr(lmcache_adapter, "LMCACHE_AVAILABLE", True)
    monkeypatch.setattr(lmcache_adapter, "LMCacheEngine", FakeEngine)
    monkeypatch.setattr(lmcache_adapter, "LMCacheEngineConfig", FakeConfig)
    monkeypatch.setattr(
        lmcache_adapter,
        "CacheEngineKey",
        lambda **kw: FakeKey(kw.get("chunk_hash") or kw.get("prefix_hash") or str(kw)),
    )

    adapter = lmcache_adapter.LMCacheAdapter()
    assert adapter.available

    # Round-trip a fake "tensor" (just a string payload for the test)
    fake_kv = "payload-bytes"
    ok = adapter.store("pfxABC", [10, 20, 30], fake_kv)
    assert ok is True

    retrieved = adapter.lookup("pfxABC", [10, 20, 30])
    assert retrieved == fake_kv

    # Miss returns None
    miss = adapter.lookup("unknown", [1])
    assert miss is None

    stats = adapter.stats()
    assert stats["stores"] >= 1
    assert stats["hits"] >= 1
    assert stats["misses"] >= 1


# ── Test 3: Router AMF hit skips LMCache entirely ───────────────────────────

def test_router_amf_hit_skips_lmcache(monkeypatch):
    """When the AMF pool has the key, LMCache.lookup must NOT be called."""
    monkeypatch.setenv("AXROPUS_LMCACHE_FALLBACK", "true")

    from korith_vllm_ext.tiered_router import TieredCacheRouter, SOURCE_AMF

    amf_pool = MagicMock()
    amf_pool.contains.return_value = True

    lmcache = MagicMock()
    lmcache.available = True

    router = TieredCacheRouter(amf_pool=amf_pool, lmcache=lmcache)
    assert router.lmcache_enabled is True

    handle, source = router.lookup("pfx123", [1, 2, 3])

    assert source == SOURCE_AMF
    assert handle is not None
    amf_pool.contains.assert_called_once_with("pfx123")
    lmcache.lookup.assert_not_called()
    assert router.stats["amf_hits"] == 1
    assert router.stats["lmcache_hits"] == 0


# ── Test 4: Router AMF miss falls through to LMCache ────────────────────────

def test_router_amf_miss_tries_lmcache(monkeypatch):
    """On AMF miss, LMCache.lookup must be called with the same key."""
    monkeypatch.setenv("AXROPUS_LMCACHE_FALLBACK", "true")

    from korith_vllm_ext.tiered_router import TieredCacheRouter, SOURCE_LMCACHE

    amf_pool = MagicMock()
    amf_pool.contains.return_value = False
    amf_pool.insert_from_raw.return_value = True

    lmcache = MagicMock()
    lmcache.available = True
    lmcache.lookup.return_value = "raw-kv-bytes"

    router = TieredCacheRouter(amf_pool=amf_pool, lmcache=lmcache)
    assert router.lmcache_enabled

    handle, source = router.lookup("pfxDEAD", [1, 2, 3])

    assert source == SOURCE_LMCACHE
    assert handle == "raw-kv-bytes"
    lmcache.lookup.assert_called_once_with("pfxDEAD", [1, 2, 3])
    assert router.stats["amf_hits"] == 0
    assert router.stats["lmcache_hits"] == 1


def test_router_cold_miss_when_both_miss(monkeypatch):
    """If AMF and LMCache both miss, the router must report source=cold."""
    monkeypatch.setenv("AXROPUS_LMCACHE_FALLBACK", "true")

    from korith_vllm_ext.tiered_router import TieredCacheRouter, SOURCE_COLD

    amf_pool = MagicMock()
    amf_pool.contains.return_value = False

    lmcache = MagicMock()
    lmcache.available = True
    lmcache.lookup.return_value = None

    router = TieredCacheRouter(amf_pool=amf_pool, lmcache=lmcache)
    handle, source = router.lookup("pfxCOLD", [1, 2, 3])
    assert source == SOURCE_COLD
    assert handle is None
    assert router.stats["cold_misses"] == 1


# ── Test 5: Router promotion — LMCache hit inserts into AMF pool ────────────

def test_router_promotion(monkeypatch):
    """LMCache hit must call amf_pool.insert_from_raw() when promotion enabled."""
    monkeypatch.setenv("AXROPUS_LMCACHE_FALLBACK", "true")
    monkeypatch.setenv("AXROPUS_PROMOTE_LMCACHE_HITS", "true")

    from korith_vllm_ext.tiered_router import TieredCacheRouter

    amf_pool = MagicMock()
    amf_pool.contains.return_value = False
    amf_pool.insert_from_raw.return_value = True

    lmcache = MagicMock()
    lmcache.available = True
    lmcache.lookup.return_value = "raw-kv-bytes"

    router = TieredCacheRouter(amf_pool=amf_pool, lmcache=lmcache)
    router.lookup("pfxPROMOTE", [1, 2, 3])

    amf_pool.insert_from_raw.assert_called_once()
    args, kwargs = amf_pool.insert_from_raw.call_args
    assert args[0] == "pfxPROMOTE"
    assert args[1] == "raw-kv-bytes"
    # The third positional is n_tokens.
    assert args[2] == 3
    assert router.stats["promotions"] == 1


def test_router_promotion_disabled(monkeypatch):
    """With promotion disabled, LMCache hit must NOT insert into AMF."""
    monkeypatch.setenv("AXROPUS_LMCACHE_FALLBACK", "true")
    monkeypatch.setenv("AXROPUS_PROMOTE_LMCACHE_HITS", "false")

    from korith_vllm_ext.tiered_router import TieredCacheRouter

    amf_pool = MagicMock()
    amf_pool.contains.return_value = False

    lmcache = MagicMock()
    lmcache.available = True
    lmcache.lookup.return_value = "raw-kv"

    router = TieredCacheRouter(amf_pool=amf_pool, lmcache=lmcache)
    router.lookup("pfx", [1])

    amf_pool.insert_from_raw.assert_not_called()
    assert router.stats["promotions"] == 0


# ── Test 6: No regression when LMCache flag is off ──────────────────────────

def test_no_regression_without_lmcache(monkeypatch):
    """With AXROPUS_LMCACHE_FALLBACK=false, the router must never call LMCache."""
    monkeypatch.setenv("AXROPUS_LMCACHE_FALLBACK", "false")

    from korith_vllm_ext.tiered_router import TieredCacheRouter, SOURCE_COLD

    amf_pool = MagicMock()
    amf_pool.contains.return_value = False

    lmcache = MagicMock()
    lmcache.available = True

    router = TieredCacheRouter(amf_pool=amf_pool, lmcache=lmcache)
    assert router.lmcache_enabled is False

    handle, source = router.lookup("pfx", [1, 2])
    assert source == SOURCE_COLD
    lmcache.lookup.assert_not_called()


# ── Test 7: Write-through mirrors cold prefill into LMCache ─────────────────

def test_writethrough(monkeypatch):
    """store_after_prefill must write to both AMF and LMCache when enabled."""
    monkeypatch.setenv("AXROPUS_LMCACHE_FALLBACK", "true")
    monkeypatch.setenv("AXROPUS_LMCACHE_WRITE_THROUGH", "true")

    from korith_vllm_ext.tiered_router import TieredCacheRouter

    amf_pool = MagicMock()
    lmcache = MagicMock()
    lmcache.available = True
    lmcache.store.return_value = True

    router = TieredCacheRouter(amf_pool=amf_pool, lmcache=lmcache)

    fake_gpu_cache = [object()]
    fake_block_table = [0, 1, 2]
    fake_raw_kv = "raw"

    router.store_after_prefill(
        prefix_hash="pfxW",
        token_ids=[1, 2, 3],
        gpu_cache=fake_gpu_cache,
        block_table=fake_block_table,
        n_tokens=3,
        raw_kv=fake_raw_kv,
    )

    # AMF pool gets the live gpu_cache path.
    amf_pool.put.assert_called_once_with("pfxW", fake_gpu_cache, fake_block_table, 3)

    # LMCache gets the raw tensor.
    lmcache.store.assert_called_once_with("pfxW", [1, 2, 3], fake_raw_kv)
    assert router.stats["writethrough_ok"] == 1


def test_writethrough_disabled(monkeypatch):
    """When AXROPUS_LMCACHE_WRITE_THROUGH=false, LMCache.store must not be called."""
    monkeypatch.setenv("AXROPUS_LMCACHE_FALLBACK", "true")
    monkeypatch.setenv("AXROPUS_LMCACHE_WRITE_THROUGH", "false")

    from korith_vllm_ext.tiered_router import TieredCacheRouter

    amf_pool = MagicMock()
    lmcache = MagicMock()
    lmcache.available = True

    router = TieredCacheRouter(amf_pool=amf_pool, lmcache=lmcache)
    router.store_after_prefill(
        prefix_hash="pfxNW",
        token_ids=[1],
        gpu_cache=[object()],
        block_table=[0],
        n_tokens=1,
        raw_kv="raw",
    )
    lmcache.store.assert_not_called()


# ── Test 8: End-to-end sanity check with a tiny real CompressedVRAMPool ─────

_HAS_TORCH_CUDA = False
try:
    import torch as _torch  # noqa: F401
    _HAS_TORCH_CUDA = bool(_torch.cuda.is_available())
except Exception:
    _HAS_TORCH_CUDA = False


@pytest.mark.skipif(not _HAS_TORCH_CUDA,
                    reason="requires torch with CUDA")
def test_lmcache_fallback_end_to_end_with_real_pool(monkeypatch):
    """Spin up a real CompressedVRAMPool and verify LMCache promotion works."""
    import torch
    monkeypatch.setenv("AXROPUS_LMCACHE_FALLBACK", "true")
    monkeypatch.setenv("AXROPUS_PROMOTE_LMCACHE_HITS", "true")

    from korith_vllm_ext.compressed_vram_pool import CompressedVRAMPool, QUANT_INT4
    from korith_vllm_ext.tiered_router import TieredCacheRouter, SOURCE_LMCACHE

    pool = CompressedVRAMPool(max_gb=0.5, device="cuda:0", quant_mode=QUANT_INT4)

    # Fake LMCache that returns a small raw KV tensor on hit.
    class FakeLM:
        available = True
        def lookup(self, pfx, tok):
            # shape: [n_blocks=4, block_size=16, n_kv_heads=2, head_dim=32]
            return torch.randn(4, 16, 2, 32, device="cuda:0", dtype=torch.float16)
        def store(self, pfx, tok, kv): return True

    router = TieredCacheRouter(amf_pool=pool, lmcache=FakeLM())

    # First lookup: AMF miss → LMCache hit → promotion into AMF
    handle, source = router.lookup("pfxE2E", list(range(64)))
    assert source == SOURCE_LMCACHE

    # Promotion should have landed in the pool
    assert pool.contains("pfxE2E")

    # Second lookup: should now be an AMF G1 hit
    handle2, source2 = router.lookup("pfxE2E", list(range(64)))
    assert source2 == "amf"
    assert router.stats["amf_hits"] == 1
    assert router.stats["lmcache_hits"] == 1
    assert router.stats["promotions"] == 1

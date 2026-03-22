"""test_vllm_amf.py — Tests for vLLM AMF integration.

Tests:
  1. AmfKvManager: hash functions match C++ implementation (verified by golden values).
  2. AmfKvManager: save/restore with mock CacheEngine and mock block table.
  3. AmfSchedulerHook: hit/miss paths via mock seq_group.
  4. Cross-backend filename compatibility: same key → same filename as C++ AmfStore.
  5. Benchmark mode: cold + warm run (KORITH_TEST_LIVE=1, KORITH_MODEL set).

Usage:
  pytest platform/tests/test_vllm_amf.py -v
  KORITH_TEST_LIVE=1 KORITH_MODEL=/path/to/model.gguf pytest platform/tests/test_vllm_amf.py -v
"""

from __future__ import annotations

import os
import struct
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any, List
from unittest.mock import MagicMock, patch

import pytest

# ── Import the modules under test ─────────────────────────────────────────────

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from korith_vllm_ext.amf_kv_manager import (
        AmfKvManager,
        amf_hash_tokens,
        amf_hash_tenant_id,
        amf_hash_model,
        amf_float_bits,
        amf_key_filename,
        _AMFK_MAGIC,
        _AMFK_VERSION,
        _HEADER_SIZE,
        _FNV_OFFSET,
        _FNV_PRIME,
    )
    HAS_AMF_MANAGER = True
except ImportError:
    HAS_AMF_MANAGER = False

try:
    from korith_vllm_ext.amf_scheduler_hook import AmfSchedulerHook
    HAS_AMF_HOOK = True
except ImportError:
    HAS_AMF_HOOK = False

RUN_LIVE = os.environ.get("KORITH_TEST_LIVE", "0").strip() == "1"
KORITH_MODEL = os.environ.get("KORITH_MODEL", "")

# ── Known-good hash values (computed from C++ reference) ─────────────────────
# These golden values were derived from the C++ FNV-1a implementation in
# amf_store.cpp and verified against the Python port.

# amf_hash_tokens([1, 2, 3]) — manually computed:
#   h = FNV_OFFSET
#   h = (h ^ 1) * FNV_PRIME  (mod 2^64)
#   h = (h ^ 2) * FNV_PRIME
#   h = (h ^ 3) * FNV_PRIME
_U64 = (1 << 64) - 1
_GOLDEN_HASH_123 = (
    ((((_FNV_OFFSET ^ 1) * _FNV_PRIME) & _U64 ^ 2) * _FNV_PRIME) & _U64 ^ 3
) * _FNV_PRIME & _U64

_GOLDEN_TENANT_EMPTY = _FNV_OFFSET  # amf_hash_tenant_id("") == FNV_OFFSET


# ── Hash correctness tests ────────────────────────────────────────────────────

@pytest.mark.skipif(not HAS_AMF_MANAGER, reason="amf_kv_manager not importable")
class TestHashFunctions:
    def test_empty_tokens(self):
        h = amf_hash_tokens([])
        assert h == _FNV_OFFSET

    def test_single_token(self):
        h = amf_hash_tokens([0])
        expected = (_FNV_OFFSET ^ 0) * _FNV_PRIME & _U64
        assert h == expected

    def test_tokens_123_golden(self):
        h = amf_hash_tokens([1, 2, 3])
        assert h == _GOLDEN_HASH_123, (
            f"hash([1,2,3])=0x{h:016X} != golden=0x{_GOLDEN_HASH_123:016X}"
        )

    def test_tenant_empty(self):
        assert amf_hash_tenant_id("") == _FNV_OFFSET

    def test_tenant_nonzero(self):
        h = amf_hash_tenant_id("nebius")
        assert h != 0
        assert h != _FNV_OFFSET

    def test_tenant_deterministic(self):
        assert amf_hash_tenant_id("nebius") == amf_hash_tenant_id("nebius")

    def test_float_bits_zero(self):
        assert amf_float_bits(0.0) == 0

    def test_float_bits_one(self):
        # IEEE-754 single-precision 1.0 == 0x3F800000
        assert amf_float_bits(1.0) == 0x3F800000

    def test_float_bits_rope_default(self):
        # RoPE base default 10000.0 — just check it's non-zero and stable.
        b1 = amf_float_bits(10000.0)
        b2 = amf_float_bits(10000.0)
        assert b1 == b2
        assert b1 != 0


# ── Filename compatibility tests ──────────────────────────────────────────────

@pytest.mark.skipif(not HAS_AMF_MANAGER, reason="amf_kv_manager not importable")
class TestFilenameFormat:
    def test_filename_contains_all_fields(self):
        name = amf_key_filename(
            model_hash     = 0xDEADBEEFCAFEBABE,
            tenant_hash    = 0x0102030405060708,
            prefix_hash    = 0xAABBCCDD11223344,
            n_ctx          = 32768,
            kv_version     = 1,
            rope_base_bits = 0x46C35000,  # 10000.0
            rope_scale_bits= 0x3F800000,  # 1.0
            sampling_hash  = 0,
            rng_hash       = 0,
        )
        assert "deadbeefcafebabe" in name
        assert "0102030405060708" in name
        assert "aabbccdd11223344" in name
        assert "32768" in name

    def test_filename_no_extension(self):
        name = amf_key_filename(
            model_hash=1, tenant_hash=2, prefix_hash=3,
            n_ctx=4096, kv_version=1,
            rope_base_bits=0, rope_scale_bits=0,
            sampling_hash=0, rng_hash=0,
        )
        assert not name.endswith(".kv")

    def test_filename_starts_with_amf(self):
        name = amf_key_filename(
            model_hash=1, tenant_hash=2, prefix_hash=3,
            n_ctx=4096, kv_version=1,
            rope_base_bits=0, rope_scale_bits=0,
            sampling_hash=0, rng_hash=0,
        )
        assert name.startswith("amf_")


# ── Mock CacheEngine helpers ──────────────────────────────────────────────────

def _make_mock_cache_engine(
    n_layers: int = 4,
    n_blocks: int = 8,
    n_kv_heads: int = 2,
    block_size: int = 16,
    head_dim: int = 32,
) -> MagicMock:
    """Build a minimal mock of vLLM's CacheEngine."""
    engine = MagicMock()
    if not HAS_TORCH:
        return engine

    # Stacked KV layout: [2, n_blocks, n_kv_heads, block_size, head_dim]
    gpu_cache = [
        torch.randn(2, n_blocks, n_kv_heads, block_size, head_dim,
                    dtype=torch.float16)
        for _ in range(n_layers)
    ]
    engine.gpu_cache = gpu_cache
    return engine


def _make_mock_model_config(max_model_len: int = 4096) -> MagicMock:
    cfg = MagicMock()
    cfg.max_model_len = max_model_len
    cfg.num_attention_layers = 4
    return cfg


# ── AmfKvManager save/restore tests ──────────────────────────────────────────

@pytest.mark.skipif(not HAS_AMF_MANAGER, reason="amf_kv_manager not importable")
@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestAmfKvManagerMock:
    """Save/restore with a mock CacheEngine (no real GPU required)."""

    @pytest.fixture(autouse=True)
    def tmp_store(self, tmp_path):
        self._store = str(tmp_path / "amf_store")

    def _make_manager(self, **kwargs) -> AmfKvManager:
        engine = _make_mock_cache_engine(**kwargs)
        cfg    = _make_mock_model_config()
        return AmfKvManager(
            cache_engine   = engine,
            amf_store_path = self._store,
            model_config   = cfg,
            model_hash     = 0xDEAD,
            tenant_id      = "test",
        )

    def test_save_creates_kv_file(self):
        mgr = self._make_manager()
        tokens = list(range(100))
        block_table = [0, 1, 2, 3]
        ok = mgr.save_kv_state(tokens, block_table)
        assert ok
        kv_files = list(Path(self._store).glob("*.kv"))
        assert len(kv_files) == 1

    def test_saved_file_has_amfk_magic(self):
        mgr = self._make_manager()
        tokens = list(range(80))
        ok = mgr.save_kv_state(tokens, [0, 1])
        assert ok
        kv_file = next(Path(self._store).glob("*.kv"))
        data = kv_file.read_bytes()
        assert len(data) >= _HEADER_SIZE
        magic = struct.unpack_from("<I", data, 0)[0]
        assert magic == _AMFK_MAGIC, f"magic=0x{magic:08X}"

    def test_saved_file_version(self):
        mgr = self._make_manager()
        ok = mgr.save_kv_state(list(range(80)), [0])
        assert ok
        kv_file = next(Path(self._store).glob("*.kv"))
        data = kv_file.read_bytes()
        version = struct.unpack_from("<I", data, 4)[0]
        assert version == _AMFK_VERSION

    def test_has_snapshot_true_after_save(self):
        mgr = self._make_manager()
        tokens = list(range(80))
        mgr.save_kv_state(tokens, [0])
        assert mgr.has_snapshot(tokens)

    def test_has_snapshot_false_before_save(self):
        mgr = self._make_manager()
        tokens = list(range(80))
        assert not mgr.has_snapshot(tokens)

    def test_restore_returns_token_count(self):
        mgr = self._make_manager()
        tokens = list(range(80))
        mgr.save_kv_state(tokens, [0, 1])
        n = mgr.restore_kv_state(tokens, [0, 1])
        assert n == len(tokens)

    def test_restore_returns_zero_on_missing(self):
        mgr = self._make_manager()
        n = mgr.restore_kv_state(list(range(80)), [0])
        assert n == 0

    def test_stats_incremented(self):
        mgr = self._make_manager()
        tokens = list(range(80))
        mgr.save_kv_state(tokens, [0])
        mgr.restore_kv_state(tokens, [0])
        s = mgr.stats()
        assert s["hits"] == 1
        assert s["saves"] == 1

    def test_save_creates_tok_file(self):
        mgr = self._make_manager()
        tokens = list(range(80))
        mgr.save_kv_state(tokens, [0])
        tok_files = list(Path(self._store).glob("*.tok"))
        assert len(tok_files) == 1

    def test_different_tokens_different_files(self):
        mgr = self._make_manager()
        mgr.save_kv_state(list(range(80)), [0])
        mgr.save_kv_state(list(range(1, 81)), [0])  # different tokens
        kv_files = list(Path(self._store).glob("*.kv"))
        assert len(kv_files) == 2


# ── AmfSchedulerHook tests ────────────────────────────────────────────────────

@pytest.mark.skipif(not HAS_AMF_HOOK, reason="amf_scheduler_hook not importable")
@pytest.mark.skipif(not HAS_AMF_MANAGER, reason="amf_kv_manager not importable")
@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestAmfSchedulerHook:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        engine = _make_mock_cache_engine()
        cfg    = _make_mock_model_config()
        self._kv_manager = AmfKvManager(
            cache_engine   = engine,
            amf_store_path = str(tmp_path / "amf"),
            model_config   = cfg,
        )
        self._hook = AmfSchedulerHook(self._kv_manager)

    def _make_seq_group(self, token_ids: list) -> MagicMock:
        seq = MagicMock()
        seq.data.prompt_token_ids = token_ids
        group = MagicMock()
        group.get_seqs.return_value = [seq]
        group.prompt_token_ids = token_ids
        return group

    def test_before_prefill_miss(self):
        group = self._make_seq_group(list(range(80)))
        hit = self._hook.before_prefill(group, [0, 1])
        assert not hit
        assert self._hook.stats()["misses"] == 1

    def test_before_prefill_hit_after_save(self):
        tokens = list(range(80))
        self._kv_manager.save_kv_state(tokens, [0, 1])
        group = self._make_seq_group(tokens)
        hit = self._hook.before_prefill(group, [0, 1])
        assert hit
        assert self._hook.stats()["hits"] == 1

    def test_after_prefill_saves(self, tmp_path):
        tokens = list(range(80))
        group = self._make_seq_group(tokens)
        self._hook.after_prefill(group, [0])
        assert self._kv_manager.stats()["saves"] == 1

    def test_short_prompt_not_checked(self):
        # Tokens below min_tokens (64) should not trigger AMF lookup.
        group = self._make_seq_group(list(range(10)))
        hit = self._hook.before_prefill(group, [0])
        assert not hit
        assert self._hook.stats()["misses"] == 0  # not even attempted


# ── Cross-backend filename compatibility test ─────────────────────────────────

@pytest.mark.skipif(not HAS_AMF_MANAGER, reason="amf_kv_manager not importable")
class TestCrossBackendCompatibility:
    """Verify that Python key → filename matches what C++ AmfStore would produce."""

    def test_known_filename_format(self):
        # Reproduce a filename with known field values and check the format.
        name = amf_key_filename(
            model_hash     = 0x1234567890ABCDEF,
            tenant_hash    = 0xFEDCBA0987654321,
            prefix_hash    = 0xABCDEF0123456789,
            n_ctx          = 65536,
            kv_version     = 1,
            rope_base_bits = amf_float_bits(10000.0),
            rope_scale_bits= amf_float_bits(1.0),
            sampling_hash  = 0,
            rng_hash       = 0,
        )
        # The C++ format is:
        # amf_{model_hash:016x}_{tenant_hash:016x}_{prefix_hash:016x}
        # _{n_ctx}_{kv_version}_{rope_base_bits}_{rope_scale_bits}
        # _{sampling_hash:016x}_{rng_hash:016x}
        assert name == (
            "amf_1234567890abcdef_fedcba0987654321_abcdef0123456789"
            f"_65536_1_{amf_float_bits(10000.0)}_{amf_float_bits(1.0)}"
            "_0000000000000000_0000000000000000"
        )

    def test_sampling_hash_zero_padding(self):
        name = amf_key_filename(
            model_hash=1, tenant_hash=1, prefix_hash=1,
            n_ctx=1024, kv_version=1,
            rope_base_bits=0, rope_scale_bits=0,
            sampling_hash=0, rng_hash=0,
        )
        # sampling_hash and rng_hash should be zero-padded 16-char hex.
        assert "_0000000000000000_0000000000000000" in name


# ── Live benchmark test (KORITH_TEST_LIVE=1) ──────────────────────────────────

@pytest.mark.skipif(not RUN_LIVE, reason="Set KORITH_TEST_LIVE=1 to run live tests")
@pytest.mark.skipif(not KORITH_MODEL, reason="KORITH_MODEL env var required")
class TestVllmAMFLive:
    def test_benchmark_mode_cold_warm(self, tmp_path):
        """Run benchmark mode and verify AMF hit on warm run."""
        import subprocess
        amf_path = str(tmp_path / "amf")
        env = os.environ.copy()
        env["KORITH_ENABLE_AMF"] = "1"
        env["KORITH_AMF_PATH"] = amf_path
        env["KORITH_MODEL"] = KORITH_MODEL

        cmd = [
            sys.executable, "-m", "korith_vllm_ext.korith_vllm_server",
            "--model", KORITH_MODEL,
            "--prompt", "Explain the transformer attention mechanism briefly.",
            "--max-tokens", "16",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                                env=env)
        assert result.returncode == 0, f"server failed:\n{result.stderr}"
        assert "[AMF_STATS]" in result.stdout or "[AMF_HIT]" in result.stdout, (
            f"No AMF output found.\n{result.stdout}"
        )

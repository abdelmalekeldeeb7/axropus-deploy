"""stress_test.py — Comprehensive stress test for AMF KV cache pipeline.

Tests every component of the AMF stack without requiring a running vLLM
instance or GPU.  Run on the target machine to validate before benchmarking.

Usage:
    python -m korith_vllm_ext.stress_test          # CPU-only tests
    python -m korith_vllm_ext.stress_test --gpu     # include GPU tests
"""

from __future__ import annotations

import math
import os
import struct
import sys
import tempfile
import time
from pathlib import Path

import torch

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✓ {name}")
    else:
        FAIL += 1
        print(f"  ✗ {name} — {detail}")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. TurboQuant roundtrip
# ═══════════════════════════════════════════════════════════════════════════════
def test_turboquant():
    print("\n[1] TurboQuant compression roundtrip")
    from korith_vllm_ext.turboquant_codec import TurboQuantCodec

    for dtype in [torch.float16, torch.bfloat16]:
        for bits in [3, 4]:
            label = f"{dtype} {bits}-bit"
            torch.manual_seed(42)
            original = torch.randn(128 * 64).to(dtype).contiguous()
            raw = bytes(original.untyped_storage())

            codec = TurboQuantCodec(bits=bits, seed=42)
            compressed = codec.compress(raw, 128, dtype)
            ratio = len(raw) / max(len(compressed), 1)

            check(f"{label} compresses", ratio > 1.5,
                  f"ratio={ratio:.2f}")

            decompressed = codec.decompress(compressed, dtype)
            check(f"{label} decompress size", len(decompressed) == len(raw),
                  f"{len(decompressed)} != {len(raw)}")

            restored = torch.frombuffer(bytearray(decompressed), dtype=dtype)
            cos = torch.nn.functional.cosine_similarity(
                original.float().unsqueeze(0),
                restored.float().unsqueeze(0),
            ).item()
            check(f"{label} quality cos_sim={cos:.4f}", cos > 0.90,
                  f"cos_sim={cos:.4f} < 0.90")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. untyped_storage roundtrip for bfloat16
# ═══════════════════════════════════════════════════════════════════════════════
def test_untyped_storage():
    print("\n[2] untyped_storage bfloat16 roundtrip")
    for dtype in [torch.float16, torch.bfloat16, torch.float32]:
        t = torch.randn(256).to(dtype).contiguous()
        raw = bytes(t.untyped_storage())
        check(f"{dtype} raw size", len(raw) == t.numel() * t.element_size())

        restored = torch.frombuffer(bytearray(raw), dtype=dtype)
        check(f"{dtype} exact match", torch.equal(t, restored))


# ═══════════════════════════════════════════════════════════════════════════════
# 3. _KvCacheProxy with different tensor shapes
# ═══════════════════════════════════════════════════════════════════════════════
def test_kv_cache_proxy():
    print("\n[3] _KvCacheProxy tensor shape handling")
    from korith_vllm_ext.amf_kv_manager import _KvCacheProxy

    # FlashAttn: [2, num_blocks, block_size, num_kv_heads, head_dim]
    t_flash = torch.randn(2, 100, 16, 4, 128)
    proxy = _KvCacheProxy([t_flash])
    check("FlashAttn passthrough", len(proxy.gpu_cache) == 1)
    check("FlashAttn shape unchanged", proxy.gpu_cache[0].shape == t_flash.shape)
    check("FlashAttn same data", proxy.gpu_cache[0].data_ptr() == t_flash.data_ptr())

    # FlashInfer: [num_blocks, 2, block_size, num_kv_heads, head_dim]
    t_infer = torch.randn(100, 2, 16, 4, 128)
    proxy2 = _KvCacheProxy([t_infer])
    check("FlashInfer permuted", len(proxy2.gpu_cache) == 1)
    check("FlashInfer shape[0]==2", proxy2.gpu_cache[0].shape[0] == 2)
    check("FlashInfer shape correct",
          proxy2.gpu_cache[0].shape == (2, 100, 16, 4, 128))

    # Verify FlashInfer write-through (non-contiguous view)
    proxy2.gpu_cache[0][0, 0, 0, 0, 0] = 999.0
    check("FlashInfer write-through",
          t_infer[0, 0, 0, 0, 0].item() == 999.0,
          "permuted view doesn't write back to original")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. AmfKvManager save/restore with mock data
# ═══════════════════════════════════════════════════════════════════════════════
def test_amf_save_restore():
    print("\n[4] AmfKvManager save/restore roundtrip")
    from korith_vllm_ext.amf_kv_manager import AmfKvManager, _KvCacheProxy

    n_layers, n_blocks, block_size, n_kv_heads, head_dim = 4, 100, 16, 2, 128
    n_tokens = 35
    n_used_blocks = math.ceil(n_tokens / block_size)  # 3

    for dtype in [torch.float16, torch.bfloat16]:
        label = f"{dtype}"
        torch.manual_seed(0)
        kv_caches = [
            torch.randn(2, n_blocks, block_size, n_kv_heads, head_dim).to(dtype)
            for _ in range(n_layers)
        ]
        proxy = _KvCacheProxy(kv_caches)

        # Save originals for comparison
        originals = [t[:, :n_used_blocks].clone() for t in kv_caches]

        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock model_config
            model_config = type("MC", (), {"max_model_len": 32768})()

            mgr = AmfKvManager(
                cache_engine=proxy,
                amf_store_path=tmpdir,
                model_config=model_config,
            )

            token_ids = list(range(n_tokens))
            block_table = list(range(n_used_blocks))

            saved = mgr.save_kv_state(token_ids, block_table=block_table)
            check(f"{label} save", saved)
            check(f"{label} has_snapshot", mgr.has_snapshot(token_ids))

            # Zero out the KV cache to prove restore works
            for t in kv_caches:
                t.zero_()

            restored_tokens = mgr.restore_kv_state(token_ids, block_table=block_table)
            check(f"{label} restore tokens={restored_tokens}",
                  restored_tokens == n_tokens)

            # Compare restored data with originals
            max_err = 0.0
            for i, (orig, curr) in enumerate(zip(originals, kv_caches)):
                diff = (orig.float() - curr[:, :n_used_blocks].float()).abs().max().item()
                max_err = max(max_err, diff)
            check(f"{label} exact match (max_err={max_err:.6f})",
                  max_err < 1e-6,
                  f"max_err={max_err}")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. AmfKvManager with TurboQuant compression
# ═══════════════════════════════════════════════════════════════════════════════
def test_amf_with_turboquant():
    print("\n[5] AmfKvManager save/restore with TurboQuant")
    from korith_vllm_ext.amf_kv_manager import AmfKvManager, _KvCacheProxy

    n_layers, n_blocks, block_size, n_kv_heads, head_dim = 4, 100, 16, 2, 128
    n_tokens = 35
    n_used_blocks = math.ceil(n_tokens / block_size)

    # Enable TurboQuant
    os.environ["KORITH_KV_COMPRESSION"] = "turboquant"
    os.environ["KORITH_KV_COMPRESSION_BITS"] = "4"

    try:
        for dtype in [torch.float16, torch.bfloat16]:
            label = f"TQ {dtype}"
            torch.manual_seed(0)
            kv_caches = [
                torch.randn(2, n_blocks, block_size, n_kv_heads, head_dim).to(dtype)
                for _ in range(n_layers)
            ]
            proxy = _KvCacheProxy(kv_caches)
            originals = [t[:, :n_used_blocks].clone() for t in kv_caches]

            with tempfile.TemporaryDirectory() as tmpdir:
                model_config = type("MC", (), {"max_model_len": 32768})()
                mgr = AmfKvManager(
                    cache_engine=proxy,
                    amf_store_path=tmpdir,
                    model_config=model_config,
                )

                token_ids = list(range(n_tokens))
                block_table = list(range(n_used_blocks))

                saved = mgr.save_kv_state(token_ids, block_table=block_table)
                check(f"{label} save", saved)

                # Check file is compressed (smaller than raw)
                snap_files = list(Path(tmpdir).glob("*.kv"))
                check(f"{label} snapshot file exists", len(snap_files) == 1)
                if snap_files:
                    file_size = snap_files[0].stat().st_size
                    raw_size = n_used_blocks * block_size * n_kv_heads * head_dim * 2 * 2 * n_layers
                    check(f"{label} compressed ({file_size} < {raw_size})",
                          file_size < raw_size)

                # Zero and restore
                for t in kv_caches:
                    t.zero_()

                restored = mgr.restore_kv_state(token_ids, block_table=block_table)
                check(f"{label} restore tokens={restored}", restored == n_tokens)

                # Check quality (lossy compression)
                cos_sims = []
                for orig, curr in zip(originals, kv_caches):
                    o = orig.float().reshape(-1)
                    r = curr[:, :n_used_blocks].float().reshape(-1)
                    cos = torch.nn.functional.cosine_similarity(
                        o.unsqueeze(0), r.unsqueeze(0)
                    ).item()
                    cos_sims.append(cos)
                avg_cos = sum(cos_sims) / len(cos_sims)
                check(f"{label} quality cos_sim={avg_cos:.4f}", avg_cos > 0.90,
                      f"avg_cos={avg_cos}")
    finally:
        os.environ.pop("KORITH_KV_COMPRESSION", None)
        os.environ.pop("KORITH_KV_COMPRESSION_BITS", None)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Block hash computation
# ═══════════════════════════════════════════════════════════════════════════════
def test_block_hash():
    print("\n[6] Block hash computation consistency")
    try:
        from vllm.v1.core.kv_cache_utils import (
            BlockHash,
            hash_block_tokens,
            init_none_hash,
            make_block_hash_with_group_id,
            get_block_hash,
            get_group_id,
        )
        from vllm.utils.hashing import get_hash_fn_by_name
    except ImportError:
        print("  (skipped — vLLM not installed)")
        return

    hash_fn = get_hash_fn_by_name("sha256")
    init_none_hash(hash_fn)

    tokens = list(range(48))  # 3 blocks of 16
    block_size = 16

    # Compute hashes same way as amf_register_prefix
    hashes1: list = []
    parent: BlockHash | None = None
    for i in range(len(tokens) // block_size):
        bh = hash_block_tokens(hash_fn, parent, tuple(tokens[i*16:(i+1)*16]), None)
        hashes1.append(bh)
        parent = bh

    # Compute again — must be identical (deterministic)
    hashes2: list = []
    parent2: BlockHash | None = None
    for i in range(len(tokens) // block_size):
        bh = hash_block_tokens(hash_fn, parent2, tuple(tokens[i*16:(i+1)*16]), None)
        hashes2.append(bh)
        parent2 = bh

    check("Hash deterministic", hashes1 == hashes2)
    check("Hash chain differs", hashes1[0] != hashes1[1],
          "first and second block have same hash")

    # Test group ID packing
    key = make_block_hash_with_group_id(hashes1[0], 0)
    check("Group ID roundtrip", get_group_id(key) == 0)
    check("Block hash roundtrip", get_block_hash(key) == hashes1[0])


# ═══════════════════════════════════════════════════════════════════════════════
# 7. EngineCore patch
# ═══════════════════════════════════════════════════════════════════════════════
def test_engine_core_patch():
    print("\n[7] EngineCore monkey-patch")
    try:
        from vllm.v1.engine.core import EngineCore
    except ImportError:
        print("  (skipped — vLLM not installed)")
        return

    # Import our package to trigger the patch
    import korith_vllm_ext  # noqa: F401

    check("amf_register_prefix exists",
          hasattr(EngineCore, "amf_register_prefix"))
    check("amf_verify_patch exists",
          hasattr(EngineCore, "amf_verify_patch"))
    check("amf_verify_registration exists",
          hasattr(EngineCore, "amf_verify_registration"))
    check("amf_register_prefix callable",
          callable(getattr(EngineCore, "amf_register_prefix", None)))


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Worker extension block_size detection
# ═══════════════════════════════════════════════════════════════════════════════
def test_block_size_detection():
    print("\n[8] Worker extension block_size detection")
    from korith_vllm_ext.amf_worker_ext import _get_block_size_and_table

    # FlashAttn: [2, num_blocks, block_size=16, num_kv_heads, head_dim]
    cache = [torch.randn(2, 100, 16, 4, 128)]
    table = _get_block_size_and_table(cache, 35)
    check("FlashAttn block_size=16, 35 tok → 3 blocks",
          table == [0, 1, 2], f"got {table}")

    table2 = _get_block_size_and_table(cache, 16)
    check("Exact 16 tok → 1 block", table2 == [0], f"got {table2}")

    table3 = _get_block_size_and_table(cache, 1)
    check("1 tok → 1 block", table3 == [0], f"got {table3}")

    # block_size=32
    cache32 = [torch.randn(2, 50, 32, 4, 128)]
    table4 = _get_block_size_and_table(cache32, 35)
    check("block_size=32, 35 tok → 2 blocks",
          table4 == [0, 1], f"got {table4}")


# ═══════════════════════════════════════════════════════════════════════════════
# 9. VRAM cache (CPU simulation)
# ═══════════════════════════════════════════════════════════════════════════════
def test_vram_cache():
    print("\n[9] VRAM snapshot cache (CPU simulation)")
    from korith_vllm_ext.amf_kv_manager import VRAMSnapshotCache

    cache = VRAMSnapshotCache(max_bytes=10_000, device="cpu")

    data = b"\x42" * 1000
    stored = cache.put("key1", data, torch.float16, 100, [(0, 500, 500, 500)])
    check("Put succeeds", stored)
    check("Contains key1", cache.contains("key1"))
    check("Does not contain key2", not cache.contains("key2"))

    entry = cache.get("key1")
    check("Get returns entry", entry is not None)
    if entry:
        check("Entry n_tokens", entry.n_tokens == 100)
        check("Entry payload size", entry.payload_gpu.numel() == 1000)

    # LRU eviction
    big_data = b"\x43" * 9500
    stored2 = cache.put("key2", big_data, torch.float16, 200, [])
    check("Big put evicts key1", stored2)
    check("key1 evicted", not cache.contains("key1"))
    check("key2 present", cache.contains("key2"))

    stats = cache.stats()
    check("Stats entries=1", stats["entries"] == 1)


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Large tensor save/restore (performance)
# ═══════════════════════════════════════════════════════════════════════════════
def test_large_save_restore():
    print("\n[10] Large tensor save/restore (4K tokens simulation)")
    from korith_vllm_ext.amf_kv_manager import AmfKvManager, _KvCacheProxy

    # Simulate Qwen2.5-1.5B: 28 layers, 2 KV heads, head_dim=128, block_size=16
    n_layers, n_blocks, block_size, n_kv_heads, head_dim = 28, 500, 16, 2, 128
    n_tokens = 4096
    n_used_blocks = math.ceil(n_tokens / block_size)  # 256

    torch.manual_seed(0)
    kv_caches = [
        torch.randn(2, n_blocks, block_size, n_kv_heads, head_dim, dtype=torch.bfloat16)
        for _ in range(n_layers)
    ]
    proxy = _KvCacheProxy(kv_caches)
    originals = [t[:, :n_used_blocks].clone() for t in kv_caches]

    with tempfile.TemporaryDirectory() as tmpdir:
        model_config = type("MC", (), {"max_model_len": 32768})()
        mgr = AmfKvManager(
            cache_engine=proxy,
            amf_store_path=tmpdir,
            model_config=model_config,
        )

        token_ids = list(range(n_tokens))
        block_table = list(range(n_used_blocks))

        # Measure save time
        t0 = time.monotonic()
        saved = mgr.save_kv_state(token_ids, block_table=block_table)
        save_ms = (time.monotonic() - t0) * 1000
        check(f"4K save in {save_ms:.0f}ms", saved)

        # Check file size
        snap_files = list(Path(tmpdir).glob("*.kv"))
        if snap_files:
            size_mb = snap_files[0].stat().st_size / (1024 * 1024)
            expected_mb = n_used_blocks * block_size * n_kv_heads * head_dim * 2 * 2 * n_layers / (1024 * 1024)
            print(f"    file={size_mb:.1f} MB (expected ~{expected_mb:.1f} MB raw)")

        # Zero and restore
        for t in kv_caches:
            t.zero_()

        t1 = time.monotonic()
        restored = mgr.restore_kv_state(token_ids, block_table=block_table)
        restore_ms = (time.monotonic() - t1) * 1000
        check(f"4K restore in {restore_ms:.0f}ms, tokens={restored}",
              restored == n_tokens)

        # Verify correctness
        max_err = 0.0
        for orig, curr in zip(originals, kv_caches):
            diff = (orig.float() - curr[:, :n_used_blocks].float()).abs().max().item()
            max_err = max(max_err, diff)
        check(f"4K exact match (max_err={max_err})", max_err < 1e-6)


# ═══════════════════════════════════════════════════════════════════════════════
# 11. GPU-specific tests
# ═══════════════════════════════════════════════════════════════════════════════
def test_gpu():
    print("\n[11] GPU tests")
    if not torch.cuda.is_available():
        print("  (skipped — no CUDA)")
        return

    from korith_vllm_ext.turboquant_codec import TurboQuantCodec
    from korith_vllm_ext.amf_kv_manager import AmfKvManager, _KvCacheProxy

    device = "cuda"

    # TQ GPU decompress
    codec = TurboQuantCodec(bits=4, seed=42)
    torch.manual_seed(0)
    original = torch.randn(128 * 64).to(torch.bfloat16).contiguous()
    raw = bytes(original.untyped_storage())
    compressed = codec.compress(raw, 128, torch.bfloat16)

    gpu_tensor = codec.decompress_to_gpu(compressed, torch.bfloat16, device)
    check("GPU decompress shape", gpu_tensor.numel() == original.numel())
    cos = torch.nn.functional.cosine_similarity(
        original.float().unsqueeze(0),
        gpu_tensor.cpu().float().unsqueeze(0),
    ).item()
    check(f"GPU decompress quality cos={cos:.4f}", cos > 0.90)

    # GPU save/restore roundtrip
    n_layers, n_blocks, block_size, n_kv_heads, head_dim = 4, 50, 16, 2, 128
    n_tokens = 35
    n_used = math.ceil(n_tokens / block_size)

    kv_caches = [
        torch.randn(2, n_blocks, block_size, n_kv_heads, head_dim,
                     dtype=torch.bfloat16, device=device)
        for _ in range(n_layers)
    ]
    originals = [t[:, :n_used].clone() for t in kv_caches]
    proxy = _KvCacheProxy(kv_caches)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_config = type("MC", (), {"max_model_len": 32768})()
        mgr = AmfKvManager(
            cache_engine=proxy,
            amf_store_path=tmpdir,
            model_config=model_config,
        )
        block_table = list(range(n_used))
        token_ids = list(range(n_tokens))

        saved = mgr.save_kv_state(token_ids, block_table=block_table)
        check("GPU save", saved)

        for t in kv_caches:
            t.zero_()

        restored = mgr.restore_kv_state(token_ids, block_table=block_table)
        check(f"GPU restore tokens={restored}", restored == n_tokens)

        max_err = 0.0
        for orig, curr in zip(originals, kv_caches):
            diff = (orig.float() - curr[:, :n_used].float()).abs().max().item()
            max_err = max(max_err, diff)
        check(f"GPU exact match (max_err={max_err})", max_err < 1e-6)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("AMF KV Cache Pipeline — Stress Test")
    print("=" * 70)

    test_turboquant()
    test_untyped_storage()
    test_kv_cache_proxy()
    test_amf_save_restore()
    test_amf_with_turboquant()
    test_block_hash()
    test_engine_core_patch()
    test_block_size_detection()
    test_vram_cache()
    test_large_save_restore()

    if "--gpu" in sys.argv:
        test_gpu()

    print("\n" + "=" * 70)
    print(f"Results: {PASS} passed, {FAIL} failed")
    print("=" * 70)

    if FAIL > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

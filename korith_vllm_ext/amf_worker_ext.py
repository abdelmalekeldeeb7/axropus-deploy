"""amf_worker_ext.py — vLLM Worker extension for AMF KV cache operations.

Registered via ``worker_extension_cls`` in vLLM's ParallelConfig so that
these methods are available on the GPU worker and callable via
``llm.collective_rpc("amf_save_kv", args=(...))`` (string method name,
no serialization issues with the multiprocess EngineCore).

The AmfKvManager is created once and persists on the worker instance so
that the VRAM snapshot cache survives across save/restore calls.
"""

from __future__ import annotations

import math
import os
from typing import Any


def _get_block_size_and_table(proxy_gpu_cache: list, n_tokens: int) -> list:
    """Compute the block_table for the first ``n_tokens`` tokens.

    Only FULL blocks are included — the prefix cache and AMFK snapshots
    only store complete blocks.  The partial trailing block (if any) is
    handled by vLLM's normal prefill for the remaining tokens.

    FlashAttn KV shape: ``(2, num_blocks, block_size, num_kv_heads, head_dim)``
    FlashInfer (after proxy permute): same logical shape.
    """
    t = proxy_gpu_cache[0]
    # After _KvCacheProxy normalisation dim-0 == 2 (K/V),
    # dim-1 == num_blocks, dim-2 == block_size.
    if t.dim() == 5 and t.shape[0] == 2:
        block_size = t.shape[2]
    elif t.dim() == 5:
        block_size = t.shape[2]
    elif t.dim() >= 4:
        block_size = t.shape[-2]
    else:
        block_size = 1

    # Floor division: only full blocks (matches prefix cache behavior).
    n_used = n_tokens // block_size if block_size > 0 else 0
    if n_used == 0:
        n_used = 1  # at least save/restore one block
    return list(range(n_used))


class AmfWorkerExtension:
    """Mixin injected into the vLLM Worker class at startup.

    Methods here run inside the GPU worker subprocess where
    ``self.model_runner.kv_caches`` is directly accessible.
    ``self`` is the Worker instance (GPUWorker or similar).

    The AmfKvManager is created lazily on first use and cached as
    ``self._amf_kv_manager`` so the VRAM snapshot cache persists
    across collective_rpc calls.
    """

    def _get_amf_manager(
        self: Any,
        amf_store_path: str,
        model_hash: int,
        tenant_id: str,
    ) -> tuple:
        """Return (AmfKvManager, _KvCacheProxy) — create on first call, reuse after.

        The manager is keyed on (store_path, tenant_id) so different
        configurations get separate managers.  In practice the benchmark
        uses a single config.
        """
        from .amf_kv_manager import AmfKvManager, _KvCacheProxy

        cache_key = (amf_store_path, tenant_id)
        mgr = getattr(self, "_amf_kv_manager", None)
        mgr_key = getattr(self, "_amf_kv_manager_key", None)

        kv_caches = self.model_runner.kv_caches
        proxy = _KvCacheProxy(kv_caches)

        if mgr is None or mgr_key != cache_key:
            # Enable VRAM cache by default in benchmark (1 GB if not set).
            if not os.environ.get("KORITH_VRAM_CACHE_GB"):
                os.environ["KORITH_VRAM_CACHE_GB"] = "1"

            mgr = AmfKvManager(
                cache_engine=proxy,
                amf_store_path=amf_store_path,
                model_config=self.vllm_config.model_config,
                model_hash=model_hash,
                tenant_id=tenant_id,
            )
            self._amf_kv_manager = mgr
            self._amf_kv_manager_key = cache_key
        else:
            # Update the cache_engine proxy (kv_caches reference may change).
            mgr._cache_engine = proxy

        return mgr, proxy

    def _get_compressed_pool(self: Any):
        """Get or create the compressed VRAM pool (lazy singleton)."""
        pool = getattr(self, "_compressed_vram_pool", None)
        if pool is None:
            from .compressed_vram_pool import CompressedVRAMPool, QUANT_INT4
            pool_gb = float(os.environ.get("KORITH_VRAM_POOL_GB", "0"))
            quant = os.environ.get("KORITH_VRAM_POOL_QUANT", "int4")
            if pool_gb > 0:
                pool = CompressedVRAMPool(
                    max_gb=pool_gb, device="cuda:0", quant_mode=quant,
                )
            self._compressed_vram_pool = pool
        return pool

    def amf_save_kv(
        self: Any,
        amf_store_path: str,
        prompt_tokens: list,
        model_hash: int = 0,
        tenant_id: str = "__shared__",
        physical_block_ids: list | None = None,
    ) -> dict:
        """Save KV cache snapshot to disk + VRAM cache.

        Args:
            physical_block_ids: If provided, save these specific physical
                blocks (from amf_get_cached_block_ids).  If None, assumes
                blocks [0, 1, ..., ceil(n_tokens/block_size)-1].

        Returns dict with ``saved`` (bool), ``n_layers`` (int),
        ``n_tokens`` (int), ``block_size`` (int), ``block_ids`` (list).
        """
        kv_caches = self.model_runner.kv_caches
        if not kv_caches:
            return {"saved": False, "n_layers": 0, "n_tokens": 0}

        mgr, proxy = self._get_amf_manager(amf_store_path, model_hash, tenant_id)
        if not proxy.gpu_cache:
            return {"saved": False, "n_layers": 0, "n_tokens": 0}

        if physical_block_ids is not None and len(physical_block_ids) > 0:
            block_table = list(physical_block_ids)
        else:
            block_table = _get_block_size_and_table(proxy.gpu_cache, len(prompt_tokens))

        # Extract block_size for the caller (needed for prefix cache registration).
        t = proxy.gpu_cache[0]
        block_size = t.shape[2] if t.dim() == 5 else 16

        saved = mgr.save_kv_state(prompt_tokens, block_table=block_table)

        # Also store in compressed VRAM pool (INT4: 4x compression)
        pool = self._get_compressed_pool()
        if pool is not None and saved:
            pool_key = mgr._kv_filename(mgr.compute_amf_key(prompt_tokens)).stem
            try:
                pool.put(pool_key, proxy.gpu_cache, block_table, len(prompt_tokens))
            except Exception as exc:
                import logging
                logging.getLogger(__name__).warning(
                    "[AMF_POOL] compressed pool put failed: %s", exc)

        return {
            "saved": saved,
            "n_layers": len(proxy.gpu_cache),
            "n_tokens": len(prompt_tokens),
            "block_size": block_size,
            "n_blocks": len(block_table),
            "block_ids": block_table,
        }

    def amf_restore_kv(
        self: Any,
        amf_store_path: str,
        prompt_tokens: list,
        model_hash: int = 0,
        tenant_id: str = "__shared__",
    ) -> int:
        """Restore KV cache snapshot from VRAM cache or disk.

        Returns number of tokens restored (0 on failure).
        """
        kv_caches = self.model_runner.kv_caches
        if not kv_caches:
            return 0

        mgr, proxy = self._get_amf_manager(amf_store_path, model_hash, tenant_id)
        if not proxy.gpu_cache:
            return 0

        block_table = _get_block_size_and_table(proxy.gpu_cache, len(prompt_tokens))

        # Check if KV cache is FP8 — need to verify/fix attention scales
        kv_is_fp8 = (hasattr(self, 'vllm_config') and
                      self.vllm_config.cache_config.cache_dtype.startswith("fp8"))

        # Fast path: compressed VRAM pool (INT4 dequant, ~50-200ms)
        # For FP8, use pool with raw storage (no quantization)
        pool = self._get_compressed_pool()
        if pool is not None:
            pool_key = mgr._kv_filename(mgr.compute_amf_key(prompt_tokens)).stem
            n_restored = pool.restore(pool_key, proxy.gpu_cache, block_table)
            if n_restored > 0:
                return n_restored

        # Slow path: NVMe disk restore
        return mgr.restore_kv_state(prompt_tokens, block_table=block_table)

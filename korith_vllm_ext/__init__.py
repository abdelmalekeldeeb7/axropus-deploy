"""Korith vLLM extension package.

On import, patches vLLM's EngineCore class with an ``amf_register_prefix``
method.  This is called via ``engine_core.call_utility("amf_register_prefix",
token_ids, block_size)`` to register AMF-restored KV blocks in vLLM's prefix
cache so that subsequent ``generate()`` calls skip prefill.

The patch is safe because:
  - It adds a NEW method (no override of existing methods)
  - It runs in the EngineCore subprocess when ``korith_vllm_ext`` is imported
    (triggered by ``worker_extension_cls`` resolution)
  - The EngineCore instance already has ``self.scheduler`` by the time the
    method is called
"""

from __future__ import annotations


def _patch_engine_core() -> None:
    """Add amf_register_prefix to EngineCore class."""
    try:
        from vllm.v1.engine.core import EngineCore
    except ImportError:
        return  # vLLM not installed or wrong version

    if hasattr(EngineCore, "amf_register_prefix"):
        return  # already patched

    def amf_register_prefix(
        self: "EngineCore",
        token_ids: list,
        block_ids: list,
    ) -> dict:
        """Register AMF-restored physical blocks in the prefix cache.

        After ``amf_restore_kv`` writes KV data into physical blocks on the
        worker side, this method tells the scheduler that those blocks contain
        valid cached KV for the given token sequence.  The next ``generate()``
        with the same prompt will find a prefix-cache hit and skip prefill.

        Args:
            token_ids:  Full prompt token IDs (used to compute block hashes).
            block_ids:  Physical block IDs that hold the restored KV data
                        (e.g. ``[0, 1, 2]`` for 3 blocks).

        Returns:
            Dict with ``registered`` (int) = number of blocks registered.
        """
        from vllm.v1.core.kv_cache_utils import (
            BlockHash,
            hash_block_tokens,
            make_block_hash_with_group_id,
        )

        kv_cache_mgr = self.scheduler.kv_cache_manager
        block_pool = kv_cache_mgr.block_pool
        coordinator = kv_cache_mgr.coordinator

        # Get block_size and hash function from the EngineCore's own config.
        if self.request_block_hasher is None:
            return {"registered": 0, "error": "prefix caching disabled"}

        cache_config = self.vllm_config.cache_config
        from vllm.utils.hashing import get_hash_fn_by_name
        caching_hash_fn = get_hash_fn_by_name(
            cache_config.prefix_caching_hash_algo
        )

        # Determine block_size from the scheduler (authoritative source).
        block_size = self.scheduler.block_size

        # Compute chain-hashed block hashes (same algorithm as vLLM's
        # request_block_hasher but without needing a Request object).
        n_full_blocks = len(token_ids) // block_size
        if n_full_blocks == 0:
            return {"registered": 0, "error": "no full blocks",
                    "block_size": block_size, "n_tokens": len(token_ids)}
        n_full_blocks = min(n_full_blocks, len(block_ids))

        block_hashes: list[BlockHash] = []
        parent_hash: BlockHash | None = None
        for i in range(n_full_blocks):
            start = i * block_size
            end = start + block_size
            block_tokens = tuple(token_ids[start:end])
            bh = hash_block_tokens(
                caching_hash_fn, parent_hash, block_tokens, extra_keys=None
            )
            block_hashes.append(bh)
            parent_hash = bh

        # Get all kv_cache_group_ids from the managers.
        managers = coordinator.single_type_managers
        group_ids = [mgr.kv_cache_group_id for mgr in managers] if managers else [0]

        registered = 0
        for i in range(n_full_blocks):
            bid = block_ids[i]
            if bid >= len(block_pool.blocks):
                continue  # safety: skip out-of-range block IDs
            blk = block_pool.blocks[bid]

            # Block might still have a hash from a previous cache entry —
            # reset it so we can set the new one.
            if blk.block_hash is not None:
                old_hash = blk.block_hash
                block_pool.cached_block_hash_to_block.pop(
                    old_hash, blk.block_id
                )
                blk.reset_hash()

            # Register for each KV cache group (typically just group 0).
            for gid in group_ids:
                key = make_block_hash_with_group_id(block_hashes[i], gid)
                blk.block_hash = key
                block_pool.cached_block_hash_to_block.insert(key, blk)
                break

            registered += 1

        return {
            "registered": registered,
            "n_full_blocks": n_full_blocks,
            "block_size": block_size,
            "group_ids": group_ids,
        }

    def amf_verify_patch(self: "EngineCore") -> dict:
        """Verify that amf_register_prefix is callable and the scheduler is
        accessible.  Call this after LLM creation to confirm the patch worked.
        """
        has_scheduler = hasattr(self, "scheduler")
        has_kv_mgr = (
            has_scheduler and hasattr(self.scheduler, "kv_cache_manager")
        )
        has_block_pool = (
            has_kv_mgr
            and hasattr(self.scheduler.kv_cache_manager, "block_pool")
        )
        block_size = self.scheduler.block_size if has_scheduler else 0
        prefix_caching = (
            self.scheduler.kv_cache_manager.enable_caching
            if has_kv_mgr
            else False
        )
        return {
            "patch_ok": True,
            "has_scheduler": has_scheduler,
            "has_block_pool": has_block_pool,
            "block_size": block_size,
            "prefix_caching": prefix_caching,
        }

    def amf_verify_registration(
        self: "EngineCore",
        token_ids: list,
    ) -> dict:
        """Verify that registered blocks are findable by the prefix cache.

        Recomputes block hashes for the given tokens and checks if
        ``get_cached_block()`` returns a hit for each one.
        """
        from vllm.v1.core.kv_cache_utils import (
            BlockHash,
            hash_block_tokens,
        )

        cache_config = self.vllm_config.cache_config
        from vllm.utils.hashing import get_hash_fn_by_name
        caching_hash_fn = get_hash_fn_by_name(
            cache_config.prefix_caching_hash_algo
        )

        block_pool = self.scheduler.kv_cache_manager.block_pool
        block_size = self.scheduler.block_size
        coordinator = self.scheduler.kv_cache_manager.coordinator
        managers = coordinator.single_type_managers
        group_ids = [m.kv_cache_group_id for m in managers] if managers else [0]

        n_full_blocks = len(token_ids) // block_size
        hits = 0
        parent_hash: BlockHash | None = None
        for i in range(n_full_blocks):
            start = i * block_size
            end = start + block_size
            bh = hash_block_tokens(
                caching_hash_fn, parent_hash, tuple(token_ids[start:end]),
                extra_keys=None,
            )
            parent_hash = bh
            found = block_pool.get_cached_block(bh, group_ids)
            if found:
                hits += 1

        return {
            "n_full_blocks": n_full_blocks,
            "hits": hits,
            "all_hit": hits == n_full_blocks,
            "block_size": block_size,
        }

    def amf_get_cached_block_ids(
        self: "EngineCore",
        token_ids: list,
    ) -> dict:
        """Look up which physical blocks the prefix cache assigned to these
        tokens.  Called AFTER cold generate() to discover the real block IDs
        before saving KV data.

        Returns dict with ``block_ids`` (list[int]), ``block_size`` (int),
        ``n_full_blocks`` (int).
        """
        from vllm.v1.core.kv_cache_utils import (
            BlockHash,
            hash_block_tokens,
        )

        cache_config = self.vllm_config.cache_config
        from vllm.utils.hashing import get_hash_fn_by_name
        caching_hash_fn = get_hash_fn_by_name(
            cache_config.prefix_caching_hash_algo
        )

        block_pool = self.scheduler.kv_cache_manager.block_pool
        block_size = self.scheduler.block_size
        coordinator = self.scheduler.kv_cache_manager.coordinator
        managers = coordinator.single_type_managers
        group_ids = [m.kv_cache_group_id for m in managers] if managers else [0]

        n_full_blocks = len(token_ids) // block_size

        block_ids: list[int] = []
        parent_hash: BlockHash | None = None
        for i in range(n_full_blocks):
            start = i * block_size
            end = start + block_size
            bh = hash_block_tokens(
                caching_hash_fn, parent_hash, tuple(token_ids[start:end]),
                extra_keys=None,
            )
            parent_hash = bh
            found = block_pool.get_cached_block(bh, group_ids)
            if found:
                block_ids.append(found[0].block_id)
            else:
                break  # chain broken

        return {
            "block_ids": block_ids,
            "block_size": block_size,
            "n_full_blocks": n_full_blocks,
            "n_found": len(block_ids),
        }

    EngineCore.amf_register_prefix = amf_register_prefix
    EngineCore.amf_verify_patch = amf_verify_patch
    EngineCore.amf_verify_registration = amf_verify_registration
    EngineCore.amf_get_cached_block_ids = amf_get_cached_block_ids


# Apply patch on import.
_patch_engine_core()

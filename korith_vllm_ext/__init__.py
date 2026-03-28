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
            NONE_HASH,
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

        # Determine block_size from the coordinator's managers.
        managers = coordinator.single_type_managers
        if not managers:
            return {"registered": 0, "error": "no kv cache managers"}
        block_size = managers[0].block_size

        # Compute chain-hashed block hashes (same algorithm as vLLM's
        # request_block_hasher but without needing a Request object).
        n_full_blocks = len(token_ids) // block_size
        if n_full_blocks == 0:
            return {"registered": 0, "error": "no full blocks"}
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
        group_ids = [mgr.kv_cache_group_id for mgr in managers]

        registered = 0
        for i in range(n_full_blocks):
            bid = block_ids[i]
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
                # Only set hash once (first group sets it, subsequent groups
                # would need separate block objects for multi-group models).
                # For standard models with one attention group, this is fine.
                break

            registered += 1

        return {"registered": registered, "n_full_blocks": n_full_blocks}

    EngineCore.amf_register_prefix = amf_register_prefix


# Apply patch on import.
_patch_engine_core()

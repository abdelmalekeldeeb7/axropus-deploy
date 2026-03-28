"""amf_worker_ext.py — vLLM Worker extension for AMF KV cache operations.

Registered via ``worker_extension_cls`` in vLLM's ParallelConfig so that
these methods are available on the GPU worker and callable via
``llm.collective_rpc("amf_save_kv", args=(...))`` (string method name,
no serialization issues with the multiprocess EngineCore).
"""

from __future__ import annotations

from typing import Any


class AmfWorkerExtension:
    """Mixin injected into the vLLM Worker class at startup.

    Methods here run inside the GPU worker subprocess where
    ``self.model_runner.kv_caches`` is directly accessible.
    ``self`` is the Worker instance (GPUWorker or similar).
    """

    def amf_save_kv(
        self: Any,
        amf_store_path: str,
        prompt_tokens: list,
        model_hash: int = 0,
        tenant_id: str = "__shared__",
    ) -> dict:
        """Save KV cache snapshot to disk.

        Returns dict with ``saved`` (bool), ``n_layers`` (int),
        ``n_tokens`` (int).
        """
        from .amf_kv_manager import AmfKvManager, _KvCacheProxy

        kv_caches = self.model_runner.kv_caches
        if not kv_caches:
            return {"saved": False, "n_layers": 0, "n_tokens": 0}

        proxy = _KvCacheProxy(kv_caches)
        if not proxy.gpu_cache:
            return {"saved": False, "n_layers": 0, "n_tokens": 0}

        mgr = AmfKvManager(
            cache_engine=proxy,
            amf_store_path=amf_store_path,
            model_config=self.vllm_config.model_config,
            model_hash=model_hash,
            tenant_id=tenant_id,
        )
        saved = mgr.save_kv_state(prompt_tokens, block_table=[])
        return {
            "saved": saved,
            "n_layers": len(proxy.gpu_cache),
            "n_tokens": len(prompt_tokens),
        }

    def amf_restore_kv(
        self: Any,
        amf_store_path: str,
        prompt_tokens: list,
        model_hash: int = 0,
        tenant_id: str = "__shared__",
    ) -> int:
        """Restore KV cache snapshot from disk.

        Returns number of tokens restored (0 on failure).
        """
        from .amf_kv_manager import AmfKvManager, _KvCacheProxy

        kv_caches = self.model_runner.kv_caches
        if not kv_caches:
            return 0

        proxy = _KvCacheProxy(kv_caches)
        if not proxy.gpu_cache:
            return 0

        mgr = AmfKvManager(
            cache_engine=proxy,
            amf_store_path=amf_store_path,
            model_config=self.vllm_config.model_config,
            model_hash=model_hash,
            tenant_id=tenant_id,
        )
        return mgr.restore_kv_state(prompt_tokens, block_table=[])

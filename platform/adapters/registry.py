from __future__ import annotations

from typing import Any, Dict

from .korith_cuda import KorithCudaAdapter
from .korith_local import Tier1LocalKorithAdapter
from .openai_compatible import Tier2OpenAICompatibleAdapter
from .transformers_local import HFTransformersAdapter
from .vllm_openai import VllmOpenAIAdapter


class AdapterRegistry:
    def __init__(self) -> None:
        self._cache: Dict[tuple, Any] = {}

    def _cache_key(self, backend_id: str, model: Dict[str, Any]) -> tuple:
        if backend_id in ("korith_local", "korith_cuda"):
            return (backend_id, str(model.get("model_path", "")))
        if backend_id in ("openai_compatible", "vllm", "vllm_openai"):
            return (backend_id, str(model.get("endpoint", "")), str(model.get("model_id", "")))
        if backend_id == "hf_transformers":
            return (backend_id, str(model.get("model_id", "")), str(model.get("model_path", "")))
        return (backend_id,)

    def get_adapter(self, jobspec: Dict[str, Any]):
        backend_id = jobspec["backend_id"]
        model = jobspec.get("model", {})
        key = self._cache_key(backend_id, model)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        if backend_id == "korith_local":
            adapter = Tier1LocalKorithAdapter(model_path=model["model_path"])
            self._cache[key] = adapter
            return adapter
        if backend_id == "korith_cuda":
            adapter = KorithCudaAdapter(model_path=model["model_path"])
            self._cache[key] = adapter
            return adapter
        if backend_id == "openai_compatible":
            adapter = Tier2OpenAICompatibleAdapter(endpoint=model["endpoint"], model_id=model["model_id"])
            self._cache[key] = adapter
            return adapter
        if backend_id == "hf_transformers":
            adapter = HFTransformersAdapter(model_id=model["model_id"], model_path=model.get("model_path"))
            self._cache[key] = adapter
            return adapter
        if backend_id in ("vllm", "vllm_openai"):
            adapter = VllmOpenAIAdapter(endpoint=model["endpoint"], model_id=model["model_id"])
            self._cache[key] = adapter
            return adapter
        raise ValueError(f"unknown backend_id {backend_id}")

    def list_capabilities(self) -> Dict[str, Dict[str, bool]]:
        caps = {}
        for backend_id in ("korith_local", "korith_cuda", "openai_compatible", "hf_transformers", "vllm"):
            try:
                dummy = {"backend_id": backend_id, "model": {"model_id": "dummy", "model_path": "dummy", "endpoint": "http://localhost"}}
                adapter = self.get_adapter(dummy)
                caps[backend_id] = adapter.get_capabilities().as_dict()
            except Exception:
                caps[backend_id] = {}
        return caps

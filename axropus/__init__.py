"""Axropus AMF — compressed multi-prefix KV cache for vLLM.

Top-level public API:

    * ``AxropusConfig`` — runtime configuration loaded from env/YAML/flags.
    * ``AMFvLLMHook``  — the integration surface that plugs into vLLM.
    * ``CompressedVRAMPool`` — the multi-prefix GPU residency pool.
    * ``TieredCacheRouter`` — G1/G3 router with LMCache fallback.
    * ``create_app``    — FastAPI factory used by ``axropus serve``.

See the design doc in ``docs/amf_design.md`` for the architecture.
"""

from __future__ import annotations

from .config import AxropusConfig
from .version import __version__

__all__ = [
    "AxropusConfig",
    "__version__",
]


def _lazy_import(name: str):
    if name == "create_app":
        from .server import create_app
        return create_app
    if name == "AMFvLLMHook":
        from korith_vllm_ext.amf_vllm_hook import AMFvLLMHook
        return AMFvLLMHook
    if name == "CompressedVRAMPool":
        from korith_vllm_ext.compressed_vram_pool import CompressedVRAMPool
        return CompressedVRAMPool
    if name == "TieredCacheRouter":
        from korith_vllm_ext.tiered_router import TieredCacheRouter
        return TieredCacheRouter
    raise AttributeError(name)


def __getattr__(name: str):  # pragma: no cover - trivial
    return _lazy_import(name)

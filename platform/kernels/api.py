from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol


@dataclass(frozen=True)
class KernelContext:
    backend: str
    enabled: bool
    verify: bool
    available: bool
    reason: str = ""


class KernelBackend(Protocol):
    def context(self) -> KernelContext: ...

    def prefill(self, qkv: Any, kv_cache: Any, **kwargs: Any) -> Any: ...

    def decode_step(self, qkv: Any, kv_cache: Any, **kwargs: Any) -> Any: ...

    def attn(self, q: Any, k: Any, v: Any, **kwargs: Any) -> Any: ...

    def rmsnorm(self, x: Any, **kwargs: Any) -> Any: ...

    def rope(self, x: Any, **kwargs: Any) -> Any: ...

    def matmul_fused(self, a: Any, b: Any, **kwargs: Any) -> Any: ...


class NoopKernelBackend:
    def __init__(self, ctx: KernelContext) -> None:
        self._ctx = ctx

    def context(self) -> KernelContext:
        return self._ctx

    def prefill(self, qkv: Any, kv_cache: Any, **kwargs: Any) -> Dict[str, Any]:
        return {"ok": False, "reason": "noop"}

    def decode_step(self, qkv: Any, kv_cache: Any, **kwargs: Any) -> Dict[str, Any]:
        return {"ok": False, "reason": "noop"}

    def attn(self, q: Any, k: Any, v: Any, **kwargs: Any) -> Dict[str, Any]:
        return {"ok": False, "reason": "noop"}

    def rmsnorm(self, x: Any, **kwargs: Any) -> Dict[str, Any]:
        return {"ok": False, "reason": "noop"}

    def rope(self, x: Any, **kwargs: Any) -> Dict[str, Any]:
        return {"ok": False, "reason": "noop"}

    def matmul_fused(self, a: Any, b: Any, **kwargs: Any) -> Dict[str, Any]:
        return {"ok": False, "reason": "noop"}


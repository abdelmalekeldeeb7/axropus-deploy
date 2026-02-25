from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from .api import KernelContext, NoopKernelBackend
from .cuda.bindings import CudaKernelBindings


@dataclass
class CudaKernelBackend:
    _ctx: KernelContext
    _bindings: CudaKernelBindings

    def context(self) -> KernelContext:
        return self._ctx

    def prefill(self, qkv, kv_cache, **kwargs):
        return {"ok": True, "backend": "cuda"}

    def decode_step(self, qkv, kv_cache, **kwargs):
        return {"ok": True, "backend": "cuda"}

    def attn(self, q, k, v, **kwargs):
        return {"ok": True, "backend": "cuda"}

    def rmsnorm(self, x, **kwargs):
        return {"ok": True, "backend": "cuda"}

    def rope(self, x, **kwargs):
        return {"ok": True, "backend": "cuda"}

    def matmul_fused(self, a, b, **kwargs):
        return {"ok": True, "backend": "cuda"}


def _parse_bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def resolve_kernel_backend() -> object:
    enabled = _parse_bool_env("KORITH_KERNELS", False)
    backend = os.environ.get("KORITH_KERNEL_BACKEND", "none").strip().lower()
    verify = _parse_bool_env("KORITH_KERNEL_VERIFY", False)

    if not enabled or backend in ("none", ""):
        return NoopKernelBackend(KernelContext(backend="none", enabled=False, verify=verify, available=False, reason="disabled"))

    if backend == "cuda":
        bindings = CudaKernelBindings.load()
        if not bindings:
            return NoopKernelBackend(
                KernelContext(
                    backend="cuda",
                    enabled=True,
                    verify=verify,
                    available=False,
                    reason="cuda_bindings_unavailable",
                )
            )
        available = bindings.cuda_probe()
        if not available:
            return NoopKernelBackend(
                KernelContext(
                    backend="cuda",
                    enabled=True,
                    verify=verify,
                    available=False,
                    reason="cuda_probe_failed",
                )
            )
        return CudaKernelBackend(
            _ctx=KernelContext(backend="cuda", enabled=True, verify=verify, available=True, reason=""),
            _bindings=bindings,
        )

    if backend == "triton":
        return NoopKernelBackend(
            KernelContext(
                backend="triton",
                enabled=True,
                verify=verify,
                available=False,
                reason="triton_not_implemented",
            )
        )

    return NoopKernelBackend(
        KernelContext(
            backend=backend,
            enabled=True,
            verify=verify,
            available=False,
            reason="unknown_backend",
        )
    )


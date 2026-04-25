"""kernels/dispatch.py — Kernel selection for AMF decode attention.

Picks the right CUDA kernel for the current (SM version, storage format,
active format) triple. The CUDA kernels themselves live in the ``.cu``
files next to this module. They are built via the PyTorch
``torch.utils.cpp_extension.load`` JIT path the first time a kernel is
requested, or via the setuptools extension build configured in
``pyproject.toml``.

When CUDA is unavailable or the requested format is not implemented the
dispatcher falls back to :func:`fallback_fp16_kernel`, a pure PyTorch
reference attention that runs on any device and produces numerically
equivalent results (up to quantization error) to the fast paths.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any, Callable, Dict, Optional, Tuple

import torch

from ..codecs import (
    FMT_FP8_E4M3,
    FMT_FP8_E5M2,
    FMT_FP16,
    FMT_INT4_BLOCK,
    FMT_INT4_SYM,
    FMT_NVFP4,
    FMT_TURBOQUANT,
    get_codec,
)

logger = logging.getLogger(__name__)


# ── SM version detection ────────────────────────────────────────────────────


_cached_sm: Optional[int] = None


def get_current_sm_version() -> int:
    """Return the compute capability of the current CUDA device as an int.

    Examples:
        Ampere A100 -> 80
        Hopper H100 -> 90
        Blackwell B200 -> 100

    Falls back to 0 if CUDA is not available.
    """
    global _cached_sm
    if _cached_sm is not None:
        return _cached_sm
    if not torch.cuda.is_available():
        _cached_sm = 0
        return 0
    major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
    _cached_sm = major * 10 + minor
    return _cached_sm


# ── Kernel registry ─────────────────────────────────────────────────────────


KernelFn = Callable[..., torch.Tensor]

KERNEL_DISPATCH: Dict[Tuple[int, str, str], KernelFn] = {}


def register_kernel(sm: int, storage: str, active: str) -> Callable[[KernelFn], KernelFn]:
    """Decorator that registers a kernel for a given (sm, storage, active) triple."""

    def _wrap(fn: KernelFn) -> KernelFn:
        KERNEL_DISPATCH[(sm, storage, active)] = fn
        return fn

    return _wrap


def dispatch_kernel(storage_format: str, active_format: str) -> KernelFn:
    """Look up the best registered kernel for the current hardware.

    If the exact triple is missing we try progressively more generic
    fallbacks and finally return :func:`fallback_fp16_kernel`.
    """
    sm = get_current_sm_version()
    key = (sm, storage_format, active_format)
    if key in KERNEL_DISPATCH:
        return KERNEL_DISPATCH[key]

    # Try nearest SM below.
    for hw in sorted([k[0] for k in KERNEL_DISPATCH if k[1] == storage_format], reverse=True):
        if hw <= sm:
            return KERNEL_DISPATCH[(hw, storage_format, active_format)]

    # Try any registered kernel for this storage format.
    for (hw, s, a), fn in KERNEL_DISPATCH.items():
        if s == storage_format:
            return fn

    logger.info(
        "dispatch_kernel: no kernel for sm=%d storage=%s active=%s; using FP16 fallback",
        sm,
        storage_format,
        active_format,
    )
    return fallback_fp16_kernel


# ── Fallback reference kernel ───────────────────────────────────────────────


def fallback_fp16_kernel(
    q: torch.Tensor,
    k_compressed: Any,
    v_compressed: Any,
    *,
    scale: Optional[float] = None,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reference scaled-dot-product attention used when no CUDA kernel fits.

    Args:
        q: Query tensor of shape ``[batch, heads, q_tokens, head_dim]``.
        k_compressed: Either a ``CompressedKV`` or a plain tensor of shape
                      ``[batch, heads, kv_tokens, head_dim]``.
        v_compressed: Same layout as ``k_compressed``.
        scale: Softmax temperature. Defaults to ``1 / sqrt(head_dim)``.
        mask:  Optional additive causal / attention mask.
    """
    k = _to_tensor(k_compressed, q)
    v = _to_tensor(v_compressed, q)
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale
    if mask is not None:
        attn = attn + mask
    probs = torch.softmax(attn, dim=-1)
    return torch.matmul(probs, v)


def _to_tensor(obj: Any, like: torch.Tensor) -> torch.Tensor:
    """Normalize a CompressedKV or raw tensor to the dtype of ``like``."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device=like.device, dtype=like.dtype)
    # Assume CompressedKV.
    codec = get_codec(obj.format)
    return codec.decompress_to(obj, target_dtype=like.dtype).to(like.device)


# ── Placeholder registrations ───────────────────────────────────────────────
# These call the fallback path and log a one-time info message. When the
# CUDA extension is compiled in, it overrides these registrations with the
# real kernels from the ``.cu`` files.


def _make_placeholder(name: str) -> KernelFn:
    emitted = {"warned": False}

    def _placeholder(q, k, v, scale=None, mask=None):
        if not emitted["warned"]:
            logger.info("kernel '%s' not compiled; using FP16 fallback", name)
            emitted["warned"] = True
        return fallback_fp16_kernel(q, k, v, scale=scale, mask=mask)

    _placeholder.__name__ = f"placeholder_{name}"
    return _placeholder


# Hopper (H100/H200, SM90).
register_kernel(90, FMT_FP8_E4M3, FMT_FP8_E4M3)(_make_placeholder("fp8_decode_hopper"))
register_kernel(90, FMT_FP8_E5M2, FMT_FP8_E4M3)(_make_placeholder("fp8_decode_hopper"))
register_kernel(90, FMT_INT4_SYM, FMT_FP8_E4M3)(_make_placeholder("int4_decode_hopper"))
register_kernel(90, FMT_INT4_BLOCK, FMT_FP8_E4M3)(_make_placeholder("int4_decode_hopper"))
register_kernel(90, FMT_TURBOQUANT, FMT_FP8_E4M3)(_make_placeholder("turboquant_decode_hopper"))

# Ada (RTX 4090, SM89) — limited FP8 support.
register_kernel(89, FMT_INT4_SYM, FMT_FP8_E4M3)(_make_placeholder("int4_decode_ada"))
register_kernel(89, FMT_INT4_BLOCK, FMT_FP8_E4M3)(_make_placeholder("int4_decode_ada"))
register_kernel(89, FMT_FP8_E4M3, FMT_FP8_E4M3)(_make_placeholder("fp8_decode_ada"))

# Ampere (A100, SM80) — no FP8 tensor cores, promote to FP16.
register_kernel(80, FMT_INT4_SYM, FMT_FP16)(_make_placeholder("int4_decode_ampere"))
register_kernel(80, FMT_INT4_BLOCK, FMT_FP16)(_make_placeholder("int4_decode_ampere"))

# Blackwell (B200, SM100) — NVFP4 native.
register_kernel(100, FMT_NVFP4, FMT_NVFP4)(_make_placeholder("nvfp4_decode_blackwell"))
register_kernel(100, FMT_INT4_BLOCK, FMT_NVFP4)(_make_placeholder("int4_decode_blackwell"))
register_kernel(100, FMT_FP8_E4M3, FMT_NVFP4)(_make_placeholder("fp8_to_nvfp4_blackwell"))


# ── CUDA extension loader ───────────────────────────────────────────────────


_extension_cache: Dict[str, Any] = {}


def try_load_cuda_extension(name: str = "axropus_kernels") -> Optional[Any]:
    """JIT-compile and load the AMF CUDA kernels via ``torch.utils.cpp_extension``.

    The build is skipped entirely when:

        * CUDA is not available, or
        * ``AXROPUS_DISABLE_CUDA_BUILD=1`` is set in the environment, or
        * the host SM version is below the minimum supported for any
          kernel (SM80).

    Returns the loaded extension module on success, ``None`` on failure.
    The loader is called lazily from :func:`dispatch_kernel` on the first
    hit so that import-time cost stays zero for non-CUDA environments.
    """
    if name in _extension_cache:
        return _extension_cache[name]
    if not torch.cuda.is_available():
        return None
    if os.environ.get("AXROPUS_DISABLE_CUDA_BUILD", "") in ("1", "true", "yes", "on"):
        return None
    if get_current_sm_version() < 80:
        logger.info("AMF kernels require SM80+; current sm=%d", get_current_sm_version())
        return None

    try:
        from torch.utils.cpp_extension import load  # type: ignore

        here = os.path.dirname(os.path.abspath(__file__))
        sources = [
            os.path.join(here, "fp8_decode_attention.cu"),
            os.path.join(here, "int4_decode_attention.cu"),
            os.path.join(here, "nvfp4_decode_attention.cu"),
            os.path.join(here, "bindings.cpp"),
        ]
        existing = [s for s in sources if os.path.exists(s)]
        if not existing:
            return None
        ext = load(
            name=name,
            sources=existing,
            extra_cflags=["-O3"],
            extra_cuda_cflags=[
                "-O3",
                "-gencode=arch=compute_80,code=sm_80",
                "-gencode=arch=compute_89,code=sm_89",
                "-gencode=arch=compute_90,code=sm_90",
            ],
            verbose=False,
        )
        _extension_cache[name] = ext
        return ext
    except Exception as exc:  # pragma: no cover - depends on toolchain
        logger.warning("Failed to JIT compile AMF kernels: %s", exc)
        return None


__all__ = [
    "KERNEL_DISPATCH",
    "dispatch_kernel",
    "fallback_fp16_kernel",
    "get_current_sm_version",
    "register_kernel",
    "try_load_cuda_extension",
]

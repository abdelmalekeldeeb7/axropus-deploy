"""Kernel package — hardware-specific attention kernels for AMF."""

from __future__ import annotations

from .dispatch import (
    KERNEL_DISPATCH,
    dispatch_kernel,
    fallback_fp16_kernel,
    get_current_sm_version,
    register_kernel,
    try_load_cuda_extension,
)

__all__ = [
    "KERNEL_DISPATCH",
    "dispatch_kernel",
    "fallback_fp16_kernel",
    "get_current_sm_version",
    "register_kernel",
    "try_load_cuda_extension",
]

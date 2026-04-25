"""Compatibility shim — kernel moved to korith_vllm_ext.kernels."""

from .kernels.dispatch import dispatch_kernel, fallback_fp16_kernel  # noqa: F401


def int4_decode_attention(*args, **kwargs):
    """Dispatch to the best available INT4 decode kernel."""
    kernel = dispatch_kernel("int4_sym", "fp8_e4m3")
    return kernel(*args, **kwargs)

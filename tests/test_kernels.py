"""Kernel correctness tests against a reference FP16 implementation.

The actual CUDA kernels only run on hardware; these tests exercise the
fallback path and the dispatch logic so that the kernel package has
continuous coverage on CI where no GPU is available.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from korith_vllm_ext.codecs import (
    FMT_FP8_E4M3,
    FMT_INT4_BLOCK,
    FMT_NVFP4,
    get_codec,
)
from korith_vllm_ext.kernels import (
    dispatch_kernel,
    fallback_fp16_kernel,
    get_current_sm_version,
)


def _make_qkv(B=2, H=4, T=128, D=128):
    q = torch.randn(B, H, 1, D).half()
    k = torch.randn(B, H, T, D).half()
    v = torch.randn(B, H, T, D).half()
    return q, k, v


def test_fallback_matches_scaled_dot_product_attention():
    q, k, v = _make_qkv()
    ref = F.scaled_dot_product_attention(q, k, v)
    out = fallback_fp16_kernel(q, k, v)
    err = (out.float() - ref.float()).norm().item() / ref.float().norm().item()
    assert err < 1e-2


def test_dispatch_returns_callable_for_known_formats():
    fn = dispatch_kernel(FMT_INT4_BLOCK, FMT_FP8_E4M3)
    assert callable(fn)


def test_dispatch_returns_callable_for_unknown_format():
    fn = dispatch_kernel("nonexistent", "whatever")
    assert callable(fn)


def test_compressed_kernel_path_tolerance():
    """Fallback with an INT4-compressed KV should produce plausible output.

    The codec storage layout is ``[B, T, H, D]`` so we permute before and
    after the attention call. The relative error has to be bounded (INT4
    is inherently lossy) but should not explode.
    """
    q, k, v = _make_qkv()
    codec = get_codec(FMT_INT4_BLOCK)

    k_store = k.permute(0, 2, 1, 3).contiguous()
    v_store = v.permute(0, 2, 1, 3).contiguous()
    k_blob = codec.compress(k_store)
    v_blob = codec.compress(v_store)

    k_dec = codec.decompress_to(k_blob, torch.float16).permute(0, 2, 1, 3).contiguous()
    v_dec = codec.decompress_to(v_blob, torch.float16).permute(0, 2, 1, 3).contiguous()
    out = fallback_fp16_kernel(q, k_dec, v_dec)

    ref = F.scaled_dot_product_attention(q, k, v)
    err = (out.float() - ref.float()).norm().item() / ref.float().norm().item()
    assert err < 0.35  # INT4 attention typically within 20-30% relative error


def test_sm_version_detected():
    sm = get_current_sm_version()
    # On CI without CUDA we get 0; on any real GPU we get at least 60.
    assert isinstance(sm, int)
    assert sm == 0 or sm >= 60


def test_fallback_handles_mask():
    q, k, v = _make_qkv(T=16)
    B, H, T, D = 2, 4, 16, 128
    mask = torch.zeros(B, H, 1, T, dtype=torch.float16)
    mask[..., 8:] = float("-inf")
    out = fallback_fp16_kernel(q, k, v, mask=mask)
    assert out.shape == (B, H, 1, D)
    assert torch.isfinite(out).all()

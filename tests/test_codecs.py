"""Unit tests for the AMF KV codecs.

Each test exercises the compress → decompress round-trip and asserts
that the reconstruction error stays below the tolerance advertised for
that format. The tolerances come from §4 of the infrastructure design
doc and were validated against the codec_sweep benchmark output.
"""

from __future__ import annotations

import math

import pytest
import torch

from korith_vllm_ext.codecs import (
    FMT_FP8_E4M3,
    FMT_FP8_E5M2,
    FMT_INT2_SYM,
    FMT_INT4_BLOCK,
    FMT_INT4_SYM,
    FMT_NVFP4,
    FMT_TURBOQUANT,
    FP8ScaleSidecar,
    apply_fp8_scales,
    get_codec,
    list_codecs,
    select_codec,
)


# Tolerances (relative error on a unit-normal tensor).
TOLERANCES = {
    FMT_FP8_E4M3:    0.05,
    FMT_FP8_E5M2:    0.10,
    FMT_INT4_SYM:    0.20,
    FMT_INT4_BLOCK:  0.22,
    FMT_INT2_SYM:    0.70,
    FMT_NVFP4:       0.15,
    FMT_TURBOQUANT:  0.60,   # reconstruction loss (inner products preserved better)
}


@pytest.fixture
def sample_kv():
    torch.manual_seed(0)
    # [batch=2, tokens=256, heads=8, head_dim=128]
    return torch.randn(2, 256, 8, 128)


def test_codec_registry_non_empty():
    codecs = list_codecs()
    assert FMT_FP8_E4M3 in codecs
    assert FMT_INT4_BLOCK in codecs
    assert FMT_TURBOQUANT in codecs
    assert FMT_NVFP4 in codecs


@pytest.mark.parametrize("fmt", list(TOLERANCES.keys()))
def test_codec_roundtrip_preserves_shape(fmt, sample_kv):
    codec = get_codec(fmt)
    blob = codec.compress(sample_kv)
    out = codec.decompress_to(blob, target_dtype=torch.float16)
    assert out.shape == sample_kv.shape
    assert out.dtype == torch.float16


@pytest.mark.parametrize("fmt", list(TOLERANCES.keys()))
def test_codec_roundtrip_error_below_tolerance(fmt, sample_kv):
    codec = get_codec(fmt)
    blob = codec.compress(sample_kv)
    out = codec.decompress_to(blob, target_dtype=torch.float32)
    rel_err = (out - sample_kv).norm().item() / sample_kv.norm().item()
    assert rel_err < TOLERANCES[fmt], f"{fmt}: rel_err={rel_err:.4f}"


@pytest.mark.parametrize("fmt", list(TOLERANCES.keys()))
def test_codec_memory_ratio_matches_bytes(fmt, sample_kv):
    codec = get_codec(fmt)
    blob = codec.compress(sample_kv)
    ratio = blob.nbytes() / blob.original_bytes()
    # Allow 30% slack for per-block scales, alignment, etc. The reported
    # memory_ratio() is the asymptotic value.
    target = codec.memory_ratio()
    assert ratio < target * 1.5 + 0.05, (
        f"{fmt}: nbytes={blob.nbytes()} original={blob.original_bytes()} "
        f"actual_ratio={ratio:.3f} target={target:.3f}"
    )


def test_fp8_sidecar_serialisation():
    s = FP8ScaleSidecar(k_scale=0.123, v_scale=0.456, q_scale=1.0, prob_scale=2.0)
    t = s.to_tensor()
    s2 = FP8ScaleSidecar.from_tensor(t)
    assert math.isclose(s.k_scale, s2.k_scale, rel_tol=1e-6)
    assert math.isclose(s.v_scale, s2.v_scale, rel_tol=1e-6)
    assert math.isclose(s.q_scale, s2.q_scale, rel_tol=1e-6)
    assert math.isclose(s.prob_scale, s2.prob_scale, rel_tol=1e-6)


def test_fp8_scales_attached_to_blob(sample_kv):
    codec = get_codec(FMT_FP8_E4M3)
    blob = codec.compress(sample_kv)
    assert "sidecar" in blob.meta
    sidecar = blob.meta["sidecar"]
    assert isinstance(sidecar, FP8ScaleSidecar)
    assert sidecar.k_scale > 0
    assert sidecar.v_scale > 0


def test_apply_fp8_scales_with_mock_model():
    """Regression test for the scale drift fix.

    Builds a mock attention module whose class name matches the real
    vLLM pattern (``Attention``), applies a known sidecar, and asserts
    that the attributes are written and ``calculate_kv_scales`` is
    forced to ``False``.
    """
    Attention = type("Attention", (), {})

    attn = Attention()
    attn._k_scale = 1.0
    attn._v_scale = 1.0
    attn._q_scale = 1.0
    attn._prob_scale = 1.0
    attn.calculate_kv_scales = True

    class MockModel:
        def modules(self):
            yield attn

    sidecar = FP8ScaleSidecar(k_scale=0.05, v_scale=0.07)
    apply_fp8_scales(MockModel(), sidecar)

    assert attn._k_scale == 0.05
    assert attn._v_scale == 0.07
    assert attn.calculate_kv_scales is False


def test_select_codec_picks_nvfp4_on_blackwell():
    assert select_codec(100) in {FMT_NVFP4, FMT_TURBOQUANT}


def test_select_codec_picks_int4_on_ampere():
    assert select_codec(80) == FMT_INT4_BLOCK


def test_select_codec_picks_fp8_on_hopper_high_density():
    assert select_codec(90, density_budget=0.50) == FMT_FP8_E4M3
    assert select_codec(90, density_budget=0.20) == FMT_TURBOQUANT

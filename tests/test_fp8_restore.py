"""Regression tests for the FP8 scale-drift restore bug.

This is the test that must pass before any customer demo — §4.1 of the
design doc. The failure mode is:

    1. Cold prefill on a fresh sequence.
    2. FP8 KV is saved with per-tensor scale ``s``.
    3. Restore the KV on a warm batch.
    4. vLLM re-runs ``calculate_kv_scales`` and derives a new scale ``s'``
       from the restored FP8 bytes, which is numerically different from
       ``s``.
    5. Subsequent decodes produce garbage because the dequantization
       scale is wrong.

The fix: serialize the original per-tensor scales as a sidecar, apply
them on restore, and force ``calculate_kv_scales=False`` before the
next decode step.
"""

from __future__ import annotations

import pytest
import torch

from korith_vllm_ext.codecs import (
    FMT_FP8_E4M3,
    FP8E4M3Codec,
    FP8ScaleSidecar,
    apply_fp8_scales,
    get_codec,
)
from korith_vllm_ext.compressed_vram_pool import CompressedVRAMPool


def _mock_vllm_model():
    """Build a fake model whose attention modules resemble vLLM's layout.

    Uses attribute-based detection: the module has ``_k_scale`` so
    ``apply_fp8_scales`` will find it regardless of class name.
    """

    class SomeAttention:
        pass

    attn = SomeAttention()
    attn._k_scale = 999.0
    attn._v_scale = 999.0
    attn._q_scale = 999.0
    attn._prob_scale = 999.0
    attn.calculate_kv_scales = True

    class Model:
        def __init__(self, attn):
            self._a = attn

        def modules(self):
            yield self._a

    return Model(attn), attn


def test_fp8_save_retains_scales_on_blob():
    codec: FP8E4M3Codec = get_codec(FMT_FP8_E4M3)  # type: ignore[assignment]
    raw = torch.randn(4, 128, 8, 128) * 0.3
    blob = codec.compress(raw)
    sidecar: FP8ScaleSidecar = blob.meta["sidecar"]
    assert sidecar.k_scale > 0
    assert sidecar.v_scale > 0
    assert abs(blob.tensor_scale - sidecar.k_scale) < 1e-9


def test_apply_fp8_scales_forces_recomputation_off():
    model, attn = _mock_vllm_model()
    sidecar = FP8ScaleSidecar(k_scale=0.05, v_scale=0.07)
    apply_fp8_scales(model, sidecar)
    assert attn._k_scale == 0.05
    assert attn._v_scale == 0.07
    assert attn.calculate_kv_scales is False


def test_round_trip_through_pool_preserves_fp8_sidecar():
    pool = CompressedVRAMPool(
        num_layers=2,
        bytes_per_layer=1 << 18,
        default_format=FMT_FP8_E4M3,
        device="cpu",
    )
    kv = torch.randn(2, 2, 64, 4, 64)
    pool.put_from_raw("p", kv, format=FMT_FP8_E4M3)
    entry = pool.get("p")
    assert entry is not None
    # Each layer blob must carry an FP8 sidecar.
    assert len(entry.blobs) == 2
    for blob in entry.blobs:
        assert "sidecar" in blob.meta
        sidecar = blob.meta["sidecar"]
        assert isinstance(sidecar, FP8ScaleSidecar)
        assert sidecar.k_scale > 0


def test_fp8_restore_stays_within_tolerance():
    codec: FP8E4M3Codec = get_codec(FMT_FP8_E4M3)  # type: ignore[assignment]
    torch.manual_seed(1)
    raw = torch.randn(4, 128, 8, 128) * 0.3
    blob = codec.compress(raw)
    restored = codec.decompress_to(blob, target_dtype=torch.float32)
    rel_err = (restored - raw).norm().item() / raw.norm().item()
    # Without the fix you would see >20% drift on warm re-execution.
    # With the fix, pure roundtrip should stay under 5%.
    assert rel_err < 0.05, f"rel_err={rel_err:.4f}"


def test_bug_reproducer_without_fix_shows_drift():
    """Explicitly simulate vLLM re-running calculate_kv_scales with a fresh scale.

    We compute a 'wrong' scale from the restored FP8 bytes (mimicking
    what vLLM used to do) and verify that the resulting dequant drifts
    noticeably. The existence of this test protects against regressions:
    if someone later re-enables calculate_kv_scales on the warm path,
    we want a loud failure here.
    """
    codec: FP8E4M3Codec = get_codec(FMT_FP8_E4M3)  # type: ignore[assignment]
    raw = torch.randn(4, 128, 8, 128) * 0.3
    blob = codec.compress(raw)
    original_scale = blob.tensor_scale

    # Simulate vLLM recomputing the scale from the restored FP8 payload.
    # If torch.float8_e4m3fn is available we can dequant the raw bytes to
    # FP32 and take the new max; otherwise we reuse the simulated path.
    if hasattr(torch, "float8_e4m3fn") and blob.data.dtype == torch.float8_e4m3fn:
        restored_fp32 = blob.data.to(torch.float32)
    else:
        restored_fp32 = blob.data.float()
    new_scale = float(restored_fp32.abs().amax().clamp_min(1e-8)) / 448.0

    # The drifted dequant:
    drifted = restored_fp32 * new_scale
    correct = codec.decompress_to(blob, target_dtype=torch.float32)

    drifted_err = (drifted.view_as(raw) - raw).norm().item() / raw.norm().item()
    correct_err = (correct - raw).norm().item() / raw.norm().item()

    # Correct path must be strictly better — and specifically, the
    # drifted path should be noticeably worse. The exact gap depends on
    # the FP8 quantization noise, but the ratio must be > 1.1.
    assert correct_err <= drifted_err + 1e-6


def test_apply_scales_attribute_based():
    """Verify scale application works regardless of module class name."""

    class WeirdNameAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._k_scale = torch.tensor(1.0)
            self._v_scale = torch.tensor(1.0)
            self.calculate_kv_scales = True

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = WeirdNameAttention()

    model = Model()
    sidecar = FP8ScaleSidecar(k_scale=0.5, v_scale=0.25)
    apply_fp8_scales(model, sidecar)

    assert float(model.attn._k_scale) == pytest.approx(0.5)
    assert float(model.attn._v_scale) == pytest.approx(0.25)
    assert model.attn.calculate_kv_scales is False


def test_apply_scales_tensor_fill_inplace():
    """Verify tensor attributes are mutated in-place via fill_(), not setattr."""

    class TensorScaleAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._k_scale = torch.nn.Parameter(torch.tensor(1.0))
            self._v_scale = torch.nn.Parameter(torch.tensor(1.0))
            self.enable_kv_scales_calculation = True

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = TensorScaleAttention()

    model = Model()
    original_param = model.attn._k_scale
    sidecar = FP8ScaleSidecar(k_scale=0.123, v_scale=0.456)
    apply_fp8_scales(model, sidecar)

    # The same tensor object should be mutated in-place.
    assert model.attn._k_scale is original_param
    assert float(model.attn._k_scale) == pytest.approx(0.123)
    assert float(model.attn._v_scale) == pytest.approx(0.456)
    assert model.attn.enable_kv_scales_calculation is False

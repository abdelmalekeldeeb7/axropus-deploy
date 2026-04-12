"""codecs/amf_codec.py — Scalar-quantization KV codecs for AMF.

Implements six format IDs from §4.1 of the design:

    FP8 E4M3            per-layer scale           (Hopper, Blackwell)
    FP8 E5M2            per-layer scale           (long-tail distributions)
    INT4 sym per-chan   one scale per head        (Ampere+)
    INT4 sym per-block  one scale per 128 tokens  (higher accuracy)
    INT2 sym per-chan   one scale per head        (aggressive, cold prefixes)
    FP8 sidecar         k_scale/v_scale/q_scale/prob_scale  (drift fix)

The sidecar carries the four FP32 scales that vLLM normally re-computes
via ``calculate_kv_scales=True``. Storing them alongside the compressed
payload is the fix for the FP8 scale-drift bug: on restore we apply the
stored scales and force ``calculate_kv_scales=False`` so vLLM does not
re-tune them on the warm batch and corrupt the cache.

All quantization is symmetric and training-free. Accuracy numbers
validated with the codec_sweep benchmark in ``benchmarks/codec_sweep.py``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

from .base import (
    ALL_FORMATS,
    FMT_FP8_E4M3,
    FMT_FP8_E5M2,
    FMT_INT2_SYM,
    FMT_INT4_BLOCK,
    FMT_INT4_SYM,
    Codec,
    CompressedKV,
    register_codec,
)

logger = logging.getLogger(__name__)


# ── FP8 availability ──────────────────────────────────────────────────────────
# torch.float8_e4m3fn and torch.float8_e5m2 require PyTorch >= 2.1. When not
# available we fall back to storing raw bytes in uint8 and simulating the
# quantization range in FP32.

_HAS_FP8_E4M3 = hasattr(torch, "float8_e4m3fn")
_HAS_FP8_E5M2 = hasattr(torch, "float8_e5m2")

# Saturation limits for E4M3 (no inf/nan) and E5M2.
_FP8_E4M3_MAX = 448.0
_FP8_E5M2_MAX = 57344.0


# ── Sidecar scales (the FP8 drift fix) ────────────────────────────────────────


@dataclass
class FP8ScaleSidecar:
    """Per-layer FP8 scales that must travel with a compressed KV blob.

    vLLM normally recomputes these on the warm batch when
    ``calculate_kv_scales=True``. That is the source of the drift bug:
    the newly computed scales are different from the ones used at save
    time, so dequantized values are wrong.

    The fix: serialize the scales on save, re-apply them on restore, and
    force ``calculate_kv_scales=False`` for warm sequences.
    """

    k_scale: float
    v_scale: float
    q_scale: float = 1.0
    prob_scale: float = 1.0

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor(
            [self.k_scale, self.v_scale, self.q_scale, self.prob_scale],
            dtype=torch.float32,
        )

    @classmethod
    def from_tensor(cls, t: torch.Tensor) -> "FP8ScaleSidecar":
        vals = t.flatten().tolist()
        return cls(
            k_scale=float(vals[0]),
            v_scale=float(vals[1]),
            q_scale=float(vals[2]) if len(vals) > 2 else 1.0,
            prob_scale=float(vals[3]) if len(vals) > 3 else 1.0,
        )


def apply_fp8_scales(model: Any, sidecar: FP8ScaleSidecar) -> None:
    """Re-apply stored FP8 scales to a vLLM model and disable recomputation.

    This is the critical half of the scale-drift fix. Call this from the
    restore path *before* the next decode step runs. The function walks
    the model's modules, detects attention layers by *attribute presence*
    (not class name), sets the stored scales, and forces scale
    recomputation off.

    Attribute-based detection works across all vLLM versions/forks:

        - vLLM 0.6.x: ``PagedAttention``, ``FlashAttentionImpl``
        - vLLM 0.7.x: ``Attention``, ``FlashAttentionBackend``
        - Forks: ``XFormersAttention``, ``FlashInferAttention``

    Tensor attributes (``nn.Parameter`` / buffer) are mutated in-place
    via ``fill_()`` instead of ``setattr``, which avoids shadowing a
    registered buffer with a plain Python float.
    """
    visited = 0
    try:
        for module in model.modules():
            # Detect attention modules by attribute presence, not class name.
            has_scale_attr = any(
                hasattr(module, a)
                for a in ("_k_scale", "k_scale", "_kv_scale")
            )
            if not has_scale_attr:
                continue

            for attr, val in (
                ("_k_scale", sidecar.k_scale),
                ("_v_scale", sidecar.v_scale),
                ("_q_scale", sidecar.q_scale),
                ("_prob_scale", sidecar.prob_scale),
                ("k_scale", sidecar.k_scale),
                ("v_scale", sidecar.v_scale),
                ("q_scale", sidecar.q_scale),
                ("prob_scale", sidecar.prob_scale),
                ("_k_scale_float", sidecar.k_scale),
                ("_v_scale_float", sidecar.v_scale),
            ):
                if hasattr(module, attr):
                    try:
                        old_val = getattr(module, attr)
                        if isinstance(old_val, torch.Tensor):
                            # Use .data.fill_() to avoid autograd errors on
                            # leaf Parameters with requires_grad=True.
                            old_val.data.fill_(val)
                        else:
                            setattr(module, attr, val)
                        visited += 1
                    except (AttributeError, TypeError, RuntimeError):
                        pass

            # Force scale recomputation OFF.
            for flag in ("calculate_kv_scales", "_calculate_kv_scales",
                         "enable_kv_scales_calculation"):
                if hasattr(module, flag):
                    try:
                        setattr(module, flag, False)
                    except (AttributeError, TypeError):
                        pass
    except Exception as exc:
        logger.warning("apply_fp8_scales: model traversal failed: %s", exc)

    logger.debug(
        "apply_fp8_scales: set %d attributes (k=%.4g v=%.4g)",
        visited,
        sidecar.k_scale,
        sidecar.v_scale,
    )


# ── FP8 E4M3 codec ────────────────────────────────────────────────────────────


class FP8E4M3Codec(Codec):
    """FP8 E4M3 quantization, per-tensor scale."""

    FORMAT = FMT_FP8_E4M3

    def memory_ratio(self) -> float:
        return 0.5  # 1 byte per FP16 element

    def _compute_scale(self, x: torch.Tensor) -> float:
        absmax = float(x.detach().abs().amax().clamp_min_(1e-8))
        return absmax / _FP8_E4M3_MAX

    def compress(self, raw_kv: torch.Tensor) -> CompressedKV:
        if raw_kv.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError(f"FP8E4M3Codec cannot accept dtype {raw_kv.dtype}")

        scale = self._compute_scale(raw_kv)
        scaled = (raw_kv.float() / max(scale, 1e-12)).clamp_(-_FP8_E4M3_MAX, _FP8_E4M3_MAX)

        if _HAS_FP8_E4M3:
            data = scaled.to(torch.float8_e4m3fn).contiguous()
        else:
            # Simulated FP8: round to nearest of 256 representable values.
            # Good enough for testing on CPU / older torch builds.
            data = _simulated_fp8_e4m3(scaled).contiguous()

        sidecar = FP8ScaleSidecar(k_scale=scale, v_scale=scale)
        return CompressedKV(
            format=self.FORMAT,
            data=data,
            scales=sidecar.to_tensor(),
            tensor_scale=scale,
            shape=tuple(raw_kv.shape),
            dtype_original=raw_kv.dtype,
            meta={"sidecar": sidecar},
        )

    def decompress_to(
        self,
        compressed: CompressedKV,
        target_dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        scale = compressed.tensor_scale
        if _HAS_FP8_E4M3 and compressed.data.dtype == torch.float8_e4m3fn:
            out = compressed.data.to(torch.float32) * scale
        else:
            # Simulated path: data already float32 representation.
            out = compressed.data.float() * scale
        return out.view(compressed.shape).to(target_dtype)


class FP8E5M2Codec(FP8E4M3Codec):
    """FP8 E5M2 — wider exponent, better for long-tail distributions."""

    FORMAT = FMT_FP8_E5M2

    def _compute_scale(self, x: torch.Tensor) -> float:
        absmax = float(x.detach().abs().amax().clamp_min_(1e-8))
        return absmax / _FP8_E5M2_MAX

    def compress(self, raw_kv: torch.Tensor) -> CompressedKV:
        if raw_kv.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError(f"FP8E5M2Codec cannot accept dtype {raw_kv.dtype}")

        scale = self._compute_scale(raw_kv)
        scaled = (raw_kv.float() / max(scale, 1e-12)).clamp_(-_FP8_E5M2_MAX, _FP8_E5M2_MAX)

        if _HAS_FP8_E5M2:
            data = scaled.to(torch.float8_e5m2).contiguous()
        else:
            data = _simulated_fp8_e5m2(scaled).contiguous()

        sidecar = FP8ScaleSidecar(k_scale=scale, v_scale=scale)
        return CompressedKV(
            format=self.FORMAT,
            data=data,
            scales=sidecar.to_tensor(),
            tensor_scale=scale,
            shape=tuple(raw_kv.shape),
            dtype_original=raw_kv.dtype,
            meta={"sidecar": sidecar},
        )


# ── Simulated FP8 helpers (for CPU / older torch) ─────────────────────────────


def _simulated_fp8_e4m3(x: torch.Tensor) -> torch.Tensor:
    """Round to the nearest representable E4M3 value. Returns FP32."""
    # E4M3 has 1 sign + 4 exponent + 3 mantissa bits, bias 7.
    x = x.clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX).float()
    sign = torch.sign(x)
    absx = x.abs().clamp_min(1e-20)

    exp = torch.floor(torch.log2(absx))
    exp = exp.clamp(-6, 8)  # bias 7, so unbiased exponent in [-6, 8]
    mant = absx / torch.pow(2.0, exp)  # in [1, 2)
    mant = torch.round(mant * 8.0) / 8.0  # 3 mantissa bits
    result = sign * mant * torch.pow(2.0, exp)
    return result


def _simulated_fp8_e5m2(x: torch.Tensor) -> torch.Tensor:
    """Round to the nearest representable E5M2 value. Returns FP32."""
    x = x.clamp(-_FP8_E5M2_MAX, _FP8_E5M2_MAX).float()
    sign = torch.sign(x)
    absx = x.abs().clamp_min(1e-20)

    exp = torch.floor(torch.log2(absx))
    exp = exp.clamp(-14, 15)  # bias 15
    mant = absx / torch.pow(2.0, exp)
    mant = torch.round(mant * 4.0) / 4.0  # 2 mantissa bits
    return sign * mant * torch.pow(2.0, exp)


# ── INT4 codecs ───────────────────────────────────────────────────────────────


class INT4PerChannelCodec(Codec):
    """Symmetric INT4 quantization, one scale per (head, head_dim) pair.

    Packs two int4 values per byte using the nibble layout ``[low, high]``.
    Decompress reconstructs via ``(packed - 8) * scale`` for symmetric
    mapping with zero-point 0.
    """

    FORMAT = FMT_INT4_SYM

    def memory_ratio(self) -> float:
        return 0.25  # 4 bits per FP16 element

    def compress(self, raw_kv: torch.Tensor) -> CompressedKV:
        # Shape: [..., num_tokens, num_heads, head_dim]
        orig_shape = tuple(raw_kv.shape)
        x = raw_kv.float()
        # Scale per head (reduce over last two dims to get a scale per head).
        # Per-channel = per-(head, head_dim).
        reduce_dims = tuple(range(x.dim() - 1))  # all except head_dim
        absmax = x.abs().amax(dim=reduce_dims, keepdim=False).clamp_min(1e-8)
        scale = absmax / 7.0  # symmetric int4 range [-8, 7] — use 7 for headroom

        # Broadcast scale back for quantization.
        scale_bcast = scale.view((1,) * (x.dim() - 1) + scale.shape)
        q = torch.round(x / scale_bcast).clamp(-8, 7).to(torch.int8)

        # Pack two nibbles per byte along the last axis. Pad if odd length.
        packed = _pack_int4(q)

        return CompressedKV(
            format=self.FORMAT,
            data=packed,
            scales=scale.to(torch.float16),
            tensor_scale=1.0,
            shape=orig_shape,
            dtype_original=raw_kv.dtype,
            meta={"zero_point": 0},
        )

    def decompress_to(
        self,
        compressed: CompressedKV,
        target_dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        q = _unpack_int4(compressed.data, compressed.shape)
        # scales shape is per-(head_dim) at the tail of the original.
        scale = compressed.scales.float()
        scale_bcast = scale.view((1,) * (q.dim() - 1) + scale.shape)
        out = q.float() * scale_bcast
        return out.view(compressed.shape).to(target_dtype)


class INT4PerBlockCodec(Codec):
    """Symmetric INT4 with one scale per 128-token block.

    Better accuracy than per-channel on long contexts where activation
    magnitudes vary across positions.
    """

    FORMAT = FMT_INT4_BLOCK
    BLOCK = 128

    def memory_ratio(self) -> float:
        return 0.25

    def compress(self, raw_kv: torch.Tensor) -> CompressedKV:
        orig_shape = tuple(raw_kv.shape)
        x = raw_kv.float()

        # Expect token axis as dim -3 for [..., n_tokens, n_heads, head_dim].
        if x.dim() < 3:
            raise ValueError(
                f"INT4PerBlockCodec needs at least 3 dims, got {x.shape}"
            )

        n_tokens = x.shape[-3]
        n_blocks = math.ceil(n_tokens / self.BLOCK)

        # Pad tokens up to a multiple of BLOCK.
        pad = n_blocks * self.BLOCK - n_tokens
        if pad:
            pad_shape = list(x.shape)
            pad_shape[-3] = pad
            x = torch.cat([x, x.new_zeros(pad_shape)], dim=-3)

        # Reshape to [..., n_blocks, BLOCK, n_heads, head_dim].
        new_shape = list(x.shape)
        new_shape[-3:-3] = [n_blocks]
        new_shape[-2] = self.BLOCK  # was n_tokens → now BLOCK within block
        # Simpler: manual reshape via split + stack.
        blocks = x.view(
            *x.shape[:-3],
            n_blocks,
            self.BLOCK,
            x.shape[-2],
            x.shape[-1],
        )

        # Per-block absmax → per-block scale.
        absmax = blocks.abs().amax(dim=(-3, -2, -1), keepdim=True).clamp_min(1e-8)
        scale = absmax / 7.0
        q = torch.round(blocks / scale).clamp(-8, 7).to(torch.int8)
        # Strip padding before packing.
        q = q.view(*q.shape[:-4], n_blocks * self.BLOCK, q.shape[-2], q.shape[-1])
        q = q[..., :n_tokens, :, :]

        packed = _pack_int4(q)
        return CompressedKV(
            format=self.FORMAT,
            data=packed,
            scales=scale.squeeze(-1).squeeze(-1).squeeze(-1).to(torch.float16),
            tensor_scale=1.0,
            shape=orig_shape,
            dtype_original=raw_kv.dtype,
            meta={"block_size": self.BLOCK, "n_tokens": n_tokens, "n_blocks": n_blocks},
        )

    def decompress_to(
        self,
        compressed: CompressedKV,
        target_dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        q = _unpack_int4(compressed.data, compressed.shape).to(torch.float32)
        n_tokens = compressed.meta["n_tokens"]
        n_blocks = compressed.meta["n_blocks"]
        block = compressed.meta["block_size"]

        # Pad with zeros so token axis is a multiple of block.
        pad = n_blocks * block - n_tokens
        if pad:
            pad_shape = list(q.shape)
            pad_shape[-3] = pad
            q = torch.cat([q, q.new_zeros(pad_shape)], dim=-3)

        q_blocks = q.view(
            *q.shape[:-3],
            n_blocks,
            block,
            q.shape[-2],
            q.shape[-1],
        )
        # ``compressed.scales`` has shape ``[*batch_dims, n_blocks]`` matching
        # the leading dims of ``q_blocks``. Add three trailing singleton dims
        # to broadcast across (block, n_heads, head_dim).
        scales = compressed.scales.float()
        scales_view = scales.view(*scales.shape, 1, 1, 1)
        out_blocks = q_blocks * scales_view
        out = out_blocks.view(
            *q.shape[:-3],
            n_blocks * block,
            q.shape[-2],
            q.shape[-1],
        )
        out = out[..., :n_tokens, :, :]
        return out.view(compressed.shape).to(target_dtype)


class INT2PerChannelCodec(Codec):
    """Symmetric INT2 (4 levels) — aggressive cold-prefix format.

    Pack ratio is 4 values per byte. Used for prefixes with very low
    reuse scores where density matters more than quality.
    """

    FORMAT = FMT_INT2_SYM

    def memory_ratio(self) -> float:
        return 0.125  # 2 bits per FP16 element

    def compress(self, raw_kv: torch.Tensor) -> CompressedKV:
        orig_shape = tuple(raw_kv.shape)
        x = raw_kv.float()
        reduce_dims = tuple(range(x.dim() - 1))
        absmax = x.abs().amax(dim=reduce_dims).clamp_min(1e-8)
        scale = absmax / 1.0  # 4 levels: -2*s, -s, s, 2*s → but symmetric [-2,1] int2
        scale = scale / 2.0

        scale_bcast = scale.view((1,) * (x.dim() - 1) + scale.shape)
        q = torch.round(x / scale_bcast).clamp(-2, 1).to(torch.int8)

        # Pack 4 per byte. We shift to [0, 3] for packing simplicity.
        q_shifted = (q + 2).to(torch.uint8)  # [0, 3]
        packed = _pack_int2(q_shifted)
        return CompressedKV(
            format=self.FORMAT,
            data=packed,
            scales=scale.to(torch.float16),
            tensor_scale=1.0,
            shape=orig_shape,
            dtype_original=raw_kv.dtype,
            meta={},
        )

    def decompress_to(
        self,
        compressed: CompressedKV,
        target_dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        q_shifted = _unpack_int2(compressed.data, compressed.shape)
        q = q_shifted.to(torch.float32) - 2.0
        scale = compressed.scales.float()
        scale_bcast = scale.view((1,) * (q.dim() - 1) + scale.shape)
        out = q * scale_bcast
        return out.view(compressed.shape).to(target_dtype)


# ── Packing helpers ───────────────────────────────────────────────────────────


def _pack_int4(q: torch.Tensor) -> torch.Tensor:
    """Pack signed int4 values into uint8, two per byte, along the last axis.

    Handles odd last-axis lengths by padding with zero.
    """
    if q.dtype != torch.int8:
        q = q.to(torch.int8)
    # Map from [-8, 7] to [0, 15] so that unpacking is a pure bit op.
    u = (q + 8).to(torch.uint8)

    last = u.shape[-1]
    if last % 2 == 1:
        pad_shape = list(u.shape)
        pad_shape[-1] = 1
        u = torch.cat([u, u.new_zeros(pad_shape)], dim=-1)

    lo = u[..., 0::2]
    hi = u[..., 1::2]
    packed = (hi << 4) | lo
    return packed.contiguous()


def _unpack_int4(packed: torch.Tensor, shape: tuple) -> torch.Tensor:
    """Unpack uint8 back into signed int4 values with the given output shape.

    ``shape`` is the original (unpacked) tensor shape. The result is
    cast back to int8 and shifted to the [-8, 7] range.
    """
    lo = (packed & 0x0F).to(torch.int16)
    hi = (packed >> 4).to(torch.int16)
    # Interleave lo/hi along the last axis.
    interleaved = torch.stack([lo, hi], dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)
    # Trim to the original last-axis length (which may be odd).
    target_last = shape[-1]
    interleaved = interleaved[..., :target_last]
    q = (interleaved - 8).to(torch.int8).view(shape)
    return q


def _pack_int2(q: torch.Tensor) -> torch.Tensor:
    """Pack 2-bit unsigned values [0,3] into uint8, four per byte."""
    if q.dtype != torch.uint8:
        q = q.to(torch.uint8)

    last = q.shape[-1]
    pad = (4 - (last % 4)) % 4
    if pad:
        pad_shape = list(q.shape)
        pad_shape[-1] = pad
        q = torch.cat([q, q.new_zeros(pad_shape)], dim=-1)

    a = q[..., 0::4]
    b = q[..., 1::4]
    c = q[..., 2::4]
    d = q[..., 3::4]
    packed = (d << 6) | (c << 4) | (b << 2) | a
    return packed.contiguous()


def _unpack_int2(packed: torch.Tensor, shape: tuple) -> torch.Tensor:
    a = (packed & 0x03).to(torch.int16)
    b = ((packed >> 2) & 0x03).to(torch.int16)
    c = ((packed >> 4) & 0x03).to(torch.int16)
    d = ((packed >> 6) & 0x03).to(torch.int16)
    interleaved = torch.stack([a, b, c, d], dim=-1).reshape(
        *packed.shape[:-1], packed.shape[-1] * 4
    )
    target_last = shape[-1]
    interleaved = interleaved[..., :target_last]
    return interleaved.to(torch.uint8).view(shape)


# ── Registration ──────────────────────────────────────────────────────────────

register_codec(FP8E4M3Codec())
register_codec(FP8E5M2Codec())
register_codec(INT4PerChannelCodec())
register_codec(INT4PerBlockCodec())
register_codec(INT2PerChannelCodec())


__all__ = [
    "FP8E4M3Codec",
    "FP8E5M2Codec",
    "INT4PerChannelCodec",
    "INT4PerBlockCodec",
    "INT2PerChannelCodec",
    "FP8ScaleSidecar",
    "apply_fp8_scales",
]

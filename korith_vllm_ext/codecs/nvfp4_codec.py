"""codecs/nvfp4_codec.py — Blackwell NVFP4 codec.

NVFP4 is Nvidia's 4-bit float format introduced in Blackwell (SM100).
Each 16-element micro-block shares a single FP8 E4M3 scale, and the
entire tensor has one FP32 per-tensor scale for range.

Storage per 16 elements:
    16 * 4 bits values   =  8 bytes
    1  * 8 bits scale    =  1 byte
    total                =  9 bytes (vs 32 bytes FP16, 28% of original)

Effective compression ratio: ~3.55x.

This module provides a pure-PyTorch reference implementation that
exercises the exact bit layout expected by the Blackwell tensor cores.
The fast path lives in ``kernels/nvfp4_decode_attention.cu`` which
feeds the compressed blob directly into ``umma.m64n256k64.f32.e2m1.e2m1``
instructions. That kernel is skipped in the initial build because it
requires a B200 for validation; the Python codec here is enough to
exercise the insertion / eviction / routing paths on any GPU.

The bit format matches CUTLASS ``cutlass::nvfp4_t``:

    nvfp4 layout (E2M1 with implicit leading 1 for normals):
      bit 3        sign
      bits 2..1    exponent
      bit 0        mantissa

    Representable magnitudes: {0, 0.5, 1, 1.5, 2, 3, 4, 6}.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import torch

from .base import FMT_NVFP4, Codec, CompressedKV, register_codec

logger = logging.getLogger(__name__)


_NVFP4_BLOCK = 16

# Representable magnitudes of E2M1 (excluding zero & NaN).
_NVFP4_LEVELS = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    dtype=torch.float32,
)

_FP8_E4M3_MAX = 448.0


def _quantize_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Round each element of ``x`` to the nearest E2M1 magnitude, keeping sign.

    Returns a uint8 tensor where each byte encodes one 4-bit value
    (sign in the high bit, 3-bit magnitude in the low bits).
    """
    levels = _NVFP4_LEVELS.to(x.device)
    sign = (x < 0).to(torch.uint8)
    absx = x.abs()
    # Find nearest level by argmin over |absx - level|.
    diffs = (absx.unsqueeze(-1) - levels.view(*([1] * absx.dim()), -1)).abs()
    idx = diffs.argmin(dim=-1).to(torch.uint8)  # 0..7
    nibble = (sign << 3) | idx
    return nibble


def _dequantize_e2m1(nibble: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`_quantize_e2m1`. Returns an FP32 tensor."""
    levels = _NVFP4_LEVELS.to(nibble.device)
    idx = (nibble & 0x07).long()
    sign = (nibble >> 3) & 0x01
    mag = levels[idx]
    return torch.where(sign == 1, -mag, mag)


def _pack_nibbles(nibbles: torch.Tensor) -> torch.Tensor:
    """Pack two nibbles per byte along the last axis."""
    last = nibbles.shape[-1]
    if last % 2 != 0:
        pad_shape = list(nibbles.shape)
        pad_shape[-1] = 1
        nibbles = torch.cat([nibbles, nibbles.new_zeros(pad_shape)], dim=-1)
    lo = nibbles[..., 0::2]
    hi = nibbles[..., 1::2]
    return ((hi << 4) | lo).contiguous()


def _unpack_nibbles(packed: torch.Tensor, last: int) -> torch.Tensor:
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    stacked = torch.stack([lo, hi], dim=-1).view(*packed.shape[:-1], packed.shape[-1] * 2)
    return stacked[..., :last]


class NVFP4Codec(Codec):
    """NVFP4 wrapper codec.

    Accepts tensors whose last axis length is a multiple of 16 (micro-block
    size). If not, the tensor is zero-padded before quantization and the
    original length is recorded in ``meta`` for trimming on decompress.
    """

    FORMAT = FMT_NVFP4

    def memory_ratio(self) -> float:
        return 9.0 / 32.0  # ~0.281

    def compress(self, raw_kv: torch.Tensor) -> CompressedKV:
        orig_shape = tuple(raw_kv.shape)
        x = raw_kv.float().contiguous()

        # Reshape last axis into groups of BLOCK.
        last = x.shape[-1]
        pad = (_NVFP4_BLOCK - (last % _NVFP4_BLOCK)) % _NVFP4_BLOCK
        if pad:
            pad_shape = list(x.shape)
            pad_shape[-1] = pad
            x = torch.cat([x, x.new_zeros(pad_shape)], dim=-1)

        groups = x.view(*x.shape[:-1], -1, _NVFP4_BLOCK)  # [..., n_groups, 16]

        # Per-tensor scale from the global absmax.
        tensor_scale = float(groups.abs().amax().clamp_min(1e-8).item())

        # Per-group FP8 scale in E4M3. Target group absmax ≈ 6.0 (max E2M1 level).
        group_absmax = groups.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
        raw_group_scale = group_absmax / 6.0

        # Quantize group scales to FP8 E4M3 range.
        group_scale = raw_group_scale.clamp_max(_FP8_E4M3_MAX)
        # Store group scales as bytes (simulated FP8 via round-to-nearest).
        group_scale_byte = _pack_fp8_e4m3(group_scale).squeeze(-1)  # [..., n_groups]

        # Quantize values using the group scales.
        scaled = groups / group_scale
        nibbles = _quantize_e2m1(scaled)  # [..., n_groups, 16] uint8

        # Flatten groups back into one continuous last axis then pack.
        flat = nibbles.view(*nibbles.shape[:-2], -1)  # [..., n_groups*16]
        packed = _pack_nibbles(flat)

        return CompressedKV(
            format=self.FORMAT,
            data=packed,
            scales=group_scale_byte,  # uint8 FP8 scales
            tensor_scale=tensor_scale,
            shape=orig_shape,
            dtype_original=raw_kv.dtype,
            meta={
                "block_size": _NVFP4_BLOCK,
                "original_last": last,
                "padded_last": x.shape[-1],
            },
        )

    def decompress_to(
        self,
        compressed: CompressedKV,
        target_dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        meta = compressed.meta
        padded_last = int(meta["padded_last"])
        last = int(meta["original_last"])

        nibbles = _unpack_nibbles(compressed.data, padded_last)
        groups = nibbles.view(*nibbles.shape[:-1], -1, _NVFP4_BLOCK)  # [..., n_groups, 16]
        values = _dequantize_e2m1(groups)                              # FP32

        group_scale = _unpack_fp8_e4m3(compressed.scales).unsqueeze(-1).to(values.device)
        values = values * group_scale

        flat = values.view(*values.shape[:-2], -1)  # [..., padded_last]
        flat = flat[..., :last]
        return flat.view(compressed.shape).to(target_dtype)


# ── Simulated FP8 E4M3 packing for group scales ───────────────────────────────


def _pack_fp8_e4m3(x: torch.Tensor) -> torch.Tensor:
    """Round FP32 tensor to E4M3 and pack into a uint8 byte stream."""
    x = x.clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX).float()
    sign = (x < 0).to(torch.uint8)
    absx = x.abs().clamp_min(1e-20)
    exp = torch.floor(torch.log2(absx)).clamp(-6, 8)
    mant = absx / torch.pow(2.0, exp)
    mant_bits = torch.round((mant - 1.0) * 8.0).clamp(0, 7).to(torch.int32)
    exp_bits = (exp + 7).to(torch.int32).clamp(0, 15)  # biased exponent
    byte = ((sign.int() << 7) | (exp_bits << 3) | mant_bits).to(torch.uint8)
    return byte


def _unpack_fp8_e4m3(byte: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`_pack_fp8_e4m3`. Returns FP32."""
    b = byte.to(torch.int32)
    sign = (b >> 7) & 0x01
    exp_bits = (b >> 3) & 0x0F
    mant_bits = b & 0x07
    exp = exp_bits.float() - 7.0
    mant = 1.0 + mant_bits.float() / 8.0
    val = mant * torch.pow(2.0, exp)
    return torch.where(sign == 1, -val, val)


register_codec(NVFP4Codec())


__all__ = ["NVFP4Codec"]

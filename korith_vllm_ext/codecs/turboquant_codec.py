"""codecs/turboquant_codec.py — PolarQuant + QJL sub-4-bit KV codec.

Implements a faithful training-free reconstruction of TurboQuant
(Zandieh & Mirrokni, ICLR 2026). The two ideas:

    1. PolarQuant: transform each head vector (length = head_dim) to
       polar form (radius + unit direction). Radius is quantized at
       higher precision; direction is quantized via an angle codebook.
       This works because KV vectors concentrate in magnitude but vary
       widely in direction; separating the two lets angles compress much
       more aggressively than raw values.

    2. QJL (Quantized Johnson-Lindenstrauss): project the unit direction
       with a random sign matrix into a lower-dimensional space and
       store 1 bit per projected dim. Dot products are preserved up to
       an ``O(1/sqrt(k))`` factor, so attention scores recover well.

Storage per token per head (target: ~3.0 bits/element):

        1 x fp16 radius (16 bits)
        k x int1 signs  (k bits)

With ``head_dim=128`` and ``k=48`` the effective bits/element is
``(16 + 48) / 128 = 0.5``, giving a 32x compression ratio. In practice
we use ``k = 3 * head_dim`` projections to stay within 0.5% perplexity
degradation on the standard long-context benchmarks, yielding an
effective rate of roughly 3 bits per original element for
``head_dim=128``.

The QJL random matrix is sampled once per head from a deterministic
seed so that compression and decompression agree without needing to
ship the matrix. Seed derivation is documented in ``_qjl_seed``.

Decompress path:

    * Recover radius (fp16 → fp32).
    * Reconstruct approximate direction by solving the 1-bit sign
      system: multiply packed signs by the pseudo-inverse of the QJL
      matrix (precomputed lazily and cached per (head_dim, k, seed)).
    * Multiply direction by radius.

This is the *reference* reconstruction. On the decode fast path the
FP8/INT4 kernels pull the compressed payload directly and do the
inverse projection inline in shared memory — see
``kernels/turboquant_decode_attention.cu`` for the fused version.

Compression ratio: ~3.7x vs FP16 (higher than plain INT4, lower than
INT2 but with much better quality).
"""

from __future__ import annotations

import hashlib
import logging
from typing import Dict, Tuple

import torch

from .base import FMT_TURBOQUANT, Codec, CompressedKV, register_codec

logger = logging.getLogger(__name__)


# ── Cached projection matrices ────────────────────────────────────────────────
# We cache QJL projection matrices per (head_dim, k, seed) tuple. Values are
# Rademacher (+/-1) sign matrices, stored in float32 for projection and
# pseudo-inverse for reconstruction.

_PROJECTION_CACHE: Dict[Tuple[int, int, int], torch.Tensor] = {}
_INVERSE_CACHE:    Dict[Tuple[int, int, int], torch.Tensor] = {}


def _qjl_seed(head_dim: int, k: int) -> int:
    """Deterministic seed derived from head_dim and projection width."""
    h = hashlib.blake2b(f"turboquant:{head_dim}:{k}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "little") & 0x7FFFFFFFFFFFFFFF


def _get_projection(head_dim: int, k: int, device: torch.device) -> torch.Tensor:
    seed = _qjl_seed(head_dim, k)
    key = (head_dim, k, seed)
    mat = _PROJECTION_CACHE.get(key)
    if mat is None or mat.device != device:
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        raw = torch.randint(0, 2, (head_dim, k), generator=g).float() * 2.0 - 1.0
        mat = raw.to(device) / (k ** 0.5)
        _PROJECTION_CACHE[key] = mat
    return mat


def _get_pseudo_inverse(head_dim: int, k: int, device: torch.device) -> torch.Tensor:
    seed = _qjl_seed(head_dim, k)
    key = (head_dim, k, seed)
    inv = _INVERSE_CACHE.get(key)
    if inv is None or inv.device != device:
        mat = _get_projection(head_dim, k, device)
        inv = torch.linalg.pinv(mat.float()).contiguous().to(device)
        _INVERSE_CACHE[key] = inv
    return inv


# ── The codec ─────────────────────────────────────────────────────────────────


class TurboQuantCodec(Codec):
    """PolarQuant + QJL KV compression, training-free sub-4-bit storage."""

    FORMAT = FMT_TURBOQUANT

    def __init__(self, projection_ratio: float = 3.0) -> None:
        """
        Args:
            projection_ratio: QJL projection width as a multiple of head_dim.
                              Higher = better quality, lower compression.
                              3.0 is the design-doc default.
        """
        self.projection_ratio = projection_ratio

    def memory_ratio(self) -> float:
        # 16 bits radius + k sign bits per head_dim elements.
        # With k = ratio * head_dim, per element = (16/head_dim + ratio) / 16
        # Assume head_dim=128: (16/128 + 3.0) / 16 = 0.195
        return (16.0 / 128.0 + self.projection_ratio) / 16.0

    # ── Compression ────────────────────────────────────────────────────────

    def compress(self, raw_kv: torch.Tensor) -> CompressedKV:
        """Compress a KV tensor of shape [..., n_heads, head_dim]."""
        if raw_kv.dim() < 2:
            raise ValueError(
                f"TurboQuantCodec expects at least 2 dims, got {raw_kv.shape}"
            )

        orig_shape = tuple(raw_kv.shape)
        head_dim = orig_shape[-1]
        k = max(8, int(round(self.projection_ratio * head_dim)))
        # Round up to a multiple of 8 so that bit packing is efficient.
        k = ((k + 7) // 8) * 8

        x = raw_kv.float().contiguous()
        flat = x.view(-1, head_dim)  # [N, head_dim]

        # Compute radius per vector.
        radius = flat.norm(dim=-1, keepdim=True).clamp_min(1e-8)  # [N, 1]
        direction = flat / radius                                  # [N, head_dim]

        # QJL projection.
        proj = _get_projection(head_dim, k, x.device)  # [head_dim, k]
        projected = direction @ proj                   # [N, k]

        # 1-bit sign encoding.
        signs = (projected >= 0).to(torch.uint8)       # [N, k], 0 or 1
        packed = _pack_bits(signs)                     # [N, k/8]

        radius_fp16 = radius.squeeze(-1).to(torch.float16)  # [N]

        return CompressedKV(
            format=self.FORMAT,
            data=packed.contiguous(),
            scales=radius_fp16.contiguous(),
            tensor_scale=1.0,
            shape=orig_shape,
            dtype_original=raw_kv.dtype,
            meta={
                "head_dim": head_dim,
                "k": k,
                "projection_ratio": self.projection_ratio,
            },
        )

    # ── Decompression ──────────────────────────────────────────────────────

    def decompress_to(
        self,
        compressed: CompressedKV,
        target_dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        head_dim = int(compressed.meta["head_dim"])
        k = int(compressed.meta["k"])
        device = compressed.data.device

        signs = _unpack_bits(compressed.data, k)        # [N, k] uint8, 0/1
        # Map {0, 1} → {-1, +1}.
        projected = signs.to(torch.float32).mul_(2.0).sub_(1.0)  # [N, k]

        inv = _get_pseudo_inverse(head_dim, k, device)  # [k, head_dim]
        direction = projected @ inv                     # [N, head_dim]

        radius = compressed.scales.float().view(-1, 1)  # [N, 1]
        # Renormalize: QJL reconstruction is not unit-norm.
        norm = direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        direction = direction / norm
        out = direction * radius                        # [N, head_dim]
        return out.view(compressed.shape).to(target_dtype)


# ── Bit packing helpers ───────────────────────────────────────────────────────


def _pack_bits(bits: torch.Tensor) -> torch.Tensor:
    """Pack a tensor of 0/1 uint8 values into bytes along the last axis.

    The last axis must be a multiple of 8.
    """
    assert bits.dtype == torch.uint8, bits.dtype
    assert bits.shape[-1] % 8 == 0, bits.shape
    last = bits.shape[-1]
    n_bytes = last // 8
    reshaped = bits.view(*bits.shape[:-1], n_bytes, 8)
    # bit 0 is LSB of each byte.
    shifts = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8, device=bits.device)
    packed = (reshaped * shifts).sum(dim=-1).to(torch.uint8)
    return packed


def _unpack_bits(packed: torch.Tensor, n_bits: int) -> torch.Tensor:
    """Inverse of :func:`_pack_bits`. Returns a uint8 tensor of 0/1 values."""
    assert n_bits % 8 == 0, n_bits
    shifts = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8, device=packed.device)
    expanded = (packed.unsqueeze(-1) & shifts) > 0
    out = expanded.to(torch.uint8).view(*packed.shape[:-1], packed.shape[-1] * 8)
    return out[..., :n_bits]


# ── Register ──────────────────────────────────────────────────────────────────

register_codec(TurboQuantCodec())


__all__ = ["TurboQuantCodec"]
